#include <HAP_farf.h>
#include <stdint.h>
#include <string.h>

#include "hex-common.h"
#include "htp-trace.h"

#include "hvx-copy.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "dma-queue.h"
#include "dma-pipeline.h"
#include "pad-reflect-ops.h"

// PAD_REFLECT_1D: pad dim 0 of each F32 row by reflection (edge sample not repeated).
//   op_params[0] = p0 (left), op_params[1] = p1 (right); requires p0, p1 < ne00.
//   [a, b, c, d] with p0 = p1 = 1 -> [b, a, b, c, d, c]

struct htp_pad_reflect_context {
    struct htp_ops_context *                     octx;
    const struct htp_pad_reflect_kernel_params * kparams;
    struct htp_pad_reflect_vtcm_layout           layout;
    uint32_t                                     row_start;
    uint32_t                                     nrows;
    uint32_t                                     nrows_per_hvx;
    uint32_t                                     p0;
    uint32_t                                     p1;
};

struct pad_reflect_pipe_state {
    const struct htp_pad_reflect_context * pctx;
    const struct htp_tensor *              src;
    const struct htp_tensor *              dst;
    uint32_t                               ir;
    uint32_t                               ir_end;
};

static inline size_t pad_reflect_row_offset(const struct htp_pad_reflect_context * pctx, const struct htp_tensor * t, uint32_t ir) {
    const struct htp_pad_reflect_kernel_params * kp = pctx->kparams;
    const uint32_t ne1 = t->ne[1];
    const uint32_t ne2 = t->ne[2];
    const uint32_t i1  = fastmodulo(ir, ne1, &kp->div_ne1);
    const uint32_t i2  = fastmodulo(fastdiv(ir, &kp->div_ne1), ne2, &kp->div_ne2);
    const uint32_t i3  = fastdiv(ir, &kp->div_ne12);
    return (size_t) i1 * t->nb[1] + (size_t) i2 * t->nb[2] + (size_t) i3 * t->nb[3];
}

static bool pad_reflect_next_task(void * ctx, struct dma_pipe_task * task) {
    struct pad_reflect_pipe_state * s = (struct pad_reflect_pipe_state *) ctx;
    if (s->ir >= s->ir_end) {
        return false;
    }
    // a block never crosses an ne1 slice, so its rows are evenly strided in both src and dst
    const uint32_t ne1       = s->src->ne[1];
    const uint32_t slice_end = (fastdiv(s->ir, &s->pctx->kparams->div_ne1) + 1) * ne1;
    task->row = s->ir;
    task->off = 0;
    task->len = MIN(MIN(s->pctx->kparams->block, s->ir_end - s->ir), slice_end - s->ir);
    task->aux = 0;
    s->ir += task->len;
    return true;
}

static void pad_reflect_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    const struct pad_reflect_pipe_state * s = (const struct pad_reflect_pipe_state *) ctx;
    const size_t src_off = pad_reflect_row_offset(s->pctx, s->src, task->row);
    dma_queue_push(dma_q, dma_make_data(in_spad, s->src->data + src_off),
                   s->pctx->layout.src_row_size_aligned, s->src->nb[1], (size_t) s->src->ne[0] * sizeof(float), task->len);
}

static void pad_reflect_compute(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    const struct pad_reflect_pipe_state * s = (const struct pad_reflect_pipe_state *) ctx;
    const uint32_t ne00 = s->src->ne[0];
    const uint32_t p0   = s->pctx->p0;
    const uint32_t p1   = s->pctx->p1;

    for (uint32_t r = 0; r < task->len; r++) {
        const float * src = (const float *) ((const uint8_t *) in_spad + (size_t) r * s->pctx->layout.src_row_size_aligned);
        float *       dst = (float *) ((uint8_t *) out_spad + (size_t) r * s->pctx->layout.dst_row_size_aligned);

        hvx_copy_f32_ua((uint8_t *) (dst + p0), (const uint8_t *) src, ne00);
        for (uint32_t i = 1; i <= p0; i++) {
            dst[p0 - i] = src[i];
        }
        for (uint32_t i = 1; i <= p1; i++) {
            dst[p0 + ne00 - 1 + i] = src[ne00 - 1 - i];
        }
    }
}

static void pad_reflect_push_out(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * out_spad) {
    const struct pad_reflect_pipe_state * s = (const struct pad_reflect_pipe_state *) ctx;
    const size_t dst_off = pad_reflect_row_offset(s->pctx, s->dst, task->row);
    dma_queue_push(dma_q, dma_make_data(s->dst->data + dst_off, out_spad),
                   s->dst->nb[1], s->pctx->layout.dst_row_size_aligned, (size_t) s->dst->ne[0] * sizeof(float), task->len);
}

static void pad_reflect_1d_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_pad_reflect_context * pctx = (const struct htp_pad_reflect_context *) data;
    struct htp_ops_context *               octx = pctx->octx;

    const uint32_t ir0 = pctx->row_start + pctx->nrows_per_hvx * ith;
    const uint32_t ir1 = MIN(ir0 + pctx->nrows_per_hvx, pctx->row_start + pctx->nrows);
    if (ir0 >= ir1) {
        return;
    }

    const struct htp_pad_reflect_vtcm_layout * L = &pctx->layout;
    uint8_t * vtcm       = (uint8_t *) octx->ctx->vtcm_base + ith * L->bytes_per_hvx;
    void *    in_buf[2]  = { vtcm + L->off_src, vtcm + L->off_src + L->src_half_bytes };
    void *    out_buf[2] = { vtcm + L->off_dst, vtcm + L->off_dst + L->dst_half_bytes };

    struct pad_reflect_pipe_state s = {
        .pctx   = pctx,
        .src    = octx->src[0],
        .dst    = octx->dst,
        .ir     = ir0,
        .ir_end = ir1,
    };

    dma_pipe_io_run(th->dma_q, th->trace, &s, in_buf, out_buf,
                    pad_reflect_next_task, pad_reflect_push_in, pad_reflect_compute, pad_reflect_push_out);
}

int op_pad_reflect_1d(struct htp_ops_context * octx) {
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    if (src->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t p0 = (uint32_t) octx->op_params[0];
    const uint32_t p1 = (uint32_t) octx->op_params[1];
    if (p0 >= src->ne[0] || p1 >= src->ne[0] || dst->ne[0] != src->ne[0] + p0 + p1) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const struct htp_pad_reflect_kernel_params * kparams = (const struct htp_pad_reflect_kernel_params *) octx->kernel_params;

    const uint32_t nr = src->ne[1] * src->ne[2] * src->ne[3];

    uint32_t row_start = 0;
    uint32_t nrows     = nr;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            nr, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }
    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, MIN(kparams->n_hvx, nrows))) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    const uint32_t n_hvx = octx->n_hvx;

    struct htp_pad_reflect_context pctx = {
        .octx          = octx,
        .kparams       = kparams,
        .row_start     = row_start,
        .nrows         = nrows,
        .nrows_per_hvx = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div),
        .p0            = p0,
        .p1            = p1,
    };

    htp_pad_reflect_vtcm_layout_build(&pctx.layout, src->ne[0], dst->ne[0], kparams->block, n_hvx);
    if (octx->ctx->vtcm_size < pctx.layout.total_bytes) {
        FARF(ERROR, "pad-reflect-1d: VTCM size %zu is too small, needed %zu\n", octx->ctx->vtcm_size, pctx.layout.total_bytes);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    work_queue_run(octx->ctx->hvx_queue, pad_reflect_1d_thread, &pctx, n_hvx);

    return HTP_STATUS_OK;
}
