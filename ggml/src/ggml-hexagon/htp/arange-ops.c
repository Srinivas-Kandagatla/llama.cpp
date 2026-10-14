#include <HAP_farf.h>
#include <stdint.h>
#include <string.h>

#include "hex-common.h"
#include "htp-trace.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "dma-queue.h"
#include "dma-pipeline.h"
#include "arange-ops.h"

// ggml op_params layout for ARANGE (all as float):
//   op_params[0] - start, op_params[1] - stop, op_params[2] - step
// dst is a contiguous 1-D F32 tensor with dst[i] = start + i * step.

struct htp_arange_context {
    struct htp_ops_context *      octx;
    struct htp_arange_vtcm_layout layout;
    uint32_t                      elem_start;
    uint32_t                      nelems;
    uint32_t                      nelems_per_hvx;
    float                         start;
    float                         step;
};

// same formula as the CPU backend (not cumulative, so there is no drift)
static __attribute__((noinline)) void arange_compute(float * restrict out, uint32_t i0, uint32_t n, float start, float step) {
    for (uint32_t j = 0; j < n; j++) {
        out[j] = start + (float) (i0 + j) * step;
    }
}

struct arange_pipe_state {
    dma_addr_t data_dst;
    float      start;
    float      step;
    uint32_t   i;
    uint32_t   i_end;
};

static bool arange_next_task(void * ctx, struct dma_pipe_task * task) {
    struct arange_pipe_state * s = (struct arange_pipe_state *) ctx;
    if (s->i >= s->i_end) {
        return false;
    }
    task->row = s->i;
    task->off = 0;
    task->len = MIN(HTP_ARANGE_CHUNK_ELEMS, s->i_end - s->i);
    task->aux = 0;
    s->i += task->len;
    return true;
}

static void arange_pipe_compute(void * ctx, const struct dma_pipe_task * task, void * out_spad) {
    const struct arange_pipe_state * s = (const struct arange_pipe_state *) ctx;
    arange_compute((float *) out_spad, task->row, task->len, s->start, s->step);
}

static void arange_push_out(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * out_spad) {
    const struct arange_pipe_state * s = (const struct arange_pipe_state *) ctx;
    const size_t nbytes = (size_t) task->len * sizeof(float);
    dma_queue_push(dma_q, dma_make_data(s->data_dst + (size_t) task->row * sizeof(float), out_spad), nbytes, nbytes, nbytes, 1);
}

// Each thread owns elements [i0, i1): chunks are computed into VTCM and DMAed out,
// two buffers so compute of one chunk overlaps the writeback of the previous one.
static void arange_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_arange_context * actx = (const struct htp_arange_context *) data;
    struct htp_ops_context *          octx = actx->octx;

    const uint32_t i0 = actx->elem_start + actx->nelems_per_hvx * ith;
    const uint32_t i1 = MIN(i0 + actx->nelems_per_hvx, actx->elem_start + actx->nelems);
    if (i0 >= i1) {
        return;
    }

    const struct htp_arange_vtcm_layout * L = &actx->layout;
    uint8_t * vtcm       = (uint8_t *) octx->ctx->vtcm_base + ith * L->bytes_per_hvx;
    void *    out_buf[2] = { vtcm + L->off_out, vtcm + L->off_out + L->out_buf_size };

    struct arange_pipe_state s = {
        .data_dst = octx->dst->data,
        .start    = actx->start,
        .step     = actx->step,
        .i        = i0,
        .i_end    = i1,
    };

    dma_pipe_out_run(th->dma_q, th->trace, &s, out_buf, 2, arange_next_task, arange_pipe_compute, arange_push_out);
}

int op_arange(struct htp_ops_context * octx) {
    const struct htp_tensor * dst = octx->dst;
    const struct htp_arange_kernel_params * kparams =
        (const struct htp_arange_kernel_params *) octx->kernel_params;

    if (dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t ne = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    uint32_t elem_start = 0;
    uint32_t nelems     = ne;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            ne, HEX_L2_LINE_SIZE / sizeof(float), octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        elem_start = range.start;
        nelems     = range.count;
    }

    if (nelems == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_hvx = MAX(1, MIN(kparams->n_hvx, (nelems + HTP_ARANGE_MIN_ELEMS_PER_HVX - 1) / HTP_ARANGE_MIN_ELEMS_PER_HVX));
    if (!htp_ops_context_set_n_hvx(octx, n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    struct htp_arange_context actx = {
        .octx       = octx,
        .elem_start = elem_start,
        .nelems     = nelems,
    };
    memcpy(&actx.start, &octx->op_params[0], sizeof(float));
    memcpy(&actx.step,  &octx->op_params[2], sizeof(float));

    htp_arange_vtcm_layout_build(&actx.layout, n_hvx);
    if (octx->ctx->vtcm_size < actx.layout.total_bytes) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    actx.nelems_per_hvx = (nelems + n_hvx - 1) / n_hvx;

    FARF(HIGH, "arange: %u elems (%u:%u) n-hvx %u\n", ne, elem_start, nelems, n_hvx);

    work_queue_run(octx->ctx->hvx_queue, arange_thread, &actx, n_hvx);

    return HTP_STATUS_OK;
}
