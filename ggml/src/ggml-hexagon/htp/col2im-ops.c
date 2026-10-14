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
#include "col2im-ops.h"

// COL2IM_1D (overlap-add of a transposed 1D convolution), F32:
//   src: columns [K*OC, T_in], dst: [T_out, OC], T_out = (T_in - 1)*s0 + K - 2*p0
//   dst[t, oc] = sum over t_in with k = t + p0 - t_in*s0 in [0, K) of src[oc*K + k, t_in]
//   op_params: [0] = s0, [1] = OC, [2] = p0
// Each output sums ceil(K/s0) columns. Outputs are independent, so threads split the time axis
// (balanced for any OC, including mono) and no atomics are needed.
// For each (oc, chunk of output time steps) the K x n_in block of src columns it reads is DMAed
// into VTCM, the outputs are summed in VTCM and DMAed out.

struct htp_col2im_context {
    struct htp_ops_context *      octx;
    struct htp_col2im_vtcm_layout layout;
    uint32_t                      t_start;          // first output time step of this dev
    uint32_t                      nt;               // output time steps of this dev
    uint32_t                      nt_per_hvx;
    uint32_t                      chunk_t;
    int32_t                       s0, p0;
    uint32_t                      K, OC, T_in, T_out, K_OC;
};

struct col2im_pipe_state {
    const struct htp_col2im_context * c;
    dma_addr_t                        src;
    dma_addr_t                        dst;
    uint32_t                          oc;
    uint32_t                          t;
    uint32_t                          t0;
    uint32_t                          t1;
};

// first and one-past-last src column that contributes to outputs [ta, tb)
static inline void col2im_in_range(const struct htp_col2im_context * c, uint32_t ta, uint32_t tb, int32_t * lo, int32_t * hi) {
    const int32_t a = (int32_t) ta + c->p0 - (int32_t) c->K + 1;
    *lo = a > 0 ? (a + c->s0 - 1) / c->s0 : 0;
    int32_t h = ((int32_t) tb - 1 + c->p0) / c->s0 + 1;
    *hi = h > (int32_t) c->T_in ? (int32_t) c->T_in : h;
}

static bool col2im_next_task(void * ctx, struct dma_pipe_task * task) {
    struct col2im_pipe_state * s = (struct col2im_pipe_state *) ctx;
    if (s->oc >= s->c->OC) {
        return false;
    }
    task->row = s->oc;
    task->off = s->t;
    task->len = MIN(s->c->chunk_t, s->t1 - s->t);
    task->aux = 0;
    s->t += task->len;
    if (s->t >= s->t1) {
        s->t = s->t0;
        s->oc++;
    }
    return true;
}

static void col2im_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    const struct col2im_pipe_state *  s = (const struct col2im_pipe_state *) ctx;
    const struct htp_col2im_context * c = s->c;

    int32_t lo, hi;
    col2im_in_range(c, task->off, task->off + task->len, &lo, &hi);
    const uint32_t n_in = hi > lo ? (uint32_t) (hi - lo) : 0;

    // rows of K floats (one per src column t_in), packed in VTCM; zero rows keep the pipeline's
    // one-descriptor-per-task contract when no column contributes
    const dma_addr_t src = s->src + ((size_t) task->row * c->K + (size_t) (n_in ? lo : 0) * c->K_OC) * sizeof(float);
    dma_queue_push(dma_q, dma_make_data(in_spad, src),
                   c->K * sizeof(float), c->K_OC * sizeof(float), c->K * sizeof(float), n_in);
}

static void col2im_compute(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    const struct col2im_pipe_state *  s = (const struct col2im_pipe_state *) ctx;
    const struct htp_col2im_context * c = s->c;

    const float * col = (const float *) in_spad;   // col[(t_in - lo) * K + k]
    float *       out = (float *) out_spad;

    const int32_t s0 = c->s0, p0 = c->p0;
    const int32_t K = (int32_t) c->K, T_in = (int32_t) c->T_in;

    int32_t lo_blk, hi_blk;
    col2im_in_range(c, task->off, task->off + task->len, &lo_blk, &hi_blk);

    for (uint32_t j = 0; j < task->len; j++) {
        const int32_t t_abs = (int32_t) (task->off + j) + p0;
        int32_t       lo    = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s0); t_abs-K+1 may be < 0
        if (t_abs - K + 1 < 0) {
            lo = 0;
        }
        int32_t hi = t_abs / s0;
        if (hi >= T_in) {
            hi = T_in - 1;
        }
        float sum = 0.0f;
        for (int32_t t_in = lo; t_in <= hi; t_in++) {
            const int32_t k = t_abs - t_in * s0;
            if (k >= 0 && k < K) {
                sum += col[(size_t) (t_in - lo_blk) * K + k];
            }
        }
        out[j] = sum;
    }
}

static void col2im_push_out(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * out_spad) {
    const struct col2im_pipe_state *  s = (const struct col2im_pipe_state *) ctx;
    const size_t nbytes = (size_t) task->len * sizeof(float);
    const dma_addr_t dst = s->dst + ((size_t) task->row * s->c->T_out + task->off) * sizeof(float);
    dma_queue_push(dma_q, dma_make_data(dst, out_spad), nbytes, nbytes, nbytes, 1);
}

static void col2im_1d_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_col2im_context * c    = (const struct htp_col2im_context *) data;
    struct htp_ops_context *          octx = c->octx;

    const uint32_t t0 = c->t_start + c->nt_per_hvx * ith;
    const uint32_t t1 = MIN(t0 + c->nt_per_hvx, c->t_start + c->nt);
    if (t0 >= t1) {
        return;
    }

    const struct htp_col2im_vtcm_layout * L = &c->layout;
    uint8_t * vtcm       = (uint8_t *) octx->ctx->vtcm_base + ith * L->bytes_per_hvx;
    void *    in_buf[2]  = { vtcm + L->off_in,  vtcm + L->off_in  + L->in_half_bytes };
    void *    out_buf[2] = { vtcm + L->off_out, vtcm + L->off_out + L->out_half_bytes };

    struct col2im_pipe_state s = {
        .c   = c,
        .src = octx->src[0]->data,
        .dst = octx->dst->data,
        .oc  = 0,
        .t   = t0,
        .t0  = t0,
        .t1  = t1,
    };

    dma_pipe_io_run(th->dma_q, th->trace, &s, in_buf, out_buf,
                    col2im_next_task, col2im_push_in, col2im_compute, col2im_push_out);
}

int op_col2im_1d(struct htp_ops_context * octx) {
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    if (src->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_col2im_kernel_params * kparams = (const struct htp_col2im_kernel_params *) octx->kernel_params;

    struct htp_col2im_context c;
    memset(&c, 0, sizeof(c));
    c.octx    = octx;
    c.s0      = octx->op_params[0];
    c.OC      = (uint32_t) octx->op_params[1];
    c.p0      = octx->op_params[2];
    c.K_OC    = src->ne[0];
    c.T_in    = src->ne[1];
    c.T_out   = dst->ne[0];
    c.chunk_t = kparams->chunk_t;
    if (c.s0 <= 0 || c.OC == 0 || c.p0 < 0 || c.K_OC % c.OC != 0 || dst->ne[1] != c.OC || c.chunk_t == 0) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    c.K = c.K_OC / c.OC;

    c.t_start = 0;
    c.nt      = c.T_out;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            c.T_out, HTP_COL2IM_T_ALIGN, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        c.t_start = range.start;
        c.nt      = range.count;
    }
    if (c.nt == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, MIN(kparams->n_hvx, (c.nt + HTP_COL2IM_T_ALIGN - 1) / HTP_COL2IM_T_ALIGN))) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    const uint32_t n_hvx = octx->n_hvx;
    c.nt_per_hvx         = hex_round_up((c.nt + n_hvx - 1) / n_hvx, HTP_COL2IM_T_ALIGN);

    htp_col2im_vtcm_layout_build(&c.layout, c.chunk_t, c.K, (uint32_t) c.s0, n_hvx);
    if (octx->ctx->vtcm_size < c.layout.total_bytes) {
        FARF(ERROR, "col2im-1d: VTCM size %zu is too small, needed %zu\n", octx->ctx->vtcm_size, c.layout.total_bytes);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    work_queue_run(octx->ctx->hvx_queue, col2im_1d_thread, &c, n_hvx);

    return HTP_STATUS_OK;
}
