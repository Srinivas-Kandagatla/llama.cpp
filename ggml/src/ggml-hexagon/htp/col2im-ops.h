#ifndef HTP_COL2IM_OPS_H
#define HTP_COL2IM_OPS_H

#include "hex-common.h"
#include "htp-vtcm.h"
#include <stddef.h>
#include <stdint.h>

// Output time steps per thread span unit (one 128B line of F32)
#define HTP_COL2IM_T_ALIGN 32

// src columns needed for an output chunk of ct time steps
static inline uint32_t htp_col2im_max_in_rows(uint32_t ct, uint32_t K, uint32_t s0) {
    return (ct + K - 2) / s0 + 2;
}

// Per-hvx-thread VTCM layout: 2 x src column block and 2 x dst chunk (ping-pong)
struct htp_col2im_vtcm_layout {
    size_t total_bytes;
    size_t bytes_per_hvx;
    size_t in_half_bytes;
    size_t out_half_bytes;
    size_t off_in;
    size_t off_out;
};

static inline void htp_col2im_vtcm_layout_build(
    struct htp_col2im_vtcm_layout * L,
    uint32_t ct,
    uint32_t K,
    uint32_t s0,
    uint32_t n_hvx
) {
    L->in_half_bytes  = hex_round_up((size_t) htp_col2im_max_in_rows(ct, K, s0) * K * sizeof(float), 128);
    L->out_half_bytes = hex_round_up((size_t) ct * sizeof(float), 128);

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_in,  2 * L->in_half_bytes);
    VTCM_LAYOUT_ALLOC(off, off_out, 2 * L->out_half_bytes);

    L->bytes_per_hvx = off;
    L->total_bytes   = off * n_hvx;
}

struct htp_col2im_kernel_params {
    uint32_t n_hvx;
    uint32_t chunk_t;    // output time steps per DMA chunk (multiple of HTP_COL2IM_T_ALIGN)
    uint32_t vtcm_size;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_col2im_kernel_params) <= 128, "htp_col2im_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_col2im_kernel_params) <= 128, "htp_col2im_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_COL2IM_OPS_H
