#ifndef HTP_PAD_REFLECT_OPS_H
#define HTP_PAD_REFLECT_OPS_H

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"
#include <stddef.h>
#include <stdint.h>

// Per-hvx-thread VTCM layout: 2 x block src rows and 2 x block dst rows (ping-pong)
struct htp_pad_reflect_vtcm_layout {
    size_t total_bytes;
    size_t bytes_per_hvx;
    size_t src_row_size_aligned;
    size_t dst_row_size_aligned;
    size_t src_half_bytes;
    size_t dst_half_bytes;
    size_t off_src;
    size_t off_dst;
};

static inline void htp_pad_reflect_vtcm_layout_build(
    struct htp_pad_reflect_vtcm_layout * L,
    uint32_t ne00,
    uint32_t ne0,
    uint32_t block,
    uint32_t n_hvx
) {
    L->src_row_size_aligned = hex_round_up((size_t) ne00 * sizeof(float), 128);
    L->dst_row_size_aligned = hex_round_up((size_t) ne0  * sizeof(float), 128);
    L->src_half_bytes       = (size_t) block * L->src_row_size_aligned;
    L->dst_half_bytes       = (size_t) block * L->dst_row_size_aligned;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src, 2 * L->src_half_bytes);
    VTCM_LAYOUT_ALLOC(off, off_dst, 2 * L->dst_half_bytes);

    L->bytes_per_hvx = off;
    L->total_bytes   = off * n_hvx;
}

struct htp_pad_reflect_kernel_params {
    uint32_t n_hvx;
    uint32_t block;      // rows per DMA block (never crosses an ne1 slice)
    uint32_t vtcm_size;

    struct fastdiv_values div_ne1;
    struct fastdiv_values div_ne2;
    struct fastdiv_values div_ne12;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_pad_reflect_kernel_params) <= 128, "htp_pad_reflect_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_pad_reflect_kernel_params) <= 128, "htp_pad_reflect_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_PAD_REFLECT_OPS_H
