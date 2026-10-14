#ifndef HTP_ARANGE_OPS_H
#define HTP_ARANGE_OPS_H

#include "hex-common.h"
#include "htp-vtcm.h"
#include <stddef.h>
#include <stdint.h>

// Below this many elements a single thread is faster than waking the workers.
#define HTP_ARANGE_MIN_ELEMS_PER_HVX 1024
// F32 elements per VTCM output buffer (16 KB)
#define HTP_ARANGE_CHUNK_ELEMS 4096

// Per-hvx-thread VTCM layout: 2 x chunk output buffers (ping-pong)
struct htp_arange_vtcm_layout {
    size_t total_bytes;
    size_t bytes_per_hvx;
    size_t off_out;
    size_t out_buf_size;
};

static inline void htp_arange_vtcm_layout_build(struct htp_arange_vtcm_layout * L, uint32_t n_hvx) {
    L->out_buf_size = (size_t) HTP_ARANGE_CHUNK_ELEMS * sizeof(float);

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_out, 2 * L->out_buf_size);

    L->bytes_per_hvx = off;
    L->total_bytes   = off * n_hvx;
}

struct htp_arange_kernel_params {
    uint32_t n_hvx;
    uint32_t vtcm_size;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_arange_kernel_params) <= 128, "htp_arange_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_arange_kernel_params) <= 128, "htp_arange_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_ARANGE_OPS_H
