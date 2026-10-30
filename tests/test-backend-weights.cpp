// Weights in a buffer with GGML_BACKEND_BUFFER_USAGE_WEIGHTS may be stored in a backend specific
// layout (for example repacked into tiles). Views of such weights must still behave like views of
// the plain ggml layout:
//   - ggml_backend_tensor_get() on a view returns the same bytes that were uploaded
//
// usage: test-backend-weights [-b <device name>]   (default: every non-CPU device)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int n_failed = 0;

#define CHECK(cond, ...)                         \
    do {                                         \
        if (!(cond)) {                           \
            printf("  FAIL: " __VA_ARGS__);      \
            printf("\n");                        \
            n_failed++;                          \
        }                                        \
    } while (0)

struct weight_case {
    ggml_type type;
    int64_t   ne0, ne1, ne2;
};

// quantize (or convert) n_rows rows of src into the bytes of a tensor of type `type`
static std::vector<uint8_t> make_weight_bytes(ggml_type type, const std::vector<float> & src, int64_t ne0, int64_t n_rows) {
    std::vector<uint8_t> out(ggml_row_size(type, ne0) * n_rows);
    if (type == GGML_TYPE_F32) {
        memcpy(out.data(), src.data(), out.size());
    } else if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src.data(), (ggml_fp16_t *) out.data(), ne0 * n_rows);
    } else {
        ggml_quantize_chunk(type, src.data(), out.data(), 0, n_rows, ne0, nullptr);
    }
    return out;
}

// views of the weight, read directly with ggml_backend_tensor_get
static void test_view_reads(const weight_case & wc, ggml_tensor * w, const std::vector<uint8_t> & bytes) {
    ggml_init_params params = { ggml_tensor_overhead() * (wc.ne2 + 1), nullptr, true };
    ggml_context * ctx = ggml_init(params);

    const size_t slice = w->nb[2];
    const size_t row   = w->nb[1];

    for (int64_t k = 0; k < wc.ne2; k++) {
        ggml_tensor * v = ggml_view_2d(ctx, w, wc.ne0, wc.ne1, w->nb[1], k * slice);
        GGML_ASSERT(ggml_backend_view_init(v) == GGML_STATUS_SUCCESS);

        std::vector<uint8_t> got(slice);
        ggml_backend_tensor_get(v, got.data(), 0, slice);
        CHECK(memcmp(got.data(), bytes.data() + k * slice, slice) == 0,
              "%s: get_tensor of view %lld (whole slice) differs from the uploaded bytes", ggml_type_name(wc.type), (long long) k);

        // one row at an offset inside the view
        const int64_t r = wc.ne1 / 2 + 1;
        std::vector<uint8_t> got_row(row);
        ggml_backend_tensor_get(v, got_row.data(), r * row, row);
        CHECK(memcmp(got_row.data(), bytes.data() + k * slice + r * row, row) == 0,
              "%s: get_tensor of view %lld row %lld differs from the uploaded bytes", ggml_type_name(wc.type), (long long) k, (long long) r);
    }

    ggml_free(ctx);
}

static void test_device(ggml_backend_dev_t dev) {
    printf("device %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend);

    // 3D weights viewed one slice at a time (like per-codebook embedding tables).
    // No F32: a backend may keep F32 weights at reduced precision (Hexagon tiles them as F16),
    // so even a read of the whole tensor need not return the uploaded bytes.
    const weight_case cases[] = {
        { GGML_TYPE_Q8_0, 256, 64, 4 },
        { GGML_TYPE_Q4_0, 256, 64, 4 },
        { GGML_TYPE_F16,  256, 64, 1 },
    };

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (const weight_case & wc : cases) {
        const int n_failed_before = n_failed;

        ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx_w = ggml_init(params);
        ggml_tensor * w = ggml_new_tensor_3d(ctx_w, wc.type, wc.ne0, wc.ne1, wc.ne2);
        ggml_set_name(w, "w");

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
        GGML_ASSERT(buf);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        std::vector<float> src(wc.ne0 * wc.ne1 * wc.ne2);
        for (float & x : src) {
            x = dist(rng);
        }
        std::vector<uint8_t> bytes = make_weight_bytes(wc.type, src, wc.ne0, wc.ne1 * wc.ne2);
        ggml_backend_tensor_set(w, bytes.data(), 0, bytes.size());

        test_view_reads(wc, w, bytes);

        printf("  %-5s [%lld, %lld, %lld]: %s\n", ggml_type_name(wc.type), (long long) wc.ne0, (long long) wc.ne1,
               (long long) wc.ne2, n_failed == n_failed_before ? "OK" : "FAIL");

        ggml_backend_buffer_free(buf);
        ggml_free(ctx_w);
    }

    ggml_backend_free(backend);
}

int main(int argc, char ** argv) {
    const char * dev_name = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            dev_name = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [-b <device name>]\n", argv[0]);
            return 1;
        }
    }

    ggml_backend_load_all();

    int n_tested = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev_name ? strcmp(ggml_backend_dev_name(dev), dev_name) != 0
                     : ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        test_device(dev);
        n_tested++;
    }

    if (n_tested == 0) {
        printf("no device to test\n");
        return dev_name ? 1 : 0;
    }

    printf("%s\n", n_failed ? "FAILED" : "OK");
    return n_failed ? 1 : 0;
}
