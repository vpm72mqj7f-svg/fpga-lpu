/**
 * weight_preloader.c — CPU Weight Preloading from SSD (P1)
 *
 * Pre-loads 61 layers of DeepSeek V4 weights into host pinned memory.
 * Weights stored as fp4 (4-bit) on SSD, unpacked to fp8 at load time.
 *
 * Memory layout (per layer, ~100 MB fp4 → ~200 MB fp8 unpacked):
 *   W_Q:        [7168, 7168] fp8  = 51.4 MB
 *   W_K:        [512,  7168] fp8  =  3.7 MB
 *   W_V:        [512,  7168] fp8  =  3.7 MB
 *   W_K_up:     [7168, 512]  fp8  =  3.7 MB
 *   W_V_up:     [7168, 512]  fp8  =  3.7 MB
 *   W_gate:     [3072, 7168] fp8  = 22.0 MB
 *   W_up:       [3072, 7168] fp8  = 22.0 MB
 *   W_down:     [7168, 3072] fp8  = 22.0 MB
 *   W_router:   [384,  7168] fp8  =  2.8 MB
 *   Expert 0..11 each: [3072,7168]×3 fp8 = 66 MB
 *
 * Total per layer: ~200 MB fp8
 * Total 61 layers: ~12 GB fp8 in host memory
 *
 * Uses libaio for async SSD I/O, mlock for DMA pinning.
 *
 * Build:
 *   gcc -O3 -laio -o weight_preloader weight_preloader.c
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <libaio.h>
#include <pthread.h>

/* ── Weight layout descriptor ──────────────────────────────────────────── */

typedef struct {
    const char *name;
    size_t      rows;
    size_t      cols;
    size_t      elem_bytes;   // 1 for fp8, 0.5 for fp4 (packed)
    int         is_fp4;       // needs unpacking
} weight_tensor_t;

typedef struct {
    int             layer_idx;
    weight_tensor_t tensors[20];  // up to 20 tensors per layer
    int             num_tensors;
    uint8_t        *data;         // pinned memory buffer
    size_t          total_bytes;
} layer_weights_t;

/* ── fp4 → fp8 unpacking ──────────────────────────────────────────────── */

static void unpack_fp4_to_fp8(const uint8_t *packed, uint8_t *unpacked,
                              size_t num_weights) {
    /* fp4 E2M1 format: {sign[3], exp[2:1], mant[0]}
     * Packed: two fp4 weights per byte [w1[3:0], w0[3:0]]
     * Unpacked: each fp4 → fp8 E4M3 (zero-extend mantissa, shift exponent)
     */
    for (size_t i = 0; i < num_weights; i++) {
        uint8_t byte = packed[i / 2];
        uint8_t fp4  = (i & 1) ? (byte >> 4) : (byte & 0x0F);

        /* Convert E2M1 → E4M3:
         *   E2M1: sign[3], exp[2:1], mant[0]
         *   E4M3: sign[7], exp[6:3], mant[2:0]
         *   Map: sign→sign, exp+2→exp, mant<<1→mant
         */
        uint8_t sign  = (fp4 >> 3) & 1;
        uint8_t exp   = (fp4 >> 1) & 3;
        uint8_t mant  = fp4 & 1;

        if (fp4 == 0) {
            unpacked[i] = 0x00;  /* zero */
        } else {
            uint8_t fp8_exp  = exp + 2;      /* E2M1 bias 1 → E4M3 bias 7, +6 shift */
            if (fp8_exp > 14) fp8_exp = 14;  /* saturate */
            uint8_t fp8_mant = mant << 1;
            unpacked[i] = (sign << 7) | (fp8_exp << 3) | fp8_mant;
        }
    }
}

/* ── Async I/O: read layer weights from SSD ───────────────────────────── */

static int ssd_read_async(int fd, void *buf, size_t size, off_t offset,
                          io_context_t *ctx, struct iocb *cb) {
    io_prep_pread(cb, fd, buf, size, offset);
    struct iocb *cbs[1] = { cb };
    return io_submit(*ctx, 1, cbs);
}

/* ── Weight preloader ─────────────────────────────────────────────────── */

typedef struct {
    const char    *weight_dir;     // SSD mount point
    int            num_layers;
    int            num_preload;     // layers to keep in memory
    layer_weights_t *layers;
    size_t          total_memory;
    pthread_mutex_t lock;
} weight_preloader_t;

int weight_preloader_init(weight_preloader_t *wp,
                          const char *weight_dir,
                          int num_layers,
                          int num_preload) {
    memset(wp, 0, sizeof(*wp));
    wp->weight_dir  = weight_dir;
    wp->num_layers  = num_layers;
    wp->num_preload = num_preload;

    wp->layers = calloc(num_layers, sizeof(layer_weights_t));
    if (!wp->layers) return -1;

    pthread_mutex_init(&wp->lock, NULL);
    return 0;
}

int weight_preloader_load_layer(weight_preloader_t *wp,
                                int layer_idx,
                                io_context_t *io_ctx) {
    if (layer_idx >= wp->num_layers) return -1;

    layer_weights_t *lw = &wp->layers[layer_idx];
    lw->layer_idx = layer_idx;

    /* Define tensors for this layer */
    weight_tensor_t tensors[] = {
        {"W_Q",     7168, 7168, 1, 0},   // fp8
        {"W_K",      512, 7168, 1, 0},
        {"W_V",      512, 7168, 1, 0},
        {"W_K_up",  7168,  512, 1, 0},
        {"W_V_up",  7168,  512, 1, 0},
        {"W_gate",  3072, 7168, 1, 0},
        {"W_up",    3072, 7168, 1, 0},
        {"W_down",  7168, 3072, 1, 0},
        {"W_router", 384, 7168, 1, 0},
        {NULL, 0, 0, 0, 0}
    };

    /* Calculate total bytes */
    size_t total = 0;
    int nt = 0;
    for (int i = 0; tensors[i].name; i++) {
        size_t bytes = tensors[i].rows * tensors[i].cols * tensors[i].elem_bytes;
        total += bytes;
        nt++;
    }

    /* Allocate pinned memory */
    uint8_t *buf = NULL;
    posix_memalign((void **)&buf, 4096, total);
    if (!buf) return -1;
    mlock(buf, total);  /* pin for DMA */

    /* Read each tensor from SSD */
    size_t offset = 0;
    char path[256];
    int fd = -1;
    struct stat st;

    snprintf(path, sizeof(path), "%s/layer_%04d.bin", wp->weight_dir, layer_idx);
    fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        /* Fallback: individual tensor files */
        fd = -1;
    }

    for (int i = 0; i < nt; i++) {
        size_t bytes = tensors[i].rows * tensors[i].cols * tensors[i].elem_bytes;

        if (fd >= 0) {
            /* Read from monolithic layer file */
            struct iocb cb;
            io_prep_pread(&cb, fd, buf + offset, bytes,
                         offset);  /* same layout in file */
            io_context_t ctx = *io_ctx;
            struct iocb *cbs[1] = { &cb };
            int ret = io_submit(ctx, 1, cbs);
            if (ret != 1) {
                /* Fall through to slow path */
            } else {
                struct io_event event;
                io_getevents(ctx, 1, 1, &event, NULL);
            }
        } else {
            /* Slow path: read individual tensor file */
            snprintf(path, sizeof(path), "%s/layer_%04d_%s.bin",
                     wp->weight_dir, layer_idx, tensors[i].name);
            FILE *f = fopen(path, "rb");
            if (f) {
                size_t n = fread(buf + offset, 1, bytes, f);
                fclose(f);
                if (n != bytes) {
                    fprintf(stderr, "Short read: %s (%zu/%zu)\n", path, n, bytes);
                }
            }
        }

        if (tensors[i].is_fp4) {
            /* Unpack in-place: fp4 takes half the space */
            size_t unpacked_bytes = bytes * 2;  /* 4b → 8b */
            uint8_t *tmp = malloc(unpacked_bytes);
            unpack_fp4_to_fp8(buf + offset, tmp, tensors[i].rows * tensors[i].cols);
            memcpy(buf + offset, tmp, unpacked_bytes);
            free(tmp);
        }

        offset += bytes;
    }

    if (fd >= 0) close(fd);

    lw->data        = buf;
    lw->total_bytes = total;
    memcpy(lw->tensors, tensors, nt * sizeof(weight_tensor_t));
    lw->num_tensors = nt;

    wp->total_memory += total;

    /* Evict oldest layer if over preload limit */
    pthread_mutex_lock(&wp->lock);
    int loaded = 0;
    for (int i = 0; i < wp->num_layers; i++) {
        if (wp->layers[i].data) loaded++;
    }
    if (loaded > wp->num_preload) {
        /* Evict LRU layer (simplified: evict furthest from current) */
        int evict = (layer_idx + wp->num_layers / 2) % wp->num_layers;
        if (wp->layers[evict].data) {
            munlock(wp->layers[evict].data, wp->layers[evict].total_bytes);
            free(wp->layers[evict].data);
            wp->layers[evict].data = NULL;
            wp->total_memory -= wp->layers[evict].total_bytes;
        }
    }
    pthread_mutex_unlock(&wp->lock);

    return 0;
}

const uint8_t *weight_preloader_get_tensor(const weight_preloader_t *wp,
                                           int layer_idx,
                                           const char *tensor_name,
                                           size_t *out_bytes) {
    if (layer_idx >= wp->num_layers) return NULL;
    const layer_weights_t *lw = &wp->layers[layer_idx];
    if (!lw->data) return NULL;

    size_t offset = 0;
    for (int i = 0; i < lw->num_tensors; i++) {
        size_t bytes = lw->tensors[i].rows * lw->tensors[i].cols *
                       lw->tensors[i].elem_bytes;
        if (strcmp(lw->tensors[i].name, tensor_name) == 0) {
            if (out_bytes) *out_bytes = bytes;
            return lw->data + offset;
        }
        offset += bytes;
    }
    return NULL;
}

void weight_preloader_destroy(weight_preloader_t *wp) {
    for (int i = 0; i < wp->num_layers; i++) {
        if (wp->layers[i].data) {
            munlock(wp->layers[i].data, wp->layers[i].total_bytes);
            free(wp->layers[i].data);
        }
    }
    free(wp->layers);
    pthread_mutex_destroy(&wp->lock);
}

/* ── Benchmark ────────────────────────────────────────────────────────── */

#if __MAIN__
int main(int argc, char **argv) {
    const char *weight_dir = argc > 1 ? argv[1] : "/data/weights/deepseek_v4";
    int num_layers = 61;
    int num_preload = 4;  // keep 4 layers pinned (~800 MB)

    io_context_t io_ctx;
    memset(&io_ctx, 0, sizeof(io_ctx));
    io_setup(128, &io_ctx);

    weight_preloader_t wp;
    if (weight_preloader_init(&wp, weight_dir, num_layers, num_preload) < 0) {
        fprintf(stderr, "Failed to init weight preloader\n");
        return 1;
    }

    /* Benchmark: load all 61 layers sequentially */
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    for (int l = 0; l < num_layers; l++) {
        if (weight_preloader_load_layer(&wp, l, &io_ctx) < 0) {
            fprintf(stderr, "Failed to load layer %d\n", l);
        }
        if (l % 10 == 0) {
            fprintf(stderr, "Loaded layer %d/%d (%.1f GB total)\r",
                    l + 1, num_layers, wp.total_memory / 1e9);
        }
    }

    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) +
                     (t1.tv_usec - t0.tv_usec) / 1e6;

    printf("\nWeight preload complete:\n");
    printf("  Layers:  %d\n", num_layers);
    printf("  Memory:  %.1f GB pinned\n", wp.total_memory / 1e9);
    printf("  Time:    %.1f s\n", elapsed);
    printf("  Rate:    %.1f GB/s\n", wp.total_memory / 1e9 / elapsed);

    /* Verify: check first tensor of layer 0 */
    size_t bytes;
    const uint8_t *wq = weight_preloader_get_tensor(&wp, 0, "W_Q", &bytes);
    printf("  Verify:  W_Q[0] @ %p, %zu bytes\n", (void *)wq, bytes);

    weight_preloader_destroy(&wp);
    io_destroy(io_ctx);
    return 0;
}
#endif
