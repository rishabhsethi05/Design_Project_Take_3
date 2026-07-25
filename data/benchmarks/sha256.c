/*
 * Standalone SHA-256 Benchmark (5x Scaled Workload)
 * Bare-metal implementation of Secure Hash Algorithm 256-bit.
 * Adapted for standalone MAPI-PRO simulation without external headers.
 */

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// SHA-256 State Variables
uint32_t state[8];
uint8_t data_buffer[64];
uint32_t datalen;
uint64_t bitlen;
uint8_t hash_output[32];

// Test input block (64 bytes)
static const uint8_t test_data[64] = {
    'm', 'a', 'p', 'i', 'p', 'r', 'o', '_', 's', 'h', 'a', '2', '5', '6', '_', 't',
    'e', 's', 't', '_', 'b', 'l', 'o', 'c', 'k', '_', 'z', 'e', 'r', 'o', '_', 'x',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
    'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v'
};

// SHA-256 Constants
static const uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xeHere is **`sha256.c`** (Cryptographic Hashing), adapted directly from standard embedded cryptographic benchmarks (BEEBS / CyaSSL / mbedTLS subsets).

It is completely standalone, header-less, and includes an internal **5x scaled workload loop** to give PACE plenty of heavy bitwise logic and working-state modifications to optimize.

---

### Save as `data/benchmarks/sha256.c`

```c
/*
 * BEEBS / Embedded SHA-256 Benchmark (5x Scaled Workload)
 * Secure Hash Algorithm (256-bit).
 * Adapted for standalone MAPI-PRO simulation without external headers.
 */

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

typedef struct {
    uint8_t data_buf[64];
    uint32_t datalen;
    unsigned long long bitlen;
    uint32_t state[8];
} SHA256_CTX;

// SHA-256 Round Constants
static const uint32_t K_sha[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef4a3f7,0xc67178f2
};

SHA256_CTX ctx;
uint8_t test_payload[256];

static void sha256_transform(SHA256_CTX *ctx_ptr, const uint8_t data[])
{
    uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
    for (; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = ctx_ptr->state[0];
    b = ctx_ptr->state[1];
    c = ctx_ptr->state[2];
    d = ctx_ptr->state[3];
    e = ctx_ptr->state[4];
    f = ctx_ptr->state[5];
    g = ctx_ptr->state[6];
    h = ctx_ptr->state[7];

    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + K_sha[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx_ptr->state[0] += a;
    ctx_ptr->state[1] += b;
    ctx_ptr->state[2] += c;
    ctx_ptr->state[3] += d;
    ctx_ptr->state[4] += e;
    ctx_ptr->state[5] += f;
    ctx_ptr->state[6] += g;
    ctx_ptr->state[7] += h;
}

static void sha256_init(SHA256_CTX *ctx_ptr)
{
    ctx_ptr->datalen = 0;
    ctx_ptr->bitlen = 0;
    ctx_ptr->state[0] = 0x6a09e667;
    ctx_ptr->state[1] = 0xbb67ae85;
    ctx_ptr->state[2] = 0x3c6ef372;
    ctx_ptr->state[3] = 0xa54ff53a;
    ctx_ptr->state[4] = 0x510e527f;
    ctx_ptr->state[5] = 0x9b05688c;
    ctx_ptr->state[6] = 0x1f83d9ab;
    ctx_ptr->state[7] = 0x5be0cd19;
}

static void sha256_update(SHA256_CTX *ctx_ptr, const uint8_t data[], size_t len)
{
    size_t i;
    for (i = 0; i < len; ++i) {
        ctx_ptr->data_buf[ctx_ptr->datalen] = data[i];
        ctx_ptr->datalen++;
        if (ctx_ptr->datalen == 64) {
            sha256_transform(ctx_ptr, ctx_ptr->data_buf);
            ctx_ptr->bitlen += 512;
            ctx_ptr->datalen = 0;
        }
    }
}