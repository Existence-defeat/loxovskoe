
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static float rand_scale(float s) {
    return ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * s;
}

static float* xmalloc_vec(int n) {
    float* p = (float*)malloc((size_t)n * sizeof(float));
    if (!p) { perror("malloc"); exit(1); }
    return p;
}

static float* xcalloc_vec(int n) {
    float* p = (float*)calloc((size_t)n, sizeof(float));
    if (!p) { perror("calloc"); exit(1); }
    return p;
}

static float** xmalloc_mat(int rows, int cols, bool zero) {
    float** m = (float**)malloc((size_t)rows * sizeof(float*));
    if (!m) { perror("malloc"); exit(1); }
    for (int i=0;i<rows;i++) {
        m[i] = zero ? (float*)calloc((size_t)cols, sizeof(float))
                    : (float*)malloc((size_t)cols * sizeof(float));
        if (!m[i]) { perror("malloc/calloc"); exit(1); }
    }
    return m;
}

static void free_mat(float** m, int rows) {
    if (!m) return;
    for (int i=0;i<rows;i++) free(m[i]);
    free(m);
}

static void softmax_inplace(float* a, int n) {
    float mx = a[0];
    for (int i=1;i<n;i++) if (a[i] > mx) mx = a[i];
    float sum = 0.0f;
    for (int i=0;i<n;i++) {
        a[i] = expf(a[i] - mx);
        sum += a[i];
    }
    if (sum < 1e-12f) sum = 1e-12f;
    for (int i=0;i<n;i++) a[i] /= sum;
}

static float* matvec(const float* v, float** W, int in_dim, int out_dim) {
    float* r = xcalloc_vec(out_dim);
    for (int i=0;i<out_dim;i++) {
        float s = 0.0f;
        for (int j=0;j<in_dim;j++) s += v[j] * W[i][j];
        r[i] = s;
    }
    return r;
}

static float* vec_add_new(const float* a, const float* b, int n) {
    float* r = xmalloc_vec(n);
    for (int i=0;i<n;i++) r[i] = a[i] + b[i];
    return r;
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / (float)M_PI) *
           (x + 0.044715f * x * x * x)));
}
float gelu1(float x) {
    const double SQRT_2_OVER_PI = 0.7978845608; 
    const double COEFF = 0.044715;

    double x3 = (double)x * (double)x * (double)x;
    double inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    double tanh_inner = tanh(inner);
    
    double term1 = 0.5 * (1.0 + tanh_inner);
    double sech2 = 1.0 - tanh_inner * tanh_inner; \
    double inner_der = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x * x);
    double term2 = 0.5 * x * sech2 * inner_der;

    return (float)term1 + term2;
}

typedef struct {
    float* gamma;
    float* beta;
    int d;
} LayerNorm;

static LayerNorm* layernorm_create(int d) {
    LayerNorm* ln = (LayerNorm*)calloc(1, sizeof(*ln));
    if (!ln) return NULL;
    ln->d = d;
    ln->gamma = xmalloc_vec(d);
    ln->beta  = xmalloc_vec(d);
    for (int i=0;i<d;i++){ ln->gamma[i]=1.0f; ln->beta[i]=0.0f; }
    return ln;
}

static void layernorm_inplace(float** x, int T, LayerNorm* ln) {
    const float eps = 1e-5f;
    int d = ln->d;
    for (int t=0;t<T;t++) {
        float mean=0.0f;
        for (int j=0;j<d;j++) mean += x[t][j];
        mean /= (float)d;

        float var=0.0f;
        for (int j=0;j<d;j++){
            float diff = x[t][j] - mean;
            var += diff*diff;
        }
        var /= (float)d;
        float inv = 1.0f / sqrtf(var + eps);

        for (int j=0;j<d;j++){
            float norm = (x[t][j] - mean) * inv;
            x[t][j] = norm * ln->gamma[j] + ln->beta[j];
        }
    }
}

static void layernorm_free(LayerNorm* ln) {
    if (!ln) return;
    free(ln->gamma);
    free(ln->beta);
    free(ln);
}

typedef struct {
    int d;
    int num_layers;    // total layers (hidden + output)
    int* layerSize;    // size of each layer output
    float** W;         // flat weights per layer
    float** b;
} FC_layer;

static FC_layer* ffn_create(int d, int num_hidden, ...) {
    FC_layer* fc = (FC_layer*)calloc(1, sizeof(*fc));
    if (!fc) return NULL;

    fc->d = d;
    fc->num_layers = num_hidden + 1;
    int L = fc->num_layers;

    fc->layerSize = (int*)calloc((size_t)L, sizeof(int));
    fc->W = (float**)calloc((size_t)L, sizeof(float*));
    fc->b = (float**)calloc((size_t)L, sizeof(float*));
    if (!fc->layerSize || !fc->W || !fc->b) { perror("calloc"); exit(1); }

    va_list args;
    va_start(args, num_hidden);

    int in = d;
    for (int l=0; l<num_hidden; l++) {
        int out = va_arg(args, int);
        fc->layerSize[l] = out;
        fc->W[l] = (float*)malloc((size_t)out * in * sizeof(float));
        fc->b[l] = (float*)malloc((size_t)out * sizeof(float));
        if (!fc->W[l] || !fc->b[l]) { perror("malloc"); exit(1); }

        float scale = sqrtf(2.0f / (in + out));
        for (int i=0;i<out*in;i++) fc->W[l][i] = rand_scale(scale);
        for (int i=0;i<out;i++) fc->b[l][i] = 0.0f;
        in = out;
    }

    fc->layerSize[L-1] = d;
    fc->W[L-1] = (float*)malloc((size_t)d * in * sizeof(float));
    fc->b[L-1] = (float*)malloc((size_t)d * sizeof(float));
    if (!fc->W[L-1] || !fc->b[L-1]) { perror("malloc"); exit(1); }

    float scale = sqrtf(2.0f / (in + d));
    for (int i=0;i<d*in;i++) fc->W[L-1][i] = rand_scale(scale);
    for (int i=0;i<d;i++) fc->b[L-1][i] = 0.0f;

    va_end(args);
    return fc;
}

typedef struct{
    float **x;    // x[0..L]  (L = num_layers)
    float **pre;  // pre[0..L-2] only hidden layers
    int L;        // fc->num_layers
    int d;        // fc->d
} easy_fc_cache;

typedef struct {
    float **w;  // w[0..L-1], each is flat [out*in]
    float **b;  // b[0..L-1], each is [out]
    int L;
} fc_cache;

static easy_fc_cache* easy_cache_create(const FC_layer* fc, bool zero)
{
    if (!fc) return NULL;

    easy_fc_cache* c = (easy_fc_cache*)calloc(1, sizeof(*c));
    if (!c) { perror("calloc"); exit(1); }

    c->L = fc->num_layers;
    c->d = fc->d;
ъ
    c->x = (float**)calloc((size_t)(c->L + 1), sizeof(float*));
    if (!c->x) { perror("calloc"); exit(1); }

    c->x[0] = zero ? (float*)calloc((size_t)c->d, sizeof(float))
                   : (float*)malloc((size_t)c->d * sizeof(float));
    if (!c->x[0]) { perror("malloc/calloc"); exit(1); }

    for (int l = 0; l < c->L; l++) {
        int out = fc->layerSize[l];
        c->x[l+1] = zero ? (float*)calloc((size_t)out, sizeof(float))
                         : (float*)malloc((size_t)out * sizeof(float));
        if (!c->x[l+1]) { perror("malloc/calloc"); exit(1); }
    }

    if (c->L >= 2) {
        c->pre = (float**)calloc((size_t)(c->L - 1), sizeof(float*));
        if (!c->pre) { perror("calloc"); exit(1); }

        for (int l = 0; l < c->L - 1; l++) {
            int out = fc->layerSize[l]; 
            c->pre[l] = zero ? (float*)calloc((size_t)out, sizeof(float))
                             : (float*)malloc((size_t)out * sizeof(float));
            if (!c->pre[l]) { perror("malloc/calloc"); exit(1); }
        }
    } else {
        c->pre = NULL;
    }

    return c;
}

static void easy_cache_zero(easy_fc_cache* c)
{
    if (!c) return;
    memset(c->x[0], 0, (size_t)c->d * sizeof(float));
    for (int l = 0; l < c->L; l++) {
        int out = 0; 
        (void)out;
    }
}

static void easy_cache_free(easy_fc_cache* c)
{
    if (!c) return;
    if (c->x) {
        for (int i = 0; i < c->L + 1; i++) free(c->x[i]);
        free(c->x);
    }
    if (c->pre) {
        for (int i = 0; i < c->L - 1; i++) free(c->pre[i]);
        free(c->pre);
    }
    free(c);
}

static fc_cache* grads_create(const FC_layer* fc, bool zero)
{
    if (!fc) return NULL;

    fc_cache* g = (fc_cache*)calloc(1, sizeof(*g));
    if (!g) { perror("calloc"); exit(1); }

    g->L = fc->num_layers;

    g->w = (float**)calloc((size_t)g->L, sizeof(float*));
    g->b = (float**)calloc((size_t)g->L, sizeof(float*));
    if (!g->w || !g->b) { perror("calloc"); exit(1); }

    for (int l = 0; l < g->L; l++) {
        int in  = (l == 0) ? fc->d : fc->layerSize[l-1];
        int out = fc->layerSize[l];

        size_t w_sz = (size_t)out * (size_t)in;
        size_t b_sz = (size_t)out;

        g->w[l] = zero ? (float*)calloc(w_sz, sizeof(float))
                       : (float*)malloc(w_sz * sizeof(float));
        g->b[l] = zero ? (float*)calloc(b_sz, sizeof(float))
                       : (float*)malloc(b_sz * sizeof(float));

        if (!g->w[l] || !g->b[l]) { perror("malloc/calloc"); exit(1); }
    }

    return g;
}

static void grads_zero(fc_cache* g, const FC_layer* fc)
{
    if (!g || !fc) return;
    for (int l = 0; l < g->L; l++) {
        int in  = (l == 0) ? fc->d : fc->layerSize[l-1];
        int out = fc->layerSize[l];
        memset(g->w[l], 0, (size_t)out * (size_t)in * sizeof(float));
        memset(g->b[l], 0, (size_t)out * sizeof(float));
    }
}

static void grads_free(fc_cache* g)
{
    if (!g) return;
    if (g->w) {
        for (int l = 0; l < g->L; l++) free(g->w[l]);
        free(g->w);
    }
    if (g->b) {
        for (int l = 0; l < g->L; l++) free(g->b[l]);
        free(g->b);
    }
    free(g);
}
static float* ffn_forward_one(const float* input, FC_layer* fc,easy_fc_cache * eaz) {
    int d = fc->d;
    float* x = xmalloc_vec(d);
    memcpy(x, input, (size_t)d*sizeof(float));
    memcpy(eaz->x[0],input,sizeof(float)*d);
    int L = fc->num_layers;
    for (int l=0;l<L;l++) {

        int in = (l==0) ? d : fc->layerSize[l-1];
        int out = fc->layerSize[l];
        float* next = xmalloc_vec(out);

        float* W = fc->W[l];
        float* b = fc->b[l];

        for (int j=0;j<out;j++){
            float s = b[j];
            for (int k=0;k<in;k++){
                s += W[j*in + k] * x[k];
            }
            if(l < L-1) eaz->pre[l][j] = s;
            next[j] = (l < L-1) ? gelu(s) : s;
            eaz->x[l+1][j] = next[j];
        }
        free(x);
        x = next;
    }
    return x;
}

void backward(FC_layer * fc , float * loss, fc_cache * qwe, easy_fc_cache * cas){
    int q = fc->d;
    float * cur_loss = xmalloc_vec(q);
    memcpy(cur_loss, loss, q * sizeof(float));

    int l = fc->num_layers - 1;
    while (l >= 0) {
        int in_size  = (l == 0) ? q : fc->layerSize[l-1];
        int out_size = fc->layerSize[l];

        for(int i = 0; i < out_size; ++i) {
            for(int j = 0; j < in_size; ++j) {
                qwe->w[l][i*in_size + j] = cur_loss[i] * cas->x[l][j];
            }
            qwe->b[l][i] = cur_loss[i];
        }

        float * dx = xmalloc_vec(in_size);
        for (int j = 0; j < in_size; j++) {
            float sum = 0.0f;
            for (int i = 0; i < out_size; i++) {
                sum += fc->W[l][i*in_size + j] * cur_loss[i];
            }
            dx[j] = sum;
        }

        if (l > 0) {
            for (int j = 0; j < in_size; j++) {
                dx[j] *= gelu1(cas->pre[l-1][j]);
            }
        }

        free(cur_loss); 
        cur_loss = dx;  
        l--;
    }
    free(cur_loss); 
}

void apply_grads(fc_cache* qwe, FC_layer* fc, float lr){
    int L = fc->num_layers;
    for (int l=0; l<L; l++){
        int in  = (l==0) ? fc->d : fc->layerSize[l-1];
        int out = fc->layerSize[l];

        float* dW = qwe->w[l];
        float* db = qwe->b[l];

        for (int i=0; i<out; i++){
            for (int j=0; j<in; j++){
                fc->W[l][i*in + j] -= lr * dW[i*in + j];
            }
            fc->b[l][i] -= lr * db[i];
        }
    }
}


static void ffn_free(FC_layer* fc) {
    if (!fc) return;
    for (int l=0;l<fc->num_layers;l++){
        free(fc->W[l]);
        free(fc->b[l]);
    }
    free(fc->W);
    free(fc->b);
    free(fc->layerSize);
    free(fc);
}
static void positional_encoding_inplace(float** x, int T, int d_model) {
    for (int t=0;t<T;t++){
        for (int j=0;j<d_model;j++){
            float freq = 1.0f / powf(10000.0f, (float)j / (float)d_model);
            if ((j & 1) == 0) x[t][j] += sinf((float)t * freq);
            else              x[t][j] += cosf((float)t * freq);
        }
    }
}

// kv cache 
typedef struct {
    int max_tokens;
    int d_model;
    float** k; // [max_tokens][d_model]
    float** v; // [max_tokens][d_model]
} kv_cache;

static kv_cache* cache_create(int max_tokens, int d_model) {
    kv_cache* c = (kv_cache*)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->max_tokens = max_tokens;
    c->d_model = d_model;
    c->k = (float**)calloc((size_t)max_tokens, sizeof(float*));
    c->v = (float**)calloc((size_t)max_tokens, sizeof(float*));
    if (!c->k || !c->v) { perror("calloc"); exit(1); }
    return c;
}

static void cache_free(kv_cache* c) {
    if (!c) return;
    for (int i=0;i<c->max_tokens;i++){
        free(c->k[i]);
        free(c->v[i]);
    }
    free(c->k);
    free(c->v);
    free(c);
}

typedef struct {
    float*** W_q;  // [H][Dh][D]
    float*** W_k;  // [H][Dh][D]
    float*** W_v;  // [H][Dh][D]
    float**  W_out; // [D][D]  (out_dim x in_dim) for matvec
    int d_model;
    int num_heads;
    int d_head;
} MultiHeadAttention;

static float*** alloc_W_hdhD(int H, int Dh, int D) {
    float*** W = (float***)malloc((size_t)H * sizeof(float**));
    if (!W) { perror("malloc"); exit(1); }
    for (int h=0;h<H;h++){
        W[h] = (float**)malloc((size_t)Dh * sizeof(float*));
        if (!W[h]) { perror("malloc"); exit(1); }
        for (int r=0;r<Dh;r++){
            W[h][r] = (float*)malloc((size_t)D * sizeof(float));
            if (!W[h][r]) { perror("malloc"); exit(1); }
        }
    }
    return W;
}

static void free_W_hdhD(float*** W, int H, int Dh) {
    if (!W) return;
    for (int h=0;h<H;h++){
        for (int r=0;r<Dh;r++) free(W[h][r]);
        free(W[h]);
    }
    free(W);
}

static MultiHeadAttention* mha_create(int d_model, int num_heads) {
    if (d_model % num_heads != 0) {
        fprintf(stderr, "пшш\n");
        return NULL;
    }

    MultiHeadAttention* mha = (MultiHeadAttention*)calloc(1, sizeof(*mha));
    if (!mha) return NULL;

    mha->d_model = d_model;
    mha->num_heads = num_heads;
    mha->d_head = d_model / num_heads;

    int H = mha->num_heads;
    int Dh = mha->d_head;
    int D = mha->d_model;

    mha->W_q = alloc_W_hdhD(H, Dh, D);
    mha->W_k = alloc_W_hdhD(H, Dh, D);
    mha->W_v = alloc_W_hdhD(H, Dh, D);

    float scale = sqrtf(2.0f / (D + Dh));
    for (int h=0;h<H;h++){
        for (int r=0;r<Dh;r++){
            for (int c=0;c<D;c++){
                mha->W_q[h][r][c] = rand_scale(scale);
                mha->W_k[h][r][c] = rand_scale(scale);
                mha->W_v[h][r][c] = rand_scale(scale);
            }
        }
    }

    mha->W_out = xmalloc_mat(D, D, false);
    float scale_out = sqrtf(2.0f / (D + D));
    for (int i=0;i<D;i++) for (int j=0;j<D;j++) mha->W_out[i][j] = rand_scale(scale_out);

    return mha;
}

static void mha_free(MultiHeadAttention* mha) {
    if (!mha) return;
    free_W_hdhD(mha->W_q, mha->num_heads, mha->d_head);
    free_W_hdhD(mha->W_k, mha->num_heads, mha->d_head);
    free_W_hdhD(mha->W_v, mha->num_heads, mha->d_head);
    free_mat(mha->W_out, mha->d_model);
    free(mha);
}

static float* project_all_heads(const float* x, MultiHeadAttention* mha, float*** W) {
    int D = mha->d_model, H = mha->num_heads, Dh = mha->d_head;
    float* out = xcalloc_vec(D);
    for (int h=0; h<H; h++) {
        float* tmp = matvec(x, W[h], D, Dh);            
        memcpy(out + h*Dh, tmp, (size_t)Dh*sizeof(float)); 
        free(tmp);
    }
    return out;
}


typedef struct {
    int T, D, H, Dh;
    float** X;        // [T][D]
    float*** Q;       // [H][T][Dh] 
    float*** K;       // [H][T][Dh]
    float*** V;       // [H][T][Dh]
    float*** A;       // [H][T][T] 
    float** O;        // [T][D]   


} AttnCache;

AttnCache* create_attn_cache(int T, int D, int H) {
    AttnCache* cache = (AttnCache*)malloc(sizeof(AttnCache));
    cache->T = T;
    cache->D = D;
    cache->H = H;
    cache->Dh = D / H; 
    
    cache->X = xmalloc_mat(T, D, false);
    
    cache->Q = (float***)malloc(H * sizeof(float**));
    cache->K = (float***)malloc(H * sizeof(float**));
    cache->V = (float***)malloc(H * sizeof(float**));
    cache->A = (float***)malloc(H * sizeof(float**));
    
    for (int h = 0; h < H; h++) {
        cache->Q[h] = xmalloc_mat(T, cache->Dh, false);
        cache->K[h] = xmalloc_mat(T, cache->Dh, false);
        cache->V[h] = xmalloc_mat(T, cache->Dh, false);
        cache->A[h] = xmalloc_mat(T, T, false);
    }
    
    cache->O = xmalloc_mat(T, D, false);
    
    return cache;
}

void free_attn_cache(AttnCache* cache) {
    if (!cache) return;
    
    free_mat(cache->X, cache->T);
    
    for (int h = 0; h < cache->H; h++) {
        free_mat(cache->Q[h], cache->T);
        free_mat(cache->K[h], cache->T);
        free_mat(cache->V[h], cache->T);
        free_mat(cache->A[h], cache->T);
    }
    free(cache->Q);
    free(cache->K);
    free(cache->V);
    free(cache->A);
    
    free_mat(cache->O, cache->T);
    free(cache);
}

static float** mha_self_attention(float** x, int T, MultiHeadAttention* mha, 
                                   bool add_posenc, AttnCache* cache) {
    int D = mha->d_model;
    int H = mha->num_heads;
    int Dh = mha->d_head;  
    
    if (cache->T < T || cache->D != D || cache->H != H) {
        fprintf(stderr, "Cache size mismatch\n");
        return NULL;
    }
    
    for (int i = 0; i < T; i++) {
        memcpy(cache->X[i], x[i], D * sizeof(float));
    }
    
    if (add_posenc) {
        positional_encoding_inplace(x, T, D);
    }
    
    float** out = xmalloc_mat(T, D, true);  
    
    for (int h = 0; h < H; h++) {
        for (int t = 0; t < T; t++) {
            float* q = matvec(x[t], mha->W_q[h], D, Dh);
            memcpy(cache->Q[h][t], q, Dh * sizeof(float));
            free(q);
            
            float* k = matvec(x[t], mha->W_k[h], D, Dh);
            memcpy(cache->K[h][t], k, Dh * sizeof(float));
            free(k);
            
            float* v = matvec(x[t], mha->W_v[h], D, Dh);
            memcpy(cache->V[h][t], v, Dh * sizeof(float));
            free(v);
        }
        
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                float dot = 0.0f;
                for (int d = 0; d < Dh; d++) {
                    dot += cache->Q[h][i][d] * cache->K[h][j][d];
                }
                cache->A[h][i][j] = dot / sqrtf((float)Dh);
            }
            
            softmax_inplace(cache->A[h][i], T);
            
            float* head_out = (float*)calloc(Dh, sizeof(float));
            for (int j = 0; j < T; j++) {
                float weight = cache->A[h][i][j];
                for (int d = 0; d < Dh; d++) {
                    head_out[d] += weight * cache->V[h][j][d];
                }
            }
            
            int offset = h * Dh;  
            for (int d = 0; d < Dh; d++) {
                out[i][offset + d] += head_out[d];  
            }
            
            free(head_out);
        }
    }
    
    for (int i = 0; i < T; i++) {
        memcpy(cache->O[i], out[i], D * sizeof(float));
    }
    
    for (int t = 0; t < T; t++) {
        float* projected = matvec(out[t], mha->W_out, D, D);
        free(out[t]);          
        out[t] = projected;  
    }
    
    return out;
}

void backprop_attention(float ** loss /*T D = num d_embed*/,int T, MultiHeadAttention * mha, AttnCache * cache_attn, float lr, bool is_masked){
    int H = mha->num_heads;
    int Dh = mha->d_head;
    int D = mha->d_model;
    float ** d0_conccat = xmalloc_mat(T,D,true);

    for(int i = 0 ; i < T ; ++i){
        for(int j = 0 ; j < D ; ++j){
            float qw = 0.0f; // concat [i][j]
            for(int k = 0 ; k < D ; k++){
                qw += mha->W_out[k][j] * loss[i][k];
            }
            d0_conccat[i][j] = qw;
        }
    }

    float ** dX = xmalloc_mat(T,D,true);
    for (int h = 0; h < H; h++) {
    float** dv = xmalloc_mat(T, Dh, true); 
    float** dA = xmalloc_mat(T, T, true);
    float ** dS = xmalloc_mat(T,T,true); 
    float ** dQ = xmalloc_mat(T,Dh,true);
    float ** dK = xmalloc_mat(T,Dh,true);
        for (int j=0; j<T; j++) {
            for (int d=0; d<Dh; d++) {
                float sum = 0;
                for (int i=0; i<T; i++) {
                    float dO = d0_conccat[i][h*Dh + d];
                    sum += cache_attn->A[h][i][j] * dO;
                }
                dv[j][d] = sum;
            }
        }
        // dA[i][j] = sum_d dO[i][d] * V[j][d]
        for (int i=0; i<T; i++){
            for (int j=0; j<T; j++){
                float sum = 0.0f;
                for (int d=0; d<Dh; d++){
                    float dO = d0_conccat[i][h*Dh + d];
                    sum += dO * cache_attn->V[h][j][d];
                }
                dA[i][j] = sum;
            }
        }
        for (int i = 0; i < T; i++) { 
            float dot = 0.0f;
            for (int k = 0; k < T; k++) {
                dot += dA[i][k] * cache_attn->A[h][i][k];
            }
            
            for (int j = 0; j < T; j++) {
                float a_val = cache_attn->A[h][i][j];
                dS[i][j] = a_val * (dA[i][j] - dot);
                if (is_masked) {
                    if (j > i) dS[i][j] = 0.0f;
                }
            }
        }
        for (int i = 0; i < T; i++) {   
            for (int j = 0; j < Dh; j++) {  
                float sum = 0.0f;
                for (int k = 0; k < T; k++) {
                    sum += dS[i][k] * cache_attn->K[h][k][j];
                }
                dQ[i][j] = sum / sqrtf((float)Dh);
            }
        }
        for (int i = 0; i < T; i++) {       
            for (int j = 0; j < Dh; j++) {  
                float sum = 0.0f;
                for (int k = 0; k < T; k++) {
                    sum += dS[k][i] * cache_attn->Q[h][k][j];
                }
                dK[i][j] = sum / sqrtf((float)Dh);
            }
        }

        for (int t = 0; t < T; t++) {
            for (int i = 0; i < D; i++) {
                float add = 0.0f;
                for (int j = 0;  j < Dh; j++) {
                    add += dQ[t][j] * mha->W_q[h][j][i];
                    add += dK[t][j] * mha->W_k[h][j][i];
                    add += dv[t][j] * mha->W_v[h][j][i];
                }
                dX[t][i] += add;
            }
        }
        for(int i = 0 ; i < D; ++i){
            for(int j = 0 ; j < Dh ; ++j){
                float q = 0.0f, k = 0.0f, v = 0.0f;
                for(int t = 0 ; t < T ; t++){
                    float x = cache_attn->X[t][i];
                    q += x * dQ[t][j];
                    k += x * dK[t][j];
                    v += x * dv[t][j]; 
                } 
                mha->W_k[h][j][i] -= lr * k;
                mha->W_q[h][j][i] -= lr * q;
                mha->W_v[h][j][i] -= lr * v;
                
            }
        }
        for(int i = 0 ; i < T ; ++i){
            free(dv[i]); free(dS[i]); free(dQ[i]); free(dK[i]); free(dA[i]);
        }  
        free(dv); free(dA); free(dS); free(dQ); free(dK);
    }
    for (int i = 0; i < D; i++) {           
        for (int j = 0; j < D; j++) {     
            float sum = 0.0f;
            for (int k = 0; k < T; k++) {
                sum += cache_attn->O[k][i] * loss[k][j];
            }
            mha->W_out[i][j] -= lr * sum;
        }
    }
    
    for(int i = 0 ; i < T ; ++i){
        memcpy(loss[i],dX[i],sizeof(float)*D);
    }
    for(int i = 0 ; i < T ; ++i){
        free(d0_conccat[i]); free(dX[i]);
    }
    free(d0_conccat); free(dX);
}
typedef struct {
  int Tq, Tkv, D, H, Dh;
  float** X_dec; // [Tq][D]
  float** X_enc; // [Tkv][D]
  float*** Q;    // [H][Tq][Dh]
  float*** K_enc;    // [H][Tkv][Dh]
  float*** V_enc;    // [H][Tkv][Dh]
  float*** A;    // [H][Tq][Tkv]
  float** Ocat;  // [Tq][D]
} CrossCache;

void free_cross_cache(CrossCache* cache);
CrossCache* create_cross_cache(int Tq, int Tkv, int D, int H) {
    CrossCache* cache = (CrossCache*)malloc(sizeof(CrossCache));
    if (!cache) return NULL;
    
    if (D % H != 0) {
        fprintf(stderr, "Error: D (%d) must be divisible by H (%d)\n", D, H);
        free(cache);
        return NULL;
    }
    
    cache->Tq = Tq;
    cache->Tkv = Tkv;
    cache->D = D;
    cache->H = H;
    cache->Dh = D / H;
    
    cache->X_dec = xmalloc_mat(Tq, D, false);
    if (!cache->X_dec) goto cleanup;
    
    cache->X_enc = xmalloc_mat(Tkv, D, false);
    if (!cache->X_enc) goto cleanup;
    
    cache->Q = (float***)malloc(H * sizeof(float**));
    if (!cache->Q) goto cleanup;
    
    for (int h = 0; h < H; h++) {
        cache->Q[h] = xmalloc_mat(Tq, cache->Dh, false);
        if (!cache->Q[h]) {
            for (int hh = 0; hh < h; hh++) {
                free_mat(cache->Q[hh], Tq);
            }
            free(cache->Q);
            goto cleanup;
        }
    }
    
    cache->K_enc = (float***)malloc(H * sizeof(float**));
    if (!cache->K_enc) goto cleanup;
    
    for (int h = 0; h < H; h++) {
        cache->K_enc[h] = xmalloc_mat(Tkv, cache->Dh, false);
        if (!cache->K_enc[h]) {
            for (int hh = 0; hh < h; hh++) {
                free_mat(cache->K_enc[hh], Tkv);
            }
            free(cache->K_enc);
            goto cleanup;
        }
    }
    
    cache->V_enc = (float***)malloc(H * sizeof(float**));
    if (!cache->V_enc) goto cleanup;
    
    for (int h = 0; h < H; h++) {
        cache->V_enc[h] = xmalloc_mat(Tkv, cache->Dh, false);
        if (!cache->V_enc[h]) {
            for (int hh = 0; hh < h; hh++) {
                free_mat(cache->V_enc[hh], Tkv);
            }
            free(cache->V_enc);
            goto cleanup;
        }
    }
    
    cache->A = (float***)malloc(H * sizeof(float**));
    if (!cache->A) goto cleanup;
    
    for (int h = 0; h < H; h++) {
        cache->A[h] = xmalloc_mat(Tq, Tkv, false);
        if (!cache->A[h]) {
            for (int hh = 0; hh < h; hh++) {
                free_mat(cache->A[hh], Tq);
            }
            free(cache->A);
            goto cleanup;
        }
    }
    
    cache->Ocat = xmalloc_mat(Tq, D, false);
    if (!cache->Ocat) goto cleanup;
    
    return cache;

cleanup:
    fprintf(stderr, "Error: Failed to allocate memory for CrossCache\n");
    free_cross_cache(cache);
    return NULL;
}

void free_cross_cache(CrossCache* cache) {
    if (!cache) return;

    if (cache->X_dec) {
        free_mat(cache->X_dec, cache->Tq);
        cache->X_dec = NULL;
    }
    
    if (cache->X_enc) {
        free_mat(cache->X_enc, cache->Tkv);
        cache->X_enc = NULL;
    }
    
    if (cache->Q) {
        for (int h = 0; h < cache->H; h++) {
            if (cache->Q[h]) {
                free_mat(cache->Q[h], cache->Tq);
                cache->Q[h] = NULL;
            }
        }
        free(cache->Q);
        cache->Q = NULL;
    }
    
    if (cache->K_enc) {
        for (int h = 0; h < cache->H; h++) {
            if (cache->K_enc[h]) {
                free_mat(cache->K_enc[h], cache->Tkv);
                cache->K_enc[h] = NULL;
            }
        }
        free(cache->K_enc);
        cache->K_enc = NULL;
    }
    
    if (cache->V_enc) {
        for (int h = 0; h < cache->H; h++) {
            if (cache->V_enc[h]) {
                free_mat(cache->V_enc[h], cache->Tkv);
                cache->V_enc[h] = NULL;
            }
        }
        free(cache->V_enc);
        cache->V_enc = NULL;
    }
    
    if (cache->A) {
        for (int h = 0; h < cache->H; h++) {
            if (cache->A[h]) {
                free_mat(cache->A[h], cache->Tq);
                cache->A[h] = NULL;
            }
        }
        free(cache->A);
        cache->A = NULL;
    }
    
    if (cache->Ocat) {
        free_mat(cache->Ocat, cache->Tq);
        cache->Ocat = NULL;
    }
    
    free(cache);
}

void reset_cross_cache(CrossCache* cache) {
    if (!cache) return;
    
    for (int t = 0; t < cache->Tq; t++) {
        memset(cache->X_dec[t], 0, cache->D * sizeof(float));
    }
    
    for (int t = 0; t < cache->Tkv; t++) {
        memset(cache->X_enc[t], 0, cache->D * sizeof(float));
    }
    
    for (int h = 0; h < cache->H; h++) {
        for (int t = 0; t < cache->Tq; t++) {
            memset(cache->Q[h][t], 0, cache->Dh * sizeof(float));
        }
    }
    
    for (int h = 0; h < cache->H; h++) {
        for (int t = 0; t < cache->Tkv; t++) {
            memset(cache->K_enc[h][t], 0, cache->Dh * sizeof(float));
        }
    }
    
    for (int h = 0; h < cache->H; h++) {
        for (int t = 0; t < cache->Tkv; t++) {
            memset(cache->V_enc[h][t], 0, cache->Dh * sizeof(float));
        }
    }
    
    for (int h = 0; h < cache->H; h++) {
        for (int t = 0; t < cache->Tq; t++) {
            memset(cache->A[h][t], 0, cache->Tkv * sizeof(float));
        }
    }
    
    for (int t = 0; t < cache->Tq; t++) {
        memset(cache->Ocat[t], 0, cache->D * sizeof(float));
    }
}


void backprop_cross_attention( float **loss, /*[T_q][D]*/, int T_q, int T_kv, 
    MultiHeadAttention *mha, CrossCache *cache,
     float lr, float **dX_dec, float **dX_enc) 
{
    int H = mha->num_heads;
    int Dh = mha->d_head;
    int D = mha->d_model;
    float ** d0_concat = xmalloc_mat(T_q,D,true);

    for(int i = 0 ; i < T_q ; ++i){
        for(int j = 0 ; j < D ; ++j){
            float qw = 0.0f; // concat [i][j]
            for(int k = 0 ; k < D ; k++){
                qw += mha->W_out[k][j] * loss[i][k];
            }
            d0_concat[i][j] = qw;
        }
    }
    for (int h = 0; h < H; h++) {
    float** dv = xmalloc_mat(T_kv, Dh, true); 
    float** dA = xmalloc_mat(T_q, T_kv, true);
    float ** dS = xmalloc_mat(T_q,T_kv,true); 
    float ** dQ = xmalloc_mat(T_q,Dh,true);
    float ** dK = xmalloc_mat(T_kv,Dh,true);

        for (int j = 0; j < T_kv; j++) {
            for (int d = 0; d < Dh; d++) {
                float sum = 0;
                for (int i = 0; i < T_q; i++) {
                    sum += cache->A[h][i][j] * d0_concat[i][h*Dh + d];
                }
                dv[j][d] = sum;
            }
        }

        for (int i = 0; i < T_q; i++) {
            for (int j = 0; j < T_kv; j++) {
                float sum = 0;
                for (int d = 0; d < Dh; d++) {
                    sum += d0_concat[i][h*Dh + d] * cache->V_enc[h][j][d];
                }
                dA[i][j] = sum;
            }
        }

        for (int i = 0; i < T_q; i++) { 
            float dot = 0.0f;
            for (int k = 0; k < T_kv; k++) {
                dot += dA[i][k] * cache->A[h][i][k];
            }
            
            for (int j = 0; j < T_kv; j++) {
                float a_val = cache->A[h][i][j];
                dS[i][j] = a_val * (dA[i][j] - dot);

            }
        }
        // 4. dQ [T_q][Dh]
        for (int i = 0; i < T_q; i++) {
            for (int j = 0; j < Dh; j++) {
                float sum = 0;
                for (int k = 0; k < T_kv; k++) {
                    sum += dS[i][k] * cache->K_enc[h][k][j];
                }
                dQ[i][j] = sum / sqrtf(Dh);
            }
        }

        // 5. dK [T_kv][Dh]
        for (int i = 0; i < T_kv; i++) {
            for (int j = 0; j < Dh; j++) {
                float sum = 0;
                for (int k = 0; k < T_q; k++) {
                    sum += dS[k][i] * cache->Q[h][k][j];
                }
                dK[i][j] = sum / sqrtf(Dh);
            }
        }


        for (int t = 0; t < T_q; t++) {
            for (int i = 0; i < D; i++) {
                float add = 0;
                for (int j = 0; j < Dh; j++) 
                    add += dQ[t][j] * mha->W_q[h][j][i];
                dX_dec[t][i] += add;
            }
        }

        for (int t = 0; t < T_kv; t++) {
            for (int i = 0; i < D; i++) {
                float add = 0;
                for (int j = 0; j < Dh; j++) {
                    add += dK[t][j] * mha->W_k[h][j][i];
                    add += dv[t][j] * mha->W_v[h][j][i];
                }
                dX_enc[t][i] += add;
            }
        }

        for(int i = 0 ; i < D; ++i){
            for(int j = 0 ; j < Dh ; ++j){
                float q = 0.0f, k = 0.0f, v = 0.0f;
                for(int t = 0 ; t < T_q ; t++){
                    float x = cache->X_dec[t][i];
                    q += x * dQ[t][j];
                } 
                for(int t = 0 ; t < T_kv; ++t){
                    float x = cache->X_enc[t][i];
                    k+= x * dK[t][j];
                    v+= x * dv[t][j];
                }
                mha->W_k[h][j][i] -= lr * k;
                mha->W_q[h][j][i] -= lr * q;
                mha->W_v[h][j][i] -= lr * v;
                
            }
        }
        for(int i = 0 ; i < T_q ; ++i) { free(dQ[i]); free(dS[i]); free(dA[i]); }
        for(int i = 0 ; i < T_kv; ++i) { free(dv[i]); free(dK[i]); }
        free(dv); free(dA); free(dS); free(dQ); free(dK);
    }
    for (int i = 0; i < D; i++) {           
        for (int j = 0; j < D; j++) {     
            float sum = 0.0f;
            for (int k = 0; k < T_q; k++) {
                sum += cache->Ocat[k][i] * loss[k][j];
            }
            mha->W_out[i][j] -= lr * sum;
        }
    }
    
    for(int i = 0 ; i < T_q ; ++i){
        free(d0_concat[i]); 
    }
    free(d0_concat);
    
}

static float** mha_masked_self_attention(float** x, int T, MultiHeadAttention* mha, kv_cache* cache, bool add_posenc) {
    int D = mha->d_model;
    int H = mha->num_heads;
    int Dh = mha->d_head;

    if (add_posenc) positional_encoding_inplace(x, T, D);

    float** out = xmalloc_mat(T, D, true);

    for (int t=0; t<T; t++) {
        if (cache) {
            free(cache->k[t]);
            free(cache->v[t]);
            cache->k[t] = project_all_heads(x[t], mha, mha->W_k);
            cache->v[t] = project_all_heads(x[t], mha, mha->W_v);
        }

        for (int h=0; h<H; h++) {
            float* q = matvec(x[t], mha->W_q[h], D, Dh);

            float* scores = xmalloc_vec(T);
            for (int j=0; j<T; j++) {
                if (j > t) { scores[j] = -INFINITY; continue; }

                const float* Kj_concat = cache ? cache->k[j] : NULL;
                float dot = 0.0f;

                if (Kj_concat) {
                    int off = h*Dh;
                    for (int d=0; d<Dh; d++) dot += q[d] * Kj_concat[off + d];
                } else {
                    float* k = matvec(x[j], mha->W_k[h], D, Dh);
                    for (int d=0; d<Dh; d++) dot += q[d] * k[d];
                    free(k);
                }

                scores[j] = dot / sqrtf((float)Dh);
            }

            softmax_inplace(scores, T);

            int off = h*Dh;
            for (int j=0; j<=t; j++) {
                float w = scores[j];
                const float* Vj_concat = cache ? cache->v[j] : NULL;
                if (Vj_concat) {
                    for (int d=0; d<Dh; d++) out[t][off + d] += w * Vj_concat[off + d];
                } else {
                    float* v = matvec(x[j], mha->W_v[h], D, Dh);
                    for (int d=0; d<Dh; d++) out[t][off + d] += w * v[d];
                    free(v);
                }
            }

            free(scores);
            free(q);
        }

        float* projected = matvec(out[t], mha->W_out, D, D);
        free(out[t]);
        out[t] = projected;
    }

    return out;
}

static float** mha_cross_attention(float** x_dec, int Tdec,
                                   kv_cache* cache_enc, int Tenc,
                                   MultiHeadAttention* mha, bool add_posenc_to_dec) {
    int D = mha->d_model;
    int H = mha->num_heads;
    int Dh = mha->d_head;

    if (add_posenc_to_dec) positional_encoding_inplace(x_dec, Tdec, D);

    float** out = xmalloc_mat(Tdec, D, true);

    for (int t=0; t<Tdec; t++) {
        for (int h=0; h<H; h++) {
            float* q = matvec(x_dec[t], mha->W_q[h], D, Dh);
            float* scores = xmalloc_vec(Tenc);

            for (int j=0; j<Tenc; j++) {
                const float* Kj = cache_enc->k[j];
                float dot=0.0f;
                int off = h*Dh;
                for (int d=0; d<Dh; d++) dot += q[d] * Kj[off + d];
                scores[j] = dot / sqrtf((float)Dh);
            }
            softmax_inplace(scores, Tenc);

            int off = h*Dh;
            for (int j=0; j<Tenc; j++) {
                float w = scores[j];
                const float* Vj = cache_enc->v[j];
                for (int d=0; d<Dh; d++) out[t][off + d] += w * Vj[off + d];
            }

            free(scores);
            free(q);
        }

        float* projected = matvec(out[t], mha->W_out, D, D);
        free(out[t]);
        out[t] = projected;
    }

    return out;
}
// ....... пока что пишу не читать.......
static float** encoder_block(float** x, int T, MultiHeadAttention* mha,
                             LayerNorm* ln1, LayerNorm* ln2,
                             FC_layer* ffn,
                             kv_cache* cache_out ) {
    int D = mha->d_model;

    float** att = mha_self_attention(x, T, mha, true);

    for (int t=0;t<T;t++){
        for (int j=0;j<D;j++) att[t][j] += x[t][j];
    }

    layernorm_inplace(att, T, ln1);

    float** ff = xmalloc_mat(T, D, false);
    for (int t=0;t<T;t++){
        float* y = ffn_forward_one(att[t], ffn);
        memcpy(ff[t], y, (size_t)D*sizeof(float));
        free(y);
    }

    for (int t=0;t<T;t++){
        for (int j=0;j<D;j++) ff[t][j] += att[t][j];
    }

    layernorm_inplace(ff, T, ln2);

    if (cache_out) {
        for (int t=0; t<T; t++) {
            free(cache_out->k[t]);
            free(cache_out->v[t]);
            cache_out->k[t] = project_all_heads(ff[t], mha, mha->W_k);
            cache_out->v[t] = project_all_heads(ff[t], mha, mha->W_v);
        }
    }

    free_mat(att, T);
    return ff; 
}

static float** decoder_block(float** x, int Tdec,
                             MultiHeadAttention* mha_self,
                             MultiHeadAttention* mha_cross,
                             LayerNorm* ln1, LayerNorm* ln2, LayerNorm* ln3,
                             FC_layer* ffn,
                             kv_cache* cache_self,     
                             kv_cache* cache_enc, int Tenc) {

    int D = mha_self->d_model;

    float** self_att = mha_masked_self_attention(x, Tdec, mha_self, cache_self, true);

    for (int t=0;t<Tdec;t++){
        for (int j=0;j<D;j++) self_att[t][j] += x[t][j];
    }
    layernorm_inplace(self_att, Tdec, ln1);

    float** cross_att = mha_cross_attention(self_att, Tdec, cache_enc, Tenc, mha_cross, false);

    for (int t=0;t<Tdec;t++){
        for (int j=0;j<D;j++) cross_att[t][j] += self_att[t][j];
    }
    layernorm_inplace(cross_att, Tdec, ln2);

    float** ff = xmalloc_mat(Tdec, D, false);
    for (int t=0;t<Tdec;t++){
        float* y = ffn_forward_one(cross_att[t], ffn);
        memcpy(ff[t], y, (size_t)D*sizeof(float));
        free(y);
    }

    for (int t=0;t<Tdec;t++){
        for (int j=0;j<D;j++) ff[t][j] += cross_att[t][j];
    }
    layernorm_inplace(ff, Tdec, ln3);

    free_mat(self_att, Tdec);
    free_mat(cross_att, Tdec);
    return ff;
}
