#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <time.h>
#include <stdbool.h>
#define M_PI 3.14159265358979323846f
typedef struct{
    int x, d;
    int num_layers;
    float **  wf, ** bf;
    float ** wi,** bi;
    float ** wc, ** bc;
    float ** w0, ** b0;
    float ** wy, ** by;
} cell;
static float rand_scale(float s) {
    return ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * s;
}
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float d_sigmoid(float out_sigmoid) {
    return out_sigmoid * (1.0f - out_sigmoid);
}
static float* xmalloc_vec(int n) {
    float* p = (float*)malloc((size_t)n * sizeof(float));
    if (!p) { perror("malloc"); exit(1); }
    return p;
}
cell * create_cell(int d,int x,int n){

    cell *rat = calloc(1, sizeof(cell));
    rat->d = d;
    rat->x = x;
    rat->num_layers = n;
    float pofig = 0.07;
    rat->wy = (float **)malloc(n*sizeof(float*)); rat->w0 = (float **)malloc(n*sizeof(float*)); rat->wc = (float **)malloc(n*sizeof(float*)); rat->wi = (float **)malloc(n*sizeof(float*)); 
    rat->wf = (float **)malloc(n*sizeof(float*));
    rat->by = (float **)malloc(n*sizeof(float*)); rat->b0 = (float **)malloc(n*sizeof(float*)); rat->bc = (float **)malloc(n*sizeof(float*)); rat->bi = (float **)malloc(n*sizeof(float*));
     rat->bf = (float **)malloc(n*sizeof(float*));
    for (int j = 0; j < n; ++j) {

            int in = (j == 0) ? x : d;   

            rat->wf[j] = malloc(d * (in + d) * sizeof(float));
            rat->wi[j] = malloc(d * (in + d) * sizeof(float));
            rat->wc[j] = malloc(d * (in + d) * sizeof(float));
            rat->w0[j] = malloc(d * (in + d) * sizeof(float));

            rat->bf[j] = malloc(d * sizeof(float));
            rat->bi[j] = malloc(d * sizeof(float));
            rat->bc[j] = malloc(d * sizeof(float));
            rat->b0[j] = malloc(d * sizeof(float));

            rat->wy[j] = malloc(d * d * sizeof(float));
            rat->by[j] = malloc(d * sizeof(float));

    for (int i = 0; i < d * (in + d); ++i) {
        rat->wf[j][i] = rand_scale(pofig);
        rat->wi[j][i] = rand_scale(pofig);
        rat->wc[j][i] = rand_scale(pofig);
        rat->w0[j][i] = rand_scale(pofig);
    }

    for (int i = 0; i < d; ++i) {
        rat->bf[j][i] = rand_scale(pofig);
        rat->bi[j][i] = rand_scale(pofig);
        rat->bc[j][i] = rand_scale(pofig);
        rat->b0[j][i] = rand_scale(pofig);
        rat->by[j][i] = rand_scale(pofig);
    }

    for (int i = 0; i < d*d; ++i) {
        rat->wy[j][i] = rand_scale(pofig);
    }
    }
    return rat;
}
void free_cell(cell * q){
    int x = q->x;
    int n = q->num_layers;
    int d = q->d;
    for(int j = 0 ; j < n ; ++ j){
        free(q->b0[j]); free(q->bc[j]); free(q->bf[j]); free(q->bi[j]); free(q->by[j]);
        
        free(q->w0[j]); free(q->wc[j]); free(q->wf[j]); free(q->wi[j]); free(q->wy[j]);
    }
    free(q->b0); free(q->bc); free(q->bf); free(q->bi); free(q->by);
        
    free(q->w0); free(q->wc); free(q->wf); free(q->wi); free(q->wy);
    free(q);
}
typedef struct {
    float **x;      // вход
    float **h_prev;
    float **c_prev;

    float **f;
    float **i;
    float **o;
    float **g;

    float **h;
    float **c;
} cache;
cache* cache_init(int T) {
    cache* k = malloc(sizeof(cache));

    k->x      = malloc(T * sizeof(float*));
    k->h_prev = malloc(T * sizeof(float*));
    k->c_prev = malloc(T * sizeof(float*));

    k->f = malloc(T * sizeof(float*));
    k->i = malloc(T * sizeof(float*));
    k->o = malloc(T * sizeof(float*));
    k->g = malloc(T * sizeof(float*));

    k->h = malloc(T * sizeof(float*));
    k->c = malloc(T * sizeof(float*));

    return k;
}
void cache_free(cache* k, int T) {
    for (int t = 0; t < T; ++t) {
        free(k->x[t]);
        free(k->h_prev[t]);
        free(k->c_prev[t]);

        free(k->f[t]);
        free(k->i[t]);
        free(k->o[t]);
        free(k->g[t]);

        free(k->h[t]);
        free(k->c[t]);
    }

    free(k->x);
    free(k->h_prev);
    free(k->c_prev);

    free(k->f);
    free(k->i);
    free(k->o);
    free(k->g);

    free(k->h);
    free(k->c);

    free(k);
}

float * copy(float * x,int sizex){
    float * q = xmalloc_vec(sizex);
    memcpy(q,x,sizex*sizeof(float));
    return q;
}
void forvard(float * x , int x_embed, float * h ,float * c_prev, int d , cell * lstm , int yoo /*типа в каком слою чувак*/,
             float * y_out, float * c_out , float * h_out, cache * kesh , int em /*типа какая итерация наврено (типа чтобы к кэшу норм обращаться)*/)
{
    kesh->x[em] = copy(x,x_embed);
    kesh->h_prev[em] = copy(h,d);
    kesh->c_prev[em] = copy(c_prev,d);
    float * f = xmalloc_vec(d);
    for ( int i = 0 ; i < d ; ++ i){

        float sum = lstm->bf[yoo][i];

        for (int j = 0; j < x_embed; j++)
            sum += lstm->wf[yoo][i*(x_embed+d) + j] * x[j];

        for (int j = 0; j < d; j++)
            sum += lstm->wf[yoo][i*(d+x_embed) + x_embed + j] * h[j];

        f[i] = sigmoid(sum);
    }
    kesh->f[em] = copy(f,d); 
    float * iq = xmalloc_vec(d);
    for ( int i = 0 ; i < d ; ++ i){

        float sum = lstm->bi[yoo][i];

        for (int j = 0; j < x_embed; j++)
            sum += lstm->wi[yoo][i*(x_embed+d) + j] * x[j];

        for (int j = 0; j < d; j++)
            sum += lstm->wi[yoo][i*(d+x_embed) + x_embed + j] * h[j];

        iq[i] = sigmoid(sum);
    }
    kesh->i[em] = copy(iq,d);
    float * g = xmalloc_vec(d);
    for ( int i = 0 ; i < d ; ++ i){

        float sum = lstm->bc[yoo][i];

        for (int j = 0; j < x_embed; j++)
            sum += lstm->wc[yoo][i*(x_embed+d) + j] * x[j];

        for (int j = 0; j < d; j++)
            sum += lstm->wc[yoo][i*(d+x_embed) + x_embed + j] * h[j];

        g[i] = tanhf(sum);
    }

    kesh->g[em] = copy(g,d);
    for(int i = 0 ; i < d ; ++ i){
        c_out[i] = f[i]*c_prev[i]+iq[i]*g[i];
    }
    float * o = xmalloc_vec(d);
    for(int i = 0 ; i < d ; ++ i){
        
        float sum = lstm->b0[yoo][i];

        for (int j = 0; j < x_embed; j++)
            sum += lstm->w0[yoo][i*(x_embed+d) + j] * x[j];

        for (int j = 0; j < d; j++)
            sum += lstm->w0[yoo][i*(d+x_embed) + x_embed + j] * h[j];

        o[i] = sigmoid(sum);
    }
    kesh->o[em] = copy(o,d);
    for(int i = 0 ;i < d ;++i ){
        h_out[i] = o[i]*tanhf(c_out[i]);

        float smex = lstm->by[yoo][i];
        for(int j = 0 ; j < d ; j++){
            smex += lstm->wy[yoo][i*d+j] * h_out[j];
        }

        y_out[i] = sigmoid(smex);
    }
        kesh->h[em] = copy(h_out, d);
        kesh->c[em] = copy(c_out, d);
     
    free(f); free(iq); free(g); free(o);
}
typedef struct {
    int x, d, num_layers;
    float **dWf, **dWi, **dWc, **dWo;
    float **dbf, **dbi, **dbc, **dbo;

    float **dWy, **dby;
} grads;
float** malloc_matrix(int rows, int cols) {
    float** m = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        m[i] = (float*)malloc(cols * sizeof(float));
    }
    return m;
}

void grads_init(grads *g, int x_dim, int d_dim, int num_layers) {
    g->x = x_dim;
    g->d = d_dim;
    g->num_layers = num_layers;

    g->dWf = malloc_matrix(num_layers, d_dim); 
    g->dWi = malloc_matrix(num_layers, d_dim);
    g->dWc = malloc_matrix(num_layers, d_dim);
    g->dWo = malloc_matrix(num_layers, d_dim);

    g->dbf = malloc_matrix(num_layers, d_dim);
    g->dbi = malloc_matrix(num_layers, d_dim);
    g->dbc = malloc_matrix(num_layers, d_dim);
    g->dbo = malloc_matrix(num_layers, d_dim);

    g->dWy = malloc_matrix(1, d_dim); 
    g->dby = malloc_matrix(1, d_dim);
}

void grads_zero(grads *g) {
    for (int i = 0; i < g->num_layers; i++) {
        memset(g->dWf[i], 0, g->d * sizeof(float));
        memset(g->dWi[i], 0, g->d * sizeof(float));
        memset(g->dWc[i], 0, g->d * sizeof(float));
        memset(g->dWo[i], 0, g->d * sizeof(float));

        memset(g->dbf[i], 0, g->d * sizeof(float));
        memset(g->dbi[i], 0, g->d * sizeof(float));
        memset(g->dbc[i], 0, g->d * sizeof(float));
        memset(g->dbo[i], 0, g->d * sizeof(float));
    }
    memset(g->dWy[0], 0, g->d * sizeof(float));
    memset(g->dby[0], 0, g->d * sizeof(float));
}

void accum_gate(
    float *dW, float *db,
    const float *W, const float *dz,
    const float *x, const float *hprev,
    int in, int H,
    float *dx_out, float *dh_prev_out
){
    // db += dz
    for(int i=0;i<H;i++) db[i] += dz[i];

    // dW += dz ⊗ [x;hprev]
    for(int i=0;i<H;i++){
        float dzi = dz[i];
        int row = i*(in+H);
        for(int k=0;k<in;k++)    dW[row + k]    += dzi * x[k];
        for(int k=0;k<H;k++)     dW[row + in+k] += dzi * hprev[k];
    }

    // dx += W_x^T dz
    if(dx_out){
        for(int k=0;k<in;k++){
            float s=0;
            for(int i=0;i<H;i++) s += W[i*(in+H) + k] * dz[i];
            dx_out[k] += s;
        }
    }

    // dh_prev += W_h^T dz
    for(int k=0;k<H;k++){
        float s=0;
        for(int i=0;i<H;i++) s += W[i*(in+H) + (in+k)] * dz[i];
        dh_prev_out[k] += s;
    }
}

void lstm_backward_step(
    cell *p, grads *gr,
    cache *k, int t, int layer,
    int in, int H,
    const float *dh, const float *dc_in,
    float *dh_prev_out, float *dc_prev_out, float *dx_out // dx_out можно NULL если не нужен
){
    float *x      = k->x[t];
    float *hprev  = k->h_prev[t];
    float *cprev  = k->c_prev[t];
    float *f      = k->f[t];
    float *ii     = k->i[t];
    float *o      = k->o[t];
    float *g      = k->g[t];
    float *c      = k->c[t];

    // 1) tanh(c)
    float *tanhc = xmalloc_vec(H);
    for(int i=0;i<H;i++) tanhc[i] = tanhf(c[i]);

    // 2) do, dc
    float *do_ = xmalloc_vec(H);
    float *dc  = xmalloc_vec(H);
    for(int i=0;i<H;i++){
        do_[i] = dh[i] * tanhc[i];
        dc[i]  = dc_in[i] + dh[i] * o[i] * (1.0f - tanhc[i]*tanhc[i]);
    }

    // 3) df, di, dg, dc_prev
    float *df = xmalloc_vec(H);
    float *di = xmalloc_vec(H);
    float *dg = xmalloc_vec(H);

    for(int i=0;i<H;i++){
        df[i] = dc[i] * cprev[i];
        di[i] = dc[i] * g[i];
        dg[i] = dc[i] * ii[i];
        dc_prev_out[i] = dc[i] * f[i];
    }

    // 4) dz*
    float *dzf = xmalloc_vec(H);
    float *dzi = xmalloc_vec(H);
    float *dzo = xmalloc_vec(H);
    float *dzg = xmalloc_vec(H);

    for(int i=0;i<H;i++){
        dzf[i] = df[i] * f[i] * (1.0f - f[i]);
        dzi[i] = di[i] * ii[i] * (1.0f - ii[i]);
        dzo[i] = do_[i] * o[i] * (1.0f - o[i]);
        dzg[i] = dg[i] * (1.0f - g[i]*g[i]);
    }


    for(int i=0;i<H;i++) dh_prev_out[i]=0;
    if(dx_out) for(int i=0;i<in;i++) dx_out[i]=0;

    accum_gate(gr->dWf[layer], gr->dbf[layer], p->wf[layer], dzf,
           x, hprev, in, H, dx_out, dh_prev_out);
    accum_gate(gr->dWi[layer], gr->dbi[layer], p->wi[layer], dzi,
           x, hprev, in, H, dx_out, dh_prev_out);
    accum_gate(gr->dWc[layer], gr->dbc[layer], p->wc[layer], dzg,
           x, hprev, in, H, dx_out, dh_prev_out);
    accum_gate(gr->dWo[layer], gr->dbo[layer], p->w0[layer], dzo,
           x, hprev, in, H, dx_out, dh_prev_out);

    free(tanhc); free(do_); free(dc);
    free(df); free(di); free(dg);
    free(dzf); free(dzi); free(dzo); free(dzg);
}
void apply_grads(cell *p, grads *g, float lr) {
    for(int l=0; l<p->num_layers; l++){
        int in = (l==0) ? p->x : p->d;

        for(int i=0;i<p->d*(in+p->d);i++){
            p->wf[l][i] -= lr * g->dWf[l][i];
            p->wi[l][i] -= lr * g->dWi[l][i];
            p->wc[l][i] -= lr * g->dWc[l][i];
            p->w0[l][i] -= lr * g->dWo[l][i];
        }

        for(int i=0;i<p->d;i++){
            p->bf[l][i] -= lr * g->dbf[l][i];
            p->bi[l][i] -= lr * g->dbi[l][i];
            p->bc[l][i] -= lr * g->dbc[l][i];
            p->b0[l][i] -= lr * g->dbo[l][i];
            p->by[l][i] -= lr * g->dby[l][i];
        }

        for(int i=0;i<p->d*p->d;i++){
            p->wy[l][i] -= lr * g->dWy[l][i];
        }
    }
}
// ....... пока что пишу не читать.......
int main(){
    int size_layers = 5;
    int L = size_layers;
    int d = 500;
    


    // идея как рабоатть должно
    float *h_state[L], *c_state[L];
    for(int i = 0 ;  i < L ; ++ i){
        h_state[i] = calloc(d,sizeof(float));
        c_state[i] = calloc(d,sizeof(float));
    }
    double t = clock();
    double T = 100000;
    while (t<= T){
        
        for (l=0; l<L; l++){
            int in_l = (l==0) ? x_embed : d;

            float *h_out = xmalloc_vec(d);
            float *c_out = xmalloc_vec(d);
            float *y_out = xmalloc_vec(d); 

            forvard(input, in_l, h_state[l], c_state[l], d, p, l,
                    y_out, c_out, h_out, k, t*L + l);

            free(h_state[l]); free(c_state[l]); 
            h_state[l] = h_out;
            c_state[l] = c_out;

            input = h_out; 
            free(y_out);   
        }
    }
    grads_zero(gr);

    dh_time_next[l]=zeros(d); dc_time_next[l]=zeros(d);

    for (t=T-1; t>=0; --t){
        float *dh_layer = dh_from_loss_at_t;
        float *dc_layer = zeros(d);

        for (l=L-1; l>=0; --l){
            int in_l = (l==0) ? x_embed : d;
            float *dh = dh_layer + dh_time_next[l];
            float *dc = dc_layer + dc_time_next[l];

            float *dh_prev = zeros(d);
            float *dc_prev = zeros(d);
            float *dx      = xmalloc_vec(in_l);

            lstm_backward_step(p, gr, k, /*t, l*/, in_l, d, dh, dc, dh_prev, dc_prev, dx);

            dh_time_next[l] = dh_prev;
            dc_time_next[l] = dc_prev;

            dh_layer = dx;  
            dc_layer = zeros(d); 
        }
    }

    apply_grads(p, gr, lr);

}
