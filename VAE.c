#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <windows.h>
#define M_PI 3.14159265358979323846f
typedef struct {
    int d; // tipa size emeding
    int num_layers;    // only hidden + output layers 
    int* layerSize;    // size of each layer output
    float** W;         // flat weights per layer
    float** b;
} FC_layer;
typedef struct {
    float** layers; // Выходы каждого слоя для вычисления градиентов
    float ** pre;
    int L;
} ForwardPass;
 float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / (float)M_PI) *
           (x + 0.044715f * x * x * x)));
}
float d_gelu(float x) {
    float c = sqrtf(2.0f / (float)M_PI);
    float x3 = x * x * x;
    float inner = c * (x + 0.044715f * x3);
    float th = tanhf(inner);
    float dg = 0.5f * (1.0f + th) + (0.5f * x * (1.0f - th * th) * c * (1.0f + 3.0f * 0.044715f * x * x));
    return dg;
}

 float rand_scale(float s) {
    return ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * s;
}
 float* xmalloc_vec(int n) {
    float* p = (float*)malloc((size_t)n * sizeof(float));
    if (!p) { perror("malloc"); exit(1); }
    return p;
}

 FC_layer* ffn_create(int d, int num_hidden, ...) {
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
    for (int l=0; l<=num_hidden; l++) {
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


    va_end(args);
    return fc;
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
 ForwardPass ffn_forward(const float* input, FC_layer* fc) {
    ForwardPass fp;
    fp.L = fc->num_layers;
    fp.layers = (float**)malloc((size_t)(fp.L + 1) * sizeof(float*)); // +1 для входного слоя
    fp.pre = (float**)malloc((size_t)(fp.L ) * sizeof(float*));
    // Сохраняем вход как нулевой слой
    fp.layers[0] = xmalloc_vec(fc->d);
    memcpy(fp.layers[0], input, (size_t)fc->d * sizeof(float));

    for (int l = 0; l < fp.L; l++) {
        int in = (l == 0) ? fc->d : fc->layerSize[l-1];
        int out = fc->layerSize[l];
        float* next = xmalloc_vec(out);
        float* nextpre = xmalloc_vec(out);
        for (int j = 0; j < out; j++) {
            float s = fc->b[l][j];
            for (int k = 0; k < in; k++) {
                s += fc->W[l][j * in + k] * fp.layers[l][k];
            }
            next[j] = (l < fp.L - 1) ? gelu(s) : s;
            nextpre[j] = s;
        }
        fp.layers[l+1] = next;
        fp.pre[l] = nextpre;
    }
    return fp;
}
void ffn_backward(
    FC_layer* fc,
    const float* grad, int grad_size,
    float lr,
    ForwardPass fp,
    bool want_input_grad,
    float* input_grad_out  // размер fc->d
){
    float* delta = xmalloc_vec(grad_size);
    memcpy(delta, grad, (size_t)grad_size * sizeof(float));

    int L = fp.L;

    for (int l = L - 1; l >= 0; l--) {
        int in_size  = (l == 0) ? fc->d : fc->layerSize[l-1];
        int out_size = fc->layerSize[l];

        float* next_delta = (float*)calloc((size_t)in_size, sizeof(float));
        if (!next_delta) { perror("calloc"); exit(1); }

        for (int j = 0; j < out_size; j++) {
            float d = delta[j];

            for (int k = 0; k < in_size; k++) {
                next_delta[k] += d * fc->W[l][j * in_size + k];

                float wgrad = d * fp.layers[l][k];
                fc->W[l][j * in_size + k] -= lr * wgrad;
            }
            fc->b[l][j] -= lr * d;
        }

        if (l > 0) {
            for (int k = 0; k < in_size; k++) {
                next_delta[k] *= d_gelu(fp.pre[l-1][k]);
            }
        }

        free(delta);
        delta = next_delta;

        if (l == 0 && want_input_grad && input_grad_out) {
            memcpy(input_grad_out, delta, (size_t)fc->d * sizeof(float));
        }
    }

    for (int i = 0; i <= L; i++) free(fp.layers[i]);
    for (int i = 0; i < L; i++) free(fp.pre[i]);
    free(fp.pre);
    free(fp.layers);
    free(delta);
}

float oshibka(float predicted, float target) {
    return (predicted - target)*(predicted - target);
}
float rand01(void) {
    float r = (float)rand() / (float)RAND_MAX;
    if (r < 1e-10f) r = 1e-10f; 
    return r;
}
float * epsil(int logva){
    float * res = xmalloc_vec(logva);
    for(int i = 0 ; i < logva ; ++i ){
    float u1 = rand01();
    float u2 = rand01();

    res[i]= (float)sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    }
    return res;
}
float * dy(float * dz,float * u,int beta,int logva){
    float * du = xmalloc_vec(logva);
    for(int i = 0 ; i<logva;++i){
        du[i] = dz[i]+ beta * u[i];
    }
    return du;
}
float * dlogvar(float * dz, float * logvar,float *eps, int beta,int logva){
    float * dlog = xmalloc_vec(logva);
    for(int i = 0 ; i < logva; ++i){
        dlog[i] = dz[i]*(0.5*eps[i]*expf(0.5*logvar[i]))+0.5*beta*(expf(logvar[i])-1);
    }
    return dlog;
}
float * sum_vectors(float * v1,float * v2, int logva){
    float * res = xmalloc_vec(logva);
    for (int i = 0 ; i < logva ; ++i){
        res[i] = v1[i]+v2[i];
    }
    return res;
}
void xz(float * x/*vxod*/,int d,float beta,FC_layer * fc1,FC_layer * fc2, float lr , int logva){

    float * copy = xmalloc_vec(d);
    memcpy(copy,x,d*sizeof(float));

    ForwardPass fq1 = ffn_forward(copy,fc1);
    float * z = fq1.layers[fq1.L];
    float * mu = xmalloc_vec(logva);
    memcpy(mu,z,logva*sizeof(float));
    float * ygar = z+logva;
    float * logvar = xmalloc_vec(logva);
    memcpy(logvar,ygar,logva*sizeof(float));
    
    float * sigma = xmalloc_vec(logva);
    float * sigma2 = xmalloc_vec(logva);
    for (int i = 0 ; i < logva; ++i){
        if (logvar[i] > 10) logvar[i] = 10;
        if (logvar[i] < -10) logvar[i] = -10;
        sigma[i] = expf(0.5f * logvar[i]);
        sigma2[i] = expf(logvar[i]);
    }

    
    float * oposym = epsil(logva);
    float * z_sample = xmalloc_vec(logva);
    for (int i = 0 ; i < logva;++i){
        z_sample[i] = mu[i]+sigma[i]*oposym[i];
    }


    ForwardPass fq2 = ffn_forward(z_sample,fc2);
    float * x_shtrix = fq2.layers[fq2.L];
    float  loss = 0.0f;
    for (int i = 0; i < d; i++) {
        loss += oshibka(x_shtrix[i], x[i]);
        if(i<logva){
            loss += 0.5 *beta*d* (sigma2[i] + mu[i]*mu[i] - 1 - logvar[i]);
        }
    }
    loss /= d;

    float * delta = xmalloc_vec(d);
    for(int j = 0 ; j < d ; ++j){
        delta[j] = 2*(x_shtrix[j]-x[j])/d;
    }
    float * dz = xmalloc_vec(logva);
    ffn_backward(fc2,delta,d,lr,fq2,true,dz);

    float * du = dy(dz,mu,beta,logva);
    float * dlog = dlogvar(dz,logvar,oposym,beta,logva);

    float* denc = xmalloc_vec(2 * logva);

    for(int i = 0 ; i < logva ; ++i){
        denc[i] = du[i];
        denc[logva+i] = dlog[i];
    }

    ffn_backward(fc1,denc,2*logva , lr,fq1,false,NULL);




    free(copy); ffn_free(fc1); ffn_free(fc2); free(logvar); free(mu);free(sigma);free(sigma2);free(z_sample);
    free(delta); free(dz); free(du); free(dlog); free(denc); free(oposym);
}
// ....... пока что пишу не читать.......
int main(){
    float lr = 0.06f;
    int logva = 3;
    HANDLE hSerial = CreateFile("\\\\.\\COM3",     
                                GENERIC_READ | GENERIC_WRITE, 
                                0,                 
                                NULL,             
                                OPEN_EXISTING,     
                                FILE_ATTRIBUTE_NORMAL, 
                                NULL);

    if (hSerial == INVALID_HANDLE_VALUE) {
        printf("Error opening port\n");
        return 1;
    }

    DCB dcb = {0};
    dcb.DCBlength = sizeof(dcb); 

    GetCommState(hSerial, &dcb);      
    dcb.BaudRate = CBR_9600;         
    dcb.ByteSize = 8;                 // Размер одного пакета данных 
    dcb.StopBits = ONESTOPBIT;       
    dcb.Parity = NOPARITY;            
    SetCommState(hSerial, &dcb);      

    char incomingByte;     
    DWORD bytesRead;       

    while(1) {           
        if (ReadFile(hSerial, &incomingByte, 1, &bytesRead, NULL)) {
            
            if (bytesRead > 0) {
                printf("Received: %c\n", incomingByte); 
            }
        }
        Sleep(10); 
    }

    CloseHandle(hSerial); 




    FC_layer * fc1 = ffn_create(d,3,216,32,2*logva);
    FC_layer * fc2 = ffn_create(logva,3,32,216,d);

}