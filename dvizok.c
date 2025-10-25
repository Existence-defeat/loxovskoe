#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdbool.h>
#include <math.h>
#include <SDL3/SDL.h>
#include <time.h>
#include <unistd.h>
#define vec3 tocka
#define pi 3.1415926535f
#define MAX_SPHERES 100  // Максимальное количество сфер

static float scene_yaw = 0.0f;
static float scene_pitch = 0.0f;
static float scene_roll = 0.0f;

typedef struct {
    float x,y,z;
} vec;
typedef struct {
    float m[4][4];
} matrix;
typedef struct{
    float x,y,z,w;
} vec4;
typedef struct{
    float x,y,z;
} tocka;
typedef struct 
{
    tocka num[3];
} Triangle;
typedef struct{
    float r,g,b;
} RGB;
typedef struct {
    SDL_FRect btn;         
    bool typing;            // рввод
    char buf[64];           // буфер 
    int len;                // длинна
    int value;              
    vec rat;
    char sost;
    bool sphere_visible;

} UI;
typedef struct {
    vec position;
    float radius;
    bool visible;
} Sphere;

float trueW(float x,int W){
    return ((x+1)/2*W);
}
float trueH(float x, int H){
    return ((1 - (x+1) / 2)*H);
}
matrix qet(
    int width,int height,
    int FOV_grad,float zNear,
    float zFar
){
    matrix R = {0};
    const float PI = 3.1415926535f;
    float aspect = (float)width / (float)height;
    float f = 1.0f / tanf((FOV_grad * PI / 180.0f) * 0.5f);

    R.m[0][0] = f / aspect;
    R.m[1][1] = f;
    R.m[2][2] = (zFar + zNear) / (zNear - zFar);
    R.m[2][3] = (2.0f * zFar * zNear) / (zNear - zFar);
    R.m[3][2] = -1.0f;
    //R.m[3][3] = 1.0f;
    return R;
}
vec4 vec4_mul_mat4x4(vec4 i, matrix m) {
    vec4 v;
    v.x = i.x * m.m[0][0] + i.y * m.m[1][0] + i.z * m.m[2][0] + i.w * m.m[3][0];
    v.y = i.x * m.m[0][1] + i.y * m.m[1][1] + i.z * m.m[2][1] + i.w * m.m[3][1];
    v.z = i.x * m.m[0][2] + i.y * m.m[1][2] + i.z * m.m[2][2] + i.w * m.m[3][2];
    v.w = i.x * m.m[0][3] + i.y * m.m[1][3] + i.z * m.m[2][3] + i.w * m.m[3][3];
    if (v.w != 0.0f) {
        v.x /= v.w;
        v.y /= v.w;
        v.z /= v.w;
        v.w = 1.0f;  
    }
    return v;}
tocka mat_mul_v3(matrix M, tocka p){
    float x = p.x, y = p.y, z = p.z;
    float rx = x*M.m[0][0] + y*M.m[1][0] + z*M.m[2][0] + 1.0f*M.m[3][0];
    float ry = x*M.m[0][1] + y*M.m[1][1] + z*M.m[2][1] + 1.0f*M.m[3][1];
    float rz = x*M.m[0][2] + y*M.m[1][2] + z*M.m[2][2] + 1.0f*M.m[3][2];
    float rw = x*M.m[0][3] + y*M.m[1][3] + z*M.m[2][3] + 1.0f*M.m[3][3];
    if (rw != 0.0f){ rx/=rw; ry/=rw; rz/=rw; }
    return (tocka){rx,ry,rz};
}    
void fiksator(tocka *q, matrix p, int W, int H , float *x, float *y) {
    vec4 v = { q->x, q->y, q->z, 1.0f };
    vec4 clip = vec4_mul_mat4x4(v, p);


    if (clip.w != 0.0f) {
        clip.x /= clip.w;
        clip.y /= clip.w;
        clip.z /= clip.w;
    }


    *x = trueW(clip.x, W);
    *y = trueH(clip.y, H);
}

matrix get_rot_x(float angle) {
    matrix mat = {0};

    mat.m[0][0] = 1.0f;
    mat.m[1][1] = cosf(angle);
    mat.m[1][2] = sinf(angle);
    mat.m[2][1] = -sinf(angle);
    mat.m[2][2] = cosf(angle);
    mat.m[3][3] = 1.0f;

    return mat;
}

matrix get_rot_y(float angle) {
    matrix mat = {0};

    mat.m[0][0] = cosf(angle);
    mat.m[0][2] = sinf(angle);
    mat.m[1][1] = 1.0f;
    mat.m[2][0] = -sinf(angle);
    mat.m[2][2] = cosf(angle);
    mat.m[3][3] = 1.0f;

    return mat;
}

matrix get_rot_z(float angle) {
    matrix mat = {0};

    mat.m[0][0] = cosf(angle);
    mat.m[0][1] = sinf(angle);
    mat.m[1][0] = -sinf(angle);
    mat.m[1][1] = cosf(angle);
    mat.m[2][2] = 1.0f;
    mat.m[3][3] = 1.0f;

    return mat;
}

static inline tocka v3(float x,float y,float z){ return (tocka){x,y,z}; }
static inline tocka v3_add(tocka a, tocka b){ return (tocka){a.x+b.x, a.y+b.y, a.z+b.z}; }

void prosto(Triangle * tr,int count){
    for (int t = 0; t < count; t++){
    for (int i = 0; i < 3; i++){
        tr[t].num[i] = v3_add(tr[t].num[i], v3(-0.5f, -0.5f, -0.5f));
    }
}
}
vec3 camera_forward(vec3 pos, vec3 target) {
    vec3 dir = { target.x - pos.x, target.y - pos.y, target.z - pos.z };


    return dir;
}
vec3 getvector(tocka a, tocka b){
    return (vec3){
        b.x-a.x,
        b.y-a.y,
        b.z-a.z,
    };
}
tocka cross(tocka a, tocka b) {
    return (tocka){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}
static inline float vlen(vec3 v){ return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); }
vec3 normalize(vec3 v){ 
    float qw = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    v.x/=qw;
    v.y/=qw;
    v.z/=qw;
    return v;
}
vec3 normal(Triangle *t){
    vec3 e1 = getvector(t->num[0], t->num[1]); 
    vec3 e2 = getvector(t->num[0], t->num[2]); 
    vec3 n  = cross(e1, e2);
    float L = vlen(n);
    if (L > 0.0f) { n.x/=L; n.y/=L; n.z/=L; }  
    return n;
}
bool is_this_triangle(Triangle *r, vec3 cam_pos){
    vec3 c = {
        (r->num[0].x + r->num[1].x + r->num[2].x) / 3.0f,
        (r->num[0].y + r->num[1].y + r->num[2].y) / 3.0f,
        (r->num[0].z + r->num[1].z + r->num[2].z) / 3.0f
    };

    vec3 v1 = (vec3){ cam_pos.x - c.x, cam_pos.y - c.y, cam_pos.z - c.z };
    float L = vlen(v1);
    if (L > 0.0f){ v1.x/=L; v1.y/=L; v1.z/=L; }

    vec3 v2 = normal(r);          
    float d  = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;

    const float EPS = 1e-3f;
    return d > EPS;                     
}
static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

vec3 glav = (vec3){ -0.1f, 0.1f, 0.6f };

void init_light(void){
    float L = vlen(glav);
    if (L > 0.0f){ glav.x/=L; glav.y/=L; glav.z/=L; }
}
static inline vec3 sub3(vec3 a, vec3 b){ return (vec3){a.x-b.x, a.y-b.y, a.z-b.z}; }
static inline float dot3(vec3 a, vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline void screen_to_ndc(float sx, float sy, int W, int H, float *nx, float *ny){
    *nx =  2.0f * sx / (float)W - 1.0f;
    *ny =  1.0f - 2.0f * sy / (float)H; // инвертируем Y экрана
}
void teni_gouraud(SDL_Renderer *ren, Triangle *t, float x[3], float y[3],
                  RGB base, vec3 obj_center_world)
{
    SDL_Vertex v[3] = {0};
    for (int i = 0; i < 3; ++i) {
        vec3 n = normalize((vec3){
            t->num[i].x - obj_center_world.x,
            t->num[i].y - obj_center_world.y,
            t->num[i].z - obj_center_world.z
        });

        float ndotl   = n.x*glav.x + n.y*glav.y + n.z*glav.z;
        float ambient = 0.1f;
        float diffuse = ndotl > 0.0f ? ndotl : 0.0f;
        float I = clampf(ambient + diffuse*0.9f, 0.0f, 1.0f);
        float fr = (base.r/255.0f) * I;
        float fg = (base.g/255.0f) * I;
        float fb = (base.b/255.0f) * I;

        v[i].position.x = x[i];
        v[i].position.y = y[i];
        v[i].tex_coord.x = 0.0f;
        v[i].tex_coord.y = 0.0f;
        v[i].color = (SDL_FColor){fr, fg, fb, 1.0f};
    }
    SDL_RenderGeometry(ren, NULL, v, 3, NULL, 0);
}
static float yaw = 0.0f;    //  Y
static float pitch = 0.0f;  // X
static float roll = 0.0f;   //  Z
static float zoom = -12.0f; 
static bool dragging = false;
void apply_transform(Triangle *src, Triangle *dst, int count,
                     float yaw, float pitch, float roll, float zoomZ)
{
    for (int j = 0; j < count; ++j) {
        for (int i = 0; i < 3; ++i) {
            tocka q = src[j].num[i];
            q = mat_mul_v3(get_rot_y(yaw),   q);
            q = mat_mul_v3(get_rot_x(pitch), q);
            q = mat_mul_v3(get_rot_z(roll),  q);
            q = v3_add(q, v3(0, 0, zoomZ));    
            dst[j].num[i] = q;
        }
    }
}

static inline vec4 mul_mat4_vec4(matrix M, vec4 v){
    vec4 r;
    r.x = M.m[0][0]*v.x + M.m[0][1]*v.y + M.m[0][2]*v.z + M.m[0][3]*v.w;
    r.y = M.m[1][0]*v.x + M.m[1][1]*v.y + M.m[1][2]*v.z + M.m[1][3]*v.w;
    r.z = M.m[2][0]*v.x + M.m[2][1]*v.y + M.m[2][2]*v.z + M.m[2][3]*v.w;
    r.w = M.m[3][0]*v.x + M.m[3][1]*v.y + M.m[3][2]*v.z + M.m[3][3]*v.w;
    return r;
}

static inline vec3 ndc_to_view(matrix proj_inv, float nx, float ny, float nz){
    vec4 c = { nx, ny, nz, 1.0f };
    vec4 v = mul_mat4_vec4(proj_inv, c);
    if (v.w != 0.0f){ v.x/=v.w; v.y/=v.w; v.z/=v.w; }
    return (vec3){ v.x, v.y, v.z };
}

static inline vec3 mouse_to_view_at_z(float mx, float my, int W, int H,
                                      matrix proj_inv, float z_view)
{
    float nx, ny;
    screen_to_ndc(mx, my, W, H, &nx, &ny);

    vec3 v_near = ndc_to_view(proj_inv, nx, ny, -1.0f);
    vec3 v_far  = ndc_to_view(proj_inv, nx, ny,  1.0f);


    float t = (z_view - v_near.z) / (v_far.z - v_near.z);

    vec3 p = {
        v_near.x + t * (v_far.x - v_near.x),
        v_near.y + t * (v_far.y - v_near.y),
        v_near.z + t * (v_far.z - v_near.z)
    };
    return p;
}

void mai(SDL_Renderer *renderer,
         Triangle *tris, matrix proj,
         int W, int H, int count,
         Triangle *tmp, float angle,bool delaem)
{
    float spin = delaem ? 0.0f : angle * 0.7f;
    apply_transform(tris, tmp, count, yaw+spin*1.9, pitch+spin*1.4, roll+spin*0.2, zoom);

    vec3 cam = (vec3){0,0,0};


    for (int i = 0; i < count; ++i) {
        Triangle *T = &tmp[i];

        
        if (!is_this_triangle(T, cam)) continue;

  
        float x[3], y[3];
        for (int j = 0; j < 3; ++j){
            fiksator(&T->num[j], proj, W, H, &x[j], &y[j]);
        }
        RGB base = { 255, 255, 225 };
        vec3 obj_center_world = {0,0,-12};
        teni_gouraud(renderer, T, x, y,base,obj_center_world);


    }
}




float frand_range(float min, float max) {
    return min + (max - min) * ((float)rand() / (float)RAND_MAX);
}
    float static fr = 120.0f/255.0f;
    float static fg = 81.0f/255.0f;
    float static fb = 169.0f/255.0f;
    static float vr = -0.15f, vg = -0.12f, vb = -0.10f;

static inline void bounce(float *x, float *v){
    if (*x < 0.0f){ *x = 0.0f; *v = fabsf(*v); }   
    if (*x > 1.0f){ *x = 1.0f; *v = -fabsf(*v); }  
}
static float t = 0.0f;
static inline float clamp01(float x){ return x < 0 ? 0 : (x > 1 ? 1 : x); }
static inline float clamp02(float x){ return x < 0 ? 0 : (x > 2 ? 2 : x); }
static inline float lerp(float a, float b, float u){ return a + (b - a) * u; }

void cratefon(SDL_Renderer *r, int W, int H, float dt){
    t += dt;
    const float amp = 0.37f;
    SDL_FColor c00 = { clamp01(fr + amp * sinf(t*0.9f + 0.0f)),
                       clamp01(fg + amp * cosf(t*1.1f + 0.4f)),
                       clamp01(fb + amp * sinf(t*1.0f + 1.3f)), 0.4f }; 

    SDL_FColor c10 = { clamp01(fr + amp * sinf(t*0.9f + 0.7f)),
                       clamp01(fg + amp * cosf(t*1.0f + 1.2f)),
                       clamp01(fb + amp * sinf(t*1.2f + 2.1f)), 0.4f }; 

    SDL_FColor c01 = { clamp01(fr + amp * sinf(t*1.1f + 2.0f)),
                       clamp01(fg + amp * cosf(t*0.8f + 2.6f)),
                       clamp01(fb + amp * sinf(t*1.3f + 3.1f)), 0.4f }; 

    SDL_FColor c11 = { clamp01(fr + amp * sinf(t*0.7f + 3.0f)),
                       clamp01(fg + amp * cosf(t*1.2f + 3.7f)),
                       clamp01(fb + amp * sinf(t*0.9f + 4.2f)), 0.4f }; 

    float X[4] = {-1, 1, -1, 1};
    float Y[4] = {-1,-1,  1, 1};
    SDL_Vertex v[4] = {0};

   
    v[0].position = (SDL_FPoint){ trueW(X[0],W), trueH(Y[0],H) }; 
    v[0].color = c00;

    v[1].position = (SDL_FPoint){ trueW(X[1],W), trueH(Y[1],H) }; 
    v[1].color = c01;

    v[2].position = (SDL_FPoint){ trueW(X[2],W), trueH(Y[2],H) }; 
    v[2].color = c10;

    v[3].position = (SDL_FPoint){ trueW(X[3],W), trueH(Y[3],H) }; 
    v[3].color = c11;

    int idx[6] = {0,1,2, 2,1,3};
    SDL_RenderGeometry(r, NULL, v, 4, idx, 6);
}

static float qq = 0.0f;
RGB animated_color(float time) {
    RGB color;
    color.r = 200 + 35 * sinf(time * 1.5f);   
    color.g = 180 + 55 * sinf(time * 2.0f);    
    color.b = 220 + 25 * sinf(time * 2.8f);   
    return color;
}
int qqq(float y,float yaw){
    if(y>=0){
        return(y-yaw);
    }
    else{return (y+yaw);}
}
void draw_sphere(SDL_Renderer *renderer,
                 matrix proj,
                 int W, int H,
                 vec center, float radius,
                 int segTheta, int segPhi,
                 float angle, bool freeze,float dt)
{
    
    float spin = freeze ? 0.0f : angle * 0.7f;
    float yawLoc   = yaw   + spin*1.9f;
    float pitchLoc = pitch + spin*1.4f;
    float rollLoc  = roll  + spin*0.2f;
    float zoomLoc  = zoom;
    qq+=dt;
    float amp = 0.41f;
    vec3 cam = (vec3){0,0,0};

    RGB base = animated_color(qq);

    vec3 obj_center_world = {center.x, center.y, center.z};

    vec3 obj_center_view  = (vec3){ center.x, center.y, center.z + zoomLoc };


    for (int it = 0; it < segTheta; ++it){
        float t0 = (float)it / (float)segTheta;
        float t1 = (float)(it+1) / (float)segTheta;

        float theta0 = t0 * (float)pi;
        float theta1 = t1 * (float)pi;

        for (int ip = 0; ip < segPhi; ++ip){
            float p0 = (float)ip / (float)segPhi;
            float p1 = (float)(ip+1) / (float)segPhi;

            float phi0 = p0 * (2.0f*(float)pi);
            float phi1 = p1 * (2.0f*(float)pi);


            tocka v00 = (tocka){
                radius * sinf(theta0)*cosf(phi0),
                  radius * cosf(theta0),
                  radius * sinf(theta0)*sinf(phi0)
            };
            tocka v01 = (tocka){
                  radius * sinf(theta0)*cosf(phi1),
                  radius * cosf(theta0),
                 radius * sinf(theta0)*sinf(phi1)
            };
            tocka v10 = (tocka){
                 radius * sinf(theta1)*cosf(phi0),
                 radius * cosf(theta1),
                 radius * sinf(theta1)*sinf(phi0)
            };
            tocka v11 = (tocka){
                radius * sinf(theta1)*cosf(phi1),
                 radius * cosf(theta1),
                 radius * sinf(theta1)*sinf(phi1)
            };

            Triangle T1 = { { v00, v10, v11 } };
            Triangle T2 = { { v00, v11, v01 } };

            for(int i=0;i<3;++i){
                T1.num[i] = mat_mul_v3(get_rot_y(yawLoc),   T1.num[i]);
    T1.num[i] = mat_mul_v3(get_rot_x(pitchLoc), T1.num[i]);
    T1.num[i] = mat_mul_v3(get_rot_z(rollLoc),  T1.num[i]);

    T1.num[i] = v3_add(T1.num[i], (tocka){ center.x, center.y, center.z });

    T1.num[i] = mat_mul_v3(get_rot_y(scene_yaw),   T1.num[i]);
    T1.num[i] = mat_mul_v3(get_rot_x(scene_pitch), T1.num[i]);
    T1.num[i] = mat_mul_v3(get_rot_z(scene_roll),  T1.num[i]);

    T1.num[i] = v3_add(T1.num[i], (tocka){ 0, 0, zoomLoc });
        }
        for(int i=0;i<3;++i){
             T2.num[i] = mat_mul_v3(get_rot_y(yawLoc),   T2.num[i]);
    T2.num[i] = mat_mul_v3(get_rot_x(pitchLoc), T2.num[i]);
    T2.num[i] = mat_mul_v3(get_rot_z(rollLoc),  T2.num[i]);

    T2.num[i] = v3_add(T2.num[i], (tocka){ center.x, center.y, center.z });

    T2.num[i] = mat_mul_v3(get_rot_y(scene_yaw),   T2.num[i]);
    T2.num[i] = mat_mul_v3(get_rot_x(scene_pitch), T2.num[i]);
    T2.num[i] = mat_mul_v3(get_rot_z(scene_roll),  T2.num[i]);

    T2.num[i] = v3_add(T2.num[i], (tocka){ 0, 0, zoomLoc });
        }


        vec3 cam = (vec3){0,0,0};

        if (is_this_triangle(&T1, cam)){
            float sx[3], sy[3];
            for (int j=0;j<3;++j)
                fiksator(&T1.num[j], proj, W, H, &sx[j], &sy[j]);

            teni_gouraud(renderer, &T1, sx, sy, base, v3_add(obj_center_view,(tocka){0,0,-0.25}));
        }
        if (is_this_triangle(&T2, cam)){
            float sx[3], sy[3];
            for (int j=0;j<3;++j)
                fiksator(&T2.num[j], proj, W, H, &sx[j], &sy[j]);
            teni_gouraud(renderer, &T2, sx, sy, base, v3_add(obj_center_view,(tocka){0,0,-0.25}));
        }
    }
}}
matrix invert4x4(matrix a) {
    matrix inv;
    float invOut[16];
    float m[16];
    for (int i=0;i<4;i++)
        for (int j=0;j<4;j++)
            m[i*4+j] = a.m[i][j];

    invOut[0] = m[5]  * m[10] * m[15] -
             m[5]  * m[11] * m[14] -
             m[9]  * m[6]  * m[15] +
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] -
             m[13] * m[7]  * m[10];

    invOut[4] = -m[4]  * m[10] * m[15] +
              m[4]  * m[11] * m[14] +
              m[8]  * m[6]  * m[15] -
              m[8]  * m[7]  * m[14] -
              m[12] * m[6]  * m[11] +
              m[12] * m[7]  * m[10];

    invOut[8] = m[4]  * m[9] * m[15] -
             m[4]  * m[11] * m[13] -
             m[8]  * m[5] * m[15] +
             m[8]  * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    invOut[12] = -m[4]  * m[9] * m[14] +
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] -
               m[8]  * m[6] * m[13] -
               m[12] * m[5] * m[10] +
               m[12] * m[6] * m[9];

    invOut[1] = -m[1]  * m[10] * m[15] +
              m[1]  * m[11] * m[14] +
              m[9]  * m[2] * m[15] -
              m[9]  * m[3] * m[14] -
              m[13] * m[2] * m[11] +
              m[13] * m[3] * m[10];

    invOut[5] = m[0]  * m[10] * m[15] -
             m[0]  * m[11] * m[14] -
             m[8]  * m[2] * m[15] +
             m[8]  * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    invOut[9] = -m[0]  * m[9] * m[15] +
              m[0]  * m[11] * m[13] +
              m[8]  * m[1] * m[15] -
              m[8]  * m[3] * m[13] -
              m[12] * m[1] * m[11] +
              m[12] * m[3] * m[9];

    invOut[13] = m[0]  * m[9] * m[14] -
              m[0]  * m[10] * m[13] -
              m[8]  * m[1] * m[14] +
              m[8]  * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    invOut[2] = m[1]  * m[6] * m[15] -
             m[1]  * m[7] * m[14] -
             m[5]  * m[2] * m[15] +
             m[5]  * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    invOut[6] = -m[0]  * m[6] * m[15] +
              m[0]  * m[7] * m[14] +
              m[4]  * m[2] * m[15] -
              m[4]  * m[3] * m[14] -
              m[12] * m[2] * m[7] +
              m[12] * m[3] * m[6];

    invOut[10] = m[0]  * m[5] * m[15] -
              m[0]  * m[7] * m[13] -
              m[4]  * m[1] * m[15] +
              m[4]  * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    invOut[14] = -m[0]  * m[5] * m[14] +
               m[0]  * m[6] * m[13] +
               m[4]  * m[1] * m[14] -
               m[4]  * m[2] * m[13] -
               m[12] * m[1] * m[6] +
               m[12] * m[2] * m[5];

    invOut[3] = -m[1] * m[6] * m[11] +
             m[1] * m[7] * m[10] +
             m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] +
             m[9] * m[3] * m[6];

    invOut[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    invOut[11] = -m[0] * m[5] * m[11] +
               m[0] * m[7] * m[9] +
               m[4] * m[1] * m[11] -
               m[4] * m[3] * m[9] -
               m[8] * m[1] * m[7] +
               m[8] * m[3] * m[5];

    invOut[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    float det = m[0] * invOut[0] + m[1] * invOut[4] + m[2] * invOut[8] + m[3] * invOut[12];
    if (det == 0.0f) return a; 

    det = 1.0f / det;
    for (int i = 0; i < 16; i++)
        invOut[i] *= det;

    for (int i=0;i<4;i++)
        for (int j=0;j<4;j++)
            inv.m[i][j] = invOut[i*4+j];

    return inv;
}
static UI ui = {0};
static vec debug_sphere_pos = {0, 0, -10};
static void draw_button(SDL_Renderer *r) {
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(r, 40, 40, 50, 180);
    SDL_RenderFillRect(r, &ui.btn);
}
static bool point_in_rect(float x, float y, const SDL_FRect *r){
    return (x >= r->x && x <= r->x + r->w && y >= r->y && y <= r->y + r->h);
}

static inline vec screen_to_world(float mx, float my, int W, int H) {
    float x = (mx / (float)W - 0.5f) * 15.0f; 
    float y = (0.5f - my / (float)H) * 10.0f;  
    float z = -13.0f;  
    
    return (vec){x, y, z};
}
float forX(float x,int W){
    return ((x/(float)W)*4.77-2.385);
}
float forY(float y,int W){
    return 3.63f - (y / (float)W) * 7.26f;
}

int main() {
    static Sphere spheres[MAX_SPHERES];
static int sphere_count = 0;
    srand((unsigned)time(NULL));
    int W = 2600;
    int H = 1700;
    float time = 0.0f ;
    SDL_Init(SDL_INIT_VIDEO);
    static bool right_click_processed = false;
    static bool eshe = false;
    
    SDL_Window *window = SDL_CreateWindow("14гитлер88",  
W,H,
    SDL_WINDOW_RESIZABLE 
);

    SDL_Renderer *renderer = SDL_CreateRenderer(window, "opengl");


    matrix x = qet(W,H,90,0.1,1000);
    matrix proj_inv = invert4x4(x);
    

    bool running = true;
    SDL_Event e;
    init_light();
    float timeq = 0.0f ;
    Uint64 prev_ms = SDL_GetTicks();
 
                ui.btn.w = 160.0f;
                ui.btn.h = 36.0f;
                ui.btn.x = (W - ui.btn.w) * 0.5f;
                ui.btn.y = H - ui.btn.h - 12.0f;  
                ui.typing = false;
                ui.len = 0;
                ui.buf[0] = '\0';
                ui.sost = 0;
                ui.rat = (vec){0,0,0};
                ui.sphere_visible=false;
                bool right_down = false;
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    while (running) {   

        Uint64 q_ms = SDL_GetTicks();
        float sd = (q_ms - prev_ms)/1000.0f;
        float sd1 = (q_ms - prev_ms)/10000.0f;
        prev_ms = q_ms;
        time+=sd;
        timeq+=sd1;
        if(timeq>=0.2f){timeq = 0.0f;}
        float tqqq = timeq;
        if (tqqq>0.1f){tqqq = 0.0f;}
       
        while (SDL_PollEvent(&e)) {
    switch (e.type) {
    case SDL_EVENT_QUIT:
        running = false;
        break;

    case SDL_EVENT_WINDOW_RESIZED:
        W = e.window.data1;
        H = e.window.data2;
        static float current_fov = 90.0f;
             x = qet(W,H,90,0.1,1000);
     proj_inv = invert4x4(x);
    
        ui.btn.x = (W - ui.btn.w) * 0.5f;
        ui.btn.y = H - ui.btn.h - 12.0f;
        break;

    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        if (e.button.button == SDL_BUTTON_LEFT) {
            float mx = (float)e.button.x;
            float my = (float)e.button.y;

            if (point_in_rect(mx, my, &ui.btn)) {

                ui.typing = !ui.typing;
                if (ui.typing) SDL_StartTextInput(window);
                else           SDL_StopTextInput(window);
            }
             else {
                dragging = true;
                SDL_SetWindowRelativeMouseMode(window, true);
            }
        }
    if (e.button.button == SDL_BUTTON_RIGHT) {
        float mx = (float)e.button.x;
        float my = (float)e.button.y;
        
        if (!point_in_rect(mx, my, &ui.btn) && sphere_count < MAX_SPHERES) {
            spheres[sphere_count].position = screen_to_world(mx, my, W, H);
            spheres[sphere_count].radius = 0.3f;
            spheres[sphere_count].visible = true;
            sphere_count++;
            
            printf("Added sphere %d at: %.2f %.2f %.2f\n", 
                   sphere_count, spheres[sphere_count-1].position.x, 
                   spheres[sphere_count-1].position.y, 
                   spheres[sphere_count-1].position.z);
        }
    }
    break;
        

    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (e.button.button == SDL_BUTTON_LEFT) {
            dragging = false;
            SDL_SetWindowRelativeMouseMode(window, false);
        }

    else if (e.button.button == SDL_BUTTON_RIGHT) {
        right_click_processed = false;
        eshe = true;
    }
    break;
    

    case SDL_EVENT_MOUSE_MOTION:
    if (dragging && !ui.typing) {
        const float sens = 0.001f;
        
 
            scene_yaw   += e.motion.xrel * sens;
            scene_pitch += e.motion.yrel * sens;
           
    }
    break;
    case SDL_EVENT_MOUSE_WHEEL:
        if (!ui.typing) {
            zoom += e.wheel.y * 0.7f;
            scene_roll += e.wheel.y * 0.1f;
            //if (zoom > -2.0f)  zoom = -2.0f;
            //if (zoom < -50.0f) zoom = -50.0f;
        }
        break;

    case SDL_EVENT_TEXT_INPUT:
        if (ui.typing) {
            const char *txt = e.text.text; 
            for (const char *p = txt; *p; ++p) {
                char c = *p;
                if ((c >= '0' && c <= '9') || (c=='-' && ui.len==0)) {
                    if (ui.len < (int)sizeof(ui.buf)-1) {
                        ui.buf[ui.len++] = c;
                        ui.buf[ui.len]   = '\0';
                    }
                }
            }
        }
        break;

    case SDL_EVENT_KEY_DOWN:
        if (!ui.typing) {
        //if (e.key.key == SDLK_1) debug_sphere_pos.z -= 1.0f;
        //if (e.key.key == SDLK_2) debug_sphere_pos.z += 1.0f;
        //if (e.key.key == SDLK_3) debug_sphere_pos.x -= 0.5f;
        //if (e.key.key == SDLK_4) debug_sphere_pos.x += 0.5f;
        //if (e.key.key == SDLK_5) debug_sphere_pos.y -= 0.5f;
        //if (e.key.key == SDLK_6) debug_sphere_pos.y += 0.5f;
        //printf("Debug sphere: %.2f, %.2f, %.2f\n", 
               //debug_sphere_pos.x, debug_sphere_pos.y, debug_sphere_pos.z);
    }
    break;

        if (e.key.key == SDLK_BACKSPACE) {
            if (ui.len > 0) ui.buf[--ui.len] = '\0';
        } else if (e.key.key == SDLK_ESCAPE) {
            ui.typing = false;
            SDL_StopTextInput(window);
        } else if (e.key.key == SDLK_RETURN || e.key.key == SDLK_KP_ENTER) {
            ui.value = (int)strtol(ui.buf, NULL, 10);
            ui.len = 0;
            ui.buf[0] = '\0';
            ui.typing = false;
            SDL_StopTextInput(window);
            printf("accepted value = %d\n", ui.value);
        }
        break;
    
    default:
        break;
    }
}  
        SDL_RenderClear(renderer);  
            cratefon(renderer, W, H,timeq);
        draw_button(renderer);
        //draw_sphere(renderer, x, W, H, debug_sphere_pos, 1.0f, 20, 20, time, dragging, tqqq);
        draw_sphere(renderer, x, W, H, 
                   (vec){0,0,-10}, 
                   0.8f, 
                   19, 19, 
                   time, dragging, tqqq);
        for (int i = 0; i < sphere_count; i++) {
    if (spheres[i].visible) {
        draw_sphere(renderer, x, W, H, 
                   spheres[i].position, 
                   spheres[i].radius, 
                   19, 19, 
                   time, dragging, tqqq);
    }
}
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
