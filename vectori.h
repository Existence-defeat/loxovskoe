#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#include <stdlib.h>

// ==================== ВЕКТОРНЫЕ ОПЕРАЦИИ ====================

/**
 * @brief Скалярное произведение двух векторов
 * @param A Первый вектор
 * @param B Второй вектор
 * @param n Размер векторов
 * @return Скалярное произведение A·B
 */
float vector_dot(float* A, float* B, int n);

/**
 * @brief Умножение вектора на скаляр
 * @param vec Исходный вектор
 * @param n Размер вектора
 * @param scalar Скаляр
 * @return Новый вектор: vec * scalar
 */
float* vector_scale(float* vec, int n, float scalar);

/**
 * @brief Поэлементное умножение векторов (Hadamard product)
 * @param A Первый вектор
 * @param B Второй вектор
 * @param n Размер векторов
 * @return Новый вектор: A[i] * B[i] для всех i
 */
float* vector_multiply_elementwise(float* A, float* B, int n);

/**
 * @brief Сложение двух векторов
 * @param A Первый вектор
 * @param B Второй вектор
 * @param n Размер векторов
 * @return Новый вектор: A[i] + B[i] для всех i
 */

// ==================== МАТРИЧНЫЕ ОПЕРАЦИИ ====================

/**
 * @brief Умножение матрицы на вектор
 * @param A Матрица размера m×n (хранится построчно)
 * @param x Вектор размера n
 * @param m Число строк матрицы A
 * @param n Число столбцов матрицы A
 * @return Вектор размера m: b = A·x
 */
float* matrix_vector_multiply(float* A, float* x, int m, int n);

/**
 * @brief Умножение двух матриц
 * @param A Матрица размера rows_A×cols_A
 * @param B Матрица размера cols_A×cols_B
 * @param rows_A Число строк матрицы A
 * @param cols_A Число столбцов матрицы A
 * @param cols_B Число столбцов матрицы B
 * @return Матрица C размера rows_A×cols_B: C = A·B
 */
float* matrix_multiply_1d(float* A, float* B, 
                         int rows_A, int cols_A, int cols_B);

/**
 * @brief Транспонирование матрицы
 * @param matrix Исходная матрица размера m×n
 * @param m Число строк исходной матрицы
 * @param n Число столбцов исходной матрицы
 * @return Транспонированная матрица размера n×m
 */
float* transpose_1d_simple(float* matrix, int m, int n);

// ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

/**
 * @brief Выделение памяти для матрицы
 * @param rows Число строк
 * @param cols Число столбцов
 * @return Указатель на матрицу
 */
float* matrix_alloc(int rows, int cols);

/**
 * @brief Освобождение памяти матрицы
 * @param matrix Указатель на матрицу
 */
void matrix_free(float* matrix);

/**
 * @brief Копирование матрицы
 * @param src Исходная матрица
 * @param dest Матрица назначения
 * @param rows Число строк
 * @param cols Число столбцов
 */
void matrix_copy(float* src, float* dest, int rows, int cols);

#endif // VECTOR_OPS_H








#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
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

static float* xmalloc_vec(int n) {
    float* p = (float*)malloc((size_t)n * sizeof(float));
    if (!p) { perror("malloc"); exit(1); }
    return p;
}
cell * create_cell(int d,int x,int n){
    cell * rat;
    rat->d = malloc(sizeof(int)*d);
    rat->d = d;
    rat->d = malloc(sizeof(int)*x);
    rat->x = x;
    rat->num_layers = malloc(sizeof(int)*n);
    rat->num_layers = n;
    float pofig = 0.07;
     rat->wy = (float *)malloc(x*(x+d)*sizeof(float)); rat->w0 = (float *)malloc((x+d)*d*sizeof(float)); rat->wc = (float *)malloc((x+d)*d*sizeof(float)); rat->wi = (float *)malloc((x+d)*d*sizeof(float)); rat->wf = (float *)malloc((x+d)*d*sizeof(float));
    rat->by = (float *)malloc(d*sizeof(float)); rat->b0 = (float *)malloc(d*sizeof(float)); rat->bc = (float *)malloc(d*sizeof(float)); rat->bi = (float *)malloc(d*sizeof(float)); rat->bf = (float *)malloc(d*sizeof(float));
    for(int j = 0 ; j < n ; ++j){
        rat->wy[j] = (float *)malloc(x*(x+d)*sizeof(float)); rat->w0[j] = (float *)malloc((x+d)*d*sizeof(float)); rat->wc[j] = (float *)malloc((x+d)*d*sizeof(float)); rat->wi[j] = (float *)malloc((x+d)*d*sizeof(float)); rat->wf[j] = (float *)malloc((x+d)*d*sizeof(float));
        rat->by[j] = (float *)malloc(d*sizeof(float)); rat->b0[j] = (float *)malloc(d*sizeof(float)); rat->bc[j] = (float *)malloc(d*sizeof(float)); rat->bi[j] = (float *)malloc(d*sizeof(float)); rat->bf[j] = (float *)malloc(d*sizeof(float));
        for(int i = 0; i < x*(x+d);++i){
            rat->wy[j][i] = rand_scale(pofig);
        }
        for(int i = 0; i < d*(x+d);++i){
            rat->w0[j][i] = rand_scale(pofig);
            rat->wc[j][i] = rand_scale(pofig);
            rat->wf[j][i] = rand_scale(pofig);
            rat->wi[j][i] = rand_scale(pofig);
        }
        for(int i = 0 ; i < d ; ++i){
            rat->b0[j][i] = rand_scale(pofig);
            rat->bc[j][i] = rand_scale(pofig);
            rat->bf[j][i] = rand_scale(pofig);
            rat->bi[j][i] = rand_scale(pofig);
            rat->by[j][i] = rand_scale(pofig);
        }
    }
    return rat;
}
