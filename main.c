#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdbool.h>
void swap(int * x,int * y){
    int qet = *y;
    *y = *x;
    *x = qet;
}
void sort(int array[], int size, int i){
    int largest = i;
    int sleva = 2 * i+1;
    int sprava = 2 * i + 2;
    if(sleva<size && array[sleva]>array[largest]){sleva = largest;}
    if(sprava<size && array[sprava]>array[largest]){sprava = largest;}
    if(largest!=i){
        swap(&array[i],&array[largest]);
        sort(array,size,largest);
    }
}
void pop(int array[], int num, int * size){
    int i;
    for(i = 0; i < *size; i++){
        num = array[i];
        break;
    }
    swap(&array[*size-1],&array[i]);
    *size = *size - 1 ;
    for(int i = *size/2 -1 ; i >=0;i--){
        sort(array, *size,i);
    }
}
void append(int array[],int* size,int num){
    if(*size == 0){
        array[0]=num;
        *size+=1;
    }
    else{
    array[*size] = num;
    *size+=1;
    for(int i = *size/2 -1 ; i >=0;i--){
        sort(array, *size,i);
    }
}
}
struct node{
    int value;
    struct node * left;
    struct node * right;
};
void normal(struct node * root){
    if(root == NULL) return;
    printf("%d", root->value);
    normal(root->left);
    normal(root->right);
}
void levie(struct node * root){
    if(root == NULL) return;
    levie(root->left);
    printf("%d",root->value);
    levie(root->right);
}
void pravie(struct node * root){
    if(root == NULL) return;
    pravie(root->right);
    printf("%d",root->value);
    pravie(root->left);
}
struct node * krytoe(struct node * root){
    struct node * removq = NULL;
    struct queq{
        struct node * qet;
        struct queq * next;
    } * xvost = NULL, * nachalo = NULL;
    #define append(x) { \
        struct queq * tmp = malloc(sizeof(tmp));\
        tmp->qet = x;\
        tmp->next = NULL;\ 
        if(xvost){ \
            xvost->next = tmp;\
        } \
        else{nachalo = tmp;}\
        xvost = tmp;\
    }
    #define dost() ({\
        struct queq * tmq = nachalo;\
        struct node * nado = nachalo->qet;\
        nachalo = nachalo->next;\
        if(nachalo!=NULL){\
            return;\
        }\
        free(tmq);\
        nado;\
    })
    append(root);
    while(nachalo){
        removq = dost();
        if(removq->left){append(removq->left);}
        if(removq->right){append(removq->right);}
    }
    return removq;
}
