#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdbool.h>
#define mb 1024 * 1024
typedef struct qet {
    size_t size;
    bool free;
    struct qet * next;
    
} qet;
static uint8_t heap[mb];
static qet* indicator = NULL;
void create_heap(){
    indicator = (qet*)heap;
    indicator->size = mb - sizeof(qet);
    indicator->free = 1;
    indicator->next = NULL;
}
qet * search(int size){
    qet * curr = indicator;
    while(curr){
        if(curr){
            if(curr->free && curr->size >=size){
                return curr;
            }
            else {
                curr = curr->next;
            }

        }  

        else return NULL;
        
    }
    return NULL;
}
void share(qet * qwe, int size){
    if(qwe->size >= size + sizeof(qwe)+8){
        qet * curr = (uint8_t*)qwe+sizeof(qet)+size;
        size_t orig = qwe->size;
        qwe->size = size;
        qwe->free = 0;
        qwe->next = curr;
        curr->free = 1;
        curr->size = orig-size-sizeof(qet);
        curr->next = NULL;
    }
}
void* mallocq(int size){
    if(size==0){return;}
    if(!indicator){create_heap();}
    qet * curr = search(size);
    if(curr){
        share(curr,size);
        curr->free =0;
        return (void*)((uint8_t*)curr+sizeof(qet));
    }
    else{
        return NULL;
    }
}
// 2 рядом стоящие
void merge(qet * qwe){
    if(!qwe){return;}
    if(qwe->free && qwe->next->free){
        qwe->size+=qwe->next->size+sizeof(qet);
        qwe->next = qwe->next->next;
    }
    else{
        return;
    }
}
void freeq(void * qwe){
    if(!qwe){return;}
    else{
        qet * curr = (uint8_t*)qwe-sizeof(qet);
        curr->free=1;
        merge(curr);
    }
}