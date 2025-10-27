#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdbool.h>
#define T 12 // min 
typedef struct  el{
    int keys[2* T + 1];//mim - T - 1,max 2*T-1
    struct el * children[2*T];
    int num_keys;
    bool is_leaf;
} root;
root * create(bool is_leaf){
    root * qet = malloc(sizeof(root));
    qet->is_leaf = is_leaf;
    qet->num_keys = 0;
    for(int i = 0; i < 2 * T; i++){
        qet->children[i]=NULL;
    }
    return qet;
}
void split_child(root* parent, int i) {
    root * glav = parent->children[i];
    root * right = create(glav->is_leaf);
    for(int i = 0; i < T-1; i++ ){
        right->keys[i] = glav->keys[i+T];
    }
    for(int i = 0; i < T; i++){
        right->children[i] = glav->children[i+T];
    } 
    for(int  j  = parent->num_keys-1;j>=i;j-- ){
        parent->keys[j+1]= parent->keys[j];
    }
    for(int  j  = parent->num_keys;j>=i+1;j-- ){
        parent->children[j+1]= parent->children[j];
    }
    parent->keys[i] = glav->keys[T-1];
    glav->num_keys = T-1;
    parent->num_keys++;
}
void polyINSERT(root * node, int key){
    int i = node->num_keys - 1;
    if(node->is_leaf){
        while(i>=0 && key < node->keys[i]){
            node->keys[i+1] = node->keys[i];
            i--;
        }
        node->keys[i+1] = key;
        node->num_keys++;
    }
    else{
        while(i>=0 && key < node->keys[i]){
            node->keys[i+1] = node->keys[i];
            i--;
        }
        i++;
        if(node->children[i]->num_keys == 2* T -1){
            split_child(node,i);
            if(node->keys[i]){i++;}
        }
        polyINSERT(node->children[i],key);
    }
}
void insert(root* rooq, int key){
    if(rooq->num_keys == 2*T-1){
        root * z = create(false);
        z->children[0]=rooq;
        split_child(z,0);
        polyINSERT(z,key);
        return z;
    }
    else{
        polyINSERT(rooq,key);
        return rooq;
    }
}
int naibolsheeSLEVA(root * qet){
    if(qet->is_leaf){
        return(qet->keys[qet->num_keys-1]);
    }
    else{
        return naibolsheeSLEVA(qet->children[qet->num_keys]);
    }
}
int naimensheeSPRAVA(root * qet){
    while(!qet->is_leaf){
        qet = qet->children[0];
    }
    return(qet->keys[0]);
}
void merge(root * parent, int idx){
    root * child = parent->children[idx];
    root * qet = parent->children[idx+1];
    child->keys[T-1] = parent->keys[idx];

    for(int i = 0; i < qet->num_keys;i++){
        child->keys[i+T] = qet->keys[i];
    }
    for(int i = 0;i<=qet->num_keys;i++){
        child->children[i+T] = qet->children[i];
    }
    for(int i = idx + 1;i<parent->num_keys;i++){
        parent->keys[i-1] = parent->keys[i];
    }
    for(int i = idx+2;i<=parent->num_keys;i++){
        parent->children[i-1] = parent->children[i];
    }
    parent->num_keys--;
    child->num_keys += qet->num_keys+1;
    free(qet);
}
void Ylevogo(root * parent, int idx){
    root * qet = parent->children[idx-1];
    root * child = parent->children[idx];
    for(int i=child->num_keys-1;i>=0;i--){
        child->keys[i+1] = child->keys[i];
    }
    child->keys[0] = parent->keys[idx-1];
    parent->keys[idx-1] = qet->keys[qet->num_keys-1];
    if(!qet->is_leaf){
        for(int i = child->num_keys; i >= 0 ; i--){
            child->children[i+1] = child->children[i];
        }
        child->children[0] = qet->children[qet->num_keys];
    }    
    qet->num_keys--;
    child->num_keys++;
}
void Ypravogo(root * parent, int idx){
    root * child = parent->children[idx];
    root * qet = parent->children[idx+1];
    child->keys[child->num_keys] = parent->keys[idx];
    child->num_keys++;
    parent->keys[idx] = qet->keys[0];
    for(int i = 0 ; i < qet->num_keys-1;i++){
        qet->keys[i] = qet->keys[i+1];
    }
    
    if(!qet->is_leaf){
        child->children[child->num_keys] = qet->children[0];
        for(int i = 0 ;i<= qet->num_keys;i++){
            qet->children[i] = qet->children[i+1];
        }
    }    
    qet->num_keys--;

}
void fill(root * parent, int idx){
    root * qet = parent->children[idx];
    if(parent->children[idx-1]->num_keys >= T){
        Ylevogo(parent,idx);
    }
    if(parent->children[idx+1]->num_keys >= T){
        Ypravogo(parent,idx);
    }
    else{
        if(idx<parent->num_keys){
            merge(parent,idx);
        }
        else{
            merge(parent, idx-1);
        }
    }
}
void delete_key(root * rooq, int key){
int idx=0;    
while(idx<rooq->num_keys && key > rooq->keys[idx]){
    idx++;
}
    if(idx<rooq->num_keys && key == rooq->keys[idx]){
        if(rooq->is_leaf){
            for (int i = idx + 1; i < rooq->num_keys; i++){
                rooq->keys[i - 1] = rooq->keys[i];}
            rooq->num_keys--;
        }
        else{
            if (rooq->children[idx]->num_keys >= T) {
                int pred = naibolsheeSLEVA(rooq->children[idx]);
                rooq->keys[idx] = pred;
                delete_key(rooq->children[idx], pred);
            } else if (rooq->children[idx + 1]->num_keys >= T) {
                int succ =  naimensheeSPRAVA(rooq->children[idx + 1]);
                rooq->keys[idx] = succ;
                delete_key(rooq->children[idx + 1], succ);
            } else {
                merge(rooq, idx);
                delete_key(rooq->children[idx], key);
            }
        }
    }
    else{
        bool qq =(idx == rooq->num_keys);
        if(rooq->children[idx]->num_keys < T){
            fill(rooq,idx);
        }
        if(qq && idx > rooq->num_keys){
            delete_key(rooq->children[idx-1],key);
        }
        else{
            delete_key(rooq->children[idx],key);
        }
    }
}