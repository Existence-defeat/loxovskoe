#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>


struct  node 
{
    int value;
    struct node * next;   
};
bool is_prime(int x){
    if(x==0) return 0;
    if (x==1)return 0;
    if(x==2) return 1;
    if(x==3)return 1;
    else{
        for(int j = 2; j*j <= x ; j++){
            if(x%j==0) {return 0;}
        }
        return 1;
    }
}
int * append(int * array, int * size, int x){
    array = realloc(array,(*size+1)*sizeof(int));
    array[*size] = x;
    (*size)++;
    return array;
}
void rekurs(struct node* x,int y) {
    if (x == NULL) return;
    struct node* last = x;
    for(int i = 0 ; i < y-1; i++){
        printf("%d ",last->value);
        last = last->next;
    }
    printf("\nlast value: %d",last->value);
}

long long dokakogo(int x){
    assert(x>=10);
    double y =x;
    double res = y * (log(y)+log(log(y)));
    return (long long ) round(res);
}
void pogna(int x){
    int * array = NULL;
    int  sizeq = 0;
    long long size = dokakogo(x);
    struct node * spisokq = malloc(sizeof(struct node)*size);
    for(int i = 0; i < size; i++){
        spisokq[i].value = i+1;
        if(i<size-1){
            spisokq[i].next = &spisokq[i+1];
        }
        else{
            spisokq[i].next=NULL;
        }
    }
    while(1){
                    struct node * arr = spisokq;
                    while(arr && arr->next){
                        if(!is_prime(arr->next->value)){
                            arr->next = arr->next->next;
                        }
                        else{
                            arr = arr->next;
                        }
                    }
                    
        
            break;
        }
    if(spisokq != NULL){
    rekurs(spisokq,x);
    }
}
int main(){
    char str[100];
    printf("vvedi chislo\n");
    fgets(str,sizeof(str),stdin);
    char * tok = strtok(str,"\n");
    int x = atoi(tok);
    pogna(x);
    return 0;
}