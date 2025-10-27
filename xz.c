#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdbool.h>
#define pi 3.14
float ygol1[320];
float ygol2[640];
void qwww(void){
    for(float i = 0;i<pi;i+=0.01){
        ygol1[(int)(i*100)]=i; 
    }
    for(float i = 0; i<=2*pi;i+=0.01){
        ygol2[(int)(i*100)]=i;
    }
}
void qqq(int r){
    float x=0;
    for(int i = 0; i < sizeof(ygol1)/sizeof(float);i++){
        for(int j = 0;j<sizeof(ygol2)/sizeof(float);j++){
            x+=j;
        }
    }
    printf("%f",(double)x);
}
int main(){
    qqq(1);
    return 1;
}