#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <dirent.h>
#define cflags "-Wall -g"
#define compiler "gcc"
#define maxlen 256
int ends_with_c(const char * str){
    size_t len = strlen(str);
    return len>2 && strcmp(str + len -2, ".c")==0;
}
char * delit(char * str){
    char * qet = strrchr(str,'.');
    if(qet){ *qet = '\0';}
    return str;
}
void create_all(){
    DIR *dir = opendir(".");
    struct dirent *lox;
    if(dir){
        while((lox = readdir(dir))){
            if(ends_with_c(lox->d_name)){
                char exe[maxlen];
                strcpy(exe,lox->d_name);
                delit(exe);

                char memory[maxlen * 2];
                snprintf(memory,sizeof(memory),"%s %s -o %s %s -lncurses",compiler,cflags,exe,lox->d_name);
                system(memory);
            }
        }
    }
    closedir(dir);
}
void delite_all(){
    DIR *dir = opendir(".");
    struct dirent *lox;
    if(dir){
        while((lox = readdir(dir))){
            if(ends_with_c(lox->d_name)){
                char exe[maxlen];
                strcpy(exe,lox->d_name);
                delit(exe);

                char memory[maxlen * 2];
                snprintf(memory,sizeof(memory),"rm -f %s",exe);
                system(memory);
            }
        }
    }
    closedir(dir);
}
int main(int argc, char * argv[]){
    if (argc < 2){
        exit(1);
    }
    if(strcmp(argv[1], "all")==0){
        create_all();
    }
    else if (strcmp(argv[1], "remove")==0)
    {
        delite_all();
    }
    
}