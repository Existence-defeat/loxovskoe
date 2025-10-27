#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdbool.h>
enum color{RED,BLACK};
typedef struct node{
    int key;
    enum color Color;
    struct node * left,*right,*parent;

} node;
node * create(int key){
    node * qet = (node *)malloc(sizeof(node));
    qet->key = key;
    qet->left = NULL;
    qet->right = NULL;
    qet->Color=RED;
    qet->parent = NULL;
    return qet;
}
void insert(int key,node ** root){
    node * qet = create(key);
    node * y = NULL;
    node * x = * root;
    while(x!=NULL){
        y = x;
        if(qet->key < x->key){
            x = x->left;
        }
        else{
            x = x->right;
        }
    }
    qet->parent = y;
    if(y==NULL){
        * root = qet;
    }
    else if(qet->key < y->key){
        y->left = qet;
    }
    else{
        y->right = qet;
    }
}
void swapleft(node ** root, node * x){
    if (x == NULL || x->right == NULL) return;
    node * y = x->right;
    if(y->left){
        x->right = y->left;
        y->left->parent = x;
    }
    y->left = x;
    if(x->parent == NULL){
        *root = y;
    }
    else if(x== x->parent->left){
        x->parent->left = y;
    }
    else{
        x->parent->right = y;
    }
    y->parent = x->parent;
    x->parent = y;

}
void swapright(node **root, node *y) {
    if (y == NULL || y->left == NULL) return;

    node *x = y->left;
    y->left = x->right;
    if (x->right)
        x->right->parent = y;

    x->parent = y->parent;

    if (y->parent == NULL)
        *root = x;
    else if (y == y->parent->left)
        y->parent->left = x;
    else
        y->parent->right = x;

    x->right = y;
    y->parent = x;
}
void fix(node ** root, node * z){
    while(z->parent && z->parent->Color == RED){
        if(z->parent->parent->right == z->parent){
            node * x = z->parent->parent->left;
            if(x && x->Color == RED){
                x->Color = BLACK;
                z->parent->Color = BLACK;
                z->parent->parent->Color = RED;
                z = z->parent->parent;
            }
            else{
                if(z ==  z->parent->left){
                    swapright(*root, z->parent);
                    z = z->right;
                }
                z->parent->Color = BLACK;
                z->parent->parent->Color = RED;
                swapleft(*root,z->parent->parent);
            }
        }
        else{
            node * x = z->parent->parent->right;
            if(x && x->Color == RED){
                x->Color = BLACK;
                z->parent->Color = BLACK;
                z->parent->parent->Color = RED;
                z = z->parent->parent;
            }
            else{
                if(z ==  z->parent->right){
                    swapleft(*root, z->parent);
                    z = z->left;
                }
                z->parent->Color = BLACK;
                z->parent->parent->Color = RED;
                swapright(*root,z->parent->parent);
            }
        }
      
    }
    (*root)->Color = BLACK;
}
void swap(node * root, node * a1 , node * a2){
    if(root == a1){
        root = a2;
    }
    else if(a1 == a1->parent->left){
        a1->parent->left = a2;
    }
    else{
        a1->parent->right = a2;
    }
}
node * maxleft(node * root){
    while(root->left!= NULL){
        root = root->left;
    }
    return root;
}
void delete(node ** root , node * z){
    node * y = z;
    enum color orig = y->Color;
    node * x;
    if(!y->left){
        x=y->right;
        swap(*root, z, x);
    }
    else if(!y->right){
        x=y->left;
        swap(*root, z , x);
    }
    else{
        y = maxleft(z);
        x = y->right;
        orig = y->Color;
        if(z->right != y){
            swap(*root,y,z->right);
            y->right = z->right;
            z->parent->right = y;
        }
        swap(*root,z,y);
        y->left = z->left;
        y->left->parent = y;
        y->Color = z->Color;

    }
    if(orig==BLACK){
        fix(*root,x);
    }
    free(z);
}