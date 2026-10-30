// Vectorized Operations
// This script includes all vector and matrix operations needed for a NN (Matmul, scalarproduct, Transpose, etc)
// Operations are written in pointer style, such that they are usable for CPU and CUDA

#include <matrix.hpp>
#include <vector>
#include <cassert>


// important TODO: for CUDA we need raw pointers (so no operator())
void matmul(const struct Matrix* A, const struct Matrix* B, struct Matrix* res) {
    assert(A->cols == B->rows);
    assert(res->rows == A->rows);
    assert(res->cols == B->cols);

    for (int i=0; i<A->rows; i++) {
        for (int j=0; j<B->cols; j++){
            float sum = 0.0f;
            for (int k=0; k<A->cols; k++){
                sum = sum + (*A)(i, k) * (*B)(k, j);
            }
            (*res)(i, j) = sum;
        }
    }
}

void transpose(const struct Matrix* A, struct Matrix* res){
    assert(res->rows == A->cols);
    assert(res->cols == A->rows);
    
    for (int i=0; i<A->rows; i++){
        for (int j=0; j<A->cols; j++){
            (*res)(i, j) = (*A)(j, i);
        }
    }
}

void dotproduct(const struct Matrix* x, const struct Matrix* y, float* res) {
    assert(x->rows == y->rows);
    assert(x->cols == 1);
    assert(y->cols == 1);

    float sum = 0;
    for (int i=0; i<x->rows; i++){
        sum = sum + x(i, 0) * y(i, 0);
    }
    (*res) = sum;
}