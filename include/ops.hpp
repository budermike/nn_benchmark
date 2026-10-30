// Vectorized Operations
// This header includes all vector and matrix operations needed for a NN (Matmul, scalarproduct, Transpose, etc)
// Operations are written in pointer style, such that they are usable for CPU and CUDA

#include <matrix.hpp>

void matmul(const struct Matrix* A, const struct Matrix* B, struct Matrix* res);

void transpose(const struct Matrix* A, struct Matrix* res);

void dotproduct(const struct Matrix* x, const struct Matrix* y, float* res);