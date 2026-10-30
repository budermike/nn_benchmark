// Matrix Types
// This header includes all kinds off matrices needed for a NN (struct, zeros, ones, random, save, load, etc)

#include <string>

struct Matrix;

struct Matrix randomized_weight(size_t rows, size_t cols);

struct Matrix zero_weights(size_t rows, size_t cols);

struct Matrix ones_weights(size_t rows, size_t cols);

void save_weights(const std::string& filename, const struct Matrix& weights);

struct Matrix load_weights(const std::string& filename);