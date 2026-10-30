// Matrix Types
// This script includes all kinds off matrices needed for a NN (struct, zeros, ones, random, save, load, etc)

#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <string>

// Random number engine (Mersenne Twister) with a fixed seed,
std::mt19937 gen(42); // Fixed seed for reproducibility

// Distribution that maps the raw random bits to floats in [-1, 1).
// Usage: dist(gen) returns one new random number per call.
std::uniform_real_distribution<float> dist(-1.0, 1.0);

// A 2D matrix stored as ONE flat vector in row-major order:
// element (i, j) lives at index i * cols + j.
struct Matrix {
    size_t rows = 0;          // number of rows (default 0 for an empty matrix)
    size_t cols = 0;          // number of columns
    std::vector<float> data;  // all elements, length rows * cols

    // Default constructor: creates an empty 0x0 matrix
    Matrix() = default;

    // Constructor: sets dimensions and takes the initial data.
    Matrix(size_t r, size_t c, std::vector<float> d) : rows(r), cols(c), data(d) {}

    // Overload the () operator for easy element access: m(i, j).
    float& operator()(size_t i, size_t j) { return data[i * cols + j]; }

    // Total number of elements (rows * cols).
    size_t size() const { return rows * cols; }

    // Raw pointer to the first element of the data.
    float* ptr() { return data.data(); }
};


// Creates a rows x cols matrix filled with random values in [-1, 1).
struct Matrix randomized_weight(size_t rows, size_t cols) {
    // First allocate space ...
    struct Matrix weights(rows, cols, std::vector<float>(rows * cols));
    // ... then draw a NEW random number for every element.
    for (size_t i = 0; i < rows * cols; ++i) {
        weights.data[i] = dist(gen);
    }

    return weights;
}

// Creates a rows x cols matrix filled with zeros.
struct Matrix zero_weights(size_t rows, size_t cols) {
    // vector(n, value) creates n elements, all set to 'value'.
    struct Matrix weights(rows, cols, std::vector<float>(rows * cols, 0.0f));

    return weights;
}

// Creates a rows x cols matrix filled with ones.
struct Matrix ones_weights(size_t rows, size_t cols) {
    // vector(n, value) creates n elements, all set to 'value'.
    struct Matrix weights(rows, cols, std::vector<float>(rows * cols, 1.0f));

    return weights;
}

// Binary file format:
//   [size_t rows][size_t cols][rows * cols floats]
// TODO: Make Errorhandeling for file operations
void save_weights(const std::string& filename, const struct Matrix& weights) {
    // Open in binary mode so no text conversion
    std::ofstream f(filename, std::ios::binary);
    size_t r = weights.rows;
    size_t c = weights.cols;

    // write(address, number_of_bytes) copies raw bytes from memory into the file.
    // reinterpret_cast<const char*> only reinterprets the pointer as a byte pointer; no values are converted.
    f.write(reinterpret_cast<const char*>(&r), sizeof(size_t));  // header: rows
    f.write(reinterpret_cast<const char*>(&c), sizeof(size_t));  // header: cols
    // Payload: all floats in one go (works because the data is contiguous).
    f.write(reinterpret_cast<const char*>(weights.data.data()),
            static_cast<std::streamsize>(sizeof(float) * weights.size()));
}

// Reads a matrix that was written by save_weights.
// The dimensions come from the file header, not from parameters.
// TODO: Make Errorhandeling for file operations
struct Matrix load_weights(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    size_t r = 0, c = 0;  // initialized, so they are not garbage if reading fails

    // Read the header first ...
    f.read(reinterpret_cast<char*>(&r), sizeof(size_t));
    f.read(reinterpret_cast<char*>(&c), sizeof(size_t));
    // ... then allocate exactly the right amount of space and read the floats.
    std::vector<float> data(r * c);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(sizeof(float) * r * c));

    return Matrix(r, c, data);
}

int main() {
    size_t rows = 3;
    size_t cols = 4;

    // Create three test matrices
    struct Matrix random_weights = randomized_weight(rows, cols);
    struct Matrix zeroed_weights = zero_weights(rows, cols);
    struct Matrix oned_weights = ones_weights(rows, cols);

    // Print the random matrix row by row
    std::cout << "Random Weights:" << "size(): " << random_weights.size() << std::endl;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << random_weights(i, j) << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }

    // Print the zero matrix
    std::cout << "Zero Weights:" << "size(): " << zeroed_weights.size() << std::endl;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << zeroed_weights(i, j) << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }

    // Print the ones matrix
    std::cout << "Ones Weights: " << "size(): " << oned_weights.size() << std::endl;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << oned_weights(i, j) << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }

    // Round trip test: save the random matrix to disk and load it back.
    // The relative path depends on the directory the program is started from.
    save_weights("../data/random_weights.bin", random_weights);
    struct Matrix loaded_weights = load_weights("../data/random_weights.bin");

    // The loaded matrix should be identical to random_weights
    std::cout << "Loaded Weights: " << "size(): " << loaded_weights.size() << std::endl;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << loaded_weights(i, j) << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }

    return 0;
}