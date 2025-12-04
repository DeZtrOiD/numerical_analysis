#include "matrix_api.h"
#include <vector>

extern "C" {

void* create_matrix(int rows, int cols, double* data) {
    std::vector<double> vec(data, data + rows * cols);
    return new Matrix(rows, cols, std::move(vec));
}

void free_matrix(void* matrix_ptr) {
    delete static_cast<Matrix*>(matrix_ptr);
}

void matrix_solve_thomas(void* matrix_ptr, double* b, double* result) {
    Matrix* mat = static_cast<Matrix*>(matrix_ptr);
    std::vector<double> vec_b(b, b + mat->get_zero_matrix(mat->_rows)._rows); // b size check
    Matrix b_mat(mat->_rows, 1, std::move(vec_b));
    Matrix x = mat->solve<Matrix::THOMAS>(b_mat);
    for (size_t i = 0; i < x._rows; i++) {
        result[i] = x(i,0);
    }
}

void matrix_print(void* matrix_ptr) {
    Matrix* mat = static_cast<Matrix*>(matrix_ptr);
    mat->print();
}

void matrix_solve_gauss(void* matrix_ptr, double* b, double* result) {
    Matrix* mat = static_cast<Matrix*>(matrix_ptr);

    std::vector<double> vec_b(b, b + mat->_rows);
    Matrix b_mat(mat->_rows, 1, std::move(vec_b));

    Matrix x = mat->solve<Matrix::GAUSS>(b_mat);

    for (size_t i = 0; i < x._rows; i++) {
        result[i] = x(i, 0);
    }
}

}
