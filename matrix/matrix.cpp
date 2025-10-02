#include "matrix.h"

#include<iostream>

template double Matrix::det<Matrix::GAUSS>() const;
template double Matrix::det<Matrix::LU>() const;
template Matrix Matrix::solve<Matrix::GAUSS>(const Matrix& b) const;
template Matrix Matrix::solve<Matrix::LU>(const Matrix& b) const;
template Matrix Matrix::inverse<Matrix::GAUSS>() const;
template Matrix Matrix::inverse<Matrix::LU>() const;

Matrix::Matrix(std::size_t rows, std::size_t columns, std::vector<double>& data):
        _rows(rows), _columns(columns), _data(data) {
    if (rows == 0 || columns == 0) {
        throw std::logic_error("A dimensionless matrix is meaningless.");
    } else if (rows * columns < (rows > columns ? rows : columns)) {
        throw std::logic_error("The matrix isn't designed for such large sizes.");
    } else if (rows * columns != _data.size()) {
        throw std::logic_error("Data size does not match matrix dimensions.");
    }
}

Matrix::Matrix(std::size_t rows, std::size_t columns, std::vector<double>&& data):
        _rows(rows), _columns(columns), _data(std::move(data)) {
    if (rows == 0 || columns == 0) {
        throw std::logic_error("A dimensionless matrix is meaningless.");
    } else if (rows * columns < (rows > columns ? rows : columns)) {
        throw std::logic_error("The matrix isn't designed for such large sizes.");
    } else if (rows * columns != _data.size()) {
        throw std::logic_error("Data size does not match matrix dimensions.");
    }
}

Matrix::Matrix(const Matrix& other):
    _rows(other._rows), _columns(other._columns), _data(other._data) {}

Matrix::Matrix(Matrix&& other):
        _rows(other._rows), _columns(other._columns), _data(std::move(other._data)) {
    other.clear();
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        this->_columns = other._columns;
        this->_rows = other._rows;
        this->_data = other._data;
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        this->_columns = other._columns;
        this->_rows = other._rows;
        this->_data = std::move(other._data);
        other.clear();
    }
    return *this;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (_rows != other._rows || _columns != other._columns) {
        throw std::logic_error("Matrix dimensions must agree for addition.");
    }
    std::vector<double> tmp(_rows * _columns);
    for (std::size_t i = 0; i < _rows * _columns; i++) {
        tmp[i] = _data[i] + other._data[i];
    }
    return Matrix(_rows, _columns, std::move(tmp));
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (_rows != other._rows || _columns != other._columns) {
        throw std::logic_error("Matrix dimensions must agree for addition.");
    }
    for (std::size_t i = 0; i < _rows * _columns; i++) {
        _data[i] += other._data[i];
    }
    return *this;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (_rows != other._rows || _columns != other._columns) {
        throw std::logic_error("Matrix dimensions must agree for subtraction.");
    }
    std::vector<double> tmp(_rows * _columns);
    for (std::size_t i = 0; i < _rows * _columns; i++) {
        tmp[i] = _data[i] - other._data[i];
    }
    return Matrix(_rows, _columns, std::move(tmp));
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (_rows != other._rows || _columns != other._columns) {
        throw std::logic_error("Matrix dimensions must agree for subtraction.");
    }
    for (std::size_t i = 0; i < _rows * _columns; i++) {
        _data[i] -= other._data[i];
    }
    return *this;
}

Matrix Matrix::operator*(double scalar) const {
    std::vector<double> tmp(_rows * _columns);
    for (std::size_t i = 0; i < _rows * _columns; i++) {
        tmp[i] = _data[i] * scalar;
    }
    return Matrix(_rows, _columns, std::move(tmp));
}

Matrix operator*(double scalar, const Matrix& M) {
    return M * scalar;
}

Matrix& Matrix::operator*=(double scalar) {
    for (std::size_t i = 0; i < _rows * _columns; i++) {
        _data[i] *= scalar;
    }
    return *this;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (_columns != other._rows) {
        throw std::logic_error("Matrix dimensions must agree for multiplication.");
    }
    std::vector<double> tmp(_rows * other._columns, 0.0);
    for (std::size_t i = 0; i < _rows; i++) {
        for (std::size_t k = 0; k < _columns; k++) {
            double aik = (*this)(i, k);
            for (std::size_t j = 0; j < other._columns; j++) {
                tmp[i * other._columns + j] += aik * other(k, j);
            }
        }
    }
    return Matrix(_rows, other._columns, std::move(tmp));
}

Matrix& Matrix::operator*=(const Matrix& other) {
    *this = *this + other;
    return *this;
}

#if __cplusplus >= 202302L
constexpr double Matrix::operator[](std::size_t row, std::size_t column) const {
    if (row < _rows && column < _columns) {
        return _data[row * _columns + column];
    }
    throw std::out_of_range("Matrix doesn't contain such entries.");
}

double& Matrix::operator[](std::size_t row, std::size_t column) {
    if (row < _rows && column < _columns) {
        return _data[row * _columns + column];
    }
    throw std::out_of_range("Matrix doesn't contain such entries.");
}
#endif

constexpr double Matrix::operator()(std::size_t row, std::size_t column) const {
    if (row < _rows && column < _columns) {
        return _data[row * _columns + column];
    }
    throw std::out_of_range("Matrix doesn't contain such entries.");
}

double& Matrix::operator()(std::size_t row, std::size_t column) {
    if (row < _rows && column < _columns) {
        return _data[row * _columns + column];
    }
    throw std::out_of_range("Matrix doesn't contain such entries.");
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    if (m._rows == 0) {
        os << "| |\n";
        return os;
    } 
    constexpr int width = 12;
    constexpr int precision = 6;
    os << std::fixed << std::setprecision(precision);
    for (std::size_t i = 0; i < m._rows; i++) {
        os << "|";
        for (std::size_t j = 0; j < m._columns; j++) {
            os << " " << std::setw(width) << std::right << m(i, j);
        }
        os << " |\n";
    }
    return os;
}

#if __cplusplus >= 202302L
void Matrix::print() {
    if (_rows == 0) {
        std::print("| |\n");
        return;
    }
    for (std::size_t i = 0; i < _rows; i++) {
        std::print("| ");
        for (std::size_t j = 0; j < _columns; j++) {
            std::print("{:8.3f} ", _data[i * _columns + j]);
        }
        std::print("|\n");
    }
}
#endif

Matrix Matrix::append_matrix(const Matrix& A, const Matrix& B) {
    if (A._rows != B._rows) {
        throw std::logic_error("Unable to append matrix.");
    }
    const std::size_t new_cols = A._columns + B._columns;
    std::vector<double> tmp(A._rows * (new_cols));
    for (std::size_t i = 0; i < A._rows; i++) {
        for (std::size_t j = 0; j < A._columns; j++) {
            tmp[i * new_cols + j] = A(i, j);
        }
        for (std::size_t j = 0; j < B._columns; j++) {
            tmp[i * new_cols + A._columns + j] = B(i, j);
        }
    }
    return Matrix(A._rows, new_cols, std::move(tmp));
}

Matrix Matrix::get_identity(std::size_t n) {
    if (n == 0) {
        throw std::logic_error("The dimensionless identity matrix is a very important and unsupported object.");
    }
    Matrix tmp{n, n, std::vector<double>(n*n, 0.0)};
    for (std::size_t i = 0; i < n; i++) {
        tmp(i, i) =  1.0;
    }
    return tmp;
}

Matrix Matrix::get_zero_matrix(std::size_t n) {
    if (n == 0) {
        throw std::logic_error("The dimensionless zero matrix is a very important and unsupported object.");
    }
    return Matrix{n, n, std::vector<double>(n*n, 0.0)};
}

constexpr void Matrix::clear() noexcept {
    _rows = 0;
    _columns = 0;
    _data.clear();
}

template<bool INVERSE_CASE>
Matrix::FLAGS Matrix::gauss_forward_elimination(Matrix& matrix) {
    FLAGS flag = POSITIVE_FLAG;
    for (std::size_t i = 0; i < (INVERSE_CASE ? matrix._rows : matrix._rows - 1); i++) {
        // Searching maximum pivot and swapping rows
        size_t max_idx = i;
        for (std::size_t j = i + 1; j < matrix._rows; j++) {
            if (std::abs(matrix(max_idx, i)) < std::abs(matrix(j, i))) {
                max_idx = j;
            }
        }
        if (std::abs(matrix(max_idx, i)) < _eps) {
            if constexpr (INVERSE_CASE) {
                return SINGULAR_FLAG;
            } else {
                continue;
            }
        }

        for (std::size_t j = i; j < matrix._columns; j++) {
            double tmp = matrix(max_idx, j);
            matrix(max_idx, j) = matrix(i, j);
            matrix(i, j) = tmp;
            if constexpr (! INVERSE_CASE) {
                if (flag == POSITIVE_FLAG) {
                    flag = NEGATIVE_FLAG;
                } else {
                    flag = POSITIVE_FLAG;
                }
            }
        }

        // Substracting rows
        if constexpr (INVERSE_CASE) {
            double diag = matrix(i, i);
            for (std::size_t j = i; j < matrix._columns; j++) {
                matrix(i, j) /= diag;
            }

            for (std::size_t j = 0; j < matrix._rows; j++) {
                if (j == i) continue;
                const double factor = matrix(j, i) / matrix(i, i);
                matrix(j, i) = 0.0;
                for (std::size_t k = i + 1; k < matrix._columns; k++) {
                    matrix(j, k) -= factor * matrix(i, k);
                }
            }
        } else {
            for (std::size_t j = i + 1; j < matrix._rows; j++) {
                const double factor = matrix(j, i) / matrix(i, i);
                matrix(j, i) = 0.0;
                for (std::size_t k = i + 1; k < matrix._columns; k++) {
                    matrix(j, k) -= factor * matrix(i, k);
                }
            }
        }
    }
    return flag;
}

template<Matrix::OPERATION_TYPE TYPE_OP>
double Matrix::det() const {
    if (_rows != _columns) {
        throw std::logic_error("Non-square matrix have no determinant*.");
    } else if (_rows == 0) {
        throw std::logic_error("Det of a dimensionless matrix isn't most interesting thing.");
    }
    if constexpr (TYPE_OP == GAUSS) {
        Matrix tmp = *this;
        double det = 1.0;
        if (gauss_forward_elimination<false>(tmp) == NEGATIVE_FLAG) det = -1.0;
        det *= tmp.det_from_diag();
        return det;
    } else if constexpr (TYPE_OP == LU) {
        auto [L, U] = this->lu_decompose();
        return L.det_from_diag() * U.det_from_diag();
    } else {
        throw std::logic_error("NOT IMPLEMENTED.");
    }
}

double Matrix::det_from_diag() {
    double trc = 1.0;
    for (std::size_t i = 0; i < this->_rows; i++) {
        trc *= (*this)(i, i);
    }
    return trc;
}

template<bool BACK>
Matrix Matrix::gauss_substitution(const Matrix& matrix) {
    Matrix res{matrix._columns - 1, 1, std::vector<double>(matrix._columns - 1, 0.0)};
    if constexpr (BACK) {
        for (std::size_t i = matrix._columns - 1; i-- > 0;) {
            double sum = 0.0;
            for (std::size_t k = i + 1; k < matrix._columns - 1; k++) {
                sum += matrix(i, k) * res(k, 0);
            }
            res(i, 0) = (matrix(i, matrix._columns - 1) - sum) / matrix(i, i);
        }
    } else {
        for (std::size_t i = 0; i < matrix._columns - 1; i++) {
            double sum = 0.0;
            for (std::size_t k = 0; k < i; k++) {
                sum += matrix(i, k) * res(k, 0);
            }
            res(i, 0) = (matrix(i, matrix._columns - 1) - sum) / matrix(i, i);
        }
    }
    return res;
}

Matrix Matrix::split_matrix(const Matrix& matrix) {
    if ((matrix._columns & 1) != 0) {
        throw std::logic_error("Split err");
    }
    std::vector<double> tmp_vec;
    tmp_vec.reserve(matrix._rows * (matrix._columns / 2));
    for (std::size_t i = 0; i < matrix._rows; i++) {
        for (std::size_t j = 0; j * 2 < matrix._columns; j++) {
            tmp_vec.push_back(matrix(i, j + (matrix._columns / 2)));
        }
    }
    return Matrix(matrix._rows, matrix._columns / 2, std::move(tmp_vec));
}

std::pair<Matrix, Matrix> Matrix::lu_decompose() const {
    if (_rows != _columns) {
        throw std::logic_error("UL decomposition requires a square matrix.");
    }
    Matrix L = Matrix::get_identity(_rows);
    Matrix U = Matrix::get_zero_matrix(_rows);

    for (std::size_t i = 0; i < _rows; i++) {
        for (std::size_t j = 0; j < _rows; j++) {
            double sum = (*this)(i, j);
            for (std::size_t k = 0; k <= i; k++) {
                sum -= L(i, k) * U(k, j);
            }
            if (i <= j) {
                if (std::abs(sum) < _eps) {
                    throw std::logic_error("Zero pivot encountered.");
                }
                U(i, j) = sum;
            } else {
                L(i, j) = sum / U(j, j);
            }
        }
    }
    return {L, U};
}

template<Matrix::OPERATION_TYPE TYPE_OP>
Matrix Matrix::solve(const Matrix& b) const {
    if (_rows < _columns) {
        throw std::logic_error("An infinite number of solutions.");
    } else if (_rows == 0) {
        throw std::logic_error("It's impossible to solve an empty linear system.");
    } else if (b._columns != 1) {
        throw std::logic_error("It's impossible to solve non linear system.");
    } else if (b._rows != _rows) {
        throw std::logic_error("The number of rows of the matrix must be equal to the number of rows of the column vector.");
    }
    if constexpr (TYPE_OP == GAUSS) {
        Matrix tmp = Matrix::append_matrix(*this, b);
        gauss_forward_elimination<false>(tmp);
        for (std::size_t i = 0; i < _columns; i++) {
            if (std::abs(tmp(i, i)) < _eps) {
                throw std::logic_error("A linear system has an infinite number of solutions.");
            }
        }
        for (std::size_t i = _rows - 1; i >= _columns; i--) {
            if (std::abs(tmp(i, i)) > _eps) {
                throw std::logic_error("A linear system is inconsistent.");
            }
        }
        return gauss_substitution<true>(tmp);
    } else if constexpr (TYPE_OP == LU) {
        // Ax = b -> LUx = b -> Ly = b -> Ux = y
        auto [L, U] = this->lu_decompose();
        Matrix res = gauss_substitution<false>(Matrix::append_matrix(L, b));
        return gauss_substitution<true>(Matrix::append_matrix(U, res));
    }
}

template<Matrix::OPERATION_TYPE TYPE_OP>
Matrix Matrix::inverse() const {
    if (_rows != _columns) {
        throw std::logic_error("The matrix must be square.");
    }
    if constexpr (TYPE_OP == GAUSS) {
        Matrix tmp = append_matrix(*this, Matrix::get_identity(_rows));
        if (gauss_forward_elimination<true>(tmp) == SINGULAR_FLAG) {
            throw std::logic_error("Matrix is singular.");
        };
        return split_matrix(tmp);
    } else if constexpr (TYPE_OP == LU) {
        auto [L, U] = this->lu_decompose();
        Matrix b {this->_rows, 1, std::vector<double>(this->_rows, 0.0)};
        b(0, 0) = 1.0;
        Matrix res = gauss_substitution<true>(
            Matrix::append_matrix(U, gauss_substitution<false>(Matrix::append_matrix(L, b)))
        );
        for (std::size_t i = 1; i < this->_rows; i++) {
            b(i - 1, 0) = 0.0;
            b(i, 0) = 1.0;
            res = Matrix::append_matrix(res,
                gauss_substitution<true>(Matrix::append_matrix(U,
                    gauss_substitution<false>(Matrix::append_matrix(L, b)))
                )
            );
        }
        return res;
    } 
}