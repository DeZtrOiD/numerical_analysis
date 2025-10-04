#include "matrix.h"

/// TODO: add complex num for matrix via template, change std::vector to new array 

template double Matrix::det<Matrix::GAUSS>() const;
template double Matrix::det<Matrix::LU>() const;
template double Matrix::det<Matrix::THOMAS>() const;

template Matrix Matrix::solve<Matrix::GAUSS>(const Matrix& b) const;
template Matrix Matrix::solve<Matrix::LU>(const Matrix& b) const;
template Matrix Matrix::solve<Matrix::THOMAS>(const Matrix& b) const;
template Matrix Matrix::solve<Matrix::JACOBI>(const Matrix& b) const;
template Matrix Matrix::solve<Matrix::SEIDEL>(const Matrix& b) const;

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
    *this = *this * other;
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

        if (max_idx != i) {
            for (std::size_t j = i; j < matrix._columns; j++) {
                double tmp = matrix(max_idx, j);
                matrix(max_idx, j) = matrix(i, j);
                matrix(i, j) = tmp;
            }
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
        auto LU = this->lu_decompose();
        double det = 1.0;
        for (std::size_t i = 0; i < LU._rows; i++) {
            // if (i < _rows) {
            //     det *= LU(LU._rows - 1 -  i, i);
            // }
            // else {
            det *= LU(i, i + LU._rows);
            // }
        }
        return det; // L.det_from_diag() * U.det_from_diag();
    } else if constexpr (TYPE_OP == THOMAS) {
        std::vector<double> tmp = thomas_method<true>(*this);
        double det = 1.0;
        for (double i : tmp) {
            det *= i;
        }
        return det;
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

// std::pair<Matrix, Matrix> 
Matrix Matrix::lu_decompose() const {
    if (_rows != _columns) {
        throw std::logic_error("LU decomposition requires a square matrix.");
    }
    Matrix L = Matrix::get_identity(_rows);
    Matrix U = Matrix::get_zero_matrix(_rows);
    Matrix LU = Matrix::append_matrix(Matrix::get_identity(_rows), Matrix::get_zero_matrix(_rows));
    for (std::size_t i = 0; i < _rows; i++) {
        for (std::size_t j = 0; j < _rows; j++) {
            double sum = (*this)(i, j);
            for (std::size_t k = 0; k <= i; k++) {
                // sum -= L(i, k) * U(k, j);
                sum -= LU(i, k) * LU(k, j + _rows);
            }
            if (i <= j) {
                if (std::abs(sum) < _eps) {
                    throw std::logic_error("Zero pivot encountered.");
                }
                LU(i, j + _rows) = sum;
            } else {
                LU(i, j) = sum / LU(j, j + _rows);
            }
        }
    }
    return LU;//{L, U};
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
        auto LU = this->lu_decompose();
        // Matrix res = gauss_substitution<false>(Matrix::append_matrix(L, b));

        return LU.LU_substitution(b);// gauss_substitution<true>(Matrix::append_matrix(U, res));
    } else if constexpr (TYPE_OP == THOMAS) {
        if (_rows != _columns) {
            throw std::logic_error("The Thomas method requires a square matrix.");
        }
        return Matrix{this->_rows, 1,  thomas_method<false>(Matrix::append_matrix(*this, b))};
    } else if constexpr (TYPE_OP == JACOBI) {
        std::size_t iter = 0;
        Matrix tmp = *this;
        Matrix tmp_b = b;
        permute_rows_for_diagonal_dominance(tmp, tmp_b);
        tmp = tmp.iteration_method<true>(tmp_b, iter);
        std::cout << "----- Iteration count JACOBI -----" << std::endl
            << iter << std::endl << "----------------------------------" << std::endl;
        return std::move(tmp);
    } else if constexpr (TYPE_OP == SEIDEL) {
        std::size_t iter = 0;
        Matrix tmp = *this;
        Matrix tmp_b = b;
        permute_rows_for_diagonal_dominance(tmp, tmp_b);
        tmp = tmp.iteration_method<false>(tmp_b, iter);
        std::cout << "----- Iteration count SEIDEL -----" << std::endl
            << iter << std::endl << "----------------------------------" << std::endl;
        return std::move(tmp);
    } else {
        throw std::logic_error("NOT IMPLEMENTED.");
    }
}

Matrix Matrix::LU_substitution(const Matrix& b) const {
    Matrix res_f{this->_rows, 1, std::vector<double>(this->_rows, 0.0)};
    Matrix res_b{this->_rows, 1, std::vector<double>(this->_rows, 0.0)};

    for (std::size_t i = 0; i < this->_rows; i++) {
        double sum = 0.0;
        for (std::size_t k = 0; k < i; k++) {
            sum += (*this)(i, k) * res_f(k, 0);
        }
        res_f(i, 0) = (b(i, 0) - sum) / (*this)(i, i);
    }

    for (std::size_t i = this->_rows; i-- > 0;) {
        double sum = 0.0;
        for (std::size_t k = i + 1; k < this->_rows; k++) {
            sum += (*this)(i, k +  this->_rows) * res_b(k, 0);
        }
        res_b(i, 0) = (res_f(i, 0) - sum) / (*this)(i, i +  this->_rows);
    }
    return res_b;
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
        auto LU = this->lu_decompose();
        Matrix b {this->_rows, 1, std::vector<double>(this->_rows, 0.0)};
        b(0, 0) = 1.0;
        Matrix res = LU.LU_substitution(b);
        for (std::size_t i = 1; i < this->_rows; i++) {
            b(i - 1, 0) = 0.0;
            b(i, 0) = 1.0;
            res = Matrix::append_matrix(res, LU.LU_substitution(b));
            // res = Matrix::append_matrix(res,
            //     gauss_substitution<true>(Matrix::append_matrix(U,
            //         gauss_substitution<false>(Matrix::append_matrix(L, b)))
            //     )
            // );
        }
        return res;
    } 
}

template<bool DETERMINANT>
std::vector<double> Matrix::thomas_method(const Matrix& matrix) {
    /// TODO: tridiagonal check
    const std::size_t n = matrix._rows;
    if constexpr (DETERMINANT) {
        std::vector<double> diag(n, 0.0);
        diag[0] = matrix[0, 0];  // new b_0
        for (std::size_t i = 1; i < n; i++) {
            double alpha = -matrix(i - 1, i) / diag[i-1];  // alpha_i-1
            diag[i] = matrix(i, i) + matrix(i, i - 1) * alpha;
        }
        return std::move(diag);
    } else {
        if (n == 1) {
            return std::vector<double>{matrix(0, n) / matrix(0, 0)};
        }
        std::vector<double> alpha_beta(n * 2, 0.0);
        alpha_beta[0] = -matrix(0, 1) / matrix(0, 0);  // alpha_1
        alpha_beta[n] = matrix(0, n) / matrix(0, 0);  // beta_1
        for (std::size_t i = 1; i < n - 1; i++) {
            alpha_beta[i] = -matrix(i, i + 1) / (
                matrix(i, i) + matrix(i, i - 1) * alpha_beta[i - 1]);
                // alpha_i = -Ci / (Bi + Ai*alpha_i-1)
            alpha_beta[i + n] = (matrix(i, n) -
                matrix(i, i - 1) * alpha_beta[i + n - 1]) /
                (matrix(i, i) + matrix(i, i - 1) * alpha_beta[i - 1]);
                // beta_i = (Fi - Ai*beta_i-1) / (Bi + Ai*alpha_i-1)
        }
        alpha_beta[n * 2 - 1] = (matrix(n - 1, n) -
            matrix(n - 1, n - 2) * alpha_beta[n * 2 - 2]) /
            (matrix(n - 1, n - 1) + matrix(n - 1, n - 2) * alpha_beta[n - 2]);
            // beta_i = (Fi - Ai*beta_i-1) / (Bi + Ai*alpha_i-1)
        std::vector<double> res(n, 0.0);
        res[n - 1] = alpha_beta[n * 2 - 1];
        for (std::size_t i = n - 1; i > 0; i--) {
            res[i - 1] = alpha_beta[i - 1] * res[i] + alpha_beta[n + i - 1];
        }
        return std::move(res);
    }
}

template<bool JACOBI_VER>
Matrix Matrix::iteration_method(const Matrix& b, std::size_t& iteration_count) const {
    Matrix res{this->_rows, 1, std::vector<double>(this->_rows, 0.0)};
    double diff = _eps * 10;
    constexpr std::size_t max_iteration = 10000;
    iteration_count = 0;
    Matrix old_x = res;
    double old_x_seidel = res(0, 0);
    do {
        iteration_count++;
        diff = 0.0;
        for (std::size_t i = 0; i < this->_rows; i++) {
            if constexpr (JACOBI_VER) {
                res(i, 0) = b(i, 0);
                for (std::size_t j = 0; j < this->_rows; j++) {
                    if (i != j) {
                        res(i, 0) -= old_x(j, 0) * (*this)(i, j);
                    }
                }
                res(i, 0) /= (*this)(i, i);
                diff += (res(i, 0) - old_x(i, 0)) * (res(i, 0) - old_x(i, 0));
            } else {
                old_x_seidel = res(i, 0);
                res(i, 0) = b(i, 0);
                for (std::size_t j = 0; j < this->_rows; j++) {
                    if (i != j) {
                        res(i, 0) -= res(j, 0) * (*this)(i, j);
                    }
                }
                res(i, 0) /= (*this)(i, i);
                diff += (res(i, 0) - old_x_seidel) * (res(i, 0) - old_x_seidel);
            }
        }
        diff = std::sqrt(diff);
        old_x = res;
    } while (std::abs(diff) > _eps && iteration_count < max_iteration);
    return std::move(res);
}

void Matrix::permute_rows_for_diagonal_dominance(Matrix& matrix, Matrix& b) {
    if (matrix._rows < 2) return;
    for (std::size_t i = 0; i < matrix._rows; i++) {
        std::size_t max_idx = 0;
        for (std::size_t j = 1; j < matrix._rows; j++) {
            if (matrix(max_idx, i) < matrix(j, i)) {
                max_idx = j;
            }
        }
        if (max_idx != i) {
            double tmp = 0.0;
            for (std::size_t j = 0; j < matrix._columns; j++) {
                tmp = matrix(max_idx, j);
                matrix(max_idx, j) = matrix(i, j);
                matrix(i, j) = tmp;
            }
            tmp = b(i, 0);
            b(i, 0) = b(max_idx, 0);
            b(max_idx, 0) = tmp;
        }
    }
}

Matrix Matrix::Householder_vector(std::size_t k) const {
    if (k >= _rows) {
        throw std::logic_error("There is no such vector.");
    }
    const std::size_t n = _rows;
    std::vector<double> vec_data(n, 0.0);

    double norm_sq = 0.0;
    for (std::size_t i = k; i < n; ++i) {
        double xi = (*this)(i, k);
        norm_sq += xi * xi;
    }
    double norm_x = std::sqrt(norm_sq);

    double xk = (*this)(k, k);
    double alpha = -std::copysign(norm_x, xk);

    vec_data[k] = xk - alpha;
    for (std::size_t i = k + 1; i < n; ++i) {
        vec_data[i] = (*this)(i, k);
    }

    return Matrix(n, 1, std::move(vec_data));
}

Matrix Matrix::transpose() const {
    std::vector<double> transposed_data(_rows * _columns);
    for (std::size_t i = 0; i < _rows; ++i) {
        for (std::size_t j = 0; j < _columns; ++j) {
            transposed_data[j * _rows + i] = _data[i * _columns + j];
        }
    }
    return Matrix(_columns, _rows, std::move(transposed_data));
}

std::pair<std::vector<double>, std::vector<double>> Matrix::block_eigenvalue(const Matrix& Ak) {
        const std::size_t n = Ak._rows;
        std::vector<double> re(n, 0.0), im(n, 0.0);
        std::size_t i = 0;
        while (i < n) {
            if (i + 1 < n && std::abs(Ak(i + 1, i)) > _eps) {
                double a = Ak(i, i), b = Ak(i, i + 1), c = Ak(i + 1, i), d = Ak(i + 1, i + 1);
                double tr = a + d;
                double det = a * d - b * c;
                double disc = tr * tr - 4.0 * det;
                if (disc >= 0.0) {
                    double s = std::sqrt(disc);
                    re[i] = (tr + s) / 2.0;
                    im[i] = 0.0;
                    re[i + 1] = (tr - s) / 2.0;
                    im[i + 1] = 0.0;
                } else {
                    double s = std::sqrt(-disc);
                    re[i] = tr / 2.0;
                    im[i] =  s / 2.0;
                    re[i + 1] = tr / 2.0;
                    im[i + 1] = -s / 2.0;
                }
                i += 2;
            } else {
                re[i] = Ak(i, i);
                im[i] = 0.0;
                ++i;
            }
        }
        return {re, im};
}

template<bool EIGENVALUES>
std::pair<Matrix, Matrix> Matrix::get_QR_algorithm(std::size_t& iter) const {
    if (_rows != _columns) {
        throw std::logic_error("QR algorithm requires a square matrix.");
    }
    const std::size_t n = _rows;
    const std::size_t max_iter = 1000;;
    iter = 0;

    if constexpr (!EIGENVALUES) {
        // Single Householder QR decomposition
        Matrix Q = Matrix::get_identity(n);
        Matrix R = *this;
        for (std::size_t k = 0; k + 1 < n; ++k) {
            Matrix v = R.Householder_vector(k);
            Matrix vT = v.transpose();
            double vTv = (vT * v)(0, 0);
            if (std::abs(vTv) < _eps / 10.0) continue;
            Matrix H = Matrix::get_identity(n) - (v * vT) * (2.0 / vTv);
            R = H * R;
            Q = Q * H;
        }
        iter = n - 1;
        return {Q, R};
    } else {
        // QR iterations
        Matrix Ak = *this;
        // Matrix A_old = Ak;
        auto eigen = block_eigenvalue(Ak); 
        for (; iter < max_iter; ++iter) {
            // compute QR of Ak
            Matrix Qk = Matrix::get_identity(n);
            Matrix Rk = Ak;
            for (std::size_t k = 0; k + 1 < n; ++k) {
                Matrix v = Rk.Householder_vector(k);
                Matrix vT = v.transpose();
                double vTv = (vT * v)(0, 0);
                if (std::abs(vTv) < 1e-16) continue;
                Matrix H = Matrix::get_identity(n) - (v * vT) * (2.0 / vTv);
                Rk = H * Rk;
                Qk = Qk * H;
            }

            Ak = Rk * Qk;

            if (qr_diff(Ak, eigen) < _eps) break;
            // A_old = Ak;
        }

        auto [re, im] = block_eigenvalue(Ak);
        Matrix reM(n, 1, std::move(re));
        Matrix imM(n, 1, std::move(im));
        return {reM, imM};
    }
}

double Matrix::qr_diff(Matrix& Ak, std::pair<std::vector<double>, std::vector<double>>& old_eigen) {
    auto [re_1, im_1] = block_eigenvalue(Ak);
    // auto [re_2, im_2] = old_eigen;
    const std::size_t n = Ak._rows;
    double max_diff = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double diff_re = (re_1[i] - old_eigen.first[i]) * (re_1[i] - old_eigen.first[i]);
        double diff_im = (im_1[i] - old_eigen.second[i]) * (im_1[i] - old_eigen.second[i]);
        double val = std::sqrt(diff_im + diff_re);
        if (val > max_diff) max_diff = val;
    }
    old_eigen = {re_1, im_1};
    return max_diff;
}

std::pair<Matrix, Matrix> Matrix::get_QR() const {
    std::size_t iter = 0;
    auto tmp = (*this).get_QR_algorithm<false>(iter);
    std::cout << "----- Iteration count get_QR -----" << std::endl
        << iter << std::endl << "----------------------------------" << std::endl;
    return tmp;
}

Matrix Matrix::get_eigenvalue() const {
    std::size_t iter = 0;
    auto [re, im] = (*this).get_QR_algorithm<true>(iter);
    std::cout << "----- Iteration count get_eigenvalue -----" << std::endl
        << iter << std::endl << "----------------------------------" << std::endl;
    return append_matrix(re, im);
}
