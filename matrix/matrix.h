#pragma once

#include <vector>
#include <stdexcept>
#include <utility>
#include <ostream>
#include <iomanip>
#include <print>
#include <algorithm>
#include <cmath>
#include<iostream>

class Matrix {
private:
    enum FLAGS{
        NEGATIVE_FLAG,
        POSITIVE_FLAG,
        SINGULAR_FLAG,
        NON_SINGULAR_FLAG
    };

    std::size_t _rows;
    std::size_t _columns;
    std::vector<double> _data;
    static constexpr double _eps = 0.0001;

    template<bool INVERSE_CASE>
    static FLAGS gauss_forward_elimination(Matrix& matrix);

    template<bool BACK>
    static Matrix gauss_substitution(const Matrix& matrix);

    static Matrix append_matrix(const Matrix& A, const Matrix& B);
    static Matrix split_matrix(const Matrix& matrix);
    double det_from_diag();
    // std::pair<Matrix, Matrix>
    Matrix lu_decompose() const;

    Matrix LU_substitution(const Matrix& matrix) const;

    template<bool DETERMINANT>
    static std::vector<double> thomas_method(const Matrix& matrix);

    static void permute_rows_for_diagonal_dominance(Matrix& matrix, Matrix& b);

    Matrix Householder_vector(std::size_t num) const;

    Matrix transpose() const;

    static double qr_diff(Matrix& Ak, std::pair<std::vector<double>, std::vector<double>>& old_eigen);

    template<bool EIGENVALUES>
    std::pair<Matrix, Matrix> get_QR_algorithm(std::size_t& iteration_count) const;

    static std::pair<std::vector<double>, std::vector<double>> block_eigenvalue(const Matrix& Ak);

    template<bool JACOBI_VER>
    Matrix iteration_method(const Matrix& b, std::size_t& iteration_count) const;

public:
    enum OPERATION_TYPE {
        GAUSS,
        LU,
        THOMAS,
        JACOBI,
        SEIDEL,
    };


    Matrix(std::size_t rows, std::size_t columns, std::vector<double>& data);
    Matrix(std::size_t rows, std::size_t columns, std::vector<double>&& data);
    Matrix(const Matrix& m);
    Matrix(Matrix&& m);
    Matrix& operator=(const Matrix&);
    Matrix& operator=(Matrix&&) noexcept;

#if __cplusplus >= 202302L
    constexpr double operator[](std::size_t row, std::size_t column) const;

    double& operator[](std::size_t row, std::size_t column);

    void print();
#endif

    Matrix operator+(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    
    Matrix operator-(const Matrix& other) const;
    Matrix& operator-=(const Matrix& other);

    Matrix operator*(double scalar) const;
    friend Matrix operator*(double scalar, const Matrix& M);
    Matrix& operator*=(double scalar);

    Matrix operator*(const Matrix& other) const;
    Matrix& operator*=(const Matrix& other);

    constexpr double operator()(std::size_t row, std::size_t column) const;

    double& operator()(std::size_t row, std::size_t column);

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

    static Matrix get_identity(std::size_t n);
    static Matrix get_zero_matrix(std::size_t n);

    constexpr void clear() noexcept;

    template<OPERATION_TYPE TYPE_OP>
    double det() const;

    template<OPERATION_TYPE TYPE_OP>
    Matrix solve(const Matrix& b) const;

    template<OPERATION_TYPE TYPE_OP>
    Matrix inverse() const;

    std::pair<Matrix, Matrix> get_QR() const;

    Matrix get_eigenvalue() const;
};

