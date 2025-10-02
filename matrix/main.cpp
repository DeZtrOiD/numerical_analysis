
#include <iostream>
#include "matrix.h"

int main() {

    Matrix var26{5, 5 , {
        3.0, -5.0, -4.0, 7.0, -11.0,
        4.0, 6.0, 9.0, 3.0, 2.0,
        -2.0, 3.0, 7.0, 5.0, 8.0,
        6.0, 5.0, 4.0, 11.0, 3.0,
        3.0, 7.0, -5.0, 6.0, -2.0
    }};
    Matrix var26_b{5, 1, {
        -23.0, -26.0, 43.0, 80.0, 27.0
    }};
    std::cout << var26 << std::endl;
    std::cout << var26.det<Matrix::GAUSS>() << std::endl;
    std::cout << var26.det<Matrix::LU>() << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << var26.solve<Matrix::GAUSS>(var26_b) << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << var26.solve<Matrix::LU>(var26_b) << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << var26.inverse<Matrix::GAUSS>() << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << var26.inverse<Matrix::LU>() << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << var26 * var26.inverse<Matrix::GAUSS>();
    std::cout << "==================================" << std::endl;
    std::cout << var26 * var26.inverse<Matrix::LU>();
    std::cout << "========================================" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}

