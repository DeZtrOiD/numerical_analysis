
#include <iostream>
#include "matrix.h"

int main() {


    std::cout << "======== Task 1: Gaussian Elimination ========" << std::endl;
    std::cout << "==============================================" << std::endl;
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
    std::cout << "==============================================" << std::endl;
    std::cout << var26.solve<Matrix::GAUSS>(var26_b) << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << var26.inverse<Matrix::GAUSS>() << std::endl;
    std::cout << std::endl;


    std::cout << "========== Task 2: UL Decomposition ==========" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << var26.det<Matrix::LU>() << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << var26.solve<Matrix::LU>(var26_b) << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << var26.inverse<Matrix::LU>() << std::endl;

    // std::cout << var26 * var26.inverse<Matrix::GAUSS>();
    // std::cout << "==============================================" << std::endl;
    // std::cout << var26 * var26.inverse<Matrix::LU>();
    // std::cout << "==============================================" << std::endl;
    std::cout << std::endl;

    std::cout << "========== Task 3: Thomas Algorithm ==========" << std::endl;
    std::cout << "==============================================" << std::endl;

    Matrix thom_A_26{
        8, 8, {
            9.0, -7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            3.0,  7.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,  8.0, 11.0, -3.0, 0.0, 0.0, 0.0, 0.0,
            0.0,  0.0, -3.0,  7.0, -4.0, 0.0, 0.0, 0.0,
            0.0,  0.0, 0.0,  3.0, 8.0, -5.0, 0.0, 0.0,
            0.0,  0.0, 0.0,  0.0, -5.0, 9.0, 4.0, 0.0,
            0.0,  0.0, 0.0,  0.0, 0.0, 3.0, -10.0, 5.0,
            0.0,  0.0, 0.0,  0.0, 0.0, 0.0, 5.0, 9.0
        }
    };

    Matrix thom_b_26{
        8, 1, {
            -26.0, 40.0, 17.0, 19.0, 61.0, -36.0, -70.0, 39.0
        }
    };

    std::cout <<"Thomas: " << thom_A_26.det<Matrix::THOMAS>() << std::endl;
    std::cout <<"Gauss: " << thom_A_26.det<Matrix::GAUSS>() << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << thom_A_26.solve<Matrix::THOMAS>(thom_b_26) << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << thom_A_26.solve<Matrix::GAUSS>(thom_b_26) << std::endl;
    std::cout << std::endl;


    std::cout << "========== Task 4: Simple Iterations =========" << std::endl;
    std::cout << "==============================================" << std::endl;
    Matrix jac_A_26{
        4, 4, {
            3.0,  16.0, -5.0,  -7.0,
            -3.0, -7.0,   4.0,  16.0,
            6.0, -5.0,  14.0,  -2.0,
            17.0, -4.0,   3.0,  -1.0
        }
    };
    Matrix jac_b_26{
        4, 1, {
            46.0,
            80.0,
            96.0,
            56.0
        }
    };
    std::cout << jac_A_26.solve<Matrix::JACOBI>(jac_b_26) << std::endl;

    std::cout << "============ Task 5: Seidel Method ===========" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << jac_A_26.solve<Matrix::SEIDEL>(jac_b_26) << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "Gauss:\n" << jac_A_26.solve<Matrix::GAUSS>(jac_b_26) << std::endl;
    std::cout << std::endl;


    std::cout << "============ Task 6: Eigenvalues =============" << std::endl;
    std::cout << "==============================================" << std::endl;
    Matrix QR_var26{
        5, 5, {
        1.0,  1.0, -4.0,  7.0,  3.0,
        4.0,  3.0,  1.0,  2.0,  2.0,
        -2.0,  7.0, -11.0, 6.0,  5.0,
        2.0,  5.0, -4.0,  1.0,  3.0,
        18.0,  3.0, -5.0,  1.0, -2.0
        }
        // 3, 3, {
        //     2, 1, 
        // }
    };
    std::cout << QR_var26.get_eigenvalue() << std::endl;
    std::cout << "==============================================" << std::endl;
    auto [Q, R] = QR_var26.get_QR();
    std::cout << "========= QQQQQQQQQQQQQQQQQQQQQQQQQQ =========" << std::endl;
    std::cout << Q << std::endl;
    std::cout << "========= RRRRRRRRRRRRRRRRRRRRRRRRRR =========" << std::endl;
    std::cout << R << std::endl;


    return 0;
}

