#include <gtest.h>
#include "Sparse_matrix.h"
#include <stdio.h> 
#include <time.h> 

TEST(Sparse_matrix, can_read_bin_coo) {
    std::string path2("../../bin_matrix/ash958.bin");
    ASSERT_NO_THROW(COO_matrix mat(path2));
}
TEST(Sparse_matrix, can_read_bin_coo_right_size) {
    std::string path2("../../bin_matrix/ash958.bin");
    COO_matrix mat(path2);
    int size= mat.get_size();
    int EXans = 1916;
    EXPECT_EQ(size, EXans);
}
TEST(Sparse_matrix, can_multiply_COO) {
    const std::string path("../../bin_matrix/ash958.bin");
    COO_matrix mat(path);
    int size=mat.get_cols();
    std::vector<double> b(size,1);
    std::vector<double> ans(size);
    ASSERT_NO_THROW(b = mat.SpMV(b));
}
TEST(Sparse_matrix, can_multiply_COO_time) {
    const std::string path("../../bin_matrix/ash958.bin");
    COO_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    double startTime, endTime;
    startTime = clock();
    for (int i = 0; i < 1000; i++) {
        b = mat.SpMV(b);
    }
    b = mat.SpMV(b);
    endTime = clock();
    double seconds = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    ASSERT_NO_THROW(printf("The time: %f seconds\n", seconds););
}
///////

TEST(Sparse_matrix, can_read_bin_diag) {
    std::string path("../../bin_matrix/ash958.bin");
    std::string path2("../../bin_matrix/ash958_diag.bin");
    ASSERT_NO_THROW(DIAG_matrix mat(path));
}
TEST(Sparse_matrix, can_multiply_diag) {
    std::string path("../../bin_matrix/ash958.bin");
    const std::string path2("C:/SpMV/SpMV/bin_matrix/ash958_diag.bin");
    DIAG_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    ASSERT_NO_THROW(b = mat.SpMV(b));
}
TEST(Sparse_matrix, can_multiply_diag_time) {
    std::string path("../../bin_matrix/ash958.bin");
    const std::string path2("C:/SpMV/SpMV/bin_matrix/ash958_diag.bin");
    DIAG_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    double startTime, endTime;
    startTime = clock();
    for (int i = 0; i < 1000; i++) {
        b = mat.SpMV(b);
    }
    b = mat.SpMV(b);
    endTime = clock();
    double seconds = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    ASSERT_NO_THROW(printf("The time: %f seconds\n", seconds););
}
/////

TEST(Sparse_matrix, can_read_bin_csr) {

    std::string path("../../bin_matrix/ash958.bin");
    std::string path2("../../bin_matrix/ash958_csr.bin");
    ASSERT_NO_THROW(CSR_matrix mat(path));
}
TEST(Sparse_matrix, can_multiply_csr) {
    std::string path("../../bin_matrix/ash958.bin");
    const std::string path2("C:/SpMV/SpMV/bin_matrix/ash958_csr.bin");
    CSR_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    ASSERT_NO_THROW(b = mat.SpMV(b));
}
TEST(Sparse_matrix, can_multiply_csr_time) {
    std::string path("../../bin_matrix/ash958.bin");
    const std::string path2("C:/SpMV/SpMV/bin_matrix/ash958_csr.bin");
    CSR_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    double startTime, endTime;
    startTime = omp_get_wtime();
    for (int i = 0; i < 1000; i++) {
        b = mat.SpMV(b);
    }
    b = mat.SpMV(b);
    endTime = omp_get_wtime();
    double seconds = (double)(endTime - startTime);// / CLOCKS_PER_SEC;
    ASSERT_NO_THROW(printf("The time: %f seconds\n", seconds););
}
///


TEST(Sparse_matrix, can_read_bin_LPack) {

    std::string path2("../../bin_matrix/ash958.bin");
    ASSERT_NO_THROW(LPack_matrix mat(path2));
}
TEST(Sparse_matrix, can_multiply_LPack) {
    std::string path("../../bin_matrix/ash958.bin");
    LPack_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    ASSERT_NO_THROW(b = mat.SpMV(b));
}
TEST(Sparse_matrix, can_multiply_LPack_time) {
    std::string path("../../bin_matrix/ash958.bin");
    const std::string path2("C:/SpMV/SpMV/bin_matrix/ash958_lpack.bin");
    LPack_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    double startTime, endTime;
    startTime = clock();
    for (int i = 0; i < 1000; i++) {
        b = mat.SpMV(b);
    }
    b = mat.SpMV(b);
    endTime = clock();
    double seconds = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    ASSERT_NO_THROW(printf("The time: %f seconds\n", seconds););
}


TEST(Sparse_matrix, can_read_bin_SELL_C) {

    std::string path("../../bin_matrix/ash958.bin");
    std::string path2("../../bin_matrix/ash958_lpack.bin");
    ASSERT_NO_THROW(SELL_C_matrix mat(path,4));
}
TEST(Sparse_matrix, can_multiply_SELL_C) {
    std::string path("../../bin_matrix/ash958.bin");
    const std::string path2("C:/SpMV/SpMV/bin_matrix/ash958_lpack.bin");
    SELL_C_matrix mat(path,4);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    ASSERT_NO_THROW(b = mat.SpMV(b));
}
TEST(Sparse_matrix, can_multiply_SELL_C_time) {
    std::string path("../../bin_matrix/ash958.bin");
    const std::string path2("C:/SpMV/SpMV/bin_matrix/ash958_lpack.bin");
    SELL_C_matrix mat(path,4);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    double startTime, endTime;
    startTime = clock();
    for (int i = 0; i < 1000; i++) {
        b = mat.SpMV(b);
    }
    b = mat.SpMV(b);
    endTime = clock();
    double seconds = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    ASSERT_NO_THROW(printf("The time: %f seconds\n", seconds););
}


TEST(Sparse_matrix, can_read_bin_SELL_C_sigma) {

    std::string path("../../bin_matrix/ash958.bin");
    ASSERT_NO_THROW(SELL_C_sigma_matrix mat(path, 4,2));
}
TEST(Sparse_matrix, can_multiply_SELL_C_sigma) {
    std::string path("../../bin_matrix/ash958.bin");
    SELL_C_sigma_matrix mat(path, 4,2);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    ASSERT_NO_THROW(b = mat.SpMV(b));
}
TEST(Sparse_matrix, can_multiply_SELL_C_sigma_time) {
    std::string path("../../bin_matrix/ash958.bin");
    SELL_C_sigma_matrix mat(path, 4,2);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    double startTime, endTime;
    startTime = clock();
    for (int i = 0; i < 1000; i++) {
        b = mat.SpMV(b);
    }
    b = mat.SpMV(b);
    endTime = clock();
    double seconds = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    ASSERT_NO_THROW(printf("The time: %f seconds\n", seconds););
}


TEST(Sparse_matrix, right_mult_coo_diag) {
    const std::string path("../../bin_matrix/ash958.bin");
    COO_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    double startTime, endTime;
    b = mat.SpMV(b);
    std::string path2("../../bin_matrix/ash958.bin");
    const std::string path3("../../bin_matrix/ash958.bin");
    DIAG_matrix mat2(path2);
    int size2 = mat2.get_cols();
    std::vector<double> b2(size2, 1);
    b2 = mat2.SpMV(b2);
    int flag=0;
    for (int i = 0; i < size2; i++) {
        if(b[i]!=b2[i])
            flag=1;
    }
    EXPECT_EQ(flag, 0);
}
TEST(Sparse_matrix, right_mult_coo_scr) {
    const std::string path("../../bin_matrix/ash958.bin");
    COO_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    double startTime, endTime;
    b = mat.SpMV(b);
    std::string path2("../../bin_matrix/ash958.bin");
    const std::string path3("../../bin_matrix/ash958.bin");
    CSR_matrix mat2(path2);
    int size2 = mat2.get_cols();
    std::vector<double> b2(size2, 1);
    b2 = mat.SpMV(b2);
    int flag=0;
    for (int i = 0; i < size2; i++) {
        if (b[i] != b2[i])
            flag = 1;
    }
    EXPECT_EQ(flag, 0);
}
TEST(Sparse_matrix, right_mult_coo_lpack) {
    const std::string path("../../bin_matrix/ash958.bin");
    COO_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    double startTime, endTime;
    b = mat.SpMV(b);
    std::string path2("../../bin_matrix/ash958.bin");
    const std::string path3("../../bin_matrix/ash958.bin");
    LPack_matrix mat2(path2);
    int size2 = mat2.get_cols();
    std::vector<double> b2(size2, 1);
    b2 = mat2.SpMV(b2);
    int flag = 0;
    for (int i = 0; i < size2; i++) {
        if (b[i] != b2[i])
            flag = 1;
    }
    EXPECT_EQ(flag, 0);
}

TEST(Sparse_matrix, right_mult_coo_SELL_C) {
    const std::string path("../../bin_matrix/ash958.bin");
    COO_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    double startTime, endTime;
    b = mat.SpMV(b);
    std::string path2("../../bin_matrix/ash958.bin");
    SELL_C_matrix mat2(path2,4);
    int size2 = mat2.get_cols();
    std::vector<double> b2(size2, 1);
    b2 = mat2.SpMV(b2);
    int flag = 0;
    for (int i = 0; i < size2; i++) {
        if (b[i] != b2[i])
            flag = 1;
    }
    EXPECT_EQ(flag, 0);
}

TEST(Sparse_matrix, right_mult_coo_SELL_C_sigma) {
    const std::string path("../../bin_matrix/ash958.bin");
    COO_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    double startTime, endTime;
    b = mat.SpMV(b);
    std::string path2("../../bin_matrix/ash958.bin");
    SELL_C_sigma_matrix mat2(path2, 2,4);
    int size2 = mat2.get_cols();
    std::vector<double> b2(size2, 1);
    b2 = mat2.SpMV(b2);
    int flag = 0;
    for (int i = 0; i < size2; i++) {
        if (b[i] != b2[i])
            flag = 1;
    }
    EXPECT_EQ(flag, 0);
}