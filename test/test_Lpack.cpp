#include <gtest.h>
#include "Sparse_matrix.h"
#include <stdio.h> 
#include <time.h> 
#include <chrono>
#include <string>

double time_Ellpack(std::string path){
    ELLPack_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    auto start = std::chrono::high_resolution_clock::now();
    b = mat.SpMV(b);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

double time_sell_c(std::string path) {
    SELL_C_matrix mat(path, 32);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    auto start = std::chrono::high_resolution_clock::now();
    b = mat.SpMV(b);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

double time_sell_c_sigma(std::string path) {
    SELL_C_sigma_matrix mat(path, 16, 1024);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    auto start = std::chrono::high_resolution_clock::now();
    b = mat.SpMV(b);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

double time_CSR(std::string path) {
    CSR_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    auto start = std::chrono::high_resolution_clock::now();
    b = mat.SpMV(b);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

double time_COO(std::string path) {
    COO_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    auto start = std::chrono::high_resolution_clock::now();
    b = mat.SpMV(b);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

double time_DIAG(std::string path) {
    DIAG_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    auto start = std::chrono::high_resolution_clock::now();
    b = mat.SpMV(b);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

//TEST(Lpack_Sparse_matrix, test_can_read_bin_LPack) {
//
//    std::string path2("../../bin_matrix/ash958.bin");
//    ASSERT_NO_THROW(LPack_matrix mat(path2));
//}
//TEST(Lpack_Sparse_matrix, test_can_multiply_LPack) {
//    std::string path("../../bin_matrix/ash958.bin");
//    LPack_matrix mat(path);
//    int size = mat.get_cols();
//    std::vector<double> b(size, 1);
//    std::vector<double> ans(size);
//    ASSERT_NO_THROW(b = mat.SpMV(b));
//}
void test_all(std::string path){
    double t1 = time_DIAG(path);
    double t2 = time_COO(path);
    double t3 = time_CSR(path);
    double t4 = time_sell_c_sigma(path);
    double t5 = time_sell_c(path);
    double t6 = time_Ellpack(path);
    printf("The time COO: %f seconds\n", t2);
    printf("The time DIAG: %f seconds\n", t1);
    printf("The time CSR: %f seconds\n", t3);
    printf("The time ELLPack: %f seconds\n", t6);
    printf("The time SELL_C: %f seconds\n", t5);
    printf("The time SELL_C_Sigma: %f seconds\n", t4);
}
TEST(TIME, test_StocF_1465) {
    std::string path("../../bin_matrix/StocF-1465.bin");
    test_all(path);
    ASSERT_NO_THROW(std::cout << path << "\n";);
}
/*
TEST(TIME, test_rajat31) {
    std::string path("../../bin_matrix/rajat31.bin");
    test_all(path);
    ASSERT_NO_THROW(std::cout << path << "\n";);
}*/

TEST(TIME, test_road_usa) {
    std::string path("../../bin_matrix/road_usa.bin");
    test_all(path);
    ASSERT_NO_THROW(std::cout << path << "\n";);
}
// TEST(Lpack_Sparse_matrix, test_can_multiply_LPack_time_big) {
//     std::string path("../../bin_matrix/atmosmodm.bin");
//     LPack_matrix mat(path);
//     int size = mat.get_cols();
//     std::vector<double> b(size, 1);
//     std::vector<double> ans(size);
//     double startTime, endTime;
//     startTime = clock();s
//     b = mat.SpMV(b);
//     endTime = clock();
//     double seconds = (double)(endTime - startTime) / CLOCKS_PER_SEC;
//     ASSERT_NO_THROW(printf("The time: %f seconds\n", seconds););
// }