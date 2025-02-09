#include <gtest.h>
#include "Sparse_matrix.h"
#include <stdio.h> 
#include <time.h> 


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
TEST(Lpack_Sparse_matrix, test_can_multiply_LPack_time) {
    std::string path("../../bin_matrix/S80PI_n1.bin");
    LPack_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    double startTime, endTime;
    startTime = clock();
    b = mat.SpMV(b);
    endTime = clock();
    double seconds = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    ASSERT_NO_THROW(printf("The time: %f seconds\n", seconds););
}
TEST(Lpack_Sparse_matrix, test_can_multiply_LPack_time) {
    std::string path("../../bin_matrix/atmosmodm.bin");
    LPack_matrix mat(path);
    int size = mat.get_cols();
    std::vector<double> b(size, 1);
    std::vector<double> ans(size);
    double startTime, endTime;
    startTime = clock();
    b = mat.SpMV(b);
    endTime = clock();
    double seconds = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    ASSERT_NO_THROW(printf("The time: %f seconds\n", seconds););
}