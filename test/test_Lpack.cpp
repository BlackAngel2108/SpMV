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

    double t1 = 0.0;
    double t6 = 0.0;
    try {
	t1 = time_DIAG(path);
 	printf("The time DIAG: %f seconds\n", t1);
    }
    catch(...){
	std::cout<<"Diag error"<<std::endl;
    }
    double t2 = time_COO(path);
    printf("The time COO: %f seconds\n", t2);
    double t3 = time_CSR(path);
    printf("The time CSR: %f seconds\n", t3);
    double t4 = time_sell_c_sigma(path);
    printf("The time SELL_C_Sigma: %f seconds\n", t4);
    double t5 = time_sell_c(path);
    printf("The time SELL_C: %f seconds\n", t5);
    try {
	t6 = time_Ellpack(path);
	printf("The time ELLPack: %f seconds\n", t6);
    }
    catch(...){
	std::cout<<"Ellpack erro"<<std::endl;
    }
    ASSERT_NO_THROW(std::cout << path << "\n";);
}
TEST(TIME, test_road_usa) {
    std::string path("../../bin_matrix/road_usa.bin");
    test_all(path);
    ASSERT_NO_THROW(std::cout << path << "\n";);
}

TEST(TIME, test_kmer_V2a) {
    std::string path("../../bin_matrix/kmer_V2a.bin");
    test_all(path);
    ASSERT_NO_THROW(std::cout << path << "\n";);
}*/

// TEST(TIME, test_nlpkkt240) {
//     std::string path("../../bin_matrix/nlpkkt240.bin");
//     test_all(path);
//     ASSERT_NO_THROW(std::cout << path << "\n";);
// }

double time_MKL_COO(const std::string& filename) {
    std::cout << "[INFO] Загружаем матрицу из файла: " << filename << std::endl;

    COO_matrix coo(filename);

    int rows = coo.get_rows();
    int cols = coo.get_cols();
    int nnz = coo.get_size();

    std::cout << "[DEBUG] Размеры матрицы: rows = " << rows << ", cols = " << cols << ", nnz = " << nnz << std::endl;

    // Проверка валидности
    if (rows <= 0 || cols <= 0 || nnz <= 0) {
        std::cerr << "[ERROR] Некорректные размеры матрицы." << std::endl;
        return -1.0;
    }

    std::vector<int> rows_int = coo.get_rows_id();
    std::vector<int> cols_int = coo.get_cols_id();
    std::vector<double> vals = coo.get_values();

    if (rows_int.size() != nnz || cols_int.size() != nnz || vals.size() != nnz) {
        std::cerr << "[ERROR] Размерность массивов не совпадает с nnz." << std::endl;
        return -1.0;
    }

    std::cout << "[INFO] Преобразуем int → MKL_INT..." << std::endl;
    std::vector<MKL_INT> rowInd(nnz);
    std::vector<MKL_INT> colInd(nnz);

    for (int i = 0; i < nnz; ++i) {
        rowInd[i] = static_cast<MKL_INT>(rows_int[i]);
        colInd[i] = static_cast<MKL_INT>(cols_int[i]);
    }

    std::vector<double> x(cols, 1.0);
    std::vector<double> y(rows, 0.0);

    sparse_matrix_t A;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    std::cout << "[INFO] Создаём MKL COO матрицу..." << std::endl;
    sparse_status_t status;
    for (int i = 0; i < nnz; ++i) {
        if (rowInd[i] >= rows || colInd[i] >= cols || rowInd[i] < 0 || colInd[i] < 0) {
            std::cerr << "[ERROR] Неверные индексы в COO: i = " << i
                      << ", row = " << rowInd[i]
                      << ", col = " << colInd[i]
                      << ", rows = " << rows
                      << ", cols = " << cols << std::endl;
            return -1.0;
        }
    }
    status = mkl_sparse_d_create_coo(&A, SPARSE_INDEX_BASE_ZERO, rows, cols, nnz,
                                      rowInd.data(), colInd.data(), vals.data());

    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "[ERROR] Ошибка при создании COO матрицы: код " << status << std::endl;
        return -1.0;
    }

    std::cout << "[INFO] Выполняем умножение матрицы на вектор..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x.data(), 0.0, y.data());

    auto end = std::chrono::high_resolution_clock::now();

    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "[ERROR] Ошибка при выполнении умножения: код " << status << std::endl;
        mkl_sparse_destroy(A);
        return -1.0;
    }

    std::cout << "[INFO] Успешное умножение. Освобождаем ресурсы..." << std::endl;
    mkl_sparse_destroy(A);

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "[INFO] Время выполнения: " << elapsed.count() << " секунд." << std::endl;

    return elapsed.count();
}


// TEST(TIME, test_mkl_nlpkkt240) {
//     std::string path("../../bin_matrix/test.bin");
//     double t = time_MKL_COO(path);
//     printf("The time MKL_COO: %f seconds\n", t);
//     ASSERT_NO_THROW(std::cout << path << "\n";);
// }

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
