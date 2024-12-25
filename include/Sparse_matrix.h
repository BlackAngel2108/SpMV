#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <omp.h>

class Sparse_matrix {
protected:
    int rows, cols;
    size_t size;

public:
    virtual std::vector<double> SpMV(std::vector<double>& vec) = 0;
    int get_cols() { return cols; }
    int get_size() { return size; }
    int get_rows() { return rows; }
};

class COO_matrix : public Sparse_matrix {
private:
    std::vector<double> data;
    std::vector<int> rows_id;
    std::vector<int> colums_id;

public:
    COO_matrix(std::string filename);
    std::vector<double> SpMV(std::vector<double>& x) override;
    std::vector<double> get_values() const { return data; }
    std::vector<int> get_rows_id() const { return rows_id; }
    std::vector<int> get_cols_id() const { return colums_id; }
};

class CSR_matrix : public Sparse_matrix {
private:
    std::vector<double> values;
    std::vector<int> column_indices;
    std::vector<int> row_pointers;

public:
    CSR_matrix(std::string filename);
    std::vector<double> SpMV(std::vector<double>& vec) override;
};

class DIAG_matrix : public Sparse_matrix {
private:
    std::map<int, std::vector<std::pair<int, double>>> diagonals;

public:
    DIAG_matrix(std::string filename);
    std::vector<double> SpMV(std::vector<double>& vec) override;
};

class ELLPack_matrix : public Sparse_matrix {
private:
    std::vector<std::vector<double>> values;      // Массив значений ненулевых элементов
    std::vector<std::vector<int>> col_indices;    // Массив индексов столбцов
    int max_non_zero;                             // Максимальное количество ненулевых элементов в строке

public:
    ELLPack_matrix(std::string filename);
    std::vector<double> SpMV(std::vector<double>& x) override;
};

class SELL_C_matrix : public Sparse_matrix {
private:
    std::vector<std::vector<double>> values;      // Массив значений ненулевых элементов
    std::vector<std::vector<int>> col_indices;    // Массив индексов столбцов
    std::vector<int> row_pointers;                // Указатели начала сегменов
    int segment_size;                             // Размер сегмента
    int max_non_zero;                             // Максимальное количество ненулевых элементов в строке

public:
    SELL_C_matrix(std::string filename, int segment_size);
    std::vector<double> SpMV(std::vector<double>& x) override;
};

class SELL_C_sigma_matrix : public Sparse_matrix {
private:
    std::vector<std::vector<double>> values;      // Массив значений ненулевых элементов
    std::vector<std::vector<int>> col_indices;    // Массив индексов столбцов
    std::vector<int> row_pointers;                // Указатели начала строк
    int segment_size;                             // Размер сегмента
    int max_non_zero;                             // Максимальное количество ненулевых элементов в строке
    int sigma;                                    // Количество ненулевых элементов на строку в сегменте

public:
    SELL_C_sigma_matrix(std::string filename, int segment_size, int sigma);
    std::vector<double> SpMV(std::vector<double>& x) override;
};