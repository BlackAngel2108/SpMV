#include "Sparse_matrix.h"

std::vector<double> CSR_matrix::SpMV(std::vector<double>& x)
{
    int n = indptr.size() - 1; // Количество строк в матрице
    std::vector<double> y(n, 0.0); // Результат умножения

    for (int i = 0; i < n; i++) {
        for (int j = indptr[i]; j < indptr[i + 1]; ++j) {
            y[i] += data[j] * x[indices[j]];
        }
    }

    return y;
}
//// Пример разреженной матрицы в формате CSR
//std::vector<double> values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
//std::vector<int> col_indices = { 0, 1, 2, 1, 2 };
//std::vector<int> row_ptr = { 0, 2, 3, 5 };
//
//// Пример вектора x
//std::vector<double> x = { 1.0, 2.0, 3.0 };

std::vector<double> COO_matrix::SpMV(std::vector<double>& x)
{
    int n = x.size();
    std::vector<double> y(n, 0.0); // Результат умножения
    int m = data.size(); // Количество элементов в матрице A
    for (int i = 0; i < m; ++i) {
        y[rows[i]] += data[i] * x[colums[i]];
    }
    return y;
}

std::vector<double> DIA_matrix::SpMV(std::vector<double>& x)
{
    int n = x.size();
    std::vector<double> y(n, 0.0);
    int m = diagonals.size();
    for (int k = 0; k < m ; k++) {
        int offset = offsets[k];
        for (int i = 0; i < n; i++) {
            int j = i + offset;
            if (j >= 0 && j < n) {
                y[i] += diagonals[k][i] * x[j];
            }
        }
    }

    return y;
}
//// Пример разреженной матрицы в формате DIA
//std::vector<std::vector<double>> diagonals = {
//    {1.0, 2.0, 3.0}, // Главная диагональ
//    {0.0, 4.0, 5.0 }, // Диагональ смещения -1
//    {7.0, 8.0, 0.0}  // Диагональ смещения 1
//};
//std::vector<int> offsets = { 0, -1, 1 };

std::vector<double> ELL_matrix::SpMV(std::vector<double>& x)
{
    int n = row_lengths.size(); // Количество строк в матрице
    std::vector<double> y(n, 0.0); // Результат умножения

    int k = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < row_lengths[i]; ++j) {
            y[i] += data[k] * x[col_indices[k]];
            ++k;
        }
    }

    return y;
}
//// Пример разреженной матрицы в формате ELL
//std::vector<double> values = { 1.0, 7.0, 4.0, 2.0, 5.0, 3.0 };
//std::vector<int> col_indices = { 0, 2, 0, 1, 1, 2 };
//std::vector<int> row_lengths = { 2, 2, 2 };