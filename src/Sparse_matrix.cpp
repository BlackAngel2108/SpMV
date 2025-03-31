#include "Sparse_matrix.h"

COO_matrix::COO_matrix(std::string filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }

    infile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    infile.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    infile.read(reinterpret_cast<char*>(&size), sizeof(size));
    //std::cout<<size<<std::endl;

    rows_id.resize(size);
    colums_id.resize(size);
    data.resize(size);

    for (size_t i = 0; i < size; ++i) {
        int row_id, col_id;
        double value;

        {
            infile.read(reinterpret_cast<char*>(&row_id), sizeof(int));
            infile.read(reinterpret_cast<char*>(&col_id), sizeof(int));
            infile.read(reinterpret_cast<char*>(&value), sizeof(double));
        }

        rows_id[i] = row_id;
        colums_id[i] = col_id;
        data[i] = value;
    }

    infile.close();
}

std::vector<double> COO_matrix::SpMV(const std::vector<double>& x) {
    std::vector<double> y(rows, 0.0);
//#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        int row = rows_id[i];
        double value = data[i];
        int col = colums_id[i];
//#pragma omp atomic
        y[row] += value * x[col];
    }
    return y;
}

CSR_matrix::CSR_matrix(std::string filename) {
    COO_matrix cooMatrix(filename);
    rows = cooMatrix.get_rows();
    cols = cooMatrix.get_cols();
    size = cooMatrix.get_size();

    std::vector<double> coo_values = cooMatrix.get_values();
    std::vector<int> coo_rows = cooMatrix.get_rows_id();
    std::vector<int> coo_cols = cooMatrix.get_cols_id();

    row_pointers.resize(rows + 1, 0);

    // Count the number of non-zero elements in each row
    for (size_t i = 0; i < size; ++i) {
        row_pointers[coo_rows[i] + 1]++;
    }

    for (int i = 1; i <= rows; ++i) {
        row_pointers[i] += row_pointers[i - 1];
    }

    values.resize(size);
    column_indices.resize(size);

    std::vector<int> currentIndex(rows, 0);
    for (size_t i = 0; i < size; ++i) {
        int row = coo_rows[i];
        int index = row_pointers[row] + currentIndex[row];
        values[index] = coo_values[i];
        column_indices[index] = coo_cols[i];
        currentIndex[row]++;
    }
}

std::vector<double> CSR_matrix::SpMV(const std::vector<double>& vec) {
    std::vector<double> result(rows, 0.0);
#ifdef simple
#ifdef omp
#pragma omp parallel for
#endif
    for (int row = 0; row < rows; ++row) {
        double tem_for_row = 0.0;
        for (int i = row_pointers[row]; i < row_pointers[row + 1]; ++i) {
            int col = column_indices[i];
            tem_for_row += values[i] * vec[col];
        }
        result[row] = tem_for_row;
    }
    return result;
}
#endif
#ifdef avx512
    #ifdef omp
    #pragma omp parallel for
    #endif
    for (int row = 0; row < rows; ++row) {
        __m512d local_sum = _mm512_setzero_pd();
        int row_start = row_pointers[row];
        int row_end = row_pointers[row + 1];
        int i = row_start;

        // Обрабатываем по 8 элементов за раз
        for (; i + 7 < row_end; i += 8) {
            // Загружаем 8 индексов столбцов
            __m256i col_idx = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&column_indices[i]));

            // Загружаем 8 значений матрицы
            __m512d mat_vals = _mm512_loadu_pd(&values[i]);

            // Собираем 8 значений из вектора vec
            __m512d vec_vals = _mm512_set_pd(
                vec[_mm256_extract_epi32(col_idx, 7)],
                vec[_mm256_extract_epi32(col_idx, 6)],
                vec[_mm256_extract_epi32(col_idx, 5)],
                vec[_mm256_extract_epi32(col_idx, 4)],
                vec[_mm256_extract_epi32(col_idx, 3)],
                vec[_mm256_extract_epi32(col_idx, 2)],
                vec[_mm256_extract_epi32(col_idx, 1)],
                vec[_mm256_extract_epi32(col_idx, 0)]
            );

            // Умножаем и добавляем к аккумулятору
            local_sum = _mm512_fmadd_pd(mat_vals, vec_vals, local_sum);
        }

        // Суммируем накопленные значения
        double vectorized_sum = _mm512_reduce_add_pd(local_sum);
        double scalar_sum = 0.0;

        // Обрабатываем оставшиеся элементы (1-7)
        for (; i < row_end; ++i) {
            scalar_sum += values[i] * vec[column_indices[i]];
        }

        result[row] = vectorized_sum + scalar_sum;
    }
    return result;
    }
#endif

DIAG_matrix::DIAG_matrix(std::string filename) {
    COO_matrix cooMatrix(filename);
    rows = cooMatrix.get_rows();
    cols = cooMatrix.get_cols();
    size = cooMatrix.get_size();

    std::vector<double> coo_values = cooMatrix.get_values();
    std::vector<int> coo_rows = cooMatrix.get_rows_id();
    std::vector<int> coo_cols = cooMatrix.get_cols_id();

    for (size_t i = 0; i < size; ++i) {
        int diagIndex = coo_cols[i] - coo_rows[i];
        diagonals[diagIndex].push_back({ coo_rows[i], coo_values[i] });
    }
}

std::vector<double> DIAG_matrix::SpMV(const std::vector<double>& vec) {
    std::vector<double> result(rows, 0.0);
    //#pragma omp parallel for
    for (auto& diag : diagonals) {
        int diagIndex = diag.first;
        for (auto& elem : diag.second) {
            int row = elem.first;
            double value = elem.second;
            int col = row + diagIndex;
            if (col >= 0 && col < cols) {
                //#pragma omp atomic
                result[row] += value * vec[col];
            }
        }
    }
    return result;
}

ELLPack_matrix::ELLPack_matrix(std::string filename) {
    COO_matrix cooMatrix(filename);
    rows = cooMatrix.get_rows();
    cols = cooMatrix.get_cols();
    size = cooMatrix.get_size();

    std::vector<double> coo_values = cooMatrix.get_values();
    std::vector<int> coo_rows = cooMatrix.get_rows_id();
    std::vector<int> coo_cols = cooMatrix.get_cols_id();

    // Count the number of non-zero elements in each row
    std::vector<int> row_counts(rows, 0);

    for (size_t i = 0; i < size; ++i) {
        row_counts[coo_rows[i]]++;
    }
    max_non_zero = *std::max_element(row_counts.begin(), row_counts.end());

    values.resize(rows, std::vector<double>(max_non_zero, 0.0));
    col_indices.resize(rows, std::vector<int>(max_non_zero, -1));

    std::vector<int> current_index(rows, 0);
    for (size_t i = 0; i < size; ++i) {
        int row = coo_rows[i];
        int col = coo_cols[i];
        double value = coo_values[i];

        values[row][current_index[row]] = value;
        col_indices[row][current_index[row]] = col;
        current_index[row]++;
    }
}

std::vector<double> ELLPack_matrix::SpMV(const std::vector<double>& x) {
#ifdef simple    
    std::vector<double> result(rows, 0.0);
  //omp_set_num_threads(4);
#ifdef omp 
#pragma omp parallel for schedule(dynamic,1000)
#endif
    for (int row = 0; row < rows; ++row) {
    double local_sum =0;
    int idx;
    for (int i = 0; i < max_non_zero; ++i){
        idx = col_indices[row][i];
            if ( idx == -1) {
                continue;
            }
            local_sum += values[row][i] * x[idx];
        }
    result[row]+= local_sum;
    }
    return result;
#endif
#ifdef avx2
        std::vector<double> result(rows, 0.0);
#ifdef omp
#pragma omp parallel for schedule(dynamic,1000)
#endif
        for (int row = 0; row < rows; ++row) {
            __m256d local_sum = _mm256_setzero_pd();  // Инициализируем аккумулятор нулями

            int i = 0;
            // Обрабатываем по 4 элемента за раз (AVX2 работает с 256-битными регистрами, 4 double)
            for (; i + 3 < max_non_zero; i += 4) {
                // Загружаем 4 индекса столбцов
                __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&col_indices[row][i]));

                // Проверяем, есть ли -1 в индексах
                __m128i mask = _mm_cmpeq_epi32(idx, _mm_set1_epi32(-1));
                if (_mm_movemask_epi8(mask) != 0) {
                    // Если есть -1, обрабатываем оставшиеся элементы скалярно
                    for (int j = i; j < i + 4; ++j) {
                        if (col_indices[row][j] == -1) continue;
                        result[row] += values[row][j] * x[col_indices[row][j]];
                    }
                    continue;
                }

                // Загружаем 4 значения из матрицы
                __m256d vals = _mm256_loadu_pd(&values[row][i]);

                // Собираем 4 значения из вектора x
                __m256d x_vals = _mm256_set_pd(
                    x[col_indices[row][i + 3]],
                    x[col_indices[row][i + 2]],
                    x[col_indices[row][i + 1]],
                    x[col_indices[row][i]]
                );

                // Умножаем и добавляем к аккумулятору
                local_sum = _mm256_add_pd(local_sum, _mm256_mul_pd(vals, x_vals));
            }

            // Обрабатываем оставшиеся элементы скалярно
            for (; i < max_non_zero; ++i) {
                if (col_indices[row][i] == -1) continue;
                result[row] += values[row][i] * x[col_indices[row][i]];
            }

            // Суммируем аккумулятор и записываем результат
            double temp[4];
            _mm256_storeu_pd(temp, local_sum);
            result[row] += temp[0] + temp[1] + temp[2] + temp[3];
        }

        return result;
#endif
#ifdef avx512
        std::vector<double> result(rows, 0.0);
#ifdef omp
#pragma omp parallel for schedule(dynamic,1000)
#endif
        for (int row = 0; row < rows; ++row) {
            __m512d local_sum = _mm512_setzero_pd();  // Инициализируем аккумулятор нулями

            int i = 0;
            // Обрабатываем по 8 элементов за раз (AVX-512 работает с 512-битными регистрами, 8 double)
            for (; i + 7 < max_non_zero; i += 8) {
                // Загружаем 8 индексов столбцов
                __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_indices[row][i]));

                // Проверяем, есть ли -1 в индексах
                __mmask8 mask = _mm256_cmpeq_epi32_mask(idx, _mm256_set1_epi32(-1));
                if (mask != 0) {
                    // Если есть -1, обрабатываем оставшиеся элементы скалярно
                    for (int j = i; j < i + 8; ++j) {
                        if (col_indices[row][j] == -1) continue;
                        result[row] += values[row][j] * x[col_indices[row][j]];
                    }
                    continue;
                }

                // Загружаем 8 значений из матрицы
                __m512d vals = _mm512_loadu_pd(&values[row][i]);

                // Собираем 8 значений из вектора x
                __m512d x_vals = _mm512_set_pd(
                    x[col_indices[row][i + 7]],
                    x[col_indices[row][i + 6]],
                    x[col_indices[row][i + 5]],
                    x[col_indices[row][i + 4]],
                    x[col_indices[row][i + 3]],
                    x[col_indices[row][i + 2]],
                    x[col_indices[row][i + 1]],
                    x[col_indices[row][i]]
                );

                // Умножаем и добавляем к аккумулятору
                local_sum = _mm512_add_pd(local_sum, _mm512_mul_pd(vals, x_vals));
            }

            // Обрабатываем оставшиеся элементы скалярно
            for (; i < max_non_zero; ++i) {
                if (col_indices[row][i] == -1) continue;
                result[row] += values[row][i] * x[col_indices[row][i]];
            }

            // Суммируем аккумулятор и записываем результат
            double temp[8];
            _mm512_storeu_pd(temp, local_sum);
            result[row] += temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
        }

        return result;
#endif
}

SELL_C_matrix::SELL_C_matrix(std::string filename, int segment_size) : segment_size(segment_size) {
    COO_matrix cooMatrix(filename);
    rows = cooMatrix.get_rows();
    cols = cooMatrix.get_cols();
    size = cooMatrix.get_size();

    std::vector<double> coo_values = cooMatrix.get_values();
    std::vector<int> coo_rows = cooMatrix.get_rows_id();
    std::vector<int> coo_cols = cooMatrix.get_cols_id();

    // Count the number of non-zero elements in each row
    std::vector<int> row_counts(rows, 0);
    for (size_t i = 0; i < size; ++i) {
        row_counts[coo_rows[i]]++;
    }
    max_non_zero = *std::max_element(row_counts.begin(), row_counts.end());

    int num_segments = (rows + segment_size - 1) / segment_size;
    values.resize(num_segments, std::vector<double>(max_non_zero * segment_size, 0.0));
    col_indices.resize(num_segments, std::vector<int>(max_non_zero * segment_size, -1));

    std::vector<int> current_index(rows, 0);
    for (size_t i = 0; i < size; ++i) {
        int row = coo_rows[i];
        int col = coo_cols[i];
        double value = coo_values[i];

        int segment = row / segment_size;
        int offset = row % segment_size;

        values[segment][offset * max_non_zero + current_index[row]] = value;
        col_indices[segment][offset * max_non_zero + current_index[row]] = col;
        current_index[row]++;
    }

    row_pointers.resize(num_segments + 1, 0);
    for (int i = 1; i <= num_segments; ++i) {
        row_pointers[i] = row_pointers[i - 1] + segment_size * max_non_zero;
    }
}

//std::vector<double> SELL_C_matrix::SpMV(std::vector<double>& x) {
//    std::vector<double> result(rows, 0.0);
//    #pragma omp parallel for
//    for (int segment = 0; segment < values.size(); segment++) {
//        for (int offset = 0; offset < segment_size; offset++) {
//            int row = segment * segment_size + offset;
//            if (row >= rows) break;
//
//            for (int i = 0; i < max_non_zero; i++) {
//                int index = offset * max_non_zero + i;
//                if (col_indices[segment][index] != -1) {
//                    #pragma omp atomic
//                    result[row] += values[segment][index] * x[col_indices[segment][index]];
//                }
//            }
//        }
//    }
//    return result;
//}

std::vector<double> SELL_C_matrix::SpMV(const std::vector<double>& x) {
    std::vector<double> result(rows, 0.0);
    int num_segments = values.size();
#ifdef simple 
#ifdef omp
#pragma omp parallel for schedule(dynamic)
#endif
    for (int segment = 0; segment < num_segments; segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            double temp_for_row = 0;
            if (row >= rows) break;

            for (int i = 0; i < segment_max_non_zero; i++) {
                int index = offset * segment_max_non_zero + i;
                int temp_col = col_indices[segment][index];
                if (temp_col == -1) {
                    continue;
                }
                temp_for_row += values[segment][index] * x[col_indices[segment][index]];
            }
            result[row] = temp_for_row;
        }
    }
    return result;
}
#endif
#ifdef avx2
#ifdef omp
#pragma omp parallel for schedule(dynamic)
#endif
    for (int segment = 0; segment < num_segments; segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;

            __m256d local_sum = _mm256_setzero_pd(); // Инициализируем аккумулятор нулями
            int i = 0;

            // Обрабатываем по 4 элемента за раз
            for (; i + 3 < segment_max_non_zero; i += 4) {
                // Загружаем 4 индекса столбцов
                __m128i col_idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&col_indices[segment][offset * segment_max_non_zero + i]));

                // Проверяем, есть ли -1 в индексах
                __m128i mask = _mm_cmpeq_epi32(col_idx, _mm_set1_epi32(-1));
                if (_mm_movemask_epi8(mask) != 0) {
                    // Если есть -1, обрабатываем оставшиеся элементы скалярно
                    for (int j = i; j < i + 4; j++) {
                        int temp_col = col_indices[segment][offset * segment_max_non_zero + j];
                        if (temp_col == -1) continue;
                        result[row] += values[segment][offset * segment_max_non_zero + j] * x[temp_col];
                    }
                    continue;
                }

                // Загружаем 4 значения из матрицы
                __m256d mat_vals = _mm256_loadu_pd(&values[segment][offset * segment_max_non_zero + i]);

                // Собираем 4 значения из вектора x
                __m256d x_vals = _mm256_set_pd(
                    x[_mm_extract_epi32(col_idx, 3)],
                    x[_mm_extract_epi32(col_idx, 2)],
                    x[_mm_extract_epi32(col_idx, 1)],
                    x[_mm_extract_epi32(col_idx, 0)]
                );

                // Умножаем и добавляем к аккумулятору
                local_sum = _mm256_add_pd(local_sum, _mm256_mul_pd(mat_vals, x_vals));
            }

            // Обрабатываем оставшиеся элементы скалярно
            for (; i < segment_max_non_zero; i++) {
                int temp_col = col_indices[segment][offset * segment_max_non_zero + i];
                if (temp_col == -1) continue;
                result[row] += values[segment][offset * segment_max_non_zero + i] * x[temp_col];
            }

            // Суммируем аккумулятор и записываем результат
            double temp[4];
            _mm256_storeu_pd(temp, local_sum);
            result[row] += temp[0] + temp[1] + temp[2] + temp[3];
        }
    }
    return result;
}
#endif
#ifdef avx512
#ifdef omp
#pragma omp parallel for schedule(dynamic)
#endif
    for (int segment = 0; segment < num_segments; segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;

            __m512d local_sum = _mm512_setzero_pd(); // Инициализируем аккумулятор нулями
            int i = 0;

            // Обрабатываем по 8 элементов за раз
            for (; i + 7 < segment_max_non_zero; i += 8) {
                // Загружаем 8 индексов столбцов
                __m256i col_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_indices[segment][offset * segment_max_non_zero + i]));

                // Проверяем, есть ли -1 в индексах
                __m256i mask = _mm256_cmpeq_epi32(col_idx, _mm256_set1_epi32(-1));
                if (_mm256_movemask_epi8(mask) != 0) {
                    // Если есть -1, обрабатываем оставшиеся элементы скалярно
                    for (int j = i; j < i + 8; j++) {
                        int temp_col = col_indices[segment][offset * segment_max_non_zero + j];
                        if (temp_col == -1) continue;
                        result[row] += values[segment][offset * segment_max_non_zero + j] * x[temp_col];
                    }
                    continue;
                }

                // Загружаем 8 значений из матрицы
                __m512d mat_vals = _mm512_loadu_pd(&values[segment][offset * segment_max_non_zero + i]);

                // Собираем 8 значений из вектора x
                __m512d x_vals = _mm512_set_pd(
                    x[_mm256_extract_epi32(col_idx, 7)],
                    x[_mm256_extract_epi32(col_idx, 6)],
                    x[_mm256_extract_epi32(col_idx, 5)],
                    x[_mm256_extract_epi32(col_idx, 4)],
                    x[_mm256_extract_epi32(col_idx, 3)],
                    x[_mm256_extract_epi32(col_idx, 2)],
                    x[_mm256_extract_epi32(col_idx, 1)],
                    x[_mm256_extract_epi32(col_idx, 0)]
                );

                // Умножаем и добавляем к аккумулятору
                local_sum = _mm512_add_pd(local_sum, _mm512_mul_pd(mat_vals, x_vals));
            }

            // Обрабатываем оставшиеся элементы скалярно
            for (; i < segment_max_non_zero; i++) {
                int temp_col = col_indices[segment][offset * segment_max_non_zero + i];
                if (temp_col == -1) continue;
                result[row] += values[segment][offset * segment_max_non_zero + i] * x[temp_col];
            }

            // Суммируем аккумулятор и записываем результат
            result[row] += _mm512_reduce_add_pd(local_sum);
        }
    }
    return result;
    }
#endif

SELL_C_sigma_matrix::SELL_C_sigma_matrix(std::string filename, int segment_size, int sigma)
    : segment_size(segment_size), sigma(sigma) {
    COO_matrix cooMatrix(filename);
    rows = cooMatrix.get_rows();
    cols = cooMatrix.get_cols();
    size = cooMatrix.get_size();

    std::vector<double> coo_values = cooMatrix.get_values();
    std::vector<int> coo_rows = cooMatrix.get_rows_id();
    std::vector<int> coo_cols = cooMatrix.get_cols_id();

    // Count the number of non-zero elements in each row
    std::vector<int> row_counts(rows, 0);
    for (size_t i = 0; i < size; ++i) {
        row_counts[coo_rows[i]]++;
    }

    // Sort rows in blocks of size sigma
    std::vector<int> row_order(rows);
    for (int i = 0; i < rows; ++i) {
        row_order[i] = i;
    }

    for (int block_start = 0; block_start < rows; block_start += sigma) {
        int block_end = std::min(block_start + sigma, rows);
        std::sort(row_order.begin() + block_start, row_order.begin() + block_end,
            [&row_counts](int a, int b) { return row_counts[a] > row_counts[b]; });
    }

    // Create a mapping from original row index to its new position after sorting
    std::vector<int> row_to_sorted_index(rows);
    for (int i = 0; i < rows; ++i) {
        row_to_sorted_index[row_order[i]] = i;
    }

    int num_segments = (rows + segment_size - 1) / segment_size;

    values.resize(num_segments);
    col_indices.resize(num_segments);

    std::vector<int> segment_max_non_zero(num_segments, 0);

    for (int segment = 0; segment < num_segments; segment++) {
        int start_row = segment * segment_size;
        int end_row = std::min(start_row + segment_size, rows);

        // Find the maximum number of non-zero elements in the current segment
        for (int row = start_row; row < end_row; row++) {
            if (row_counts[row_order[row]] > segment_max_non_zero[segment]) {
                segment_max_non_zero[segment] = row_counts[row_order[row]];
            }
        }

        values[segment].resize(static_cast<size_t>(segment_size) * static_cast<size_t>(segment_max_non_zero[segment]), 0.0);
        col_indices[segment].resize(static_cast<size_t>(segment_size) * static_cast<size_t>(segment_max_non_zero[segment]), -1);
    }

    std::vector<int> current_index(rows, 0);
    for (size_t i = 0; i < size; ++i) {
        int row = coo_rows[i];
        int col = coo_cols[i];
        double value = coo_values[i];

        // Use the mapping to find the correct segment and offset
        int sorted_index = row_to_sorted_index[row];
        int segment = sorted_index / segment_size;
        int offset = sorted_index % segment_size;

        values[segment][offset * segment_max_non_zero[segment] + current_index[row]] = value;
        col_indices[segment][offset * segment_max_non_zero[segment] + current_index[row]] = col;
        current_index[row]++;
    }

    row_pointers.resize(num_segments + 1, 0);
    for (int i = 1; i <= num_segments; i++) {
        row_pointers[i] = row_pointers[i - 1] + segment_size * segment_max_non_zero[i - 1];
    }
}

std::vector<double> SELL_C_sigma_matrix::SpMV(const std::vector<double>& x) {
    std::vector<double> result(rows, 0.0);
    int num_segments = values.size();
#ifdef simple 
#ifdef omp
    //#pragma omp parallel for schedule(dynamic)
#endif
    for (int segment = 0; segment < num_segments; segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            double temp_for_row = 0;
            if (row >= rows) break;

            for (int i = 0; i < segment_max_non_zero; i++) {
                int index = offset * segment_max_non_zero + i;
                int temp_col = col_indices[segment][index];
                if (temp_col == -1) {
                    continue;
                }
                temp_for_row += values[segment][index] * x[col_indices[segment][index]];
            }
            result[row] = temp_for_row;
        }
    }
    return result;
}
#endif
#ifdef avx2
#ifdef omp
#pragma omp parallel for schedule(dynamic)
#endif
for (int segment = 0; segment < num_segments; segment++) {
    int segment_max_non_zero = values[segment].size() / segment_size;

    for (int offset = 0; offset < segment_size; offset++) {
        int row = segment * segment_size + offset;
        if (row >= rows) break;

        __m256d local_sum = _mm256_setzero_pd(); // Инициализируем аккумулятор нулями
        int i = 0;

        // Обрабатываем по 4 элемента за раз
        for (; i + 3 < segment_max_non_zero; i += 4) {
            // Загружаем 4 индекса столбцов
            __m128i col_idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&col_indices[segment][offset * segment_max_non_zero + i]));

            // Проверяем, есть ли -1 в индексах
            __m128i mask = _mm_cmpeq_epi32(col_idx, _mm_set1_epi32(-1));
            if (_mm_movemask_epi8(mask) != 0) {
                // Если есть -1, обрабатываем оставшиеся элементы скалярно
                for (int j = i; j < i + 4; j++) {
                    int temp_col = col_indices[segment][offset * segment_max_non_zero + j];
                    if (temp_col == -1) continue;
                    result[row] += values[segment][offset * segment_max_non_zero + j] * x[temp_col];
                }
                continue;
            }

            // Загружаем 4 значения из матрицы
            __m256d mat_vals = _mm256_loadu_pd(&values[segment][offset * segment_max_non_zero + i]);

            // Собираем 4 значения из вектора x
            __m256d x_vals = _mm256_set_pd(
                x[_mm_extract_epi32(col_idx, 3)],
                x[_mm_extract_epi32(col_idx, 2)],
                x[_mm_extract_epi32(col_idx, 1)],
                x[_mm_extract_epi32(col_idx, 0)]
            );

            // Умножаем и добавляем к аккумулятору
            local_sum = _mm256_add_pd(local_sum, _mm256_mul_pd(mat_vals, x_vals));
        }

        // Обрабатываем оставшиеся элементы скалярно
        for (; i < segment_max_non_zero; i++) {
            int temp_col = col_indices[segment][offset * segment_max_non_zero + i];
            if (temp_col == -1) continue;
            result[row] += values[segment][offset * segment_max_non_zero + i] * x[temp_col];
        }

        // Суммируем аккумулятор и записываем результат
        double temp[4];
        _mm256_storeu_pd(temp, local_sum);
        result[row] += temp[0] + temp[1] + temp[2] + temp[3];
    }
}
return result;
}
#endif
#ifdef avx512
#ifdef omp
#pragma omp parallel for schedule(dynamic)
#endif
    for (int segment = 0; segment < num_segments; segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;

            __m512d local_sum = _mm512_setzero_pd(); // Инициализируем аккумулятор нулями
            int i = 0;

            // Обрабатываем по 8 элементов за раз
            for (; i + 7 < segment_max_non_zero; i += 8) {
                // Загружаем 8 индексов столбцов
                __m256i col_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    &col_indices[segment][offset * segment_max_non_zero + i]));

                // Проверяем, есть ли -1 в индексах
                __m256i mask = _mm256_cmpeq_epi32(col_idx, _mm256_set1_epi32(-1));
                if (_mm256_movemask_epi8(mask) != 0) {
                    // Если есть -1, обрабатываем оставшиеся элементы скалярно
                    for (int j = i; j < i + 8; j++) {
                        int temp_col = col_indices[segment][offset * segment_max_non_zero + j];
                        if (temp_col == -1) continue;
                        result[row] += values[segment][offset * segment_max_non_zero + j] * x[temp_col];
                    }
                    continue;
                }

                // Загружаем 8 значений из матрицы
                __m512d mat_vals = _mm512_loadu_pd(&values[segment][offset * segment_max_non_zero + i]);

                // Собираем 8 значений из вектора x
                __m512d x_vals = _mm512_set_pd(
                    x[_mm256_extract_epi32(col_idx, 7)],
                    x[_mm256_extract_epi32(col_idx, 6)],
                    x[_mm256_extract_epi32(col_idx, 5)],
                    x[_mm256_extract_epi32(col_idx, 4)],
                    x[_mm256_extract_epi32(col_idx, 3)],
                    x[_mm256_extract_epi32(col_idx, 2)],
                    x[_mm256_extract_epi32(col_idx, 1)],
                    x[_mm256_extract_epi32(col_idx, 0)]
                );

                // Умножаем и добавляем к аккумулятору
                local_sum = _mm512_add_pd(local_sum, _mm512_mul_pd(mat_vals, x_vals));
            }

            // Обрабатываем оставшиеся элементы скалярно
            for (; i < segment_max_non_zero; i++) {
                int temp_col = col_indices[segment][offset * segment_max_non_zero + i];
                if (temp_col == -1) continue;
                result[row] += values[segment][offset * segment_max_non_zero + i] * x[temp_col];
            }

            // Суммируем аккумулятор и записываем результат
            result[row] += _mm512_reduce_add_pd(local_sum);
        }
    }
    return result;
    }
#endif