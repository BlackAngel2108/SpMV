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
#if defined  (simple) || defined(risc) || defined(avx512)
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

#if defined (avx512_test) || defined(avx2)
    #ifdef omp
    #pragma omp parallel for schedule (dynamic)
    #endif
    for (int row = 0; row < rows; ++row) {
        __m512d local_sum = _mm512_setzero_pd();
        int row_start = row_pointers[row];
        int row_end = row_pointers[row + 1];
        int k = row_end;
        int i = row_start;
        for (; i+8<=row_end; k -= 8, i += 8) {
            // Загружаем 8 индексов столбцов
            __m256i col_idx = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&column_indices[i]));

            // Загружаем 8 значений матрицы
            __m512d mat_vals = _mm512_loadu_pd(&values[i]);

            // Собираем 8 значений из вектора vec
            __m512d vec_vals = _mm512_i32gather_pd(col_idx, vec.data(), 8);

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
    col_indices.resize(rows, std::vector<int>(max_non_zero,0.0));

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
        local_sum += values[row][i] * x[idx];
    }
    result[row]+= local_sum;
    }
    return result;
    }
#endif
#ifdef avx2
        std::vector<double> result(rows, 0.0);
#ifdef omp
#pragma omp parallel for schedule (dynamiс)
#endif
for (int row = 0; row < rows; ++row) {
            __m256d local_sum = _mm256_setzero_pd();  // Инициализируем аккумулятор нулями

            // Обрабатываем по 4 элемента за раз (AVX2 работает с 256-битными регистрами, 4 double)
            int k = max_non_zero;
            int i = 0;
            for (; i+4<=max_non_zero; k -= 4, i += 4) {
                // Загружаем 4 индекса столбцов
                __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&col_indices[row][i]));

                // Загружаем 4 значения из матрицы
                __m256d vals = _mm256_loadu_pd(&values[row][i]);

                // Собираем значения из вектора x по индексам
                __m256d x_vals = _mm256_i32gather_pd(x.data(), idx,4);

                // Умножаем и добавляем к аккумулятору
                local_sum = _mm256_fmadd_pd(vals, x_vals, local_sum);
            }

            // Горизонтальное суммирование аккумулятора
            __m128d sum128 = _mm_add_pd(_mm256_extractf128_pd(local_sum, 1),
                _mm256_castpd256_pd128(local_sum));
            sum128 = _mm_hadd_pd(sum128, sum128);
            result[row] = _mm_cvtsd_f64(sum128);

            // Обрабатываем оставшиеся элементы скалярно
            for (; i < max_non_zero; ++i) {
                result[row] += values[row][i] * x[col_indices[row][i]];
            }
        }
        return result;
}
#endif

#ifdef avx512
        std::vector<double> result(rows, 0.0);
    #ifdef omp
    #pragma omp parallel for schedule (dynamic)
    #endif
	for (int row = 0; row < rows; ++row) {
            __m512d local_sum = _mm512_setzero_pd();  // Инициализируем аккумулятор нулями

            // Обрабатываем по 8 элементов за раз (AVX-512 работает с 512-битными регистрами, 8 double)
            int k = max_non_zero;
            int i = 0;
            for (; i+8<=max_non_zero ; k -= 8, i += 8) {
                // Загружаем 8 индексов столбцов
                __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_indices[row][i]));
                // Загружаем 8 значений из матрицы
                __m512d vals = _mm512_loadu_pd(&values[row][i]);

                __m512d x_vals = _mm512_i32gather_pd(idx, x.data(), 8);

                // Умножаем и добавляем к аккумулятору
                local_sum = _mm512_fmadd_pd(vals, x_vals, local_sum);
            }
            // Суммируем аккумулятор и записываем результат
            double temp = _mm512_reduce_add_pd(local_sum);
            result[row] += temp;
            
            // Обрабатываем оставшиеся элементы скалярно
            for (; i < max_non_zero; ++i) {
                result[row] += values[row][i] * x[col_indices[row][i]];
            }
        }
return result;
}
#endif

#ifdef risc
        std::vector<double> result(rows, 0.0);
#ifdef omp
#pragma omp parallel for schedule(dynamic)
#endif
        for (int row = 0; row < rows; ++row) {
            float64_t scalar_sum = 0.0;

            size_t vlmax = vsetvlmax_e64m1();// Устанавливаем максимальную длину для double

            vfloat64m1_t vec_sum =vfmv_v_f_f64m1(0.0, vlmax);// Векторный аккумулятор

            int i = 0;
            int k = max_non_zero;
            size_t vl=0;
            for (vl; k>0; k-=vl, i += vl) {
                vl = vsetvl_e64m1(k);
                
        	vuint32mf2_t vec_indices = vle32_v_u32mf2(&col_indices[row][i], vl);
                vec_indices = vsll_vx_u32mf2(vec_indices, 3 , vl);
                                                
                // заг��зка x_val �лемен�ов по индек�ам
        	vfloat64m1_t x_vals = vluxei32_v_f64m1(&x[0], vec_indices, vl);

                // Загрузка значений матрицы
                vfloat64m1_t mat_vals = vle_v_f64m1(&values[row][i], vl);

                // Умножение и сложение (FMA)
                vec_sum = vfmacc_vv_f64m1(vec_sum, mat_vals, x_vals, vl);
            }
            vfloat64m1_t v_reduce_sum = vfredosum_vs_f64m1_f64m1(vec_sum,0.0,vlmax);
	    vse64_v_f64m1(&scalar_sum,v_reduce_sum,vlmax);
            result[row] = scalar_sum;;
        }

        return result;
}
#endif

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
    col_indices.resize(num_segments, std::vector<int>(max_non_zero * segment_size,0));

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

                // Загружаем 4 значения из матрицы
                __m256d mat_vals = _mm256_loadu_pd(&values[segment][offset * segment_max_non_zero + i]);

                // Собираем 4 значения из вектора x
                __m256d x_vals = _mm256_i32gather_pd(x.data(), col_idx, 8);

                // Умножаем и добавляем к аккумулятору
                local_sum = _mm256_fmadd_pd(mat_vals, x_vals, local_sum);
            }

            // Обрабатываем оставшиеся элементы скалярно
            for (; i < segment_max_non_zero; i++) {
                int temp_col = col_indices[segment][offset * segment_max_non_zero + i];
                result[row] += values[segment][offset * segment_max_non_zero + i] * x[temp_col];
            }
            // Горизонтальное суммирование аккумулятора
            __m128d sum128 = _mm_add_pd(_mm256_extractf128_pd(local_sum, 1),
                _mm256_castpd256_pd128(local_sum));
            sum128 = _mm_hadd_pd(sum128, sum128);
            result[row] = _mm_cvtsd_f64(sum128);

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

            __m512d local_sum = _mm512_setzero_pd(); // �������������� ����������� ������
            int i = 0;

            // ������������ �� 8 ��������� �� ���
            for (; i + 7 < segment_max_non_zero; i += 8) {
                // ��������� 8 �������� ��������
                __m256i col_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_indices[segment][offset * segment_max_non_zero + i]));

                // ��������� 8 �������� �� �������
                __m512d mat_vals = _mm512_loadu_pd(&values[segment][offset * segment_max_non_zero + i]);

                // �������� 8 �������� �� ������� x
                __m512d x_vals = _mm512_i32gather_pd(col_idx, x.data(), 8);

                // �������� � ��������� � ������������
                local_sum = _mm512_fmadd_pd(mat_vals, x_vals,local_sum);
            }

            // ������������ ���������� �������� ��������
            for (; i < segment_max_non_zero; i++) {
                int temp_col = col_indices[segment][offset * segment_max_non_zero + i];
                result[row] += values[segment][offset * segment_max_non_zero + i] * x[temp_col];
            }

            // ��������� ����������� � ���������� ���������
            result[row] += _mm512_reduce_add_pd(local_sum);
        }
    }
    return result;
    }
#endif

#ifdef risc
    #ifdef omp
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int segment = 0; segment < num_segments; segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;
            size_t vlmax = vsetvlmax_e64m1();
            vfloat64m1_t vec_sum = vfmv_v_f_f64m1(0.0, vlmax); // Векторный аккумулятор

            int i=0;
            int k = segment_max_non_zero;
            for (size_t vl; k > 0; k -= vl, i += vl) {
                vl = vsetvl_e64m1(k);
                int index = offset * segment_max_non_zero + i;

                vint32m1_t vec_indices = vle_v_i32m1(&col_indices[segment][index], vl);
                // Преобразование индексов в 64-битные
                vint64m1_t vec_indices_64 = vwadd_vx_i64m8(vec_indices, 0, vl);

                vfloat64m1_t x_vals = vluxei64_v_f64m1(x.data(), vec_indices_64, vl);

                vfloat64m1_t mat_vals = vle_v_f64m1(&values[segment][index], vl);
                vec_sum = vfmacc_vv_f64m1(vec_sum, mat_vals, x_vals, vl);

            }
            // Скалярное суммирование оставшихся элементов
            
            double scalar_sum = vfmv_f_s_f64m1_f64(vec_sum);
            result[row] = scalar_sum;
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

        __m256d local_sum = _mm256_setzero_pd(); // �������������� ����������� ������
        int i = 0;

        // ������������ �� 4 �������� �� ���
        for (; i + 3 < segment_max_non_zero; i += 4) {
            // ��������� 4 ������� ��������
            __m128i col_idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&col_indices[segment][offset * segment_max_non_zero + i]));

            // ���������, ���� �� -1 � ��������
            __m128i mask = _mm_cmpeq_epi32(col_idx, _mm_set1_epi32(-1));
            if (_mm_movemask_epi8(mask) != 0) {
                // ���� ���� -1, ������������ ���������� �������� ��������
                for (int j = i; j < i + 4; j++) {
                    int temp_col = col_indices[segment][offset * segment_max_non_zero + j];
                    if (temp_col == -1) continue;
                    result[row] += values[segment][offset * segment_max_non_zero + j] * x[temp_col];
                }
                continue;
            }

            // ��������� 4 �������� �� �������
            __m256d mat_vals = _mm256_loadu_pd(&values[segment][offset * segment_max_non_zero + i]);

            // �������� 4 �������� �� ������� x
            __m256d x_vals = _mm256_set_pd(
                x[_mm_extract_epi32(col_idx, 3)],
                x[_mm_extract_epi32(col_idx, 2)],
                x[_mm_extract_epi32(col_idx, 1)],
                x[_mm_extract_epi32(col_idx, 0)]
            );

            // �������� � ��������� � ������������
            local_sum = _mm256_add_pd(local_sum, _mm256_mul_pd(mat_vals, x_vals));
        }

        // ������������ ���������� �������� ��������
        for (; i < segment_max_non_zero; i++) {
            int temp_col = col_indices[segment][offset * segment_max_non_zero + i];
            if (temp_col == -1) continue;
            result[row] += values[segment][offset * segment_max_non_zero + i] * x[temp_col];
        }

        // ��������� ����������� � ���������� ���������
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

            __m512d local_sum = _mm512_setzero_pd(); // �������������� ����������� ������
            int i = 0;

            // ������������ �� 8 ��������� �� ���
            for (; i + 7 < segment_max_non_zero; i += 8) {
                // ��������� 8 �������� ��������
                __m256i col_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_indices[segment][offset * segment_max_non_zero + i]));

                // ��������� 8 �������� �� �������
                __m512d mat_vals = _mm512_loadu_pd(&values[segment][offset * segment_max_non_zero + i]);

                // �������� 8 �������� �� ������� x
                __m512d x_vals = _mm512_i32gather_pd(col_idx, x.data(), 8);

                // �������� � ��������� � ������������
                local_sum = _mm512_fmadd_pd(mat_vals, x_vals,local_sum);
            }

            // ������������ ���������� �������� ��������
            for (; i < segment_max_non_zero; i++) {
                int temp_col = col_indices[segment][offset * segment_max_non_zero + i];
                result[row] += values[segment][offset * segment_max_non_zero + i] * x[temp_col];
            }

            // ��������� ����������� � ���������� ���������
            result[row] += _mm512_reduce_add_pd(local_sum);
        }
    }


    return result;
    }
#endif

#ifdef risc
    #ifdef omp
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int segment = 0; segment < num_segments; segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;
            vfloat64m1_t vec_sum = vfmv_v_f_f64m1(0.0, vlmax); // Векторный аккумулятор

            int i=0;
            int k = segment_max_non_zero;
            for (size_t vl; k > 0; k -= vl, i += vl) {
                vl = __riscv_vsetvl_e64m1(k);
                int index = offset * segment_max_non_zero + i;

                vint32m1_t vec_indices = vle32_v_i32m1(&col_indices[segment][index], vl);
                // Преобразование индексов в 64-битные
                vint64m1_t vec_indices_64 = vwadd_vx_i64m1(vec_indices, 0, vl);

                vfloat64m1_t x_vals = vluxei64_v_f64m1(x.data(), vec_indices_64, vl);

                vfloat64m1_t mat_vals = vle64_v_f64m1(&values[segment][index], vl);
                vec_sum = vfmacc_vv_f64m1(vec_sum, mat_vals, x_vals, vl);

            }
            // Скалярное суммирование оставшихся элементов
            double scalar_sum = vfmv_f_s_f64m1_f64(vec_sum);
            result[row] = scalar_sum;
        }
    }
    return result;
    }
#endif
