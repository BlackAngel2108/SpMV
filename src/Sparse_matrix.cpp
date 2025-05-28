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
#if defined  (simple) || defined(risc) || defined(avx512) || defined(avx2)
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

#if defined (avx512_test) || defined(avx2_test)
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

    // Resize the single-dimensional arrays
    values.resize(rows * max_non_zero, 0.0);
    col_indices.resize(rows * max_non_zero, 0);

    // Initialize row_starts
    row_starts.resize(rows + 1, 0);
    for (int i = 0; i < rows; ++i) {
        row_starts[i + 1] = row_starts[i] + row_counts[i];
    }

    std::vector<int> current_index(rows, 0);
    for (size_t i = 0; i < size; ++i) {
        int row = coo_rows[i];
        int col = coo_cols[i];
        double value = coo_values[i];

        int index = row_starts[row] + current_index[row];
        values[index] = value;
        col_indices[index] = col;
        current_index[row]++;
    }
}


std::vector<double> ELLPack_matrix::SpMV(const std::vector<double>& x) {
#ifdef simple
    std::vector<double> result(rows, 0.0);
  //omp_set_num_threads(4);
#ifdef omp 
#pragma omp parallel for schedule(dynamic, 1000)
#endif
    for (int i = 0; i < rows; ++i) {
        for (int j = row_starts[i]; j < row_starts[i + 1]; ++j) {
            result[i] += values[j] * x[col_indices[j]];
        }
    }
    return result;
}
#endif

#ifdef avx2
std::vector<double> result(rows, 0.0);
#ifdef omp
#pragma omp parallel for schedule(dynamic, 1000)
#endif
for (int row = 0; row < rows; row++) {
    __m256d local_sum = _mm256_setzero_pd();  // 256-битный аккумулятор (4 doubles)

    int row_start = row_starts[row];
    int row_end = row_starts[row + 1];
    int length = row_end - row_start;

    int i = 0;
    for (; i + 4 <= length; i += 4) {
        // Загружаем 4 индекса (int)
        __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&col_indices[row_start + i]));

        // Загружаем 4 значения из матрицы
        __m256d mat_vals = _mm256_loadu_pd(&values[row_start + i]);

        // Собираем 4 значения из вектора x по индексам
        __m256d x_vals = _mm256_i32gather_pd(x.data(), idx, 8);  // 8 = sizeof(double)

        // Умножение и сложение
        local_sum = _mm256_fmadd_pd(mat_vals, x_vals, local_sum);
        // __m256d mul = _mm256_mul_pd(mat_vals, x_vals);
        // local_sum = _mm256_add_pd(local_sum, mul);
    }

    // Суммируем 4 значения в аккумуляторе
    __m128d low128  = _mm256_castpd256_pd128(local_sum);
    __m128d high128 = _mm256_extractf128_pd(local_sum, 1);
    __m128d sum128  = _mm_add_pd(low128, high128);
    sum128 = _mm_hadd_pd(sum128, sum128);
    double temp = _mm_cvtsd_f64(sum128);

    // Обработка оставшихся элементов скалярно
    for (; i < length; i++) {
        temp += values[row_start + i] * x[col_indices[row_start + i]];
    }

    result[row] = temp;
}
return result;
}
#endif

#ifdef avx512
    std::vector<double> result(rows, 0.0);

#ifdef omp
#pragma omp parallel for schedule(dynamic, 1000)
#endif
    for (int row = 0; row < rows; ++row) {
        __m512d local_sum = _mm512_setzero_pd();  // Инициализируем аккумулятор нулями

        // Обрабатываем по 8 элементов за раз (AVX-512 работает с 512-битными регистрами, 8 double)
        int k = row_starts[row + 1] - row_starts[row];
        int i = row_starts[row];
        for (; i + 8 <= row_starts[row + 1]; k -= 8, i += 8) {
            // Загружаем 8 индексов столбцов
            __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_indices[i]));
            // Загружаем 8 значений из матрицы
            __m512d vals = _mm512_loadu_pd(&values[i]);

            __m512d x_vals = _mm512_i32gather_pd(idx, x.data(), 8);

            // Умножаем и добавляем к аккумулятору
            local_sum = _mm512_fmadd_pd(vals, x_vals, local_sum);
        }
        // Суммируем аккумулятор и записываем результат
        double temp = _mm512_reduce_add_pd(local_sum);
        result[row] += temp;

        // Обрабатываем оставшиеся элементы скалярно
        for (; i < row_starts[row + 1]; ++i) {
            result[row] += values[i] * x[col_indices[i]];
        }
    }
    return result;
}
#endif

// #ifdef risc
//         std::vector<double> result(rows, 0.0);
// #ifdef omp
// #pragma omp parallel for schedule(dynamic)
// #endif
//         for (int row = 0; row < rows; ++row) {
//             float64_t scalar_sum = 0.0;

//             size_t vlmax = vsetvlmax_e64m1();// Устанавливаем максимальную длину для double

//             vfloat64m1_t vec_sum =vfmv_v_f_f64m1(0.0, vlmax);// Векторный аккумулятор

//             int i = 0;
//             int k = max_non_zero;
//             size_t vl=0;
//             for (vl; k>0; k-=vl, i += vl) {
//                 vl = vsetvl_e64m1(k);
                
//                 vuint32mf2_t vec_indices = vle32_v_u32mf2(&col_indices[row][i], vl);
//                 vec_indices = vsll_vx_u32mf2(vec_indices, 3 , vl);
                                                    
//                 vfloat64m1_t x_vals = vluxei32_v_f64m1(&x[0], vec_indices, vl);

//                 vfloat64m1_t mat_vals = vle_v_f64m1(&values[row][i], vl);

//                 // Умножение и сложение (FMA)
//                 vec_sum = vfmacc_vv_f64m1(vec_sum, mat_vals, x_vals, vl);
//             }
//             vfloat64m1_t v_reduce_sum = vfredosum_vs_f64m1_f64m1(vec_sum,0.0,vlmax);
// 	        vse64_v_f64m1(&scalar_sum,v_reduce_sum,vlmax);
//             result[row] = scalar_sum;;
//         }

//         return result;
// }
// #endif
#ifdef risc
std::vector<double> ELLPack_matrix::SpMV(const std::vector<double>& x) {
    std::vector<double> result(rows, 0.0);

#ifdef omp
#pragma omp parallel for schedule(dynamic)
#endif
    for (int row = 0; row < rows; ++row) {
        float64_t scalar_sum = 0.0;

        size_t vlmax = vsetvlmax_e64m1(); // Устанавливаем максимальную длину для double

        vfloat64m1_t vec_sum = vfmv_v_f_f64m1(0.0, vlmax); // Векторный аккумулятор

        int i = row_starts[row];
        int k = row_starts[row + 1] - row_starts[row];
        size_t vl = 0;
        for (vl; k > 0; k -= vl, i += vl) {
            vl = vsetvl_e64m1(k);

            vuint32mf2_t vec_indices = vle32_v_u32mf2(&col_indices[i], vl);
            vec_indices = vsll_vx_u32mf2(vec_indices, 3, vl);

            vfloat64m1_t x_vals = vluxei32_v_f64m1(&x[0], vec_indices, vl);

            vfloat64m1_t mat_vals = vle_v_f64m1(&values[i], vl);

            // Умножение и сложение (FMA)
            vec_sum = vfmacc_vv_f64m1(vec_sum, mat_vals, x_vals, vl);
        }
        vfloat64m1_t v_reduce_sum = vfredosum_vs_f64m1_f64m1(vec_sum, 0.0, vlmax);
        vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
        result[row] = scalar_sum;
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

    // Подсчет количества ненулевых элементов в каждой строке
    std::vector<int> row_counts(rows, 0);
    for (size_t i = 0; i < size; ++i) {
        row_counts[coo_rows[i]]++;
    }

    int num_segments = (rows + segment_size - 1) / segment_size;
    std::vector<int> segment_non_zero_counts(num_segments, 0);

    // Подсчет количества ненулевых элементов в каждом сегменте
    for (int row = 0; row < rows; ++row) {
        int segment = row / segment_size;
        if (row_counts[row] > segment_non_zero_counts[segment]) {
            segment_non_zero_counts[segment] = row_counts[row];
        }
    }

    // Инициализация segment_starts
    segment_starts.resize(num_segments + 1, 0);
    for (int i = 1; i <= num_segments; ++i) {
        segment_starts[i] = segment_starts[i - 1] + segment_size * segment_non_zero_counts[i - 1];
    }

    // Изменение размера одномерных массивов
    int total_non_zero = 0;
    for (int segment_non_zero : segment_non_zero_counts) {
        total_non_zero += segment_non_zero * segment_size;
    }
    values.resize(total_non_zero, 0.0);
    col_indices.resize(total_non_zero, 0);

    std::vector<int> current_index(rows, 0);
    for (size_t i = 0; i < size; ++i) {
        int row = coo_rows[i];
        int col = coo_cols[i];
        double value = coo_values[i];

        int segment = row / segment_size;
        int offset = row % segment_size;

        int index = segment_starts[segment] + offset * segment_non_zero_counts[segment] + current_index[row];
        values[index] = value;
        col_indices[index] = col;
        current_index[row]++;
    }
}


std::vector<double> SELL_C_matrix::SpMV(const std::vector<double>& x) {
    std::vector<double> result(rows, 0.0);
    int num_segments = values.size();
#ifdef simple
#ifdef omp
#pragma omp parallel for schedule(dynamic, 1000)
#endif
for (int segment = 0; segment < (rows + segment_size - 1) / segment_size; ++segment) {
    int segment_max_non_zero = (segment_starts[segment + 1] - segment_starts[segment]) / segment_size;

    for (int offset = 0; offset < segment_size; ++offset) {
        int row = segment * segment_size + offset;
        double temp_for_row = 0;
        if (row >= rows) break;

        for (int i = 0; i < segment_max_non_zero; ++i) {
            int index = segment_starts[segment] + offset * segment_max_non_zero + i;
            int temp_col = col_indices[index];
            temp_for_row += values[index] * x[temp_col];
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
    //int segment_max_non_zero = values[segment].size() / segment_size;
    int segment_max_non_zero = (segment_starts[segment + 1] - segment_starts[segment]) / segment_size;

    for (int offset = 0; offset < segment_size; offset++) {
        int row = segment * segment_size + offset;
        if (row >= rows) break;

        __m256d local_sum = _mm256_setzero_pd(); // 4 doubles accumulator
        int i = 0;

        // SIMD блок — по 4 значения
        for (; i + 3 < segment_max_non_zero; i += 4) {
            int index = segment_starts[segment] + offset * segment_max_non_zero + i;
            __m128i col_idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                &col_indices[index]));

            __m256d mat_vals = _mm256_loadu_pd(
                &values[index]);

            __m256d x_vals = _mm256_i32gather_pd(
                x.data(), col_idx, 8);  // 8 байт = sizeof(double)

            //local_sum = _mm256_fmadd_pd(mat_vals, x_vals, local_sum);
            __m256d mul = _mm256_mul_pd(mat_vals, x_vals);
            local_sum = _mm256_add_pd(local_sum, mul);

        }

        // Горизонтальное суммирование: local_sum = [a, b, c, d] → temp = a + b + c + d
        __m128d sum128 = _mm_add_pd(
            _mm256_extractf128_pd(local_sum, 1),
            _mm256_castpd256_pd128(local_sum));
        sum128 = _mm_hadd_pd(sum128, sum128);
        double temp = _mm_cvtsd_f64(sum128);

        // Скалярные остатки
        for (; i < segment_max_non_zero; i++) {
            int index = segment_starts[segment] + offset * segment_max_non_zero + i;
            int temp_col = col_indices[index];
            temp += values[index] * x[temp_col];
        }

        result[row] = temp;
    }
}
return result;
}
#endif

// #ifdef avx512
// #ifdef omp
// #pragma omp parallel for schedule(dynamic, 1000)
// #endif
// for (int segment = 0; segment < num_segments; segment++) {
//     int segment_max_non_zero = values[segment].size() / segment_size;

//     for (int offset = 0; offset < segment_size; offset++) {
//         int row = segment * segment_size + offset;
//         if (row >= rows) break;

//         __m512d local_sum = _mm512_setzero_pd(); 
//         int i = 0;

//         for (; i + 7 < segment_max_non_zero; i += 8) {
//             __m256i col_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_indices[segment][offset * segment_max_non_zero + i]));

//             __m512d mat_vals = _mm512_loadu_pd(&values[segment][offset * segment_max_non_zero + i]);

//             __m512d x_vals = _mm512_i32gather_pd(col_idx, x.data(), 8);

//             local_sum = _mm512_fmadd_pd(mat_vals, x_vals,local_sum);
//         }

//         for (; i < segment_max_non_zero; i++) {
//             int temp_col = col_indices[segment][offset * segment_max_non_zero + i];
//             result[row] += values[segment][offset * segment_max_non_zero + i] * x[temp_col];
//         }

//         result[row] += _mm512_reduce_add_pd(local_sum);
//     }
// }
// return result;
// }
// #endif

#ifdef avx512
#ifdef omp
#pragma omp parallel for schedule(dynamic, 1000)
#endif
    for (int segment = 0; segment < num_segments; segment++) {
        int segment_max_non_zero = (segment_starts[segment + 1] - segment_starts[segment]) / segment_size;

        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;

            __m512d local_sum = _mm512_setzero_pd();
            int i = 0;

            for (; i + 7 < segment_max_non_zero; i += 8) {
                int index = segment_starts[segment] + offset * segment_max_non_zero + i;
                if (index + 7 >= segment_starts[segment + 1]) break; // Проверка выхода за границы

                __m256i col_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_indices[index]));

                __m512d mat_vals = _mm512_loadu_pd(&values[index]);

                __m512d x_vals = _mm512_i32gather_pd(col_idx, x.data(), 8);

                local_sum = _mm512_fmadd_pd(mat_vals, x_vals, local_sum);
            }

            for (; i < segment_max_non_zero; i++) {
                int index = segment_starts[segment] + offset * segment_max_non_zero + i;
                if (index >= segment_starts[segment + 1]) break; // Проверка выхода за границы

                int temp_col = col_indices[index];
                result[row] += values[index] * x[temp_col];
            }

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
            // вектор с нулями
            vfloat64m1_t v_zero = vfmv_v_f_f64m1(0.0, vlmax);
            int i=0;
            int k = segment_max_non_zero;
            for (size_t vl; k > 0; k -= vl, i += vl) {
                vl = vsetvl_e64m1(k);
                int index = offset * segment_max_non_zero + i;

                vuint32mf2_t vec_indices32 = vle32_v_u32mf2(&col_indices[segment][index], vl);
                vec_indices32 = vsll_vx_u32mf2(vec_indices, 3 , vl);

                vfloat64m1_t x_vals = vluxei32_v_f64m1(&x[0], vec_indices32, vl);

                vfloat64m1_t mat_vals = vle_v_f64m1(&values[segment][index], vl);
                vec_sum = vfmacc_vv_f64m1(vec_sum, mat_vals, x_vals, vl);

            }
            // сложение всех элементов вектора
            vfloat64m1_t v_reduce_sum = __riscv_vfredusum_vs_f64m1_f64m1(vec_sum, v_zero, vlmax);
            // Выгрузка значения из вектора v_reduce_sum в переменную scalar_sum
            __riscv_vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
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

    // Подсчет количества ненулевых элементов в каждой строке
    std::vector<int> row_counts(rows, 0);
    for (size_t i = 0; i < size; ++i) {
        row_counts[coo_rows[i]]++;
    }

    // Сортировка строк блоками размера sigma
    std::vector<int> row_order(rows);
    for (int i = 0; i < rows; ++i) {
        row_order[i] = i;
    }

    for (int block_start = 0; block_start < rows; block_start += sigma) {
        int block_end = std::min(block_start + sigma, rows);
        std::sort(row_order.begin() + block_start, row_order.begin() + block_end,
            [&row_counts](int a, int b) { return row_counts[a] > row_counts[b]; });
    }

    // Создание отображения из исходного индекса строки в её новую позицию после сортировки
    std::vector<int> row_to_sorted_index(rows);
    for (int i = 0; i < rows; ++i) {
        row_to_sorted_index[row_order[i]] = i;
    }
    sorted_to_row_index.resize(rows);
    for (int i = 0; i < rows; ++i) {
        sorted_to_row_index[row_to_sorted_index[i]] = i;
    }

    int num_segments = (rows + segment_size - 1) / segment_size;
    std::vector<int> segment_non_zero_counts(num_segments, 0);

    // Подсчет количества ненулевых элементов в каждом сегменте
    for (int row = 0; row < rows; ++row) {
        int segment = row_to_sorted_index[row] / segment_size;
        if (row_counts[row] > segment_non_zero_counts[segment]) {
            segment_non_zero_counts[segment] = row_counts[row];
        }
    }

    // Инициализация segment_starts
    segment_starts.resize(num_segments + 1, 0);
    for (int i = 1; i <= num_segments; ++i) {
        segment_starts[i] = segment_starts[i - 1] + segment_size * segment_non_zero_counts[i - 1];
    }

    // Изменение размера одномерных массивов
    int total_non_zero = 0;
    for (int segment_non_zero : segment_non_zero_counts) {
        total_non_zero += segment_non_zero * segment_size;
    }
    values.resize(total_non_zero, 0.0);
    col_indices.resize(total_non_zero, 0);

    std::vector<int> current_index(rows, 0);
    for (size_t i = 0; i < size; ++i) {
        int row = coo_rows[i];
        int col = coo_cols[i];
        double value = coo_values[i];

        // Использование отображения для нахождения правильного сегмента и смещения
        int sorted_index = row_to_sorted_index[row];
        int segment = sorted_index / segment_size;
        int offset = sorted_index % segment_size;

        int index = segment_starts[segment] + offset * segment_non_zero_counts[segment] + current_index[row];
        values[index] = value;
        col_indices[index] = col;
        current_index[row]++;
    }
}

std::vector<double> SELL_C_sigma_matrix::SpMV(const std::vector<double>& x) {
#ifdef simple 
std::vector<double> result(rows, 0.0);
int num_segments = (rows + segment_size - 1) / segment_size;
#ifdef omp
    #pragma omp parallel for schedule(dynamic, 1000)
#endif
for (int segment = 0; segment < num_segments; segment++) {
    int segment_max_non_zero = (segment_starts[segment + 1] - segment_starts[segment]) / segment_size;

    for (int offset = 0; offset < segment_size; offset++) {
        int row = segment * segment_size + offset;
        double temp_for_row = 0;
        if (row >= rows) break;

        for (int i = 0; i < segment_max_non_zero; i++) {
            int index = segment_starts[segment] + offset * segment_max_non_zero + i;
            int temp_col = col_indices[index];
            temp_for_row += values[index] * x[temp_col];
        }
        // Использование отображения из sorted_row обратно в исходный индекс строки
        int original_row = sorted_to_row_index[row];
        result[original_row] = temp_for_row;
    }
}
return result;
}

#endif
#ifdef avx2
std::vector<double> result(rows, 0.0);
int num_segments = (rows + segment_size - 1) / segment_size;
#ifdef omp
#pragma omp parallel for schedule(dynamic)
#endif
for (int segment = 0; segment < num_segments; segment++) {
    //int segment_max_non_zero = values[segment].size() / segment_size;
    int segment_max_non_zero = (segment_starts[segment + 1] - segment_starts[segment]) / segment_size;
    for (int offset = 0; offset < segment_size; offset++) {
        int row = segment * segment_size + offset;
        if (row >= rows) break;

        __m256d local_sum = _mm256_setzero_pd(); // 4 doubles accumulator
        int i = 0;

        // SIMD блок — по 4 значения
        for (; i + 3 < segment_max_non_zero; i += 4) {
            int index = segment_starts[segment] + offset * segment_max_non_zero + i;
            __m128i col_idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                &col_indices[index]));

            __m256d mat_vals = _mm256_loadu_pd(
                &values[index]);

            __m256d x_vals = _mm256_i32gather_pd(
                x.data(), col_idx, 8);  // 8 байт = sizeof(double)

            local_sum = _mm256_fmadd_pd(mat_vals, x_vals, local_sum);
        }

        // Горизонтальное суммирование: local_sum = [a, b, c, d] → temp = a + b + c + d
        __m128d sum128 = _mm_add_pd(
            _mm256_extractf128_pd(local_sum, 1),
            _mm256_castpd256_pd128(local_sum));
        sum128 = _mm_hadd_pd(sum128, sum128);
        double temp = _mm_cvtsd_f64(sum128);
        int original_row = sorted_to_row_index[row];
        // Скалярные остатки
        for (; i < segment_max_non_zero; i++) {
            int index = segment_starts[segment] + offset * segment_max_non_zero + i;
            int temp_col = col_indices[index];
            temp += values[index] * x[temp_col];
        }

        result[original_row] = temp;
    }
}
return result;
}
#endif


#ifdef avx512
std::vector<double> result(rows, 0.0);
int num_segments = (rows + segment_size - 1) / segment_size;
#ifdef omp
#pragma omp parallel for schedule(dynamic, 1000)
#endif
for (int segment = 0; segment < num_segments; segment++) {
        //int segment_max_non_zero = values[segment].size() / segment_size;
        int segment_max_non_zero = (segment_starts[segment + 1] - segment_starts[segment]) / segment_size;
        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;

            __m512d local_sum = _mm512_setzero_pd(); 
            int i = 0;

            for (; i + 7 < segment_max_non_zero; i += 8) {
                int index = segment_starts[segment] + offset * segment_max_non_zero + i;
                __m256i col_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_indices[index]));

                __m512d mat_vals = _mm512_loadu_pd(&values[index]);

                __m512d x_vals = _mm512_i32gather_pd(col_idx, x.data(), 8);

                local_sum = _mm512_fmadd_pd(mat_vals, x_vals,local_sum);
            }
            // Use mapping from sorted_row back to original row index
            int original_row = sorted_to_row_index[row];
            for (; i < segment_max_non_zero; i++) {
                int index = segment_starts[segment] + offset * segment_max_non_zero + i;
                int temp_col = col_indices[index];
                result[original_row] += values[index] * x[temp_col];
            }

            result[original_row] += _mm512_reduce_add_pd(local_sum);
        }
    }


return result;
}
#endif

#ifdef risc
std::vector<double> result(rows, 0.0);
int num_segments = (rows + segment_size - 1) / segment_size;
    #ifdef omp
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int segment = 0; segment < num_segments; segment++) {
        //int segment_max_non_zero = values[segment].size() / segment_size;
        int segment_max_non_zero = (segment_starts[segment + 1] - segment_starts[segment]) / segment_size;
        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;
            
            // вычисление максимального количества элементов double
            size_t vlmax = vsetvlmax_e64m1();

            // создание вектора аккумулятора
            vfloat64m1_t vec_sum = vfmv_v_f_f64m1(0.0, vlmax);
                    // вектор с нулями
            vfloat64m1_t v_zero = vfmv_v_f_f64m1(0.0, vlmax);
            double scalar_sum = 0.0;
            int i=0;
            int k = segment_max_non_zero;
            for (size_t vl; k > 0; k -= vl, i += vl) {
                int index = segment_starts[segment] + offset * segment_max_non_zero + i;
                vl = __riscv_vsetvl_e64m1(k);
                int index = offset * segment_max_non_zero + i;

                vuint32mf2_t vec_indices32 = vle32_v_u32mf2(&col_indices[index], vl);
                vec_indices32 = vsll_vx_u32mf2(vec_indices32, 3 , vl);
                vfloat64m1_t x_vals = vluxei32_v_f64m1(&x[0], vec_indices32, vl);

                vfloat64m1_t mat_vals = vle64_v_f64m1(&values[index], vl);
                vec_sum = vfmacc_vv_f64m1(vec_sum, mat_vals, x_vals, vl);

                
            }
            // сложение всех элементов вектора
            vfloat64m1_t v_reduce_sum = vfredusum_vs_f64m1_f64m1(vec_sum, v_zero, vlmax);
            // Выгрузка значения из вектора v_reduce_sum в переменную scalar_sum
            vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
            // Use mapping from sorted_row back to original row index
            int original_row = sorted_to_row_index[row];
            result[original_row] = temp_for_row;
        }
    }
    return result;
    }
#endif
