#include "Sparse_matrix.h"

COO_matrix::COO_matrix(std::string filename) {
    std::ifstream infile(filename, std::ios::binary);
    try {
        if (!infile.is_open()) {
            throw std::runtime_error("Error opening file for reading: " + filename);
        }
    } 
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
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
#pragma omp parallel for
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
    col_indices.resize(rows, std::vector<uint32_t>(max_non_zero, 0));

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
#endif
#ifdef risc
        std::vector<double> result(rows, 0.0);
        // вычисление максимального количества элементов double
        size_t vlmax = __riscv_vsetvlmax_e64m1();

        #ifdef omp
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (int row = 0; row < rows; ++row) {
            // вектор с нулями
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
            
            double scalar_sum = 0.0;

            // создание вектора аккумулятора и заполнение 0.0
            vfloat64m1_t vec_sum = __riscv_vfmv_v_f_f64m1(0.0, vlmax);

            int i = 0;
            int k = max_non_zero;
            for (size_t vl; k>0; k-=vl, i += vl) {
                vl = __riscv_vsetvl_e64m1(k);

                // загрузка индексов столбцов (32-битные unsigned int), грузим в половину вектора
                vuint32mf2_t vec_indices = __riscv_vle32_v_u32mf2(&col_indices[row][i], vl);
                vec_indices = __riscv_vsll_vx_u32mf2(vec_indices, 3 , vl);

                // загрузка x_val элементов по индексам
                vfloat64m1_t x_vals = __riscv_vluxei32_v_f64m1(&x[0], vec_indices, vl);

                // загрузка элементов из матрицы
                vfloat64m1_t mat_vals = __riscv_vle64_v_f64m1(&values[row][i], vl);

                // умножение и сложение с накоплением (FMA)
                vec_sum = __riscv_vfmacc_vv_f64m1(vec_sum, mat_vals, x_vals, vl);
            }
            // сложение всех элементов вектора
            vfloat64m1_t v_reduce_sum = __riscv_vfredusum_vs_f64m1_f64m1(vec_sum, v_zero, vlmax);
            // Выгрузка значения из вектора v_reduce_sum в переменную scalar_sum
            __riscv_vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
            result[row] = scalar_sum;
        }

        return result;        
#endif
#ifdef riscLMUL2
        std::vector<double> result(rows, 0.0);
        // вычисление максимального количества элементов double
        size_t vlmax = __riscv_vsetvlmax_e64m2();

        #ifdef omp
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (int row = 0; row < rows; ++row) {
            // вектор с нулями
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
            
            double scalar_sum = 0.0;

            // создание вектора аккумулятора и заполнение 0.0
            vfloat64m2_t vec_sum = __riscv_vfmv_v_f_f64m2(0.0, vlmax);

            int i = 0;
            int k = max_non_zero;
            for (size_t vl; k>0; k-=vl, i += vl) {
                vl = __riscv_vsetvl_e64m2(k);

                // загрузка индексов столбцов (32-битные unsigned int), грузим в половину вектора
                vuint32m1_t vec_indices = __riscv_vle32_v_u32m1(&col_indices[row][i], vl);
                vec_indices = __riscv_vsll_vx_u32m1(vec_indices, 3 , vl);

                // загрузка x_val элементов по индексам
                vfloat64m2_t x_vals = __riscv_vluxei32_v_f64m2(&x[0], vec_indices, vl);

                // загрузка элементов из матрицы
                vfloat64m2_t mat_vals = __riscv_vle64_v_f64m2(&values[row][i], vl);

                // умножение и сложение с накоплением (FMA)
                vec_sum = __riscv_vfmacc_vv_f64m2(vec_sum, mat_vals, x_vals, vl);
            }
            // сложение всех элементов вектора
            vfloat64m1_t v_reduce_sum = __riscv_vfredusum_vs_f64m2_f64m1(vec_sum, v_zero, vlmax);
            // Выгрузка значения из вектора v_reduce_sum в переменную scalar_sum
            __riscv_vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
            result[row] = scalar_sum;
        }

        return result;        
#endif
#ifdef riscLMUL4
        std::vector<double> result(rows, 0.0);
        // вычисление максимального количества элементов double
        size_t vlmax = __riscv_vsetvlmax_e64m4();

        #ifdef omp
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (int row = 0; row < rows; ++row) {
            // вектор с нулями
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
            
            double scalar_sum = 0.0;

            // создание вектора аккумулятора и заполнение 0.0
            vfloat64m4_t vec_sum = __riscv_vfmv_v_f_f64m4(0.0, vlmax);

            int i = 0;
            int k = max_non_zero;
            for (size_t vl; k>0; k-=vl, i += vl) {
                vl = __riscv_vsetvl_e64m4(k);

                // загрузка индексов столбцов (32-битные unsigned int), грузим в половину вектора
                vuint32m2_t vec_indices = __riscv_vle32_v_u32m2(&col_indices[row][i], vl);
                vec_indices = __riscv_vsll_vx_u32m2(vec_indices, 3 , vl);

                // загрузка x_val элементов по индексам
                vfloat64m4_t x_vals = __riscv_vluxei32_v_f64m4(&x[0], vec_indices, vl);

                // загрузка элементов из матрицы
                vfloat64m4_t mat_vals = __riscv_vle64_v_f64m4(&values[row][i], vl);

                // умножение и сложение с накоплением (FMA)
                vec_sum = __riscv_vfmacc_vv_f64m4(vec_sum, mat_vals, x_vals, vl);
            }
            // сложение всех элементов вектора
            vfloat64m1_t v_reduce_sum = __riscv_vfredusum_vs_f64m4_f64m1(vec_sum, v_zero, vlmax);
            // Выгрузка значения из вектора v_reduce_sum в переменную scalar_sum
            __riscv_vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
            result[row] = scalar_sum;
        }

        return result;        
#endif
#ifdef riscLMUL8
        std::vector<double> result(rows, 0.0);
        // вычисление максимального количества элементов double
        size_t vlmax = __riscv_vsetvlmax_e64m8();

        #ifdef omp
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (int row = 0; row < rows; ++row) {
            // вектор с нулями
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
            
            double scalar_sum = 0.0;

            // создание вектора аккумулятора и заполнение 0.0
            vfloat64m8_t vec_sum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);

            int i = 0;
            int k = max_non_zero;
            for (size_t vl; k>0; k-=vl, i += vl) {
                vl = __riscv_vsetvl_e64m8(k);

                // загрузка индексов столбцов (32-битные unsigned int), грузим в половину вектора
                vuint32m4_t vec_indices = __riscv_vle32_v_u32m4(&col_indices[row][i], vl);
                vec_indices = __riscv_vsll_vx_u32m4(vec_indices, 3 , vl);

                // загрузка x_val элементов по индексам
                vfloat64m8_t x_vals = __riscv_vluxei32_v_f64m8(&x[0], vec_indices, vl);

                // загрузка элементов из матрицы
                vfloat64m8_t mat_vals = __riscv_vle64_v_f64m8(&values[row][i], vl);

                // умножение и сложение с накоплением (FMA)
                vec_sum = __riscv_vfmacc_vv_f64m8(vec_sum, mat_vals, x_vals, vl);
            }
            // сложение всех элементов вектора
            vfloat64m1_t v_reduce_sum = __riscv_vfredusum_vs_f64m8_f64m1(vec_sum, v_zero, vlmax);
            // Выгрузка значения из вектора v_reduce_sum в переменную scalar_sum
            __riscv_vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
            result[row] = scalar_sum;
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
    col_indices.resize(num_segments, std::vector<unsigned int>(max_non_zero * segment_size, 0));

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
                temp_for_row += values[segment][index] * x[col_indices[segment][index]];
            }
            result[row] = temp_for_row;
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
            
            // вычисление максимального количества элементов double
            size_t vlmax = __riscv_vsetvlmax_e64m1();

            // создание вектора аккумулятора
            vfloat64m1_t vec_sum = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
                    // вектор с нулями
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
            double scalar_sum = 0.0;
            int i=0;
            int k = segment_max_non_zero;
            for (size_t vl; k > 0; k -= vl, i += vl) {
                vl = __riscv_vsetvl_e64m1(k);
                int index = offset * segment_max_non_zero + i;

                vuint32mf2_t vec_indices32 = __riscv_vle32_v_u32mf2(&col_indices[segment][index], vl);
                vec_indices32 = __riscv_vsll_vx_u32mf2(vec_indices32, 3 , vl);
                vfloat64m1_t x_vals = __riscv_vluxei32_v_f64m1(&x[0], vec_indices32, vl);

                vfloat64m1_t mat_vals = __riscv_vle64_v_f64m1(&values[segment][index], vl);
                vec_sum = __riscv_vfmacc_vv_f64m1(vec_sum, mat_vals, x_vals, vl);

                
            }
            // сложение всех элементов вектора
            vfloat64m1_t v_reduce_sum = __riscv_vfredusum_vs_f64m1_f64m1(vec_sum, v_zero, vlmax);
            // Выгрузка значения из вектора v_reduce_sum в переменную scalar_sum
            __riscv_vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
            result[row] = scalar_sum;
            // // Переводим значение [0] вектора суммы в скалярную величину
            // double scalar_sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
            // result[row] = scalar_sum;
        }
    }
    return result;
}
#endif
#ifdef riscLMUL2
    #ifdef omp
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int segment = 0; segment < num_segments; segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;
            
            // вычисление максимального количества элементов double
            size_t vlmax = __riscv_vsetvlmax_e64m2();

            // создание вектора аккумулятора
            vfloat64m2_t vec_sum = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
                    // вектор с нулями
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
            double scalar_sum = 0.0;
            int i=0;
            int k = segment_max_non_zero;
            for (size_t vl; k > 0; k -= vl, i += vl) {
                vl = __riscv_vsetvl_e64m2(k);
                int index = offset * segment_max_non_zero + i;

                vuint32m1_t vec_indices32 = __riscv_vle32_v_u32m1(&col_indices[segment][index], vl);
                vec_indices32 = __riscv_vsll_vx_u32m1(vec_indices32, 3 , vl);
                vfloat64m2_t x_vals = __riscv_vluxei32_v_f64m2(&x[0], vec_indices32, vl);

                vfloat64m2_t mat_vals = __riscv_vle64_v_f64m2(&values[segment][index], vl);
                vec_sum = __riscv_vfmacc_vv_f64m2(vec_sum, mat_vals, x_vals, vl);
  
            }
            // сложение всех элементов вектора
            vfloat64m1_t v_reduce_sum = __riscv_vfredusum_vs_f64m2_f64m1(vec_sum, v_zero, vlmax);
            // Выгрузка значения из вектора v_reduce_sum в переменную scalar_sum
            __riscv_vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
            result[row] = scalar_sum;
            // // Переводим значение [0] вектора суммы в скалярную величину
            // double scalar_sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
            // result[row] = scalar_sum;
        }
    }
    return result;
}
#endif
#ifdef riscLMUL4
    #ifdef omp
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int segment = 0; segment < num_segments; segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;
            
            // вычисление максимального количества элементов double
            size_t vlmax = __riscv_vsetvlmax_e64m4();

            // создание вектора аккумулятора
            vfloat64m4_t vec_sum = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
                    // вектор с нулями
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
            double scalar_sum = 0.0;
            int i=0;
            int k = segment_max_non_zero;
            for (size_t vl; k > 0; k -= vl, i += vl) {
                vl = __riscv_vsetvl_e64m4(k);
                int index = offset * segment_max_non_zero + i;

                vuint32m2_t vec_indices32 = __riscv_vle32_v_u32m2(&col_indices[segment][index], vl);
                vec_indices32 = __riscv_vsll_vx_u32m2(vec_indices32, 3 , vl);
                vfloat64m4_t x_vals = __riscv_vluxei32_v_f64m4(&x[0], vec_indices32, vl);

                vfloat64m4_t mat_vals = __riscv_vle64_v_f64m4(&values[segment][index], vl);
                vec_sum = __riscv_vfmacc_vv_f64m4(vec_sum, mat_vals, x_vals, vl);
  
            }
            // сложение всех элементов вектора
            vfloat64m1_t v_reduce_sum = __riscv_vfredusum_vs_f64m4_f64m1(vec_sum, v_zero, vlmax);
            // Выгрузка значения из вектора v_reduce_sum в переменную scalar_sum
            __riscv_vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
            result[row] = scalar_sum;
            // // Переводим значение [0] вектора суммы в скалярную величину
            // double scalar_sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
            // result[row] = scalar_sum;
        }
    }
    return result;
}
#endif
#ifdef riscLMUL8
    #ifdef omp
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int segment = 0; segment < num_segments; segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;
            
            // вычисление максимального количества элементов double
            size_t vlmax = __riscv_vsetvlmax_e64m8();

            // создание вектора аккумулятора
            vfloat64m8_t vec_sum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
                    // вектор с нулями
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
            double scalar_sum = 0.0;
            int i=0;
            int k = segment_max_non_zero;
            for (size_t vl; k > 0; k -= vl, i += vl) {
                vl = __riscv_vsetvl_e64m8(k);
                int index = offset * segment_max_non_zero + i;

                vuint32m4_t vec_indices32 = __riscv_vle32_v_u32m4(&col_indices[segment][index], vl);
                vec_indices32 = __riscv_vsll_vx_u32m4(vec_indices32, 3 , vl);
                vfloat64m8_t x_vals = __riscv_vluxei32_v_f64m8(&x[0], vec_indices32, vl);

                vfloat64m8_t mat_vals = __riscv_vle64_v_f64m8(&values[segment][index], vl);
                vec_sum = __riscv_vfmacc_vv_f64m8(vec_sum, mat_vals, x_vals, vl);
  
            }
            // сложение всех элементов вектора
            vfloat64m1_t v_reduce_sum = __riscv_vfredusum_vs_f64m8_f64m1(vec_sum, v_zero, vlmax);
            // Выгрузка значения из вектора v_reduce_sum в переменную scalar_sum
            __riscv_vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
            result[row] = scalar_sum;
            // // Переводим значение [0] вектора суммы в скалярную величину
            // double scalar_sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
            // result[row] = scalar_sum;
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
    sorted_to_row_index.resize(rows);
    for (int i = 0; i < rows; ++i) {
        sorted_to_row_index[row_to_sorted_index[i]] = i;
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
        col_indices[segment].resize(static_cast<size_t>(segment_size) * static_cast<size_t>(segment_max_non_zero[segment]), 0);
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
                temp_for_row += values[segment][index] * x[col_indices[segment][index]];
            }
            // Use mapping from sorted_row back to original row index
            int original_row = sorted_to_row_index[row];
            result[original_row] = temp_for_row;
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
            
            // вычисление максимального количества элементов double
            size_t vlmax = __riscv_vsetvlmax_e64m1();

            // создание вектора аккумулятора
            vfloat64m1_t vec_sum = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
                    // вектор с нулями
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
            double scalar_sum = 0.0;
            int i=0;
            int k = segment_max_non_zero;
            for (size_t vl; k > 0; k -= vl, i += vl) {
                vl = __riscv_vsetvl_e64m1(k);
                int index = offset * segment_max_non_zero + i;

                vuint32mf2_t vec_indices32 = __riscv_vle32_v_u32mf2(&col_indices[segment][index], vl);
                vec_indices32 = __riscv_vsll_vx_u32mf2(vec_indices32, 3 , vl);
                vfloat64m1_t x_vals = __riscv_vluxei32_v_f64m1(&x[0], vec_indices32, vl);

                vfloat64m1_t mat_vals = __riscv_vle64_v_f64m1(&values[segment][index], vl);
                vec_sum = __riscv_vfmacc_vv_f64m1(vec_sum, mat_vals, x_vals, vl);

                
            }
            // сложение всех элементов вектора
            vfloat64m1_t v_reduce_sum = __riscv_vfredusum_vs_f64m1_f64m1(vec_sum, v_zero, vlmax);
            // Выгрузка значения из вектора v_reduce_sum в переменную scalar_sum
            __riscv_vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
            // Use mapping from sorted_row back to original row index
            int original_row = sorted_to_row_index[row];
            result[original_row] = temp_for_row;
            // // Переводим значение [0] вектора суммы в скалярную величину
            // double scalar_sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
            // result[row] = scalar_sum;
        }
    }
    return result;
}
#endif
#ifdef riscLMUL2
    #ifdef omp
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int segment = 0; segment < num_segments; segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;
            
            // вычисление максимального количества элементов double
            size_t vlmax = __riscv_vsetvlmax_e64m2();

            // создание вектора аккумулятора
            vfloat64m2_t vec_sum = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
                    // вектор с нулями
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
            double scalar_sum = 0.0;
            int i=0;
            int k = segment_max_non_zero;
            for (size_t vl; k > 0; k -= vl, i += vl) {
                vl = __riscv_vsetvl_e64m2(k);
                int index = offset * segment_max_non_zero + i;

                vuint32m1_t vec_indices32 = __riscv_vle32_v_u32m1(&col_indices[segment][index], vl);
                vec_indices32 = __riscv_vsll_vx_u32m1(vec_indices32, 3 , vl);
                vfloat64m2_t x_vals = __riscv_vluxei32_v_f64m2(&x[0], vec_indices32, vl);

                vfloat64m2_t mat_vals = __riscv_vle64_v_f64m2(&values[segment][index], vl);
                vec_sum = __riscv_vfmacc_vv_f64m2(vec_sum, mat_vals, x_vals, vl);
  
            }
            // сложение всех элементов вектора
            vfloat64m1_t v_reduce_sum = __riscv_vfredusum_vs_f64m2_f64m1(vec_sum, v_zero, vlmax);
            // Выгрузка значения из вектора v_reduce_sum в переменную scalar_sum
            __riscv_vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
            // Use mapping from sorted_row back to original row index
            int original_row = sorted_to_row_index[row];
            result[original_row] = temp_for_row;
            // // Переводим значение [0] вектора суммы в скалярную величину
            // double scalar_sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
            // result[row] = scalar_sum;
        }
    }
    return result;
}
#endif
#ifdef riscLMUL4
    #ifdef omp
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int segment = 0; segment < num_segments; segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;
            
            // вычисление максимального количества элементов double
            size_t vlmax = __riscv_vsetvlmax_e64m4();

            // создание вектора аккумулятора
            vfloat64m4_t vec_sum = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
                    // вектор с нулями
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
            double scalar_sum = 0.0;
            int i=0;
            int k = segment_max_non_zero;
            for (size_t vl; k > 0; k -= vl, i += vl) {
                vl = __riscv_vsetvl_e64m4(k);
                int index = offset * segment_max_non_zero + i;

                vuint32m2_t vec_indices32 = __riscv_vle32_v_u32m2(&col_indices[segment][index], vl);
                vec_indices32 = __riscv_vsll_vx_u32m2(vec_indices32, 3 , vl);
                vfloat64m4_t x_vals = __riscv_vluxei32_v_f64m4(&x[0], vec_indices32, vl);

                vfloat64m4_t mat_vals = __riscv_vle64_v_f64m4(&values[segment][index], vl);
                vec_sum = __riscv_vfmacc_vv_f64m4(vec_sum, mat_vals, x_vals, vl);
  
            }
            // сложение всех элементов вектора
            vfloat64m1_t v_reduce_sum = __riscv_vfredusum_vs_f64m4_f64m1(vec_sum, v_zero, vlmax);
            // Выгрузка значения из вектора v_reduce_sum в переменную scalar_sum
            __riscv_vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
            // Use mapping from sorted_row back to original row index
            int original_row = sorted_to_row_index[row];
            result[original_row] = temp_for_row;
            // // Переводим значение [0] вектора суммы в скалярную величину
            // double scalar_sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
            // result[row] = scalar_sum;
        }
    }
    return result;
}
#endif
#ifdef riscLMUL8
    #ifdef omp
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int segment = 0; segment < num_segments; segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; offset++) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;
            
            // вычисление максимального количества элементов double
            size_t vlmax = __riscv_vsetvlmax_e64m8();

            // создание вектора аккумулятора
            vfloat64m8_t vec_sum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
                    // вектор с нулями
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
            double scalar_sum = 0.0;
            int i=0;
            int k = segment_max_non_zero;
            for (size_t vl; k > 0; k -= vl, i += vl) {
                vl = __riscv_vsetvl_e64m8(k);
                int index = offset * segment_max_non_zero + i;

                vuint32m4_t vec_indices32 = __riscv_vle32_v_u32m4(&col_indices[segment][index], vl);
                vec_indices32 = __riscv_vsll_vx_u32m4(vec_indices32, 3 , vl);
                vfloat64m8_t x_vals = __riscv_vluxei32_v_f64m8(&x[0], vec_indices32, vl);

                vfloat64m8_t mat_vals = __riscv_vle64_v_f64m8(&values[segment][index], vl);
                vec_sum = __riscv_vfmacc_vv_f64m8(vec_sum, mat_vals, x_vals, vl);
  
            }
            // сложение всех элементов вектора
            vfloat64m1_t v_reduce_sum = __riscv_vfredusum_vs_f64m8_f64m1(vec_sum, v_zero, vlmax);
            // Выгрузка значения из вектора v_reduce_sum в переменную scalar_sum
            __riscv_vse64_v_f64m1(&scalar_sum, v_reduce_sum, vlmax);
            // Use mapping from sorted_row back to original row index
            int original_row = sorted_to_row_index[row];
            result[original_row] = temp_for_row;
            // // Переводим значение [0] вектора суммы в скалярную величину
            // double scalar_sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
            // result[row] = scalar_sum;
        }
    }
    return result;
}
#endif