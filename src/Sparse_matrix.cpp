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
    // Чтение данных параллельно
//#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        int row_id, col_id;
        double value;

        // Каждый поток читает свой блок данных
//#pragma omp critical
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
//#pragma omp parallel for
    for (int row = 0; row < rows; ++row) {
        for (int i = row_pointers[row]; i < row_pointers[row + 1]; ++i) {
            int col = column_indices[i];
            result[row] += values[i] * vec[col];
        }
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
//#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
//#pragma omp atomic
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
    std::vector<double> result(rows, 0.0);

  //omp_set_num_threads(4);
#pragma omp parallel for schedule(dynamic,1000)

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

//#pragma omp parallel for
    for (int segment = 0; segment < values.size(); segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; ++offset) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;

            for (int i = 0; i < segment_max_non_zero; i++) {
                int index = offset * segment_max_non_zero + i;

                if (col_indices[segment][index] != -1) {
//#pragma omp atomic
                    result[row] += values[segment][index] * x[col_indices[segment][index]];
                }
            }
        }
    }
    return result;
}

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

//#pragma omp parallel for
    for (int segment = 0; segment < values.size(); segment++) {
        int segment_max_non_zero = values[segment].size() / segment_size;

        for (int offset = 0; offset < segment_size; ++offset) {
            int row = segment * segment_size + offset;
            if (row >= rows) break;

            for (int i = 0; i < segment_max_non_zero; i++) {
                int index = offset * segment_max_non_zero + i;

                if (col_indices[segment][index] != -1) {
//#pragma omp atomic
                    result[row] += values[segment][index] * x[col_indices[segment][index]];
                }
            }
        }
    }
    return result;
}
