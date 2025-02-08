#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <ctime>
#include <sstream>
#include <unordered_set>
#include "Sparse_matrix.h"

namespace fs = std::filesystem;

double SPMV_time(std::vector<double>& ans, Sparse_matrix& matrix, std::vector<double>& vector) {
    double startTime = omp_get_wtime();
    ans = matrix.SpMV(vector);
    double endTime = omp_get_wtime();
    return (double)(endTime - startTime);// / CLOCKS_PER_SEC;
}

// Function to read already processed matrices from a file
std::unordered_set<std::string> read_processed_matrices(const std::string& filename) {
    std::unordered_set<std::string> processed_matrices;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        return processed_matrices; // If the file does not exist, return an empty set
    }

    std::string line;
    bool is_first_line = true; // Skip header
    while (std::getline(infile, line)) {
        if (is_first_line) {
            is_first_line = false;
            continue; // Skip the first line (header)
        }
        if (line.empty()) {
            continue; // Skip empty lines
        }
        std::istringstream iss(line);
        std::string matrix_name;
        if (std::getline(iss, matrix_name, ',')) {
            processed_matrices.insert(matrix_name);
        }
    }
    infile.close();
    return processed_matrices;
}

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "Russian");
    int skip_files;
    int target_file;

    // Check if command line arguments are provided
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <number_of_files_to_skip> <target_file_number>" << std::endl;
        return 1;
    }
    skip_files = std::stoi(argv[1]);
    target_file = std::stoi(argv[2]);

    // Parallel block for testing
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::cout << "Hello from thread " << thread_id << std::endl;
    }

    std::string bin_folder = "../../bin_matrix";
    std::string output_file = "../../results.csv";    // Output file name

    // Check if the file results.csv exists
    bool file_exists = fs::exists(output_file);

    std::ofstream outfile;
    if (file_exists) {
        // Open the file for appending data
        outfile.open(output_file, std::ios::app);
    }
    else {
        // Create a new file and write the header
        outfile.open(output_file);
        outfile << "Matrix," << "COO_time," << " CSR_time," << " DIAG_time," << " ELLPack_time," << "ELL_C_time," << "SELL_C_sigma_time," << std::endl;
    }

    if (!outfile.is_open()) {
        std::cerr << "Error: unable to open file " << output_file << " for writing." << std::endl;
        return 1;
    }

    // Read already processed matrices
    std::unordered_set<std::string> processed_matrices = read_processed_matrices(output_file);

    // If the file exists, check the number of already written rows
    int existing_rows = 0;
    if (file_exists) {
        std::ifstream infile(output_file);
        std::string line;
        while (std::getline(infile, line)) {
            existing_rows++;
        }
        infile.close();
    }

    // If the number of rows is less than skip_files, add the missing rows
    if (existing_rows < skip_files + 1) { // +1 for the header
        int missing_rows = skip_files + 1 - existing_rows;
        for (int i = 0; i < missing_rows; ++i) {
            outfile << ", , , , , " << std::endl; // Write empty rows
        }
    }

    if (!fs::exists(bin_folder) || !fs::is_directory(bin_folder)) {
        std::cerr << "Error: folder " << bin_folder << " does not exist or is not a directory." << std::endl;
        return 1;
    }

    // Iterate over all files in the folder
    int file_count = 0;
    for (const auto& entry : fs::directory_iterator(bin_folder)) {
        if (entry.is_regular_file()) { // Check if it is a file
            file_count++;

            // Skip files until skip_files
            if (file_count <= skip_files) {
                std::cout << "Skipping file: " << entry.path().string() << std::endl;
                continue;
            }

            // If target_file != -1, process only the specified file
            if (target_file != -1 && file_count != target_file) {
                std::cout << "Skipping file: " << entry.path().string() << std::endl;
                continue;
            }

            std::string filename = entry.path().string(); // Full path to the file
            std::string base_name = entry.path().stem().string(); // File name without extension

            // Check if the matrix has already been processed
            if (processed_matrices.find(base_name) != processed_matrices.end()) {
                std::cout << "Matrix " << base_name << " already processed. Skipping." << std::endl;
                continue; // Skip this file
            }

            std::cout << "Reading file: " << filename << std::endl;
            COO_matrix coo_matrix(filename);
            int size = coo_matrix.get_cols();
            std::vector<double> b(size, 1);
            std::vector<double> ans(size);
            std::cout << "    COO_matrix: " << filename << std::endl;
            double t1 = SPMV_time(ans, coo_matrix, b);

            std::cout << "    CSR_matrix: " << filename << std::endl;
            CSR_matrix csr_matrix(filename);
            double t2 = SPMV_time(ans, csr_matrix, b);

            std::cout << "    DIAG_matrix: " << filename << std::endl;
            DIAG_matrix diag_matrix(filename);
            double t3 = SPMV_time(ans, diag_matrix, b);

            double t4 = 0;
            try {
                std::cout << "   ELLPack_matrix: " << filename << std::endl;
                ELLPack_matrix ellpack_matrix(filename);
                t4 = SPMV_time(ans, ellpack_matrix, b);
            }
            catch (...) {
                t4 = 0;
                std::cout << "ELLPACK ERROR: " << std::endl;
            }

            double t5 = 0;
            try {
                std::cout << "    SELL_C_matrix: " << filename << std::endl;
                SELL_C_matrix sell_c_matrix(filename, 4);
                t5 = SPMV_time(ans, sell_c_matrix, b);
            }
            catch (...) {
                t5 = 0;
                std::cout << "SELL_C ERROR: " << std::endl;
            }

            double t6 = 0;

            outfile << base_name << ", " << t1 << ", " << t2 << ", " << t3 << "," << t4 << ", " << t5 << ", " << t6 << std::endl;
        }
    }
    outfile.close();
    std::cout << "Processing complete. Results written to file " << output_file << std::endl;
    return 0;
}
