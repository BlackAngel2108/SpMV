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

// Функция для чтения уже обработанных матриц из файла
std::unordered_set<std::string> read_processed_matrices(const std::string& filename) {
    std::unordered_set<std::string> processed_matrices;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        return processed_matrices; // Если файл не существует, возвращаем пустой набор
    }

    std::string line;
    bool is_first_line = true; // Пропускаем заголовок
    while (std::getline(infile, line)) {
        if (is_first_line) {
            is_first_line = false;
            continue; // Пропускаем первую строку (заголовок)
        }
        if (line.empty()) {
            continue; // Пропускаем пустые строки
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

    // Проверяем, переданы ли аргументы командной строки
    if (argc < 3) {
        std::cerr << "Использование: " << argv[0] << " <количество_файлов_для_пропуска> <номер_файла_для_обработки>" << std::endl;
        return 1;
    }
    skip_files = std::stoi(argv[1]);
    target_file = std::stoi(argv[2]);

    // Параллельный блок для тестирования
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::cout << "Hello from thread " << thread_id << std::endl;
    }

    std::string bin_folder = "../../bin_matrix";
    std::string output_file = "../../results.csv";    // Имя выходного файла

    // Проверяем, существует ли файл results.csv
    bool file_exists = fs::exists(output_file);

    std::ofstream outfile;
    if (file_exists) {
        // Открываем файл для добавления данных
        outfile.open(output_file, std::ios::app);
    }
    else {
        // Создаем новый файл и записываем заголовок
        outfile.open(output_file);
        outfile << "Matrix," << "COO_time," << " CSR_time," << " DIAG_time," << " ELLPack_time," << "ELL_C_time," << "SELL_C_sigma_time," << std::endl;
    }

    if (!outfile.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << output_file << " для записи." << std::endl;
        return 1;
    }

    // Читаем уже обработанные матрицы
    std::unordered_set<std::string> processed_matrices = read_processed_matrices(output_file);

    // Если файл существует, проверяем количество уже записанных строк
    int existing_rows = 0;
    if (file_exists) {
        std::ifstream infile(output_file);
        std::string line;
        while (std::getline(infile, line)) {
            existing_rows++;
        }
        infile.close();
    }

    // Если количество строк меньше skip_files, добавляем недостающие строки
    if (existing_rows < skip_files + 1) { // +1 для заголовка
        int missing_rows = skip_files + 1 - existing_rows;
        for (int i = 0; i < missing_rows; ++i) {
            outfile << ", , , , , " << std::endl; // Записываем пустые строки
        }
    }

    if (!fs::exists(bin_folder) || !fs::is_directory(bin_folder)) {
        std::cerr << "Ошибка: папка " << bin_folder << " не существует или не является директорией." << std::endl;
        return 1;
    }

    // Проходим по всем файлам в папке
    int file_count = 0;
    for (const auto& entry : fs::directory_iterator(bin_folder)) {
        if (entry.is_regular_file()) { // Проверяем, что это файл
            file_count++;

            // Пропускаем файлы до skip_files
            if (file_count <= skip_files) {
                std::cout << "Пропускаем файл: " << entry.path().string() << std::endl;
                continue;
            }

            // Если target_file != -1, обрабатываем только указанный файл
            if (target_file != -1 && file_count != target_file) {
                std::cout << "Пропускаем файл: " << entry.path().string() << std::endl;
                continue;
            }

            std::string filename = entry.path().string(); // Полный путь к файлу
            std::string base_name = entry.path().stem().string(); // Имя файла без расширения

            // Проверяем, была ли матрица уже обработана
            if (processed_matrices.find(base_name) != processed_matrices.end()) {
                std::cout << "Матрица " << base_name << " уже обработана. Пропускаем." << std::endl;
                continue; // Пропускаем этот файл
            }

            std::cout << "Чтение файла: " << filename << std::endl;
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
    std::cout << "Обработка завершена. Результаты записаны в файл " << output_file << std::endl;
    return 0;
}