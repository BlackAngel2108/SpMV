#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;
struct MatrixElement {
    int row;
    int col;
    double value;
};

// Функция для чтения матрицы из файла формата .mtx
std::vector<MatrixElement> readMatrixFromMTX(const std::string& filename, int& rows, int& cols) {
    std::ifstream file(filename);
    std::vector<MatrixElement> matrix;
    if (!file.is_open()) {
        std::cerr << "Ошибка открытия файла " << filename << std::endl;
        return matrix;
    }

    std::string line;
    bool sizeRead = false;
    while (std::getline(file, line)) {
        if (line[0] == '%') continue; // Пропуск комментариев

        std::istringstream iss(line);
        if (!sizeRead) {
            iss >> rows >> cols; // Считывание размерности
            sizeRead = true;
            continue;
        }

        int row, col;
        double value = 1.0; // Значение по умолчанию
        iss >> row >> col;
        if (!(iss >> value)) {
            value = 1.0;
        }
        matrix.push_back({ row - 1, col - 1, value }); // Преобразование индексов к нулевой основе
    }

    file.close();
    return matrix;
}

// Функция для записи матрицы в бинарный файл COO формат
void writeMatrixToBinaryCOO(const std::string& filename, const std::vector<MatrixElement>& matrix, int rows, int cols) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Ошибка открытия файла для записи " << filename << std::endl;
        return;
    }

    outfile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    outfile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    size_t size = matrix.size();
    outfile.write(reinterpret_cast<const char*>(&size), sizeof(size));

    for (const auto& elem : matrix) {
        outfile.write(reinterpret_cast<const char*>(&elem.row), sizeof(int));
        outfile.write(reinterpret_cast<const char*>(&elem.col), sizeof(int));
        outfile.write(reinterpret_cast<const char*>(&elem.value), sizeof(double));
    }

    outfile.close();
}

// Функция для чтения матрицы из бинарного файла COO формат
std::vector<MatrixElement> readMatrixFromBinaryCOO(const std::string& filename, int& rows, int& cols) {
    std::ifstream infile(filename, std::ios::binary);
    std::vector<MatrixElement> matrix;

    if (!infile.is_open()) {
        std::cerr << "Ошибка открытия файла для чтения " << filename << std::endl;
        return matrix;
    }

    infile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    infile.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    size_t size;
    infile.read(reinterpret_cast<char*>(&size), sizeof(size));

    matrix.resize(size);
    for (size_t i = 0; i < size; ++i) {
        infile.read(reinterpret_cast<char*>(&matrix[i].row), sizeof(matrix[i].row));
        infile.read(reinterpret_cast<char*>(&matrix[i].col), sizeof(matrix[i].col));
        infile.read(reinterpret_cast<char*>(&matrix[i].value), sizeof(matrix[i].value));
    }

    infile.close();
    return matrix;
}

int main() {
    setlocale(LC_ALL, "Russian");

    std::string matrix_folder = "../../matrix_to_git";
    std::string bin_folder = "../../bin_matrix";

    if (!fs::exists(matrix_folder) || !fs::is_directory(matrix_folder)) {
        std::cerr << "Ошибка: папка " << matrix_folder << " не существует или не является директорией." << std::endl;
        return 1;
    }

    if (!fs::exists(bin_folder)) {
        fs::create_directory(bin_folder);
    }

    for (const auto& entry : fs::directory_iterator(matrix_folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mtx") {
            std::string mtx_filename = entry.path().string(); // Полный путь к .mtx файлу
            std::string base_filename = entry.path().stem().string(); // Имя файла без расширения
            std::string bin_filename = bin_folder + "/" + base_filename + ".bin"; // Путь для сохранения бинарного файла
            std::string bin_filename2 = bin_folder + "/" +"big" +"/"+ base_filename + ".bin";
            std::cout << "Чтение файла: " << mtx_filename << std::endl;

            if (fs::exists(bin_filename) || fs::exists(bin_filename2) ){
                std::cout << "Файл " << bin_filename << " уже существует. Пропускаем создание." << std::endl;
                continue;
            }
            int rows = 0, cols = 0;
            std::vector<MatrixElement> matrix = readMatrixFromMTX(mtx_filename, rows, cols);
            if (matrix.empty()) {
                std::cerr << "Ошибка: не удалось прочитать матрицу из файла " << mtx_filename << std::endl;
                continue;
            }

            std::cout << "Преобразование в COO формат и запись в бинарный файл: " << bin_filename << std::endl;

            writeMatrixToBinaryCOO(bin_filename, matrix, rows, cols);
        }
    }

    std::cout << "Обработка завершена." << std::endl;
    return 0;
    }
/*
    // Чтение из .mtx файла
    std::vector<MatrixElement> matrix = readMatrixFromMTX("../../matrix/ash958.mtx", rows, cols);
    if (matrix.empty()) {
        std::cerr << "Ошибка: не удалось прочитать матрицу из файла " << std::endl;
        return 1; // Завершаем программу при ошибке
    }

    // Запись в бинарный файл COO формат
    writeMatrixToBinaryCOO("../../bin_matrix/ash958_coo.bin", matrix, rows, cols);

    // Чтение из бинарного файла COO формат
    int readRows = 0, readCols = 0;
    std::vector<MatrixElement> readMatrix = readMatrixFromBinaryCOO("../../bin_matrix/ash958_coo.bin", readRows, readCols);

    // Вывод для проверки COO формата
    std::cout << "Считанная матрица (COO формат, " << readRows << "x" << readCols << "):" << std::endl;
    for (const auto& elem : readMatrix) {
        std::cout << "Строка: " << elem.row << ", Колонка: " << elem.col << ", Значение: " << elem.value << std::endl;
    }
*/
