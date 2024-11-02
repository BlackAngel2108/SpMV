#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

// Структура для хранения элементов матрицы
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
    while (std::getline(file, line)) {
        if (line[0] == '%') continue;  // Пропуск комментариев

        std::istringstream iss(line);
        if (rows == 0 && cols == 0) {
            iss >> rows >> cols;  // Считывание размерности
            continue;
        }

        int row, col;
        double value;
        iss >> row >> col >> value;
        matrix.push_back({ row, col, value });
    }
    file.close();
    return matrix;
}

// Функция для записи матрицы в бинарный файл
void writeMatrixToBinary(const std::string& filename, const std::vector<MatrixElement>& matrix, int rows, int cols) {
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
        outfile.write(reinterpret_cast<const char*>(&elem), sizeof(elem));
    }
    outfile.close();
}

// Функция для чтения матрицы из бинарного файла
std::vector<MatrixElement> readMatrixFromBinary(const std::string& filename, int& rows, int& cols) {
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
        infile.read(reinterpret_cast<char*>(&matrix[i]), sizeof(matrix[i]));
    }

    infile.close();
    return matrix;
}

int main() {
    setlocale(LC_ALL, "Russian");
    int rows = 0, cols = 0;

    // Чтение из .mtx файла
    std::vector<MatrixElement> matrix = readMatrixFromMTX("ash958.mtx", rows, cols);
    if (matrix.empty()) {
        return 1;  // Ошибка при чтении файла
    }

    // Запись в бинарный файл
    writeMatrixToBinary("matrix.bin", matrix, rows, cols);

    // Чтение из бинарного файла
    int readRows = 0, readCols = 0;
    std::vector<MatrixElement> readMatrix = readMatrixFromBinary("matrix.bin", readRows, readCols);

    // Вывод для проверки
    std::cout << "Считанная матрица (" << readRows << "x" << readCols << "):" << std::endl;
    for (const auto& elem : readMatrix) {
        std::cout << "Строка: " << elem.row << ", Колонка: " << elem.col << ", Значение: " << elem.value << std::endl;
    }

    return 0;
}
