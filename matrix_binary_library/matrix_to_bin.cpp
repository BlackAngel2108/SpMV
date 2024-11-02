#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

// ��������� ��� �������� ��������� �������
struct MatrixElement {
    int row;
    int col;
    double value;
};

// ������� ��� ������ ������� �� ����� ������� .mtx
std::vector<MatrixElement> readMatrixFromMTX(const std::string& filename, int& rows, int& cols) {
    std::ifstream file(filename);
    std::vector<MatrixElement> matrix;

    if (!file.is_open()) {
        std::cerr << "������ �������� ����� " << filename << std::endl;
        return matrix;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == '%') continue;  // ������� ������������

        std::istringstream iss(line);
        if (rows == 0 && cols == 0) {
            iss >> rows >> cols;  // ���������� �����������
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

// ������� ��� ������ ������� � �������� ����
void writeMatrixToBinary(const std::string& filename, const std::vector<MatrixElement>& matrix, int rows, int cols) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "������ �������� ����� ��� ������ " << filename << std::endl;
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

// ������� ��� ������ ������� �� ��������� �����
std::vector<MatrixElement> readMatrixFromBinary(const std::string& filename, int& rows, int& cols) {
    std::ifstream infile(filename, std::ios::binary);
    std::vector<MatrixElement> matrix;

    if (!infile.is_open()) {
        std::cerr << "������ �������� ����� ��� ������ " << filename << std::endl;
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

    // ������ �� .mtx �����
    std::vector<MatrixElement> matrix = readMatrixFromMTX("ash958.mtx", rows, cols);
    if (matrix.empty()) {
        return 1;  // ������ ��� ������ �����
    }

    // ������ � �������� ����
    writeMatrixToBinary("matrix.bin", matrix, rows, cols);

    // ������ �� ��������� �����
    int readRows = 0, readCols = 0;
    std::vector<MatrixElement> readMatrix = readMatrixFromBinary("matrix.bin", readRows, readCols);

    // ����� ��� ��������
    std::cout << "��������� ������� (" << readRows << "x" << readCols << "):" << std::endl;
    for (const auto& elem : readMatrix) {
        std::cout << "������: " << elem.row << ", �������: " << elem.col << ", ��������: " << elem.value << std::endl;
    }

    return 0;
}
