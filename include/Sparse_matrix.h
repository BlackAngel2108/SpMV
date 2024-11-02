#pragma once
#include <iostream>
#include <vector>
class Sparse_matrix
{
protected:
    int n,m;
    //std::vector<std::vector<double>> data;
public:
    virtual std::vector<double> SpMV(std::vector<double>&){}
};

class CSR_matrix : public Sparse_matrix
{
private:
    std::vector<double> data; //no zero elements
    std::vector<int> indices; //colums
    std::vector<int> indptr; //index of element in data in i line
public:
    CSR_matrix(std::vector<double> data_, std::vector<int> indices_, std::vector<int> indptr_):
        data(data_), indices(indices_), indptr(indptr_){}
    std::vector<double> SpMV(std::vector<double>&) override;
    std::vector<double> SELLPACK(std::vector<double>);
    std::vector<double> Sell_c_sigma(std::vector<double>);
    std::vector<double> Sell_c_R(std::vector<double>);
    std::vector<double> LAV_1Seg(std::vector<double>);
    std::vector<double> LAV(std::vector<double>);
};

class COO_matrix : public Sparse_matrix
{
private:
    std::vector<double> data;
    std::vector<int> rows; 
    std::vector<int> colums; 
public:
    COO_matrix(std::vector<double> data_, std::vector<int> rows_, std::vector<int> colums_) :
        data(data_), rows(rows_), colums(colums_) {}
    std::vector<double> SpMV(std::vector<double>&) override;
};


class DIA_matrix : public Sparse_matrix
{
private:
    std::vector<std::vector<double>> diagonals;
    std::vector<int> offsets;
public:
    DIA_matrix(std::vector<std::vector<double>> diagonals_, std::vector<int> offsets_) :
        diagonals(diagonals_), offsets(offsets_) {}
    std::vector<double> SpMV(std::vector<double>&) override;
};

class ELL_matrix : public Sparse_matrix
{
private:
    std::vector<double> data;
    std::vector<int> col_indices;
    std::vector<int> row_lengths;
public:
    ELL_matrix(std::vector<double> data_, std::vector<int> col_indices_, std::vector<int> row_lengths_) :
        data(data_), col_indices(col_indices_), row_lengths(row_lengths_) {}
    std::vector<double> SpMV(std::vector<double>&) override;
};

class HYB_matrix : public Sparse_matrix //Hybrid
{
private:
    std::vector<double> data;
    std::vector<int> indices;
    std::vector<int> indptr;
public:
    HYB_matrix(std::vector<double> data_, std::vector<int> indices_, std::vector<int> indptr_) :
        data(data_), indices(indices_), indptr(indptr_) {}
    std::vector<double> SpMV(std::vector<double>&) override;
};

class HDC_matrix : public Sparse_matrix //Hybrid DIA/CSR 
{
private:
    std::vector<double> data;
    std::vector<int> indices;
    std::vector<int> indptr;
public:
    HDC_matrix(std::vector<double> data_, std::vector<int> indices_, std::vector<int> indptr_) :
        data(data_), indices(indices_), indptr(indptr_) {}
    std::vector<double> SpMV(std::vector<double>&) override;
};