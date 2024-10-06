#pragma once
#include <iostream>
#include <vector>
class Sparse_matrix
{
protected:
    int n,m;
    //std::vector<std::vector<double>> data;
public:
    virtual std::vector<double> SpMV(std::vector<double>){}
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
    std::vector<double> SpMV(std::vector<double>) override;
    std::vector<double> SELLPACK(std::vector<double>);
    std::vector<double> Sell_c_sigma(std::vector<double>);
    std::vector<double> Sell_c_R(std::vector<double>);
    std::vector<double> LAV_1Seg(std::vector<double>);
    std::vector<double> LAV(std::vector<double>);
};
