#include "NeuralMatrix.h"
#include "NumDistinguish.h"
#include "Sample.h"
#include <cstdlib> // Header file needed to use srand and rand
#include <ctime>   // Header file needed to use time

using namespace std;
NeuralMatrix::NeuralMatrix(int row_, int colum_)
    : row(row_), column(colum_)
{
    static bool initial = false;
    if (!initial)
    {
        srand((unsigned int) time(0));
        initial = true;
    }
    bias.resize(row_);
    for (int r = 0; r < row_; r++)
    {
        vector<double> row;
        row.resize(colum_);
        for (int c = 0; c < colum_; c++)
        {
            row[c] = rand() * 2.0 / RAND_MAX - 1.0;
        }
        matrix.emplace_back(row);
        bias[r] = rand() * 2.0 / RAND_MAX - 1.0;
    }
}

NeuralMatrix::NeuralMatrix(const NeuralMatrix &neural)
    : row(neural.row), column(neural.column)
{
    bias.resize(neural.bias.size(), 0);
    for (int r = 0; r < neural.row; r++)
    {
        vector<double> row;
        row.resize(column, 0);
        matrix.emplace_back(row);
    }
}

NeuralMatrix::~NeuralMatrix()
{
}
