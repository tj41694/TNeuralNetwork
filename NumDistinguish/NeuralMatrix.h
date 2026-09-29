#pragma once
#include <vector>
using namespace std;
class NeuralMatrix
{
  public:
    NeuralMatrix(int row_, int colum_, float bias_);
    NeuralMatrix(const NeuralMatrix &neural, bool zeroIze);
    ~NeuralMatrix();

  public:
    vector<vector<double>> matrix;
    int row;
    int column;
    double bias;

  private:
};
