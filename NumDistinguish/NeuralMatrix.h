#pragma once
#include <vector>
using namespace std;
class NeuralMatrix
{
  public:
    NeuralMatrix(int row_, int colum_);
    NeuralMatrix(const NeuralMatrix &neural);
    ~NeuralMatrix();

  public:
    vector<vector<double>> matrix;
    int row;
    int column;
    vector<double> bias;

  private:
};
