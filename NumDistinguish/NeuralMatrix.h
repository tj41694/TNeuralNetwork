#pragma once
#include <vector>
class NeuralMatrix {
public:

	NeuralMatrix(int row_, int colum_, float bias_);
	NeuralMatrix(const NeuralMatrix & neural, bool zeroIze);
	~NeuralMatrix();

public:
	std::vector<std::vector<double>>	matrix;
	int									row;
	int									column;
	double								bias;

private:
};

