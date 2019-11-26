#pragma once
#include <vector>
class NeuralMatrix {
public:

	NeuralMatrix(int row_, int colum_, float bias_);
	~NeuralMatrix();

public:
	std::vector<std::vector<double>>	matrix;
	int									row;
	int									column;
	float								bias;

private:
};

