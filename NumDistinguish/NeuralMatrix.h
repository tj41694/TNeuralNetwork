#pragma once
class Sample;
class NeuralMatrix {
public:

	NeuralMatrix(int row_, int colum_, float bias_);
	~NeuralMatrix();

public:
	float*	matrix;
	int		row;
	int		column;
	float	bias;

private:
};

