#pragma once
class Sample;
class NeuralLayer {
public:
	NeuralLayer(int row_, int colum_, float bias_);
	Sample* MatrixMultiply(const Sample& ly);
	~NeuralLayer();
private:
	float* matrix;
	int row;
	int column;
	float* bias;
};

