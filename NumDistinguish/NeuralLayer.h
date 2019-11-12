#pragma once
class Layer;
class NeuralLayer {
public:
	NeuralLayer(int row_, int colum_, float bias_);
	Layer* MatrixMultiply(float* data, unsigned int length);
	~NeuralLayer();
private:
	float* matrix;
	int row;
	int column;
	float* bias;
};

