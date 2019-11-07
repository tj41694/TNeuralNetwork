#pragma once
struct DataElem;
class NeuralLayer {
public:
	NeuralLayer(int row_, int colum_, float bias_);
	DataElem Multiply(float* data, unsigned int length);
	~NeuralLayer();
private:
	float* matrix;
	int row;
	int column;
	float* bias;
};

