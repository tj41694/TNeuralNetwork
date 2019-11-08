#include "NeuralLayer.h"
#include <cstdlib> // Header file needed to use srand and rand
#include <ctime> // Header file needed to use time
#include "NumDistinguish.h"

NeuralLayer::NeuralLayer(int row_, int colum_, float bias_) :row(row_), column(colum_) {
	matrix = new float[(size_t)row_ * colum_];
	bias = new float[row_] { bias_ };
	srand((unsigned int)time(0));
	for (int i = 0; i < row_ * colum_; i++) {
		float x = (float)(rand() * 2.0 / RAND_MAX - 1.0);
		matrix[i] = x;
	}
}

DataElem NeuralLayer::MatrixMultiply(float* data, unsigned int length) {
	if (length != column) {
		printf("err.. Dimension not match..");
	}
	DataElem result;
	result.pixs = new float[row];
	result.size = row;
	for (int r = 0; r < row; r++) { //矩阵乘数据(向量)
		float val = 0;
		for (int c = 0; c < column; c++) {
			val += data[c] * matrix[r * column + c];
		}
		result.pixs[r] = val + bias[r];
	}
	return result;
}
NeuralLayer::~NeuralLayer() {
	delete[] matrix;
}

