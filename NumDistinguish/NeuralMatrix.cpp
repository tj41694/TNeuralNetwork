#include "NeuralMatrix.h"
#include <cstdlib> // Header file needed to use srand and rand
#include <ctime> // Header file needed to use time
#include "NumDistinguish.h"
#include "Sample.h"

NeuralMatrix::NeuralMatrix(int row_, int colum_, float bias_) :row(row_), column(colum_) {
	matrix = new float[(size_t)row_ * colum_];
	bias = bias_;
	srand((unsigned int)time(0));
	for (int i = 0; i < row_ * colum_; i++) {
		float x = (float)(rand() * 2.0 / RAND_MAX - 1.0);
		matrix[i] = x;
	}
}

NeuralMatrix::~NeuralMatrix() {
	delete[] matrix;
}

