#include "Shuffle.h"
#include "NumDistinguish.h"
#include <random>
#include <chrono>       // std::chrono::system_clock
#include "Layer.h"

using namespace std;
Shuffle::Shuffle(const vector<Layer>& datas) : size(datas.size()) {
	indeces.resize(datas.size());
	for (unsigned int i = 0; i < size; i++) { indeces[i] = i; }
	Random_Shuffle();
}

void Shuffle::Random_Shuffle() {
	// obtain a time-based seed:
	unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle(indeces.begin(), indeces.end(), std::default_random_engine(seed));
	curIndex = 0;
}

vector<unsigned int>* Shuffle::GetShuffledData(int count) {
	vector<unsigned int>* dataIndex = new vector<unsigned int>();

	while (dataIndex->size() != count) {
		dataIndex->emplace_back(indeces[curIndex++]);
		if (curIndex == size) {
			Random_Shuffle();
		}
	}
	return dataIndex;
}


Shuffle::~Shuffle() {}
