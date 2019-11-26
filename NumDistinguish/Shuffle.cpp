#include "Shuffle.h"
#include "NumDistinguish.h"
#include <random>
#include <chrono>       // std::chrono::system_clock
#include "Sample.h"

using namespace std;

Shuffle::Shuffle(size_t size_) : size(size_) {
	// obtain a time-based seed:
	seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();

	randomIndeces.resize(size);

	for (size_t i = 0; i < size; i++) { randomIndeces[i] = i; } 

	Random_Shuffle();
}

void Shuffle::Random_Shuffle() {
	std::shuffle(randomIndeces.begin(), randomIndeces.end(), std::default_random_engine(seed));
	curIndex = 0;
}

const vector<size_t>& Shuffle::GetShuffledData(int count) {

	shuffleIndeces.clear();
	shuffleIndeces.resize(count);

	for (int i = 0; i < count; i++) {
		shuffleIndeces[i] = randomIndeces[curIndex++];
		if (curIndex == size - 1) {
			Random_Shuffle();
		}
	}
	return shuffleIndeces;
}

Shuffle::~Shuffle() {
}

