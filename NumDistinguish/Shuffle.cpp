#include "Shuffle.h"
#include <algorithm>
#include <chrono>
#include <random>

using namespace std;

Shuffle::Shuffle(size_t size_)
{
    // obtain a time-based seed:
    auto seed = (unsigned) chrono::system_clock::now().time_since_epoch().count();

    randomIndeces.resize(size_);

    for (size_t i = 0; i < size_; i++)
        randomIndeces[i] = i;

    shuffle(randomIndeces.begin(), randomIndeces.end(), default_random_engine(seed));
}

void Shuffle::GetShuffledData(int count, vector<size_t>& shuffleIndeces)
{
    shuffleIndeces.clear();
    shuffleIndeces.resize(count);
    auto size = randomIndeces.size();

    for (int i = 0; i < count; i++)
    {
        shuffleIndeces[i] = randomIndeces[(curIndex++) % size];
    }
    if (curIndex + count > size)
    {
        auto seed = (unsigned) chrono::system_clock::now().time_since_epoch().count();
        shuffle(randomIndeces.begin(), randomIndeces.end(), default_random_engine(seed));
    }
    curIndex %= size;
}

Shuffle::~Shuffle()
{
}
