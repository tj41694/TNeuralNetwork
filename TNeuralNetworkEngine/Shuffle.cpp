#include "Shuffle.h"
#include "TnRandom.h"
#include <algorithm>

using namespace std;

Shuffle::Shuffle(size_t size_)
{
    // 与权重初始化共用同一个随机源，否则 seed 无法复现整轮训练.
    randomIndeces.resize(size_);

    for (size_t i = 0; i < size_; i++)
        randomIndeces[i] = i;

    RandomShuffle(randomIndeces);
}

void Shuffle::GetShuffledData(const vector<Sample *> &samples, int count, vector<Sample *> &shuffledData)
{
    shuffledData.clear();
    shuffledData.resize(count);
    auto size = randomIndeces.size();

    for (int i = 0; i < count; i++)
    {
        shuffledData[i] = samples[randomIndeces[(curIndex++) % size]];
    }
    if (curIndex + count > size)
    {
        RandomShuffle(randomIndeces);
    }
    curIndex %= size;
}

Shuffle::~Shuffle()
{
}
