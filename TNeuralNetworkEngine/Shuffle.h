#pragma once
#include "Sample.h"
#include <vector>

using namespace std;
class Shuffle
{
  public:
    Shuffle(size_t size_);
    ~Shuffle();

    void GetShuffledData(const vector<Sample *> &samples, int count, vector<Sample *> &shuffledData);

  private:
    vector<size_t> randomIndeces; // 随机索引池
    size_t curIndex = 0;
};
