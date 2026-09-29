#include <vector>

using namespace std;
class Shuffle
{
  public:
    Shuffle(size_t size_);
    ~Shuffle();

    void GetShuffledData(int count, vector<size_t> &shuffleIndeces);

  private:
    vector<size_t> randomIndeces; // 随机索引池
    size_t curIndex = 0;
};
