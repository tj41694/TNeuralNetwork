#include <vector>

using namespace std;
class Shuffle
{
  public:
    Shuffle(size_t size_);
    ~Shuffle();

    const vector<size_t> &GetShuffledData(int count);

  private:
    void Random_Shuffle();

  private:
    vector<size_t> randomIndeces;  // 随机索引池
    vector<size_t> shuffleIndeces; // 供返回的索引池
    const size_t size;
    size_t curIndex = 0;
    unsigned seed;
};
