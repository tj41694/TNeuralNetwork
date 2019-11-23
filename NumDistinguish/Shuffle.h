#include <vector>
class Sample;
class Shuffle {
public:

	Shuffle(const std::vector<Sample*>& data);
	~Shuffle();

	const std::vector<unsigned int>& GetShuffledData(int count);

private:

	void Random_Shuffle();

private:
	std::vector<unsigned int>	randomIndeces;	//随机索引池
	std::vector<unsigned int>	shuffleIndeces;	//供返回的索引池
	const size_t				size;
	size_t						curIndex = 0;
	unsigned					seed;
};
