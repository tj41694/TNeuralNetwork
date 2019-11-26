#include <vector>
class Sample;
class Shuffle {
public:

	Shuffle(size_t size_);
	~Shuffle();

	const std::vector<size_t>& GetShuffledData(int count);

private:

	void Random_Shuffle();

private:
	std::vector<size_t>	randomIndeces;	//���������
	std::vector<size_t>	shuffleIndeces;	//�����ص�������
	const size_t				size;
	size_t						curIndex = 0;
	unsigned					seed;
};
