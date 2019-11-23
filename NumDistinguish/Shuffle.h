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
	std::vector<unsigned int>	randomIndeces;	//���������
	std::vector<unsigned int>	shuffleIndeces;	//�����ص�������
	const size_t				size;
	size_t						curIndex = 0;
	unsigned					seed;
};
