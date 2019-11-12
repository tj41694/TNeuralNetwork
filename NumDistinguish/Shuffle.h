#include <vector>
class Layer;
class Shuffle {
public:
	Shuffle(const std::vector<Layer*>& data);
	std::vector<unsigned int>* GetShuffledData(int count);
	~Shuffle();


private:
	void Random_Shuffle();
private:
	std::vector<unsigned int> indeces;
	const size_t size;
	size_t curIndex;
};
