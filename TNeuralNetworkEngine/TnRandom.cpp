#include "TnRandom.h"
#include <algorithm>

namespace
{
struct RandomState
{
    std::mt19937 engine;
    uint32_t seed = 0;
    bool seeded = false;
};

RandomState &State()
{
    static RandomState state;
    return state;
}

uint32_t DeriveSeed()
{
    std::random_device rd;
    uint64_t v = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    v ^= v >> 32;
    // 保证 0 不会被当成"未设置"以外的语义
    return static_cast<uint32_t>(v);
}
} // namespace

void SeedRandom(uint32_t seed)
{
    RandomState &state = State();
    state.seed = seed;
    state.engine.seed(seed);
    state.seeded = true;
}

uint32_t RandomSeed()
{
    RandomState &state = State();
    if (!state.seeded)
    {
        SeedRandom(DeriveSeed());
    }
    return state.seed;
}

std::mt19937 &RandomEngine()
{
    RandomSeed();
    return State().engine;
}

double RandomNormal(double mean, double stddev)
{
    std::normal_distribution<double> dist(mean, stddev);
    return dist(RandomEngine());
}

void RandomShuffle(std::vector<size_t> &vec)
{
    std::shuffle(vec.begin(), vec.end(), RandomEngine());
}
