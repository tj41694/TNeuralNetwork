#pragma once
#include <cstdint>
#include <random>
#include <vector>

// 全局随机源。训练里所有的随机性（权重初始化、mini-batch 打乱）都必须走这里，
// 否则即使把 seed 写进 meta.json 也复现不出同一轮训练。
void SeedRandom(uint32_t seed);

// 返回当前使用的种子；若从未显式指定，则用 random_device 派生一个并固定下来，
// 这样调用方总是能拿到"本次实际用的种子"并写进 meta.json。
uint32_t RandomSeed();

std::mt19937 &RandomEngine();

double RandomUniform(double low, double high);

void RandomShuffle(std::vector<size_t> &vec);
