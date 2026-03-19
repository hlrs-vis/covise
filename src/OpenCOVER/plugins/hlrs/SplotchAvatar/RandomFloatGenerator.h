#ifndef COVER_PLUGIN_SPLOTCH_AVATAR_GET_RANDOM_FLOAT_GENERATOR_H
#define COVER_PLUGIN_SPLOTCH_AVATAR_GET_RANDOM_FLOAT_GENERATOR_H

#include <random>

class RandomFloatGenerator
{
    std::random_device m_randomDevice{};
    std::mt19937 m_engine{m_randomDevice()};

public:
    float generate(float start, float end)
    {
        return std::uniform_real_distribution<float>{start, end}(m_engine);
    }
};

#endif // COVER_PLUGIN_SPLOTCH_AVATAR_GET_RANDOM_FLOAT_GENERATOR_H
