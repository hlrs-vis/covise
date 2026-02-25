#pragma once
#include <array>

namespace core::interface
{
struct Color
{
    float r, g, b, a;
    Color()
        : r(1.0f)
        , g(1.0f)
        , b(1.0f)
        , a(1.0f)
    {
    }
    Color(float r, float g, float b, float a)
        : r(r)
        , g(g)
        , b(b)
        , a(a)
    {
    }
    Color(const float rgba[4])
        : r(rgba[0])
        , g(rgba[1])
        , b(rgba[2])
        , a(rgba[3])
    {
    }
    Color(const std::array<float, 4> &rgba)
        : r(rgba[0])
        , g(rgba[1])
        , b(rgba[2])
        , a(rgba[3])
    {
    }

    bool operator==(const Color &other) const
    {
        return r == other.r && g == other.g && b == other.b && a == other.a;
    }
};

class IColorable
{
public:
    IColorable() = default;
    virtual ~IColorable() = default;
    IColorable(const IColorable &) = delete;
    IColorable &operator=(const IColorable &) = delete;
    virtual void applyColor(const Color &color) = 0;
};
} // namespace core::interface
