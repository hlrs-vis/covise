// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cmath>
#include <ostream>

#ifdef __CUDACC__
#define ASG_FUNC __host__ __device__
#else
#define ASG_FUNC
#endif

namespace asg
{
    struct Vec2f
    {
        float x;
        float y;
    };

    struct Vec3f
    {
        float x;
        float y;
        float z;
    };

    struct Vec4f
    {
        float x;
        float y;
        float z;
        float w;
    };

    struct Vec2i
    {
        int x;
        int y;
    };

    struct Vec3i
    {
        int x;
        int y;
        int z;
    };

    struct Vec4i
    {
        int x;
        int y;
        int z;
        int w;
    };

    struct Box2f
    {
        Vec2f min;
        Vec2f max;
    };

    struct Box3f
    {
        Vec3f min;
        Vec3f max;
    };

    struct Box2i
    {
        Vec2i min;
        Vec2i max;
    };

    struct Box3i
    {
        Vec3i min;
        Vec3i max;
    };

    struct Mat3f
    {
        Vec3f col0;
        Vec3f col1;
        Vec3f col2;
    };

    struct Mat4x3f
    {
        Vec3f col0;
        Vec3f col1;
        Vec3f col2;
        Vec3f col3;
    };

    struct Mat4f
    {
        Vec4f col0;
        Vec4f col1;
        Vec4f col2;
        Vec4f col3;
    };

    enum class Axis
    {
        X,
        Y,
        Z,
    };

    //--- General -----------------------------------------

    template <typename T>
    ASG_FUNC inline T min(T const& a, T const& b)
    {
        return b < a ? b : a;
    }

    template <typename T>
    ASG_FUNC inline T max(T const& a, T const& b)
    {
        return a < b ? b : a;
    }

    template <typename T, typename S>
    ASG_FUNC inline T lerp(T const& a, T const& b, S const& x)
    {
        return (S(1.0f) - x) * a + x * b;
    }

    template <typename T>
    ASG_FUNC inline T clamp(T const& x, T const& a, T const& b)
    {
        return max(a, min(x, b));
    }

    template <typename T>
    ASG_FUNC inline T saturate(T const& x)
    {
        return clamp(x, T(0.0), T(1.0));
    }

    template <typename T>
    ASG_FUNC inline T div_up(T a, T b)
    {
        return (a + b - 1) / b;
    }


    //--- Vec2f -------------------------------------------

    ASG_FUNC inline bool operator==(Vec2f const& a, Vec2f const& b)
    {
        return a.x == b.x && a.y == b.y;
    }

    ASG_FUNC inline bool operator!=(Vec2f const& a, Vec2f const& b)
    {
        return !(a == b);
    }

    ASG_FUNC inline Vec2f operator+(Vec2f const& a, Vec2f const& b)
    {
        return { a.x + b.x, a.y + b.y };
    }

    ASG_FUNC inline Vec2f operator-(Vec2f const& a, Vec2f const& b)
    {
        return { a.x - b.x, a.y - b.y };
    }

    ASG_FUNC inline Vec2f operator*(Vec2f const& a, Vec2f const& b)
    {
        return { a.x * b.x, a.y * b.y };
    }

    ASG_FUNC inline Vec2f operator/(Vec2f const& a, Vec2f const& b)
    {
        return { a.x / b.x, a.y / b.y };
    }

    ASG_FUNC inline Vec2f& operator+=(Vec2f& a, Vec2f const& b)
    {
        a = a + b;
        return a;
    }

    ASG_FUNC inline Vec2f& operator-=(Vec2f& a, Vec2f const& b)
    {
        a = a - b;
        return a;
    }

    ASG_FUNC inline Vec2f& operator*=(Vec2f& a, Vec2f const& b)
    {
        a = a * b;
        return a;
    }

    ASG_FUNC inline Vec2f& operator/=(Vec2f& a, Vec2f const& b)
    {
        a = a / b;
        return a;
    }

    inline std::ostream& operator<<(std::ostream& out, Vec2f const& v)
    {
        out << '(' << v.x << ',' << v.y << ')';
        return out;
    }


    //--- Vec3f -------------------------------------------

    ASG_FUNC inline bool operator==(Vec3f const& a, Vec3f const& b)
    {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }

    ASG_FUNC inline bool operator!=(Vec3f const& a, Vec3f const& b)
    {
        return !(a == b);
    }

    ASG_FUNC inline Vec3f operator-(Vec3f const& a)
    {
        return { -a.x, -a.y, -a.z };
    }

    ASG_FUNC inline Vec3f operator+(Vec3f const& a, Vec3f const& b)
    {
        return  { a.x + b.x, a.y + b.y, a.z + b.z };
    }

    ASG_FUNC inline Vec3f operator-(Vec3f const& a, Vec3f const& b)
    {
        return  { a.x - b.x, a.y - b.y, a.z - b.z };
    }

    ASG_FUNC inline Vec3f operator*(Vec3f const& a, Vec3f const& b)
    {
        return { a.x * b.x, a.y * b.y, a.z * b.z };
    }

    ASG_FUNC inline Vec3f operator/(Vec3f const& a, Vec3f const& b)
    {
        return { a.x / b.x, a.y / b.y, a.z / b.z };
    }

    ASG_FUNC inline Vec3f operator+(Vec3f const& a, float b)
    {
        return  { a.x + b, a.y + b, a.z + b };
    }

    ASG_FUNC inline Vec3f operator-(Vec3f const& a, float b)
    {
        return  { a.x - b, a.y - b, a.z - b };
    }

    ASG_FUNC inline Vec3f operator*(Vec3f const& a, float b)
    {
        return  { a.x * b, a.y * b, a.z * b };
    }

    ASG_FUNC inline Vec3f operator/(Vec3f const& a, float b)
    {
        return { a.x / b, a.y / b, a.z / b };
    }

    ASG_FUNC inline Vec3f operator*(float a, Vec3f const& b)
    {
        return  { a * b.x, a * b.y, a * b.z };
    }

    ASG_FUNC inline Vec3f operator/(float a, Vec3f const& b)
    {
        return { a / b.x, a / b.y, a / b.z };
    }

    ASG_FUNC inline Vec3f& operator+=(Vec3f& a, Vec3f const& b)
    {
        a = a + b;
        return a;
    }

    ASG_FUNC inline Vec3f& operator-=(Vec3f& a, Vec3f const& b)
    {
        a = a - b;
        return a;
    }

    ASG_FUNC inline Vec3f& operator*=(Vec3f& a, Vec3f const& b)
    {
        a = a * b;
        return a;
    }

    ASG_FUNC inline Vec3f& operator/=(Vec3f& a, Vec3f const& b)
    {
        a = a / b;
        return a;
    }

    ASG_FUNC inline Vec3f& operator+=(Vec3f& a, float b)
    {
        a = a + b;
        return a;
    }

    ASG_FUNC inline Vec3f& operator-=(Vec3f& a, float b)
    {
        a = a - b;
        return a;
    }

    ASG_FUNC inline Vec3f& operator*=(Vec3f& a, float b)
    {
        a = a * b;
        return a;
    }

    ASG_FUNC inline Vec3f& operator/=(Vec3f& a, float b)
    {
        a = a / b;
        return a;
    }

    ASG_FUNC inline Vec3f cross(Vec3f const& a, Vec3f const& b)
    {
        return { a.y * b.z - a.z * b.y,
                 a.z * b.x - a.x * b.z,
                 a.x * b.y - a.y * b.x };
    }

    ASG_FUNC inline float dot(Vec3f const& a, Vec3f const& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    ASG_FUNC inline float length(Vec3f const& a)
    {
        return sqrtf(dot(a, a));
    }

    ASG_FUNC inline Vec3f normalize(Vec3f const& a)
    {
        return a / length(a);
    }

    ASG_FUNC inline Vec3f min(Vec3f const& a, Vec3f const& b)
    {
        return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
    }

    ASG_FUNC inline Vec3f max(Vec3f const& a, Vec3f const& b)
    {
        return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
    }

    inline std::ostream& operator<<(std::ostream& out, Vec3f const& v)
    {
        out << '(' << v.x << ',' << v.y << ',' << v.z << ')';
        return out;
    }


    //--- Vec4f -------------------------------------------

    ASG_FUNC inline bool operator==(Vec4f const& a, Vec4f const& b)
    {
        return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
    }

    ASG_FUNC inline bool operator!=(Vec4f const& a, Vec4f const& b)
    {
        return !(a == b);
    }

    ASG_FUNC inline Vec4f operator+(Vec4f const& a, Vec4f const& b)
    {
        return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
    }

    ASG_FUNC inline Vec4f operator-(Vec4f const& a, Vec4f const& b)
    {
        return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
    }

    ASG_FUNC inline Vec4f operator*(Vec4f const& a, Vec4f const& b)
    {
        return { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
    }

    ASG_FUNC inline Vec4f operator/(Vec4f const& a, Vec4f const& b)
    {
        return { a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
    }

    ASG_FUNC inline Vec4f& operator+=(Vec4f& a, Vec4f const& b)
    {
        a = a + b;
        return a;
    }

    ASG_FUNC inline Vec4f& operator-=(Vec4f& a, Vec4f const& b)
    {
        a = a - b;
        return a;
    }

    ASG_FUNC inline Vec4f& operator*=(Vec4f& a, Vec4f const& b)
    {
        a = a * b;
        return a;
    }

    ASG_FUNC inline Vec4f& operator/=(Vec4f& a, Vec4f const& b)
    {
        a = a / b;
        return a;
    }

    inline std::ostream& operator<<(std::ostream& out, Vec4f const& v)
    {
        out << '(' << v.x << ',' << v.y << ',' << v.z << ',' << v.w << ')';
        return out;
    }


    //--- Vec2i -------------------------------------------

    ASG_FUNC inline bool operator==(Vec2i const& a, Vec2i const& b)
    {
        return a.x == b.x && a.y == b.y;
    }

    ASG_FUNC inline bool operator!=(Vec2i const& a, Vec2i const& b)
    {
        return !(a == b);
    }

    ASG_FUNC inline Vec2i operator+(Vec2i const& a, Vec2i const& b)
    {
        return { a.x + b.x, a.y + b.y };
    }

    ASG_FUNC inline Vec2i operator-(Vec2i const& a, Vec2i const& b)
    {
        return { a.x - b.x, a.y - b.y };
    }

    ASG_FUNC inline Vec2i operator*(Vec2i const& a, Vec2i const& b)
    {
        return { a.x * b.x, a.y * b.y };
    }

    ASG_FUNC inline Vec2i operator/(Vec2i const& a, Vec2i const& b)
    {
        return { a.x / b.x, a.y / b.y };
    }

    ASG_FUNC inline Vec2i& operator+=(Vec2i& a, Vec2i const& b)
    {
        a = a + b;
        return a;
    }

    ASG_FUNC inline Vec2i& operator-=(Vec2i& a, Vec2i const& b)
    {
        a = a - b;
        return a;
    }

    ASG_FUNC inline Vec2i& operator*=(Vec2i& a, Vec2i const& b)
    {
        a = a * b;
        return a;
    }

    ASG_FUNC inline Vec2i& operator/=(Vec2i& a, Vec2i const& b)
    {
        a = a / b;
        return a;
    }

    inline std::ostream& operator<<(std::ostream& out, Vec2i const& v)
    {
        out << '(' << v.x << ',' << v.y << ')';
        return out;
    }


    //--- Vec3i -------------------------------------------

    ASG_FUNC inline bool operator==(Vec3i const& a, Vec3i const& b)
    {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }

    ASG_FUNC inline bool operator!=(Vec3i const& a, Vec3i const& b)
    {
        return !(a == b);
    }

    ASG_FUNC inline Vec3i operator+(Vec3i const& a, Vec3i const& b)
    {
        return { a.x + b.x, a.y + b.y, a.z + b.z };
    }

    ASG_FUNC inline Vec3i operator-(Vec3i const& a, Vec3i const& b)
    {
        return { a.x - b.x, a.y - b.y, a.z - b.z };
    }

    ASG_FUNC inline Vec3i operator*(Vec3i const& a, Vec3i const& b)
    {
        return { a.x * b.x, a.y * b.y, a.z * b.z };
    }

    ASG_FUNC inline Vec3i operator/(Vec3i const& a, Vec3i const& b)
    {
        return { a.x / b.x, a.y / b.y, a.z / b.z };
    }

    ASG_FUNC inline Vec3i operator*(Vec3i const& a, int b)
    {
        return { a.x * b, a.y * b, a.z * b };
    }

    ASG_FUNC inline Vec3i& operator+=(Vec3i& a, Vec3i const& b)
    {
        a = a + b;
        return a;
    }

    ASG_FUNC inline Vec3i& operator-=(Vec3i& a, Vec3i const& b)
    {
        a = a - b;
        return a;
    }

    ASG_FUNC inline Vec3i& operator*=(Vec3i& a, Vec3i const& b)
    {
        a = a * b;
        return a;
    }

    ASG_FUNC inline Vec3i& operator/=(Vec3i& a, Vec3i const& b)
    {
        a = a / b;
        return a;
    }

    ASG_FUNC inline Vec3i& operator*=(Vec3i& a, int b)
    {
        a = a * b;
        return a;
    }

    ASG_FUNC inline Vec3i min(Vec3i const& a, Vec3i const& b)
    {
        return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
    }

    ASG_FUNC inline Vec3i max(Vec3i const& a, Vec3i const& b)
    {
        return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
    }

    inline std::ostream& operator<<(std::ostream& out, Vec3i const& v)
    {
        out << '(' << v.x << ',' << v.y << ',' << v.z << ')';
        return out;
    }


    //--- Vec4i -------------------------------------------

    ASG_FUNC inline bool operator==(Vec4i const& a, Vec4i const& b)
    {
        return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
    }

    ASG_FUNC inline bool operator!=(Vec4i const& a, Vec4i const& b)
    {
        return !(a == b);
    }

    ASG_FUNC inline Vec4i operator+(Vec4i const& a, Vec4i const& b)
    {
        return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
    }

    ASG_FUNC inline Vec4i operator-(Vec4i const& a, Vec4i const& b)
    {
        return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
    }

    ASG_FUNC inline Vec4i operator*(Vec4i const& a, Vec4i const& b)
    {
        return { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
    }

    ASG_FUNC inline Vec4i operator/(Vec4i const& a, Vec4i const& b)
    {
        return { a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
    }

    ASG_FUNC inline Vec4i& operator+=(Vec4i& a, Vec4i const& b)
    {
        a = a + b;
        return a;
    }

    ASG_FUNC inline Vec4i& operator-=(Vec4i& a, Vec4i const& b)
    {
        a = a - b;
        return a;
    }

    ASG_FUNC inline Vec4i& operator*=(Vec4i& a, Vec4i const& b)
    {
        a = a * b;
        return a;
    }

    ASG_FUNC inline Vec4i& operator/=(Vec4i& a, Vec4i const& b)
    {
        a = a / b;
        return a;
    }

    inline std::ostream& operator<<(std::ostream& out, Vec4i const& v)
    {
        out << '(' << v.x << ',' << v.y << ',' << v.z << ',' << v.w << ')';
        return out;
    }


    //--- MatN --------------------------------------------

    template <typename MatN>
    ASG_FUNC inline MatN makeIdentity()
    {
        return MatN{};
    }


    //--- Mat3f -------------------------------------------

    template <>
    ASG_FUNC inline Mat3f makeIdentity()
    {
        return Mat3f{
            { 1.f, 0.f, 0.f },
            { 0.f, 1.f, 0.f },
            { 0.f, 0.f, 1.f }
        };
    }

    ASG_FUNC inline Mat3f makeRotation(Vec3f const& axis, float angle)
    {
        Vec3f v = normalize(axis);
        float s = sinf(angle);
        float c = cosf(angle);

        return Mat3f{
            { v.x*v.x*(1.f-c)+c, v.x*v.y*(1.f-c)+s*v.z, v.x*v.z*(1.f-c)-s*v.y },
            { v.y*v.x*(1.f-c)-s*v.z, v.y*v.y*(1.f-c)+c, v.y*v.z*(1.f-c)+s*v.x },
            { v.x*v.x*(1.f-c)+s*v.y, v.z*v.y*(1.f-c)-s*v.x, v.z*v.z*(1.f-c)+c }
        };
    }

    ASG_FUNC inline Mat3f operator*(Mat3f const& a, Mat3f const& b)
    {
        return Mat3f{
            { a.col0.x*b.col0.x + a.col1.x*b.col0.y + a.col2.x*b.col0.z,
              a.col0.y*b.col0.x + a.col1.y*b.col0.y + a.col2.y*b.col0.z,
              a.col0.z*b.col0.x + a.col1.z*b.col0.y + a.col2.z*b.col0.z },
            { a.col0.x*b.col1.x + a.col1.x*b.col1.y + a.col2.x*b.col1.z,
              a.col0.y*b.col1.x + a.col1.y*b.col1.y + a.col2.y*b.col1.z,
              a.col0.z*b.col1.x + a.col1.z*b.col1.y + a.col2.z*b.col1.z },
            { a.col0.x*b.col2.x + a.col1.x*b.col2.y + a.col2.x*b.col2.z,
              a.col0.y*b.col2.x + a.col1.y*b.col2.y + a.col2.y*b.col2.z,
              a.col0.z*b.col2.x + a.col1.z*b.col2.y + a.col2.z*b.col2.z }
        };
    }

    ASG_FUNC inline Vec3f operator*(Mat3f const& a, Vec3f const& b)
    {
        return { 
            a.col0.x * b.x + a.col1.x * b.y + a.col2.x * b.z,
            a.col0.y * b.x + a.col1.y * b.y + a.col2.y * b.z,
            a.col0.z * b.x + a.col1.z * b.y + a.col2.z * b.z
            };
    }

    inline std::ostream& operator<<(std::ostream& out, Mat3f const& m)
    {
        out << '(' << m.col0 << ',' << m.col1 << ',' << m.col2 << ')';
        return out;
    }


    //--- Mat4x3f -----------------------------------------

    template <>
    ASG_FUNC inline Mat4x3f makeIdentity()
    {
        return Mat4x3f{
            { 1.f, 0.f, 0.f },
            { 0.f, 1.f, 0.f },
            { 0.f, 0.f, 1.f },
            { 0.f, 0.f, 0.f }
        };
    }

    ASG_FUNC inline Vec3f operator*(Mat4x3f const& a, Vec4f const& b)
    {
        return {
            a.col0.x * b.x + a.col1.x * b.y + a.col2.x * b.z + a.col3.x * b.w,
            a.col0.y * b.x + a.col1.y * b.y + a.col2.y * b.z + a.col3.y * b.w,
            a.col0.z * b.x + a.col1.z * b.y + a.col2.z * b.z + a.col3.z * b.w,
            };
    }

    ASG_FUNC inline Vec4f operator*(Vec3f const& a, Mat4x3f const& b)
    {
        return {
            a.x * b.col0.x + a.y * b.col0.y + a.z * b.col0.z,
            a.x * b.col1.x + a.y * b.col1.y + a.z * b.col1.z,
            a.x * b.col2.x + a.y * b.col2.y + a.z * b.col2.z,
            a.x * b.col3.x + a.y * b.col3.y + a.z * b.col3.z
            };
    }

    inline std::ostream& operator<<(std::ostream& out, Mat4x3f const& m)
    {
        out << '(' << m.col0 << ',' << m.col1 << ',' << m.col2 << ',' << m.col3 << ')';
        return out;
    }

} // asg


