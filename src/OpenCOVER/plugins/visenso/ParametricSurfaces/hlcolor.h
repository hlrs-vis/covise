/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __hlcolor_h__
#define __hlcolor_h__

#include "hlmacros.h"
#include "hlvector.h"

class HlColor
{

private:
    double mRed, mGreen, mBlue;

public:
    HlColor()
        : mRed(0)
        , mGreen(0)
        , mBlue(0)
    {
    }
    HlColor(double r, double g, double b)
        : mRed(r)
        , mGreen(g)
        , mBlue(b)
    {
    }
    HlColor(const HlVector &rgb)
    {
        setRGB(rgb);
    }
    double getRed() const
    {
        return mRed;
    }
    double getGreen() const
    {
        return mGreen;
    }
    double getBlue() const
    {
        return mBlue;
    }
    void setRed(double red)
    {
        mRed = red;
    }
    void setGreen(double green)
    {
        mGreen = green;
    }
    void setBlue(double blue)
    {
        mBlue = blue;
    }
    HlVector getRGB() const
    {
        return HlVector(mRed, mGreen, mBlue);
    }
    void setRGB(const HlVector &rgb)
    {
        mRed = rgb.mX;
        mGreen = rgb.mY;
        mBlue = rgb.mZ;
    }
    HlVector getHSV(const HlVector &hsv)
    {
        return RGB2HSV(getRGB());
    }
    void setHSV(const HlVector &hsv)
    {
        setRGB(HSV2RGB(hsv));
    }
    static HlVector RGB2HSV(const HlVector &RGB)
    {
        return RGB2HSV(RGB.mX, RGB.mY, RGB.mZ);
    }
    static HlVector RGB2HSV(double r, double g, double b);
    static HlVector HSV2RGB(const HlVector &HSV)
    {
        return HSV2RGB(HSV.mX, HSV.mY, HSV.mZ);
    }
    static HlVector HSV2RGB(double h, double s, double v);
};

inline HlVector HlColor::RGB2HSV(double r, double g, double b)
{
    // RGB are each on [0, 1]. S and V are returned on [0, 1] and H is
    // returned on [0, 6]. Exception: H is returned UNDEFINED if S==0.

    int i;
    double x, v, f;

    x = MIN3(r, g, b);
    v = MAX3(r, g, b);
    if (v == x)
        return HlVector(-1, 0, v);
    f = (r == x) ? g - b : ((g == x) ? b - r : r - g);
    i = (r == x) ? 3 : ((g == x) ? 5 : 1);
    return HlVector(i - f / (v - x), (v - x) / v, v);
}

inline HlVector HlColor::HSV2RGB(double h, double s, double v)
{

    // H is given on [0, 6] or UNDEFINED. S and V are given on [0, 1].
    // RGB are each returned on [0, 1].
    double m, n, f;
    int i;

    if (h == -1)
        return HlVector(v, v, v);
    i = (int)floor(h);
    f = h - i;
    if (!(i & 1))
        f = 1 - f; // if i is even
    m = v * (1 - s);
    n = v * (1 - s * f);
    switch (i)
    {
    case 6:
    case 0:
        return HlVector(v, n, m);
    case 1:
        return HlVector(n, v, m);
    case 2:
        return HlVector(m, v, n);
    case 3:
        return HlVector(m, n, v);
    case 4:
        return HlVector(n, m, v);
    case 5:
        return HlVector(v, m, n);
    }
    return HlVector(0, 0, 0);
}

#endif // __hlcolor_h__
