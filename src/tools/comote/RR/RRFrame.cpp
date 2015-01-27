/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// RRFrame.cpp

#include "../Debug.h"
#include "RRFrame.h"

static const int DctSize = 8;

static int roundup(int i, int round)
{
    return ((i + round - 1) / round) * round;
}

RRFrame::RRFrame(bool planar)
    : data(0)
    , width(0)
    , height(0)
    , rowBytes(0)
    , pixelSize(4)
    , horizSub(1)
    , vertSub(1)
    , sub(1)
    , planar(planar)
{
}

RRFrame::~RRFrame()
{
    delete[] data;
}

bool RRFrame::resize(int w, int h)
{
    QMutexLocker guard(&mutex);

    if (data && width == w && height == h)
    {
        return true;
    }

    delete[] data;

    width = w;
    height = h;

    if (planar)
    {
        rowBytes = roundup(width, DctSize * horizSub);
    }
    else
    {
        rowBytes = width * pixelSize;
    }

    try
    {
        size_t size = 0;
        if (planar)
        {
            size = rowBytes * roundup(height, DctSize * vertSub) + 2 * (rowBytes / horizSub) * (roundup(height, DctSize * vertSub) / vertSub);
        }
        else
        {
            size = height * width * pixelSize;
        }
        data = new unsigned char[size];
    }
    catch (std::exception &e)
    {
        return false;
    }

    return true;
}

void RRFrame::setSubSampling(int s, int h, int v)
{
    sub = s;
    horizSub = h;
    vertSub = v;
}

unsigned char *RRFrame::yData(int x, int y) const
{
    return data
           + y * rowBytes + x;
}

unsigned char *RRFrame::uData(int x, int y) const
{
    return data
           + rowBytes * roundup(height, DctSize * vertSub)
           + (y / vertSub) * (rowBytes / horizSub) + x / horizSub;
}

unsigned char *RRFrame::vData(int x, int y) const
{
    return data
           + rowBytes * roundup(height, DctSize * vertSub)
           + (rowBytes / vertSub) * (roundup(height, DctSize * vertSub) / vertSub)
           + (y / vertSub) * (rowBytes / horizSub) + x / horizSub;
}
