/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// RRFrame.h

#ifndef RR_FRAME_H
#define RR_FRAME_H

#include <QMutex>
#include <QMutexLocker>
#include <QVector>

#include "RRCompressedTile.h"
#include "RRDecompressor.h"

class RRFrame
{
public:
    RRFrame(bool planar);
    ~RRFrame();

    inline void lock()
    {
        mutex.lock();
    }

    inline void unlock()
    {
        mutex.unlock();
    }

    // Resize the buffer(s)
    bool resize(int w, int h);

    void setSubSampling(int rrsub, int h, int v);

    inline unsigned char *getData() const
    {
        return data;
    }

    inline int getWidth() const
    {
        return width;
    }

    inline int getHeight() const
    {
        return height;
    }

    inline int getRowBytes() const
    {
        return rowBytes;
    }

    inline int getPixelSize() const
    {
        return pixelSize;
    }

    inline int getSubSampling() const
    {
        return sub;
    }
    inline int getHorizSubSampling() const
    {
        return horizSub;
    }
    inline int getVertSubSampling() const
    {
        return vertSub;
    }

    unsigned char *yData(int x, int y) const;
    unsigned char *uData(int x, int y) const;
    unsigned char *vData(int x, int y) const;

private:
    QMutex mutex;

    unsigned char *data;
    int width;
    int height;
    int rowBytes;
    int pixelSize;
    int horizSub;
    int vertSub;
    int sub;
    bool planar;
};
#endif
