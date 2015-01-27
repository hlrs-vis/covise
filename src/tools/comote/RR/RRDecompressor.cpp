/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// RRDecompressor.cpp

#include "../Debug.h"

#include "RRFrame.h"
#include "RRDecompressor.h"
#include "RRCompressedTile.h"

#include "tjplanar.h"

#ifdef _OPENMP
#include <omp.h>
#endif

RRTileDecompressor::RRTileDecompressor(bool planar)
    : planar(planar)
{
    tjctx.push_back(tjInitDecompress());
    ASSERT(tjctx.back() != 0);
}

RRTileDecompressor::~RRTileDecompressor()
{
    for (int i = 0; i < tjctx.size(); ++i)
        tjDestroy(tjctx[i]);
    tjctx.clear();
}

int RRTileDecompressor::run(TileVector *tv, RRFrame *frame, bool deleteTileBuffer)
{
    if (tv->begin() >= tv->end())
    {
        return 0; // Nothing to do
    }

    frame->lock();
    {
        const int n = tv->size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
        for (int i = 0; i < n; ++i)
        {
            RRCompressedTile &tile = tv->at(i);

#ifdef _OPENMP
            const int tnum = omp_get_thread_num();
#pragma omp critical
            {
                if (tjctx.size() <= tnum)
                {
                    tjctx.resize(tnum + 1);
                }
                tjhandle &tj = tjctx[tnum];
                if (tj == 0)
                    tj = tjInitDecompress();
            }
#else
            const int tnum = 0;
#endif

            tjhandle tj = tjctx[tnum];
            int w = std::min(tile.w, frame->getWidth() - tile.x);
            int h = std::min(tile.h, frame->getHeight() - tile.y);
            int x = tile.x;
            int y = tile.y;

            if (planar)
            {
                unsigned char *planes[4] = {
                    frame->yData(x, y),
                    frame->uData(x, y),
                    frame->vData(x, y),
                    NULL
                };

                tjDecompressPlanar(tj,
                                   tile.buffer,
                                   tile.size,
                                   planes,
                                   w,
                                   frame->getRowBytes(),
                                   h,
                                   3,
                                   frame->getSubSampling(),
                                   TJ_BGR);
            }
            else
            {
                tjDecompress(tj,
                             tile.buffer,
                             tile.size,
                             frame->getData() + y * frame->getRowBytes() + x * frame->getPixelSize(),
                             w,
                             frame->getRowBytes(),
                             h,
                             frame->getPixelSize(),
                             TJ_BGR);
            }

            if (deleteTileBuffer)
            {
                delete[] tile.buffer;
            }
        }
    }
    frame->unlock();

    return 0;
}
