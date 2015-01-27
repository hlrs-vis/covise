/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// RRCompressedTile.h

#ifndef RR_COMPRESSED_TILE_H
#define RR_COMPRESSED_TILE_H

struct RRCompressedTile
{
    unsigned char *buffer;
    int size;
    int x;
    int y;
    int w;
    int h;

    RRCompressedTile()
        : buffer(0)
        , size(0)
        , x(0)
        , y(0)
        , w(0)
        , h(0)
    {
    }
};
#endif
