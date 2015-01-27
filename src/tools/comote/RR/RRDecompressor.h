/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// RRDecompressor.h

#ifndef RR_DECOMPRESSOR_H
#define RR_DECOMPRESSOR_H

#include "RRCompressedTile.h"

#include <vector>

#include <turbojpeg.h>

class RRFrame;

class RRTileDecompressor
{
public:
    typedef std::vector<RRCompressedTile> TileVector;

    RRTileDecompressor(bool planar);
    ~RRTileDecompressor();

    // Decompress the given list of tiles into frame
    int run(TileVector *tv, RRFrame *frame, bool deleteTileBuffer = false);

private:
    std::vector<tjhandle> tjctx;
    bool planar;
};
#endif
