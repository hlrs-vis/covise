/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TILED_DISPLAY_SERVER_VV_H
#define TILED_DISPLAY_SERVER_VV_H

//#define TILED_DISPLAY_SYNC
#include "TiledDisplayServer.h"

#ifdef TILE_ENCODE_JPEG
#include <util/JPEG/SGJpegDecoder.h>
#endif

// FIXME: only use virvo for volume rendering - I don't care....
#include <virvo/vvsocketio.h>

class TiledDisplayServerVV : public TiledDisplayServer
{

public:
    TiledDisplayServerVV(int number);
    virtual ~TiledDisplayServerVV();

    bool accept();
    void run();

protected:
    vvSocketIO *socket;

#ifdef TILE_ENCODE_JPEG
    SGJpegImage jpegImage;
    SGJpegDecoder jpegDecoder;
#endif
};

#endif
