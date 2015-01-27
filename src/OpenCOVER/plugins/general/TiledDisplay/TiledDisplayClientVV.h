/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TILED_DISPLAY_CLIENT_VV_H
#define TILED_DISPLAY_CLIENT_VV_H

#include <virvo/vvsocketio.h>

#include "TiledDisplayClient.h"

class TiledDisplayClientVV : public TiledDisplayClient
{

public:
    TiledDisplayClientVV(int number, const std::string &compositor);
    virtual ~TiledDisplayClientVV();

    virtual bool connect();
    virtual void run();

private:
    vvSocketIO *socket;

#ifdef TILE_ENCODE_JPEG
    SGJpegEncoder encoder;
    SGJpegImage jpegImage;
#endif
};

#endif
