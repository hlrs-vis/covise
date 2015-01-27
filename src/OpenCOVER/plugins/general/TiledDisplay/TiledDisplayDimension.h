/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TILED_DISPLAY_DIMENSION_H
#define TILED_DISPLAY_DIMENSION_H

#include "TiledDisplayDefines.h"

class TiledDisplayDimension
{
public:
    int width;
    int height;

    TiledDisplayDimension(int w, int h)
    {
        width = w;
        height = h;
    }
    TiledDisplayDimension()
    {
        width = 0;
        height = 0;
    }
};
#endif
