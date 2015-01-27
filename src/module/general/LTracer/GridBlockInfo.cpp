/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GridBlockInfo.h"

GridBlockInfo::GridBlockInfo()
{
    grid = NULL;
    stepNo = -1;
    blockNo = -1;
    block = NULL;
    step = NULL;
    x = y = z = 0.0;
    sbcIdx = 0;
    u = v = w = 0.0;
    g11 = g12 = g13 = 0.0;
    g21 = g22 = g23 = 0.0;
    g31 = g32 = g33 = 0.0;
    type = _GBI_UNDEFINED;
    return;
}

GridBlockInfo::~GridBlockInfo()
{
    // dummy
}
