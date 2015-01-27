/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/sginterface/vruiVec.h>

namespace vrui
{

vruiVec operator-(const vruiVec &first, const vruiVec &second)
{
    int size = (first.size <= second.size) ? first.size : second.size;
    vruiVec rv(size);
    for (int ctr = 0; ctr < size; ++ctr)
        rv[ctr] = first[ctr] - second[ctr];
    return rv;
}
}
