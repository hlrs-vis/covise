/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vruiMatrix.h"

vrui::vruiMatrix::~vruiMatrix()
{
}

bool vrui::vruiMatrix::isIdentity() const
{
    for (int i=0; i<4; ++i)
    {
        for (int j=0; j<4; ++j)
            if ((*this)(i,j) != (double)(i==j))
                return false;
    }

    return true;
}
