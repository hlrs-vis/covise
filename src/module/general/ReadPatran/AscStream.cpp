/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AscStream.h"
#include <util/coviseCompat.h>
bool
AscStream::getline(char *buf, int maxnum)
{
    bool ret = false;
    buf[0] = '\0';
    while (buf[0] == '\0')
    {
        if (in_->getline(buf, maxnum))
        {
            ret = true;
        }
        else if (in_->eof())
        {
            return false;
        }
    }
    return ret;
}
