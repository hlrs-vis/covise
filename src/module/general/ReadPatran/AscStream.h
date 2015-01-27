/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __ASC_STREAM_H
#define __ASC_STREAM_H
#include <util/coviseCompat.h>

/// class to read ASCII files line per line
/// Can handle DOS line ending \13\10 and UNIX ending \n
class AscStream
{

public:
    AscStream(ifstream *in)
    {
        in_ = in;
    };

    bool getline(char *buf, int maxnum);

private:
    ifstream *in_;
};
#endif
