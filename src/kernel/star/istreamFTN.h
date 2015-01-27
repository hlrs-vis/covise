/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ISTREAMFTN_H_
#define _ISTREAMFTN_H_

#include <covise/covise.h>
#include "istreamFTN.h"
#include "util/coTypes.h"

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

namespace covise
{

class STAREXPORT istreamFTN
{

private:
    int d_actBlockNo;
    int d_fileDesc;
    int d_errFlag;
    int d_byteSwap;

public:
    istreamFTN(int filedes)
        : d_actBlockNo(1)
        , d_fileDesc(filedes)
        , d_errFlag(0)
        , d_byteSwap(0){};

    ~istreamFTN()
    {
        close(d_fileDesc);
    }

    int readFTN(void *data, int length);
    int readFTN_BS(void *data, int length);

    int skipBlocks(int numBlocks);
    int skipBlocks(int maxidx, int block)
    {
        if (maxidx > 0)
            return skipBlocks((maxidx - 1) / block + 1);
        else
            return 1;
    }

    int readFTNfloat(float *data, int num, int perBlock);
    int readFTNint(int *data, int num, int perBlock);

    int getActualBlockNo() const
    {
        return d_actBlockNo;
    }

    int fail()
    {
        return (d_errFlag != 0);
    }

    /// scan forward until you find a block of the apprpriate size, then read it
    int scanForSizeFloat(float *data, int num);
    int scanForSizeInt(int *data, int num);

    /// skip forward behind a block of the given size: return 0 in success
    int skipForSize(int size);

    /// rewind the file : also resets error flag
    void rewind();

    // switch Byteswapping on/off
    void setByteSwap(int val)
    {
        d_byteSwap = (val != 0);
    }
};
}
#endif
