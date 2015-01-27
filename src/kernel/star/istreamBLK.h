/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ISTREAMBLK_H_
#define _ISTREAMBLK_H_

#include <covise/covise.h>
#include <util/coTypes.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif
#if defined(__linux__) || defined(CO_hp) || defined(_WIN32) || defined(__APPLE__) || defined(__hpux)
#define lseek64 lseek
#define off64_t off_t
#endif

namespace covise
{

class STAREXPORT istreamBLK
{
private:
    off64_t d_blocksize, d_blockMask;
    int d_filedesc;
    int d_errFlag;
    off64_t d_preRead;
    int d_byteSwap;

public:
    istreamBLK(int filedesc, long blockSize);
    ~istreamBLK()
    {
        if (d_filedesc > 0)
            close(d_filedesc);
    }

    /////  read/skip unformatted bytes
    off64_t read(void *data, off64_t len);

    off64_t skipBlk(off64_t len); // skip len bytes, filled to full block
    off64_t skip(off64_t len); // skip len bytes, d_preRead in Block
    off64_t skipBlocks(int blocks); // skip blocks
    off64_t seekBlock(int blockNo); // seek block, start with 1 !

    /////  read/skip ints/floats/doubles in existing Array

    off64_t read(int *data, off64_t len)
    {
        return read((void *)data, len * sizeof(int));
    }

    off64_t read(float *data, off64_t len)
    {
        return read((void *)data, len * sizeof(float));
    }

    off64_t read(double *data, off64_t len)
    {
        return read((void *)data, len * sizeof(double));
    }

    off64_t skipInt(off64_t len)
    {
        return skip(len * sizeof(int));
    }

    off64_t skipFloat(off64_t len)
    {
        return skip(len * sizeof(float));
    }

    off64_t skipDouble(off64_t len)
    {
        return skip(len * sizeof(double));
    }

    off64_t skipIntBlk(off64_t len)
    {
        return skipBlk(len * sizeof(int));
    }

    off64_t skipFloatBlk(off64_t len)
    {
        return skipBlk(len * sizeof(float));
    }

    off64_t skipDoubleBlk(off64_t len)
    {
        return skipBlk(len * sizeof(double));
    }

    off64_t numRec(off64_t len) const
    {
        return ((len * sizeof(float) - 1) & d_blockMask) / d_blocksize + 1;
    }

    int fail()
    {
        return (d_errFlag != 0);
    }

    void resetErrorFlag()
    {
        d_errFlag = 0;
    }

    void setByteSwap(int val)
    {
        d_byteSwap = (val != 0);
    }

    void rewind();
};
}
#endif
