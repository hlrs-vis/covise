/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <covise/covise.h>
#include "istreamBLK.h"

using namespace covise;

istreamBLK::istreamBLK(int filedesc, long blockSize)
    : d_blocksize(blockSize)
    , d_blockMask(~(blockSize - 1))
    , d_filedesc(filedesc)
    , d_errFlag((filedesc >= 0) ? 0 : -1)
    , d_preRead(0)
    , d_byteSwap(0)
{
    ;
}

off64_t istreamBLK::read(void *data, off64_t len)
{
    off64_t readsize = ((len + d_preRead - 1) & d_blockMask) + d_blocksize - d_preRead;
    d_preRead = 0;
    if (readsize < 0)
    {
        d_errFlag = 1;
        return -1;
    }
    off64_t rd = ::read(d_filedesc, (char *)data, len);

    ///  change this if unsigned/int != 32 bit
    if (d_byteSwap)
    {
        int i, num = len >> 2;
        unsigned int val;
        unsigned int *udata = (unsigned int *)data;
        for (i = 0; i < num; i++)
        {
            val = udata[i];
            udata[i] = ((val & 0x000000ff) << 24)
                       | ((val & 0x0000ff00) << 8)
                       | ((val & 0x00ff0000) >> 8)
                       | ((val & 0xff000000) >> 24);
        }
    }

    if (rd < len)
    {
        d_errFlag = 1;
        return -1;
    }
    if (readsize > len)
    {
        if (lseek64(d_filedesc, readsize - len - d_preRead, SEEK_CUR) < 0)
        {
            d_errFlag = 1;
            return -1;
        }
    }
    return rd;
}

off64_t istreamBLK::skipBlk(off64_t len)
{
    off64_t readsize = ((len - 1) & d_blockMask) + d_blocksize;
    if (lseek64(d_filedesc, readsize, SEEK_CUR) < 0)
    {
        d_errFlag = 1;
        return -1;
    }
    return readsize;
}

off64_t istreamBLK::skip(off64_t len)
{
    d_preRead = len;
    if (lseek64(d_filedesc, len, SEEK_CUR) < 0)
    {
        d_errFlag = 1;
        return -1;
    }
    return len;
}

off64_t istreamBLK::skipBlocks(int blocks)
{
    off64_t blk = blocks; // usr 64bit
    blk *= d_blocksize;
    if (lseek64(d_filedesc, blk, SEEK_CUR) < 0)
    {
        d_errFlag = 1;
        return -1;
    }
    return blocks * d_blocksize;
}

off64_t istreamBLK::seekBlock(int blockNo)
{
    blockNo -= 1;
    off64_t blk = blockNo;
    blk *= d_blocksize;
    if (lseek64(d_filedesc, blk, SEEK_SET) < 0)
    {
        d_errFlag = 1;
        return -1;
    }
    return blockNo * d_blocksize;
}

void istreamBLK::rewind()
{
    lseek64(d_filedesc, 0, SEEK_SET);
}
