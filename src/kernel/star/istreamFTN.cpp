/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <covise/covise.h>
#include "istreamFTN.h"

#undef VERBOSE

using namespace covise;

///  change this if unsigned/int != 32 bit
#define SWAP(x) x = ((x & 0x000000ff) << 24)  \
                    | ((x & 0x0000ff00) << 8) \
                    | ((x & 0x00ff0000) >> 8) \
                    | ((x & 0xff000000) >> 24)

#define SWAP_N(data, len)                                                                                                               \
    do                                                                                                                                  \
    {                                                                                                                                   \
        int i, num = (len) >> 2;                                                                                                        \
        unsigned int val;                                                                                                               \
        unsigned int *udata = (unsigned int *)(data);                                                                                   \
        for (i = 0; i < num; i++)                                                                                                       \
        {                                                                                                                               \
            val = udata[i];                                                                                                             \
            udata[i] = ((val & 0x000000ff) << 24) | ((val & 0x0000ff00) << 8) | ((val & 0x00ff0000) >> 8) | ((val & 0xff000000) >> 24); \
        }                                                                                                                               \
    } while (false)

int istreamFTN::readFTN(void *data, int len)
{
#ifdef VERBOSE
    cerr << "# " << d_actBlockNo
         << ": istreamFTN::readFTN(void *data," << len << ")"
         << endl;
#endif

    int blocklen, bytesRead, rd;

    rd = read(d_fileDesc, (void *)(&blocklen), sizeof(int));
    if (d_byteSwap)
        SWAP(blocklen);

    if ((rd < sizeof(int)) || (len > blocklen))
    {
#ifdef VERBOSE
        cerr << "# " << d_actBlockNo
             << ": Block length error: block should be "
             << len << " bytes, but is " << blocklen << " bytes"
             << endl;
#endif
        d_errFlag = 1;
        return -1;
    }

    bytesRead = read(d_fileDesc, (void *)data, len);

    if (bytesRead < len)
    {
        d_errFlag = 1;
        return -1;
    }

    if (lseek(d_fileDesc, blocklen - len, SEEK_CUR) < 0)
    {
        d_errFlag = 1;
        return -1;
    }

    int dummy;
    rd = read(d_fileDesc, (void *)&dummy, sizeof(int));
    if (d_byteSwap)
        SWAP(dummy);

    if ((rd < sizeof(int)) || (dummy != blocklen))
    {
        d_errFlag = 1;
        return -1;
    }

    d_actBlockNo++;
    return bytesRead;
}

int istreamFTN::readFTN_BS(void *data, int len)
{
#ifdef VERBOSE
    cerr << "# " << d_actBlockNo
         << ": istreamFTN::readFTN_BS(void *data," << len << ")"
         << endl;
#endif

    int blocklen, bytesRead, rd;

    rd = read(d_fileDesc, (void *)(&blocklen), sizeof(int));
    if (d_byteSwap)
        SWAP(blocklen);

    if ((rd < sizeof(int)) || (len > blocklen))
    {
#ifdef VERBOSE
        cerr << "# " << d_actBlockNo
             << ": Block length error: block should be "
             << len << " bytes, but is " << blocklen << " bytes"
             << endl;
#endif
        d_errFlag = 1;
        return -1;
    }

    bytesRead = read(d_fileDesc, (void *)data, len);
    if (d_byteSwap)
        SWAP_N(data, len);

    if (bytesRead < len)
    {
#ifdef VERBOSE
        cerr << "# " << d_actBlockNo
             << ": read error: wanted "
             << len << " bytes, but read " << bytesRead << " bytes"
             << endl;
#endif
        d_errFlag = 1;
        return -1;
    }

    if (lseek(d_fileDesc, blocklen - len, SEEK_CUR) < 0)
    {
        d_errFlag = 1;
        return -1;
    }

    int dummy;
    rd = read(d_fileDesc, (void *)&dummy, sizeof(int));
    if (d_byteSwap)
        SWAP(dummy);

    if ((rd < sizeof(int)) || (dummy != blocklen))
    {
        d_errFlag = 1;
        return -1;
    }

    d_actBlockNo++;
    return bytesRead;
}

int istreamFTN::skipBlocks(int numBlocks)
{
#ifdef VERBOSE
    cout << "# " << d_actBlockNo
         << ": void istreamFTN::skipBlocks(" << numBlocks << ")" << endl;
#endif
    int blocklen = 0;

    d_actBlockNo += numBlocks;

    while (numBlocks > 0)
    {
        ssize_t retval;
        retval = read(d_fileDesc, (void *)(&blocklen), sizeof(int));
        if (retval == -1)
        {
            std::cerr << "istreamFTN::skipBlocks: sscanf failed" << std::endl;
            return -1;
        }
        if (d_byteSwap)
            SWAP(blocklen);

        off_t off = lseek(d_fileDesc, blocklen, SEEK_CUR);
#ifdef VERBOSE
        cout << "# " << d_actBlockNo
             << ": void istreamFTN::skipBlocks(" << numBlocks << "): offset " << off << endl;
#endif
        if (off == -1)
        {
            std::cerr << "istreamFTN::skipBlocks: lseek failed" << std::endl;
            return -1;
        }

        int dummy;
        retval = read(d_fileDesc, (void *)&dummy, sizeof(int));
        if (retval == -1)
        {
            std::cerr << "istreamFTN::skipBlocks: sscanf failed" << std::endl;
            return -1;
        }
        if (d_byteSwap)
            SWAP(dummy);

        if (dummy != blocklen)
        {
#ifdef VERBOSE
            cerr << "Block not terminated correctly: Termination length "
                 << dummy << " instead of expected " << blocklen
                 << endl;
#endif
            d_errFlag = 1;
            return -1;
        }
        numBlocks--;
    }
    return blocklen;
}

#ifndef _MSC_VER
inline int min(int i, int k)
{
    return (i < k) ? i : k;
}
#endif

int istreamFTN::readFTNfloat(float *data, int num, int perBlock)
{
#ifdef VERBOSE
    cerr << "int istreamFTN::readFTNfloat(float *data,"
         << num << ", " << perBlock << ")"
         << endl;
#endif
    int readNow;
    int readLast = 0;
    int readAll = 0;
    while (num > 0)
    {
        readNow = std::min(num, perBlock);
        readLast = readFTN_BS((void *)data, sizeof(float) * readNow);
        if (readLast != sizeof(float) * readNow)
        {
#ifdef VERBOSE
            cerr << "# " << d_actBlockNo
                 << " short read in readFTNfloat: expected " << sizeof(float) * readNow
                 << " bytes, read only " << readLast << endl;
#endif
            d_errFlag = 1;
            return -1;
        }
        readAll += readLast;
        data += perBlock;
        num -= readNow;
    }
    return readLast / sizeof(float);
}

int istreamFTN::readFTNint(int *data, int num, int perBlock)
{
#ifdef VERBOSE
    cerr << "int istreamFTN::readFTNfloat(int *data,"
         << num << ", " << perBlock << ")"
         << endl;
#endif
    int readNow;
    int readLast = 0;
    int readAll = 0;
    while (num > 0)
    {
        readNow = std::min(num, perBlock);
        readLast = readFTN_BS((void *)data, sizeof(int) * readNow);
        if (readLast != sizeof(int) * readNow)
        {
#ifdef VERBOSE
            cerr << "short read in readFTNfloat: expected " << sizeof(int) * readNow
                 << " bytes, read only " << readLast << endl;
#endif
            d_errFlag = 1;
            return -1;
        }
        readAll += readLast;
        data += perBlock;
        num -= readNow;
    }
    return readLast / sizeof(int);
}

/// scan forward until you find a block of the apprpriate size, then read it
int istreamFTN::scanForSizeFloat(float *data, int num)
{
#ifdef VERBOSE
    cerr << "# " << d_actBlockNo
         << ": istreamFTN::scanForSizeFloat(" << num << ")"
         << endl;
#endif
    int size = sizeof(float) * num;
    int blocklen, rd;

    rd = read(d_fileDesc, (void *)(&blocklen), sizeof(int));
    if (d_byteSwap)
        SWAP(blocklen);

    d_actBlockNo++;
    while (rd == sizeof(int) && blocklen != size)
    {
        lseek(d_fileDesc, blocklen + sizeof(int), SEEK_CUR);

        rd = read(d_fileDesc, (void *)(&blocklen), sizeof(int));
        if (d_byteSwap)
            SWAP(blocklen);

        d_actBlockNo++;
#ifdef VERBOSE
        cerr << "   skipped " << blocklen / 4 << " words"
             << endl;
#endif
    }

    if (rd == sizeof(int))
    {
        rd = read(d_fileDesc, (void *)data, size);
        if (d_byteSwap)
            SWAP_N(data, size);

        lseek(d_fileDesc, sizeof(int), SEEK_CUR);
        return 0;
    }
    else
        return -1;
}

/// scan forward until you find a block of the apprpriate size, then read it
int istreamFTN::scanForSizeInt(int *data, int num)
{
#ifdef VERBOSE
    cerr << "# " << d_actBlockNo
         << ": istreamFTN::scanForSizeFloat(" << num << ")"
         << endl;
#endif
    int size = sizeof(int) * num;
    int blocklen, rd;

    rd = read(d_fileDesc, (void *)(&blocklen), sizeof(int));
    if (d_byteSwap)
        SWAP(blocklen);

    d_actBlockNo++;
    while (rd == sizeof(int) && blocklen != size)
    {
        lseek(d_fileDesc, blocklen + sizeof(int), SEEK_CUR);
        rd = read(d_fileDesc, (void *)(&blocklen), sizeof(int));
        if (d_byteSwap)
            SWAP(blocklen);

        d_actBlockNo++;
#ifdef VERBOSE
        cerr << "   skipped " << blocklen / 4 << " words"
             << endl;
#endif
    }

    if (rd == sizeof(int))
    {
        rd = read(d_fileDesc, (void *)data, size);
        if (d_byteSwap)
            SWAP_N(data, size);

        lseek(d_fileDesc, sizeof(int), SEEK_CUR);
        return 0;
    }
    else
        return -1;
}

/// skip forward behind a block of the given size: return 0 in success
int istreamFTN::skipForSize(int size)
{
#ifdef VERBOSE
    cerr << "# " << d_actBlockNo
         << ": istreamFTN::skipForSize(" << size << ")"
         << endl;
#endif

    int blocklen, rd;

    rd = read(d_fileDesc, (void *)(&blocklen), sizeof(int));
    if (d_byteSwap)
        SWAP(blocklen);

    d_actBlockNo++;
    while (rd == sizeof(int) && blocklen != size)
    {
        lseek(d_fileDesc, blocklen + sizeof(int), SEEK_CUR);
        rd = read(d_fileDesc, (void *)(&blocklen), sizeof(int));
        if (d_byteSwap)
            SWAP(blocklen);

        d_actBlockNo++;
#ifdef VERBOSE
        cerr << "   skipped " << blocklen / 4 << " words"
             << endl;
#endif
    }

    if (rd == sizeof(int))
    {
        lseek(d_fileDesc, blocklen + sizeof(int), SEEK_CUR);
        return 0;
    }
    else
        return -1;
}

/// rewind the file : also resets error flag
void istreamFTN::rewind()
{
    lseek(d_fileDesc, 0, SEEK_SET);
}
