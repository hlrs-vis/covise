/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "istreamFTN.h"
#include <util/coviseCompat.h>

void istreamFTN::parseString(char *string, int start, int end, int *result)
{
    if (end > (int)strlen(string) - 1)
        return;

    int i;
    for (i = start; i <= end; i++)
        parse_buffer[i - start] = string[i];
    parse_buffer[end - start + 1] = '\n';
    parse_buffer[end - start + 2] = '\0';
    if (sscanf(parse_buffer, "%d", result) != 1)
    {
        fprintf(stderr, "istreamFTN::parseString: sscanf failed\n");
    }
    return;
}

void istreamFTN::parseString(char *string, int start, int end, float *result)
{
    if (end > (int)strlen(string) - 1)
        return;

    int i;
    for (i = start; i <= end; i++)
        parse_buffer[i - start] = string[i];
    parse_buffer[end - start + 1] = '\n';
    parse_buffer[end - start + 2] = '\0';
    if (sscanf(parse_buffer, "%f", result) != 1)
    {
        fprintf(stderr, "istreamFTN::parseString: sscanf2 failed\n");
    }
    return;
}

int istreamFTN::readFTN(void *data, int len)
{
#ifdef VERBOSE
    cerr << "# " << actualBlockNo
         << ": istreamFTN::readFTN(void *data," << len << ")"
         << endl;
#endif

    int blocklen;
    int rd;
    int bytesRead;

    rd = read(fd, (void *)(&blocklen), (size_t)sizeof(int));
    if (machineIsLittleEndian())
    {
        byteSwap(blocklen);
    }

    if ((rd < sizeof(int)) || (len > blocklen))
    {
        error = 1;
        return -1;
    }

    bytesRead = read(fd, (void *)data, len);
    if (bytesRead < len)
    {
        error = 1;
        return -1;
    }
    if (machineIsLittleEndian())
    {
        byteSwap((int *)data, len / sizeof(int));
    }

    if (lseek(fd, blocklen - len, SEEK_CUR) < 0)
    {
        error = 1;
        return -1;
    }

    int dummy;
    rd = read(fd, (void *)&dummy, (size_t)sizeof(int));
    if (machineIsLittleEndian())
    {
        byteSwap(dummy);
    }
    if ((rd < sizeof(int)) || (dummy != blocklen))
    {
        error = 1;
        return -1;
    }

    actualBlockNo++;
    return bytesRead;
}

int istreamFTN::skipBlocks(int numBlocks)
{
#ifdef VERBOSE
    cout << "void istreamFTN::skipBlocks(" << numBlocks << ")" << endl;
#endif
    int blocklen;

    actualBlockNo += numBlocks;

    while (numBlocks > 0)
    {
        if (read(fd, (void *)(&blocklen), sizeof(int)) != sizeof(int))
        {
            fprintf(stderr, "istreamFTN::skipBlocks: short read1\n");
        }
        lseek(fd, blocklen, SEEK_CUR);
        int dummy;
        if (read(fd, (void *)&dummy, sizeof(int)) != sizeof(int))
        {
            fprintf(stderr, "istreamFTN::skipBlocks: short read2\n");
        }
        if (dummy != blocklen)
        {
#ifdef VERBOSE
            cerr << "Block not terminated correctly: Termination length "
                 << dummy << " instead of expected " << blocklen
                 << endl;
#endif
            error = 1;
            return -1;
        }
        numBlocks--;
    }
    return 1;
}

#ifndef _MSC_VER
inline int min(int i, int k)
{
    return (i < k) ? i : k;
}
#endif

//This function is unused?
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
        readLast = readFTN((void *)data, sizeof(float) * readNow);
        if (machineIsLittleEndian())
        {
            byteSwap(data, readNow);
        }
        if (readLast != sizeof(float) * readNow)
        {
#ifdef VERBOSE
            cerr << "short read in readFTNfloat: expected " << sizeof(float) * readNow
                 << " bytes, read only " << readLast << endl;
#endif
            error = 1;
            return -1;
        }
        readAll += readLast;
        data += perBlock;
        num -= readNow;
    }
    return readLast / sizeof(float);
}

//This function is unused?
int istreamFTN::readFTNint(int *data, int num, int perBlock)
{
#ifdef VERBOSE
    cerr << "int istreamFTN::readFTNint(int *data,"
         << num << ", " << perBlock << ")"
         << endl;
#endif
    int readNow;
    int readLast = 0;
    int readAll = 0;
    while (num > 0)
    {
        readNow = std::min(num, perBlock);
        readLast = readFTN((void *)data, sizeof(int) * readNow);
        if (machineIsLittleEndian())
        {
            byteSwap(data, readNow);
        }
        if (readLast != sizeof(int) * readNow)
        {
#ifdef VERBOSE
            cerr << "short read in readFTNfloat: expected " << sizeof(int) * readNow
                 << " bytes, read only " << readLast << endl;
#endif
            error = 1;
            return -1;
        }
        readAll += readLast;
        data += perBlock;
        num -= readNow;
    }
    return readLast / sizeof(int);
}
