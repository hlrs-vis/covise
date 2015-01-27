/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ISTREAMFTN_H_
#define _ISTREAMFTN_H_

#include <util/coviseCompat.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

class istreamFTN
{

private:
    int fd; // file descriptor
    int actualBlockNo;
    int error;
    char parse_buffer[500];

public:
    istreamFTN()
        : fd(-1)
        , actualBlockNo(1)
        , error(0){};
    istreamFTN(int filedes)
        : fd(filedes)
        , actualBlockNo(1)
        , error(0){};

#ifdef _WIN32
    ~istreamFTN()
    {
        if (fd >= 0)
            _close(fd);
    }
#else
    ~istreamFTN()
    {
        if (fd >= 0)
            close(fd);
    }
#endif

    int readFTN(void *data, int length);

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
    void parseString(char *string, int start, int end, int *result);
    void parseString(char *string, int start, int end, float *result);
    void parseString(char *string, int start, int end, double *result);

    int getActualBlockNo() const
    {
        return actualBlockNo;
    }

    int fail()
    {
        return (error != 0);
    }
};
#endif
