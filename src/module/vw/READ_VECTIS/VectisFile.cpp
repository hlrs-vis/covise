/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VectisFile.h"
#include <appl/ApplInterface.h>

int VectisFile::read_record(int &length, char **data)
{
    int blocksize;
#ifdef BYTESWAP
    int *iptr, i;
#endif

    //	printf("new block --------------------- ");
    read(hdl, &blocksize, sizeof(int));
#ifdef BYTESWAP
    blocksize = (blocksize & 0xff000000) >> 16 | (blocksize & 0x00ff0000) >> 16 | (blocksize & 0x0000ff00) << 16 | (blocksize & 0x000000ff) << 16;
#endif
    //	printf("size: %d\n", blocksize);
    length = blocksize;
    *data = new char[length];
    read(hdl, *data, length);
#ifdef BYTESWAP
    iptr = (int *)*data;
    for (i = 0; i < length / sizeof(int); i++)
        iptr[i] = (iptr[i] & 0xff000000) >> 16 | (iptr[i] & 0x00ff0000) >> 16 | (iptr[i] & 0x0000ff00) << 16 | (iptr[i] & 0x000000ff) << 16;
#endif
    read(hdl, &blocksize, sizeof(int));
    return 0;
}

int VectisFile::read_record(int length, char *data)
{
    int blocksize;

#ifdef BYTESWAP
    int *iptr, i;
#endif

    //	printf("new block --------------------- ");
    read(hdl, &blocksize, sizeof(int));
#ifdef BYTESWAP
    blocksize = (blocksize & 0xff000000) >> 16 | (blocksize & 0x00ff0000) >> 16 | (blocksize & 0x0000ff00) << 16 | (blocksize & 0x000000ff) << 16;
#endif
    //	printf("size: %d\n", blocksize);
    if (length != blocksize)
        return 0;
    read(hdl, data, length);
#ifdef BYTESWAP
    iptr = (int *)*data;
    for (i = 0; i < length / sizeof(int); i++)
        iptr[i] = (iptr[i] & 0xff000000) >> 16 | (iptr[i] & 0x00ff0000) >> 16 | (iptr[i] & 0x0000ff00) << 16 | (iptr[i] & 0x000000ff) << 16;
#endif
    read(hdl, &blocksize, sizeof(int));
    return 0;
}

int VectisFile::skip_record()
{
    int blocksize;
#ifdef BYTESWAP
    int *iptr, i;
#endif

    //	printf("new block --------------------- ");
    read(hdl, &blocksize, sizeof(int));
#ifdef BYTESWAP
    blocksize = (blocksize & 0xff000000) >> 16 | (blocksize & 0x00ff0000) >> 16 | (blocksize & 0x0000ff00) << 16 | (blocksize & 0x000000ff) << 16;
#endif
    printf("skipping: %d\n", blocksize);
    lseek(hdl, blocksize, SEEK_CUR);
    read(hdl, &blocksize, sizeof(int));
    return 0;
}

int VectisFile::read_record(int &data)
{
    int blocksize;
#ifdef BYTESWAP
    int *iptr, i;
#endif

    //	printf("new block --------------------- ");
    read(hdl, &blocksize, sizeof(int));
#ifdef BYTESWAP
    blocksize = (blocksize & 0xff000000) >> 16 | (blocksize & 0x00ff0000) >> 16 | (blocksize & 0x0000ff00) << 16 | (blocksize & 0x000000ff) << 16;
#endif
    if (blocksize != sizeof(float))
        return 0;

    //	printf("size: %d\n", blocksize);
    read(hdl, &data, sizeof(int));
#ifdef BYTESWAP
    data = (data & 0xff000000) >> 16 | (data & 0x00ff0000) >> 16 | (data & 0x0000ff00) << 16 | (data & 0x000000ff) << 16;
#endif
    read(hdl, &blocksize, sizeof(int));
    return 0;
}

int VectisFile::read_record(float &fdata)
{
    int blocksize;
#ifdef BYTESWAP
    int *data, i;
#endif

    //	printf("new block --------------------- ");
    read(hdl, &blocksize, sizeof(int));
#ifdef BYTESWAP
    blocksize = (blocksize & 0xff000000) >> 16 | (blocksize & 0x00ff0000) >> 16 | (blocksize & 0x0000ff00) << 16 | (blocksize & 0x000000ff) << 16;
#endif
    if (blocksize != sizeof(float))
        return 0;

    //	printf("size: %d\n", blocksize);
    read(hdl, &fdata, sizeof(float));

#ifdef BYTESWAP
    data = (int *)&fdata;
    *data = (*data & 0xff000000) >> 16 | (*data & 0x00ff0000) >> 16 | (*data & 0x0000ff00) << 16 | (*data & 0x000000ff) << 16;
#endif
    read(hdl, &blocksize, sizeof(int));
    return 0;
}

int VectisFile::read_textrecord(char **data)
{
    int blocksize, i;
#ifdef BYTESWAP
    int *iptr;
#endif
    char tmp_data[81];

    tmp_data[80] = 0;

    //	printf("new block --------------------- ");
    read(hdl, &blocksize, sizeof(int));
#ifdef BYTESWAP
    blocksize = (blocksize & 0xff000000) >> 16 | (blocksize & 0x00ff0000) >> 16 | (blocksize & 0x0000ff00) << 16 | (blocksize & 0x000000ff) << 16;
#endif
    //	printf("size: %d\n", blocksize);
    if (blocksize != 80)
        return 0;
    read(hdl, tmp_data, 80);
#ifdef BYTESWAP
    iptr = (int *)tmp_data;
    for (i = 0; i < 80 / sizeof(int); i++)
        iptr[i] = (iptr[i] & 0xff000000) >> 16 | (iptr[i] & 0x00ff0000) >> 16 | (iptr[i] & 0x0000ff00) << 16 | (iptr[i] & 0x000000ff) << 16;
#endif
    for (i = 79; i >= 0 && tmp_data[i] == ' '; i--)
        tmp_data[i] = 0;

    *data = new char[strlen(tmp_data) + 1];
    strcpy(*data, tmp_data);
    read(hdl, &blocksize, sizeof(int));
    return 0;
}

VectisFile::VectisFile(char *n)
{
    hdl = Covise::open(n, O_RDONLY);
    if (hdl == -1)
    {
        perror("trouble reading Vectis File");
        name = 0L;
        return;
    }
    name = new char[strlen(n) + 1];
    strcpy(name, n);
}
