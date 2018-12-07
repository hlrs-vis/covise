/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <cstdio>
#ifndef WIN32
#include <unistd.h>
#endif
#include <cctype>
#include <cstring>

static void
byteSwap(int no_points, void *buffer)
{
    int i;
    unsigned int *i_buffer = static_cast<unsigned int *>(buffer);
    for (i = 0; i < no_points; i++)
    {
        unsigned &val = i_buffer[i];
        val = ((val & 0xff000000) >> 24)
              | ((val & 0x00ff0000) >> 8)
              | ((val & 0x0000ff00) << 8)
              | ((val & 0x000000ff) << 24);
    }
}

/////////////////////////////////////////////////////
FILE *covOpenOutFile(const char *filename)
{
    FILE *fi = fopen(filename, "w");
    if (fi == NULL)
    {
        perror(filename);
        return fi;
    }
#ifdef BYTESWAP
    fwrite("COV_BE", 6, 1, fi);
#else
    fwrite("COV_LE", 6, 1, fi);
#endif
    return fi;
}

/////////////////////////////////////////////////////
int covWriteSetBegin(FILE *fi, int numSteps)
{
    fwrite("SETELE", 6, 1, fi);

    return fwrite(&numSteps, sizeof(int), 1, fi);
}

/////////////////////////////////////////////////////
int covWriteAttrib(FILE *fi, int numAttrib,
                   const char *const *attrName,
                   const char *const *attrValue)
{
    int size = sizeof(int);
    int i;
    for (i = 0; i < numAttrib; i++)
        size += strlen(attrName[i]) + strlen(attrValue[i]) + 2;
    fwrite(&size, sizeof(int), 1, fi);
    fwrite(&numAttrib, sizeof(int), 1, fi);
    for (i = 0; i < numAttrib; i++)
    {
        fwrite(attrName[i], strlen(attrName[i]) + 1, 1, fi);
        fwrite(attrValue[i], strlen(attrValue[i]) + 1, 1, fi);
    }
    return 0;
}

/////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        std::cerr << "Call: " << argv[0] << " <file> <file> ..." << std::endl;
        return(1);
    }

    FILE *dataFile = covOpenOutFile("data.covise");

    int numFiles = argc - 1;

    covWriteSetBegin(dataFile, numFiles);

    // loop over files, create data files on-the-fly
    int fileNo;
    for (fileNo = 0; fileNo < numFiles; fileNo++)
    {
        // open input file
        FILE *fi = fopen(argv[fileNo + 1], "r");
        if (!fi)
        {
            perror(argv[fileNo + 1]);
            return (0);
        }
        std::cout << "File " << argv[fileNo + 1] << std::endl;

        struct
        {
            int nx, ny, nz;
        } hdr;

        fread(&hdr, sizeof(hdr), 1, fi);
        byteSwap(sizeof(hdr) / 4, &hdr);

        int numCells = hdr.nx * hdr.ny * hdr.nz;

        const char *strsdt = "STRSDT";
        fwrite(strsdt, 6, 1, dataFile);
        int size;
        size = hdr.nz;
        fwrite(&size, sizeof(int), 1, dataFile);
        size = hdr.ny;
        fwrite(&size, sizeof(int), 1, dataFile);
        size = hdr.nx;
        fwrite(&size, sizeof(int), 1, dataFile);
        float *flgroup = new float[numCells];
        fread(flgroup, sizeof(float), numCells, fi);
        byteSwap(numCells, flgroup);

        fwrite(flgroup, sizeof(float), numCells, dataFile);

        covWriteAttrib(dataFile, 0, NULL, NULL);
        delete[] flgroup;

        // close input file
        fclose(fi);
    }

    // finish sets by writing their Attributes
    char buffer[64];
    sprintf(buffer, "0 %d", numFiles);
    const char *attrName[] = { "TIMESTEP", "CREATOR" };
    const char *attrVal[] = { buffer, argv[0] };

    covWriteAttrib(dataFile, 2, attrName, attrVal);

    fclose(dataFile);

    return 0;
}
