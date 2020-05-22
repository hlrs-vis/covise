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
    int i;

    if (argc != 2)
    {
        std::cerr << "Call: " << argv[0] << " Meshfile" << std::endl;
        return(1);
    }

    FILE *input = fopen(argv[1], "r");

    struct
    {
        int nx, ny, nz;
        int numMat;
        float xMin, yMin, zMin;
    } hdr;

    //////////////////////////////////////////////////////////////////////////
    ///// Header
    fread(&hdr, sizeof(hdr), 1, input);
    byteSwap(sizeof(hdr) / 4, &hdr);

    printf("%d x %d x %d Cells, %d Materials\n\n", hdr.nx, hdr.ny, hdr.nz, hdr.numMat);
    int numCells = hdr.nx * hdr.ny * hdr.nz;

    FILE *infoFile = fopen("MeshInfo", "w");
    fprintf(infoFile, "Converted file: %s\n\n", argv[1]);
    fprintf(infoFile, "%d x %d x %d Cells, %d Materials\n\n", hdr.nx, hdr.ny, hdr.nz, hdr.numMat);

    //////////////////////////////////////////////////////////////////////////
    ///// Dx/DY/Dz
    float *dx = new float[hdr.nx];
    fread(dx, sizeof(float), hdr.nx, input);
    byteSwap(hdr.nx, dx);

    float *dy = new float[hdr.ny];
    fread(dy, sizeof(float), hdr.ny, input);
    byteSwap(hdr.ny, dy);

    float *dz = new float[hdr.nz];
    fread(dz, sizeof(float), hdr.nz, input);
    byteSwap(hdr.nz, dz);

    //////////////////////////////////////////////////////////////////////////
    ///// Mat_Groups
    int *matGroups = new int[hdr.numMat];
    fread(matGroups, sizeof(int), hdr.numMat, input);
    byteSwap(hdr.numMat, matGroups);

    //////////////////////////////////////////////////////////////////////////
    ///// Mat_ids
    int *matID = new int[hdr.numMat];
    fread(matID, sizeof(int), hdr.numMat, input);
    byteSwap(hdr.numMat, matID);

    //////////////////////////////////////////////////////////////////////////
    ///// Materials
    char *materials = new char[numCells];
    fread(materials, sizeof(char), numCells, input);

    //////////////////////////////////////////////////////////////////////////
    ///// info file

    fprintf(infoFile, "%8s %5s %5s %9s\n", "Material", "Grp", "ID", "count");
    fprintf(infoFile, "-----------------------------------\n");
    int idx;
    for (idx = 0; idx < hdr.numMat; idx++)
    {
        int count = 0;
        for (i = 0; i < numCells; i++)
            if (materials[i] == idx)
                count++;
        fprintf(infoFile, "%8d %5d %5d %9d\n", idx, matGroups[idx], matID[idx], count);
    }
    fclose(infoFile);

    //////////////////////////////////////////////////////////////////////////
    ///// Grid
    const char *attribName[] = { "CREATOR" };
    const char *attribVal[] = { "magma2cov v1.0" };

    FILE *outfile = covOpenOutFile("Mesh.covise");
    static const char *rectgrd = "RCTGRD";
    fwrite(rectgrd, 6, 1, outfile);

    // buffer > all individual sizes
    float *buffer = new float[hdr.nx + hdr.ny + hdr.nz];
    int size;

    /// outsides grid is one larger
    size = hdr.nz + 1;
    fwrite(&size, sizeof(int), 1, outfile);
    size = hdr.ny + 1;
    fwrite(&size, sizeof(int), 1, outfile);
    size = hdr.nx + 1;
    fwrite(&size, sizeof(int), 1, outfile);

    /// cell outsides grid
    buffer[0] = hdr.zMin;
    for (i = 0; i < hdr.nz; i++)
        buffer[i + 1] = buffer[i] + dz[i];
    fwrite(buffer, sizeof(int), hdr.nz + 1, outfile);

    buffer[0] = hdr.yMin;
    for (i = 0; i < hdr.ny; i++)
        buffer[i + 1] = buffer[i] + dy[i];
    fwrite(buffer, sizeof(int), hdr.ny + 1, outfile);

    buffer[0] = hdr.xMin;
    for (i = 0; i < hdr.nx; i++)
        buffer[i + 1] = buffer[i] + dx[i];
    fwrite(buffer, sizeof(int), hdr.nx + 1, outfile);

    covWriteAttrib(outfile, 1, attribName, attribVal);
    fclose(outfile);

    //////////////////////////////////////////////////////////////////////////
    ///// GridCC

    outfile = covOpenOutFile("MeshCC.covise");
    fwrite(rectgrd, 6, 1, outfile);

    ///  cell center grid
    size = hdr.nz;
    fwrite(&size, sizeof(int), 1, outfile);
    size = hdr.ny;
    fwrite(&size, sizeof(int), 1, outfile);
    size = hdr.nx;
    fwrite(&size, sizeof(int), 1, outfile);

    /// cell centers  Z
    buffer[0] = hdr.zMin + 0.5f * dz[0];
    for (i = 1; i < hdr.nz; i++)
        buffer[i] = buffer[i - 1] + 0.5f * (dz[i - 1] + dz[i]);
    fwrite(buffer, sizeof(int), hdr.nz, outfile);

    /// cell centers  Y
    buffer[0] = hdr.yMin + 0.5f * dy[0];
    for (i = 1; i < hdr.ny; i++)
        buffer[i] = buffer[i - 1] + 0.5f * (dy[i - 1] + dy[i]);
    fwrite(buffer, sizeof(int), hdr.ny, outfile);

    /// cell centers  X
    buffer[0] = hdr.xMin + 0.5f * dx[0];
    for (i = 1; i < hdr.nx; i++)
        buffer[i] = buffer[i - 1] + 0.5f * (dx[i - 1] + dx[i]);
    fwrite(buffer, sizeof(int), hdr.nx, outfile);

    covWriteAttrib(outfile, 1, attribName, attribVal);
    fclose(outfile);

    //////////////////////////////////////////////////////////////////////////
    /// group info as float
    outfile = covOpenOutFile("MatGrp_F.covise");
    const char *strsdt = "STRSDT";
    fwrite(strsdt, 6, 1, outfile);
    size = hdr.nz;
    fwrite(&size, sizeof(int), 1, outfile);
    size = hdr.ny;
    fwrite(&size, sizeof(int), 1, outfile);
    size = hdr.nx;
    fwrite(&size, sizeof(int), 1, outfile);
    float *flgroup = new float[numCells];
    for (i = 0; i < numCells; i++)
        flgroup[i] = matGroups[materials[i]];
    fwrite(flgroup, sizeof(float), numCells, outfile);
    covWriteAttrib(outfile, 1, attribName, attribVal);
    fclose(outfile);

    //////////////////////////////////////////////////////////////////////////
    /// matID info as float
    outfile = covOpenOutFile("MatID_F.covise");
    fwrite(strsdt, 6, 1, outfile);
    size = hdr.nz;
    fwrite(&size, sizeof(int), 1, outfile);
    size = hdr.ny;
    fwrite(&size, sizeof(int), 1, outfile);
    size = hdr.nx;
    fwrite(&size, sizeof(int), 1, outfile);
    for (i = 0; i < numCells; i++)
        flgroup[i] = matID[materials[i]];
    fwrite(flgroup, sizeof(float), numCells, outfile);
    covWriteAttrib(outfile, 1, attribName, attribVal);
    fclose(outfile);

    //////////////////////////////////////////////////////////////////////////
    /// mat Idx info as float
    outfile = covOpenOutFile("MatIdx_F.covise");
    fwrite(strsdt, 6, 1, outfile);
    size = hdr.nz;
    fwrite(&size, sizeof(int), 1, outfile);
    size = hdr.ny;
    fwrite(&size, sizeof(int), 1, outfile);
    size = hdr.nx;
    fwrite(&size, sizeof(int), 1, outfile);
    for (i = 0; i < numCells; i++)
        flgroup[i] = materials[i];
    fwrite(flgroup, sizeof(float), numCells, outfile);
    covWriteAttrib(outfile, 1, attribName, attribVal);
    fclose(outfile);

    //////////////////////////////////////////////////////////////////////////
    /// group info
    outfile = covOpenOutFile("MatGrp.covise");
    const char *intarr = "INTARR";
    fwrite(intarr, 6, 1, outfile);
    // 1 dimension
    size = 1;
    fwrite(&size, sizeof(int), 1, outfile);
    // num. Elements
    size = numCells;
    fwrite(&size, sizeof(int), 1, outfile);
    // list of dimensions
    size = numCells;
    fwrite(&size, sizeof(int), 1, outfile);
    int *group = new int[numCells];
    for (i = 0; i < numCells; i++)
        group[i] = matGroups[materials[i]];
    fwrite(group, sizeof(int), numCells, outfile);
    covWriteAttrib(outfile, 1, attribName, attribVal);
    fclose(outfile);

    //////////////////////////////////////////////////////////////////////////
    /// matID
    outfile = covOpenOutFile("MatID.covise");
    fwrite(intarr, 6, 1, outfile);
    // 1 dimension
    size = 1;
    fwrite(&size, sizeof(int), 1, outfile);
    // num. Elements
    size = numCells;
    fwrite(&size, sizeof(int), 1, outfile);
    // list of dimensions
    size = numCells;
    fwrite(&size, sizeof(int), 1, outfile);
    int *grp = new int[numCells];
    for (i = 0; i < numCells; i++)
        group[i] = matID[materials[i]];
    fwrite(grp, sizeof(int), numCells, outfile);
    covWriteAttrib(outfile, 1, attribName, attribVal);
    fclose(outfile);

    //////////////////////////////////////////////////////////////////////////
    /// matIndex
    outfile = covOpenOutFile("MatIdx.covise");
    fwrite(intarr, 6, 1, outfile);
    // 1 dimension
    size = 1;
    fwrite(&size, sizeof(int), 1, outfile);
    // num. Elements
    size = numCells;
    fwrite(&size, sizeof(int), 1, outfile);
    // list of dimensions
    size = numCells;
    fwrite(&size, sizeof(int), 1, outfile);
    for (i = 0; i < numCells; i++)
        group[i] = materials[i];
    fwrite(grp, sizeof(int), numCells, outfile);
    covWriteAttrib(outfile, 1, attribName, attribVal);
    fclose(outfile);

    return 0;
}
