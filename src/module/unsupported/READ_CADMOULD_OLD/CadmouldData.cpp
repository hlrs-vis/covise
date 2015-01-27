/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CadmouldData.h"
#include <appl/ApplInterface.h>
#include <ctype.h>

CadmouldData::CadmouldData()
{
    elem[0] = elem[1] = elem[2] = NULL;
    thickness = NULL;
    points[0] = points[1] = points[2] = NULL;
    value = NULL;
    connect = NULL;
    fill_time = NULL;
}

int CadmouldData::init(int allocPoints, int allocElem)
{
    for (int i = 0; i < 3; i++)
    {
        if (!(elem[i] = new int[allocElem]))
            return FAIL;
        if (!(points[i] = new float[allocPoints]))
            return FAIL;
    }
    if (!(thickness = new float[allocElem]))
        return FAIL;
    if (!(value = new float[allocPoints]))
        return FAIL;
    if (!(connect = new int[allocPoints]))
        return FAIL;
    if (!(fill_time = new float[allocPoints]))
        return FAIL;

    return SUCCESS;
}

CadmouldData::~CadmouldData()
{
    for (int i = 0; i < 3; i++)
    {
        if (elem[i])
            delete[] elem[i];
        if (points[i])
            delete[] points[i];
    }
    if (thickness)
        delete[] thickness;
    if (value)
        delete[] value;
    if (connect)
        delete[] connect;
    if (fill_time)
        delete[] fill_time;
}

// check: if we find an unprintable char, it is binary
static bool isBinary(FILE *fi)
{
    bool bin = false;
    int k = 0;

    while (!feof(fi) && k < 1000 && !bin)
    {
        enum
        {
            CR = 13,
            NL = 10
        };
        char c = getc(fi);
        if (!isprint(c) && c != '\n' && c != NL && c != CR)
            bin = true;
        k++;
    }
    rewind(fi);

    return bin;
}

inline void byteswap(int &val)
{
    val = ((val & 0xff000000) >> 24)
          | ((val & 0x00ff0000) >> 8)
          | ((val & 0x0000ff00) << 8)
          | ((val & 0x000000ff) << 24);
}

inline void byteswap(float &fval)
{
    int &val = (int &)fval;
    val = ((val & 0xff000000) >> 24)
          | ((val & 0x00ff0000) >> 8)
          | ((val & 0x0000ff00) << 8)
          | ((val & 0x000000ff) << 24);
}

inline void byteswap(int *field, int numElem)
{
    int i;
    for (i = 0; i < numElem; i++)
    {
        int &val = field[i];
        val = ((val & 0xff000000) >> 24)
              | ((val & 0x00ff0000) >> 8)
              | ((val & 0x0000ff00) << 8)
              | ((val & 0x000000ff) << 24);
    }
}

inline void byteswap(float *field, int numElem)
{
    int i;
    for (i = 0; i < numElem; i++)
    {
        int &val = (int &)field[i];
        val = ((val & 0xff000000) >> 24)
              | ((val & 0x00ff0000) >> 8)
              | ((val & 0x0000ff00) << 8)
              | ((val & 0x000000ff) << 24);
    }
}

int CadmouldData::load(const char *meshfile, const char *datafile)
{
    FILE *mesh, *data;
    char in[1024];
    int i;

    ////////////////////////////////////////////////////////////////////
    /// Mesh

    mesh = Covise::fopen(meshfile, "r");
    if (NULL == mesh)
        return FAIL;
    fgets(in, 1023, mesh);
    sscanf(in, "%d %d", &no_points, &no_elements);

    // everthing is written out in 4-multiples
    int readPoints = ((no_points + 3) / 4) * 4;
    int readElem = ((no_elements + 3) / 4) * 4;

    // allocate memory including overhang for 4-filling
    if (init(readPoints, readElem) == FAIL)
        return FAIL;

    // read element descriptions
    for (i = 0; i < no_elements; i++)
    {
        int type;
        fgets(in, 1023, mesh);
        sscanf(in, "%d %d %d %f %d", elem[0] + i, elem[1] + i, elem[2] + i,
               thickness + i, &type);
    }

    // read vertex coordinates
    for (i = 0; i < no_points; i++)
    {
        fgets(in, 1023, mesh);
        sscanf(in, "%f %f %f", points[0] + i, points[1] + i, points[2] + i);
    }

    fclose(mesh);

    ////////////////////////////////////////////////////////////////////
    /// Data

    data = Covise::fopen(datafile, "r");
    if (NULL == data)
        return FAIL;

    ///////////////////////// Binary data
    if (isBinary(data))
    {
        int mustswap = 0;

        struct
        {
            int numnode, maxfill;
            float min, max;
        } hdr;

        // Read Header
        fread(&hdr, sizeof(hdr), 1, data);

        // number of nodes incorrect ? - might be byteswapped
        if (hdr.numnode != no_points)
        {
            byteswap(hdr.numnode);
            mustswap = 1;

            // still not ok: error message
            if (hdr.numnode != no_points)
            {
                return FAIL;
            }
        }

        // Fuellstand
        fread(value, sizeof(float), readPoints, data);
        if (mustswap)
            byteswap(value, readPoints);

        // Vorgänger
        fread(connect, sizeof(int), readPoints, data);
        if (mustswap)
            byteswap(connect, readPoints);

        // Skip Header
        fread(&hdr, sizeof(hdr), 1, data);

        // Time
        fread(fill_time, sizeof(float), readPoints, data);
        if (mustswap)
            byteswap(fill_time, readPoints);

        return SUCCESS; // OK
    }

    ///////////////////////// ASCII data
    else
    {
        // skip case name
        fgets(in, 1023, data);

        // Num Nodes | min | max | Anzahl Fuellbilder in Datei
        fgets(in, 1023, data);

        // Fuellstand
        for (i = 0; i < readPoints; i += 4)
        {
            fgets(in, 1023, data);
            sscanf(in, "%e %e %e %e", &value[i], &value[i + 1],
                   &value[i + 2], &value[i + 3]);
        }

        // Vorgänger
        for (i = 0; i < readPoints; i += 4)
        {
            fgets(in, 1023, data);
            sscanf(in, "%d %d %d %d", &connect[i], &connect[i + 1],
                   &connect[i + 2], &connect[i + 3]);
        }

        // Num Nodes | min | max | Anzahl Fuellbilder in Datei
        fgets(in, 127, data);

        // Fillup time
        for (i = 0; i < readPoints; i += 4)
        {
            fgets(in, 1023, data);
            sscanf(in, "%f %f %f %f", &fill_time[i], &fill_time[i + 1],
                   &fill_time[i + 2], &fill_time[i + 3]);
        }
    }
    return SUCCESS;
}
