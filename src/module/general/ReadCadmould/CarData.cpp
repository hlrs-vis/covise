/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS CarData
//
// This class reads filling data
//
// Initial version: 2002-05-06 [sk]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "CarData.h"

#include <util/coviseCompat.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef WIN32
#define DIR_SEP '\\'
#else
#define DIR_SEP '/'
#endif

#undef VERBOSE

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
CarData::CarData(const char *path,
                 bool byteswap)

    : CadmouldData(byteswap)
{
    d_path = new char[strlen(path) + 1];
    strcpy(d_path, path);

    d_state = 0;
    d_numFields = 0;
    d_numVert = 0;
    d_fieldNames = NULL;
    d_fieldTypes = NULL;
    title = NULL;
    buf = NULL;
    fpos = 0;

    // check that file is there
    d_file = fopen(d_path, "r");
    if (!d_file)
    {
        if (errno)
            d_state = errno;
        else
            d_state = -1;
        return;
    }

    struct stat statRec;
    if (0 != fstat(fileno(d_file), &statRec))
    {
        if (errno)
            d_state = errno;
        else
            d_state = -1;
        return;
    }
    // Read Header
    // @@@ assume binary
    if (fread(&car_hdr, sizeof(CAR_HEADER), 1, d_file) != 1)
    {
        fprintf(stderr, "ReadCadmould: fread for header failed\n");
    }

#ifdef VERBOSE
    fprintf(stderr, "header content: \nVersion: %s\nProg: %s\n", car_hdr.ver, car_hdr.prg);
#endif

    if (byte_swap)
    {
        byteSwap((sizeof(CAR_HEADER) - 40) / sizeof(float), &car_hdr.tim);
    }

#ifdef VERBOSE
    fprintf(stderr, "File: %s\nTime: %f\n", d_path, car_hdr.tim);
    fprintf(stderr, "Level: %f\nNumLines: %d\nNumCol: %d\n", car_hdr.lev, car_hdr.lin, car_hdr.num_col);
    fprintf(stderr, "Flags: %d\nMod: %d\n", car_hdr.floatg, car_hdr.mod);
#endif

    d_numVert = ((car_hdr.lin + 3) / 4) * 4;
    d_numFields = car_hdr.num_col;

    d_fieldTypes = new CadmouldData::FieldType[d_numFields];

    int i;
    signed char typ;
    for (i = 0; i < d_numFields; i++)
    {
        if (fread(&typ, sizeof(char), 1, d_file) != 1)
        {
            fprintf(stderr, "ReadCadmould: fread for typ failed\n");
        }
        if (typ > 0)
        {
            d_fieldTypes[i] = CadmouldData::SCALAR_FLOAT;
        }
        else
        {
            d_fieldTypes[i] = CadmouldData::SCALAR_INT;
        }
        if (typ != 1 && typ != -1)
        {
            cerr << "Only vertex based data supported!" << endl << "type : " << (int)typ << " in file: "
                 << path << " at column: " << i << endl;
            d_state = -1;
            return;
        }
    }

    fseek(d_file, sizeof(STR) * d_numFields, SEEK_CUR);

    title = new STR[d_numFields];
    d_fieldNames = new char *[d_numFields];
    for (i = 0; i < d_numFields; i++)
    {
        readSTR(&title[i]); //title
        d_fieldNames[i] = (char *)&title[i].txt;
    }

#ifdef VERBOSE
    fprintf(stderr, "opened %s : %d nodes\n", path, d_numVert);
#endif
    fclose(d_file);
}

void
CarData::readSTR(STR *str)
{
    int dummy;
    if (fread(&dummy, sizeof(int), 1, d_file) != 1)
    {
        fprintf(stderr, "CarData::readSTR: fread1 failed\n");
    }
    if (fread(&str->txt, sizeof(char), 36, d_file) != 36)
    {
        fprintf(stderr, "CarData::readSTR: fread2 failed\n");
    }
}

CadmouldData::FieldType
CarData::getFieldType(int fieldNo)
{
    if (d_fieldTypes)
    {
        return d_fieldTypes[fieldNo];
    }
    else
    {
        return CadmouldData::SCALAR_INT;
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CarData::~CarData()
{
    delete[] title;
    delete[] d_fieldNames;
    delete[] d_fieldTypes;
    delete[] buf;
    delete[] d_path;
}

int
CarData::numTimeSteps()
{
    return 0;
}

int
CarData::numFields()
{
    return d_numFields;
}

int
CarData::numVert()
{
    return d_numVert;
}

const char *
CarData::getName(int fieldNo)
{
    if (d_fieldNames)
    {
        return d_fieldNames[fieldNo];
    }
    else
    {
        return NULL;
    }
}

int
CarData::percent(int)
{
    return -1;
}

int
CarData::readField(int fieldNo, int, void *buffer)
{
    int i;

    d_state = 0;

    if (!buf)
    {
        // check that file is there
        d_file = fopen(d_path, "r");
        if (!d_file)
        {
            if (errno)
                d_state = errno;
            else
                d_state = -1;
            return d_state;
        }

        struct stat statRec;
        if (0 != fstat(fileno(d_file), &statRec))
        {
            if (errno)
                d_state = errno;
            else
                d_state = -1;
            return 0;
        }

        buf = new char[statRec.st_size];
        if (!buf)
        {
            cerr << "Not enough memory" << endl;
            return 0;
        }
        if (fread(buf, statRec.st_size, 1, d_file) != 1)
        {
            fprintf(stderr, "CarData::readField: fread failed\n");
        }
        fclose(d_file);
        // if( byte_swap ) {
        //    byteSwap(statRec.st_size/sizeof(int)+sizeof(CAR_HEADER), buf);
        // }
    }

    fpos = sizeof(CAR_HEADER); // pos after header

    getObject(&col.typ, sizeof(char), fieldNo);

    getString(&col.fmt, fieldNo);
    getString(&col.tit, fieldNo);
    getString(&col.uni, fieldNo);
    getString(&col.xpl, fieldNo);

    getObject(&col.cmp, sizeof(int), fieldNo);
    if (byte_swap)
    {
        byteSwap(1, &col.cmp);
    }

    getObject(&col.bit, sizeof(char), fieldNo);

    getValue(&col.sca, col.typ, fieldNo);
    getValue(&col.min, col.typ, fieldNo);
    getValue(&col.max, col.typ, fieldNo);

#ifdef VERBOSE
    fprintf(stderr, "FLOAT?: %d\n", col.typ);
    fprintf(stderr, "Titel: %s\nUnit: %s\nxpl: %s\n", col.tit.txt, col.uni.txt, col.xpl.txt);
    fprintf(stderr, "Flags: %d\nMod: %d\n", col.cmp, col.bit);
    if (d_fieldTypes[fieldNo] == CadmouldData::SCALAR_INT)
    {
        fprintf(stderr, "Scale: %d\nMin: %d\nMax: %d\n", col.sca.i, col.min.i, col.max.i);
    }
    else
    {
        fprintf(stderr, "Scale: %f\nMin: %f\nMax: %f\n", col.sca.f, col.min.f, col.max.f);
    }
#endif

    //memset( buffer, 0, sizeof(float)*car_hdr.lin);
    if (d_fieldTypes[fieldNo] == CadmouldData::SCALAR_FLOAT)
    {
        float *res = (float *)buffer;
        for (i = 0; i < car_hdr.lin; i++)
        {
            getObject(&res[i], sizeof(float), fieldNo);
            if (byte_swap)
            {
                byteSwap(1, &res[i]);
            }
            // check for corruption
            if (res[i] < col.min.f)
            {
                if (res[i] <= -1.0e+36 && col.sca.f != 0.0)
                {
                    res[i] = -1.0 / col.sca.f;
                }
                else
                {
                    res[i] = col.min.f;
                }
            }
            else if (res[i] > col.max.f)
            {
                res[i] = col.max.f;
            }

            res[i] *= col.sca.f;
        }
    }
    else
    {
        int *res = (int *)buffer;
        for (i = 0; i < car_hdr.lin; i++)
        {
            getObject(&res[i], sizeof(int), fieldNo);
            if (byte_swap)
            {
                byteSwap(1, &res[i]);
            }
            // check for corruption
            if (res[i] < col.min.i)
            {
                res[i] = col.min.i;
            }
            else if (res[i] > col.max.i)
            {
                res[i] = col.max.i;
            }

            res[i] *= col.sca.i;
        }
    }

    return 0;
}

void
CarData::getString(STR *tgt, int fieldNo)
{
    getObject(tgt, sizeof(STR), fieldNo);
}

void
CarData::getObject(void *tgt, size_t size, int fieldNo)
{
    memcpy(tgt, &buf[fpos + fieldNo * size], size);
    fpos += car_hdr.num_col * size;
}

void
CarData::getValue(VAL *tgt, char type, int fieldNo)
{
    if (type > 0)
    {
        getObject(&tgt->f, sizeof(float), fieldNo);
        if (byte_swap)
        {
            byteSwap(1, &tgt->f);
        }
    }
    else
    {
        getObject(&tgt->i, sizeof(int), fieldNo);
        if (byte_swap)
        {
            byteSwap(1, &tgt->i);
        }
    }
}
