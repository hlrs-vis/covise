/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_STL_H
#define _READ_STL_H

// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadSTL
//
// This module interpolates data values from Cell to Vertex
// based data representation
//
// Initial version: 2002-07-17 we
// +++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++
// Changes:
#include <util/coviseCompat.h>
#include <api/coModule.h>
using namespace covise;

class ReadStl : public coModule
{
public:
    typedef struct
    {
        float x, y, z;
    } Vect3;

    typedef struct
    {
        float r;
        float g;
        float b;
        float a;
    } Color;

    struct Facet
    {
        Vect3 norm;
        Vect3 vert[3];
        uint16_t colors;
    } facet;

    enum Format
    {
        STL_ASCII,
        STL_BINARY,
        STL_BINARY_BS,
        STL_NONE
    };
    enum ColType
    {
        AUTO = 0,
        MAGICS,
        VISCAM,
        ONE
    } colType;

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *paraName, bool inMapLoading);

    // ports
    coOutputPort *p_polyOut;
    coOutputPort *p_normOut;
    coOutputPort *p_linesOut;
    coOutputPort *p_colorOut;

    // parameter
    coFileBrowserParam *p_filename;
    coChoiceParam *p_format;
    coChoiceParam *p_colType;
    coStringParam *p_color;
    coBooleanParam *p_removeDoubleVert;
    coBooleanParam *p_showFeatureLines;
    coFloatSliderParam *p_angle;
    coBooleanParam *p_flipNormals;

    // utility functions
    void readHeader();
    int readBinary();
    int readASCII();

    // covise-specific calls are
    // as far as possible lumped together in this function
    void outputObjects(vector<float> &x, vector<float> &y, vector<float> &z,
                       vector<int> &connList, vector<int> &elemList,
                       vector<float> &nx, vector<float> &ny, vector<float> &nz,
                       vector<float> &lx, vector<float> &ly, vector<float> &lz,
                       vector<int> &vl, vector<int> &ll, vector<Color> &colors);

    // which format was identified (if so)
    Format d_format;

    // already opened file, alway rewound after use
    FILE *d_file;

public:
    ReadStl(int argc, char *argv[]);
    virtual ~ReadStl();
};
#endif
