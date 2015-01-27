/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_A3R_H
#define _READ_A3R_H
/****************************************************************************\ 
 * ReadA3R module
 *
 *
\****************************************************************************/

#include <api/coModule.h>
using namespace covise;

class ReadA3R : public coModule
{

private:
    float cx, cy, cz; // coodinates of the cubes p_center
    float sMin, sMax, sVal; // edge length of the cube

    coFileBrowserParam *p_file;

    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);
    virtual void postInst();

    int read_a3r(const char *fname, vector<float> &x, vector<float> &y, vector<float> &z);

    coDoPoints *create_do_points(const char *fname, const char *do_name);
    bool indexed_file_name(char *s, int n);

    //  Ports
    coOutputPort *p_ptsOut;

    coBooleanParam *p_use_timesteps;
    coIntScalarParam *p_first_step;
    coIntScalarParam *p_last_step;
    coIntScalarParam *p_inc;

public:
    ReadA3R(int argc, char *argv[]);
    virtual ~ReadA3R();
};

struct Vect3D
{
    struct SFLOAT
    {
        float x;
        float y;
        float z;
    };
};

struct A3R_HEADER
{
    A3R_HEADER(); // Initialization of the structure members

    char file_type[4]; // "a3r"
    int count; // (4 bytes) number of particles
    int data_start; // (4 bytes) address of the start of the particles data
    char version[10]; // (10 bytes) version of a3r format
    double r; // (8 bytes) particle radius
    int count_1; // (4 bytes) reserved for the future use;
};
#endif
