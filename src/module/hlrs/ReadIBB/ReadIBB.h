/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                              2008      **
 **                                                                        **
 ** Description:  ReadIBB                                                  **
 **                                                                        **
 ** Covise read module for IBB GID files                                   **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner                                                  **
 **                                                                        **
 **                                                                        **
 ** Date:  19.07.08  V1.0                                                  **
 \**************************************************************************/

#include <api/coModule.h>
using namespace covise;
#include <util/coviseCompat.h>

struct coordinate
{
    float x;
    float y;
    float z;
    void set(float xc, float yc, float zc)
    {
        x = xc;
        y = yc;
        z = zc;
    };
};

struct element
{
    int v[8];
    int type;
};

class ReadIBB : public coModule
{

private:
    virtual int compute(const char *port);
    virtual void postInst();
    virtual void quit();

    FILE *gridFP;
    FILE *dataFP;
    // parameters
    coFileBrowserParam *p_geoFile;
    coFileBrowserParam *p_simFile;

    // ports
    coOutputPort *port_grid;
    coOutputPort *port_displacement;
    coOutputPort *port_velocity;
    coOutputPort *port_pressure;
    coOutputPort *port_k;
    coOutputPort *port_eps;
    coOutputPort *port_wall;
    coOutputPort *port_pressrb;
    coOutputPort *port_bila;
    coOutputPort *port_boco;
    coOutputPort *port_bcin;

    coIntScalarParam *p_numt;
    coIntScalarParam *p_skip;
    coIntScalarParam *p_firstStep;

public:
    ReadIBB(int argc, char *argv[]);
};
