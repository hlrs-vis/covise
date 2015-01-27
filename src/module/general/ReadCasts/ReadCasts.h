/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_CASTS_H_
#define _READ_CASTS_H_

#include <list>

#include <util/coviseCompat.h>
#include <api/coModule.h>
using namespace covise;
#include <api/coSimpleModule.h>

#ifndef YAC
class ReadCasts : public coSimpleModule
{
#else
class ReadCasts : public coFunctionModule
{
#endif
    COMODULE
public:
    ReadCasts(int argc, char *argv[]);
    virtual ~ReadCasts();

private:
    enum
    {
        NUM,
        MAT,
        COORD_HEADER,
        COORD,
        ELEM_HEADER,
        ELEM,
        DEF,
        UNKNOWN
    };

    struct material
    {
        int num;
        int h_type;
        int h_num;
        int h_end;
        int p_type;
        int p_num;
        int p_end;
        int t_type;
        int t_num;
        int t_end;
    };

    virtual int compute(const char *port);
    virtual void param(const char *paraName, bool inMapLoading);

    bool readGrid(const char *fileName, int numTimeSteps);
    bool readData(const char *fileName, int &numTimeSteps);

    int swapData(const char *fileName);

    coOutputPort *p_gridOut;
    coOutputPort *p_dataOut;
    coFileBrowserParam *p_gridFile;
    coFileBrowserParam *p_dataFile;

    std::list<struct material *> materials;
};
#endif
