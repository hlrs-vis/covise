/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_EAS_H
#define _READ_EAS_H
/**************************************************************************\
**                                                   	      (C)2002 RUS **
**                                                                        **
** Description: READ EAS result files             	                  **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: Uwe Woessner                                                   **                             **
**                                                                        **
\**************************************************************************/

#include "reader/coReader.h"
#include "reader/ReaderControl.h"
using namespace covise;
#include <util/coviseCompat.h>

#ifdef _WIN32
#ifndef _OFF_T_DEFINED
typedef int64_t off_t;
#endif
#define fseeko _fseeki64
#define ftello _ftelli64
#endif

#define ATTRLEN 10
#define UDEFLEN 20

#define NUM_DATA_PORTS 5

class ReadEAS : public coReader
{

private:
    struct EASHeader
    {
        int64_t type;
        int64_t datasize;
        int64_t nzs;
        int64_t npar;
        int64_t ndim1;
        int64_t ndim2;
        int64_t ndim3;
        int64_t attribMode;
        int64_t geomModeTime;
        int64_t geomModeParam;
        int64_t geomModeDim1;
        int64_t geomModeDim2;
        int64_t geomModeDim3;

        int64_t sizeTime;
        int64_t sizeParam;
        int64_t sizeDim1;
        int64_t sizeDim2;
        int64_t sizeDim3;

        int64_t userData;

        int64_t UserDataCharSize;
        int64_t UserDataIntSize;
        int64_t UserDataRealSize;
    };
    struct Kennsatz
    {
        char kennung[20];
        EASHeader header;
        //Zeitschritt Feld: nzs x 8 byte
        int64_t *timestepData;
        char **timestepAttrib;
        //wenn attribute mode = EAS3_ALL_ATTR
        char **paramAttrib;
        char **dimAttrib;

        //wenn geometry mode > EAS3_NO_G

        double *timeData;
        double *paramData;
        double *dim1Data;
        double *dim2Data;
        double *dim3Data;

        char **userDataChar;
        int64_t *userDataInt;
        double *userDataReal;
    };

    enum FileType
    {
        EAS2 = 1,
        EAS3 = 2
    };
    enum FloatSize
    {
        IEEES = 1,
        IEEED = 2,
        IEEEQ = 3
    };
    enum AttribMode
    {
        EAS3_NO_ATTR = 1,
        EAS3_ALL_ATTR = 2
    };
    enum GeomMode
    {
        EAS3_NO_G = 1,
        EAS3_X0DX_G = 2,
        EAS3_UDEF_G = 3,
        EAS3_ALL_G = 4,
        EAS3_FULL_G = 5
    };
    enum UserData
    {
        EAS3_NO_UDEF = 1,
        EAS3_ALL_UDEF = 2,
        EAS3_INT_UDEF = 3
    };

    //  member functions
    virtual int compute(const char *port);
    int readHeader(const char *filename);
    void freeHeader();
    void initHeader();
    virtual void param(const char *paraName, bool inMapLoading);
    int headerState; // can be FAIL=-1 or SUCCESS=0 or 1 (if not read)

    off_t dataPos;
    Kennsatz ks;

    coDistributedObject *makegrid(const char *objName);
    coDistributedObject *makeDataObject(const char *objName, int paramNumber);
    FILE *fp;
    std::string fileName;

public:
    enum ParamTypes
    {
        EAS_BROWSER,
        MESHPORT,
        DPORT1,
        DPORT2,
        DPORT3,
        DPORT4,
        DPORT5
    };
    ReadEAS(int argc, char *argv[]);
    virtual ~ReadEAS();
};

#endif
