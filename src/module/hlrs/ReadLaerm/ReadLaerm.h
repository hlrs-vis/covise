/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_Laerm_H
#define _READ_Laerm_H
/**************************************************************************\
**                                                   	      (C)2002 RUS **
**                                                                        **
** Description: READ Laerm result files             	                  **
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

#define NUM_DATA_PORTS 5

class ReadLaerm : public coReader
{

private:
    struct LaermHeader
    {
        float ox;
		float oy;
		float gridSizeX;
		float gridSizeY;
        int ndimx;
        int ndimy;
        int ndimz;
		int npar;
		std::vector<std::string> variables;
    };
	float *xc, *yc, *zc, *gc, *LD, *Nacht, *l3, *l4;

    //  member functions
    virtual int compute(const char *port);
    int readHeader(const char *filename);
    void freeHeader();
    void initHeader();
    virtual void param(const char *paraName, bool inMapLoading);
    int headerState; // can be FAIL=-1 or SUCCESS=0 or 1 (if not read)
	LaermHeader header;
	size_t dataPos;

    coDistributedObject *makegrid(const char *objName);
    coDistributedObject *makeDataObject(const char *objName, int paramNumber);
    FILE *fp;
    std::string fileName;

public:
    enum ParamTypes
    {
        Laerm_BROWSER,
        MESHPORT,
        DPORT1,
        DPORT2,
        DPORT3,
        DPORT4,
        DPORT5
    };
    ReadLaerm(int argc, char *argv[]);
    virtual ~ReadLaerm();
};

#endif
