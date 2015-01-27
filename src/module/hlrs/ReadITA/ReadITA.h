/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_RADIOSS_H
#define _READ_RADIOSS_H
/************************************************************************
 *									*
 *          								*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30a				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 ************************************************************************/

#define ProgrammName "Reader for ITA Hannover data"
#define Kurzname "ReadITA"
#define Copyright "(c) 2012 HLRS, Uni Stuttgart"
#define Autor "Uwe Woessner"

/************************************************************************/

#include <util/coviseCompat.h>

#include <api/coModule.h>
#include <list>
#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <do/coDoPolygons.h>
using namespace covise;

struct Amplitudes
{
    float phi;
    float theta;
    std::vector<float> values;
};

class ReadITA : public coModule
{

public:
    ReadITA(int argc, char *argv[]);
    virtual ~ReadITA();

private:
    virtual int compute(const char *port);
    virtual void quit(void);

    const char *filename;

    coOutputPort *gridPort;
    coOutputPort *dataPort;
    coFileBrowserParam *fileParam;

    coDoPolygons *gridDataObject;
    coDoFloat *scalarDataObject; // output object for vector data
    const char *scalarDataName; // output object name assigned by the controller

    char infobuf[500]; // buffer for COVISE info and error messages
    char line[20000]; // line buffer
    std::vector<Amplitudes> table;
};

#endif
