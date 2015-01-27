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

#define ProgrammName "Reader for IHF Volume Data"
#define Kurzname "ReadSeis"
#define Copyright "(c) 2002 RUS Rechenzentrum der Uni Stuttgart"
#define Autor "Uwe Woessner"

/************************************************************************/

#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoUniformGrid.h>

#include <api/coModule.h>
using namespace covise;

class ReadSeis : public coModule
{

public:
    ReadSeis(int argc, char *argv[]);
    virtual ~ReadSeis();

private:
    virtual int compute(const char *port);
    virtual void quit(void);

    const char *filename;

    coOutputPort *gridPort;
    coOutputPort *dataPort;
    coFileBrowserParam *fileParam;

    coDoUniformGrid *gridDataObject;
    coDoFloat *scalarDataObject; // output object for vector data
    const char *scalarDataName; // output object name assigned by the controller

    uint16_t xDimension;
    uint16_t yDimension;
    uint16_t zDimension;

    float gridspacing; // mm

    char infobuf[500]; // buffer for COVISE info and error messages
    char line[20000]; // line buffer
};

#endif
