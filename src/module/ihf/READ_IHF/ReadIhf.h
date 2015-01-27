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
#define Kurzname "ReadIhf"
#define Copyright "(c) 2002 RUS Rechenzentrum der Uni Stuttgart"
#define Autor "Andreas Kopecki"
#define letzteAenderung "ak - 07.03.2003"

/**
 * This is a simple reader for a special volume input format,
 * used in a master thesis at the IHF.
 * File format is as follows:
 * HEADER: 10 byte;
 *         int16 (2 bytes each) : x-dimension, y-dimension, z-dimension,
 *         float (4 bytes)      : spacing (cubic) in mm
 * BODY  : vector of floats (x, y, z) (6 bytes per entry)
 *         runs [[[x]y]z]
 */

#ifdef _STANDARD_C_PLUS_PLUS
#include <fstream>
using namespace std;
#else
#include <fstream.h>
#endif

#include <api/coModule.h>
using namespace covise;

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

class ReadIhf : public coModule
{

public:
    ReadIhf();
    virtual ~ReadIhf();

private:
    virtual int compute(void);
    virtual void quit(void);
    void param(const char *name);

    const char *filename;

    coOutputPort *gridPort;
    coOutputPort *dataPort;
    coFileBrowserParam *fileParam;

    coDoUniformGrid *gridDataObject;
    coDoVec3 *vectorDataObject; // output object for vector data
    const char *vectorDataName; // output object name assigned by the controller

    uint16_t xDimension;
    uint16_t yDimension;
    uint16_t zDimension;

    float gridspacing; // mm

    char infobuf[500]; // buffer for COVISE info and error messages
};
#endif
