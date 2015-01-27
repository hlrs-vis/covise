/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_DASIM_H
#define _READ_DASIM_H
/**************************************************************************\ 
 **                                                   	      (C)2002 RUS **
 **                                                                        **
 ** Description: READ SoundVol result files             	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Uwe Woessner                                                   **                             **
 **                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <api/coModule.h>
using namespace covise;

#define TYPELENGTH 3

struct AtomColor
{
    char type[TYPELENGTH];
    float color[4];
};

class Displacement : public coModule
{

private:
    virtual int compute(const char *port);

    coInputPort *m_portPoints;
    coInputPort *m_portID;
    coOutputPort *m_portDisplacement;

    coIntScalarParam *m_pTimestep;

public:
    Displacement(int argc, char *argv[]);
    virtual ~Displacement();
};
#endif
