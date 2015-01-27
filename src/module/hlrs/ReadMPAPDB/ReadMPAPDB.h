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

#include <config/coConfig.h>

#define TYPELENGTH 3

struct AtomColor
{
    char type[TYPELENGTH];
    float color[4];
};

class ReadMPAPDB : public coModule
{

private:
    virtual int compute(const char *port);

    coOutputPort *m_portPoints;
    coOutputPort *m_portAtomType;
    coOutputPort *m_portAtomID;

    coFileBrowserParam *m_pParamFile;
    coBooleanParam *m_pUseIDFromFile;

    coIntScalarParam *m_pNTimesteps;
    coIntScalarParam *m_pStepTimesteps;

    coConfigGroup *m_mapConfig;

    map<string, int> AtomID;
    map<string, int>::iterator it;

    std::vector<string> m_atomtype;
    std::vector<AtomColor> m_rgb;
    std::vector<float> m_radius;

public:
    ReadMPAPDB(int argc, char *argv[]);
    virtual ~ReadMPAPDB();
};
#endif
