/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ReadIVDTubes_H
#define _ReadIVDTubes_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                      (C)2005 HLRS   ++
// ++ Description: ReadIVDTubes module                                      ++
// ++                                                                     ++
// ++ Author:  Uwe                                                        ++
// ++                                                                     ++
// ++                                                                     ++
// ++ Date:  2.2006                                                      ++
// ++**********************************************************************/

#include <api/coModule.h>
using namespace covise;

#define LINE_SIZE 1024
#define MAXTIMESTEPS 2048
#define NUM_SCALAR 14

class ReadIVDTubes : public coModule
{
public:
    ReadIVDTubes(int argc, char *argv[]);

private:
    //////////  inherited member functions
    virtual int compute(const char *port);

    char line[LINE_SIZE];
    char bfr[2048];

    FILE *file;
    char *c;
    ////////// ports
    coOutputPort *m_portLines;
    coOutputPort **m_varPorts;

    ///////// params
    coFileBrowserParam *m_pParamFile;

    ////////// member variables;
    int numLines;
    int numPoints;
    int *numSegments;

    char *m_filename;
};
#endif
