/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READIFF_H
#define _READIFF_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                      (C)2005 HLRS   ++
// ++ Description: ReadIFF module                                         ++
// ++                                                                     ++
// ++ Author:  Uwe                                                        ++
// ++                                                                     ++
// ++                                                                     ++
// ++ Date:  10.2005                                                      ++
// ++**********************************************************************/

#include <api/coModule.h>
using namespace covise;

#define LINE_SIZE 1024
#define MAXTIMESTEPS 2048

class ReadIFF : public coModule
{
public:
    ReadIFF(int argc, char *argv[]);

private:
    //////////  inherited member functions
    virtual int compute(const char *port);

    ////////// ports
    coOutputPort *m_portPoints;
    coOutputPort *m_portColors;

    ///////// params
    coFileBrowserParam *m_pParamFile;

    ////////// member variables;
    int maxNumberOfMolecules;
    int numberOfTimesteps;
    char *m_filename;
};
#endif
