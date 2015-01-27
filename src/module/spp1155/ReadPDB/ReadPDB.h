/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2004 ZAIK/RRZK  ++
// ++ Description: ReadPDB module                                         ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                       Thomas van Reimersdahl                        ++
// ++               Institute for Computer Science (Prof. Lang)           ++
// ++                        University of Cologne                        ++
// ++                         Robert-Koch-Str. 10                         ++
// ++                             50931 Köln                              ++
// ++                                                                     ++
// ++ Date:  26.09.2004                                                   ++
// ++**********************************************************************/

#ifndef _READ_PDB_H
#define _READ_PDB_H

// includes
#include <stdlib.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <api/coModule.h>
using namespace covise;
#include <appl/ApplInterface.h>
using namespace covise;
#include <util/coviseCompat.h>

class vtkMyPDBReader;

class ReadPDB : public coModule
{
public:
    ReadPDB(int argc, char *argv[]);

private:
    // main
    int compute(const char *port);

    // parameters
    char *m_filename;
    void readPDBFile();

    // virtual methods
    virtual void param(const char *name, bool inMapLoading);

    bool fileExists(const char *filename);

private:
    vtkMyPDBReader *m_pReader;

    coOutputPort *m_portPoints;
    coOutputPort *m_portBondsLines;
    coOutputPort *m_portGroupsLines;
    coOutputPort *m_portAtomType;

    coFileBrowserParam *m_pParamFile;
    coBooleanParam *m_pTime;
    coIntScalarParam *m_pTimeMin;
    coIntScalarParam *m_pTimeMax;

    int m_iTimestep;
    int m_iTimestepMin;
    int m_iTimestepMax;

    std::string m_sFileNamePattern;
};
#endif
