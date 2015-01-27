/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READITT_H
#define _READITT_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2004 ZAIK/RRZK  ++
// ++ Description: ReadITT module                                         ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                       Thomas van Reimersdahl                        ++
// ++               Institute for Computer Science (Prof. Lang)           ++
// ++                        University of Cologne                        ++
// ++                         Robert-Koch-Str. 10                         ++
// ++                             50931 K�n                              ++
// ++                                                                     ++
// ++ The sources for reading the molecule structures are based on        ++
// ++ the VRMoleculeViewer plugin of OpenCOVER.                           ++
// ++                                                                     ++
// ++ Date:  26.12.2004                                                   ++
// ++**********************************************************************/

#include <api/coModule.h>
using namespace covise;
#include <util/coMatrix.h>
#include <api/coSimLib.h>
#include <util/coVector.h>
#include "Molecule.h"

#define LINE_SIZE 512

class ReadITT : public coSimLib
{
public:
    ReadITT(int argc, char *argv[]);
    void enableLookAhead(bool bOn = true);
    void setLookAhead(int iTimestep = 1);

private:
    void loadData(const char *moleculepath);
    coDoLines *createBox(const char *objectName,
                         float ox, float oy, float oz, float size_x, float size_y, float size_z);

    //////////  inherited member functions
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);
    virtual void postInst();

    ////////// ports
    coOutputPort *m_portPoints;
    coOutputPort *m_portAtomType;
    coOutputPort *m_portSpheres;
    coOutputPort *m_portVolumeBox;
    coOutputPort *m_portColors;
    coOutputPort *m_portRadii;

    ///////// params
    coFileBrowserParam *m_pParamFile;
    coBooleanParam *m_pLookAhead;
    coIntScalarParam *m_pLookAheadValue;
    coIntScalarParam *m_pSleepSeconds;

    ////////// member variables
    MoleculeStructure *structure;
    int m_iLookAhead;
    int m_bLookAhead;
    int m_iSleepSeconds;
    bool m_bDoSelfExec;
};
#endif
