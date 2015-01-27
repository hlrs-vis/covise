/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// checkin Tue Nov 13 10:26:40 MEZ 2001   Fehler in euler winkeln behoben

#ifndef _MODIFY_ADD_PART_H
#define _MODIFY_ADD_PART_H

// modify tetin geometry
// project VISiT
// Author:  M. Friedrichs, R. Lang, D. Rainer, A. Werner
// History: Dec 99  module frame
//          Feb 99  usage of projection library

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <api/coModule.h>
using namespace covise;

#include "coTetin.h"
#include "coTetin__tetinFile.h"
#include "coTetin__transGeom.h"
#include "coTetin__replayFile.h"
#include "coTetin__configDir.h"
#include "coTetin__Hexa.h"
#include "coTetin__bocoFile.h"
#include "coTetin__OutputInterf.h"

class ModifyAddPart : public coModule
{

private:
    // Maximum number of attached vents and different vent types
    enum
    {
        MAX_VENTS = 16,
        MAX_NAMES = 30
    };

    //////////  member functions
    virtual int compute();
    virtual void quit();
    virtual void param(const char *name);
    virtual void postInst();

    ////////// all our parameter ports
    coStringParam *p_ventDir;
    coChoiceParam *p_action;
    coChoiceParam *p_vent;

    ////////// the data in- and output ports
    coInputPort *inTetin;
    coOutputPort *solverText;
    coOutputPort *outPolygon;
    coOutputPort *prostarData;

    // path for vent descriptions
    char *ventFilePath;

    // no of attached vents and different vent type descriptions
    int numVentDirs, numVents, currVent;

    // The parameters for each vent
    float pos[MAX_VENTS][3]; // position of the vent
    float euler[MAX_VENTS][3]; // rotation in euler angles
    float axis[MAX_VENTS][3]; // channel axis, def is x-axis
    float coverRot[MAX_VENTS][9]; // rotation matrix as vector which is set only from COVER
    // directory name
    int currVentFile[MAX_VENTS];
    int fileid[MAX_VENTS], exist[MAX_VENTS];
    char *ventdirs[MAX_NAMES]; // directory for vent
    char *polynames[MAX_VENTS]; // object name
    coChoiceParam *p_name[MAX_VENTS];
    coFloatVectorParam *p_pos[MAX_VENTS]; // parameter position of the vents
    coFloatVectorParam *p_rot[MAX_VENTS]; // parameter for setting rotation as matrix (disabled)
    coFloatVectorParam *p_euler[MAX_VENTS]; // parameter for setting rotation as euler angles
    coDoPolygons *polyobj[MAX_VENTS];

    void createVentParam();
    void getVentDirs();
    void getNumOfVents();

    void computeEulerAngles(int i);
    void computeCoverRot(int i);

public:
    ModifyAddPart();
};
#endif // _MODIFY_ADD_PART_H
