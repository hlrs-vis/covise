/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MODIFY_CABIN_H
#define _MODIFY_CABIN_H

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
#include "coTetin__trianFam.h"
#include "coTetin__apprxCurve.h"
#include "coTetin__trianTol.h"
#include "coTetin__tetinFile.h"
#include "coTetin__Proj.h"
#include "coTetin__transGeom.h"
#include "coTetin__defCurve.h"
#include "coTetin__replayFile.h"
#include "coTetin__configDir.h"
#include "coTetin__Hexa.h"
#include "coTetin__OutputInterf.h"
#include "coTetin__bocoFile.h"
#include "coTetin__prescPnt.h"
#include "coTetin__getprescPnt.h"

#include "coTetin__bocoFile.h"
#include "coTetin__OutputInterf.h"

#define NUM_ALLOC_LIST 100 // number of entries in familyNameList to allocate
// in one step

class ModifyCabin : public coModule
{

private:
    //////////  member functions
    virtual int compute();
    virtual void quit();
    virtual void param(const char *name);
    virtual void postInst();

    int MAX_MOVE;

    ////////// all our parameter ports
    coFileBrowserParam *p_baseTetin, *p_configFile;
    coFileBrowserParam *p_replayFile, *p_tetinIncrFile;
    coChoiceParam *p_moveName, *p_projName, *p_solver, *p_reset;
    coFloatSliderParam *p_trans_value, *p_direction;
    coFloatParam *p_tolerance;
    coStringParam *p_caseName, *p_configDir;

    ////////// the data in- and output ports
    coOutputPort *p_tetinObject;
    coOutputPort *p_hexaObject;
    coOutputPort *p_addPartObject;
    coOutputPort *p_feedback;

    char *baseTetinName;
    char *configFileName;
    char *replayFile;
    char *configDir;
    const char *solverName;
    char *caseName;

    const char *moveNames[20], *projNames[20];
    int numMoveNames, currMoveName, resetName;
    int numProjNames, currProjName;
    float tolerance;

    /// command to execute in compute callback
    enum
    {
        NONE = 0,
        SEND_BASE_TETIN,
        TRANSLATE_GEOM,
        TRANSLATE_RESET,
        PROJ_CURVES,
        PROJ_RESET,
        CONFIG_FILE,
        CREATE_LINE,
    };
    int command;
    bool newGeom; // flag indicating that either a new base tetin or
    // new config file was selected
    void selfExec();
    int parser(char *, char **, int, char *);
    int projCurves(coTetin *, char *);
    int projPoints(coTetin *, char *);
    int transGeom();
    int resetGeom();
    int sendBaseInfo();
    const char *resetNames[2];

    struct myMoveList
    {
        const char *name;
        const char **familyNames;
        int numFamilies;
        float trans_vec[3];
        float dmin, dmax, value, oldvalue;
    };
    struct myMoveList moveList[20];

    struct myProjList
    {
        const char *name;
        const char **curveNames;
        const char *familyName;
        const char *projName;
        int numCurves;
        float trans_vec[3];
        float dmin, dmax, value, oldvalue;
    };
    struct myProjList projList[20];

public:
    ModifyCabin();
};
#endif // _MODIFY_CABIN_H
