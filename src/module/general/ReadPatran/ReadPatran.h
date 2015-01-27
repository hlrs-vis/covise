/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_PATRAN_NEW_H
#define _READ_PATRAN_NEW_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE Tube application module                           **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
 **                                                                        **
 **                                                                        **
 ** Author:  Reiner Beller, Sasha Cioringa                                 **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
 ** Date:  08.11.00                                                        **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;

#include "NeutralFile.h"
#include "NodalFile.h"
#include "ElementFile.h"
#include "ElementAsciiFile.h"
#include "StepFile.h"

class Patran : public coModule
{

private:
    virtual int compute(const char *port);
    virtual void param(const char *, bool inMapLoading);
    // parameters

    coFileBrowserParam *p_gridpath;
    coFileBrowserParam *p_displpath;
    coFileBrowserParam *p_nshpath;
    coFileBrowserParam *p_elempath;
    coChoiceParam *p_option;
    coIntScalarParam *p_timesteps;
    coIntScalarParam *p_skip;
    coIntScalarParam *p_columns;

    // ports
    coOutputPort *p_outPort1;
    coOutputPort *p_outPort2;
    coOutputPort *p_outPort3;
    coOutputPort *p_outPort4;

    // private data
    NeutralFile *gridFile;
    NodalFile *nodal_displFile;
    NodalFile *nodal_stressFile;
    ElementFile *elemFile;
    ElementAscFile *elemAscFile;

    char init_path[100];
    const char *grid_path;
    const char *displ_path;
    const char *nsh_path;
    const char *elem_path;

    int has_timesteps;

public:
    Patran(int argc, char *argv[]);
    virtual ~Patran();
};
#endif
