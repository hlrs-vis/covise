/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_OPTRES_H
#define _READ_OPTRES_H
/**************************************************************************\ 
 **                                                   	      (C)1999 RUS **
 **                                                                        **
 ** Description: Read Optres V6.0C binary files      	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Uwe Woessner                                                   **
 **                                                                        **
 ** History:                                                               **
 ** December 99         v1                                                 **                               **
 **                                                                        **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;
#define NUM_PORTS 3

class ReadOptres : public coModule
{

private:
    //  member functions
    virtual int compute();
    virtual void postInst();

    //  ports
    coOutputPort *p_bars;
    coOutputPort *p_patches;
    coOutputPort *p_volumes;
    coOutputPort *p_geometry;

    coOutputPort *p_bar_data[NUM_PORTS];
    coOutputPort *p_patch_data[NUM_PORTS];
    coOutputPort *p_volume_data[NUM_PORTS];

    coFileBrowserParam *p_filename;
    coChoiceParam *p_selection[NUM_PORTS];
    coChoiceParam *p_layer;
    coIntScalarParam *p_numTimesteps;

    coStringParam *p_partSelection;
    coStringParam *p_partMaterial;
    coStringParam *p_dataSelection;

public:
    ReadOptres();
    virtual ~ReadOptres();
};
#endif
