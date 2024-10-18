/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Revit_VrmlNodePhases_PLUGIN_H
#define _Revit_VrmlNodePhases_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2022 HLRS  **
 **                                                                          **
 ** Description: Revit Plugin VrmlInterface    **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** Mar-09  v1	    				       		                                   **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <util/coTypes.h>


#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>


using namespace vrml;


class VrmlNodePhases : public VrmlNodeChild
{

public:
    // Define the fields of Timesteps nodes
    static void initFields(VrmlNodePhases *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodePhases(VrmlScene* scene = 0);
    VrmlNodePhases(const VrmlNodePhases& n);
    virtual void addToScene(VrmlScene* s, const char*);

    virtual void render(Viewer*);

    static void update();

    static VrmlNodePhases* instance() { return theInstance; };

    void setPhase(int phase);
    void setPhase(const std::string &phaseName);
    void setNumPhases(int numphase);

private:
    // Fields
    VrmlSFInt d_numPhases;
    VrmlSFInt d_Phase;
    VrmlSFString d_PhaseName;
    static VrmlNodePhases* theInstance;
};

#endif
