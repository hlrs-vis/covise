/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2011 Visenso  **
 **                                                                        **
 ** Description: AtomBuilderPlugin                                         **
 **              for CyberClassroom                                        **
 **                                                                        **
 ** Author: C. Spenrath, D. Rainer                                         **
 **                                                                        **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _ATOM_BUILDER_PLUGIN_H
#define _ATOM_BUILDER_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>

#include <config/CoviseConfig.h>
#include <vrbclient/VRBClient.h>
#include <PluginUtil/GenericGuiObject.h>

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Material>
using namespace opencover;
using namespace covise;

class AtomBuilder;

class AtomBuilderPlugin : public coVRPlugin, public GenericGuiObject
{
public:
    // constructor destructor
    AtomBuilderPlugin();
    virtual ~AtomBuilderPlugin();

    // variables of class
    static AtomBuilderPlugin *plugin;

    // inherit from coVRPlugin or MenuListener
    virtual bool init();
    virtual void guiToRenderMsg(const char *msg);
    virtual void preFrame();

protected:
    void guiParamChanged(GuiParam *guiParam);

private:
    GuiParamString *p_element;
    GuiParamBool *p_puzzleMode;
    GuiParamBool *p_nextPresStepAllowed;
    AtomBuilder *atomBuilder;
};

#endif
