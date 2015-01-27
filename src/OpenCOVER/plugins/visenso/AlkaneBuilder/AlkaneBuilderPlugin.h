/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2012 Visenso  **
 **                                                                        **
 ** Description: AlkaneBuilderPlugin                                       **
 **              for CyberClassroom                                        **
 **                                                                        **
 ** Author: D. Rainer                                                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _ALKANE_BUILDER_PLUGIN_H
#define _ALKANE_BUILDER_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>

#include <config/CoviseConfig.h>
#include <vrbclient/VRBClient.h>
#include <PluginUtil/GenericGuiObject.h>

class AlkaneBuilder;

class AlkaneBuilderPlugin : public opencover::coVRPlugin, public opencover::GenericGuiObject
{
public:
    // AlkaneBuilderPlugin destructor
    AlkaneBuilderPlugin();
    virtual ~AlkaneBuilderPlugin();

    // variables of class
    static AlkaneBuilderPlugin *plugin;

    // inherit from coVRPlugin or MenuListener
    virtual bool init();
    virtual void guiToRenderMsg(const char *msg);
    virtual void preFrame();

protected:
    void guiParamChanged(opencover::GuiParam *guiParam);

private:
    opencover::GuiParamString *p_formula;
    opencover::GuiParamBool *p_reset; // send a reset message in this presentation step
    opencover::GuiParamBool *p_mode; // show a molekule (false) or build a molekule (true)
    opencover::GuiParamBool *p_xform; // allow xform
    opencover::GuiParamBool *p_intersect; // allow intersect
    opencover::GuiParamBool *p_showPlane; //show plane in whhich atom balls move
    opencover::GuiParamBool *p_showDescription; // show decsription text
    opencover::GuiParamBool *p_nextPresStepAllowed; // allow forward to next step
    AlkaneBuilder *alkaneBuilder;
};

#endif
