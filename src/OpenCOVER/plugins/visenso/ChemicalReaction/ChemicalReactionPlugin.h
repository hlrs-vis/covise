/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2011 Visenso  **
 **                                                                        **
 ** Description: ChemicalReactionPlugin                                    **
 **              for CyberClassroom                                        **
 **                                                                        **
 ** Author: C. Spenrath                                                    **
 **                                                                        **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _CHEMICAL_REACTION_PLUGIN_H
#define _CHEMICAL_REACTION_PLUGIN_H

#include "MoleculeHandler.h"
#include "StartButton.h"
#include "ReactionArea.h"

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>

#include <config/CoviseConfig.h>
#include <vrbclient/VRBClient.h>
#include <PluginUtil/GenericGuiObject.h>
#include <cover/coHud.h>

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Material>

using namespace opencover;
using namespace covise;

class ChemicalReactionPlugin : public coVRPlugin, public GenericGuiObject
{
public:
    // constructor destructor
    ChemicalReactionPlugin();
    virtual ~ChemicalReactionPlugin();

    // variables of class
    static ChemicalReactionPlugin *plugin;

    // inherit from coVRPlugin or MenuListener
    virtual bool init();
    virtual void guiToRenderMsg(const char *msg);
    virtual void preFrame();

protected:
    void guiParamChanged(GuiParam *guiParam);

private:
    void updateOkIndicator();

    MoleculeHandler *moleculeHandler;
    ReactionArea *reactionArea;
    StartButton *startButton;

    osg::ref_ptr<osg::Node> geometryOk;
    osg::ref_ptr<osg::Node> geometryNotOk;
    osg::ref_ptr<osg::MatrixTransform> indicatorTransform;

    GuiParamBool *p_nextPresStepAllowed;
    GuiParamString *p_startMolecule[3];
    GuiParamString *p_endMolecule;

    bool validTask;

    coHud *hud; // hud for messages
    float hudTime;
};

#endif
