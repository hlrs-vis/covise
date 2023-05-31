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

#include <net/message.h>
#include <grmsg/coGRKeyWordMsg.h>
#include <cover/coVRNavigationManager.h>

#include "AlkaneDatabase.h"
#include "AlkaneBuilder.h"
#include "AlkaneBuilderPlugin.h"

AlkaneBuilderPlugin *AlkaneBuilderPlugin::plugin = NULL;

AlkaneBuilderPlugin::AlkaneBuilderPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, GenericGuiObject("AlkaneBuilderPlugin")
{
    if (opencover::cover->debugLevel(0))
        fprintf(stderr, "\nAlkaneBuilderPlugin::AlkaneBuilderPlugin\n");
}

AlkaneBuilderPlugin::~AlkaneBuilderPlugin()
{
    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "\nAlkaneBuilderPlugin::~AlkaneBuilderPlugin\n");
    ///delete alkaneBuilder;
}

bool AlkaneBuilderPlugin::init()
{
    if (plugin)
        return false;

    if (opencover::cover->debugLevel(0))
        fprintf(stderr, "\nAlkaneBuilderPlugin::init\n");

    // set plugin
    AlkaneBuilderPlugin::plugin = this;

    // string on alkanebuilder panel to enter the chemical formula
    p_mode = addGuiParamBool("Build manually (true) | Build automatically(false)", false); //start in show mode
    p_reset = addGuiParamBool("Reset Atom Postitions", false);
    p_xform = addGuiParamBool("Xform", true); // start in xform mode
    p_intersect = addGuiParamBool("Intersect", false); // disable intersection at start
    p_showDescription = addGuiParamBool("Show Text", false); // disable text at start
    p_showPlane = addGuiParamBool("Show Table", false); // disable plane at start
    p_formula = addGuiParamString("Chemical Formula", "");
    p_nextPresStepAllowed = addNextPresStepAllowed(true);

    // create alkane builder
    alkaneBuilder = new AlkaneBuilder();
    opencover::coVRNavigationManager::instance()->setNavMode(opencover::coVRNavigationManager::XForm);
    alkaneBuilder->enableIntersection(false);
    alkaneBuilder->showPlane(false);

    return true;
}

void AlkaneBuilderPlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{

    GenericGuiObject::guiToRenderMsg(msg);

    if (msg.isValid()&& msg.getType() == grmsg::coGRMsg::KEYWORD)
    {
        auto &keywordmsg = msg.as<grmsg::coGRKeyWordMsg>();
        const char *keyword = keywordmsg.getKeyWord();

        if (strcmp(keyword, "presForward") == 0)
        {
            fprintf(stderr, "AlkaneBuilderPlugin::guiToRenderMsg msg=%s ---------\n", keyword);
            //alkaneBuilder->presForward();
        }
        if (strcmp(keyword, "presBackward") == 0)
        {
            fprintf(stderr, "AlkaneBuilderPlugin::guiToRenderMsg msg=%s ---------\n", keyword);
            //alkaneBuilder->presForward();
        }
        if (strcmp(keyword, "showNotReady") == 0)
        {
            alkaneBuilder->showErrorPanel();
        }
    }
}

void AlkaneBuilderPlugin::preFrame()
{
    //fprintf(stderr,"AlkaneBuilderPlugin::preFrame\n");
    alkaneBuilder->update();
    p_nextPresStepAllowed->setValue(alkaneBuilder->getStatus());
}

void AlkaneBuilderPlugin::guiParamChanged(opencover::GuiParam *guiParam)
{
    fprintf(stderr, "AlkaneBuilderPlugin::guiParamChanged(%s)\n", guiParam->getName().c_str());
    if (guiParam == p_formula)
    {
        Alkane alkane = AlkaneDatabase::Instance()->findByFormula(p_formula->getValue());
        //cerr << alkane.name << endl;
        // set new configuration
        alkaneBuilder->setAlkane(alkane);
    }

    else if (guiParam == p_reset)
    {
        if (p_reset->getValue())
            alkaneBuilder->reset();
    }
    else if (guiParam == p_mode)
    {
        if (p_mode->getValue())
            alkaneBuilder->setModeBuild(true);
        else
            alkaneBuilder->setModeBuild(false);
    }
    else if (guiParam == p_xform)
    {
        if (p_xform->getValue())
        {
            opencover::coVRNavigationManager::instance()->setNavMode(opencover::coVRNavigationManager::XForm);
        }
        else
        {
            osg::Matrix identm;
            opencover::cover->setXformMat(identm);
            opencover::cover->setScale(1.0);
            opencover::coVRNavigationManager::instance()->setNavMode(opencover::coVRNavigationManager::NavNone);
        }
    }

    else if (guiParam == p_intersect)
    {
        if (p_intersect->getValue())
        {
            alkaneBuilder->enableIntersection(true);
        }
        else
        {
            alkaneBuilder->enableIntersection(false);
        }
    }

    else if (guiParam == p_showPlane)
    {
        if (p_showPlane->getValue())
        {
            alkaneBuilder->showPlane(true);
        }
        else
        {
            alkaneBuilder->showPlane(false);
        }
    }

    else if (guiParam == p_showDescription)
    {
        if (p_showDescription->getValue())
        {
            alkaneBuilder->showInstructionText(true);
            alkaneBuilder->showStatusText(true);
        }
        else
        {
            alkaneBuilder->showInstructionText(false);
            alkaneBuilder->showStatusText(false);
        }
    }
}

COVERPLUGIN(AlkaneBuilderPlugin)
