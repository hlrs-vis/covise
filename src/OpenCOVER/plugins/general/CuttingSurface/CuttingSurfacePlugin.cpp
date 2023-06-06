/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#endif

#include <cover/coVRPluginSupport.h>
#include <cover/coInteractor.h>
#include "CuttingSurfacePlugin.h"
#include "CuttingSurfaceInteraction.h"
#include <cover/RenderObject.h>

#include <grmsg/coGRObjVisMsg.h>
#include <grmsg/coGRObjMoveInterMsg.h>
#include <grmsg/coGRObjSetCaseMsg.h>
#include <grmsg/coGRObjSetNameMsg.h>
#include <grmsg/coGRObjRestrictAxisMsg.h>
#include <grmsg/coGRObjAttachedClipPlaneMsg.h>

using namespace grmsg;
using namespace opencover;

void CuttingSurfacePlugin::newInteractor(const RenderObject *container, coInteractor *i)
{
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "\n--- coVRNewInteractor from module %s\n", i->getModuleName());
       // if (container)
       //     fprintf(stderr, "container %ld %s\n", (long int)container, container->getName());
    }
    const char *moduleName = i->getModuleName();
    if ((strncmp(moduleName, "CuttingSurfaceComp", 18) == 0) || (strncmp(moduleName, "CuttingSurface", 14) == 0))
    {
        add(container, i);
    }

#ifndef COVISE_BUILD
    if ((strncmp(moduleName, "CutGeometry", 11) == 0))
    {
        add(container, i);
    }
#endif

    ModuleFeedbackPlugin::newInteractor(container, i);
}

void CuttingSurfacePlugin::removeObject(const char *objName, bool r)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\n--- coVRRemoveObject objectName=[%s]\n", objName);

    // replace is handeled in addObject
    if (!r)
    {
        remove(objName);
    }
}
void CuttingSurfacePlugin::addNode(osg::Node *node, const RenderObject *obj)
{
    //fprintf(stderr,"CuttingSurfacePlugin::addNode %s\n", obj->getName());
    if (obj)
    {
        addNodeToCase(obj->getName(), node);
    }
}
void
CuttingSurfacePlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "\n--- CuttingSurfacePlugin::guiToRenderMsg\n");
    }
    if (msg.isValid())
    {
        switch (msg.getType())
        {
        case coGRMsg::GEO_VISIBLE:
        {
            auto &geometryVisibleMsg = msg.as<coGRObjVisMsg>();
            const char *objectName = geometryVisibleMsg.getObjName();

            if (cover->debugLevel(3))
                fprintf(stderr, "CuttingSurfacePlugin::guiToRenderMsg coGRMsg::GEO_VISIBLE object=%s visible=%d\n", objectName, geometryVisibleMsg.isVisible());

            handleGeoVisibleMsg(objectName, geometryVisibleMsg.isVisible());
            updateInteractorVisibility(objectName);
        }
        break;
        case coGRMsg::INTERACTOR_VISIBLE:
        {
            auto &interactorVisibleMsg = msg.as<coGRObjVisMsg>();
            const char *objectName = interactorVisibleMsg.getObjName();
            if (cover->debugLevel(3))
                fprintf(stderr, "CuttingSurfacePlugin::guiToRenderMsg coGRMsg::INTERACTOR_VISIBLE object=%s\n", objectName);
            handleInteractorVisibleMsg(objectName, interactorVisibleMsg.isVisible());
        }
        break;
        case coGRMsg::SET_CASE:
        {
            auto &setCaseMsg = msg.as<coGRObjSetCaseMsg>();
            const char *objectName = setCaseMsg.getObjName();
            if (cover->debugLevel(3))
                fprintf(stderr, "CuttingSurfacePlugin::guiToRenderMsg coGRMsg::SET_CASE object=%s\n", objectName);
            const char *caseName = setCaseMsg.getCaseName();
            handleSetCaseMsg(objectName, caseName);
            handleInteractorSetCaseMsg(objectName, caseName);
        }
        break;
        case coGRMsg::SET_NAME:
        {
            auto &setNameMsg = msg.as<coGRObjSetNameMsg>();
            const char *coviseObjectName = setNameMsg.getObjName();
            const char *newName = setNameMsg.getNewName();
            if (cover->debugLevel(3))
                fprintf(stderr, "CuttingSurfacePlugin::guiToRenderMsg oGRMsg::SET_NAME object=%s name=%s\n", coviseObjectName, newName);
            handleSetNameMsg(coviseObjectName, newName);
        }
        break;
        case coGRMsg::MOVE_INTERACTOR:
        {
            if (cover->debugLevel(3))
                fprintf(stderr, "CuttingSurfacePlugin::guiToRenderMsg coGRMsg::MOVE_INTERACTOR\n");
            auto &moveInteractorMsg = msg.as<coGRObjMoveInterMsg>();
            const char *objectName = moveInteractorMsg.getObjName();

            const char *interactorName = moveInteractorMsg.getInteractorName();
            if (cover->debugLevel(3))
                fprintf(stderr, "\tobject=%s interactor=%s\n", objectName, interactorName);
            float x = moveInteractorMsg.getX();
            float y = moveInteractorMsg.getY();
            float z = moveInteractorMsg.getZ();
            handleMoveInteractorMsg(objectName, interactorName, x, y, z);
        }
        break;
        case coGRMsg::RESTRICT_AXIS:
        {
            auto &restrictAxisMsg = msg.as<coGRObjRestrictAxisMsg>();
            const char *objectName = restrictAxisMsg.getObjName();
            if (cover->debugLevel(3))
                fprintf(stderr, "CuttingSurfacePlugin::guiToRenderMsg coGRMsg::RESTRICT_AXIS object=%s\n", objectName);
            handleRestrictAxisMsg(objectName, restrictAxisMsg.getAxisName());
        }
        break;
        case coGRMsg::ATTACHED_CLIPPLANE:
        {
            auto &clipPlaneMsg = msg.as<coGRObjAttachedClipPlaneMsg>();
            const char *objectName = clipPlaneMsg.getObjName();
            if (cover->debugLevel(3))
                fprintf(stderr, "CuttingSurfacePlugin::guiToRenderMsg coGRMsg::ATTACHED_CLIPPLANE object=%s\n", objectName);
            handleAttachedClipPlaneMsg(objectName, clipPlaneMsg.getClipPlaneIndex(), clipPlaneMsg.getOffset(), clipPlaneMsg.isFlipped());
        }
        break;
        default:
        {
            if (cover->debugLevel(3))
                fprintf(stderr, "CuttingSurfacePlugin::guiToRenderMsg NOT-USED\n");
        }
        break;
        }
    }
}

void CuttingSurfacePlugin::preFrame()
{
    ModuleFeedbackPlugin::preFrame();
}

CuttingSurfacePlugin::CuttingSurfacePlugin()
    : ModuleFeedbackPlugin("CuttingSurface")
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nCuttingSurfacePlugin::CuttingSurfacePlugin\n");
}

CuttingSurfacePlugin::~CuttingSurfacePlugin()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nCuttingSurfacePlugin::~CuttingSurfacePlugin\n");
}

void
CuttingSurfacePlugin::updateInteractorVisibility(const char *objectName)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfacePlugin::updateInteractorVisibility(%s)\n", objectName);

    myInteractions_.reset();
    while (myInteractions_.current())
    {
        if (myInteractions_.current()->compare(objectName))
        {
            ((CuttingSurfaceInteraction *)myInteractions_.current())->updatePickInteractorVisibility();
            //((CuttingSurfaceInteraction*)myInteractions_.current())->updateDirectInteractorVisibility();
            break;
        }
        myInteractions_.next();
    }
}

ModuleFeedbackManager *
CuttingSurfacePlugin::NewModuleFeedbackManager(const RenderObject *container, coInteractor *interactor, const RenderObject *, const char *pluginName)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfacePlugin::NewModuleFeedbackManager\n");

    return new CuttingSurfaceInteraction(container, interactor, pluginName, this);
}

void
CuttingSurfacePlugin::handleInteractorVisibleMsg(const char *objectName, bool show)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfacePlugin::showInteractor(%s, %d)\n", objectName, show);
    myInteractions_.reset();
    while (myInteractions_.current())
    {
        if (myInteractions_.current()->compare(objectName))
        {
            ((CuttingSurfaceInteraction *)myInteractions_.current())->setShowInteractorFromGui(show);
            break;
        }
        myInteractions_.next();
    }
}

void
CuttingSurfacePlugin::handleInteractorSetCaseMsg(const char *objectName, const char *caseName)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfacePlugin::handleInteractorSetCaseMsg(%s, %s)\n", objectName, caseName);

    myInteractions_.reset();
    while (myInteractions_.current())
    {
        if (myInteractions_.current()->compare(objectName))
        {

            ((CuttingSurfaceInteraction *)myInteractions_.current())->interactorSetCaseFromGui(caseName);
        }
        myInteractions_.next();
    }
}

void
CuttingSurfacePlugin::handleMoveInteractorMsg(const char *objectName, const char *interactorName, float x, float y, float z)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfacePlugin::handleMoveInteractorMsg(objectName=%s interactorName=%s)\n", objectName, interactorName);

    //fprintf(stderr,"\tsearching the appropriate module feedback manager in list...\n");

    myInteractions_.reset();
    while (myInteractions_.current())
    {

        if (myInteractions_.current()->compare(objectName))
        {
            //fprintf(stderr,"found... now searching the right interactor for %s\n", interactorName);
            if (strcmp(interactorName, "point") == 0)
            {
                //fprintf(stderr,"interactorName=point\n");
                static_cast<CuttingSurfaceInteraction *>(myInteractions_.current())->setInteractorPointFromGui(x, y, z);
                break;
            }
            else if (strcmp(interactorName, "normal") == 0)
            {
                //fprintf(stderr,"interactorName=normal\n");
                static_cast<CuttingSurfaceInteraction *>(myInteractions_.current())->setInteractorNormalFromGui(x, y, z);
                break;
            }
        }
        myInteractions_.next();
    }
}

void
CuttingSurfacePlugin::handleRestrictAxisMsg(const char *objectName, const char *axisName)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfacePlugin::handleRestrictAxisMsg(objectName=%s axisName=%s)\n", objectName, axisName);

    myInteractions_.reset();
    while (myInteractions_.current())
    {

        if (myInteractions_.current()->compare(objectName))
        {
            //fprintf(stderr,"found... now searching the right interactor for %s\n", interactorName);
            if (strcmp(axisName, "xAxis") == 0)
            {
                static_cast<CuttingSurfaceInteraction *>(myInteractions_.current())->setRestrictXFromGui();
                break;
            }
            else if (strcmp(axisName, "yAxis") == 0)
            {
                static_cast<CuttingSurfaceInteraction *>(myInteractions_.current())->setRestrictYFromGui();
                break;
            }
            else if (strcmp(axisName, "zAxis") == 0)
            {
                static_cast<CuttingSurfaceInteraction *>(myInteractions_.current())->setRestrictZFromGui();
                break;
            }
            else if (strcmp(axisName, "freeAxis") == 0)
            {
                static_cast<CuttingSurfaceInteraction *>(myInteractions_.current())->setRestrictNoneFromGui();
                break;
            }
        }
        myInteractions_.next();
    }
}

void
CuttingSurfacePlugin::handleAttachedClipPlaneMsg(const char *objectName, int clipPlaneIndex, float offset, bool flip)
{
    myInteractions_.reset();
    while (myInteractions_.current())
    {
        if (myInteractions_.current()->compare(objectName))
        {

            ((CuttingSurfaceInteraction *)myInteractions_.current())->setClipPlaneFromGui(clipPlaneIndex, offset, flip);
        }
        myInteractions_.next();
    }
}

COVERPLUGIN(CuttingSurfacePlugin)
