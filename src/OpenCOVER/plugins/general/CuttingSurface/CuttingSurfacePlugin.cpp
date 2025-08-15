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

    if ((strncmp(moduleName, "CutGeometry", 11) == 0) || (strncmp(moduleName, "Clip", 4) == 0))
    {
        add(container, i);
    }

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
        ModuleFeedbackPlugin::guiToRenderMsg(msg);
        switch (msg.getType())
        {
        case coGRMsg::INTERACTOR_VISIBLE:
        {
            auto &interactorVisibleMsg = msg.as<coGRObjVisMsg>();
            const char *objectName = interactorVisibleMsg.getObjName();
            if (cover->debugLevel(3))
                fprintf(stderr, "CuttingSurfacePlugin::guiToRenderMsg coGRMsg::INTERACTOR_VISIBLE object=%s\n", objectName);
            handleInteractorVisibleMsg(objectName, interactorVisibleMsg.isVisible());
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
    for (auto *i: myInteractions_)
    {
        if (i->compare(objectName))
        {
            ((CuttingSurfaceInteraction *)i)->setShowInteractorFromGui(show);
            break;
        }
    }
}

void
CuttingSurfacePlugin::handleMoveInteractorMsg(const char *objectName, const char *interactorName, float x, float y, float z)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfacePlugin::handleMoveInteractorMsg(objectName=%s interactorName=%s)\n", objectName, interactorName);

    //fprintf(stderr,"\tsearching the appropriate module feedback manager in list...\n");

    for (auto *i: myInteractions_)
    {
        if (i->compare(objectName))
        {
            //fprintf(stderr,"found... now searching the right interactor for %s\n", interactorName);
            if (strcmp(interactorName, "point") == 0)
            {
                //fprintf(stderr,"interactorName=point\n");
                static_cast<CuttingSurfaceInteraction *>(i)->setInteractorPointFromGui(x, y, z);
                break;
            }
            else if (strcmp(interactorName, "normal") == 0)
            {
                //fprintf(stderr,"interactorName=normal\n");
                static_cast<CuttingSurfaceInteraction *>(i)->setInteractorNormalFromGui(x, y, z);
                break;
            }
        }
    }
}

void
CuttingSurfacePlugin::handleRestrictAxisMsg(const char *objectName, const char *axisName)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfacePlugin::handleRestrictAxisMsg(objectName=%s axisName=%s)\n", objectName, axisName);

    for (auto *i: myInteractions_)
    {
        if (i->compare(objectName))
        {
            //fprintf(stderr,"found... now searching the right interactor for %s\n", interactorName);
            if (strcmp(axisName, "xAxis") == 0)
            {
                static_cast<CuttingSurfaceInteraction *>(i)->setRestrictXFromGui();
                break;
            }
            else if (strcmp(axisName, "yAxis") == 0)
            {
                static_cast<CuttingSurfaceInteraction *>(i)->setRestrictYFromGui();
                break;
            }
            else if (strcmp(axisName, "zAxis") == 0)
            {
                static_cast<CuttingSurfaceInteraction *>(i)->setRestrictZFromGui();
                break;
            }
            else if (strcmp(axisName, "freeAxis") == 0)
            {
                static_cast<CuttingSurfaceInteraction *>(i)->setRestrictNoneFromGui();
                break;
            }
        }
    }
}

void
CuttingSurfacePlugin::handleAttachedClipPlaneMsg(const char *objectName, int clipPlaneIndex, float offset, bool flip)
{
    for (auto *i: myInteractions_)
    {
        if (i->compare(objectName))
        {
            ((CuttingSurfaceInteraction *)i)->setClipPlaneFromGui(clipPlaneIndex, offset, flip);
        }
    }
}

COVERPLUGIN(CuttingSurfacePlugin)
