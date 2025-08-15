/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "TracerPlugin.h"

#include "TracerInteraction.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coInteractor.h>
#include <cover/coVRMSController.h>
#include <config/CoviseConfig.h>

#include <map>

#include <cover/RenderObject.h>

#include <grmsg/coGRObjVisMsg.h>
#include <grmsg/coGRObjMoveInterMsg.h>
#include <grmsg/coGRObjSetCaseMsg.h>
#include <grmsg/coGRObjSetNameMsg.h>

#include <PluginUtil/PluginMessageTypes.h>

using namespace grmsg;
using namespace std;

int TracerPlugin::debugLevel_ = 0;

void TracerPlugin::newInteractor(const RenderObject *container, coInteractor *i)
{
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "\n--- TracerPlugin::newInteractor from module %s\n", i->getModuleName());
        if (container)
            fprintf(stderr, "container %p %s\n", container, container->getName());
    }
    const char *moduleName = i->getModuleName();
    if ((strncmp(moduleName, "TracerComp", 10) == 0) || (strncmp(moduleName, "Tracer", 6) == 0))
    {
        add(container, i);
    }

    ModuleFeedbackPlugin::newInteractor(container, i);
}

void TracerPlugin::preFrame()
{
    ModuleFeedbackPlugin::preFrame();
}

void TracerPlugin::removeObject(const char *objName, bool r)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "\n--- TracerPlugin::removeObject objectName=[%s] replace=%d\n", objName, r);

    // replace is handeled in addObject
    if (!r)
    {
        remove(objName);
        removeSmoke(objName);
    }
}

void
TracerPlugin::addObject(const RenderObject *container, osg::Group * /*root*/, const RenderObject *grid, const RenderObject *velo, const RenderObject *, const RenderObject *)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n---- TracerPlugin::addObject\n");

    // check if this is from TracerComp and the sampled grid
    if (container)
    {
        if (container->getAttribute("CREATOR_MODULE_NAME")
            && (strstr(container->getAttribute("CREATOR_MODULE_NAME"), "TracerComp") != NULL))
        {
            //RenderObject *grid = container->getGeometry();
            if (!grid)
                fprintf(stderr, "!grid\n");

            if (!grid || !grid->isUniformGrid())
            {
                fprintf(stderr, "not a UNIGRD\n");
                return;
            }
            //RenderObject *velo = container->getNormals();
            if (!velo)
                fprintf(stderr, "!velo\n");
            if (!velo || !velo->isVectors())
            {
                fprintf(stderr, "not a USTVDT\n");
                return;
            }
            addSmoke(container->getName(), grid, velo);
        }
        else
        {
            if (cover->debugLevel(3))
                fprintf(stderr, "from TracerComp but no sampled grid\n");
        }
    }
    else
    {
        if (cover->debugLevel(0))
            fprintf(stderr, "...no container\n");
    }
}
void TracerPlugin::addNode(osg::Node *node, const RenderObject *obj)
{
    //fprintf(stderr,"TracerPlugin::addNode %s\n", obj->getName());
    if (obj)
    {
        addNodeToCase(obj->getName(), node);
    }
}

void
TracerPlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- TracerPlugin:: guiToRenderMsg\n");
        
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
                fprintf(stderr, "\tcoGRMsg::INTERACTOR_VISIBLE object=%s\n", objectName);
            handleInteractorVisibleMsg(objectName, interactorVisibleMsg.isVisible() != 0);
        }
        break;
                case coGRMsg::SMOKE_VISIBLE:
        {
            auto &smokeVisibleMsg = msg.as<coGRObjVisMsg>();
            const char *objectName = smokeVisibleMsg.getObjName();
            if (cover->debugLevel(3))
                fprintf(stderr, "\tcoGRMsg::SMOKE_VISIBLE object=%s\n", objectName);

            handleSmokeVisibleMsg(objectName, smokeVisibleMsg.isVisible() != 0);
        }
        break;
        case coGRMsg::MOVE_INTERACTOR:
        {
            if (cover->debugLevel(3))
                fprintf(stderr, "\tcoGRMsg::MOVE_INTERACTOR\n");
            auto &moveInteractorMsg = msg.as<coGRObjMoveInterMsg>();
            const char *objectName = moveInteractorMsg.getObjName();

            const char *interactorName = moveInteractorMsg.getInteractorName();
            //fprintf(stderr,"interactorName=[%s]\n",interactorName );

            if (cover->debugLevel(3))
                fprintf(stderr, "\tobject=%s interactor=%s\n", objectName, interactorName);

            float x = moveInteractorMsg.getX();
            float y = moveInteractorMsg.getY();
            float z = moveInteractorMsg.getZ();
            handleMoveInteractorMsg(objectName, interactorName, x, y, z);
        }
        break;
        case coGRMsg::INTERACTOR_USED:
        {
            auto &interactorUsedMsg = msg.as<coGRObjVisMsg>();
            const char *objectName = interactorUsedMsg.getObjName();
            if (cover->debugLevel(3))
                fprintf(stderr, "\tcoGRMsg::INTERACTOR_USED object=%s\n", objectName);

            handleUseInteractorMsg(objectName, interactorUsedMsg.isVisible() != 0);
        }
        break;
        default:
        {
            if (cover->debugLevel(3))
                fprintf(stderr, "\t msg NOT-USED\n");
        }
        break;
        }
    }
}

//-----------------------------------------------------------------------------

TracerPlugin::TracerPlugin()
    : ModuleFeedbackPlugin("Tracer")
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerPlugin::TracerPlugin\n");

    if (coVRMSController::instance()->isMaster())
    {
        TracerPlugin::debugLevel_ = coCoviseConfig::getInt("COVER.Plugin.Tracer.DebugLevel", 0);
    }
}

TracerPlugin::~TracerPlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerPlugin::~TracerPlugin\n");
}

ModuleFeedbackManager *
TracerPlugin::NewModuleFeedbackManager(const RenderObject *container, coInteractor *interactor, const RenderObject *, const char *pluginName)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerPlugin::NewModuleFeedbackManager for %s\n", container->getName());

    return new TracerInteraction(container, interactor, pluginName, this);
}

void
TracerPlugin::addSmoke(const char *containerName, const RenderObject *grid, const RenderObject *velo)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerPlugin::addSmoke %s\n", containerName);

    // find the TracerInteraction (ModuleFeedbackManager), where this object belongs to
    for (auto *i: myInteractions_)
    {
        if (i->compare(containerName))
        {
            ((TracerInteraction *)i)->addSmoke(grid, velo);
            break;
        }
    }
}

void
TracerPlugin::removeSmoke(const char *objName)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerPlugin::removeSmoke %s\n", objName);

    // find the TracerInteraction (ModuleFeedbackManager), where this object belongs to
    for (auto *i: myInteractions_)
    {
        if (i->compare(objName))
        {
            ((TracerInteraction *)i)->addSmoke(NULL, NULL);
            break;
        }
    }
}

void
TracerPlugin::handleInteractorVisibleMsg(const char *objectName, bool show)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "TracerPlugin::handleInteractorVisibleMsg(%s, %d)\n", objectName, show);
    for (auto *i: myInteractions_)
    {
        if (i->compare(objectName))
        {
            ((TracerInteraction *)i)->setShowInteractorFromGui(show);
            break;
        }
    }
}

void
TracerPlugin::handleSmokeVisibleMsg(const char *objectName, bool show)
{
    //fprintf(stderr,"TracerPlugin::showSmoke(%s, %d)\n", objectName, show);
    for (auto *i: myInteractions_)
    {
        if (i->compare(objectName))
        {
            ((TracerInteraction *)i)->setShowSmokeFromGui(show);
            break;
        }
    }
}

void
TracerPlugin::handleMoveInteractorMsg(const char *objectName, const char *interactorName, float x, float y, float z)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "TracerPlugin::moveInteractor(objectName=%s interactorName=%s)\n", objectName, interactorName);

    //fprintf(stderr,"\tsearching the appropriate module feedback manager in list...\n");

    for (auto *i: myInteractions_)
    {
        if (i->compare(objectName))
        {
            //fprintf(stderr,"found... now searching the right interactor for %s\n", interactorName);
            if (strcmp(interactorName, "s1") == 0)
            {
                //fprintf(stderr,"interactorName=s1\n");
                static_cast<TracerInteraction *>(i)->setStartpoint1FromGui(x, y, z);
                break;
            }
            else if (strcmp(interactorName, "s2") == 0)
            {
                //fprintf(stderr,"interactorName=s2\n");
                static_cast<TracerInteraction *>(i)->setStartpoint2FromGui(x, y, z);
                break;
            }
            else if (strcmp(interactorName, "direction") == 0)
            {
                //fprintf(stderr,"interactorName=direction\n");
                static_cast<TracerInteraction *>(i)->setDirectionFromGui(x, y, z);
                break;
            }
            //else
            //   fprintf(stderr,"interactorName [%s] is unknown\n");
        }
    }
}

void
TracerPlugin::handleUseInteractorMsg(const char * /*objectName*/, bool /*use*/)
{
    //fprintf(stderr,"TracerPlugin::useInteractor(%s, %d)\n", objectName, use);
    //for (auto *i: myInteractions_)
    //{
    //   if (i->compare(objectName))
    //   {
    //      ((TracerInteraction*)i)->setUseInteractorFromGui(use);
    //      break;
    //   }
    //}
}

COVERPLUGIN(TracerPlugin)

// local variables:
// c-basic-offset: 3
// end:
