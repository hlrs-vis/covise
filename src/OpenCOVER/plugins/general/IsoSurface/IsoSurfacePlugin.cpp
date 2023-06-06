/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coInteractor.h>
#include "IsoSurfacePlugin.h"
#include "IsoSurfaceInteraction.h"
#include <cover/RenderObject.h>

#include <grmsg/coGRObjVisMsg.h>
#include <grmsg/coGRObjMoveInterMsg.h>
#include <grmsg/coGRObjSetCaseMsg.h>
#include <grmsg/coGRObjSetNameMsg.h>

using namespace covise;
using namespace grmsg;
using namespace opencover;

IsoSurfacePlugin::IsoSurfacePlugin()
    : ModuleFeedbackPlugin("IsoSurface")
{
}

IsoSurfacePlugin::~IsoSurfacePlugin()
{
}

void
IsoSurfacePlugin::preFrame()
{
    ModuleFeedbackPlugin::preFrame();
}

void
IsoSurfacePlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "\n--- IsoSurfacePlugin:: guiToRenderMsg\n");
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
                fprintf(stderr, "IsoSurfacePlugin::guiToRenderMsg coGRMsg::GEO_VISIBLE object=%s visible=%d\n", objectName, geometryVisibleMsg.isVisible());

            handleGeoVisibleMsg(objectName, geometryVisibleMsg.isVisible());
        }
        break;
        case coGRMsg::SET_CASE:
        {
            auto &setCaseMsg = msg.as<coGRObjSetCaseMsg>();
            const char *objectName = setCaseMsg.getObjName();
            if (cover->debugLevel(3))
                fprintf(stderr, "IsoSurfacePlugin::guiToRenderMsg coGRMsg::SET_CASE object=%s\n", objectName);
            const char *caseName = setCaseMsg.getCaseName();
            handleSetCaseMsg(objectName, caseName);
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
        
        default:
        {
            if (cover->debugLevel(3))
                fprintf(stderr, "CuttingSurfacePlugin::guiToRenderMsg NOT-USED\n");
        }
        break;
        }
    }
}
void IsoSurfacePlugin::addNode(osg::Node *node, const RenderObject *obj)
{
    //fprintf(stderr,"IsoSurfacePlugin::addNode %s\n", obj->getName());
    if (obj)
    {
        addNodeToCase(obj->getName(), node);
    }
}
ModuleFeedbackManager *
IsoSurfacePlugin::NewModuleFeedbackManager(const RenderObject *container, coInteractor *interactor, const RenderObject *, const char *pluginName)
{
    return new IsoSurfaceInteraction(container, interactor, pluginName, this);
}

// called whenever cover receives a covise object
// with feedback info appended
void
IsoSurfacePlugin::newInteractor(const RenderObject *container, coInteractor *i)
{

    const char *moduleName = i->getModuleName();
    if ((strncmp(moduleName, "IsoSurfaceComp", 14) == 0) || (strncmp(moduleName, "IsoSurface", 10) == 0))
    {
        add(container, i);
    }

    ModuleFeedbackPlugin::newInteractor(container, i);
}

void IsoSurfacePlugin::removeObject(const char *objName, bool r)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\n--- coVRRemoveObject objectName=[%s]\n", objName);

    // replace is handeled in addObject
    if (!r)
    {
        remove(objName);
    }
}

COVERPLUGIN(IsoSurfacePlugin)
