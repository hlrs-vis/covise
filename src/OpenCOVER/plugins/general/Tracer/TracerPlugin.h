/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TRACER_PLUGIN_H
#define _TRACER_PLUGIN_H

namespace opencover
{
class ModuleFeedbackManager;
}
#include <PluginUtil/ModuleFeedbackPlugin.h>
#include <cover/coVRPluginSupport.h>

using namespace covise;
using namespace opencover;

class TracerPlugin : public ModuleFeedbackPlugin
{
public:
    TracerPlugin();
    virtual ~TracerPlugin();

    virtual void preFrame();
    void newInteractor(const RenderObject *container, coInteractor *i);
    void removeObject(const char *objName, bool r);
    void addObject(const RenderObject *container, osg::Group *, const RenderObject *, const RenderObject *, const RenderObject *, const RenderObject *);
    virtual void addNode(osg::Node *, const RenderObject * = NULL);
    void guiToRenderMsg(const char *msg);
    // prepare smoke data for tracer line
    void addSmoke(const char *name, const RenderObject *, const RenderObject *);
    void removeSmoke(const char *name);
    static int debugLevel_;

protected:
    virtual ModuleFeedbackManager *NewModuleFeedbackManager(const RenderObject *, coInteractor *, const RenderObject *, const char *);
    // msg from gui
    void updateInteractorVisibility(const char *objectName);
    void handleSmokeVisibleMsg(const char *objectName, bool show);
    void handleInteractorVisibleMsg(const char *objectName, bool show);
    void handleMoveInteractorMsg(const char *objectName, const char *interactorName, float x, float y, float z);
    void handleUseInteractorMsg(const char *objectName, bool use);
    void handleInteractorSetCaseMsg(const char *objectName, const char *caseName);
};
#endif
