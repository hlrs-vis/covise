/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUTTINGSURFACE_PLUGIN_H_
#define _CUTTINGSURFACE_PLUGIN_H_

#include <cover/coVRPlugin.h>
#include <PluginUtil/ModuleFeedbackPlugin.h>

using namespace opencover;

class CuttingSurfacePlugin : public ModuleFeedbackPlugin
{
public:
    CuttingSurfacePlugin();
    virtual ~CuttingSurfacePlugin();

    void guiToRenderMsg(const char *msg);
    virtual void preFrame();
    void newInteractor(const RenderObject *container, coInteractor *i);
    void removeObject(const char *objName, bool r);
    virtual void addNode(osg::Node *, const RenderObject * = NULL);

protected:
    virtual ModuleFeedbackManager *NewModuleFeedbackManager(const RenderObject *, coInteractor *, const RenderObject *, const char *);

    // called if msg from gui arrives
    void handleInteractorVisibleMsg(const char *objectName, bool show);
    void updateInteractorVisibility(const char *objectName);
    void handleInteractorSetCaseMsg(const char *objectName, const char *caseName);
    void handleMoveInteractorMsg(const char *objectName, const char *interactorName, float x, float y, float z);
    void handleRestrictAxisMsg(const char *objectName, const char *axisName);
    void handleAttachedClipPlaneMsg(const char *objectName, int clipPlaneIndex, float offset, bool flip);
};
#endif
