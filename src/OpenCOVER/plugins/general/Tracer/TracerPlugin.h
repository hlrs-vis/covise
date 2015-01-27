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
    void newInteractor(RenderObject *container, coInteractor *i);
    void removeObject(const char *objName, bool r);
    void addObject(RenderObject *container, RenderObject * /*geomobj*/,
                   RenderObject * /*normObj*/, RenderObject * /*colorObj*/, RenderObject * /*texObj*/,
                   osg::Group * /*root*/, int /*numCol*/, int /*colorBinding*/, int /*colorPacking*/,
                   float * /*r*/, float * /*g*/, float * /*b*/, int * /*packedCol*/,
                   int /*numNormals*/, int /*normalBinding*/, float * /*xn*/, float * /*yn*/, float * /*zn*/,
                   float /*transparency*/);
    virtual void addNode(osg::Node *, RenderObject * = NULL);
    void guiToRenderMsg(const char *msg);
    // prepare smoke data for tracer line
    void addSmoke(const char *name, RenderObject *, RenderObject *);
    void removeSmoke(const char *name);
    static int debugLevel_;

protected:
    virtual ModuleFeedbackManager *NewModuleFeedbackManager(RenderObject *, coInteractor *, RenderObject *, const char *);
    // msg from gui
    void updateInteractorVisibility(const char *objectName);
    void handleSmokeVisibleMsg(const char *objectName, bool show);
    void handleInteractorVisibleMsg(const char *objectName, bool show);
    void handleMoveInteractorMsg(const char *objectName, const char *interactorName, float x, float y, float z);
    void handleUseInteractorMsg(const char *objectName, bool use);
    void handleInteractorSetCaseMsg(const char *objectName, const char *caseName);
};
#endif
