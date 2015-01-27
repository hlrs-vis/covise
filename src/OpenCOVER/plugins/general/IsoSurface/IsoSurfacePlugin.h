/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ISOSURFACE_PLUGIN_H_
#define _ISOSURFACE_PLUGIN_H_

#include <cover/coVRPlugin.h>
#include <PluginUtil/ModuleFeedbackPlugin.h>
#include <cover/RenderObject.h>

class IsoSurfacePlugin : public opencover::ModuleFeedbackPlugin
{
public:
    IsoSurfacePlugin();
    virtual ~IsoSurfacePlugin();

    void guiToRenderMsg(const char *msg);
    virtual void preFrame();
    void newInteractor(opencover::RenderObject *container, opencover::coInteractor *i);
    void removeObject(const char *objName, bool replace);
    virtual void addNode(osg::Node *, opencover::RenderObject * = NULL);

protected:
    // this returns in fact an IsoSurfaceInteraction pointer
    virtual opencover::ModuleFeedbackManager *NewModuleFeedbackManager(opencover::RenderObject *, opencover::coInteractor *, opencover::RenderObject *, const char *);

private:
};
#endif
