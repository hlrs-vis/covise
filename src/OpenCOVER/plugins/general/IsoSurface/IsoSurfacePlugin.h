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

    void guiToRenderMsg(const grmsg::coGRMsg &msg) override;
    void preFrame() override;
    void newInteractor(const opencover::RenderObject *container, opencover::coInteractor *i) override;
    void removeObject(const char *objName, bool replace) override;
    virtual void addNode(osg::Node *, const opencover::RenderObject * = NULL) override;

protected:
    // this returns in fact an IsoSurfaceInteraction pointer
    virtual opencover::ModuleFeedbackManager *NewModuleFeedbackManager(const opencover::RenderObject *,
                                                                       opencover::coInteractor *,
                                                                       const opencover::RenderObject *,
                                                                       const char *) override;

private:
};
#endif
