/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ISOSURFACE_INTERACTION_H_
#define _ISOSURFACE_INTERACTION_H_

#include <PluginUtil/ModuleInteraction.h>

class IsoSurfacePoint;
class IsoSurfacePlugin;

namespace vrui
{
class coCheckboxMenuItem;
class coSliderMenuItem;
}

namespace opencover
{
class IsoSurfaceInteraction : public ModuleInteraction
{
public:
    IsoSurfaceInteraction(RenderObject *container, coInteractor *inter, const char *pluginName, IsoSurfacePlugin *p);
    virtual ~IsoSurfaceInteraction();
    virtual void update(RenderObject *container, coInteractor *inter);
    virtual void preFrame();

    // react to pickInteractor and directInteractor checkbox
    virtual void updatePickInteractors(bool);
    virtual void updateDirectInteractors(bool);

    static const char *ISOPOINT;
    static const char *ISOVALUE;
    IsoSurfacePlugin *plugin;

private:
    bool wait_;
    bool newObject_;
    IsoSurfacePoint *isoPoint_;
    float minValue_, maxValue_, isoValue_;
    void createMenu();
    void updateMenu();
    void deleteMenu();
    void updateInteractorVisibility();
    vrui::coSliderMenuItem *valueSlider_;
    virtual void menuEvent(vrui::coMenuItem *menuItem);
    virtual void menuReleaseEvent(vrui::coMenuItem *menuItem);
};
}
#endif
