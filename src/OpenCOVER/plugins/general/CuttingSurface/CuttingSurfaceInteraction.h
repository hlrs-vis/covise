/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUTSURFACE_INTERACTION_H_
#define _CUTSURFACE_INTERACTION_H_

#include <PluginUtil/ModuleInteraction.h>

class CuttingSurfacePlugin;
class CuttingSurfacePlane;
class CuttingSurfaceCylinder;
class CuttingSurfaceSphere;

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxGroup;
class coSliderMenuItem;
}

namespace opencover
{

class CuttingSurfaceInteraction : public ModuleInteraction
{
public:
    CuttingSurfaceInteraction(RenderObject *container, coInteractor *inter, const char *pluginName, CuttingSurfacePlugin *p);
    virtual ~CuttingSurfaceInteraction();
    virtual void update(RenderObject *container, coInteractor *inter);
    virtual void preFrame();
    virtual void updatePickInteractors(bool);
    virtual void updateDirectInteractors(bool);

    static const char *VERTEX;
    static const char *POINT;
    static const char *OPTION;
    static const char *SCALAR;

    enum restrictAxis
    {
        RESTRICT_NONE = 0,
        RESTRICT_X,
        RESTRICT_Y = 2,
        RESTRICT_Z = 3
    };

    // if msg arrives from gui
    //      void setShowInteractorFromGui(bool state);
    void updatePickInteractorVisibility();
    void interactorSetCaseFromGui(const char *interactorCasename);
    void setInteractorPointFromGui(float x, float y, float z);
    void setInteractorNormalFromGui(float x, float y, float z);
    void setRestrictXFromGui();
    void setRestrictYFromGui();
    void setRestrictZFromGui();
    void setRestrictNoneFromGui();
    void setClipPlaneFromGui(int index, float offset, bool flip);

private:
    bool wait_;
    bool newObject_;
    bool planeOptionsInMenu_;

    // create/update/delete the contents of the tracer submenu
    void createMenu();
    void updateMenu();
    void deleteMenu();
    void menuEvent(vrui::coMenuItem *menuItem);

    void getParameters();

    int restrictToAxis_;
    void restrictX();
    void restrictY();
    void restrictZ();
    void restrictNone();
    vrui::coCheckboxMenuItem *orientX_, *orientY_, *orientZ_, *orientFree_;
    vrui::coCheckboxGroup *orientGroup_;

    enum
    {
        OPTION_NONE = -1,
        OPTION_PLANE = 0,
        OPTION_SPHERE,
        OPTION_CYLX,
        OPTION_CYLY,
        OPTION_CYLZ,
        NumSurfaceStyles
    };
    int option_, oldOption_;
    vrui::coSubMenuItem *optionButton_;
    vrui::coRowMenu *optionMenu_;
    vrui::coCheckboxMenuItem *optionPlane_, *optionCylX_, *optionCylY_, *optionCylZ_, *optionSphere_;
    vrui::coCheckboxGroup *optionGroup_;

    CuttingSurfacePlane *csPlane_;
    CuttingSurfaceCylinder *csCylX_, *csCylY_, *csCylZ_;
    CuttingSurfaceSphere *csSphere_;

    int activeClipPlane_;
    vrui::coRowMenu *clipPlaneMenu_;
    vrui::coSubMenuItem *clipPlaneMenuItem_;
    vrui::coCheckboxGroup *clipPlaneIndexGroup_;
    vrui::coCheckboxMenuItem *clipPlaneNoneCheckbox_;
    vrui::coCheckboxMenuItem *clipPlaneIndexCheckbox_[6];
    vrui::coSliderMenuItem *clipPlaneOffsetSlider_;
    vrui::coCheckboxMenuItem *clipPlaneFlipCheckbox_;
    CuttingSurfacePlugin *plugin;

    void sendClipPlaneToGui();
    void switchClipPlane(int index);
    void sendClipPlaneVisibilityMsg(int index, bool enabled);
    void sendClipPlanePositionMsg(int index);
    void sendShowPickInteractorMsg();
    void sendHidePickInteractorMsg();
    void sendRestrictXMsg();
    void sendRestrictYMsg();
    void sendRestrictZMsg();
    void sendRestrictNoneMsg();
};
}
#endif
