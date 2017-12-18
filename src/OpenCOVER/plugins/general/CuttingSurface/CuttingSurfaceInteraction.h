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

#ifdef VRUI
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxGroup;
class coSliderMenuItem;
}
#else
namespace opencover
{
namespace ui
{
class Menu;
class Button;
class Slider;
class ButtonGroup;
class SelectionList;
}
}
#endif


namespace opencover
{

class CuttingSurfaceInteraction : public ModuleInteraction
{
public:
    CuttingSurfaceInteraction(const RenderObject *container, coInteractor *inter, const char *pluginName, CuttingSurfacePlugin *p);
    virtual ~CuttingSurfaceInteraction();
    virtual void update(const RenderObject *container, coInteractor *inter);
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
        RESTRICT_X = 1,
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
    bool newObject_;
    bool planeOptionsInMenu_;

    // create/update/delete the contents of the tracer submenu
    void createMenu();
    void updateMenu();
    void deleteMenu();
#ifdef VRUI
    void menuEvent(vrui::coMenuItem *menuItem);
#endif

    void getParameters();

    int restrictToAxis_ = RESTRICT_NONE;
    void restrictX();
    void restrictY();
    void restrictZ();
    void restrictNone();
    ui::SelectionList *orient_=nullptr;

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
    int option_ = OPTION_NONE;
#if 0
    ui::Menu *optionMenu_=nullptr;
    ui::Button *optionPlane_=nullptr, *optionCylX_=nullptr, *optionCylY_=nullptr, *optionCylZ_=nullptr, *optionSphere_=nullptr;
    ui::ButtonGroup *optionGroup_=nullptr;
#endif
    ui::SelectionList *optionChoice_=nullptr;

    CuttingSurfacePlane *csPlane_;
    CuttingSurfaceCylinder *csCylX_, *csCylY_, *csCylZ_;
    CuttingSurfaceSphere *csSphere_;

    int activeClipPlane_;
    ui::Menu *clipPlaneMenu_=nullptr;
    ui::ButtonGroup *clipPlaneIndexGroup_=nullptr;
    ui::Button *clipPlaneNoneCheckbox_=nullptr;
    ui::Button *clipPlaneIndexCheckbox_[6]={nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    ui::Slider *clipPlaneOffsetSlider_=nullptr;
    ui::Button *clipPlaneFlipCheckbox_=nullptr;
    CuttingSurfacePlugin *plugin=nullptr;

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
