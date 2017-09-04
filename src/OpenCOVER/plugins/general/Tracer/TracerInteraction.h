/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TRACER_INTERACTION_H
#define _TRACER_INTERACTION_H

#include <PluginUtil/ModuleInteraction.h>

class TracerLine;
class TracerPlane;
class TracerFreePoints;

#ifdef USE_COVISE
#include <CovisePluginUtil/SmokeGeneratorSolutions.h>
#endif
#include <osg/Group>
#ifdef VRUI
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>

namespace vrui
{
class coPotiMenuItem;
class coSubMenuItem;
class coCheckboxGroup;
class coButtonMenuItem;
}
#else
#include <cover/ui/Owner.h>
namespace opencover {
namespace ui {
class Menu;
class Button;
class Slider;
class ButtonGroup;
class SelectionList;
}
}
#endif

class TracerPlugin;

class TracerInteraction : public opencover::ModuleInteraction
{
public:
    // constructor
    TracerInteraction(const opencover::RenderObject *container, opencover::coInteractor *inter, const char *pluginName, TracerPlugin *p);

    // destructor
    virtual ~TracerInteraction();

    // update covise stuff and menus
    virtual void update(const opencover::RenderObject *container, opencover::coInteractor *inter);

    // direct interaction
    virtual void preFrame();

    // react to pickInteractor and directInteractor checkbox
    virtual void updatePickInteractors(bool);
    virtual void updateDirectInteractors(bool);

    // add smoke if available
    void addSmoke(const opencover::RenderObject *smokeGrid, const opencover::RenderObject *smokeVelo);

    // module was copied --> new mode
    //void setNew();

    // hide also interactors if geometry is hidden
    void updatePickInteractorVisibility();

    // mgs from gui
    void setStartpoint1FromGui(float x, float y, float z);
    void setStartpoint2FromGui(float x, float y, float z);
    void setDirectionFromGui(float x, float y, float z);
    void setShowSmokeFromGui(bool state);
    void interactorSetCaseFromGui(const char *interactorCasename);
    //    void setUseInteractorFromGui(bool use);
    //    void interactorSetCaseFromGui(const char *interactorCasename);

    bool guiSliderFirsttime_;

    static const char *P_NO_STARTPOINTS; // no_startp!SLIDER
    static const char *P_STARTPOINT1; // startpoint1!VECTOR
    static const char *P_STARTPOINT2; // startpoint2!VECTOR
    static const char *P_DIRECTION; // direction!VECTOR
    static const char *P_TDIRECTION; // tdirection!CHOICE
    static const char *P_TASKTYPE; // tasktype!CHOICE
    static const char *P_STARTSTYLE; // startstyle!CHOICE
    static const char *P_TRACE_LEN; // trace_len!SCALAR
    static const char *P_FREE_STARTPOINTS; // freeStartPonts!STRING

    // uniform grid and velocity for smoke available
    bool smoke_;
    // uniform grid info
    float xmin_, xmax_;
    float ymin_, ymax_;
    float zmin_, zmax_;
    int nx_, ny_, nz_;
    // velocity  info
    vector<float> u_;
    vector<float> v_;
    vector<float> w_;

private:
    bool newObject_; // indicates, that in preFrame visibiliy is checked again
    bool debugSmoke_;
    bool interactorUsed_; //if TracerComp uses 2D part interactorUsed=false

#ifdef VRUI
    virtual void menuEvent(vrui::coMenuItem *menuItem);
    virtual void menuReleaseEvent(vrui::coMenuItem *menuItem);

    vrui::coPotiMenuItem *_numStartPointsPoti;
    int _numStartPointsMin, _numStartPointsMax, _numStartPoints;

    vrui::coPotiMenuItem *traceLenPoti_;
    float traceLenMin_, traceLenMax_, traceLen_;

    vrui::coSubMenuItem *_taskTypeButton;
    vrui::coRowMenu *_taskTypeMenu;
    vrui::coCheckboxMenuItem *_streamlinesCheckbox, *_particlesCheckbox, *_pathlinesCheckbox, *_streaklinesCheckbox;
    vrui::coCheckboxGroup *_taskTypeGroup;
    int _numTaskTypes;
    char **_taskTypeNames;
    enum
    {
        TASKTYPE_STREAMLINES = 0,
        TASKTYPE_PARTICLES,
        TASKTYPE_PATHLINES,
        TASKTYPE_STREAKLINES
    };
    int _selectedTaskType;

    vrui::coSubMenuItem *_startStyleButton;
    vrui::coRowMenu *_startStyleMenu;
    vrui::coCheckboxMenuItem *_planeCheckbox, *_lineCheckbox, *_freeCheckbox, *_cylinderCheckbox;
    vrui::coCheckboxGroup *_startStyleGroup;
#else
    opencover::ui::Slider *_numStartPointsPoti = nullptr;
    int _numStartPointsMin, _numStartPointsMax, _numStartPoints;

    opencover::ui::Slider *traceLenPoti_;
    float traceLenMin_, traceLenMax_, traceLen_;

    opencover::ui::SelectionList *_taskType = nullptr;
#if 0
    opencover::ui::Menu *_taskTypeMenu;
    opencover::ui::Button *_streamlinesCheckbox, *_particlesCheckbox, *_pathlinesCheckbox, *_streaklinesCheckbox;
    opencover::ui::ButtonGroup *_taskTypeGroup;
#endif
    int _numTaskTypes;
    char **_taskTypeNames;
    enum
    {
        TASKTYPE_STREAMLINES = 0,
        TASKTYPE_PARTICLES,
        TASKTYPE_PATHLINES,
        TASKTYPE_STREAKLINES
    };
    int _selectedTaskType;

    opencover::ui::SelectionList *_startStyle = nullptr;
#if 0
    opencover::ui::Menu *_startStyleMenu;
    opencover::ui::Button *_planeCheckbox, *_lineCheckbox, *_freeCheckbox, *_cylinderCheckbox;
    opencover::ui::ButtonGroup *_startStyleGroup;
#endif
#endif
    int _numStartStyles;
    char **_startStyleNames;
    enum
    {
        STARTSTYLE_LINE = 0,
        STARTSTYLE_PLANE = 1,
        STARTSTYLE_FREE = 2,
        STARTSTYLE_CYLINDER = 2
    };
    int _selectedStartStyle, _oldStartStyle;

    int _numTimeDirections;
    char **_timeDirectionNames;
    int _selectedTimeDirection;

    TracerPlane *_tPlane;
    TracerLine *_tLine;
    TracerFreePoints *_tFree;
#ifdef USE_COVISE
    SmokeGeneratorSolutions solutions_;
#endif

#ifdef VRUI
    vrui::coCheckboxMenuItem *smokeCheckbox_;
#else
    opencover::ui::Button *smokeCheckbox_;
#endif
    bool smokeInMenu_;

    // create/update/delete the contents of the tracer submenu
    void createMenuContents();
    void updateMenuContents();
    void deleteMenuContents();

    char *_containerName;
    bool isComplex;

    void displaySmoke();
    osg::ref_ptr<osg::Group> smokeRoot; ///< Geometry node
    osg::ref_ptr<osg::Geode> smokeGeode_;
    osg::ref_ptr<osg::Geometry> smokeGeometry_;
    osg::ref_ptr<osg::Vec4Array> smokeColor_;
    bool showSmoke_;
    void showSmoke();
    void hideSmoke();
    void updateSmokeLine();
    void updateSmokePlane();

    void getParameters();

    osg::MatrixTransform *interDCS_;
    TracerPlugin *plugin;
};
#endif
