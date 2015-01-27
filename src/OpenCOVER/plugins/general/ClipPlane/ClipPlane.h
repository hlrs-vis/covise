/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*  GL Clipping Plane module

//  Author : Daniela Rainer
//  Date   : 04-Feb-99

*/

#include <cover/coTabletUI.h>
#include <cover/coVRPluginSupport.h>

#include <cover/coVRPlugin.h>
#include <osg/ClipPlane>

namespace vrui
{
class coButtonMenuItem;
class coSubMenuItem;
class coRowMenu;
class coCheckboxMenuItem;
class coTrackerButtonInteraction;
}

using namespace vrui;
using namespace opencover;

#include <PluginUtil/coVR3DTransRotInteractor.h>

class ClipPlanePlugin : public coVRPlugin, public coTUIListener, public coMenuListener
{
private:
    class Plane
    {
    public:
        bool enabled;
        bool valid; // valid is false before first use
        osg::ref_ptr<osg::ClipPlane> clip;
        coTUIToggleButton *tuiEnableButton;
        coTUIToggleButton *tuiDirectInteractorButton;
        coTUIToggleButton *tuiPickInteractorButton;
        coCheckboxMenuItem *vruiEnableCheckbox;
        coCheckboxMenuItem *vruiDirectInteractorCheckbox;
        coCheckboxMenuItem *vruiPickInteractorCheckbox;
        coTrackerButtonInteraction *directInteractor;
        coVR3DTransRotInteractor *pickInteractor;
        bool showPickInteractor_, showDirectInteractor_;
        Plane()
        {
            valid = false;
            enabled = false;
            showPickInteractor_ = false;
            showDirectInteractor_ = false;
            clip = NULL;
            directInteractor = NULL;
            pickInteractor = NULL;
            tuiEnableButton = NULL;
            tuiDirectInteractorButton = NULL;
            tuiPickInteractorButton = NULL;
            vruiEnableCheckbox = NULL;
            vruiDirectInteractorCheckbox = NULL;
            vruiPickInteractorCheckbox = NULL;
        }
        ~Plane()
        {
            if (directInteractor)
                delete directInteractor;
            delete pickInteractor;
            delete tuiEnableButton;
            delete tuiDirectInteractorButton;
            delete tuiPickInteractorButton;
            delete vruiEnableCheckbox;
            delete vruiDirectInteractorCheckbox;
            delete vruiPickInteractorCheckbox;
        }
    };
    Plane plane[coVRPluginSupport::MAX_NUM_CLIP_PLANES];

    coTUITab *clipTab;
    coSubMenuItem *clipMenuItem;
    coRowMenu *clipMenu;

    osg::ref_ptr<osg::Geode> visibleClipPlaneGeode; // transparent plane
    bool active; // FLAG: ON = POSSIBLE TO CHANGE/SET CHOSEN CLIPPING PLANE
    osg::Matrix pointerMatrix; // needed for computing eqn
    osg::ref_ptr<osg::MatrixTransform> pointerTransform;
    osg::ref_ptr<osg::MatrixTransform> interactorTransform; // needed as parent node for the visible plane

    osg::Geode *loadPlane();

    osg::Vec4d matrixToEquation(const osg::Matrix &mat);

    void setInitialEquation(int);

public:
    ClipPlanePlugin();
    virtual ~ClipPlanePlugin();
    bool init();
    coVRPlugin *getModule()
    {
        return this;
    };
    virtual void menuEvent(coMenuItem *item);
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);
    void message(int type, int len, const void *buf);
    void preFrame();
    static ClipPlanePlugin *thisInstance;

    coMenuItem *getMenuButton(const std::string &buttonName);
};
