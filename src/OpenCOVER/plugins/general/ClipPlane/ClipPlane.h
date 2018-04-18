/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*  GL Clipping Plane module

//  Author : Daniela Rainer
//  Date   : 04-Feb-99

*/

#include <cover/ui/Owner.h>
#include <osg/ClipPlane>

#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>

namespace vrui
{
class coTrackerButtonInteraction;
class coRelativeInputInteraction;
}

namespace opencover {
namespace ui {
class Button;
class Menu;
class Group;
}
}

using namespace opencover;


class ClipPlanePlugin : public coVRPlugin, public ui::Owner
{
private:
    class Plane
    {
    public:
        bool enabled=false;
        bool valid=false; // valid is false before first use
        osg::ref_ptr<osg::ClipPlane> clip;
        ui::Group *UiGroup = nullptr;
        ui::Button *EnableButton = nullptr;
        ui::Button *DirectInteractorButton = nullptr;
        ui::Button *PickInteractorButton = nullptr;
        vrui::coTrackerButtonInteraction *directInteractor = nullptr;
        vrui::coRelativeInputInteraction *relativeInteractor = nullptr;
        coVR3DTransRotInteractor *pickInteractor = nullptr;
        bool showPickInteractor_=false, showDirectInteractor_=false;
        Plane();
        ~Plane();
    };
    Plane plane[coVRPluginSupport::MAX_NUM_CLIP_PLANES];

    ui::Menu *clipMenu = nullptr;

    osg::ref_ptr<osg::Geode> visibleClipPlaneGeode; // transparent plane
    bool active = false; // FLAG: ON = POSSIBLE TO CHANGE/SET CHOSEN CLIPPING PLANE
    osg::Matrix pointerMatrix; // needed for computing eqn
    osg::ref_ptr<osg::MatrixTransform> pointerTransform;
    osg::ref_ptr<osg::MatrixTransform> interactorTransform; // needed as parent node for the visible plane

    osg::Geode *loadPlane();

    osg::Vec4d matrixToEquation(const osg::Matrix &mat);

    void setInitialEquation(int);
    bool m_directInteractorShow = false, m_directInteractorEnable = false;

public:
    ClipPlanePlugin();
    virtual ~ClipPlanePlugin();
    bool init();
    void message(int toWhom, int type, int len, const void *buf);
    void preFrame();
};
