/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*  GL Clipping Plane module

//  Author : Daniela Rainer
//  Date   : 04-Feb-99

*/

#include <cover/mui/support/Listener.h>

#include <cover/coVRPlugin.h>
#include <osg/ClipPlane>

namespace vrui
{
class coButtonMenuItem;
class coTrackerButtonInteraction;
}

namespace mui
{
class ToggleButton;
class Tab;
}

using namespace vrui;
using namespace opencover;

#include <PluginUtil/coVR3DTransRotInteractor.h>

class ClipPlanePlugin : public coVRPlugin, public mui::Listener
{
private:
    class Plane
    {
    public:
        bool enabled;
        bool valid; // valid is false before first use
        osg::ref_ptr<osg::ClipPlane> clip;
        mui::ToggleButton *EnableButton;
        mui::ToggleButton *DirectInteractorButton;
        mui::ToggleButton *PickInteractorButton;
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
            EnableButton = NULL;
            DirectInteractorButton = NULL;
            PickInteractorButton = NULL;
        }
        ~Plane()
        {
            if (directInteractor)
                delete directInteractor;
            delete pickInteractor;
            delete EnableButton;
            delete DirectInteractorButton;
            delete PickInteractorButton;
        }
    };
    Plane plane[coVRPluginSupport::MAX_NUM_CLIP_PLANES];

    mui::Tab *clipTab;

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
    void muiEvent(mui::Element *muiItem);
    void muiPressEvent(mui::Element *muiItem);
    void muiReleaseEvent(mui::Element *muiItem);
    void message(int type, int len, const void *buf);
    void preFrame();
    static ClipPlanePlugin *thisInstance;
};
