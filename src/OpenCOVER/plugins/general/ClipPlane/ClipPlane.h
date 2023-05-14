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
#include <vrb/client/SharedState.h>

#include <array>
#include <memory>

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
class SelectionList;
}
}

#include <OpenConfig/value.h>

using namespace opencover;


class ClipPlanePlugin : public coVRPlugin, public ui::Owner
{
private:
    struct Plane
    {
        int num = -1;
        std::unique_ptr<ConfigBool> enabled;
        std::unique_ptr<ConfigFloatArray> equation;
        bool inLocalUpdate = false; // inhibit updating interactor from equation
        bool valid=false; // valid is false before first use
        osg::ref_ptr<osg::ClipPlane> clip;
        ui::Group *UiGroup = nullptr;
        ui::Button *EnableButton = nullptr;
        ui::Button *DirectInteractorButton = nullptr;
        ui::Button *PickInteractorButton = nullptr;
        ui::SelectionList *RootChoice = nullptr;
        vrui::coTrackerButtonInteraction *directInteractor = nullptr;
        vrui::coRelativeInputInteraction *relativeInteractor = nullptr;
        coVR3DTransRotInteractor *pickInteractor = nullptr;
        bool showPickInteractor_=false, showDirectInteractor_=false;
        Plane();
        ~Plane();
        void set(const osg::Vec4d &eq);
        osg::ref_ptr<osg::ClipNode> clipNode;
        osg::ClipNode *getClipNode() const;
        void setClipNode(osg::ClipNode *cn);
    };
    struct ClipNodeData
    {
        std::string plugin;
        std::string name;
        osg::ref_ptr<osg::ClipNode> node;
    };
    std::array<Plane, coVRPluginSupport::MAX_NUM_CLIP_PLANES> plane;
    std::unique_ptr<vrb::SharedState<std::vector<double>>> sharedPlanes[coVRPluginSupport::MAX_NUM_CLIP_PLANES];
    //vrb::SharedState<std::vector<double>> sharedPlane_1, sharedPlane_2, sharedPlane_3;
    void updateRootChoices();

    ui::Menu *clipMenu = nullptr;

    osg::ref_ptr<osg::Geode> visibleClipPlaneGeode; // transparent plane
    bool active = false; // FLAG: ON = POSSIBLE TO CHANGE/SET CHOSEN CLIPPING PLANE
    osg::Matrix pointerMatrix; // needed for computing eqn
    osg::ref_ptr<osg::MatrixTransform> pointerTransform;
    osg::ref_ptr<osg::MatrixTransform> interactorTransform; // needed as parent node for the visible plane

    osg::Geode *loadPlane();

    osg::Vec4d matrixToEquation(const osg::Matrix &mat);

    void setInitialEquation(int);
    osg::Matrix equationToMatrix(int);
    bool m_directInteractorShow = false, m_directInteractorEnable = false;
    std::list<ClipNodeData> clipNodes;

public:
    ClipPlanePlugin();
    ~ClipPlanePlugin() override;
    bool init() override;
    void message(int toWhom, int type, int len, const void *buf) override;
    void preFrame() override;
    void addNodeFromPlugin(osg::Node *node, const RenderObject *, coVRPlugin *addingPlugin) override;
    void removeNode(osg::Node *node, bool isGroup, osg::Node *realNode) override;
};
