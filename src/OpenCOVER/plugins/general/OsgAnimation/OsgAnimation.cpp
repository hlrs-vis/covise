/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/ui/FileBrowser.h>
#include <cover/ui/Owner.h>
#include <osg/MatrixTransform>
#include <map>
#include <cover/ui/Menu.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Action.h>

#include <osg/NodeVisitor>

#include <osgAnimation/BasicAnimationManager>
#include <osgAnimation/RigTransformHardware>
#include <osgAnimation/Animation>
#include <osgAnimation/Skeleton>
#include <osgGA/GUIEventHandler>
#include <osgAnimation/StackedRotateAxisElement>

using namespace covise;
using namespace opencover;
using namespace ui;

struct AnimationManagerFinder : public osg::NodeVisitor
{
    osg::ref_ptr<osgAnimation::BasicAnimationManager> m_am;
    AnimationManagerFinder() : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN) {}
    void apply(osg::Node &node) override
    {

        if (m_am.valid())
            return;

        if (node.getUpdateCallback())
        {
            m_am = dynamic_cast<osgAnimation::BasicAnimationManager *>(node.getUpdateCallback());
            return;
        }

        traverse(node);
    }
};

class OsgAnimation : public coVRPlugin, public ui::Owner
{
public:
    void loadAnimations()
    {
        cover->getObjectsRoot()->accept(m_amFinder);
        if (m_amFinder.m_am.valid())
        {
            std::cerr << "Found AnimationManager" << std::endl;
        }
        else
        {
            std::cerr << "No AnimationManager found" << std::endl;
        }
        for(const auto & anim : m_amFinder.m_am->getAnimationList())
        {
            auto slider = new ui::Slider(m_menu, anim->getName());
            slider->setBounds(0, 1);
            slider->setCallback([this, &anim](double val, bool x) {
                m_amFinder.m_am->playAnimation(anim, 1, val);
            });
            m_sliders.push_back(slider);

        }
    }
    OsgAnimation()
    :coVRPlugin(COVER_PLUGIN_NAME)
    , Owner(COVER_PLUGIN_NAME, cover->ui)
    , m_menu(new ui::Menu("OsgAnimation", this))
    {
        auto reloadButton = new ui::Action(m_menu, "Reload Animations");
        reloadButton->setCallback([this]() {
            for (auto slider : m_sliders)
                delete slider;
            m_sliders.clear();
            loadAnimations();
        });

        loadAnimations();
    }


private:

    AnimationManagerFinder m_amFinder;
    std::vector<ui::Slider *> m_sliders;
    ui::Menu *m_menu = nullptr;
};

COVERPLUGIN(OsgAnimation)