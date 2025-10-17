/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COLORANIM_PLUGIN_H
#define _COLORANIM_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <cover/ui/Owner.h>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <vector>
#include <string>

namespace opencover
{
namespace ui
{
class Button;
class Menu;
class Slider;
class Label;
}
}

using namespace opencover;

class ColorAnimPlugin : public coVRPlugin, public ui::Owner
{
public:
    ColorAnimPlugin();
    ~ColorAnimPlugin();

    bool init();
    void preFrame();

private:
    // UI elements
    ui::Menu *animMenu = nullptr;
    ui::Slider *speedSlider = nullptr;
    ui::Button *playButton = nullptr;
    ui::Button *resetButton = nullptr;
    ui::Label *frameLabel = nullptr;

    // Scene graph elements
    osg::ref_ptr<osg::MatrixTransform> brainTransform;
    osg::ref_ptr<osg::Geode> brainGeode;
    osg::ref_ptr<osg::Geometry> brainGeometry;

    // Animation data
    std::vector<osg::ref_ptr<osg::Vec4Array>> colorFrames;
    int numFrames;
    float currentFrame;
    float animationSpeed;
    bool isPlaying;

    // Helper methods
    bool loadBrainModels(const std::string &path);
    void updateColors();
    osg::Vec4Array* interpolateColors(int frame1, int frame2, float t);
};

#endif
