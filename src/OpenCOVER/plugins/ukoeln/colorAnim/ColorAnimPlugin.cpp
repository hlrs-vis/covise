/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ColorAnimPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Label.h>
#include <osg/Material>
#include <osgDB/ReadFile>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace opencover;

ColorAnimPlugin::ColorAnimPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("ColorAnimPlugin", cover->ui)
, numFrames(275)
, currentFrame(0.0f)
, animationSpeed(1.0f)
, isPlaying(false)
{
}

ColorAnimPlugin::~ColorAnimPlugin()
{
    if (brainTransform.valid() && cover->getObjectsRoot())
    {
        cover->getObjectsRoot()->removeChild(brainTransform.get());
    }
}

bool ColorAnimPlugin::init()
{
    fprintf(stderr, "ColorAnimPlugin::init\n");

    // Create UI Menu
    animMenu = new ui::Menu("Brain Color Animation", this);

    // Play/Pause button
    playButton = new ui::Button(animMenu, "Play");
    playButton->setState(false);
    playButton->setCallback([this](bool state) {
        isPlaying = state;
        if (state)
            playButton->setText("Pause");
        else
            playButton->setText("Play");
    });

    // Reset button
    resetButton = new ui::Button(animMenu, "Reset");
    resetButton->setState(false);
    resetButton->setCallback([this](bool state) {
        if (state)
        {
            currentFrame = 0.0f;
            updateColors();
            resetButton->setState(false);
        }
    });

    // Speed slider
    speedSlider = new ui::Slider(animMenu, "Speed");
    speedSlider->setBounds(0.1, 10.0);
    speedSlider->setValue(1.0);
    speedSlider->setCallback([this](double value, bool released) {
        animationSpeed = (float)value;
    });

    // Frame label
    frameLabel = new ui::Label(animMenu, "Frame");
    frameLabel->setText("Frame: 0 / 275");

    // Load brain models
    std::string brainPath = coCoviseConfig::getEntry("value", "COVER.Plugin.ColorAnim.ModelPath", "");
    if (brainPath.empty())
    {
        fprintf(stderr, "ColorAnimPlugin: No model path configured. Please set COVER.Plugin.ColorAnim.ModelPath in config\n");
        fprintf(stderr, "ColorAnimPlugin: Trying to load from current directory ./brain_*.obj\n");
        brainPath = "./brain";
    }

    if (!loadBrainModels(brainPath))
    {
        fprintf(stderr, "ColorAnimPlugin: Failed to load brain models\n");
        return false;
    }

    // Create scene graph structure
    brainTransform = new osg::MatrixTransform();
    brainGeode = new osg::Geode();

    if (brainGeometry.valid())
    {
        brainGeode->addDrawable(brainGeometry.get());
        brainTransform->addChild(brainGeode.get());

        // Add to scene
        cover->getObjectsRoot()->addChild(brainTransform.get());

        // Set initial colors
        updateColors();
    }

    fprintf(stderr, "ColorAnimPlugin: Initialization complete\n");
    return true;
}

bool ColorAnimPlugin::loadBrainModels(const std::string &basePath)
{
    fprintf(stderr, "ColorAnimPlugin: Loading brain models from %s\n", basePath.c_str());

    // Try to load the first model to get geometry
    std::ostringstream firstPath;
    firstPath << basePath << "_001.obj";

    osg::ref_ptr<osg::Node> firstModel = osgDB::readNodeFile(firstPath.str());
    if (!firstModel.valid())
    {
        // Try alternative naming
        firstPath.str("");
        firstPath << basePath << "_0.obj";
        firstModel = osgDB::readNodeFile(firstPath.str());
    }

    if (!firstModel.valid())
    {
        fprintf(stderr, "ColorAnimPlugin: Could not load first model: %s\n", firstPath.str().c_str());
        return false;
    }

    // Extract geometry from the loaded model
    osg::Geode *geode = dynamic_cast<osg::Geode*>(firstModel.get());
    if (!geode && firstModel->asGroup() && firstModel->asGroup()->getNumChildren() > 0)
    {
        geode = dynamic_cast<osg::Geode*>(firstModel->asGroup()->getChild(0));
    }

    if (!geode || geode->getNumDrawables() == 0)
    {
        fprintf(stderr, "ColorAnimPlugin: Could not find geometry in model\n");
        return false;
    }

    osg::Geometry *sourceGeom = dynamic_cast<osg::Geometry*>(geode->getDrawable(0));
    if (!sourceGeom)
    {
        fprintf(stderr, "ColorAnimPlugin: Could not extract geometry\n");
        return false;
    }

    // Create our own geometry with the same structure
    brainGeometry = new osg::Geometry();
    brainGeometry->setVertexArray(sourceGeom->getVertexArray());
    brainGeometry->setNormalArray(sourceGeom->getNormalArray());
    brainGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    // Copy primitive sets
    for (unsigned int i = 0; i < sourceGeom->getNumPrimitiveSets(); ++i)
    {
        brainGeometry->addPrimitiveSet(sourceGeom->getPrimitiveSet(i));
    }

    // Set material
    osg::StateSet *stateSet = brainGeometry->getOrCreateStateSet();
    osg::Material *material = new osg::Material();
    material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);

    // Load all color frames
    fprintf(stderr, "ColorAnimPlugin: Loading %d color frames...\n", numFrames);

    for (int i = 0; i < numFrames; ++i)
    {
        std::ostringstream modelPath;
        modelPath << basePath << "_" << std::setfill('0') << std::setw(3) << (i + 1) << ".obj";

        osg::ref_ptr<osg::Node> model = osgDB::readNodeFile(modelPath.str());

        // Try alternative naming if failed
        if (!model.valid())
        {
            modelPath.str("");
            modelPath << basePath << "_" << i << ".obj";
            model = osgDB::readNodeFile(modelPath.str());
        }

        if (!model.valid())
        {
            fprintf(stderr, "ColorAnimPlugin: Warning - Could not load frame %d: %s\n", i, modelPath.str().c_str());
            // Create a default color array
            osg::Vec4Array *defaultColors = new osg::Vec4Array();
            int numVerts = sourceGeom->getVertexArray()->getNumElements();
            for (int v = 0; v < numVerts; ++v)
            {
                defaultColors->push_back(osg::Vec4(0.8f, 0.8f, 0.8f, 1.0f));
            }
            colorFrames.push_back(defaultColors);
            continue;
        }

        // Extract color array from this model
        osg::Geode *frameGeode = dynamic_cast<osg::Geode*>(model.get());
        if (!frameGeode && model->asGroup() && model->asGroup()->getNumChildren() > 0)
        {
            frameGeode = dynamic_cast<osg::Geode*>(model->asGroup()->getChild(0));
        }

        if (frameGeode && frameGeode->getNumDrawables() > 0)
        {
            osg::Geometry *frameGeom = dynamic_cast<osg::Geometry*>(frameGeode->getDrawable(0));
            if (frameGeom && frameGeom->getColorArray())
            {
                osg::Vec4Array *colors = dynamic_cast<osg::Vec4Array*>(frameGeom->getColorArray());
                if (colors)
                {
                    colorFrames.push_back(colors);
                }
                else
                {
                    // Convert Vec3Array to Vec4Array if needed
                    osg::Vec3Array *colors3 = dynamic_cast<osg::Vec3Array*>(frameGeom->getColorArray());
                    if (colors3)
                    {
                        osg::Vec4Array *colors4 = new osg::Vec4Array();
                        for (size_t v = 0; v < colors3->size(); ++v)
                        {
                            colors4->push_back(osg::Vec4((*colors3)[v], 1.0f));
                        }
                        colorFrames.push_back(colors4);
                    }
                }
            }
        }

        if ((i + 1) % 50 == 0)
        {
            fprintf(stderr, "ColorAnimPlugin: Loaded %d frames...\n", i + 1);
        }
    }

    fprintf(stderr, "ColorAnimPlugin: Loaded %d color frames\n", (int)colorFrames.size());
    return colorFrames.size() > 0;
}

void ColorAnimPlugin::updateColors()
{
    if (!brainGeometry.valid() || colorFrames.empty())
        return;

    int frame1 = (int)currentFrame;
    int frame2 = (frame1 + 1) % colorFrames.size();
    float t = currentFrame - frame1;

    // Get interpolated colors
    osg::Vec4Array *colors = interpolateColors(frame1, frame2, t);

    brainGeometry->setColorArray(colors);
    brainGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    // Update frame label
    std::ostringstream ss;
    ss << "Frame: " << (int)currentFrame << " / " << numFrames;
    frameLabel->setText(ss.str());
}

osg::Vec4Array* ColorAnimPlugin::interpolateColors(int frame1, int frame2, float t)
{
    if (frame1 >= (int)colorFrames.size())
        frame1 = colorFrames.size() - 1;
    if (frame2 >= (int)colorFrames.size())
        frame2 = colorFrames.size() - 1;

    osg::Vec4Array *colors1 = colorFrames[frame1].get();
    osg::Vec4Array *colors2 = colorFrames[frame2].get();

    osg::Vec4Array *result = new osg::Vec4Array();

    size_t numColors = std::min(colors1->size(), colors2->size());
    for (size_t i = 0; i < numColors; ++i)
    {
        osg::Vec4 c1 = (*colors1)[i];
        osg::Vec4 c2 = (*colors2)[i];
        osg::Vec4 interpolated = c1 * (1.0f - t) + c2 * t;
        result->push_back(interpolated);
    }

    return result;
}

void ColorAnimPlugin::preFrame()
{
    if (!isPlaying || colorFrames.empty())
        return;

    // Update animation
    float frameIncrement = animationSpeed * cover->frameDuration() * 30.0f; // 30 fps base
    currentFrame += frameIncrement;

    // Loop animation
    if (currentFrame >= (float)colorFrames.size())
    {
        currentFrame = 0.0f;
    }

    updateColors();
}

COVERPLUGIN(ColorAnimPlugin)
