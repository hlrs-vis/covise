/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ColorAnimPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/VRSceneGraph.h>
#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Label.h>
#include <cover/ui/Group.h>
#include <config/CoviseConfig.h>
#include <osg/Material>
#include <osg/CullFace>
#include <osgDB/ReadFile>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace covise;
using namespace opencover;

ColorAnimPlugin::ColorAnimPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("ColorAnimPlugin", cover->ui)
, numFrames(275)
, currentFrame(0.0f)
, animationSpeed(0.02f)
, isPlaying(false)
, isPingPong(false)
, animationDirection(1.0f)
, normalsFlipped(false)
, interpolationMode(LINEAR)
{
}

ColorAnimPlugin::~ColorAnimPlugin()
{
    if (brainTransform.valid() && cover->getObjectsRoot())
    {
        cover->getObjectsRoot()->removeChild(brainTransform.get());
    }
}

// Rekursive Hilfsfunktion
osg::Geode* findFirstGeode(osg::Node* node)
{
    if (!node) return nullptr;
    
    osg::Geode* geode = dynamic_cast<osg::Geode*>(node);
    if (geode) return geode;
    
    osg::Group* group = dynamic_cast<osg::Group*>(node);
    if (group)
    {
        for (unsigned int i = 0; i < group->getNumChildren(); ++i)
        {
            osg::Geode* foundGeode = findFirstGeode(group->getChild(i));
            if (foundGeode) return foundGeode;
        }
    }
    return nullptr;
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
    speedSlider->setBounds(0.001, 0.1);
    speedSlider->setValue(0.02);
    speedSlider->setCallback([this](double value, bool released) {
        animationSpeed = (float)value;
    });

    // Ping-pong mode button
    pingPongButton = new ui::Button(animMenu, "Ping-Pong Mode");
    pingPongButton->setState(false);
    pingPongButton->setCallback([this](bool state) {
        isPingPong = state;
    });

    // Flip normals button
    flipNormalsButton = new ui::Button(animMenu, "Flip Normals (Inside View)");
    flipNormalsButton->setState(false);
    flipNormalsButton->setCallback([this](bool state) {
        flipNormals();
        normalsFlipped = state;
    });

    // Frame label
    frameLabel = new ui::Label(animMenu, "Frame");
    frameLabel->setText("Frame: 0 / 275");

    // Interpolation mode selection
    interpGroup = new ui::Group(animMenu, "Interpolation");

    interpLinear = new ui::Button(interpGroup, "Linear");
    interpLinear->setState(true);
    interpLinear->setCallback([this](bool state) {
        if (state) setInterpolationMode(LINEAR);
    });

    interpSmoothstep = new ui::Button(interpGroup, "Smoothstep");
    interpSmoothstep->setState(false);
    interpSmoothstep->setCallback([this](bool state) {
        if (state) setInterpolationMode(SMOOTHSTEP);
    });

    interpSmootherstep = new ui::Button(interpGroup, "Smootherstep");
    interpSmootherstep->setState(false);
    interpSmootherstep->setCallback([this](bool state) {
        if (state) setInterpolationMode(SMOOTHERSTEP);
    });

    interpEaseInOut = new ui::Button(interpGroup, "Ease In-Out");
    interpEaseInOut->setState(false);
    interpEaseInOut->setCallback([this](bool state) {
        if (state) setInterpolationMode(EASE_IN_OUT);
    });

    interpCubic = new ui::Button(interpGroup, "Cubic");
    interpCubic->setState(false);
    interpCubic->setCallback([this](bool state) {
        if (state) setInterpolationMode(CUBIC);
    });

    // Load brain models    
    std::string projectDir = coCoviseConfig::getEntry("value", "COVER.Plugin.ColorAnim.ProjectDir", "");
    bool highRes = coCoviseConfig::isOn("COVER.Plugin.ColorAnim.high_res", false);
    
    if (projectDir.empty())
    {
        fprintf(stderr, "ColorAnimPlugin: No project directory configured. Please set COVER.Plugin.ColorAnim.ProjectDir in config\n");
        fprintf(stderr, "ColorAnimPlugin: Example: /path/to/data/Horn/\n");
        return false;
    }

    std::string firstFilePath = "";
    if (highRes)
        firstFilePath = projectDir + "surfaces/dynamic/cortex_high_res/highres_cortex_001.ply";
    else
        firstFilePath = projectDir + "surfaces/dynamic/cortex/cortex_001.ply";

    if (!loadBrainModels(firstFilePath))
    {
        fprintf(stderr, "ColorAnimPlugin: Failed to load brain models\n");
        return false;
    }

    // Create scene graph structure
    brainTransform = new osg::MatrixTransform();
    brainTransform->setName("BrainTransform");
    brainGeode = new osg::Geode();
    brainGeode->setName("BrainGeode");

    if (brainGeometry.valid())
    {
        brainGeode->addDrawable(brainGeometry.get());
        brainTransform->addChild(brainGeode.get());

        // Add to scene
        cover->getObjectsRoot()->addChild(brainTransform.get());

        // Set initial colors
        updateColors();
    }

    // Load additional static models with transparency settings
    struct AnatomyModel {
        std::string filename;
        float transparency;
    };
    
    
    electrodeTransform = new osg::MatrixTransform();
    electrodeTransform->setName("ElectrodeTransform"); 

    brainGroup = new osg::Group();
    brainTransform->addChild(brainGroup.get());

    brainGroup->addChild(electrodeTransform.get());

    std::vector<AnatomyModel> anatomyModels = {
        {"surfaces/static/anatomy_halb.wrl", 1.0f},
        {"surfaces/static/left_electrode.wrl", 1.0f}
    };

    

    for (const auto& model : anatomyModels)
    {       
        std::string fn = projectDir + model.filename;
        std::cout << "Loading Anatomy file: " << fn.c_str() << std::endl << std::flush;
        osg::Node* loadedNode = coVRFileManager::instance()->loadFile(
            fn.c_str(),
            nullptr,
            electrodeTransform.get(),
            nullptr
        );

        
        if(loadedNode){
            //electrodeTransform->addChild(loadedNode);
        } else {
            std::cout << "Failed to load: " << fn.c_str() << std::endl << std::flush;
        }




        /*
        osg::Geode* geode = findFirstGeode(loadedNode);
        if (geode)
        {
            VRSceneGraph::instance()->setTransparency(geode, model.transparency);
            osg::StateSet* ss = geode->getOrCreateStateSet();
            ss->setMode(GL_BLEND, osg::StateAttribute::ON);
            ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
            ss->setNestRenderBins(false);

            //osg::StateSet* stateTransp = VRSceneGraph::instance()->loadTransparentGeostate(osg::Material::ColorMode::AMBIENT_AND_DIFFUSE);
            //geode->setStateSet(stateTransp);        
            
        }
        */
    }

    fprintf(stderr, "ColorAnimPlugin: Initialization complete\n");
    return true;
}

bool ColorAnimPlugin::loadBrainModels(const std::string &firstFilePath)
{
    fprintf(stderr, "ColorAnimPlugin: Loading brain models starting from %s\n", firstFilePath.c_str());

    // Parse the first file path to extract directory, base name, number, and extension
    size_t lastSlash = firstFilePath.find_last_of("/\\");
    std::string directory = (lastSlash != std::string::npos) ? firstFilePath.substr(0, lastSlash + 1) : "";
    std::string filename = (lastSlash != std::string::npos) ? firstFilePath.substr(lastSlash + 1) : firstFilePath;

    // Find the last sequence of digits in the filename
    size_t digitEnd = filename.length();
    size_t digitStart = digitEnd;

    // Find extension (last dot)
    size_t extPos = filename.find_last_of('.');
    std::string extension = (extPos != std::string::npos) ? filename.substr(extPos) : "";

    // Look for digits before the extension
    if (extPos != std::string::npos)
    {
        digitEnd = extPos;
        digitStart = extPos;

        // Find the start of the digit sequence
        while (digitStart > 0 && isdigit(filename[digitStart - 1]))
        {
            digitStart--;
        }
    }

    if (digitStart >= digitEnd)
    {
        fprintf(stderr, "ColorAnimPlugin: Could not find number pattern in filename: %s\n", filename.c_str());
        return false;
    }

    std::string baseFileName = filename.substr(0, digitStart);
    std::string numberStr = filename.substr(digitStart, digitEnd - digitStart);
    int startNumber = std::atoi(numberStr.c_str());
    int numberWidth = numberStr.length(); // Zero-padding width

    fprintf(stderr, "ColorAnimPlugin: Parsed pattern - base: '%s', start: %d, width: %d, ext: '%s'\n",
            baseFileName.c_str(), startNumber, numberWidth, extension.c_str());

    // Try to load the first model to get geometry
    osg::ref_ptr<osg::Node> firstModel = osgDB::readNodeFile(firstFilePath);

    if (!firstModel.valid())
    {
        fprintf(stderr, "ColorAnimPlugin: Could not load first model: %s\n", firstFilePath.c_str());
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
        modelPath << directory << baseFileName << std::setfill('0') << std::setw(numberWidth)
                  << (startNumber + i) << extension;

        osg::ref_ptr<osg::Node> model = osgDB::readNodeFile(modelPath.str());

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

    // Apply interpolation curve
    float t_curved = applyInterpolationCurve(t);

    size_t numColors = std::min(colors1->size(), colors2->size());
    for (size_t i = 0; i < numColors; ++i)
    {
        osg::Vec4 c1 = (*colors1)[i];
        osg::Vec4 c2 = (*colors2)[i];
        osg::Vec4 interpolated = c1 * (1.0f - t_curved) + c2 * t_curved;
        result->push_back(interpolated);
    }

    return result;
}

float ColorAnimPlugin::applyInterpolationCurve(float t)
{
    // Clamp t to [0, 1]
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    switch (interpolationMode)
    {
    case LINEAR:
        return t;

    case SMOOTHSTEP:
        // Smoothstep: 3t^2 - 2t^3
        return t * t * (3.0f - 2.0f * t);

    case SMOOTHERSTEP:
        // Smootherstep: 6t^5 - 15t^4 + 10t^3
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);

    case EASE_IN_OUT:
        // Cosine-based ease in-out
        return 0.5f - 0.5f * std::cos(t * M_PI);

    case CUBIC:
        // Cubic ease in-out
        if (t < 0.5f)
            return 4.0f * t * t * t;
        else
        {
            float f = (2.0f * t - 2.0f);
            return 0.5f * f * f * f + 1.0f;
        }

    default:
        return t;
    }
}

void ColorAnimPlugin::setInterpolationMode(InterpolationMode mode)
{
    interpolationMode = mode;

    // Update button states
    interpLinear->setState(mode == LINEAR);
    interpSmoothstep->setState(mode == SMOOTHSTEP);
    interpSmootherstep->setState(mode == SMOOTHERSTEP);
    interpEaseInOut->setState(mode == EASE_IN_OUT);
    interpCubic->setState(mode == CUBIC);

    // Immediately update colors to show the effect
    updateColors();
}

void ColorAnimPlugin::preFrame()
{
    if (!isPlaying || colorFrames.empty())
        return;

    // Update animation
    float frameIncrement = animationSpeed * animationDirection * cover->frameDuration() * 30.0f; // 30 fps base
    currentFrame += frameIncrement;

    // Handle animation boundaries
    if (isPingPong)
    {
        // Ping-pong mode: reverse direction at boundaries
        if (currentFrame >= (float)colorFrames.size())
        {
            currentFrame = (float)colorFrames.size() - 0.01f;
            animationDirection = -1.0f;
        }
        else if (currentFrame < 0.0f)
        {
            currentFrame = 0.0f;
            animationDirection = 1.0f;
        }
    }
    else
    {
        // Loop mode: wrap around at end
        if (currentFrame >= (float)colorFrames.size())
        {
            currentFrame = 0.0f;
        }
        else if (currentFrame < 0.0f)
        {
            currentFrame = (float)colorFrames.size() - 0.01f;
        }
    }

    updateColors();
}

void ColorAnimPlugin::flipNormals()
{
    if (!brainGeometry.valid())
        return;

    osg::Vec3Array *normals = dynamic_cast<osg::Vec3Array*>(brainGeometry->getNormalArray());
    if (!normals)
        return;

    // Flip all normals
    for (size_t i = 0; i < normals->size(); ++i)
    {
        (*normals)[i] = -(*normals)[i];
    }

    // Mark the array as modified
    normals->dirty();
    brainGeometry->dirtyBound();

    // Also toggle backface culling to see the flipped geometry properly
    osg::StateSet *stateSet = brainGeometry->getOrCreateStateSet();
    osg::CullFace *cullFace = dynamic_cast<osg::CullFace*>(
        stateSet->getAttribute(osg::StateAttribute::CULLFACE));

    if (normalsFlipped)
    {
        // When flipping back to normal, cull back faces (default)
        if (!cullFace)
        {
            cullFace = new osg::CullFace(osg::CullFace::BACK);
            stateSet->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
        }
        else
        {
            cullFace->setMode(osg::CullFace::BACK);
        }
    }
    else
    {
        // When viewing from inside, cull front faces or disable culling
        if (!cullFace)
        {
            cullFace = new osg::CullFace(osg::CullFace::FRONT);
            stateSet->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
        }
        else
        {
            cullFace->setMode(osg::CullFace::FRONT);
        }
    }
}

void ColorAnimPlugin::setupVertexColorMaterial(osg::Node *node)
{
    if (!node)
        return;

    // Set up material to use vertex colors
    osg::StateSet *stateSet = node->getOrCreateStateSet();
    osg::Material *material = new osg::Material();
    material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);

    // If this is a geode, also set the material on each drawable
    osg::Geode *geode = dynamic_cast<osg::Geode*>(node);
    if (geode)
    {
        for (unsigned int i = 0; i < geode->getNumDrawables(); ++i)
        {
            osg::Drawable *drawable = geode->getDrawable(i);
            if (drawable)
            {
                osg::StateSet *drawableStateSet = drawable->getOrCreateStateSet();
                osg::Material *drawableMaterial = new osg::Material();
                drawableMaterial->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
                drawableStateSet->setAttributeAndModes(drawableMaterial, osg::StateAttribute::ON);
            }
        }
    }

    // Recursively process child nodes
    osg::Group *group = node->asGroup();
    if (group)
    {
        for (unsigned int i = 0; i < group->getNumChildren(); ++i)
        {
            setupVertexColorMaterial(group->getChild(i));
        }
    }
}

COVERPLUGIN(ColorAnimPlugin)
