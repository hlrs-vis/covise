/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// VolumePlugin.cpp
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <vector>

#include <boost/make_shared.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/stream.hpp>

#include "VolumePlugin.h"

#include <util/common.h>
#include <util/unixcompat.h>
#include <assert.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRCollaboration.h>
#include <cover/coVRNavigationManager.h>
#include <cover/VRSceneGraph.h>
#include <cover/VRViewer.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginList.h>
#include <cover/coCollabInterface.h>

#include <cover/ui/Label.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Menu.h>

#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>

#include <PluginUtil/PluginMessageTypes.h>

#include <cover/VRViewer.h>

#include "coClipSphere.h"
#include "coDefaultFunctionEditor.h"
#include "coPinEditor.h"
#include <cover/RenderObject.h>
#include <virvo/vvdebugmsg.h>
#include <virvo/vvtoolshed.h>
#include <virvo/vvvecmath.h>
#include <virvo/vvfileio.h>

#include <OpenVRUI/osg/mathUtils.h>
#include <config/CoviseConfig.h>

#include <osg/Material>
#include <osg/StateSet>
#include <osgDB/ReadFile>
#include <osg/io_utils>

using namespace osg;
using namespace vrui;

covise::TokenBuffer& operator<<(covise::TokenBuffer& tb, const vvTransFunc& id);
covise::TokenBuffer& operator>>(covise::TokenBuffer& tb, vvTransFunc& id);

namespace covise
{
	template <>
	TokenBufferDataType getTokenBufferDataType < vvTransFunc >(const vvTransFunc& type) {
		return TokenBufferDataType::TRANSFERFUNCTION;
	}
}
#undef VERBOSE

VolumePlugin *VolumePlugin::plugin = NULL;
scoped_ptr<coCOIM> VolumeCoim; // keep before other items (to be destroyed last)

VolumePlugin::Volume::Volume()
{
    inScene = false;
    curChannel = 0;
    multiDimTF = true;
    useChannelWeights = true;

    std::string rendererName = covise::coCoviseConfig::getEntry("COVER.Plugin.Volume.Renderer");
    std::string voxType = covise::coCoviseConfig::getEntry("voxelType", "COVER.Plugin.Volume.Renderer");

    drawable = new virvo::VolumeDrawable(rendererName, voxType);
    drawable->enableFlatDisplay(coVRConfig::instance()->haveFlatDisplay());
    drawable->setROIPosition(Vec3(0., 0., 0.));

    ref_ptr<Material> mtl = new Material();
    mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    mtl->setShininess(Material::FRONT_AND_BACK, 16.0f);

    ref_ptr<StateSet> geoState = new StateSet();
    geoState->setGlobalDefaults();
    geoState->setAttributeAndModes(mtl.get(), StateAttribute::ON);
    geoState->setMode(GL_LIGHTING, StateAttribute::ON);
    geoState->setMode(GL_BLEND, StateAttribute::ON);
    geoState->setRenderingHint(StateSet::TRANSPARENT_BIN);
    geoState->setNestRenderBins(false);

    node = new osg::Geode();
    node->setStateSet(geoState.get());
    node->setNodeMask(node->getNodeMask() & ~Isect::Intersection);
    node->addDrawable(drawable.get());

    transform = new osg::MatrixTransform();
    auto mirror = osg::Matrix::identity();
    mirror(0,0) = -1;
    transform->setMatrix(osg::Matrix::rotate(M_PI*0.5, osg::Vec3(0,1,0)) * osg::Matrix::rotate(M_PI, osg::Vec3(1,0,0)) * mirror);
    transform->addChild(node);

    min = max = osg::Vec3(0., 0., 0.);

    roiPosObj[0] = INITIAL_POS_X;
    roiPosObj[1] = INITIAL_POS_Y;
    roiPosObj[2] = INITIAL_POS_Z;

    roiCellSize = 0.3;
    roiMode = false;
    boundaries = false;
    preIntegration = false;
    lighting = false;
    interpolation = true;
    blendMode = virvo::VolumeDrawable::AlphaBlend;
    mapTF = true;
}

/** This creates an empty icon texture.
 */
Geode *VolumePlugin::Volume::createImage(string &filename)
{
    const float WIDTH = 256.0f;
    const float HEIGHT = 256.0f;
    const float ZPOS = 130.0f;

#ifdef VERBOSE
    cerr << "createImage " << filename << endl;
#endif

    Geometry *geom = new Geometry();
    Texture2D *icon = new Texture2D();
    Image *image = NULL;
    /*
  image = new Image();
  char* img = new char[2*2*4];
  memset(img, 0, 2*2*4);
  image->setImage(2, 2, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, (unsigned char*)img, Image::USE_NEW_DELETE);
*/
    image = osgDB::readImageFile(filename);
    icon->setImage(image);

    Vec3Array *vertices = new Vec3Array(4);
    // bottom left
    (*vertices)[0].set(-WIDTH / 2.0, -HEIGHT / 2.0, ZPOS);
    // bottom right
    (*vertices)[1].set(WIDTH / 2.0, -HEIGHT / 2.0, ZPOS);
    // top right
    (*vertices)[2].set(WIDTH / 2.0, HEIGHT / 2.0, ZPOS);
    // top left
    (*vertices)[3].set(-WIDTH / 2.0, HEIGHT / 2.0, ZPOS);
    geom->setVertexArray(vertices);

    Vec2Array *texcoords = new Vec2Array(4);
    (*texcoords)[0].set(0.0, 0.0);
    (*texcoords)[1].set(1.0, 0.0);
    (*texcoords)[2].set(1.0, 1.0);
    (*texcoords)[3].set(0.0, 1.0);
    geom->setTexCoordArray(0, texcoords);

    Vec3Array *normals = new Vec3Array(1);
    (*normals)[0].set(0.0f, 0.0f, 1.0f);
    geom->setNormalArray(normals);
    geom->setNormalBinding(Geometry::BIND_OVERALL);

    Vec4Array *colors = new Vec4Array(1);
    (*colors)[0].set(1.0, 1.0, 1.0, 1.0);
    geom->setColorArray(colors);
    geom->setColorBinding(Geometry::BIND_OVERALL);
    geom->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));

    // Texture:
    StateSet *stateset = geom->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, StateAttribute::OFF);
    stateset->setTextureAttributeAndModes(0, icon, StateAttribute::ON);

    Geode *imageGeode = new Geode();
    imageGeode->addDrawable(geom);

    return imageGeode;
}

void VolumePlugin::Volume::addToScene(osg::Group *group)
{
#ifdef VERBOSE
    cerr << "add volume to scene" << endl;
#endif
    if (!inScene)
    {
        if (group)
            group->addChild(transform.get());
        else
            cover->getObjectsRoot()->addChild(transform.get());
    }

    inScene = true;
}

void VolumePlugin::Volume::removeFromScene()
{
    if (cover->debugLevel(3))
        cerr << "remove volume from scene: draw=" << drawable.get() << endl;
    coVRAnimationManager::instance()->removeTimestepProvider(drawable.get());

    if (inScene && transform.get())
    {
        osg::Node::ParentList parents = transform->getParents();
        for (osg::Node::ParentList::iterator parent = parents.begin();
             parent != parents.end();
             ++parent)
            (*parent)->removeChild(transform.get());
        inScene = false;
    }
}

VolumePlugin::Volume::~Volume()
{
    if (cover->debugLevel(3))
        cerr << "delete volume" << endl;
}

FileEntry::FileEntry(const char *fN, const char *mN)
{
    char *buf;

    fileName = new char[strlen(fN) + 1];
    strcpy(fileName, fN);

    // Register menu name:
    if (mN == NULL)
    {
        // If NULL then use file name as menu name:
        buf = new char[strlen(fN) + 1];
        vvToolshed::extractBasename(buf, fN);
        menuName = new char[strlen(buf) + 1];
        strcpy(menuName, buf);
        delete[] buf;
    }
    else
    {
        menuName = new char[strlen(mN) + 1];
        strcpy(menuName, mN);
    }

    // Create a menu entry:
    fileMenuItem = new ui::Action(VolumePlugin::plugin->filesGroup, menuName);
    fileMenuItem->setCallback([this](){
        cover->sendMessage(VolumePlugin::plugin,
                           coVRPluginSupport::TO_SAME, PluginMessageTypes::VolumeLoadFile,
                           strlen(fileName) + 1, fileName);
    });
}

FileEntry::~FileEntry()
{
    delete[] fileName;
    delete[] menuName;
    delete fileMenuItem;
}

int
VolumePlugin::loadUrl(const Url &url, Group *parent, const char *ck)
{
    if (url.scheme() != "file" && url.scheme() != "dicom")
        return -1;

    const char *filename = url.path().c_str();
    vvVolDesc vd(filename);

    if (url.scheme() == "dicom") {
        auto q = url.query();
        auto pos = q.find("entry");
        if (pos != std::string::npos)
        {
            auto entry = q.substr(pos);
            auto valpos = entry.find('=');
            if (valpos != std::string::npos)
            {
                auto valstr = entry.substr(valpos+1);
                int ent;
                std::stringstream str(valstr);
                str >> ent;
                vd.setEntry(ent);
            }
        }
    }

    int result = plugin->loadFile(filename, parent, &vd);
    return (result == 0) ? -1 : 0;
}

int
VolumePlugin::loadVolume(const char *filename, osg::Group *parent, const char *)
{
#ifdef VERBOSE
    fprintf(stderr, "VolumePlugin::loadVolume(%s)\n", filename);
#endif
    int result = plugin->loadFile(filename, parent);
    return (result == 0) ? -1 : 0;
}

int
VolumePlugin::unloadVolume(const char *filename, const char *)
{
    if (plugin && filename)
    {
        bool ok = plugin->updateVolume(filename, NULL);
        return ok ? 0 : -1;
    }
    return -1;
}

FileHandler fileHandler[] = {
    { NULL,
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "tif" },
    { NULL,
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "tiff" },
    { NULL,
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "xvf" },
    { NULL,
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "rvf" },
    { NULL,
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "avf" },
    { NULL,
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "nii" },
    { NULL,
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "nii.gz" },
    { VolumePlugin::loadUrl,
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "dicomdir" },
};

/// Constructor
VolumePlugin::VolumePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("VolumePlugin", cover->ui)
, editor(NULL)
{
}

bool VolumePlugin::init()
{
    std::cerr << "VolumePlugin::init" << std::endl;

    vvDebugMsg::msg(1, "VolumePlugin::VolumePlugin()");

    std::string rendererName = covise::coCoviseConfig::getEntry("COVER.Plugin.Volume.Renderer");
    bool enableSphereClipping = rendererName.find("rayrend") == 0;

    // set virvo debug level ------------------------------

    // Debug level value may be either [NO_MESSAGES|FEW_MESSAGES|MOST_MESSAGES|ALL_MESSAGES]
    // Or, in the same order and meaning the same as the string equivalents [0|1|2|3]
    bool debugLevelExists = false;
    int debugLevelInt = covise::coCoviseConfig::getInt("COVER.Plugin.Volume.DebugLevel", 0, &debugLevelExists);

    if (debugLevelExists)
    {
        if ((debugLevelInt >= 0) && (debugLevelInt <= 9))
        {
            vvDebugMsg::setDebugLevel(debugLevelInt);
        }
        else
        {
            // In that case, the debug level was specified as a string literal
            std::string debugLevelStr = covise::coCoviseConfig::getEntry("COVER.Plugin.Volume.DebugLevel");
            if (!debugLevelStr.empty())
            {
                if (strcasecmp(debugLevelStr.c_str(), "NO_MESSAGES") == 0)
                {
                    vvDebugMsg::setDebugLevel(vvDebugMsg::NO_MESSAGES);
                }
                else if (strcasecmp(debugLevelStr.c_str(), "FEW_MESSAGES") == 0)
                {
                    vvDebugMsg::setDebugLevel(vvDebugMsg::FEW_MESSAGES);
                }
                else if (strcasecmp(debugLevelStr.c_str(), "MOST_MESSAGES") == 0)
                {
                    vvDebugMsg::setDebugLevel(vvDebugMsg::MOST_MESSAGES);
                }
                else if (strcasecmp(debugLevelStr.c_str(), "ALL_MESSAGES") == 0)
                {
                    vvDebugMsg::setDebugLevel(vvDebugMsg::ALL_MESSAGES);
                }
            }
        }
    }


    // ----------------------------------------------------

    backgroundColor = BgDefault;
    bool ignore;
    computeHistogram = covise::coCoviseConfig::isOn("value", "COVER.Plugin.Volume.UseHistogram", true, &ignore);
    maxHistogramVoxels = covise::coCoviseConfig::getLong("value", "COVER.Plugin.Volume.MaxHistogramVoxels", maxHistogramVoxels);
    showTFE = covise::coCoviseConfig::isOn("value", "COVER.Plugin.Volume.ShowTFE", true, &ignore);
    lighting = covise::coCoviseConfig::isOn("value", "COVER.Plugin.Volume.Lighting", false, &ignore);
    preIntegration = covise::coCoviseConfig::isOn("value", "COVER.Plugin.Volume.PreIntegration", false, &ignore);

    tfeBackgroundTexture.resize(TEXTURE_RES_BACKGROUND * TEXTURE_RES_BACKGROUND * 4);

    currentVolume = volumes.end();

    volDesc = NULL;
    reenableCulling = false;

    tfApplyCBData.volume = NULL;
    tfApplyCBData.drawable = NULL;
    tfApplyCBData.tfe = NULL;

    // Initializations:
    plugin = this;
    showClipOutlines = true;
    lastRoll = 0.0;
    roiMode = false;
    unregister = false;
    instantMode = false;
    highQualityOversampling = MAX_QUALITY;
    currentQuality = 1.;
    chosenFPS = INITIAL_FPS;
    allVolumesActive = true;
    radiusScale[0] = INITIAL_CLIP_SPHERE_RADIUS;
    radiusScale[1] = INITIAL_CLIP_SPHERE_RADIUS;
    radiusScale[2] = INITIAL_CLIP_SPHERE_RADIUS;

    interactionA = new coCombinedButtonInteraction(coInteraction::ButtonA, "ROIMode", coInteraction::Menu);
    interactionB = new coCombinedButtonInteraction(coInteraction::ButtonB, "ROIMode", coInteraction::Menu);
    fpsMissed = 0;

    vvDebugMsg::msg(1, "VolumePlugin::initApp()");

    int numHandlers = sizeof(fileHandler) / sizeof(fileHandler[0]);
    for (int i = 0; i < numHandlers; ++i)
    {
        coVRFileManager::instance()->registerFileHandler(&fileHandler[i]);
    }

    VolumeCoim.reset(new coCOIM(this));

    editor = new coDefaultFunctionEditor(applyDefaultTransferFunction, &tfApplyCBData);
    editor->setSaveFunc(saveDefaultTransferFunction);

    tfApplyCBData.tfe = editor;
    volumeMenu = new ui::Menu("Volume", this);

    // create the TabletUI interface
    functionEditorTab = new coTUIFunctionEditorTab("Volume TFE", coVRTui::instance()->mainFolder->getID());
    functionEditorTab->setPos(0, 0);
    functionEditorTab->setEventListener(this);

    const vvTransFunc &func = editor->getTransferFunc();
    for (std::vector<vvTFWidget *>::const_iterator it = func._widgets.begin();
         it != func._widgets.end(); ++it)
    {
        vvTFWidget *widget = *it;
        vvTFColor *colorWidget = dynamic_cast<vvTFColor *>(widget);
        if (colorWidget != NULL)
        {
            coTUIFunctionEditorTab::colorPoint cp;
            cp.r = colorWidget->_col[0];
            cp.g = colorWidget->_col[1];
            cp.b = colorWidget->_col[2];
            cp.x = colorWidget->_pos[0];
            cp.y = colorWidget->_pos[1];
            functionEditorTab->colorPoints.push_back(cp);
        }
        else
        {
            vvTFPyramid *pyramidWidget = dynamic_cast<vvTFPyramid *>(widget);
            if (pyramidWidget != NULL)
            {
                coTUIFunctionEditorTab::alphaPoint ap;
                ap.kind = coTUIFunctionEditorTab::TF_PYRAMID;
                ap.alpha = pyramidWidget->_opacity;
                ap.xPos = pyramidWidget->_pos[0];
                ap.xParam1 = pyramidWidget->_bottom[0];
                ap.xParam2 = pyramidWidget->_top[0];
                ap.yPos = pyramidWidget->_pos[1];
                ap.yParam1 = pyramidWidget->_bottom[1];
                ap.yParam2 = pyramidWidget->_top[1];
                ap.additionalDataElems = 0;
                ap.additionalData = NULL;
                functionEditorTab->alphaPoints.push_back(ap);
            }
        }
    }

    // Create main volume menu:

    tfeItem = new ui::Button(volumeMenu, "ShowTFE");
    tfeItem->setText("Show TFE");
    tfeItem->setCallback([this](bool state){
        if (state)
            editor->show();
        else
            editor->hide();
    });

    boundItem = new ui::Button(volumeMenu, "Boundaries");
    boundItem->setState(false);
    boundItem->setCallback([this](bool state){
        applyToVolumes([this, state](Volume &vol){
            vol.drawable->setBoundaries(state);
            vol.boundaries = state;
        });
    });

    ROIItem = new ui::Button(volumeMenu, "ROI");
    ROIItem->setText("Region of interest (ROI)");
    ROIItem->setState(false);
    ROIItem->setCallback([this](bool state){
        setROIMode(state);
    });

    auto cropItem = new ui::Action(volumeMenu, "CropToRoi");
    cropItem->setText("Crop to ROI");
    cropItem->setCallback([this](){
        cropVolume();
    });

    auto renderGroup = new ui::Group(volumeMenu, "Rendering");

    lightingItem = new ui::Button(renderGroup, "Lighting");
    lightingItem->setState(lighting);
    lightingItem->setCallback([this](bool state){
        applyToVolumes([this, state](Volume &vol){
            vol.drawable->setLighting(state);
            vol.lighting = state;
        });
    });

    interpolItem = new ui::Button(renderGroup, "Interpolation");
    interpolItem->setState(false);
    interpolItem->setCallback([this](bool state){
        applyToVolumes([this, state](Volume &vol){
            vol.drawable->setInterpolation(state);
            vol.interpolation = state;
        });
    });

    auto colorsItem = new ui::Slider(renderGroup, "DiscreteColors");
    colorsItem->setBounds(0., 32.);
    colorsItem->setIntegral(true);
    colorsItem->setText("Discrete colors");
    colorsItem->setPresentation(ui::Slider::AsDial);
    colorsItem->setCallback([this](double value, bool released){
        discreteColors = (int)value;
        editor->updateColorBar();
        editor->setDiscreteColors(discreteColors);
    });

    blendModeItem = new ui::SelectionList(renderGroup, "BlendMode");
    blendModeItem->setText("Blend mode");
    blendModeItem->append("Alpha");
    blendModeItem->append("Alpha (dark)");
    blendModeItem->append("Alpha (light)");
    blendModeItem->append("Maximum intensity");
    blendModeItem->append("Minimum intensity");
    blendModeItem->setCallback([this](int mode){
        virvo::VolumeDrawable::BlendMode blend = virvo::VolumeDrawable::AlphaBlend;
        if (mode == 4)
            blend = virvo::VolumeDrawable::MinimumIntensity;
        else if (mode == 3)
            blend = virvo::VolumeDrawable::MaximumIntensity;

        Vec4 bg(0., 0., 0., 1.);
        switch (blend)
        {
        case virvo::VolumeDrawable::AlphaBlend:
            if (mode == 0)
            {
                backgroundColor = BgDefault;
                bg[0] = covise::coCoviseConfig::getFloat("r", "COVER.Background", 0.f);
                bg[1] = covise::coCoviseConfig::getFloat("g", "COVER.Background", 0.f);
                bg[2] = covise::coCoviseConfig::getFloat("b", "COVER.Background", 0.f);
            }
            else if (mode == 1)
            {
                backgroundColor = BgDark;
                bg[0] = bg[1] = bg[2] = 0.30f;
            }
            else if (mode == 2)
            {
                backgroundColor = BgLight;
                bg[0] = bg[1] = bg[2] = 0.75f;
            }
            break;
        case virvo::VolumeDrawable::MinimumIntensity:
            bg[0] = bg[1] = bg[2] = 1.;
            break;
        case virvo::VolumeDrawable::MaximumIntensity:
            bg[0] = bg[1] = bg[2] = 0.;
            break;
        }
        VRViewer::instance()->setClearColor(bg);

        applyToVolumes([this, blend, mode](Volume &vol){
            vol.drawable->setBlendMode(blend);
            vol.blendMode = blend;
        });
    });

    auto hqItem = new ui::Slider(renderGroup, "Oversampling");
    hqItem->setBounds(1., MAX_QUALITY*2.);
    hqItem->setValue(highQualityOversampling);
    hqItem->setCallback([this](double value, bool released){
        highQualityOversampling = value;
    });

    auto fpsItem = new ui::Slider(renderGroup, "FrameRate");
    fpsItem->setText("Frame rate");
    fpsItem->setBounds(5.0, 60.0);
    fpsItem->setValue(chosenFPS);
    fpsItem->setCallback([this](double value, bool released){
        chosenFPS = value;
    });

    preintItem = new ui::Button(renderGroup, "PreIntegration");
    preintItem->setText("Pre-integration");
    preintItem->setState(preIntegration);
    preintItem->setCallback([this](bool state){
        applyToVolumes([this, state](Volume &vol){
            vol.drawable->setPreintegration(state);
            vol.preIntegration = state;
        });
    });

    auto volumesGroup = new ui::Group(volumeMenu, "Volumes");

    filesMenu = new ui::Menu(volumesGroup, "Files");

    auto saveItem = new ui::Action(filesMenu, "SaveVolume");
    saveItem->setText("Save volume");
    saveItem->setCallback([this](){
        saveVolume();
    });

    auto unloadItem = new ui::Action(filesMenu, "Unload");
    unloadItem->setText("Unload current file");
    unloadItem->setCallback([this](){
        if (currentVolume != volumes.end())
        {
            std::string filename = currentVolume->second.filename;
            if (!filename.empty())
                updateVolume(filename, NULL);
        }
    });

    filesGroup = new ui::Group(filesMenu, "Files");

    auto allVolumesActiveItem = new ui::Button(volumesGroup, "AllVolumesActive");
    allVolumesActiveItem->setState(allVolumesActive);
    allVolumesActiveItem->setText("All volumes active");
    allVolumesActiveItem->setCallback([this](bool state){
        allVolumesActive = state;
    });

    auto sideBySideItem = new ui::Button(volumesGroup, "SideBySide");
    sideBySideItem->setText("Side by side");
    sideBySideItem->setState(false);
    sideBySideItem->setCallback([this](bool state){
        if (state)
        {
            osg::Vec3 maxSize(0.f, 0.f, 0.f);
            for (VolumeMap::iterator it = volumes.begin(); it != volumes.end(); ++it)
            {
                osg::Vec3 sz = it->second.max - it->second.min;
                for (int i = 0; i < 3; ++i)
                    if (maxSize[i] < sz[i])
                        maxSize[i] = sz[i];
            }
            int i = 0;
            for (VolumeMap::iterator it = volumes.begin(); it != volumes.end(); ++it)
            {
                osg::Vec3 translate(i * maxSize[0], 0.f, 0.f);
                translate -= it->second.min;
                osg::Matrix mat = it->second.transform->getMatrix();
                mat.setTrans(translate);
                it->second.transform->setMatrix(mat);
                ++i;
            }
        }
        else
        {
            osg::Matrix ident;
            ident.makeIdentity();
            for (VolumeMap::iterator it = volumes.begin(); it != volumes.end(); ++it)
            {
                osg::Matrix mat = it->second.transform->getMatrix();
                mat.setTrans(osg::Vec3(0,0,0));
                it->second.transform->setMatrix(mat);
            }
        }
    });

    auto cycleVolumeItem = new ui::Action(volumesGroup, "CycleVolume");
    cycleVolumeItem->setText("Cycle active volume");
    cycleVolumeItem->setCallback([this](){
        VolumeMap::iterator cur = currentVolume;
        if (cur != volumes.end())
        {
            ++cur;
        }
        if (volumes.end() == cur)
        {
            cur = volumes.begin();
        }

        makeVolumeCurrent(cur);
    });

    currentVolumeItem = new ui::Label(volumesGroup, "CurrentVolume");
    currentVolumeItem->setText("[]");

    // Create clipping menu
    clipMenu = new ui::Menu(volumeMenu, "Clipping");

    auto clipSingleSlice = new ui::Button(clipMenu, "SingleSliceClipping");
    clipSingleSlice->setText("Single slice clipping");
    clipSingleSlice->setState(singleSliceClipping);
    clipSingleSlice->setCallback([this](bool state){
        singleSliceClipping = state;
        applyToVolumes([this, state](Volume &vol){
            vol.drawable->setSingleSliceClipping(state);
            vol.drawable->setOpaqueClipping(opaqueClipping && state);
            if (state)
                vol.drawable->setLighting(false);
            else
                vol.drawable->setLighting(vol.lighting);
        });
    });

    auto clipOpaqueItem = new ui::Button(clipMenu, "OpaqueClipping");
    clipOpaqueItem->setText("Opaque clipping");
    clipOpaqueItem->setState(opaqueClipping);
    clipOpaqueItem->setCallback([this](bool state){
        opaqueClipping = state;
        applyToVolumes([this, state](Volume &vol){
            if (singleSliceClipping) {
                vol.drawable->setOpaqueClipping(state);
            }
        });
    });

    auto clipOutlinesItem = new ui::Button(clipMenu, "ClipOutlines");
    clipOutlinesItem->setText("Show box intersections");
    clipOutlinesItem->setState(showClipOutlines);
    clipOutlinesItem->setCallback([this](bool state){
        showClipOutlines = state;
    });

    auto followCoverClippingItem = new ui::Button(clipMenu, "ClipCover");
    followCoverClippingItem->setText("Track clip planes");
    followCoverClippingItem->setState(followCoverClipping);
    followCoverClippingItem->setCallback([this](bool state){
        followCoverClipping = state;
    });

    auto ignoreCoverClippingItem = new ui::Button(clipMenu, "ClipCoverIgnore");
    ignoreCoverClippingItem->setText("Ignore clip planes");
    ignoreCoverClippingItem->setState(ignoreCoverClipping);
    ignoreCoverClippingItem->setCallback([this](bool state){
        ignoreCoverClipping = state;
    });

    if (enableSphereClipping)
    {
        // Initialize clip spheres
        for (int i = 0; i < NumClipSpheres; ++i)
        {
            auto group = new ui::Group(clipMenu, "Sphere"+std::to_string(i));
            group->setText("Clip sphere "+std::to_string(i));

            auto clipSphereActiveItem = new ui::Button(group, "SphereActive"+std::to_string(i));
            clipSphereActiveItem->setText("Sphere "+std::to_string(i)+" active");
            clipSphereActiveItem->setState(false);
            clipSphereActiveItem->setCallback([this, i](bool state){
                clipSpheres.at(i)->setActive(state);
            });

            auto clipSphereInteractorItem = new ui::Button(group, "SphereInteractor"+std::to_string(i));
            clipSphereInteractorItem->setText("Sphere "+std::to_string(i)+" interactor");
            clipSphereInteractorItem->setState(false);
            clipSphereInteractorItem->setCallback([this, i](bool state){
                clipSpheres.at(i)->setInteractorActive(state);
            });

            auto clipSphereRadiusItem = new ui::Slider(group, "SphereRadius"+std::to_string(i));
            clipSphereRadiusItem->setText("Sphere "+std::to_string(i)+" radius");
            clipSphereRadiusItem->setBounds(0.1, 1.0);
            clipSphereRadiusItem->setValue(radiusScale[i]);
            clipSphereRadiusItem->setCallback([this, i](double value, bool released){
                radiusScale[i] = value;
            });

            clipSpheres.push_back(boost::make_shared<coClipSphere>());
        }
    }

    // Read volume file entries from covise.config:
    for (const auto &entry : covise::coCoviseConfig::getScopeEntries("COVER.Plugin.Volume.Files"))
        fileList.push_back(new FileEntry(entry.second.c_str(), entry.first.c_str()));
    // Load volume data:
    std::string line = covise::coCoviseConfig::getEntry("COVER.Plugin.Volume.VolumeFile");
    if (!line.empty())
    {
        loadFile(line.c_str(), NULL);
    }

    return true;
}

bool VolumePlugin::update()
{
    ++updateCount;
    return false;
}

/// Destructor
VolumePlugin::~VolumePlugin()
{
    vvDebugMsg::msg(1, "VolumePlugin::~VolumePlugin()");

    int numHandlers = sizeof(fileHandler) / sizeof(fileHandler[0]);
    for (int i = 0; i < numHandlers; ++i)
    {
        coVRFileManager::instance()->unregisterFileHandler(&fileHandler[i]);
    }

#ifdef VERBOSE
    cerr << "~VolumePlugin: delete renderer\n";
#endif

    tfApplyCBData.volume = NULL;
    tfApplyCBData.drawable = NULL;
    tfApplyCBData.tfe = NULL;
    delete editor;
    for (VolumeMap::iterator it = volumes.begin();
         it != volumes.end();
         it++)
    {
        it->second.removeFromScene();
    }
    volumes.clear();

    for (list<FileEntry *>::iterator i = fileList.begin(); i != fileList.end(); ++i)
        delete *i;
    fileList.clear();

    delete interactionA;
    delete interactionB;

    for (VolumeMap::iterator it = volumes.begin();
         it != volumes.end();
         it++)
    {
        Node::ParentList parents = it->second.transform->getParents();
        for (Node::ParentList::iterator parent = parents.begin(); parent != parents.end(); ++parent)
            (*parent)->removeChild(it->second.transform.get());
    }

    delete functionEditorTab;
}

/*
*
* Handle TabletUI Events from Pushbuttons etc.
*/
void VolumePlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == functionEditorTab)
    {
        cout << "VolumePlugin::tabletPressEvent: FunctionEditorTab " << std::endl;

        TFApplyCBData *data = &tfApplyCBData;
        if (data && data->drawable)
        {
            vvVolDesc *vd = data->drawable->getVolumeDescription();

            // if the dimensionality is different (do not match that of data)
            // adjust that.
            if (functionEditorTab->getDimension() > 1) // 1 is always ok
            {
                // if we want 2D TF but we have only one channel, use the
                // first-order derivative as the second channel
                if (vd->getChan() < 2)
                {
                    vd->addGradient(0, vvVolDesc::GRADIENT_MAGNITUDE);
                    assert(vd->getChan() >= 2);
                    functionEditorTab->setDimension(vd->getChan());

                    // refresh the histogram
                    int buckets[2];
                    buckets[0] = coTUIFunctionEditorTab::histogramBuckets;
                    buckets[1] = coTUIFunctionEditorTab::histogramBuckets;

                    delete[] functionEditorTab -> histogramData;
                    functionEditorTab->histogramData = NULL;
                    functionEditorTab->histogramData = new int[buckets[0] * buckets[1]];
                    vd->makeHistogram(0, 0, 2, buckets, functionEditorTab->histogramData,0 , 1);

                    functionEditorTab->sendHistogramData();
                }
            }

            coDefaultFunctionEditor *tfe = data->tfe;
            if (tfe)
            {
                //update tfe
                vvTransFunc func;

                // first color points
                for (int i = 0; i < functionEditorTab->colorPoints.size(); ++i)
                {
                    if (functionEditorTab->getDimension() == 1)
                    {
                        func._widgets.push_back(new vvTFColor(vvColor(functionEditorTab->colorPoints[i].r,
                                                                      functionEditorTab->colorPoints[i].g,
                                                                      functionEditorTab->colorPoints[i].b),
                                                              functionEditorTab->colorPoints[i].x));
                    }
                    else
                    {
                        func._widgets.push_back(new vvTFColor(vvColor(functionEditorTab->colorPoints[i].r,
                                                                      functionEditorTab->colorPoints[i].g,
                                                                      functionEditorTab->colorPoints[i].b),
                                                              functionEditorTab->colorPoints[i].x,
                                                              functionEditorTab->colorPoints[i].y));
                    }
                }

                // then, the alpha widgets
                for (int i = 0; i < functionEditorTab->alphaPoints.size(); ++i)
                {
                    //TF_PYRAMID == 1, TF_CUSTOM == 4
                    //
                    switch (functionEditorTab->alphaPoints[i].kind)
                    {
                    case 1: //TUITFEWidget::TF_TRIANGLE:
                    {
                        if (functionEditorTab->getDimension() == 1)
                        {
                            func._widgets.push_back(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f),
                                                                    false,
                                                                    functionEditorTab->alphaPoints[i].alpha,
                                                                    functionEditorTab->alphaPoints[i].xPos,
                                                                    functionEditorTab->alphaPoints[i].xParam1,
                                                                    functionEditorTab->alphaPoints[i].xParam2));
                        }
                        else
                        {
                            vvColor color(1.0f, 1.0f, 1.0f);

                            if (functionEditorTab->alphaPoints[i].ownColor)
                            {
                                color[0] = functionEditorTab->alphaPoints[i].r;
                                color[1] = functionEditorTab->alphaPoints[i].g;
                                color[2] = functionEditorTab->alphaPoints[i].b;
                            }
                            func._widgets.push_back(new vvTFPyramid(color,
                                                                    functionEditorTab->alphaPoints[i].ownColor,
                                                                    functionEditorTab->alphaPoints[i].alpha,
                                                                    functionEditorTab->alphaPoints[i].xPos,
                                                                    functionEditorTab->alphaPoints[i].xParam1,
                                                                    functionEditorTab->alphaPoints[i].xParam2,
                                                                    functionEditorTab->alphaPoints[i].yPos,
                                                                    functionEditorTab->alphaPoints[i].yParam1,
                                                                    functionEditorTab->alphaPoints[i].yParam2));
                        }
                    }
                    break;

                    case 2: //TUITFEWidget::TF_BELL:
                    {
                        vvColor color(1.0f, 1.0f, 1.0f);

                        if (functionEditorTab->alphaPoints[i].ownColor)
                        {
                            color[0] = functionEditorTab->alphaPoints[i].r;
                            color[1] = functionEditorTab->alphaPoints[i].g;
                            color[2] = functionEditorTab->alphaPoints[i].b;
                        }
                        func._widgets.push_back(new vvTFBell(color,
                                                             functionEditorTab->alphaPoints[i].ownColor,
                                                             functionEditorTab->alphaPoints[i].alpha,
                                                             functionEditorTab->alphaPoints[i].xPos,
                                                             functionEditorTab->alphaPoints[i].xParam1,
                                                             functionEditorTab->alphaPoints[i].yPos,
                                                             functionEditorTab->alphaPoints[i].yParam1));
                    }
                    break;

                    case 4: //TUITFEWidget::TF_CUSTOM:
                    {
                        int numOpacityPoints = functionEditorTab->alphaPoints[i].additionalDataElems;

                        if (numOpacityPoints > 0)
                        {
                            float posBegin = functionEditorTab->alphaPoints[i].additionalData[0];
                            float posEnd = functionEditorTab->alphaPoints[i].additionalData[(numOpacityPoints - 1) * 2];

                            //float begin=0.0f, end=0.0f;
                            vvTFCustom *cuw = new vvTFCustom(0.5f, 1.0f); //position, size. Cover all the area.
                            for (int j = 0; j < numOpacityPoints * 2; j += 2)
                            {
                                cuw->_points.push_back(new vvTFPoint(functionEditorTab->alphaPoints[i].additionalData[j + 1],
                                                                     functionEditorTab->alphaPoints[i].additionalData[j]));
                            }

                            // Adjust widget size:
                            cuw->_size[0] = posEnd - posBegin;
                            cuw->_pos[0] = (posBegin + posEnd) / 2.0f;

                            // Adjust point positions:
                            list<vvTFPoint *>::iterator iter;
                            for (iter = cuw->_points.begin(); iter != cuw->_points.end(); iter++)
                            {
                                (*iter)->_pos[0] -= cuw->_pos[0];
                            }

                            func._widgets.push_back(cuw);
                        }
                    }
                    break;

                    case 5: //TUITFEWidget::TF_CUSTOM_2D:
                        //TODO!!
                        break;

                    case 6: //TUITFEWidget::TF_MAP:
                    {
                        vvTFCustomMap *w = new vvTFCustomMap(functionEditorTab->alphaPoints[i].xPos,
                                                             functionEditorTab->alphaPoints[i].xParam1,
                                                             functionEditorTab->alphaPoints[i].yPos,
                                                             functionEditorTab->alphaPoints[i].yParam1);

                        w->setOwnColor(functionEditorTab->alphaPoints[i].ownColor);
                        if (w->hasOwnColor())
                        {
                            w->_col[0] = functionEditorTab->alphaPoints[i].r;
                            w->_col[1] = functionEditorTab->alphaPoints[i].g;
                            w->_col[2] = functionEditorTab->alphaPoints[i].b;
                        }
                        // store map info
                        //w->_dim = functionEditorTab->alphaPoints[i].additionalDataElems;
                        int dim = w->_dim[0] * w->_dim[1] * w->_dim[2];
                        memcpy(w->_map, functionEditorTab->alphaPoints[i].additionalData, dim * sizeof(float));

                        func._widgets.push_back(w);
                    }
                    break;
                    }
                } // end for alpha widgets
                tfe->setTransferFunc(func, 0);

                for (VolumeMap::iterator it = volumes.begin();
                     it != volumes.end();
                     it++)
                {
                    if (allVolumesActive || data->drawable == it->second.drawable)
                    {
                        if (it->second.multiDimTF == data->volume->multiDimTF)
                        {
                            it->second.curChannel = tfe->getActiveChannel();
                            it->second.tf[tfe->getActiveChannel()] = func;
                            it->second.drawable->setTransferFunctions(it->second.tf);
                            if (it->second.mapTF)
                                mapTFToMinMax(it, vd);

                            it->second.useChannelWeights = tfe->getUseChannelWeights();
                            it->second.drawable->setUseChannelWeights(it->second.useChannelWeights);
                            it->second.channelWeights = tfe->getChannelWeights();
                        }
                    }
                }
            }
        }
    }
}
/*
* 
* Handle TabletUI Events from ToggleButtons, Sliders, etc.
*/
void VolumePlugin::tabletEvent(coTUIElement * /*tUIItem*/)
{
}

bool VolumePlugin::pointerInROI(bool *isMouse)
{
    vvDebugMsg::msg(1, "VolumePlugin::pointerInROI()");
    *isMouse = false;

    if (currentVolume == volumes.end())
        return false;

    const osg::Matrix &mat = currentVolume->second.transform->getMatrix();
    osg::Matrix inv = osg::Matrix::inverse(mat);

    Vec3 pointerPos1Wld;
    Vec3 pointerPos2Wld;
    Vec3 pointerPos1Obj;
    Vec3 pointerPos2Obj;

    pointerPos1Wld = cover->getPointerMat().getTrans();
    pointerPos1Obj = pointerPos1Wld * cover->getInvBaseMat() * inv;
    pointerPos1Wld.set(0.0, 1000.0, 0.0);
    pointerPos2Wld = pointerPos1Wld * cover->getPointerMat();
    pointerPos2Obj = pointerPos2Wld * cover->getInvBaseMat() * inv;

    Vec3 H, N;
    H = currentVolume->second.roiPosObj - pointerPos1Obj;
    N = pointerPos2Obj - pointerPos1Obj;
    H.normalize();
    N.normalize();
    float h = (currentVolume->second.roiPosObj - pointerPos1Obj) * (currentVolume->second.roiPosObj - pointerPos1Obj);
    float sprod = H * N;
    sprod *= sprod;
    float dist = sqrt(fabs(h * sprod - h));
    //cerr << "Dist:: " << dist << "    " << myUserData->cellSize/2.0*myUserData->maxSize << endl;
    bool result = (dist < roiCellSize / 2.0 * roiMaxSize);
    if (result || !coVRNavigationManager::instance()->mouseNav())
        return result;

    *isMouse = true;

    pointerPos1Wld = cover->getMouseMat().getTrans();
    pointerPos1Obj = pointerPos1Wld * cover->getInvBaseMat() * inv;
    pointerPos1Wld.set(0.0, 1000.0, 0.0);
    pointerPos2Wld = pointerPos1Wld * cover->getMouseMat();
    pointerPos2Obj = pointerPos2Wld * cover->getInvBaseMat() * inv;

    H = currentVolume->second.roiPosObj - pointerPos1Obj;
    N = pointerPos2Obj - pointerPos1Obj;
    H.normalize();
    N.normalize();
    h = (currentVolume->second.roiPosObj - pointerPos1Obj) * (currentVolume->second.roiPosObj - pointerPos1Obj);
    sprod = H * N;
    sprod *= sprod;
    dist = sqrt(fabs(h * sprod - h));
    //cerr << "Dist:: " << dist << "    " << myUserData->cellSize/2.0*myUserData->maxSize << endl;
    return (dist < roiCellSize / 2.0 * roiMaxSize);
}

bool VolumePlugin::roiVisible()
{
    vvDebugMsg::msg(1, "VolumePlugin::roiVisible()");

    if (currentVolume == volumes.end())
        return false;

    Vec3 &roiPosObj = currentVolume->second.roiPosObj;
    Vec3 &min = currentVolume->second.min;
    Vec3 &max = currentVolume->second.max;

    if ((roiPosObj[0] + (roiCellSize * roiMaxSize) / 2.0) > min[0] && (roiPosObj[0] - (roiCellSize * roiMaxSize) / 2.0) < max[0] && (roiPosObj[1] + (roiCellSize * roiMaxSize) / 2.0) > min[1] && (roiPosObj[1] - (roiCellSize * roiMaxSize) / 2.0) < max[1] && (roiPosObj[2] + (roiCellSize * roiMaxSize) / 2.0) > min[2] && (roiPosObj[2] - (roiCellSize * roiMaxSize) / 2.0) < max[2])
        return true;
    else
        return false;
}

void VolumePlugin::applyDefaultTransferFunction(void *userData)
{
    VolumePlugin::plugin->applyAllTransferFunctions(userData);
}

void VolumePlugin::applyAllTransferFunctions(void *userData)
{
    vvDebugMsg::msg(2, "VolumePlugin::applyDefaultTransferFunction()");

    TFApplyCBData *data = (TFApplyCBData *)userData;

    if (data && data->drawable && data->tfe)
    {
        coDefaultFunctionEditor *tfe = data->tfe;
        for (VolumeMap::iterator it = volumes.begin();
             it != volumes.end();
             it++)
        {
            if (it->second.drawable == data->drawable || allVolumesActive)
            {
                vvVolDesc *vd = it->second.drawable.get()->getVolumeDescription();

                it->second.tf = tfe->getTransferFuncs();
                it->second.drawable->setTransferFunctions(it->second.tf);
                if (it->second.mapTF)
                    mapTFToMinMax(it, vd);
                  
                it->second.useChannelWeights = tfe->getUseChannelWeights();
                it->second.channelWeights = tfe->getChannelWeights();
                it->second.drawable->setChannelWeights(it->second.channelWeights);
                it->second.drawable->setUseChannelWeights(it->second.useChannelWeights);
            }
        }
    }
}

void VolumePlugin::saveDefaultTransferFunction(void *userData)
{
    vvDebugMsg::msg(2, "VolumePlugin::saveDefaultTransferFunction()");

    if (coVRMSController::instance()->isSlave())
        return;

    TFApplyCBData *data = (TFApplyCBData *)userData;

    if (data && data->drawable)
    {
        vvVolDesc *vd = data->drawable->getVolumeDescription();
        if (!vd)
            return;

        vd->setFilename("cover-transferfunction.xvf");
        vvFileIO fio;

        vvFileIO::ErrorType err = fio.saveVolumeData(vd, false, vvFileIO::TRANSFER);

        int filenumber = 0;
        while (err == vvFileIO::FILE_EXISTS)
        {
            ++filenumber;

            int digits = 1;
            int tmpFilenumber = filenumber / 10;
            while (tmpFilenumber > 0)
            {
                ++digits;
                tmpFilenumber /= 10;
            }

            tmpFilenumber = filenumber;
            char *filenumberStr = new char[digits + 1];
            int ite = digits - 1;
            while (tmpFilenumber > 0)
            {
                // Convert to ascii.
                filenumberStr[ite] = (tmpFilenumber % 10) + 48;
                tmpFilenumber /= 10;
                --ite;
            }
            filenumberStr[digits] = '\0';
            std::stringstream str;
            str << "cover->transferfunction_(";
            str << filenumberStr;
            str << ".xvf";
            delete[] filenumberStr;
            vd->setFilename(str.str().c_str());
            err = fio.saveVolumeData(vd, false, vvFileIO::TRANSFER);
        }

        if (err == vvFileIO::OK)
        {
            cerr << "transfer function saved to " << vd->getFilename() << endl;
        }
        else
        {
            cerr << "failed to save transfer function to " << vd->getFilename() << endl;
        }
    }
}

int VolumePlugin::loadFile(const char *fName, osg::Group *parent, const vvVolDesc *params)
{
    vvDebugMsg::msg(1, "VolumePlugin::loadFile()");

    const char *fn = coVRFileManager::instance()->getName(fName);
    if (!fn)
    {
        cerr << "Invalid file name: " << (fName ? fName : "(null)") << endl;
        return 0;
    }

    std::string fileName(fn);
#ifdef VERBOSE
    cerr << "Loading volume file: " << fileName << endl;
#endif

    vvVolDesc *vd = nullptr;
    if (params)
    {
        vd = new vvVolDesc(params);
        vd->setFilename(fileName.c_str());
    }
    else
    {
        vd =  new vvVolDesc(fileName.c_str());
    }

    vvFileIO fio;
    if (fio.loadVolumeData(vd) != vvFileIO::OK)
    {
        cerr << "Cannot load volume file: " << fileName << endl;
        delete vd;
        vd = NULL;
        return 0;
    }

    vd->printInfoLine("Loaded");

    // a volumefile will be loaded now , so show the TFE
    if (showTFE)
    {
        editor->show();
        tfeItem->setState(true);
    }

    updateVolume(fileName, vd, false, fileName, nullptr, parent);

    return 1;
}

void VolumePlugin::sendROIMessage(osg::Vec3 roiPos, float size)
{
    vvDebugMsg::msg(1, "VolumePlugin::sendROIMessage()");

    ROIData pd;
    pd.x = roiPos[0];
    pd.y = roiPos[1];
    pd.z = roiPos[2];
    pd.size = size;
    cover->sendMessage(this,
                       coVRPluginSupport::TO_SAME, PluginMessageTypes::VolumeROIMsg,
                       sizeof(ROIData), &pd);
}

void VolumePlugin::message(int toWhom, int type, int len, const void *buf)
{
    vvDebugMsg::msg(1, "VolumePlugin::VRMessage()");

    if (type == PluginMessageTypes::VolumeLoadFile)
    {
        loadFile((const char *)buf, NULL);
    }
    else if (type == PluginMessageTypes::VolumeROIMsg)
    {
        ROIData *pd;
        pd = (ROIData *)buf;
        if (currentVolume != volumes.end())
        {
            currentVolume->second.roiPosObj.set(pd->x, pd->y, pd->z);

            virvo::VolumeDrawable *drawable = getCurrentDrawable();
            if (drawable)
            {
                drawable->setROIPosition(currentVolume->second.roiPosObj);
                drawable->setROISize(pd->size);
            }

            if ((coVRCollaboration::instance()->getCouplingMode() == coVRCollaboration::TightCoupling))
            {
                if (drawable && drawable->getROISize() > 0.)
                {
                    roiMode = true;
                }
                else
                {
                    roiMode = false;
                }
                ROIItem->setState(roiMode);
            }
        }
    }
    else if (type == PluginMessageTypes::VolumeClipMsg)
    {
    }
    else
    {
        VolumeCoim->receiveMessage(type, len, buf);
    }
}

void VolumePlugin::addObject(const RenderObject *container, osg::Group *group, const RenderObject *geometry, const RenderObject *, const RenderObject *colorObj, const RenderObject *)
{
    vvDebugMsg::msg(1, "VolumePlugin::VRAddObject()");
    int shader = -1;

    if (container->getAttribute("VOLUME_SHADER"))
    {
        std::string s = container->getAttribute("VOLUME_SHADER");
        shader = atoi(s.c_str());
    }

    // Check if valid volume data was added:
    if (!geometry || !geometry->isUniformGrid())
    {
        return;
    }

#ifdef VERBOSE
    fprintf(stderr, "add volume: geo=%s, color=%s\n", geometry->getName(), colorObj->getName());
#endif
    int sizeX, sizeY, sizeZ;
    geometry->getSize(sizeX, sizeY, sizeZ);

    if (sizeX<=1 || sizeY<=1 || sizeZ<=1)
    {
        // ignore 2-dimensional grids: already handled by COVISE plugin
        return;
    }

    float minX, maxX, minY, maxY, minZ, maxZ;
    geometry->getMinMax(minX, maxX, minY, maxY, minZ, maxZ);
    osg::Matrix tfMat = geometry->transform;

    bool showEditor = showTFE;
    if (colorObj)
    {
        const uchar *byteData = colorObj->getByte(Field::Byte);
        const int *packedColor = colorObj->getInt(Field::RGBA);
        const float *red = colorObj->getFloat(Field::Red), *green = colorObj->getFloat(Field::Green), *blue = colorObj->getFloat(Field::Blue);

        std::vector<const uchar*> byteChannels(Field::NumChannels);
        std::vector<const float*> floatChannels(Field::NumChannels);

        bool have_byte_chans = false;
        bool have_float_chans = false;

        int noChan = 0;

        float min[Field::NumChannels], max[Field::NumChannels], irange[Field::NumChannels];
        for (int c = Field::Channel0; c < Field::NumChannels; ++c)
        {
            min[c] = colorObj->getMin(c);
            max[c] = colorObj->getMax(c);
            if (max[c] - min[c] <= 0.f)
                irange[c] = 1.f;
            else
                irange[c] = 1.f/(max[c] - min[c]);

            byteChannels[c] = colorObj->getByte((Field::Id)c);
            if (byteChannels[c])
                have_byte_chans = true;

            floatChannels[c] = colorObj->getFloat((Field::Id)c);
            if (floatChannels[c])
                have_float_chans = true;

            if (byteChannels[c] || floatChannels[c])
                ++noChan;
        }

        uchar *data = NULL;
        size_t noVox = sizeX * sizeY * sizeZ;
        if (packedColor)
        {
            showEditor = false;
            noChan = 4;
            data = new uchar[noVox * 4];
            uchar *p = data;
            for (size_t i=0; i<noVox; ++i)
            {
                *p++ = (packedColor[i] >> 24) & 0xff;
                *p++ = (packedColor[i] >> 16) & 0xff;
                *p++ = (packedColor[i] >> 8) & 0xff;
                *p++ = (packedColor[i] >> 0) & 0xff;
            }
        }
        else if (have_byte_chans || have_float_chans)
        {
            data = new uchar[noVox * noChan];
            uchar *p = data;
            if (!have_byte_chans)
            {
                const float *chan[Field::NumChannels];
                float range[Field::NumChannels];
                int ch=0;
                for (int c = Field::Channel0; c < Field::NumChannels; ++c)
                {
                    if (floatChannels[c])
                    {
                        range[ch] = irange[c]*255.99f;
                        chan[ch] = floatChannels[c];
                        ++ch;
                    }
                }
                assert(ch == noChan);
                for (size_t i=0; i<noVox; ++i)
                {
                    for (int c = 0; c<noChan; ++c)
                    {
                        *p++ = (uchar)((chan[c][i]-min[c])*range[c]);
                    }
                }
            }
            else if (!have_float_chans)
            {
                const uchar *chan[Field::NumChannels];
                int ch=0;
                for (int c = Field::Channel0; c < Field::NumChannels; ++c)
                {
                    if (byteChannels[c])
                    {
                        chan[ch] = byteChannels[c];
                        ++ch;
                    }
                }
                assert(ch == noChan);
                for (size_t i=0; i<noVox; ++i)
                {
                    for (int c = 0; c<noChan; ++c)
                    {
                        *p++ = chan[c][i];
                    }
                }
            }
            else
            {
                for (size_t i=0; i<noVox; ++i)
                {
                    for (int c = Field::Channel0; c < Field::NumChannels; ++c)
                    {
                        if (byteChannels[c])
                        {
                            *p++ = byteChannels[c][i];
                        }
                        else if (floatChannels[c])
                        {
                            *p++ = (uchar)((floatChannels[c][i]-min[c])*irange[c] * 255.99);
                        }
                    }
                }
            }
        }
        else if (red && green && blue)
        {
            noChan = 3;
            data = new uchar[noVox * 3];
            uchar *p = data;
            for (size_t i=0; i<noVox; ++i)
            {
                *p++ = (uchar)((red[i]-min[0])*irange[0] * 255.99);
                *p++ = (uchar)((green[i]-min[1])*irange[1] * 255.99);
                *p++ = (uchar)((blue[i]-min[2])*irange[2] * 255.99);
            }
        }
        else if (red && green)
        {
            noChan = 2;
            data = new uchar[noVox * 2];
            uchar *p = data;
            for (size_t i=0; i<noVox; ++i)
            {
                *p++ = (uchar)((red[i]-min[0])*irange[0] * 255.99);
                *p++ = (uchar)((green[i]-min[1])*irange[1] * 255.99);
            }
        }
        else if (red)
        {
            noChan = 1;
            data = new uchar[noVox];
            uchar *p = data;
            for (size_t i=0; i<noVox; ++i)
            {
                *p++ = (uchar)((red[i]-min[0])*irange[0] * 255.99);
            }
        }
        else if (byteData)
        {
            noChan = 1;
            data = new uchar[noVox];
            memcpy(data, byteData, noVox);
        }
        else
        {
            cerr << "no data received" << endl;
            return;
        }

        // add to timestep series if necessary
        if (container && container->getName() && volumes.find(container->getName()) != volumes.end())
        {
            volDesc->addFrame(data, vvVolDesc::ARRAY_DELETE);
            volDesc->frames = volDesc->getStoredFrames();
#ifdef VERBOSE
            fprintf(stderr, "added timestep to %s: %d steps\n", container->getName(), (int)volDesc->frames);
#endif
        }
        else
        {
            volDesc = new vvVolDesc("COVISEXX",
                                    sizeZ, sizeY, sizeX, 1, 1, noChan, &data, vvVolDesc::ARRAY_DELETE);
            volDesc->pos = vvVector3(minZ+maxZ, -minY-maxY, -minX-maxX) * .5f;
            volDesc->setDist((maxZ - minZ) / sizeZ,
                             (maxY - minY) / sizeY,
                             (maxX - minX) / sizeX);
        }

        if (packedColor)
        {
            volDesc->tf[0].setDefaultColors(3, 0., 1.);
            volDesc->tf[0].setDefaultAlpha(0, 0., 1.);
        }

        for (size_t c = 0; c < volDesc->getChan(); ++c)
        {
            volDesc->range(c)[0] = colorObj->getMin(c);
            volDesc->range(c)[1] = colorObj->getMax(c);

            if (volDesc->range(c)[1] == 0 && volDesc->range(c)[0] == 0)
                volDesc->findMinMax(c, volDesc->range(c)[0], volDesc->range(c)[1]);
        }
        
        int groupID = -1;
        std::string objName;
        if (container)
            objName= container->getName();
        else if (geometry)
            objName= geometry->getName();
        
        groupID = std::isdigit(objName[0]) ? objName[0] : -1;

        if (groupID >= 0){
            if(minData.find(groupID) == minData.end()){
                std::vector<float> newVec;
                minData[groupID] = newVec;
                maxData[groupID] = newVec;
            }
            if (minData[groupID].size() < volDesc->getChan()){
                minData[groupID].clear();
                maxData[groupID].clear();
                for (size_t c = 0; c < volDesc->getChan(); ++c)
                {
                    minData[groupID].push_back(colorObj->getMin(c));
                    maxData[groupID].push_back(colorObj->getMax(c));
                }
            }else{
                bool updateAll = false;
                for (size_t c = 0; c < volDesc->getChan(); ++c)
                {
                    updateAll = false;
                    if(minData[groupID][c]>colorObj->getMin(c)){
                        minData[groupID][c] = colorObj->getMin(c);
                        updateAll = true;
                    }
                    if(maxData[groupID][c]<colorObj->getMax(c)){
                        maxData[groupID][c] = colorObj->getMax(c);
                        updateAll = true;
                    }
                    if (updateAll) {
                        for (VolumeMap::iterator it = volumes.begin(); it != volumes.end(); it++) {
                            std::string itName = it->first;
                            if (std::strncmp(objName.c_str(), itName.c_str(),2)==0) {
                                vvVolDesc *vd = it->second.drawable->getVolumeDescription();
                                updateVolume(it->first,vd,true);
                            }
                        }
                    }
                }
            }
        }
        
        if (container->getName()){
            updateVolume(container->getName(), volDesc, true, "", container, group);
            updateTransform(container->getName(), tfMat);
        }else if (geometry && geometry->getName()) {
            updateVolume(geometry->getName(), volDesc, true, "", container, group);
            updateTransform(geometry->getName(), tfMat);
        }else {
            const char *name = "Anonymous COVISE object";
            updateVolume(name, volDesc, true, "", container, group);
            updateTransform(name, tfMat);
        }

        if (shader >= 0 && currentVolume != volumes.end())
        {
            virvo::VolumeDrawable *drawable = currentVolume->second.drawable.get();
            if (drawable)
            {
                drawable->setShader(shader);
            }
        }
    }

    // a volume file will be loaded now, so show the TFE
    if (showEditor)
    {
        editor->show();
        tfeItem->setState(true);
    }
}

void VolumePlugin::mapTFToMinMax(VolumeMap::iterator it, vvVolDesc *vd){
    int groupID = std::isdigit(it->first.c_str()[0]) ? it->first.c_str()[0] : -1;
    typedef std::vector<vvTFWidget *> Widgets;
    for (size_t chan = 0; chan < vd->tf.size(); ++chan)
    {
        for (Widgets::iterator widg = vd->tf[chan]._widgets.begin();
             widg != vd->tf[chan]._widgets.end();
             ++widg)
        {
            vvTFWidget *w = *widg;
            if (groupID<0)
                w->mapFrom01(vd->range(chan)[0], vd->range(chan)[1]);
            else
                w->mapFrom01(minData[groupID][chan], maxData[groupID][chan]);
        }
    }
}

bool VolumePlugin::sameObject(VolumeMap::iterator it1, VolumeMap::iterator it2) {
    std::string name1 = it1->first;
    std::string name2 = it2->first;
    if (std::isdigit(name1.c_str()[0])) {
        if (std::strncmp(name1.c_str(), name2.c_str(),2)==0)
            return true;
    }
    return false;
}

void VolumePlugin::setTimestep(int t)
{
    for (VolumeMap::iterator it = volumes.begin();
         it != volumes.end();
         it++)
    {
        it->second.drawable->setCurrentFrame(t);
    }
}

void VolumePlugin::removeObject(const char *name, bool)
{
    vvDebugMsg::msg(2, "VolumePlugin::VRRemoveObject()");

    updateVolume(name, NULL);

}

void VolumePlugin::cropVolume()
{
    if (currentVolume == volumes.end())
        return;

    virvo::VolumeDrawable *drawable = currentVolume->second.drawable.get();
    if (!drawable)
        return;

    std::string name = currentVolume->first;

    vvVolDesc *vd = drawable->getVolumeDescription();
    if (!vd)
        return;

    const Vec3 &roiPosObj = currentVolume->second.roiPosObj;
    const float &roiSize = currentVolume->second.roiCellSize;

    ssize_t x = vd->vox[0], y = vd->vox[1], z = vd->vox[2];
    ssize_t w = x * roiSize, h = y * roiSize, s = z * roiSize;

    Vec3 sz = currentVolume->second.max - currentVolume->second.min;
    Vec3 relCenter = roiPosObj - currentVolume->second.min;
    for (int i = 0; i < 3; ++i)
    {
        relCenter[i] /= sz[i];
    }
    Vec3 relCorner = relCenter - Vec3(roiSize, roiSize, roiSize) * 0.5f;
    x = relCorner[0] * x;
    y = relCorner[1] * y;
    y = vd->vox[1] - y - 1 - h;
    z = relCorner[2] * z;
    z = vd->vox[2] - 1 - z - s;

    if (x < 0)
    {
        w += x;
        x = 0;
    }
    if (y < 0)
    {
        h += y;
        y = 0;
    }
    if (z < 0)
    {
        s += z;
        z = 0;
    }
    if (w < 1)
        w = 1;
    if (h < 1)
        h = 1;
    if (s < 1)
        s = 1;

    vd->crop(x, y, z, w, h, s);
    setROIMode(false);
    roiCellSize = 1.f;
    drawable->setVolumeDescription(vd);
    updateVolume(name.c_str(), vd);
    currentVolume->second.roiCellSize = roiCellSize;
}

void VolumePlugin::syncTransferFunction()
{
    if (currentVolume == volumes.end())
        return;

	for (int i = 0; i < currentVolume->second.tf.size(); ++i)
	{

		*currentVolume->second.tfState[i] = currentVolume->second.tf[i];
	}
}

void VolumePlugin::saveVolume()
{
    if (coVRMSController::instance()->isSlave())
        return;

    if (currentVolume == volumes.end())
        return;

    virvo::VolumeDrawable *drawable = currentVolume->second.drawable.get();
    if (!drawable)
        return;

    vvVolDesc *vd = drawable->getVolumeDescription();
    if (!vd)
        return;

    vd->setFilename("cover-volume.xvf");
    int num = 0;
    for (;;)
    {
        vvFileIO fio;
        const int err = fio.saveVolumeData(vd, false);
        if (err == vvFileIO::OK)
        {
            cerr << "volume saved to " << vd->getFilename() << endl;
            break;
        }
        else if (err == vvFileIO::FILE_EXISTS)
        {
            ++num;
            std::stringstream str;
            str << "cover-volume" << num << ".xvf";
            vd->setFilename(str.str().c_str());
        }
        else
        {
            cerr << "failed to save volume to " << vd->getFilename() << endl;
            break;
        }
    }
}

bool VolumePlugin::updateTransform(const std::string &name, const osg::Matrix &tfMat)
{
    VolumeMap::iterator volume = volumes.find(name);
    if (volume == volumes.end())
        return false;

    auto mirror = osg::Matrix::identity();
    mirror(0, 0) = -1;
    auto &t = volume->second.transform;
    t->setMatrix(osg::Matrix::rotate(M_PI * 0.5, osg::Vec3(0, 1, 0)) * osg::Matrix::rotate(M_PI, osg::Vec3(1, 0, 0)) *
                 mirror * tfMat);
    return true;
}

bool VolumePlugin::updateVolume(const std::string &name, vvVolDesc *vd, bool mapTF, const std::string &filename,
                                const RenderObject *container, osg::Group *group)
{
    if (!vd)
    {
        VolumeMap::iterator volume = volumes.find(name);
        if (volume == volumes.end())
            return false;

        coVRPluginList::instance()->removeNode(volume->second.transform, false, volume->second.transform);

        if (volume == currentVolume)
        {
            VolumeMap::iterator cur = currentVolume;
            if (cur != volumes.end())
                ++cur;
            if (cur == volumes.end())
                cur = volumes.begin();
            makeVolumeCurrent(cur);
        }
        volume->second.removeFromScene();
        if (volume == currentVolume)
            currentVolume = volumes.end();
        volumes.erase(volume);
        return true;
    }

    VolumeMap::iterator volume = volumes.find(name);
    if (volume == volumes.end())
    {
        volumes[name].transform->setName("Volume: "+name);
        volumes[name].addToScene(group);
        volumes[name].filename = filename;
        volumes[name].multiDimTF = vd->getChan() == 1;
        volumes[name].preIntegration = preintItem->state();
        volumes[name].lighting = lightingItem->state();
        volumes[name].mapTF = mapTF;
        if (!container && volumes[name].transform) {
            volumes[name].transform->setMatrix(osg::Matrix::identity());
        }
        if (volumes[name].multiDimTF)
        {
            volumes[name].tf.resize(1);
			volumes[name].tfState.resize(1);
            if (vd->tf[0].isEmpty())
            {
                volumes[name].tf[0] = editor->getTransferFunc(0);
            }
            else
            {
                volumes[name].tf[0] = vd->tf[0];
            }
			volumes[name].tfState[0].reset(new vrb::SharedState<vvTransFunc>(("TransFunc" + name + "0"), volumes[name].tf[0], vrb::ALWAYS_SHARE));
			volumes[name].tfState[0]->setUpdateFunction([this, name]() {
				volumes[name].tf[0] = volumes[name].tfState[0]->value();
				});
        }
        else
        {
			volumes[name].tf.resize(vd->getChan());
			volumes[name].tfState.resize(vd->getChan());
            if (vd->tf.empty() || vd->tf.size() != vd->getChan())
            {
                for (int i = 0; i < volumes[name].tf.size(); ++i)
                {
                    volumes[name].tf[i].setDefaultColors(4 + i, 0, 1);
                    volumes[name].tf[i].setDefaultAlpha(0, 0, 1);
                }
            }
            else
            {
                volumes[name].tf = vd->tf;
            }
			for (int i = 0; i < volumes[name].tf.size(); ++i)
			{
				volumes[name].tfState[i].reset(new vrb::SharedState<vvTransFunc>(("TransFunc" + name + std::to_string(i)), volumes[name].tf[i], vrb::ALWAYS_SHARE));
				volumes[name].tfState[i]->setUpdateFunction([this, name, i]() {
					volumes[name].tf[i] = volumes[name].tfState[i]->value();
					});
			}

            if (vd->channelWeights.size() != vd->getChan())
            {
                vd->channelWeights.resize(vd->getChan());
                std::fill(vd->channelWeights.begin(), vd->channelWeights.end(), 1.0f);
            }
            volumes[name].channelWeights = vd->channelWeights;
        }
        volume = volumes.find(name);
        coVRPluginList::instance()->addNode(volume->second.transform, container, this);
    }

    virvo::VolumeDrawable *drawable = volume->second.drawable.get();
    if (vd != drawable->getVolumeDescription())
    {
        VRViewer::instance()->culling(false);
        reenableCulling = true;
        drawable->setVolumeDescription(vd);
    }

    volume->second.drawable->getBoundingBox(&volume->second.min, &volume->second.max);
    volume->second.roiPosObj = (volume->second.min + volume->second.max) * .5;

    // Update animation frame:
    coVRAnimationManager::instance()->setNumTimesteps(drawable->getNumFrames(), drawable);

    makeVolumeCurrent(volume);

    drawable->setTransferFunctions(volume->second.tf);
    if (mapTF)
        mapTFToMinMax(volume, vd);
         
    drawable->setPreintegration(volume->second.preIntegration);
    drawable->setLighting(volume->second.lighting);
    drawable->setChannelWeights(volume->second.channelWeights);
    drawable->setUseChannelWeights(volume->second.useChannelWeights);

    return true;
}

void VolumePlugin::makeVolumeCurrent(VolumeMap::iterator it)
{
    if (currentVolume != volumes.end())
    {
        virvo::VolumeDrawable *drawable = currentVolume->second.drawable.get();
        if (drawable)
            drawable->setBoundariesActive(false);
        const int curChan = editor->getActiveChannel();
        currentVolume->second.curChannel = curChan;
        currentVolume->second.tf = editor->getTransferFuncs();
        currentVolume->second.channelWeights = editor->getChannelWeights();
    }
    currentVolume = it;
    if (currentVolume != volumes.end())
    {
        virvo::VolumeDrawable *drawable = currentVolume->second.drawable.get();
        drawable->setBoundariesActive(true);
        Vec3 &roiPosObj = currentVolume->second.roiPosObj;
        Vec3 &min = currentVolume->second.min;
        Vec3 &max = currentVolume->second.max;

        roiMaxSize = fabs(max[0] - min[0]);
        if (fabs(max[1] - min[1]) > roiMaxSize)
            roiMaxSize = fabs(max[1] - min[1]);
        if (fabs(max[2] - min[2]) > roiMaxSize)
            roiMaxSize = fabs(max[2] - min[2]);
        drawable->setROIPosition(roiPosObj);

        setROIMode(currentVolume->second.roiMode);
        ROIItem->setState(currentVolume->second.roiMode);

        lightingItem->setState(currentVolume->second.lighting);
        preintItem->setState(currentVolume->second.preIntegration);
        boundItem->setState(currentVolume->second.boundaries);
        interpolItem->setState(currentVolume->second.interpolation);

        if (currentVolume->second.blendMode == virvo::VolumeDrawable::AlphaBlend)
        {
            if (backgroundColor == BgDefault)
                blendModeItem->select(0);
            else if (backgroundColor == BgDark)
                blendModeItem->select(1);
            else if (backgroundColor == BgLight)
                blendModeItem->select(2);
        }
        else if (currentVolume->second.blendMode == virvo::VolumeDrawable::MaximumIntensity)
        {
            blendModeItem->select(3);
        }
        else if (currentVolume->second.blendMode == virvo::VolumeDrawable::MinimumIntensity)
        {
            blendModeItem->select(4);
        }

        std::string displayName = currentVolume->second.filename;
        std::string::size_type slash = displayName.rfind('/');
        if (slash != std::string::npos)
            displayName = displayName.substr(slash + 1);
        if (currentVolume->second.filename.empty())
            displayName = "[COVISE]";
        currentVolumeItem->setText(displayName);
    }
    else
    {
        currentVolumeItem->setText("(none)");
    }

    updateTFEData();
}

void VolumePlugin::updateTFEData()
{
    if (!editor)
        return;

    if (currentVolume != volumes.end())
    {
        tfApplyCBData.volume = &currentVolume->second;
        tfApplyCBData.drawable = currentVolume->second.drawable.get();
        if (tfApplyCBData.drawable)
        {
            vvVolDesc *vd = tfApplyCBData.drawable->getVolumeDescription();
            if (vd)
            {
                if (computeHistogram && vd->getFrameVoxels() < maxHistogramVoxels)
                {
                    size_t res[] = { TEXTURE_RES_BACKGROUND, TEXTURE_RES_BACKGROUND };
                    vvColor fg(1.0f, 1.0f, 1.0f);
                    vd->makeHistogramTexture(0, 0, 1, res, &tfeBackgroundTexture[0], vvVolDesc::VV_LOGARITHMIC, &fg, 0,1);
                    editor->updateBackground(&tfeBackgroundTexture[0]);
                    editor->pinedit->setBackgroundType(0); // histogram
                    editor->enableHistogram(true);
                }
                else
                {
                    editor->enableHistogram(false);
                }

                editor->setNumChannels(vd->getChan());

                editor->setTransferFuncs(currentVolume->second.tf);
                
                int objID = std::isdigit(currentVolume->first.c_str()[0]) ? currentVolume->first.c_str()[0] : -1;
                for (int c = 0; c < vd->getChan(); ++c)
                {
                    editor->setActiveChannel(c);
                    editor->setMin(objID > 0 ? minData[objID][c] : vd->range(c)[0]);
                    editor->setMax(objID > 0 ? maxData[objID][c] : vd->range(c)[1]);
                }

                editor->setActiveChannel(currentVolume->second.curChannel);
                tfApplyCBData.drawable->setTransferFunctions(editor->getTransferFuncs());
                if (tfApplyCBData.volume->mapTF)
                    mapTFToMinMax(currentVolume, vd);
                
                tfApplyCBData.drawable->setChannelWeights(editor->getChannelWeights());
                tfApplyCBData.drawable->setUseChannelWeights(editor->getUseChannelWeights());

                instantMode = tfApplyCBData.drawable->getInstantMode();
                editor->setInstantMode(instantMode);

                //after loading data, if we have a tabletUI,
                // 1) notify the tabletUI function editor of its dimensionality (channel number)
                // 2) send it the histogram if the plugin is configured this way
                functionEditorTab->setDimension(vd->getChan());
                delete[] functionEditorTab -> histogramData;
                functionEditorTab->histogramData = NULL;

                int buckets[2] = {
                    coTUIFunctionEditorTab::histogramBuckets,
                    vd->getChan() == 1 ? 1 : coTUIFunctionEditorTab::histogramBuckets
                };
                if (computeHistogram && vd->getFrameVoxels() < maxHistogramVoxels)
                {
                    functionEditorTab->histogramData = new int[buckets[0] * buckets[1]];
                    if (vd->getChan() == 1)
                        vd->makeHistogram(0, 0, 1, buckets, functionEditorTab->histogramData, 0,1);
                        else
                        //TODO: allow to pass in multiple min/max pairs
                            vd->makeHistogram(0, 0, 2, buckets, functionEditorTab->histogramData,0,1);
                    editor->enableHistogram(true);
                    std::cerr << "enabling histogram" << std::endl;
                }
                else
                {
                    std::cerr << "disabling histogram" << std::endl;
                    editor->enableHistogram(false);
                }
            }
        }
    }
    editor->update();
}

void VolumePlugin::preFrame()
{
    const float SPEED = 0.3f; // adaption speed
    float change;
    static bool firstFPSmeasurement = true;

    bool framesSkipped = updateCount > 1;
    updateCount = 0;

    vvDebugMsg::msg(3, "VolumePlugin::VRPreFrame()");

    if (editor)
    {
        virvo::VolumeDrawable *drawable = getCurrentDrawable();
        if (drawable)
        {
            if (instantMode != drawable->getInstantMode())
            {
                instantMode = drawable->getInstantMode();
                editor->setInstantMode(instantMode);
            }
        }
        editor->update();
        tfeItem->setState(editor->isVisible());
    }
    else
    {
        fprintf(stderr, "no editor\n");
    }


    typedef std::vector<boost::shared_ptr<coClipSphere> >::iterator SphereIt;
    {
        virvo::VolumeDrawable *drawable = getCurrentDrawable();
        for (SphereIt it = clipSpheres.begin(); it != clipSpheres.end(); ++it)
        {
            boost::shared_ptr<coClipSphere> sphere = *it;
            if (drawable && sphere->active() && !sphere->valid())
            {
                // If this clip sphere is activated for the first time,
                // start out at the min corner of the volume's bounding box
                osg::Vec3 min_corner;
                osg::Vec3 max_corner;
                drawable->getBoundingBox(&min_corner, &max_corner);
                osg::Matrix transform = osg::Matrix::translate(min_corner);
                (void)max_corner;
                sphere->setMatrix(transform);
                sphere->setValid(true);
            }
            sphere->preFrame();
        }
    }

    // Measure fps:
    double end = cover->frameTime();
    float fps = INITIAL_FPS;
    if (firstFPSmeasurement)
    {
        firstFPSmeasurement = false;
        fps = INITIAL_FPS;
    }
    else
    {
        fps = 1.0f / (end - start);
    }
    start = end; // start stopwatch

    float quality = currentQuality;
    if (cover->isHighQuality())
    {
        quality = highQualityOversampling;
        fpsMissed = 0;
    }
    else
    {
        float threshold = 0.1f * chosenFPS;
        if (fabs(fps - chosenFPS) > threshold)
        {
            if (!framesSkipped)
                fpsMissed++;
        }
        else
        {
            fpsMissed = 0;
        }

        // skip fps variations due to transfer function changes (especially when pre-integrating)
        if (fpsMissed > 1)
        {
            fpsMissed = 0;
            if (chosenFPS > 0.0f)
            {
                change = (fps - chosenFPS) / chosenFPS;
                currentQuality += currentQuality * change * SPEED;
            }
            else
            {
                currentQuality = MAX_QUALITY;
            }

            //cerr << "chosenFPS=" << chosenFPS << ", fps=" << fps << "q=" << currentQuality << endl;

            currentQuality = coClamp(currentQuality, MIN_QUALITY, MAX_QUALITY);
            currentQuality = std::min(currentQuality, highQualityOversampling);
        }
        quality = currentQuality;
    }

    Matrix vMat = cover->getViewerMat();
    // viewing direction, world space (Performer format)
    Vec3 viewDirWld(vMat(1, 0), vMat(1, 1), vMat(1, 2));
    // viewing direction, object space (Performer format)
    Vec3 viewDirObj = osg::Matrix::transform3x3(viewDirWld, cover->getInvBaseMat()); // 3x3 only

    for (VolumeMap::iterator it = volumes.begin();
         it != volumes.end();
         it++)
    {
        // Set direction from viewer to object  for renderer:
        Vec3 center((it->second.min + it->second.max) * .5);
        if (roiMode && it == currentVolume)
        {
            center = it->second.roiPosObj;
        }
        Vec3 centerWorld = center;// * cover->getBaseMat();
        Vec3 viewerPosWorld = cover->getViewerMat().getTrans();
        Vec3 objDirWorld = centerWorld - viewerPosWorld;
        Vec3 objDirObj = osg::Matrix::transform3x3(objDirWorld, cover->getInvBaseMat()); // 3x3 only
        objDirObj.normalize();
        //cerr << "Vec: " << objDirObj[0] << "   "<< objDirObj[1] << "   " << objDirObj[2] << endl;

        virvo::VolumeDrawable *drawable = it->second.drawable.get();
        // Adjust image quality, viewing & object direction
        if (drawable)
        {
            const osg::Matrix &t = it->second.transform->getMatrix();
            auto invT = osg::Matrix::inverse(t);
            drawable->setQuality(quality);
            drawable->setViewDirection(viewDirObj*invT);
            drawable->setObjectDirection(objDirObj*invT);

            typedef vvRenderState::ParameterType PT;

            int maxClipPlanes = drawable->getMaxClipPlanes();
            int numClipPlanes = 0;

            if (cover->isClippingOn())
            {
                StateSet *state = drawable->getOrCreateStateSet();
                ClipNode *cn = cover->getObjectsRoot();
                for (int i = 0; i < std::min((int)cn->getNumClipPlanes(), maxClipPlanes); ++i)
                {
                    ClipPlane *cp = cn->getClipPlane(i);
                    Vec4 v = cp->getClipPlane() * invT;

                    boost::shared_ptr<vvClipPlane> plane = vvClipPlane::create();
                    plane->normal = -virvo::vec3(v.x(), v.y(), v.z());
                    plane->offset = v.w();

                    drawable->setParameter(PT(vvRenderState::VV_CLIP_OBJ0 + i), plane);
                    drawable->setParameter(PT(vvRenderState::VV_CLIP_OUTLINE0 + i), showClipOutlines);

                    if (ignoreCoverClipping || followCoverClipping)
                    {
                        state->setMode(GL_CLIP_PLANE0 + cp->getClipPlaneNum(), StateAttribute::OFF);
                    }
                    else
                    {
                        state->removeMode(GL_CLIP_PLANE0 + cp->getClipPlaneNum());
                    }
                    drawable->setParameter(PT(vvRenderState::VV_CLIP_OBJ_ACTIVE0 + cp->getClipPlaneNum()), singleSliceClipping || (!ignoreCoverClipping && followCoverClipping));

                    ++numClipPlanes;
                }

                drawable->setStateSet(state);
            }

            // Get bounding box diagonal to determine clip sphere radius
            osg::Vec3 minCorner;
            osg::Vec3 maxCorner;
            drawable->getBoundingBox(&minCorner, &maxCorner);
            osg::Vec3 diagonal = maxCorner - minCorner;

            int numClipSpheres = 0;
            SphereIt it = clipSpheres.begin();
            int objId = numClipPlanes;
            for (int i = numClipPlanes; i < maxClipPlanes && it != clipSpheres.end(); ++i, ++it)
            {
                if ((*it)->active())
                {
                    osg::Vec3 center = (*it)->getPosition();
                    boost::shared_ptr<vvClipSphere> sphere = vvClipSphere::create();
                    sphere->center = virvo::vec3(center.z(), -center.y(), center.x());
                    sphere->radius = diagonal.length() * 0.5f * radiusScale[i - numClipPlanes];

                    drawable->setParameter(PT(vvRenderState::VV_CLIP_OBJ0 + objId), sphere);
                    drawable->setParameter(PT(vvRenderState::VV_CLIP_OBJ_ACTIVE0 + objId), true);
                    drawable->setParameter(PT(vvRenderState::VV_CLIP_OUTLINE0 + i), showClipOutlines);

                    ++objId;
                    ++numClipSpheres;
                }
            }

            int numClipObjs = numClipPlanes + numClipSpheres;

            // Disable all clip objects that are not used
            for (PT i = PT(vvRenderState::VV_CLIP_OBJ_ACTIVE0 + numClipObjs);
                    i != vvRenderState::VV_CLIP_OBJ_ACTIVE_LAST;
                    i = PT(i + 1))
            {
                drawable->setParameter(i, false);
            }

            drawable->setParameter(vvRenderState::VV_FOCUS_CLIP_OBJ, cover->getActiveClippingPlane());
        }
    }

    virvo::VolumeDrawable *drawable = getCurrentDrawable();

    // Process ROI mode:
    bool mouse = false;
    if (roiMode && pointerInROI(&mouse))
    {
        if (drawable)
        {
            drawable->setROISelected(true);
            drawable->setBoundaries(true);
        }
        if (!interactionA->isRegistered())
        {
            coInteractionManager::the()->registerInteraction(interactionA);
            interactionA->setHitByMouse(mouse);
        }
        if (!interactionB->isRegistered())
        {
            coInteractionManager::the()->registerInteraction(interactionB);
            interactionB->setHitByMouse(mouse);
        }
    }
    else
    {
        unregister = true;
    }

    if (interactionA->wasStarted())
    {
        // start ROI move
        invStartMove.invert(interactionA->is2D() ? cover->getMouseMat() : cover->getPointerMat());
        if (currentVolume != volumes.end())
            startPointerPosWorld = currentVolume->second.roiPosObj * currentVolume->second.transform->getMatrix() * cover->getBaseMat();
        else
            startPointerPosWorld = Vec3(0., 0., 0.) * cover->getBaseMat();
    }
    //int rollCoord = interactionB->is2D() ? 1 : 2;
    int rollCoord = interactionB->is2D() ? 0 : 2;
    if (interactionB->wasStarted())
    {
        coCoord mouseCoord(interactionB->is2D() ? cover->getMouseMat() : cover->getPointerMat());
        lastRoll = mouseCoord.hpr[rollCoord];
    }

    if (interactionA->isRunning())
    {
        Matrix moveMat;
        moveMat.mult(invStartMove, interactionA->is2D() ? cover->getMouseMat() : cover->getPointerMat());
        Vec3 roiPosWorld = startPointerPosWorld * moveMat;
        if (currentVolume != volumes.end())
        {
            osg::Matrix t = osg::Matrix::inverse(currentVolume->second.transform->getMatrix());
            currentVolume->second.roiPosObj = roiPosWorld * cover->getInvBaseMat() * t;
        }

        if (drawable)
        {
            if (drawable->getROISize() <= 0.0f)
                drawable->setROISize(0.00001f);
            if (coVRCollaboration::instance()->getCouplingMode() != coVRCollaboration::MasterSlaveCoupling
                || coVRCollaboration::instance()->isMaster())
            {
                drawable->setROIPosition(currentVolume->second.roiPosObj);
                if (coVRCollaboration::instance()->getCouplingMode() != coVRCollaboration::LooseCoupling)
                {
                    sendROIMessage(drawable->getROIPosition(), drawable->getROISize());
                }
            }
        }
    }
    if (interactionB->isRunning())
    {
        if (coVRCollaboration::instance()->getCouplingMode() != coVRCollaboration::MasterSlaveCoupling
                || coVRCollaboration::instance()->isMaster())
        {
            bool mouse = interactionB->is2D();
            coCoord mouseCoord(mouse ? cover->getMouseMat() : cover->getPointerMat());
            if (lastRoll != mouseCoord.hpr[rollCoord])
            {
                if ((lastRoll - mouseCoord.hpr[rollCoord]) > 180)
                    lastRoll -= 360;
                if ((lastRoll - mouseCoord.hpr[rollCoord]) < -180)
                    lastRoll += 360;

                float rollDiff = (lastRoll - (float)mouseCoord.hpr[rollCoord]) / 90.0f;
                roiCellSize += rollDiff * (mouse ? 10 : 1);
                lastRoll = (float)mouseCoord.hpr[rollCoord];

                if (roiCellSize <= 0.1)
                    roiCellSize = 0.1; // retain minimum size for ROI to be visible
                if (roiCellSize > 1.0)
                    roiCellSize = 1.0;
                cerr << "roi=" << roiCellSize << ", mouse=" << mouse << endl;
                if (drawable)
                    drawable->setROISize(roiCellSize);
                if (currentVolume != volumes.end())
                {
                    currentVolume->second.roiCellSize = roiCellSize;
                }
                if (drawable && coVRCollaboration::instance()->getCouplingMode() != coVRCollaboration::LooseCoupling)
                {
                    sendROIMessage(drawable->getROIPosition(), drawable->getROISize());
                }
            }
        }
    }
    if (interactionA->wasStopped())
    {
        if (coVRCollaboration::instance()->getCouplingMode() != coVRCollaboration::MasterSlaveCoupling
            || coVRCollaboration::instance()->isMaster())
        {
            if (!roiVisible())
            {
                ROIItem->setState(false);
#ifdef VERBOSE
                cerr << "pointer Released (ROI not visible)" << endl;
#endif
                if (currentVolume != volumes.end())
                {
                    currentVolume->second.roiPosObj = (currentVolume->second.min + currentVolume->second.max) * .5;
                    if (drawable)
                    {
                        drawable->setROIPosition(currentVolume->second.roiPosObj);
                        drawable->setROISize(0.f);
                    }
                }
                roiMode = false;
                return;
            }

            if (drawable && coVRCollaboration::instance()->getCouplingMode() != coVRCollaboration::LooseCoupling)
            {
                sendROIMessage(drawable->getROIPosition(), drawable->getROISize());
            }
        }
    }

    if (unregister)
    {
        if (interactionA->isRegistered() && (interactionA->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionA);
        }
        if (interactionB->isRegistered() && (interactionB->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionB);
        }
        if ((!interactionA->isRegistered()) && (!interactionB->isRegistered()))
        {
            unregister = false;
            if (drawable)
            {
                drawable->setROISelected(false);
                if (currentVolume != volumes.end())
                    drawable->setBoundaries(currentVolume->second.boundaries);
                else
                    drawable->setBoundaries(false);
            }
        }
    }
}

void VolumePlugin::postFrame()
{
    vvDebugMsg::msg(3, "VolumePlugin::VRPostFrame");
    volDesc = NULL;
    if (reenableCulling)
    {
        reenableCulling = false;
        VRViewer::instance()->culling(true);
    }
}

void VolumePlugin::applyToVolumes(std::function<void(Volume &)> func)
{
    for (VolumeMap::iterator it = allVolumesActive ? volumes.begin() : currentVolume; it != volumes.end(); ++it)
    {
        virvo::VolumeDrawable *drawable = it->second.drawable;
        if (!drawable)
            continue;

        func(it->second);

        if (!allVolumesActive)
            break;
    }
}

void VolumePlugin::setROIMode(bool newMode)
{
    vvDebugMsg::msg(1, "VolumePlugin::setROIMode()");

    if (newMode)
    {
        if (currentVolume != volumes.end())
        {
            roiCellSize = currentVolume->second.roiCellSize;
            if (roiCellSize <= 0.1f)
                roiCellSize = 0.1f;
            if (roiCellSize > 1.0f)
                roiCellSize = 1.0f;
            currentVolume->second.roiMode = true;
        }
        if (coVRCollaboration::instance()->getCouplingMode() != coVRCollaboration::MasterSlaveCoupling
            || coVRCollaboration::instance()->isMaster())
        {
            roiMode = true;
        }
        else
        {
            ROIItem->setState(false);
        }
    }
    else
    {
        if (coVRCollaboration::instance()->getCouplingMode() != coVRCollaboration::MasterSlaveCoupling
            || coVRCollaboration::instance()->isMaster())
        {
            roiCellSize = 0.0f;
            roiMode = false;
        }
        else
        {
            ROIItem->setState(true);
        }
        if (currentVolume != volumes.end())
        {
            currentVolume->second.roiMode = false;
        }
    }

    virvo::VolumeDrawable *drawable = getCurrentDrawable();
    if (drawable)
    {
        drawable->setROISize(roiCellSize);

        if (coVRCollaboration::instance()->getCouplingMode() != coVRCollaboration::MasterSlaveCoupling
            || coVRCollaboration::instance()->isMaster())
        {
            if (coVRCollaboration::instance()->getCouplingMode() != coVRCollaboration::LooseCoupling)
            {
                sendROIMessage(drawable->getROIPosition(), drawable->getROISize());
            }
        }
    }
}

void VolumePlugin::setClippingMode(bool newMode)
{
    vvDebugMsg::msg(1, "VolumePlugin::setClippingMode()");

    virvo::VolumeDrawable *drawable = getCurrentDrawable();
    if (drawable)
    {
        drawable->setClipping(newMode);
    }
}

virvo::VolumeDrawable *VolumePlugin::getCurrentDrawable()
{
    if (currentVolume != volumes.end())
        return currentVolume->second.drawable.get();
    else
        return NULL;
}

covise::TokenBuffer& operator<<(covise::TokenBuffer& tb, const vvTransFunc& id)
{
	std::vector<char> buf;
	typedef boost::iostreams::back_insert_device<std::vector<char> > sink_type;
	typedef boost::iostreams::stream<sink_type> stream_type;

	sink_type sink(buf);
	stream_type stream(sink);

	// Create a serializer
	boost::archive::binary_oarchive archive(stream);

	// Serialize the message
	archive << id;

	// Don't forget to flush the stream!!!
	stream.flush();
	tb << int(buf.size());
	tb.addBinary(&buf[0], buf.size());
	return tb;
}

covise::TokenBuffer& operator>>(covise::TokenBuffer& tb, vvTransFunc& id)
{
	int size;
	tb >> size;
    const auto *begin = tb.getBinary(size);
    std::vector<char> buf(begin, begin + size);
    typedef boost::iostreams::basic_array_source<char> source_type;
	typedef boost::iostreams::stream<source_type> stream_type;

	source_type source(&buf[0], buf.size());
	stream_type stream(source);

	// Create a deserialzer
	boost::archive::binary_iarchive archive(stream);

	// Deserialize the message
	archive >> id;
	return tb;
}

COVERPLUGIN(VolumePlugin)
