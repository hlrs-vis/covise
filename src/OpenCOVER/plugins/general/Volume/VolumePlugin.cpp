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

#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coFrame.h>
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
    transform->setMatrix(osg::Matrix::rotate(M_PI*0.5, osg::Vec3(0,1,0)) * osg::Matrix::rotate(M_PI, osg::Vec3(1,0,0)));
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

void VolumePlugin::Volume::addToScene()
{
#ifdef VERBOSE
    cerr << "add volume to scene" << endl;
#endif
    if (!inScene)
    {
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
    fileMenuItem = new coButtonMenuItem(menuName);
    fileMenuItem->setMenuListener(VolumePlugin::plugin);
    VolumePlugin::plugin->filesMenu->add(fileMenuItem);
}

FileEntry::~FileEntry()
{
    delete[] fileName;
    delete[] menuName;
    delete fileMenuItem;
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
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "tif" },
    { NULL,
      VolumePlugin::loadVolume,
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "tiff" },
    { NULL,
      VolumePlugin::loadVolume,
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "xvf" },
    { NULL,
      VolumePlugin::loadVolume,
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "rvf" },
    { NULL,
      VolumePlugin::loadVolume,
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "avf" },
    { NULL,
      VolumePlugin::loadVolume,
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "nii" },
    { NULL,
      VolumePlugin::loadVolume,
      VolumePlugin::loadVolume,
      VolumePlugin::unloadVolume,
      "nii.gz" },
};

/// Constructor
VolumePlugin::VolumePlugin()
    : editor(NULL)
{
}

bool VolumePlugin::init()
{
    vvDebugMsg::msg(1, "VolumePlugin::VolumePlugin()");

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
    computeHistogram = covise::coCoviseConfig::isOn("value", "COVER.Plugin.Volume.UseHistogram", false, &ignore);
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
    pinboardEntry.reset(new coSubMenuItem("Volume..."));
    cover->getMenu()->add(pinboardEntry.get());
    volumeMenu.reset(new coRowMenu("Volume", cover->getMenu()));
    pinboardEntry->setMenu(volumeMenu.get());

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

    filesItem.reset(new coSubMenuItem("Files..."));
    volumeMenu->add(filesItem.get());
    filesMenu.reset(new coRowMenu("Files", volumeMenu.get()));
    filesItem->setMenu(filesMenu.get());

    clipItem.reset(new coSubMenuItem("Clipping..."));
    clipMenu.reset(new coRowMenu("Clipping", volumeMenu.get()));
    clipItem->setMenu(clipMenu.get());
    volumeMenu->add(clipItem.get());

    unloadItem.reset(new coButtonMenuItem("Unload Current File"));
    filesMenu->add(unloadItem.get());
    tfeItem.reset(new coButtonMenuItem("Display TFE"));
    ROIItem.reset(new coCheckboxMenuItem("Region of Interest", false));
    cropItem.reset(new coButtonMenuItem("Crop to ROI"));
    saveItem.reset(new coButtonMenuItem("Save Volume"));
    fpsItem.reset(new coSliderMenuItem("Frame Rate", 5.0, 60.0, chosenFPS));
    boundItem.reset(new coCheckboxMenuItem("Boundaries", false));
    interpolItem.reset(new coCheckboxMenuItem("Interpolation", false));
    preintItem.reset(new coCheckboxMenuItem("Pre-integration", preIntegration));
    lightingItem.reset(new coCheckboxMenuItem("Lighting", lighting));
    colorsItem.reset(new coPotiMenuItem("Discrete Colors", 0.0, 32.0, 0, VolumeCoim.get(), "DISCRETE_COLORS"));
    hqItem.reset(new coSliderMenuItem("Oversampling", 1.0, MAX_QUALITY * 2., highQualityOversampling));
    allVolumesActiveItem.reset(new coCheckboxMenuItem("All Volumes Active", allVolumesActive));
    cycleVolumeItem.reset(new coButtonMenuItem("Cycle Active Volume"));
    currentVolumeItem.reset(new coLabelMenuItem("[]"));
    sideBySideItem.reset(new coCheckboxMenuItem("Side By Side", false));

    blendModeItem.reset(new coSubMenuItem("Blending..."));
    blendModeMenu.reset(new coRowMenu("Blending", volumeMenu.get()));
    blendModeItem->setMenu(blendModeMenu.get());

    coCheckboxGroup *blendGroup = new coCheckboxGroup();
    alphaDefBlendItem.reset(new coCheckboxMenuItem("Alpha Blending (default)", true, blendGroup));
    alphaLightBlendItem.reset(new coCheckboxMenuItem("Alpha Blending (light)", false, blendGroup));
    alphaDarkBlendItem.reset(new coCheckboxMenuItem("Alpha Blending (dark)", false, blendGroup));
    minIntensityItem.reset(new coCheckboxMenuItem("Minimum Intensity", false, blendGroup));
    maxIntensityItem.reset(new coCheckboxMenuItem("Maximum Intensity", false, blendGroup));
    blendModeMenu->add(alphaDefBlendItem.get());
    blendModeMenu->add(alphaLightBlendItem.get());
    blendModeMenu->add(alphaDarkBlendItem.get());
    blendModeMenu->add(minIntensityItem.get());
    blendModeMenu->add(maxIntensityItem.get());

    // Create clipping menu

    clipModeItem.reset(new coCheckboxMenuItem("Opaque Clipping", false));
    clipOutlinesItem.reset(new coCheckboxMenuItem("Show Box Intersections", true));
    clipSphereActive0Item.reset(new coCheckboxMenuItem("ClipSphere 0 enable", false));
    clipSphereInteractorActive0Item.reset(new coCheckboxMenuItem("ClipSphere 0 Interactor", false));
    clipSphereRadius0Item.reset(new coSliderMenuItem("Radius", 0.1, 1.0, radiusScale[0]));
    clipSphereActive1Item.reset(new coCheckboxMenuItem("ClipSphere 1 enable", false));
    clipSphereInteractorActive1Item.reset(new coCheckboxMenuItem("ClipSphere 1 Interactor", false));
    clipSphereRadius1Item.reset(new coSliderMenuItem("Radius", 0.1, 1.0, radiusScale[1]));
    clipSphereActive2Item.reset(new coCheckboxMenuItem("ClipSphere 2 enable", false));
    clipSphereInteractorActive2Item.reset(new coCheckboxMenuItem("ClipSphere 2 Interactor", false));
    clipSphereRadius2Item.reset(new coSliderMenuItem("Radius", 0.1, 1.0, radiusScale[2]));

    // Set event listeners

    filesItem->setMenuListener(this);
    clipItem->setMenuListener(this);
    blendModeItem->setMenuListener(this);
    ROIItem->setMenuListener(this);
    unloadItem->setMenuListener(this);

    clipModeItem->setMenuListener(this);
    clipOutlinesItem->setMenuListener(this);
    clipSphereActive0Item->setMenuListener(this);
    clipSphereInteractorActive0Item->setMenuListener(this);
    clipSphereRadius0Item->setMenuListener(this);
    clipSphereActive1Item->setMenuListener(this);
    clipSphereInteractorActive1Item->setMenuListener(this);
    clipSphereRadius1Item->setMenuListener(this);
    clipSphereActive2Item->setMenuListener(this);
    clipSphereInteractorActive2Item->setMenuListener(this);
    clipSphereRadius2Item->setMenuListener(this);
    preintItem->setMenuListener(this);
    lightingItem->setMenuListener(this);
    fpsItem->setMenuListener(this);
    hqItem->setMenuListener(this);
    boundItem->setMenuListener(this);
    interpolItem->setMenuListener(this);
    interpolItem->setState(true);
    colorsItem->setMenuListener(this);
    colorsItem->setInteger(true);
    cropItem->setMenuListener(this);
    saveItem->setMenuListener(this);
    tfeItem->setMenuListener(this);
    cycleVolumeItem->setMenuListener(this);
    allVolumesActiveItem->setMenuListener(this);
    alphaDefBlendItem->setMenuListener(this);
    alphaDarkBlendItem->setMenuListener(this);
    alphaLightBlendItem->setMenuListener(this);
    maxIntensityItem->setMenuListener(this);
    minIntensityItem->setMenuListener(this);
    sideBySideItem->setMenuListener(this);

    volumeMenu->add(boundItem.get());
    volumeMenu->add(ROIItem.get());
    volumeMenu->add(cropItem.get());
    volumeMenu->add(saveItem.get());

    volumeMenu->add(interpolItem.get());
    volumeMenu->add(blendModeItem.get());
    volumeMenu->add(preintItem.get());
    volumeMenu->add(lightingItem.get());
    volumeMenu->add(fpsItem.get());
    volumeMenu->add(hqItem.get());
    volumeMenu->add(colorsItem.get());
    volumeMenu->add(tfeItem.get());
    volumeMenu->add(cycleVolumeItem.get());
    volumeMenu->add(currentVolumeItem.get());
    volumeMenu->add(sideBySideItem.get());
    volumeMenu->add(allVolumesActiveItem.get());

    clipMenu->add(clipModeItem.get());
    clipMenu->add(clipOutlinesItem.get());
    clipMenu->add(clipSphereActive0Item.get());
    clipMenu->add(clipSphereInteractorActive0Item.get());
    clipMenu->add(clipSphereRadius0Item.get());
    clipMenu->add(clipSphereActive1Item.get());
    clipMenu->add(clipSphereInteractorActive1Item.get());
    clipMenu->add(clipSphereRadius1Item.get());
    clipMenu->add(clipSphereActive2Item.get());
    clipMenu->add(clipSphereInteractorActive2Item.get());
    clipMenu->add(clipSphereRadius2Item.get());

    // Initialize clip spheres
    for (int i = 0; i < NumClipSpheres; ++i)
    {
        clipSpheres.push_back(boost::make_shared<coClipSphere>());
    }

    // Read volume file entries from covise.config:
    covise::coCoviseConfig::ScopeEntries e = covise::coCoviseConfig::getScopeEntries("COVER.Plugin.Volume.Files");
    const char **entries = e.getValue();
    if (entries)
    {
        while (*entries)
        {
            const char *menuName = *entries;
            entries++;
            const char *fileName = *entries;
            entries++;
            //if(fileName && menuName)
            // cerr << "VolumePlugin: file " << fileName << "   " << menuName << endl;
            fileList.push_back(new FileEntry(fileName, menuName));
        }
    }
    // Load volume data:
    std::string line = covise::coCoviseConfig::getEntry("COVER.Plugin.Volume.VolumeFile");
    if (!line.empty())
    {
        loadFile(line.c_str(), NULL);
    }

    return true;
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
                if (vd->chan < 2)
                {
                    vd->addGradient(0, vvVolDesc::GRADIENT_MAGNITUDE);
                    assert(vd->chan >= 2);
                    functionEditorTab->setDimension(vd->chan);

                    // refresh the histogram
                    unsigned int buckets[2];
                    buckets[0] = coTUIFunctionEditorTab::histogramBuckets;
                    buckets[1] = coTUIFunctionEditorTab::histogramBuckets;

                    delete[] functionEditorTab -> histogramData;
                    functionEditorTab->histogramData = NULL;
                    functionEditorTab->histogramData = new int[buckets[0] * buckets[1]];
                    vd->makeHistogram(0, 0, 2, buckets, functionEditorTab->histogramData, 0, 1);

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
                    if (allVolumesActiveItem->getState() || data->drawable == it->second.drawable)
                    {
                        if (it->second.multiDimTF == data->volume->multiDimTF)
                        {
                            it->second.curChannel = tfe->getActiveChannel();
                            it->second.tf[tfe->getActiveChannel()] = func;
                            it->second.drawable->setTransferFunctions(it->second.tf);
                            if (it->second.mapTF)
                                it->second.drawable->mapTransferFunctionsFrom01();
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
            if (it->second.drawable == data->drawable || allVolumesActiveItem->getState())
            {
                it->second.tf = tfe->getTransferFuncs();
                it->second.drawable->setTransferFunctions(it->second.tf);
                if (it->second.mapTF)
                    it->second.drawable->mapTransferFunctionsFrom01();
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
            const char *filename = "cover-transferfunction_(";
            const char *suffix = ").xvf";
            const size_t len = strlen(filename) + strlen(suffix) + digits + 1;
            char *dest = (char *)calloc(len, sizeof(char));
            strncat(dest, filename, strlen(filename));
            strncat(dest, filenumberStr, digits);
            strncat(dest, suffix, strlen(suffix));
            dest[len - 1] = '\0';
            vd->setFilename(dest);
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

int VolumePlugin::loadFile(const char *fName, osg::Group *parent)
{
    (void)parent;
    vvDebugMsg::msg(1, "VolumePlugin::loadFile()");

    const char *fn = coVRFileManager::instance()->getName(fName);
    if (!fn)
    {
        cerr << "Invalid file name" << endl;
        return 0;
    }

    std::string fileName(fn);
#ifdef VERBOSE
    cerr << "Loading volume file: " << fileName << endl;
#endif

    vvVolDesc *vd = new vvVolDesc(fileName.c_str());
    vvFileIO fio;
    if (fio.loadVolumeData(vd) != vvFileIO::OK)
    {
        cerr << "Cannot load volume file: " << fileName << endl;
        delete vd;
        vd = NULL;
        return 0;
    }

    vd->printInfoLine("Loaded");

    editor->show();
    // a volumefile will be loaded now , so show the TFE

    updateVolume(fileName, vd, false, fileName);

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

void VolumePlugin::message(int type, int len, const void *buf)
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

            if ((coVRCollaboration::instance()->getSyncMode() == coVRCollaboration::TightCoupling))
            {
                if (drawable && drawable->getROISize() > 0.)
                {
                    roiMode = true;
                    cover->setButtonState("Region of Interest", 1);
                }
                else
                {
                    roiMode = false;
                    cover->setButtonState("Region of Interest", 0);
                }
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

void VolumePlugin::addObject(const RenderObject *container, osg::Group *, const RenderObject *geometry, const RenderObject *, const RenderObject *colorObj, const RenderObject *)
{
    vvDebugMsg::msg(1, "VolumePlugin::VRAddObject()");
    int shader = -1;

    // colorMap is not passed as parameter..
    size_t MaxColorMap = 8;
    std::vector<RenderObject *> colorMap(MaxColorMap);
    for (int c = 0; c < colorMap.size(); ++c)
    {
        colorMap[c] = container->getColorMap(c);
    }

    if (container->getAttribute("VOLUME_SHADER"))
    {
        std::string s = container->getAttribute("VOLUME_SHADER");
        shader = atoi(s.c_str());
    }

    // Check if valid volume data was added:
    if (geometry && geometry->isUniformGrid())
    {
#ifdef VERBOSE
        fprintf(stderr, "add volume: %s\n", geometry->getName());
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

#ifdef VERBOSE
        cerr << "@ APP @@\n";
        cerr << "@ APP @@ Color object type is " << colorObj->getType() << endl;
#endif

        bool showEditor = true;
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
            }

            // add to timestep series if necessary
            if (container && container->getName() && volumes.find(container->getName()) != volumes.end())
            {
                volDesc->addFrame(data, vvVolDesc::ARRAY_DELETE);
                volDesc->frames = volDesc->getStoredFrames();
#ifdef VERBOSE
                fprintf(stderr, "added timestep to %s: %d steps\n", container->getName(), volDesc->frames);
#endif
            }
            else
            {
                volDesc = new vvVolDesc("COVISEXX",
                                        sizeZ, sizeY, sizeX, 1, 1, noChan, &data, vvVolDesc::ARRAY_DELETE);
                volDesc->pos = vvVector3(minZ+maxZ, -minY-maxY, minX+maxX) * .5f;
                volDesc->dist[0] = (maxZ - minZ) / sizeZ;
                volDesc->dist[1] = (maxY - minY) / sizeY;
                volDesc->dist[2] = (maxX - minX) / sizeX;
            }

            if (packedColor)
            {
                volDesc->tf[0].setDefaultColors(3, 0., 1.);
                volDesc->tf[0].setDefaultAlpha(0, 0., 1.);
            }

            for (size_t c = 0; c < volDesc->chan; ++c)
            {
                volDesc->real[c][0] = colorObj->getMin(c);
                volDesc->real[c][1] = colorObj->getMax(c);

                if (volDesc->real[c][1] == 0 && volDesc->real[c][0] == 0)
                    volDesc->real[c][1] = 1.0f;
            }

            // Append color maps as additional transfer functions
            for (int c = 0; c < MaxColorMap; ++c)
            {
                if (colorMap[c] && colorMap[c]->getNumElements() > 0)
                {
                    volDesc->tf.resize(volDesc->tf.size() + 1);
                    volDesc->real.push_back(virvo::vec2(0.0f, 1.0f));

                    const float* rgbax = colorMap[c]->getFloat((Field::Id)c);

                    for (int i = 0; i < colorMap[c]->getNumElements(); ++i)
                    {
                        float r = rgbax[i * 5];
                        float g = rgbax[i * 5 + 1];
                        float b = rgbax[i * 5 + 2];
                        float a = rgbax[i * 5 + 3];
                        volDesc->tf.back()._widgets.push_back(new vvTFPyramid(
                                vvColor(r, g, b),
                                true,       // has own color
                                a,          // opacity
                                i / 255.0f, // xpos
                                1 / 255.0f, // width bottom
                                1 / 255.0f  // width top
                                )
                            );
                    }
                }
            }

            if (container->getName())
                updateVolume(container->getName(), volDesc);
            else if (geometry && geometry->getName())
                updateVolume(geometry->getName(), volDesc);
            else
                updateVolume("Anonymous COVISE object", volDesc);
            if (currentVolume != volumes.end())
            {
                coVRPluginList::instance()->addNode(currentVolume->second.transform, container, this);
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
            editor->show();
    }
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

bool VolumePlugin::updateVolume(const std::string &name, vvVolDesc *vd, bool mapTF, const std::string &filename)
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
        volumes[name].removeFromScene();
        volumes.erase(name);
        if (volume == currentVolume)
        {
            currentVolume = volumes.end();
        }
        return true;
    }

    VolumeMap::iterator volume = volumes.find(name);
    if (volume == volumes.end())
    {
        volumes[name].addToScene();
        volumes[name].filename = filename;
        volumes[name].multiDimTF = vd->chan == 1;
        volumes[name].preIntegration = preintItem->getState();
        volumes[name].lighting = lightingItem->getState();
        volumes[name].mapTF = mapTF;
        if (volumes[name].multiDimTF)
        {
            volumes[name].tf.resize(1);
            if (vd->tf[0].isEmpty())
            {
                volumes[name].tf[0] = editor->getTransferFunc(0);
            }
            else
            {
                volumes[name].tf[0] = vd->tf[0];
            }
        }
        else
        {
            volumes[name].tf.resize(vd->chan);
            if (vd->tf.empty() || vd->tf.size() != vd->chan)
            {
                for (int i = 0; i < volumes[name].tf.size(); ++i)
                {
                    volumes[name].tf[i].setDefaultColors(4 + i, 0., 1.);
                    volumes[name].tf[i].setDefaultAlpha(0, 0., 1.);
                }
            }
            else
            {
                volumes[name].tf = vd->tf;
            }

            if (vd->channelWeights.size() != vd->chan)
            {
                vd->channelWeights.resize(vd->chan);
                std::fill(vd->channelWeights.begin(), vd->channelWeights.end(), 1.0f);
            }
            volumes[name].channelWeights = vd->channelWeights;
        }
        volume = volumes.find(name);
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
    if (mapTF) drawable->mapTransferFunctionsFrom01();
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
            alphaDefBlendItem->setState(backgroundColor == BgDefault);
            alphaDarkBlendItem->setState(backgroundColor == BgDark);
            alphaLightBlendItem->setState(backgroundColor == BgLight);
        }
        else
        {
            alphaDefBlendItem->setState(false);
            alphaDarkBlendItem->setState(false);
            alphaLightBlendItem->setState(false);
        }
        maxIntensityItem->setState(currentVolume->second.blendMode == virvo::VolumeDrawable::MaximumIntensity);
        minIntensityItem->setState(currentVolume->second.blendMode == virvo::VolumeDrawable::MinimumIntensity);

        std::string displayName = currentVolume->second.filename;
        std::string::size_type slash = displayName.rfind('/');
        if (slash != std::string::npos)
            displayName = displayName.substr(slash + 1);
        if (currentVolume->second.filename.empty())
            displayName = "[COVISE]";
        currentVolumeItem->setLabel(displayName);
    }
    else
    {
        currentVolumeItem->setLabel("(none)");
    }

    updateTFEData();
}

void VolumePlugin::updateTFEData()
{
    if (editor)
    {
        if (currentVolume != volumes.end())
        {
            tfApplyCBData.volume = &currentVolume->second;
            tfApplyCBData.drawable = currentVolume->second.drawable.get();
            if (tfApplyCBData.drawable)
            {
                vvVolDesc *vd = tfApplyCBData.drawable->getVolumeDescription();
                if (vd)
                {
                    if (computeHistogram)
                    {
                        size_t res[] = { TEXTURE_RES_BACKGROUND, TEXTURE_RES_BACKGROUND };
                        vvColor fg(1.0f, 1.0f, 1.0f);
                        vd->makeHistogramTexture(0, 0, 1, res, &tfeBackgroundTexture[0], vvVolDesc::VV_LINEAR, &fg, 0., 1.);
                        editor->updateBackground(&tfeBackgroundTexture[0]);
                    }

                    editor->setNumChannels(vd->chan);

                    editor->setTransferFuncs(currentVolume->second.tf);

                    for (int c = 0; c < vd->chan; ++c)
                    {
                        editor->setActiveChannel(c);
                        editor->setMin(vd->real[c][0]);
                        editor->setMax(vd->real[c][1]);
                    }

                    editor->setActiveChannel(currentVolume->second.curChannel);
                    tfApplyCBData.drawable->setTransferFunctions(editor->getTransferFuncs());
                    if (tfApplyCBData.volume->mapTF)
                        tfApplyCBData.drawable->mapTransferFunctionsFrom01();

                    tfApplyCBData.drawable->setChannelWeights(editor->getChannelWeights());
                    tfApplyCBData.drawable->setUseChannelWeights(editor->getUseChannelWeights());

                    instantMode = tfApplyCBData.drawable->getInstantMode();
                    editor->setInstantMode(instantMode);

                    //after loading data, if we have a tabletUI,
                    // 1) notify the tabletUI function editor of its dimensionality (channel number)
                    // 2) send it the histogram if the plugin is configured this way
                    functionEditorTab->setDimension(vd->chan);
                    delete[] functionEditorTab -> histogramData;
                    functionEditorTab->histogramData = NULL;

                    unsigned int buckets[2] = {
                        coTUIFunctionEditorTab::histogramBuckets,
                        vd->chan == 1 ? 1 : coTUIFunctionEditorTab::histogramBuckets
                    };
                    if (computeHistogram)
                    {
                        functionEditorTab->histogramData = new int[buckets[0] * buckets[1]];
                        if (vd->chan == 1)
                            vd->makeHistogram(0, 0, 1, buckets, functionEditorTab->histogramData, 0, 1);
                        else
                            vd->makeHistogram(0, 0, 2, buckets, functionEditorTab->histogramData, 0, 1);
                    }
                }
            }
        }
        editor->update();
    }
}

void VolumePlugin::preFrame()
{
    const float SPEED = 0.3f; // adaption speed
    float change;
    static bool firstFPSmeasurement = true;

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
        Vec3 centerWorld = center * cover->getBaseMat();
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
            drawable->setQuality(quality);
            drawable->setViewDirection(viewDirObj*t);
            drawable->setObjectDirection(objDirObj*t);
        }

        if (drawable)
        {
            typedef vvRenderState::ParameterType PT;

            int maxClipPlanes = drawable->getMaxClipPlanes();
            int numClipPlanes = 0;

            if (cover->isClippingOn())
            {
                StateSet *state = drawable->getOrCreateStateSet();
                ClipNode *cn = cover->getObjectsRoot();
                for (unsigned int i = 0; i < std::min((int)cn->getNumClipPlanes(), maxClipPlanes); ++i)
                {
                    ClipPlane *cp = cn->getClipPlane(i);
                    Vec4 v = cp->getClipPlane();

                    boost::shared_ptr<vvClipPlane> plane = vvClipPlane::create();
                    plane->normal = virvo::vec3(-v.z(), v.y(), -v.x());
                    plane->offset = v.w();

                    drawable->setParameter(PT(vvRenderState::VV_CLIP_OBJ0 + i), plane);
                    drawable->setParameter(PT(vvRenderState::VV_CLIP_OBJ_ACTIVE0 + i), true);
                    drawable->setParameter(PT(vvRenderState::VV_CLIP_OUTLINE0 + i), showClipOutlines);

                    state->setMode(GL_CLIP_PLANE0 + cp->getClipPlaneNum(), StateAttribute::OFF);

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
            if (coVRCollaboration::instance()->getSyncMode() != coVRCollaboration::MasterSlaveCoupling
                || coVRCollaboration::instance()->isMaster())
            {
                drawable->setROIPosition(currentVolume->second.roiPosObj);
                if (coVRCollaboration::instance()->getSyncMode() != coVRCollaboration::LooseCoupling)
                {
                    sendROIMessage(drawable->getROIPosition(), drawable->getROISize());
                }
            }
        }
    }
    if (interactionB->isRunning())
    {
        if (coVRCollaboration::instance()->getSyncMode() != coVRCollaboration::MasterSlaveCoupling
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
                if (drawable && coVRCollaboration::instance()->getSyncMode() != coVRCollaboration::LooseCoupling)
                {
                    sendROIMessage(drawable->getROIPosition(), drawable->getROISize());
                }
            }
        }
    }
    if (interactionA->wasStopped())
    {
        if (coVRCollaboration::instance()->getSyncMode() != coVRCollaboration::MasterSlaveCoupling
            || coVRCollaboration::instance()->isMaster())
        {
            if (!roiVisible())
            {
                cover->setButtonState("Region of Interest", 0);
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

            if (drawable && coVRCollaboration::instance()->getSyncMode() != coVRCollaboration::LooseCoupling)
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

void VolumePlugin::menuEvent(coMenuItem *item)
{
    vvDebugMsg::msg(2, "VolumePlugin::menuEvent()");

    for (VolumeMap::iterator it = allVolumesActive ? volumes.begin() : currentVolume; it != volumes.end(); ++it)
    {
        virvo::VolumeDrawable *drawable = it->second.drawable;
        if (!drawable)
            continue;

        if (item == clipModeItem.get())
        {
            drawable->setSingleSliceClipping(clipModeItem->getState());
        }

        else if (item == preintItem.get())
        {
            drawable->setPreintegration(preintItem->getState());
            it->second.preIntegration = preintItem->getState();
        }
        else if (item == lightingItem.get())
        {
            drawable->setLighting(lightingItem->getState());
            it->second.lighting = lightingItem->getState();
        }

        else if (item == boundItem.get())
        {
            drawable->setBoundaries(boundItem->getState());
            it->second.boundaries = boundItem->getState();
        }

        else if (item == interpolItem.get())
        {
            drawable->setInterpolation(interpolItem->getState());
            it->second.interpolation = interpolItem->getState();
        }

        else if (item == alphaDefBlendItem.get()
                 || item == alphaDarkBlendItem.get()
                 || item == alphaLightBlendItem.get()
                 || item == maxIntensityItem.get()
                 || item == minIntensityItem.get())
        {
            virvo::VolumeDrawable::BlendMode mode = virvo::VolumeDrawable::AlphaBlend;
            if (minIntensityItem->getState())
                mode = virvo::VolumeDrawable::MinimumIntensity;
            else if (maxIntensityItem->getState())
                mode = virvo::VolumeDrawable::MaximumIntensity;

            drawable->setBlendMode(mode);
            it->second.blendMode = mode;

            Vec4 bg(0., 0., 0., 1.);
            switch (mode)
            {
            case virvo::VolumeDrawable::AlphaBlend:
                if (alphaDefBlendItem->getState())
                {
                    backgroundColor = BgDefault;
                    bg[0] = covise::coCoviseConfig::getFloat("r", "COVER.Background", 0.f);
                    bg[1] = covise::coCoviseConfig::getFloat("g", "COVER.Background", 0.f);
                    bg[2] = covise::coCoviseConfig::getFloat("b", "COVER.Background", 0.f);
                }
                else if (alphaDarkBlendItem->getState())
                {
                    backgroundColor = BgDark;
                    bg[0] = bg[1] = bg[2] = 0.30f;
                }
                else if (alphaLightBlendItem->getState())
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
        }

        if (!allVolumesActive)
            break;
    }

    virvo::VolumeDrawable *drawable = getCurrentDrawable();

    if (item == ROIItem.get())
    {
        setROIMode(ROIItem->getState());
    }

    else if (item == clipOutlinesItem.get())
    {
        showClipOutlines = clipOutlinesItem->getState();
    }

    else if (item == clipSphereActive0Item.get())
    {
        clipSpheres.at(0)->setActive(clipSphereActive0Item->getState());
    }

    else if (item == clipSphereInteractorActive0Item.get())
    {
        clipSpheres.at(0)->setInteractorActive(clipSphereInteractorActive0Item->getState());
    }

    else if (item == clipSphereRadius0Item.get())
    {
        radiusScale[0] = clipSphereRadius0Item->getValue();
    }

    else if (item == clipSphereActive1Item.get())
    {
        clipSpheres.at(1)->setActive(clipSphereActive1Item->getState());
    }

    else if (item == clipSphereInteractorActive1Item.get())
    {
        clipSpheres.at(1)->setInteractorActive(clipSphereInteractorActive1Item->getState());
    }

    else if (item == clipSphereRadius1Item.get())
    {
        radiusScale[1] = clipSphereRadius1Item->getValue();
    }

    else if (item == clipSphereActive2Item.get())
    {
        clipSpheres.at(2)->setActive(clipSphereActive2Item->getState());
    }

    else if (item == clipSphereInteractorActive2Item.get())
    {
        clipSpheres.at(2)->setInteractorActive(clipSphereInteractorActive2Item->getState());
    }

    else if (item == clipSphereRadius2Item.get())
    {
        radiusScale[2] = clipSphereRadius2Item->getValue();
    }

    else if (item == fpsItem.get())
    {
        chosenFPS = fpsItem->getValue();
    }

    else if (item == colorsItem.get())
    {
        discreteColors = (int)colorsItem->getValue();
        editor->updateColorBar();
        editor->setDiscreteColors(discreteColors);
    }

    else if (item == tfeItem.get())
    {
        editor->show();
    }

    else if (item == hqItem.get())
    {
        highQualityOversampling = hqItem->getValue();
    }

    else if (item == allVolumesActiveItem.get())
    {
        allVolumesActive = allVolumesActiveItem->getState();
    }

    else if (item == cycleVolumeItem.get())
    {
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
    }

    else if (item == unloadItem.get())
    {
        if (currentVolume != volumes.end())
        {
            std::string filename = currentVolume->second.filename;
            if (!filename.empty())
                updateVolume(filename, NULL);
        }
    }
    else if (item == saveItem.get())
    {
        saveVolume();
    }
    else if (item == cropItem.get())
    {
        cropVolume();
    }
    else if (item == sideBySideItem.get())
    {
        if (sideBySideItem->getState())
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
    }
    else
    {
        bool isMaterial = false;

        // Otherwise assume it must be a file entry:
        if (!isMaterial)
        {
            for (list<FileEntry *>::iterator fe = fileList.begin(); fe != fileList.end(); ++fe)
            {
                if ((*fe)->fileMenuItem == item)
                {
                    cover->sendMessage(this,
                                       coVRPluginSupport::TO_SAME, PluginMessageTypes::VolumeLoadFile,
                                       strlen((*fe)->fileName) + 1, (*fe)->fileName);
                    break;
                }
            }
        }
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
        if (coVRCollaboration::instance()->getSyncMode() != coVRCollaboration::MasterSlaveCoupling
            || coVRCollaboration::instance()->isMaster())
        {
            roiMode = true;
        }
        else
        {
            cover->setButtonState("Region of Interest", 0);
        }
    }
    else
    {
        if (coVRCollaboration::instance()->getSyncMode() != coVRCollaboration::MasterSlaveCoupling
            || coVRCollaboration::instance()->isMaster())
        {
            roiCellSize = 0.0f;
            roiMode = false;
        }
        else
        {
            cover->setButtonState("Region of Interest", 1);
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

        if (coVRCollaboration::instance()->getSyncMode() != coVRCollaboration::MasterSlaveCoupling
            || coVRCollaboration::instance()->isMaster())
        {
            if (coVRCollaboration::instance()->getSyncMode() != coVRCollaboration::LooseCoupling)
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

COVERPLUGIN(VolumePlugin)
