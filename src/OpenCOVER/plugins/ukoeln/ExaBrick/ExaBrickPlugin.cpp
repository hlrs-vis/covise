/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: Hello OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include "ExaBrickPlugin.h"
#include "coDefaultFunctionEditor.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <config/CoviseConfig.h>

#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Action.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Container.h>
#include <cover/ui/Element.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Owner.h>
#include <cover/ui/TextField.h>

#include <cover/coVRFileManager.h>
#include <cover/coVRConfig.h>
#include <cover/RenderObject.h>
// #include <cover/VRViewer.h>
#include "owl/owl.h"
#include "programs/VolumeData.h"
#include "exa/ColorMapper.h"
#include "exa/mat4.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "submodules/3rdParty/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION 1
#include "submodules/3rdParty/stb_image.h"
#include "exa/embedded_colormaps.h"

#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
// #include "coDefaultFunctionEditor.h"
#include <chrono>
#include <thread>


using namespace opencover;
using namespace exa;
class coDefaultFunctionEditor;

ExaBrickPlugin *ExaBrickPlugin::plugin = nullptr;

static FileHandler TextHandler = {
    NULL,
    ExaBrickPlugin::loadFile,
    NULL,
    "exa" 
};

// some small tool functions
math::mat4f osg_cast(osg::Matrixd const &m)
{
  float arr[16];
  std::copy(m.ptr(), m.ptr() + 16, arr);
  return math::mat4f(arr);
}

int notsame(math::mat4f a, math::mat4f b){
   int i, j;
   for (i = 0; i < 4; i++)
      for (j = 0; j < 4; j++)
         if (a(i,j) != b(i,j))
      return 1;
   return 0;
}

void osg_print(osg::Matrixd const &mat){
  for (int m = 0; m < 4; m++) {
    printf("  [");
    for (int n = 0; n < 4; n++) {
      printf("%f ",mat(n,m));
    }
    printf("]\n");
  }
}

static std::vector<std::string> string_split(std::string s, char delim)
{
    std::vector<std::string> result;

    std::istringstream stream(s);

    for (std::string token; std::getline(stream, token, delim); )
    {
        result.push_back(token);
    }

    return result;
}


/*! tell renderer to re-start frame accumulatoin; this _has_ to be
  called every time something changes that would change the
  converged image to be rendererd (eg, change to transfer
  runctoin, camera, etc) */
void ExaBrickPlugin::resetAccumulation()
{
  plugin->accumID =0;
}

ExaBrickPlugin *ExaBrickPlugin::instance()
{
    return plugin;
}

ExaBrickPlugin::ExaBrickPlugin()
: ui::Owner("ExaBrickPlugin",cover->ui)
, editor(NULL)
//, coVRPlugin(COVER_PLUGIN_NAME)  //for new covise
{
  fprintf(stderr, "ExaBrickPlugin::ExaBrickPlugin\n");
  //register filehandler
  fprintf(stderr, "registering file handler\n");
  coVRFileManager::instance()->registerFileHandler(&TextHandler);
  // VRViewer::instance()->setRenderToTexture(true);
}

void ExaBrickPlugin::usage(const std::string &msg)
{
  if (msg != "")
    std::cout << "Error: " << msg << std::endl << std::endl;
  // std::cout << "usage : ./exaViewer path/to/configFile.exa" << std::endl;
  // std::cout << "--camera pos.x pos.y pos.z at.x at.y at.z up.x up.y up.z" << std::endl;
  // std::cout << "--size windowSize.x windowSize.y" << std::endl;
  std::cout << std::endl;
  exit((msg == "") ? 0 : 1);
}

bool ExaBrickPlugin::init()
{
  cout<< "calling init function"<<"\n";
  if (cover->debugLevel(1)) fprintf(stderr, "\n    ExaBrickPlugin::init\n");

  if (plugin) return false;

  plugin = this;

  tfApplyCBData.tfe = NULL;

  // editor = new coDefaultFunctionEditor(applyDefaultTransferFunction, &tfApplyCBData);
  // editor->setSaveFunc(saveDefaultTransferFunction);

  // tfApplyCBData.tfe = editor;

  return true;
}//init()

void ExaBrickPlugin::setOpacityScale(float xfos){
  plugin->cmdline.xfOpacityScale = xfos;
  plugin->updateTransferFunction();
}

void ExaBrickPlugin::updateTransferFunction(){
  if (!plugin->renderer)
  return
  // reset Accumulation
  plugin->resetAccumulation();
  // update Transferfunction
  for (size_t c=0;c<plugin->renderer->scalarFields.size();++c) {
    for (auto f : plugin->cmdline.xfAlpha[c]) std::cout << f << ' ';
    plugin->renderer->updateXF(c,
                        plugin->cmdline.xfAlpha[c].data(),
                        plugin->xfColor[plugin->xfColorMap[c]],
                        plugin->xfDomain[c],
                        plugin->cmdline.xfOpacityScale);

  }
}

void ExaBrickPlugin::applyDefaultTransferFunction(void *userData)
{
    TFApplyCBData *data = (TFApplyCBData *)userData;

    // if (data && data->drawable && data->tfe)
    // {
    //     coDefaultFunctionEditor *tfe = data->tfe;
    //     for (VolumeMap::iterator it = volumes.begin();
    //          it != volumes.end();
    //          it++)
    //     {
    //         if (it->second.drawable == data->drawable || allVolumesActive)
    //         {
    //             vvVolDesc *vd = it->second.drawable.get()->getVolumeDescription();

    //             it->second.tf = tfe->getTransferFuncs();
    //             it->second.drawable->setTransferFunctions(it->second.tf);
    //             if (it->second.mapTF)
    //                 mapTFToMinMax(it, vd);
                  
    //             it->second.useChannelWeights = tfe->getUseChannelWeights();
    //             it->second.channelWeights = tfe->getChannelWeights();
    //             it->second.drawable->setChannelWeights(it->second.channelWeights);
    //             it->second.drawable->setUseChannelWeights(it->second.useChannelWeights);
    //         }
    //     }
    // }
}

 int ExaBrickPlugin::loadFile(const char *filename, osg::Group *loadParent, const char *covise_key)
{ 
  #if 1
  cout<< "calling loadFile function"<<"\n";
  fprintf(stderr, "ExaBrickPlugin::loadFile(%s,%p,%s)\n", filename, loadParent, covise_key);
  
  plugin->cmdline.exaFileName = filename;

  plugin->config = Config::parseConfigFile(filename);
  if (!plugin->config)
    usage("no config file specified");
  
  if (!plugin->config->bricks.sp)
    usage("no bricks file specified");

  ExaBricks::SP input = plugin->config->bricks.sp;//ExaBricks::load(argv[1],{argv[2]});
  // const box3f bounds = plugin->config->getBounds(); //The bounds of bricks is: [(0,0,0):(8,8,16)
  // std::cout<< "The lower bounds of bricks is: "<< bounds.lower <<"\n";
  // std::cout<< "The upper x bounds of bricks is: "<< bounds.upper.x <<"\n";

  // plugin->volumeData.sizeX = bounds.upper.x;
  // plugin->volumeData.sizeY = bounds.upper.y;
  // plugin->volumeData.sizeZ = bounds.upper.z;
  printf("plugin->config->scalarField.valueRange: (%f,%f)\n",plugin->config->scalarFields[0]->valueRange.lower,plugin->config->scalarFields[0]->valueRange.upper);

  std::vector<TriangleMesh::SP> surfaces = plugin->config->surfaces;
  size_t numVerts = 0;
  size_t numIndices = 0;
  for (const auto &s : surfaces) {
      numVerts += s->vertex.size();
      numIndices += s->index.size();
  }
  if (numIndices != 0) {
      std::cout << "Loaded mesh consisting of "
                << owl::prettyDouble(numIndices) << " triangles\n"
                << "Vertices Memory: "
                << owl::prettyNumber(numIndices * sizeof(vec3i)) << " bytes\n"
                << "Indices Memory: "
                << owl::prettyNumber(numIndices * sizeof(vec3f)) << " bytes\n";
  }
    /* ==================================================================
    create the optix renderer 
    ================================================================== */
  if (!plugin->renderer && plugin->config){
    cout<<"creating renderer\n";
    plugin->renderer
    = std::make_shared<OptixRenderer>(plugin->config->bricks.sp,
                                    plugin->config->surfaces,
                                    plugin->config->scalarFields);    
    // std::cout << "The size of OptixRenderer is: " << sizeof(exa::OptixRenderer) << "\n";
    plugin->renderer->setVoxelSpaceTransform(plugin->config->bricks.voxelSpaceTransform);
    const box3f bounds =plugin->renderer->regionDomain;
    // const box3f bounds = plugin->config->getBounds(); //The bounds of bricks is: [(0,0,0):(8,8,16)]
    std::cout << "Regions domain bounds upper: " << bounds.upper << "\n";
    std::cout << "Regions domain bounds lower: " << bounds.lower << "\n";
    plugin->volumeData.sizeU = bounds.lower.x;
    plugin->volumeData.sizeV = bounds.lower.y;
    plugin->volumeData.sizeW = bounds.lower.z;
    plugin->volumeData.sizeX = bounds.upper.x;
    plugin->volumeData.sizeY = bounds.upper.y;
    plugin->volumeData.sizeZ = bounds.upper.z;

    //init viewer settings
    // exa::ColorMapper colorMapper;
    ColorMapper* customColorMap = nullptr;
    // ColorMapper cm = *plugin->customColorMap;
    plugin->xfColorMap.resize(plugin->renderer->scalarFields.size());
    plugin->xfDomain.resize(plugin->renderer->scalarFields.size());

    std::vector<ColorMapper> colormaps = {
        ColorMapper(paraview_cool_warm, sizeof(paraview_cool_warm)),
        ColorMapper(rainbow, sizeof(rainbow)),
        ColorMapper(matplotlib_plasma, sizeof(matplotlib_plasma)),
        ColorMapper(matplotlib_virdis, sizeof(matplotlib_virdis)),
        ColorMapper(samsel_linear_green, sizeof(samsel_linear_green)),
        ColorMapper(samsel_linear_ygb_1211g, sizeof(samsel_linear_ygb_1211g)),
        ColorMapper(cool_warm_extended, sizeof(cool_warm_extended)),
        ColorMapper(blackbody, sizeof(blackbody)),
        ColorMapper(jet, sizeof(jet)),
        ColorMapper(blue_gold, sizeof(blue_gold)),
        ColorMapper(ice_fire, sizeof(ice_fire)),
        ColorMapper(nic_edge, sizeof(nic_edge)),
        ColorMapper(exa::covise, sizeof(exa::covise)),
        ColorMapper(jamie_draft, sizeof(jamie_draft)),
        ColorMapper(hsv, sizeof(hsv)),
        ColorMapper({vec3f(0,0,0), vec3f(1,1,1)})
    };

    // ==================================================================
    // set up transfer function
    // ==================================================================
    int XF_ALPHA_COUNT = INITIAL_XF_ALPHA_COUNT;
    plugin->xfColor.resize(colormaps.size());//16
    printf("XF_ALPHA_COUNT: %d\n",XF_ALPHA_COUNT);
    for (size_t i=0;i<plugin->xfColor.size();i++) {
      plugin->xfColor[i].resize(XF_ALPHA_COUNT);
    }


    for (int i=0;i<XF_ALPHA_COUNT;i++) {
      float t=i/(float)(XF_ALPHA_COUNT-1);
      for (size_t j = 0; j < colormaps.size(); ++j) {
          plugin->xfColor[j][i] = colormaps[j](t);
      }
    }
        // to print the xfColor
        // for (size_t i = 0; i < plugin->xfColor.size(); ++i) {
        //     for (size_t j = 0; j < plugin->xfColor[i].size(); ++j) {
        //         std::cout << "plugin->xfColor[" << i << "][" << j << "]: "
        //                   << plugin->xfColor[i][j].x << ", "
        //                   << plugin->xfColor[i][j].y << ", "
        //                   << plugin->xfColor[i][j].z << std::endl;
        //     }
        // }
    if (plugin->cmdline.xfAlpha.empty()) {
      plugin->cmdline.xfAlpha.resize(plugin->renderer->scalarFields.size());
      for (size_t c=0;c<plugin->renderer->scalarFields.size();++c) {
        for (int i=0;i<XF_ALPHA_COUNT;i++) {
          float t=i/(float)(XF_ALPHA_COUNT-1);
          // plugin->cmdline.xfAlpha[c][i] = 1;
          plugin->cmdline.xfAlpha[c][i] = t;
        }
      }
    }
          // for (size_t i = 0; i <  plugin->cmdline.xfAlpha.size(); ++i) {
          //     for (size_t j = 0; j <  plugin->cmdline.xfAlpha[i].size(); ++j) {
          //         std::cout << "cmdline.xfAlpha[" << i << "][" << j << "]: "
          //                   <<  plugin->cmdline.xfAlpha[i][j] << std::endl;
          //     }
          // }
    cout << "plugin->renderer->scalarFields.size(): " << plugin->renderer->scalarFields.size() << std::endl;
    // update Transferfunction
    for (size_t c=0;c<plugin->renderer->scalarFields.size();++c) {
        // plugin->xfColorMap[c] = 0;
        // printf("plugin->renderer->scalarFields: (%f,%f)\n",plugin->renderer->scalarFields[c]->valueRange.lower,plugin->renderer->scalarFields[c]->valueRange.upper);
        plugin->xfDomain[c] = plugin->renderer->scalarFields[c]->valueRange; 
        plugin->renderer->updateXF(c,
                           plugin->cmdline.xfAlpha[c].data(),
                           plugin->xfColor[plugin->xfColorMap[c]],
                           plugin->xfDomain[c],
                           plugin->cmdline.xfOpacityScale);

        // for (int i = 0; i < plugin->xfColor[plugin->xfColorMap[c]].size(); ++i) {
        // printf("plugin->xfColor[xfColorMap[c]][%d]: %f %f %f %f\n",
        // i,
        //   plugin->xfColor[plugin->xfColorMap[c]][i].x,
        //     plugin->xfColor[plugin->xfColorMap[c]][i].y,
        //     plugin->xfColor[plugin->xfColorMap[c]][i].z,
        //       plugin->xfColor[plugin->xfColorMap[c]][i].w);
        // }
        std::cout << "plugin->xfColorMap[c]: "<<plugin->xfColorMap[c] << std::endl;
    }

      // for (size_t i = 0; i < plugin->cmdline.xfAlpha.size(); ++i) {
      //     for (size_t j = 0; j < plugin->cmdline.xfAlpha[i].size(); ++j) {
      //         std::cout << "plugin->cmdline.xfAlpha[" << i << "][" << j << "]: "
      //                   << plugin->cmdline.xfAlpha[i][j] << std::endl;
      //     }
      // }
        
    // ==================================================================
    // iso plane(s) editor
    // ==================================================================
    for (int i=0;i<MAX_ISO_SURFACES;i++) {
          plugin->isoSurfaceEnabled[i] = 0;
          plugin->isoSurfaceValue[i]
              = plugin->xfDomain[plugin->isoChannel].lower
              + ((i+1)/(MAX_ISO_SURFACES+1.f))*(plugin->xfDomain[plugin->isoChannel].upper-plugin->xfDomain[plugin->isoChannel].lower);
      if (plugin->cmdline.isochannels.size() > i) {
        plugin->isoSurfaceChannel[i] = plugin->cmdline.isochannels[i];
      }
      plugin->isoSurfaceChannel[i] = 0; //for the init now
    }
    plugin->renderer->updateIsoValues(plugin->isoSurfaceValue,
                              plugin->isoSurfaceChannel,
                              plugin->isoSurfaceEnabled);

    // ==================================================================
    // set up clipbox, ao and clockScale
    // ==================================================================
    plugin->renderer->frameState.clipBox.enabled = plugin->cmdline.clipBox.enabled;
    const box3f wsbounds = plugin->renderer->worldSpaceBounds;
    plugin->renderer->frameState.clipBox.coords.lower
      = vec3f(wsbounds.lower) + plugin->cmdline.clipBox.coords.lower * vec3f(wsbounds.span());
    plugin->renderer->frameState.clipBox.coords.upper
      = vec3f(wsbounds.lower) + plugin->cmdline.clipBox.coords.upper * vec3f(wsbounds.span());

    plugin->renderer->frameState.ao.enabled = plugin->cmdline.ao.enabled;
    plugin->renderer->frameState.ao.length = plugin->cmdline.ao.length;

    plugin->renderer->frameState.clockScale = plugin->cmdline.clockScale;

    // for (size_t c=0;c<plugin->renderer->scalarFields.size();++c) {
    //   // xfColorMap[c] = 0;
    //   // xfDomain[c] = renderer->scalarFields[c]->valueRange;
    //   const float opacity = 0.000000;
    //   plugin->renderer->updateXF(0, //c,
    //                       plugin->cmdline.xfAlpha[c].data(), //xfAlpha[c].data(),
    //                       plugin->xfDomain[0], //xfDomain[c],
    //                       1);//cmdline.xfOpacityScale
    // }

    // init channelInfos
    // int numChannels = coVRConfig::instance()->numChannels();
    for (size_t i=0; i<plugin->channelInfos.size(); ++i) {
      // Causes frame re-initialization
      plugin->channelInfos[i].width = 1;
      plugin->channelInfos[i].height = 1;
      // plugin->channelInfos[i].mv = osg::Matrix::identity();
      // plugin->channelInfos[i].pr = osg::Matrix::identity();
      plugin->channelInfos[i].mv = math::mat4f::identity();
      plugin->channelInfos[i].pr = math::mat4f::identity();
    }

    // init bgcolor
    float r = coCoviseConfig::getFloat("r", "COVER.Background", 0.0f);
    float g = coCoviseConfig::getFloat("g", "COVER.Background", 0.0f);
    float b = coCoviseConfig::getFloat("b", "COVER.Background", 0.0f);
    // float bgcolor[] = {r,g,b,1.f};
    // float bgcolor[] = {r,g,b};
    owl::vec3f bgcolor(r,g,b);
    plugin->renderer->setBgColor(bgcolor);
    int numChannels = coVRConfig::instance()->numChannels();
  } 
  // plugin->renderer->setGradientShadingDVR(1);
  //     plugin->renderer->setGradientShadingISO(1);

  // ==================================================================
  // set up ui menu
  // ==================================================================

  // set up menu Transfer Function
  plugin->tf_menu = new ui::Menu("Transfer Function", plugin);

  auto opacityScaleSlider = new ui::Slider(plugin->tf_menu, "OpacityScale");
  opacityScaleSlider->setText("Opacity Scale");
  opacityScaleSlider->setBounds(0., 1.);
  opacityScaleSlider->setValue(plugin->cmdline.xfOpacityScale);
  opacityScaleSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          // plugin->setOpacityScale(value);
          plugin->cmdline.xfOpacityScale = value;
          plugin->updateTransferFunction();
      }
  });

  // xfDomain upper and lower
  auto domainLowerSlider = new ui::Slider(plugin->tf_menu, "domainLower");
  domainLowerSlider->setText("xfDomain Lower");
  domainLowerSlider->setBounds(-100., 100.);
  if (plugin->renderer != NULL){
    for (size_t c=0;c<plugin->renderer->scalarFields.size();++c) {
      cout<<"lower: "<<plugin->xfDomain[c].lower<<"\n";
      domainLowerSlider->setValue(plugin->xfDomain[c].lower);
    }
  }
  domainLowerSlider->setCallback([plugin](double value, bool released){
    if (plugin->renderer != NULL){
      for (size_t c=0;c<plugin->renderer->scalarFields.size();++c) {
        plugin->xfDomain[c].lower = value; 
      }
      plugin->updateTransferFunction();
    }
  });

  auto domainUpperSlider = new ui::Slider(plugin->tf_menu, "domainUpper");
  domainUpperSlider->setText("xfDomain Upper");
  domainUpperSlider->setBounds(-100., 100.);
  if (plugin->renderer != NULL){
    for (size_t c=0;c<plugin->renderer->scalarFields.size();++c) {
      cout<<"upper: "<<plugin->xfDomain[c].upper<<"\n";
      domainUpperSlider->setValue(plugin->xfDomain[c].upper);
    }
  }
  domainUpperSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          for (size_t c=0;c<plugin->renderer->scalarFields.size();++c) {
            plugin->xfDomain[c].upper = value; 
          }
          plugin->updateTransferFunction();
      }
  });

  // Embedded colormaps
  const static std::vector<std::string> colormapNames = {
      "Paraview Cool Warm",
      "Rainbow",
      "Matplotlib Plasma",
      "Matplotlib Virdis",
      "Samsel Linear Green",
      "Samsel Linear YGB 1211g",
      "Cool Warm Extended",
      "Blackbody",
      "Jet",
      "Blue Gold",
      "Ice Fire",
      "Nic Edge",
      "Covise",
      "JamieDraft",
      "HSV",
      "Custom"
  };
  auto xfColorMapList = new ui::SelectionList(plugin->tf_menu, "xfColorMap");
  xfColorMapList->setText("xfColorMap: ");
  for (const auto& colormap : colormapNames) {
    xfColorMapList->append(colormap);
  }
  xfColorMapList->setCallback([plugin](int idx){
      if (plugin->renderer != NULL){
          for (size_t c=0;c<plugin->renderer->scalarFields.size();++c) {
            plugin->xfColorMap[c] = idx;
          }
          plugin->updateTransferFunction();
      }
  });

  auto channelList = new ui::SelectionList(plugin->tf_menu, "channel");
  channelList->setText("Channel: density");
  channelList->append("density");

  //// set up menu Render Settings
  plugin->rs_menu = new ui::Menu("Render Settings", plugin);

  // AO
  auto aoEditField = new ui::EditField(plugin->rs_menu, "AO");

  auto aoEnabledButton = new ui::Button(plugin->rs_menu, "enabled?");
  aoEnabledButton->setState(false);
  aoEnabledButton->setCallback([plugin](bool state){
    plugin->cmdline.ao.enabled = state;
    plugin->renderSettingsChangedCB();
  });

  auto ao_length = new ui::Slider(plugin->rs_menu, "ao_length");
  ao_length->setText("Length");
  ao_length->setBounds(0., 2000.);
  ao_length->setValue(plugin->cmdline.ao.length);
  ao_length->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.ao.length = value;
          plugin->renderSettingsChangedCB();
      }
  });

  // Clock Scale
  auto cloackScaleEditField = new ui::EditField(plugin->rs_menu, "Clock Scale");

  auto clockScale = new ui::Slider(plugin->rs_menu, "clockScale");
  clockScale->setText("Clock Scale/ Timevis");
  clockScale->setBounds(0., 1000.);
  clockScale->setValue(plugin->cmdline.clockScale);
  clockScale->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.clockScale = value;
          plugin->renderSettingsChangedCB();
      }
  });

  // Clip Box
  auto clipBoxEditField = new ui::EditField(plugin->rs_menu, "Clip Box");

  auto clipBoxOnButton = new ui::Button(plugin->rs_menu, "Clip Box on?");
  clipBoxOnButton->setState(false);
  clipBoxOnButton->setCallback([plugin](bool state){
    plugin->cmdline.clipBox.enabled = state;
    plugin->renderSettingsChangedCB();
  });

  auto loxSlider = new ui::Slider(plugin->rs_menu, "loxSlider");
  loxSlider->setText("lo x");
  loxSlider->setBounds(-100., 100.);
  loxSlider->setValue(plugin->cmdline.clipBox.coords.lower.x);
  loxSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.clipBox.coords.lower.x = value;
          plugin->renderSettingsChangedCB();
      }
  });

  auto loySlider = new ui::Slider(plugin->rs_menu, "loySlider");
  loySlider->setText("lo y");
  loySlider->setBounds(-100., 100.);
  loySlider->setValue(plugin->cmdline.clipBox.coords.lower.y);
  loySlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.clipBox.coords.lower.y = value;
          plugin->renderSettingsChangedCB();
      }
  });

  auto lozSlider = new ui::Slider(plugin->rs_menu, "lozSlider");
  lozSlider->setText("lo z");
  lozSlider->setBounds(-100., 100.);
  lozSlider->setValue(plugin->cmdline.clipBox.coords.lower.z);
  lozSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.clipBox.coords.lower.z = value;
          plugin->renderSettingsChangedCB();
      }
  });

  auto hixSlider = new ui::Slider(plugin->rs_menu, "hixSlider");
  hixSlider->setText("hi x");
  hixSlider->setBounds(-100., 100.);
  hixSlider->setValue(plugin->cmdline.clipBox.coords.upper.x);
  hixSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.clipBox.coords.upper.x = value;
          plugin->renderSettingsChangedCB();
      }
  });

  auto hiySlider = new ui::Slider(plugin->rs_menu, "hiySlider");
  hiySlider->setText("hi y");
  hiySlider->setBounds(-100., 100.);
  hiySlider->setValue(plugin->cmdline.clipBox.coords.upper.y);
  hiySlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.clipBox.coords.upper.y = value;
          plugin->renderSettingsChangedCB();
      }
  });

  auto hizSlider = new ui::Slider(plugin->rs_menu, "hizSlider");
  hizSlider->setText("hi z");
  hizSlider->setBounds(-100., 100.);
  hizSlider->setValue(plugin->cmdline.clipBox.coords.upper.z);
  hizSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.clipBox.coords.upper.z = value;
          plugin->renderSettingsChangedCB();
      }
  });

  // auto resetRSAction = new ui::Slider(plugin->rs_menu, "Reset Render Settings");
  // resetRSAction->setCallback([plugin](){
  //   if (plugin->renderer != NULL){
  //     // clipBoxOnButton->setState(false);
  //     // loxSlider->setValue(0);
  //     // loySlider->setValue(0);
  //     // lozSlider->setValue(0);
  //     // hixSlider->setValue(1);
  //     // hiySlider->setValue(1);
  //     // hizSlider->setValue(1);
  //     plugin->renderSettingsChangedCB();
  //   }
  // });

  // set up menu ISO Surfaces
  plugin->is_menu = new ui::Menu("ISO Surfaces", plugin);

  for (int i=0;i<MAX_ISO_SURFACES;i++) {
    auto enabledButton = new ui::Button(plugin->is_menu, "enabled?");
    enabledButton->setState(false);
    enabledButton->setCallback([plugin, i](bool state){
      if (state){
        plugin->isoSurfaceEnabled[i]=1;}
      else{
        plugin->isoSurfaceEnabled[i]=0;}
      plugin->isoSurfacesChangedCB();
    });

    auto isoValueSlider = new ui::Slider(plugin->is_menu, "isoValueSlider");
    isoValueSlider->setText("Value");
    isoValueSlider->setBounds(plugin->xfDomain[0].lower,
		    	                    plugin->xfDomain[0].upper);
    isoValueSlider->setValue(plugin->isoSurfaceValue[i]);
    isoValueSlider->setCallback([plugin, i](double value, bool released){
        if (plugin->renderer != NULL){
            plugin->isoSurfaceValue[i] = value;
            plugin->isoSurfacesChangedCB();
        }
    });

    auto isoChannelList = new ui::SelectionList(plugin->is_menu, "isoChannel");
    isoChannelList->setText("Channel: density");
    isoChannelList->append("density");
  }

  // set up menu Contour Planes
  plugin->cp_menu = new ui::Menu("Contour Planes", plugin);
  for (int i=0;i<MAX_CONTOUR_PLANES;i++) {
    if (plugin->cmdline.contourplanes.size() > i) {
      plugin->contourPlaneEnabled[i] = 1;
      plugin->contourPlaneNormal[i] = vec3f(plugin->cmdline.contourplanes[i]);
      plugin->contourPlaneOffset[i] = plugin->cmdline.contourplanes[i].w;
      plugin->contourPlaneChannel[i] = 0;
    } else {
      plugin->contourPlaneEnabled[i] = 0;
      plugin->contourPlaneNormal[i] = vec3f(0.f);
      plugin->contourPlaneNormal[i][i%3] = 1.f;
      plugin->contourPlaneOffset[i] = .5f;
    }

    if (plugin->cmdline.contourchannels.size() > 1)
      plugin->contourPlaneChannel[i] = plugin->cmdline.contourchannels[i];
    else
      plugin->contourPlaneChannel[i] = 0;

    auto contourPlaneEnabledButton = new ui::Button(plugin->cp_menu, "enabled?");
    contourPlaneEnabledButton->setState(false);
    contourPlaneEnabledButton->setCallback([plugin, i](bool state){
      if (state){
        plugin->contourPlaneEnabled[i]=1;}
      else{
        plugin->contourPlaneEnabled[i]=0;}
      plugin->contourPlanesChangedCB();
    });
    
    auto cpNormalXSlider = new ui::Slider(plugin->cp_menu, "cpNormalXSlider");
    cpNormalXSlider->setText("Normal x");
    cpNormalXSlider->setBounds(-5., 5.);
    cpNormalXSlider->setValue(plugin->contourPlaneNormal[i][0]);
    cpNormalXSlider->setCallback([plugin, i](double value, bool released){
        if (plugin->renderer != NULL){
            plugin->contourPlaneNormal[i][0] = value;
            plugin->contourPlanesChangedCB();
        }
    });

    auto cpNormalYSlider = new ui::Slider(plugin->cp_menu, "cpNormalYSlider");
    cpNormalYSlider->setText("Normal y");
    cpNormalYSlider->setBounds(-5., 5.);
    cpNormalYSlider->setValue(plugin->contourPlaneNormal[i][1]);
    cpNormalYSlider->setCallback([plugin, i](double value, bool released){
        if (plugin->renderer != NULL){
            plugin->contourPlaneNormal[i][1] = value;
            plugin->contourPlanesChangedCB();
        }
    });

    auto cpNormalZSlider = new ui::Slider(plugin->cp_menu, "cpNormalZSlider");
    cpNormalZSlider->setText("Normal z");
    cpNormalZSlider->setBounds(-5., 5.);
    cpNormalZSlider->setValue(plugin->contourPlaneNormal[i][2]);
    cpNormalZSlider->setCallback([plugin, i](double value, bool released){
        if (plugin->renderer != NULL){
            plugin->contourPlaneNormal[i][2] = value;
            plugin->contourPlanesChangedCB();
        }
    });

    auto cpOffsetSlider = new ui::Slider(plugin->cp_menu, "cpOffsetSlider");
    cpOffsetSlider->setText("Offset");
    cpOffsetSlider->setBounds(-5., 5.);
    cpOffsetSlider->setValue(plugin->contourPlaneOffset[i]);
    cpOffsetSlider->setCallback([plugin, i](double value, bool released){
        if (plugin->renderer != NULL){
            plugin->contourPlaneOffset[i] = value;
            plugin->contourPlanesChangedCB();
        }
    });

    auto cpChannelList = new ui::SelectionList(plugin->cp_menu, "cpChannelList");
    cpChannelList->setText("Channel: density");
    cpChannelList->append("density");
  }

  // set up menu Tracer
  plugin->tr_menu = new ui::Menu("Tracer", plugin);

  auto trEnabledButton = new ui::Button(plugin->tr_menu, "enabled?");
  trEnabledButton->setState(false);
  trEnabledButton->setCallback([plugin](bool state){
    if (state){
      plugin->cmdline.traces.enabled=1;}
    else{
      plugin->cmdline.traces.enabled=0;}
    plugin->tracerSettingsChangedCB();
  });

  auto trChannelXList = new ui::SelectionList(plugin->tr_menu, "trChannelXList");
  trChannelXList->setText("Channel 0: density");
  trChannelXList->append("density");

  auto trChannelYList = new ui::SelectionList(plugin->tr_menu, "trChannelYList");
  trChannelYList->setText("Channel 1: density");
  trChannelYList->append("density");

  auto trChannelZList = new ui::SelectionList(plugin->tr_menu, "trChannelZList");
  trChannelZList->setText("Channel 2: density");
  trChannelZList->append("density");

  auto numSeedsSlider = new ui::Slider(plugin->tr_menu, "numSeedsSlider");
  numSeedsSlider->setText("Number of Seeds");
  numSeedsSlider->setBounds(0., 2000.);
  numSeedsSlider->setValue(plugin->cmdline.traces.numTraces);
  numSeedsSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.traces.numTraces = value;
          plugin->contourPlanesChangedCB();
      }
  });

  auto timeStepsSlider = new ui::Slider(plugin->tr_menu, "timeStepsSlider");
  timeStepsSlider->setText("Time Steps");
  timeStepsSlider->setBounds(-0., 1000.);
  timeStepsSlider->setValue(plugin->cmdline.traces.numTimesteps);
  timeStepsSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.traces.numTimesteps = value;
          plugin->contourPlanesChangedCB();
      }
  });

  auto stepSizeSlider = new ui::Slider(plugin->tr_menu, "stepSizeSlider");
  stepSizeSlider->setText("Step Size");
  stepSizeSlider->setBounds(1e-8f, 1e+8f);
  stepSizeSlider->setValue(plugin->cmdline.traces.steplen);
  stepSizeSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.traces.steplen = value;
          plugin->contourPlanesChangedCB();
      }
  });

  auto seedRegionEditField = new ui::EditField(plugin->tr_menu, "Seed Region");

  auto trSRloXSlider = new ui::Slider(plugin->tr_menu, "trSRloXSlider");
  trSRloXSlider->setText("Lower x");
  trSRloXSlider->setBounds(-5., 5.);
  trSRloXSlider->setValue(plugin->cmdline.traces.seedRegion.lower.x);
  trSRloXSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.traces.seedRegion.lower.x = value;
          plugin->tracerSettingsChangedCB();
      }
  });

  auto trSRloYSlider = new ui::Slider(plugin->tr_menu, "trSRloYSlider");
  trSRloYSlider->setText("Number of Seeds");
  trSRloYSlider->setBounds(-5., 5.);
  trSRloYSlider->setValue(plugin->cmdline.traces.seedRegion.lower.y);
  trSRloYSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.traces.seedRegion.lower.y = value;
          plugin->tracerSettingsChangedCB();
      }
  });

  auto trSRloZSlider = new ui::Slider(plugin->tr_menu, "trSRloZSlider");
  trSRloZSlider->setText("Number of Seeds");
  trSRloZSlider->setBounds(-5., 5.);
  trSRloZSlider->setValue(plugin->cmdline.traces.seedRegion.lower.z);
  trSRloZSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.traces.seedRegion.lower.z = value;
          plugin->tracerSettingsChangedCB();
      }
  });

  auto trSRhiXSlider = new ui::Slider(plugin->tr_menu, "trSRhiXSlider");
  trSRhiXSlider->setText("Upper x");
  trSRhiXSlider->setBounds(-5., 5.);
  trSRhiXSlider->setValue(plugin->cmdline.traces.seedRegion.upper.x);
  trSRhiXSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.traces.seedRegion.upper.x = value;
          plugin->tracerSettingsChangedCB();
      }
  });

  auto trSRhiYSlider = new ui::Slider(plugin->tr_menu, "trSRhiYSlider");
  trSRhiYSlider->setText("Upper y");
  trSRhiYSlider->setBounds(-5., 5.);
  trSRhiYSlider->setValue(plugin->cmdline.traces.seedRegion.upper.y);
  trSRhiYSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.traces.seedRegion.upper.y = value;
          plugin->tracerSettingsChangedCB();
      }
  });

  auto trSRhiZSlider = new ui::Slider(plugin->tr_menu, "trSRhiZSlider");
  trSRhiZSlider->setText("Upper z");
  trSRhiZSlider->setBounds(-5., 5.);
  trSRhiZSlider->setValue(plugin->cmdline.traces.seedRegion.upper.z);
  trSRhiZSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL){
          plugin->cmdline.traces.seedRegion.upper.z = value;
          plugin->tracerSettingsChangedCB();
      }
  });

  // set up menu Other
  // plugin->other_menu = new ui::Menu("SS/PR/DVR/ISO", plugin);
  plugin->other_menu = new ui::Menu("Other Settings", plugin);

  auto spaceSkippingEnabledButton = new ui::Button(plugin->other_menu, "Enable Space Skipping");
  spaceSkippingEnabledButton->setState(true);
  spaceSkippingEnabledButton->setCallback([plugin](bool state){
    if (state){
      plugin->doSpaceSkipping=1;}
    else{
      plugin->doSpaceSkipping=0;}
    plugin->spaceSkippingChangedCB();
  });

  auto progressiveRefinementButton = new ui::Button(plugin->other_menu, "Progressive Refinement");
  progressiveRefinementButton->setState(true);
  progressiveRefinementButton->setCallback([plugin](bool state){
    if (state){
      plugin->cmdline.doProgressiveRefinement=1;}
    else{
      plugin->cmdline.doProgressiveRefinement=0;}
    plugin->progressiveRefinementCB();
  });

  auto gradientShadingDVRButton = new ui::Button(plugin->other_menu, "Enable Gradient Shading (DVR)");
  gradientShadingDVRButton->setState(true);
  gradientShadingDVRButton->setCallback([plugin](bool state){
    if (state){
      plugin->gradientShadingDVR=1;}
    else{
      plugin->gradientShadingDVR=0;}
    plugin->gradientShadingDVRChangedCB();
  });

  auto gradientShadingISOButton = new ui::Button(plugin->other_menu, "Enable Gradient Shading (ISO)");
  gradientShadingISOButton->setState(true);
  gradientShadingISOButton->setCallback([plugin](bool state){
    if (state){
      plugin->gradientShadingISO=1;}
    else{
      plugin->gradientShadingISO=0;}
    plugin->gradientShadingISOChangedCB();
  });

  auto autoFPSButton = new ui::Button(plugin->other_menu, "Enable Auto FPS");
  autoFPSButton->setState(true);
  autoFPSButton->setCallback([plugin](bool state){
    if (state){
      plugin->autoFPS=1;}
    else{
      plugin->autoFPS=0;}
    plugin->autoFPSChanged();
  });

  plugin->fpsSlider = new ui::Slider(plugin->other_menu, "FrameRate");
  plugin->fpsSlider->setText("Frame rate");
  plugin->fpsSlider->setBounds(5.0, 60.0);
  plugin->fpsSlider->setValue(plugin->cmdline.chosenFPS);
  plugin->fpsSlider->setCallback([plugin](double value, bool released){
    plugin->cmdline.chosenFPS = value;
  });
  // if (plugin->autoFPS){
  //   plugin->fpsSlider->setEnabled(true);
  // }
  // else{
  //   plugin->fpsSlider->setEnabled(false);
  // }

  plugin->rayMarchingStepSizeSlider = new ui::Slider(plugin->other_menu, "rayMarchingStepSizeSlider");
  plugin->rayMarchingStepSizeSlider->setText("Ray Marching Step Size");
  plugin->rayMarchingStepSizeSlider->setBounds(0.1, 5.);
  plugin->rayMarchingStepSizeSlider->setValue(plugin->cmdline.dt);
  plugin->rayMarchingStepSizeSlider->setCallback([plugin](double value, bool released){
      if (plugin->renderer != NULL ){
        if(value<=0){ // renderer crush if dt = 0
          plugin->cmdline.dt = 0.1;}
        else{
          plugin->cmdline.dt = value;}
          plugin->resetAccumulation();
      }
  });
  // if (plugin->autoFPS){
  plugin->rayMarchingStepSizeSlider->setEnabled(false); //default auto fps mode
  // }
  // else{
  //   rayMarchingStepSizeSlider->setEnabled(true);
  // }

  // set up Load Button
  auto load_button = new ui::Button(  plugin->tf_menu, "Load TF");
  load_button->setState(false);
  load_button->setCallback([plugin](bool state){
    plugin->loadTransferFunction();
  });
  

  #endif

  return 0;

}

// ==================================================================
// menu callback functions
// ==================================================================
void ExaBrickPlugin::renderSettingsChangedCB()
{
  plugin->renderer->frameState.clipBox.enabled = plugin->cmdline.clipBox.enabled;
  const box3f bounds = plugin->renderer->worldSpaceBounds;
  plugin->renderer->frameState.clipBox.coords.lower
    = vec3f(bounds.lower) + plugin->cmdline.clipBox.coords.lower * vec3f(bounds.span());
  plugin->renderer->frameState.clipBox.coords.upper
    = vec3f(bounds.lower) + plugin->cmdline.clipBox.coords.upper * vec3f(bounds.span());
  
  plugin->renderer->updateRenderSettings(plugin->cmdline.ao.enabled, 
                                        plugin->cmdline.ao.length, 
                                        plugin->cmdline.clockScale) ;
  plugin->resetAccumulation();
}

void ExaBrickPlugin::isoSurfacesChangedCB()
{
  plugin->resetAccumulation();
  plugin->renderer->updateIsoValues(plugin->isoSurfaceValue,
                            plugin->isoSurfaceChannel,
                            plugin->isoSurfaceEnabled);
}

void ExaBrickPlugin::contourPlanesChangedCB()
{
  plugin->resetAccumulation();
  plugin->renderer->updateContourPlanes(plugin->contourPlaneNormal,
                              plugin->contourPlaneOffset,
                              plugin->contourPlaneChannel,
                              plugin->contourPlaneEnabled);
}

void ExaBrickPlugin::tracerSettingsChangedCB()
{
  plugin->renderer->traces.tracerChannels = plugin->cmdline.traces.channels;
  plugin->renderer->traces.seedRegion = plugin->cmdline.traces.seedRegion;
  plugin->renderer->traces.numTraces = plugin->cmdline.traces.numTraces;
  plugin->renderer->traces.numTimesteps = plugin->cmdline.traces.numTimesteps;
  plugin->renderer->traces.steplen = plugin->cmdline.traces.steplen;
  plugin->renderer->traces.tracerEnabled = plugin->cmdline.traces.enabled;
  plugin->renderer->resetTracer();
  plugin->resetAccumulation();
}

void ExaBrickPlugin::spaceSkippingChangedCB()
{
  plugin->resetAccumulation();
  plugin->renderer->setSpaceSkipping(plugin->doSpaceSkipping);
}

void ExaBrickPlugin::gradientShadingDVRChangedCB()
{
  plugin->resetAccumulation();
  plugin->renderer->setGradientShadingDVR(plugin->gradientShadingDVR);
}

void ExaBrickPlugin::gradientShadingISOChangedCB()
{
  plugin->resetAccumulation();
  plugin->renderer->setGradientShadingISO(plugin->gradientShadingISO);
}

void ExaBrickPlugin::progressiveRefinementCB()
{
  plugin->resetAccumulation();
}

void ExaBrickPlugin::loadTransferFunction(){
  int XF_ALPHA_COUNT = INITIAL_XF_ALPHA_COUNT;
  std::string fileName = plugin->cmdline.exaFileName + ".xf";
  std::ifstream xfFile(fileName,std::ios::binary);
  if (plugin->cmdline.xfAlpha.empty()) {
    plugin->cmdline.xfAlpha.resize(1);
  }
  xfFile.read((char*)plugin->cmdline.xfAlpha[0].data(),XF_ALPHA_COUNT*sizeof(float));
  // for (size_t c=0;c<renderer->scalarFields.size();++c) {
  //   for (int i=0;i<XF_ALPHA_COUNT;i++) {
  //     float t=i/(float)(XF_ALPHA_COUNT-1);
  //     cmdline.xfAlpha[c][i] = t;
  //   }
  // }
  plugin->updateTransferFunction();
  std::cout << "#viewer: done writing transfer function (opacities) to '" << fileName << "'" << std::endl;
}

void ExaBrickPlugin::autoFPSChanged(){
  if (plugin->autoFPS){
    plugin->fpsSlider->setEnabled(true);
    plugin->rayMarchingStepSizeSlider->setEnabled(false);
  }
  else{
    plugin->fpsSlider->setEnabled(false);
    plugin->rayMarchingStepSizeSlider->setEnabled(true);
  }
}
// ==================================================================
// rendering
// ==================================================================

void ExaBrickPlugin::preDraw(osg::RenderInfo &info)
{ 
  if(!plugin||!plugin->renderer) return;

  renderFrame(info);
}

void ExaBrickPlugin::preFrame()
{ 
  if(!plugin||!plugin->renderer) return;

  if(plugin->autoFPS){
    static bool firstFPSmeasurement = true; 
    float start = 0.0f;
    float fps = INITIAL_FPS; // initially desired frame rate
    float amplitude = INITIAL_AMPLITUDE; // initially desired fps swing
    float speed = INITIAL_SPEED; //change speed of dt
    int frequence = INITIAL_FREQUENCE; //change frequence of dt in ms

    // Measure fps:
    // float end = cover->frameTime();
    float end = cover->frameDuration();
    if (firstFPSmeasurement)
    {
      firstFPSmeasurement = false;
      fps = INITIAL_FPS;
    }
    else
    {
      // add a wait to prevent the loop from running too fast
      std::this_thread::sleep_for(std::chrono::milliseconds(frequence));
      fps = 1.0f / (end - start);
    }
    // cout<<"end: "<<end<<"   ";
    // cout<<"fps: "<<fps<<"   ";

    // increase ray marching step to let fps increase
    if (fps < plugin->cmdline.chosenFPS - 0.5*amplitude){
      // cout<<"dt: "<<plugin->cmdline.dt<<"\n"; 
      plugin->cmdline.dt += speed; // change speed of dt
      plugin->rayMarchingStepSizeSlider->setValue(plugin->cmdline.dt);
      plugin->resetAccumulation();
      // add a wait to prevent the loop from running too fast
      std::this_thread::sleep_for(std::chrono::milliseconds(frequence));
    }
    // drop ray marching step to let fps drop
    if (fps > plugin->cmdline.chosenFPS + 0.5*amplitude){
      // cout<<"dt: "<<plugin->cmdline.dt<<"\n"; 
      plugin->cmdline.dt -= speed; // change speed of dt
      plugin->rayMarchingStepSizeSlider->setValue(plugin->cmdline.dt);
      plugin->resetAccumulation();
      // add a wait to prevent the loop from running too fast
      std::this_thread::sleep_for(std::chrono::milliseconds(frequence));
    }
  }
}

void ExaBrickPlugin::renderFrame(osg::RenderInfo &info)
{ 
  if(!plugin||!plugin->renderer) return;

  int numChannels = coVRConfig::instance()->numChannels();
  if (!multiChannelDrawer) {
    std::cout<< "creating multiChannelDrawer..."<<"\n";
    multiChannelDrawer = new MultiChannelDrawer(false, false);
    multiChannelDrawer->setMode(MultiChannelDrawer::AsIs);  
    cover->getScene()->addChild(multiChannelDrawer);
    channelInfos.resize(numChannels);
  }

  // for (unsigned chan=0; chan<multiChannelDrawer->numViews(); ++chan) {
  for (unsigned chan=0; chan<numChannels; ++chan) {
    renderFrame(info, chan);
  }

}

void ExaBrickPlugin::renderFrame(osg::RenderInfo &info, unsigned chan)
{ 
  #if 1
  if(!plugin||!plugin->renderer) return;

  // Viewport
  auto cam = coVRConfig::instance()->channels[chan].camera;
  auto vp = cam->getViewport();
  int width = vp->width();
  int height = vp->height();

  // resize if the window size changed
  if (channelInfos[chan].width != width || channelInfos[chan].height != height) {
    channelInfos[chan].width = width;
    channelInfos[chan].height = height;
    // multiChannelDrawer->setViewEye(chan, current_eye);
    multiChannelDrawer->resizeView(chan, width, height,
                                    channelInfos[chan].depthFormat,
                                    channelInfos[chan].colorFormat);
    multiChannelDrawer->clearColor(chan);
    // multiChannelDrawer->clearDepth(chan);

    unsigned imgSize[] = {(unsigned)width,(unsigned)height};
    const char* vendor = (const char*)glGetString(GL_VENDOR);
    cout<< "graphics card is being used: "<<vendor<<"\n";

    // reallocating GPU memory
    cudaFree(d_fbPointer);
    cudaMalloc(&d_fbPointer, width*height*sizeof(unsigned));
    if (d_fbPointer != NULL) {
      plugin->renderer->resizeFrameBuffer(d_fbPointer, {width, height});
    }
    else {
      // Handle the case when fbPointer is NULL
      cout<<"fbPointer is NULL"<<"\n";
    }
  }

  //get depth buffer
  plugin->db = reinterpret_cast<char *>(multiChannelDrawer->depth(chan));
  //write 1 into depth buffer of all pixel, so they will not cover the menu
  float* depthBuffer = reinterpret_cast<float*>(db);
  int bufferSize = width * height;
  for (int i = 0; i < bufferSize; ++i) {
      depthBuffer[i] = 1.0f;
  }

  // get the matrices and send to Renderer
  osg::Matrix mv = multiChannelDrawer->modelMatrix(chan) * multiChannelDrawer->viewMatrix(chan);
  osg::Matrix pr = multiChannelDrawer->projectionMatrix(chan);

  // cout<<"mv:"<<"\n";
  // osg_print(mv);
  // cout<<"pr:"<<"\n";
  // osg_print(pr);

  // cast osg matrices into mat4f type matrices
  math::mat4f view = osg_cast(mv);
  math::mat4f proj = osg_cast(pr);
    // cout<<"view:"<<"\n";
    // math::printMat4f(view);
    // cout<<"proj:"<<"\n";
    // math::printMat4f(proj);
  // update camera when the viewport changes
  // if (channelInfos[chan].mv != view || channelInfos[chan].pr != proj) {
  if (notsame(channelInfos[chan].mv,view) || notsame(channelInfos[chan].pr,proj)) {
    // cout<<"updating matrices......."<<"\n";
    channelInfos[chan].mv = view;
    channelInfos[chan].pr = proj;
    plugin->renderer->setCameraMat(view,proj);
  }
  

  plugin->renderer->updateDt(plugin->cmdline.dt);
    // cout<<"renderframe accumID: "<<accumID<<"\n";
  plugin->renderer->updateFrameID(accumID);
  plugin->accumID++;
  plugin->renderer->render();

  void *h_fbPointer = (uint32_t *)multiChannelDrawer->rgba(0);
  cudaMemcpy(h_fbPointer, d_fbPointer, width*height*sizeof(unsigned),
              cudaMemcpyDeviceToHost);

  multiChannelDrawer->update();
  multiChannelDrawer->swapFrame();
  #endif
}

/* void ExaBrickPlugin::initFrames()
{  
  framesInitialized = true;
} */


void ExaBrickPlugin::expandBoundingSphere(osg::BoundingSphere &bs)
{
  #if 1
  if(!plugin||!plugin->renderer)
    return;
  cout<< "expandBoundingSphere"<<"\n";

  float bounds[6] = { 1e30f, 1e30f, 1e30f,
                      -1e30f,-1e30f,-1e30f };
      // bounds[0] = volumeData.sizeU;
      // bounds[1] = volumeData.sizeV;
      // bounds[2] = volumeData.sizeW;
      // bounds[3] = volumeData.sizeX;
      // bounds[4] = volumeData.sizeY;
      // bounds[5] = volumeData.sizeZ;
      // bounds[0] = fminf(bounds[0], 0.f);
      // bounds[1] = fminf(bounds[1], 0.f);
      // bounds[2] = fminf(bounds[2], 0.f);
  bounds[0] =  0.f;
  bounds[1] =  0.f;
  bounds[2] =  0.f;
  bounds[3] = fmaxf(bounds[3], volumeData.sizeX);
  bounds[4] = fmaxf(bounds[4], volumeData.sizeY);
  bounds[5] = fmaxf(bounds[5], volumeData.sizeZ);

  osg::Vec3f minCorner(bounds[0],bounds[1],bounds[2]);
  osg::Vec3f maxCorner(bounds[3],bounds[4],bounds[5]);

// cout<< minCorner"<< minCorner.x() << ", " << minCorner.y() << ", " << minCorner.z() << std::endl;
// cout<< maxCorner"<< maxCorner.x() << ", " << maxCorner.y() << ", " << maxCorner.z() << std::endl;
  osg::Vec3f center = (minCorner+maxCorner)*.5f;
  float radius = (center-minCorner).length();
  bs.set(center, radius);
  #endif
}


// this is called if the plugin is removed at runtime
ExaBrickPlugin::~ExaBrickPlugin()
{
  coVRFileManager::instance()->unregisterFileHandler(&TextHandler);
  cover->getScene()->removeChild(multiChannelDrawer);
  fprintf(stderr, "Goodbye!\n");
}

COVERPLUGIN(ExaBrickPlugin)