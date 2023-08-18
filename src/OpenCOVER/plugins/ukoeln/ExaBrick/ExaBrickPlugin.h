/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// #ifndef _HELLO_PLUGIN_H
// #define _HELLO_PLUGIN_H
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
#pragma once

// #include <cover/ui/Owner.h>
// #include <cover/ui/Menu.h>
// #include <cover/ui/ButtonGroup.h>
#include <cover/ui/Button.h>
#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
// #include <config/CoviseConfig.h>
#include <PluginUtil/MultiChannelDrawer.h>
#include "exa/Config.h"
#include "exa/OptixRenderer.h"
# define EXPLICIT_BASIS_METHOD 1
// #if EXPLICIT_BASIS_METHOD
# include "exa/Regions.h"
# include "exa/mat4.h"
// # include "ExaBrickGlobals.h"
// #endif
// #include "GL/glui.h"

using namespace exa;
using namespace covise;
using namespace opencover;
using namespace vrui;

#define INITIAL_XF_ALPHA_COUNT 128

#define INITIAL_FPS 20.0f // initially desired frame rate
#define INITIAL_AMPLITUDE 1.0f // initially desired fps swing
#define INITIAL_SPEED 0.01f //change speed of dt
#define INITIAL_FREQUENCE 5 //change frequence of dt in ms


namespace opencover {
namespace ui {
class Element;
class Group;
class Slider;
class Menu;
class Button;
}
}

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coButtonMenuItem;
class coTrackerButtonInteraction;
}

class coDefaultFunctionEditor;


class ExaBrickPlugin : public opencover::coVRPlugin, public ui::Owner
{


public:
    static ExaBrickPlugin *plugin;
    static ExaBrickPlugin *instance();
    // osg::ref_ptr<osg::Group> m_scene;

    static void usage(const std::string &msg); 
    ExaBrickPlugin();    
    ~ExaBrickPlugin();
    bool init();
    static int loadFile(const char *filename, osg::Group *loadParent, const char *covise_key);
    void preDraw(osg::RenderInfo &info);
    void preFrame() override;
    void renderFrame(osg::RenderInfo &info);
    void renderFrame(osg::RenderInfo &info, unsigned chan);
    // viewing_params getViewingParams(const osg::RenderInfo &info) const;
    void resetAccumulation();
    void expandBoundingSphere(osg::BoundingSphere &bs);
    void setOpacityScale(float xfOpacityScale);
    void updateTransferFunction();
    void renderSettingsChangedCB();
    void isoSurfacesChangedCB();
    void contourPlanesChangedCB();
    void tracerSettingsChangedCB();
    void spaceSkippingChangedCB();
    void progressiveRefinementCB();
    void gradientShadingDVRChangedCB();
    void gradientShadingISOChangedCB();
    void loadTransferFunction();
    void autoFPSChanged();
    // //virtual bool destroy();
    // void addObject(const RenderObject *container, osg::Group *parent,
    //                const RenderObject *geometry, const RenderObject *normals,
    //                const RenderObject *colors, const RenderObject *texture);


protected:
    void *d_fbPointer{ nullptr };
    void *db{ nullptr };

private:
    ui::Menu *tf_menu = nullptr;
    ui::Menu *rs_menu = nullptr;
    ui::Menu *is_menu = nullptr;
    ui::Menu *cp_menu = nullptr;
    ui::Menu *tr_menu = nullptr;
    ui::Menu *other_menu = nullptr;
    ui::Slider *fpsSlider = nullptr;
    ui::Slider *rayMarchingStepSizeSlider = nullptr;
    // ui::Button *load_button = nullptr;
    // ui::Button *spaceSkippingEnabledButton = nullptr;
    // ui::Button *progressiveRefinementButton = nullptr;
    // ui::Button *gradientShadingDVRButton = nullptr;
    // ui::Button *gradientShadingISOButton = nullptr;
    coDefaultFunctionEditor *editor; ///< transfer function editor
    static void applyDefaultTransferFunction(void *userData);
    
    struct TFApplyCBData
    {
        coDefaultFunctionEditor *tfe;
    };

    TFApplyCBData tfApplyCBData;

    float xfOpacityScale=0.f;

    Config::SP config = nullptr;
    OptixRenderer::SP renderer = nullptr;
    mutable size_t          m_total_frame_num = 0;
    osg::ref_ptr<opencover::MultiChannelDrawer> multiChannelDrawer{nullptr};
    struct ChannelInfo {
        int width=1, height=1;
        GLenum colorFormat=GL_UNSIGNED_BYTE;
        GLenum depthFormat=GL_FLOAT;
        // osg::Matrix mv, pr;
        math::mat4f mv,pr;
    };
    std::vector<ChannelInfo> channelInfos;  
    
    /*! currently active channel */
    int channel=0;
    /*! isos only from channel 0 */
    const int isoChannel=0;
    std::vector<std::vector<vec3f>> xfColor;
    std::vector<int> xfColorMap;
    std::vector<interval<float>> xfDomain;
    float isoSurfaceValue[MAX_ISO_SURFACES];
    int   isoSurfaceChannel[MAX_ISO_SURFACES];
    int   isoSurfaceEnabled[MAX_ISO_SURFACES];

    vec3f contourPlaneNormal[MAX_CONTOUR_PLANES];
    float contourPlaneOffset[MAX_CONTOUR_PLANES];
    int   contourPlaneChannel[MAX_CONTOUR_PLANES];
    int   contourPlaneEnabled[MAX_CONTOUR_PLANES];

    // glui::GLUI_String customColorMapString;
    struct {
        const void *data;
        int sizeU, sizeV, sizeW;
        int sizeX, sizeY, sizeZ;
        // int bpc;
        // float minValue, maxValue;

        // std::vector<float> voxels;
        // std::vector<float> rgbLUT;
        // std::vector<float> alphaLUT;
    
        // bool changed = false;
        // bool deleteData = false;
    } volumeData;

    /*! accumulation ID */
    int accumID { 0 };
    /*! whether to progressively refine - must be an int to match the glui data type */
    int doSpaceSkipping { 1 };
    int gradientShadingDVR { 1 };
    int gradientShadingISO { 1 };
    // GLuint textureID { 0 };
    int autoFPS {1};

    struct {
      std::string exaFileName;
      std::vector<std::array<float, 128>> xfAlpha;
      
      // struct {
      //   vec3f vp = vec3f(0.f);
      //   vec3f vu = vec3f(0.f);
      //   vec3f vi = vec3f(0.f);
      //   float fov = 70.f;
      // } camera;
      // vec2i windowSize = vec2i(0.f);

      // std::string displayString;
      struct {
        float length  = 1000.f;
        bool  enabled = false;
      } ao;

      float clockScale = 0.f;
      int doProgressiveRefinement { 1 };
      int gradientShadingDVR { 1 };
      int gradientShadingISO { 1 };
      // int showColorBar { 0 };
      // int colorBarChannel { -1 };
      
      struct {
        box3f coords = box3f(vec3f(0),vec3f(1));
        // float orientation[4][4] = { {1,0,0,0},  {0,1,0,0},  {0,0,1,0},  {0,0,0,1} };
        // vec3f origin { vec3f(.5f) };
        bool  enabled = false;
        // int   inverted { 0 };
      } clipBox;

      struct {
        vec3i channels { 0,1,2 };
        box3f seedRegion {{.3f,.3f,.5f},{.8f,.8f,.5f}};
        int numTraces = 1000;
        int numTimesteps = 100;
        float steplen = 1e-6f;
        int enabled { 0 };
      } traces;

      // interval<float> valueRange = interval<float>(1e20f, -1e20f);

      std::vector<float> isovalues;
      std::vector<int> isochannels;
      std::vector<vec4f> contourplanes; // normal (xyz), offset (w)
      std::vector<int> contourchannels;
      // std::vector<std::string> colormaps;
      // std::string customColorMapString;
      // int orbitCount = 0;
      // vec3f orbitUp = vec3f(0, 1, 0);
      // vec3f orbitCenter = vec3f(1e20f);
      // float orbitRadius = -1.f;

      float xfOpacityScale = 1.f;

      float dt = .5f; //ray march step (>0) should be something like 0.5 , 1 or 2
      float chosenFPS = INITIAL_FPS; // initially desired frame rate
    } cmdline;
  
};
// #endif
