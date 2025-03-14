/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <util/common.h>
#include <map>

#include <vsg/maths/vec4.h>
#include <vsg/lighting/Light.h>
#include <vsg/core/ref_ptr.h>
namespace vive {
namespace ui {
class Menu;
class Button;
class Slider;
}
}

namespace vive
{

class VVCORE_EXPORT vvLighting
{
    static vvLighting *s_instance;
    vvLighting();

public:
    ui::Menu *lightingMenu_ = nullptr;
    ui::Button *switchHeadlight_;
    bool headlightState;

    ui::Button *switchOtherlights_;
    bool otherlightsState;

    ui::Slider *strengthSpecularlight_;
    float specularlightStrength = 0.f;

    ui::Button *switchSpotlight_;
    bool spotlightState;

    multimap<string, vsg::Light *> m;

public:

    enum { MaxNumLights = 8 };

    typedef struct
    {
        float r, g, b;
    } RGB;
    typedef struct
    {
        float x, y, z, w;
    } XYZW;
    typedef struct
    {
        float x, y, z, expo, angle;
    } SpotDef;
    typedef struct
    {
        RGB diff, spec, amb;
        XYZW pos;
        SpotDef spot;
    } LightDef;

    vsg::ref_ptr<vsg::Light> light1;
    vsg::ref_ptr<vsg::Light> light2;

    // rescue values of specular lights
    vsg::vec4 headlightSpec, light1Spec, light2Spec, spotlightSpec;

    /**
       *  Create a light source
       *  @param   configName   Look for Cover<configName> section in covise.config
       *  @param   defValue     Default values
       *  @param   force        Create even when no values in covise.config
       *  @retun   NULL if section missing and no 'force', pointer to Light otherwise
       */
    vsg::ref_ptr<vsg::Light> createLightSource(const char *configName,
                                        const LightDef &defValue,
                                        bool force);

    void initSunLight();
    void initLampLight();
    void initOtherLight();
    //      void initMenuLight();

    void initMenu();

    struct Light
    {
        Light(vsg::Light *ls, vsg::Node *r)
        {
            source = ls;
            root = r;
        }
        vsg::ref_ptr<vsg::Light> source;
        vsg::ref_ptr<vsg::Node> root;
        bool on;
    };
    vector<Light> lightList;

    void config();
    void init();

public:
    vsg::ref_ptr<vsg::Light> headlight;
    vsg::ref_ptr<vsg::Light> spotlight;
    vsg::ref_ptr<vsg::Light> shadowlight;
    static vvLighting *instance();

    virtual ~vvLighting();

    void update();

    // add light to scene
    int addLight(vsg::ref_ptr<vsg::Light> &ls, vsg::Group *parent = NULL, vsg::Node *root = NULL, const char *menuName = NULL);

    // remove light from scene
    vsg::Light *removeLight(vsg::Light *ls);

    // switch light
    // if limitToBranch is set, switch inside it only
    vsg::Light *switchLight(vsg::Light *ls, bool on, vsg::Node *limitToBranch = NULL);

    // switch lights other than headlight and spot light
    void switchOtherLights(bool on);
    
    vsg::Light *getShadowLight(){return shadowlight;};
    void setShadowLight(vsg::Light *ls);

    bool isLightEnabled(size_t ln) const;
};
}
