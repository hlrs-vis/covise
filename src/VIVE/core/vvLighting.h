/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_VR_LIGHTING_H
#define CO_VR_LIGHTING_H

/*! \file
 \brief  manage light sources

 \author (C)
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#include <util/common.h>
#include <map>

#include <osg/Vec4>
#include <osg/LightSource>
#include <osg/ref_ptr>
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

    multimap<string, osg::LightSource *> m;

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

    vsg::ref_ptr<osg::LightSource> light1;
    vsg::ref_ptr<osg::LightSource> light2;

    // rescue values of specular lights
    vsg::vec4 headlightSpec, light1Spec, light2Spec, spotlightSpec;

    /**
       *  Create a light source
       *  @param   configName   Look for Cover<configName> section in covise.config
       *  @param   defValue     Default values
       *  @param   force        Create even when no values in covise.config
       *  @retun   NULL if section missing and no 'force', pointer to Light otherwise
       */
    osg::LightSource *createLightSource(const char *configName,
                                        const LightDef &defValue,
                                        bool force);

    void initSunLight();
    void initLampLight();
    void initOtherLight();
    //      void initMenuLight();

    void initMenu();

    struct Light
    {
        Light(osg::LightSource *ls, vsg::Node *r)
        {
            source = ls;
            root = r;
        }
        vsg::ref_ptr<osg::LightSource> source;
        vsg::ref_ptr<vsg::Node> root;
        bool on;
    };
    vector<Light> lightList;

    void config();
    void init();

public:
    vsg::ref_ptr<osg::LightSource> headlight;
    vsg::ref_ptr<osg::LightSource> spotlight;
    vsg::ref_ptr<osg::LightSource> shadowlight;
    static vvLighting *instance();

    virtual ~vvLighting();

    void update();

    // add light to scene
    int addLight(osg::LightSource *ls, vsg::Group *parent = NULL, vsg::Node *root = NULL, const char *menuName = NULL);

    // remove light from scene
    osg::LightSource *removeLight(osg::LightSource *ls);

    // switch light
    // if limitToBranch is set, switch inside it only
    osg::LightSource *switchLight(osg::LightSource *ls, bool on, vsg::Node *limitToBranch = NULL);

    // switch lights other than headlight and spot light
    void switchOtherLights(bool on);
    
    osg::LightSource *getShadowLight(){return shadowlight;};
    void setShadowLight(osg::LightSource *ls);

    bool isLightEnabled(size_t ln) const;
};
}
#endif
