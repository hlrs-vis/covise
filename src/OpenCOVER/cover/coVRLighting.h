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
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>

namespace opencover
{
class buttonSpecCell;

class COVEREXPORT coVRLighting : public vrui::coMenuListener
{
public:
    vrui::coSubMenuItem *lightingButton_;
    vrui::coRowMenu *lightingMenu_;
    vrui::coCheckboxMenuItem *switchHeadlight_;
    bool headlightState;

    vrui::coCheckboxMenuItem *switchOtherlights_;
    bool otherlightsState;

    vrui::coCheckboxMenuItem *switchSpecularlight_;
    bool specularlightState;

    vrui::coCheckboxMenuItem *switchSpotlight_;
    bool spotlightState;

    virtual void menuEvent(vrui::coMenuItem *menuItem);

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

    osg::LightSource *light1;
    osg::LightSource *light2;

    // rescue values of specular lights
    osg::Vec4 headlightSpec, light1Spec, light2Spec, spotlightSpec;

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
        Light(osg::LightSource *ls, osg::Node *r)
        {
            source = ls;
            root = r;
        }
        osg::ref_ptr<osg::LightSource> source;
        osg::ref_ptr<osg::Node> root;
        bool on;
    };
    vector<Light> lightList;

    void config();
    void init();

public:
    osg::LightSource *headlight;
    osg::LightSource *spotlight;
    osg::LightSource *shadowlight;
    static void manipulateCallback(void *sceneGraph, buttonSpecCell *spec);
    static coVRLighting *instance();

    // process key events
    bool keyEvent(int type, int keySym, int mod);

    coVRLighting();
    virtual ~coVRLighting();

    void update();

    // add light to scene
    int addLight(osg::LightSource *ls, osg::Group *parent = NULL, osg::Node *root = NULL, const char *menuName = NULL);

    // remove light from scene
    osg::LightSource *removeLight(osg::LightSource *ls);

    // switch light
    // if limitToBranch is set, switch inside it only
    osg::LightSource *switchLight(osg::LightSource *ls, bool on, osg::Node *limitToBranch = NULL);

    // switch lights other than headlight and spot light
    void switchOtherLights(bool on);
    
    osg::LightSource *getShadowLight(){return shadowlight;};
    void setShadowLight(osg::LightSource *ls);

};
}
#endif
