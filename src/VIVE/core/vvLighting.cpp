/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *                                                                      *
 *                                                                      *
 *                            (C) 1996-200                              *
 *              Computer Centre University of Stuttgart                 *
 *                         Allmandring 30                               *
 *                       D-70550 Stuttgart                              *
 *                            Germany                                   *
 *									*
 *									*
 *	File            vvLighting.cpp                                *
 *									*
 *	Description     scene graph class                               *
 *                                                                      *
 *	Author          D. Rainer				        *
 *                 F. Foehl                                             *
 *                 U. Woessner                                          *
 *                                                                      *
 *	Date            20.08.97                                        *
 *                 10.07.98 Performer C++ Interface                     *
 *                 20.11.00 Pinboard config through covise.config       *
 ************************************************************************/

#include <config/CoviseConfig.h>

#include "vvLighting.h"
#include "vvSceneGraph.h"
#include "vvPluginSupport.h"
//#include "coVRShadowManager.h"
#include <vsg/nodes/Group.h>
#include <vsg/maths/mat4.h>
#include <vsg/nodes/MatrixTransform.h>
#include <vsg/lighting/PointLight.h> 
#include <vsg/lighting/SpotLight.h> 
#include <iostream>

#include <ui/Button.h>
#include <ui/Menu.h>
#include <ui/SelectionList.h>
#include <ui/Slider.h>

using namespace vsg;
using namespace vive;
using covise::coCoviseConfig;

static void setSpecularLight(vsg::Light *l, const vsg::vec4 &c, float f)
{
    PointLight* pl= dynamic_cast<vsg::PointLight*>(l);
    //if (pl)
     //   pl->setSpecular(vsg::vec4(std::min(c[0]*f,1.f), std::min(c[1]*f,1.f), std::min(c[2]*f,1.f), 1.));
}

vvLighting *vvLighting::s_instance = NULL;

vvLighting *vvLighting::instance()
{
    if (!s_instance)
        s_instance = new vvLighting;
    return s_instance;
}

vvLighting::vvLighting()
    : lightingMenu_(NULL)
    , switchHeadlight_(NULL)
    , headlightState(true)
    , switchOtherlights_(NULL)
    , otherlightsState(false)
    , strengthSpecularlight_(NULL)
    , specularlightStrength(0.f)
    , switchSpotlight_(NULL)
    , spotlightState(false)
{
    config();
    init();
}

void vvLighting::init()
{
    if (vv->debugLevel(2))
        fprintf(stderr, "\nnew vvLighting\n");

    config();

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //  Create sun  (overhead light)
    initSunLight();

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //  Create Lamp  (hand light)
    initLampLight();

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //  Create Menu Light
    //initMenuLight();

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //  Create other lights  (overhead light)
    initOtherLight();
}

vvLighting::~vvLighting()
{
    if (vv->debugLevel(2))
        fprintf(stderr, "\ndelete vvLighting\n");

    s_instance = NULL;
}

void vvLighting::config()
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvLighting::readConfigFile\n");

    spotlightState = coCoviseConfig::isOn("VIVE.Spotlight", spotlightState);
    specularlightStrength = coCoviseConfig::isOn("VIVE.Specular", specularlightStrength>0.) ? 1.f : 0.f;
}

void vvLighting::initSunLight()
{
    static const LightDef headlightDefault = {
        { // diffuse
          1.0, 1.0, 1.0
        },
        { // specular
          1.0, 1.0, 1.0
        },
        { // ambient
          0.3f, 0.3f, 0.3f
        },
        { // position
          0.0, -10000.0, 10000.0, 1.0
        },
        { // Spot: x,y,z,expo,angle
          0.0, 0.0, -1.0, 0.0, 180.0
        }
    };
    headlight = createLightSource("Sun", headlightDefault, true);
    if (headlight)
    {
        // save values in case we switch off specular
        //headlightSpec = ((osg::Light *)headlight->getLight())->getSpecular();
        const auto f = specularlightStrength;
        //setSpecularLight(headlight->getLight(), headlightSpec, f);

        addLight(headlight);
        headlightState = coCoviseConfig::isOn("VIVE.Headlight", headlightState);
        switchLight(headlight, headlightState);
    }
    shadowlight = headlight;
}

//void vvLighting::initMenuLight()
//{
//   static const LightDef menulightDefault =
//   {
//      {                                           // diffuse
//         1.0,      1.0,     1.0
//      },
//      {                                           // specular
//         1.0,      1.0,     1.0
//      },
//      {                                           // ambient
//         0.3,      0.3,     0.3
//      },
//      {                                           // position
//         0.0, -1000.0, 0.0, 1.0
//      },
//      {                                           // Spot: x,y,z,expo,angle
//         0.0,      1.0,    0.0, 0.0, 180.0
//      }
//   };
//   menulight = createLightSource("Menu",menulightDefault,true);
//   if (menulight)
//   {
//      addLight(menulight, vvSceneGraph::instance()->getMenuGroup(), vvSceneGraph::instance()->getMenuGroup());
//      switchLight(menulight, true);
//   }
//}

void vvLighting::initLampLight()
{
    static const LightDef spotlightDefault = {
        { // diffuse
          1.0f, 1.0f, 1.0f
        },
        { // specular
          1.0f, 1.0f, 1.0f
        },
        { // abmient
          0.2f, 0.2f, 0.2f
        },
        { // position
          0.0f, -1.0f, 0.0f, 1.0f
        },
        { // Spot: x,y,z,expo,angle
          0.0f, 1.0f, 0.0f, 1.0f, 30.0f
        }
    };

    spotlight = createLightSource("Lamp", spotlightDefault, true);
    if (spotlight)
    {
        //spotlightSpec = ((osg::Light *)spotlight->getLight())->getSpecular();
        const auto f = specularlightStrength;
      //  setSpecularLight(spotlight->getLight(), spotlightSpec, f);
        addLight(spotlight, vvSceneGraph::instance()->getHandTransform());
        // turn on/off spotlight according to covise.config setting
        // COVERConfig.SPOTLIGHT
        switchLight(spotlight, spotlightState);
    }
}

void vvLighting::initOtherLight()
{
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //  default values for Lights
    static const LightDef lightDefault = {
        { // diffuse
          0.5f, 0.5f, 0.5f
        },
        { // specular
          0.5f, 0.5f, 0.5f
        },
        { // abmient
          0.1f, 0.1f, 0.1f
        },
        { // position
          0.0f, -10000.0f, 0.0f, 1.0f
        },
        { // Spot: x,y,z,expo,angle
          0.0f, 0.0f, -1.0f, 0.0f, float(M_PI)
        }
    };

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //  Create Light 1  (overhead light)
    light1 = createLightSource("Light1", lightDefault, false);
    if (light1)
    {
       /* light1Spec = ((osg::Light*)light1->getLight())->getSpecular();
        const auto f = specularlightStrength;
        setSpecularLight(light1->getLight(), light1Spec, f);
        addLight(light1);
        switchLight(light1, true);
        otherlightsState = true;*/
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //  Create Light 2  (overhead light)
    light2 = createLightSource("Light2", lightDefault, false);
    if (light2)
    {
       /* light2Spec = ((osg::Light*)light2->getLight())->getSpecular();
        const auto f = specularlightStrength;
        setSpecularLight(light2->getLight(), light2Spec, f);
        addLight(light2);
        switchLight(light2, true);
        otherlightsState = true;*/
    }
}

vsg::ref_ptr<vsg::Light>
vvLighting::createLightSource(const char *configName, const LightDef &def, bool force)
{
    vsg::ref_ptr<vsg::Light> light = NULL;

    char *configEntry = new char[128]; // hope it won't ever be bigger
    strcpy(configEntry, "VIVE.Lights.");
    strcat(configEntry, configName);
    strcat(configEntry, ".");
    char *configBase = configEntry + strlen(configEntry);

    strcpy(configBase, "Specular");
    bool specOn = coCoviseConfig::isOn(configEntry, false);

    strcpy(configBase, "Diffuse");
    bool diffOn = coCoviseConfig::isOn(configEntry, false);

    strcpy(configBase, "Ambient");
    bool ambiOn = coCoviseConfig::isOn(configEntry, false);

    strcpy(configBase, "Position");
    bool posiOn = coCoviseConfig::isOn(configEntry, false);

    strcpy(configBase, "Spot");
    bool spotOn = coCoviseConfig::isOn(configEntry, false);

    // if any of these is set - do it
    if (specOn || diffOn || ambiOn || posiOn || force)
    {
        //fprintf(stderr, "creating light %s\n", configName);
        float x, y, z, w, r, g, b, expo, angle;

        auto pointLight = vsg::PointLight::create();

        pointLight->name = configName;
        pointLight->color.set(1.0f, 1.0f, 1.0f);
        pointLight->intensity = static_cast<float>(1.0);
        pointLight->position.set(static_cast<float>(def.pos.x), static_cast<float>(def.pos.y), static_cast<float>(def.pos.z));
        pointLight->radius = 1E20;
        //pointLight->intensity *= pointLight->radius;
        light = pointLight;
       /* vsg::Light* vsgLight = new vsg::Light();
        light->setLight(osgLight);

        osgLight->setName(configName);
        light->setName(configName);*/

        // read all colors here - we'll save the values for the specular color
        // somewhere else if specular is off

        // TODO coConfig:vector
        if (specOn)
        {
            strcpy(configBase, "Specular");
            r = coCoviseConfig::getFloat("r", configEntry, 0.0f);
            g = coCoviseConfig::getFloat("g", configEntry, 0.0f);
            b = coCoviseConfig::getFloat("b", configEntry, 0.0f);
           // ((osg::Light *)light->getLight())->setSpecular(vsg::vec4(r, g, b, 1));
        }
       // else
           // ((osg::Light *)light->getLight())->setSpecular(vsg::vec4(def.spec.r, def.spec.g, def.spec.b, 1));

        if (diffOn)
        {
            strcpy(configBase, "Diffuse");
            r = coCoviseConfig::getFloat("r", configEntry, 0.0f);
            g = coCoviseConfig::getFloat("g", configEntry, 0.0f);
            b = coCoviseConfig::getFloat("b", configEntry, 0.0f);
           // ((osg::Light *)light->getLight())->setDiffuse(vsg::vec4(r, g, b, 1));
        }
       // else
          //  ((osg::Light *)light->getLight())->setDiffuse(vsg::vec4(def.diff.r, def.diff.g, def.diff.b, 1));

        if (ambiOn)
        {
            strcpy(configBase, "Ambient");
            r = coCoviseConfig::getFloat("r", configEntry, 0.0f);
            g = coCoviseConfig::getFloat("g", configEntry, 0.0f);
            b = coCoviseConfig::getFloat("b", configEntry, 0.0f);
           // ((osg::Light *)light->getLight())->setAmbient(vsg::vec4(r, g, b, 1));
        }
       // else
           // ((osg::Light *)light->getLight())->setAmbient(vsg::vec4(def.amb.r, def.amb.g, def.amb.b, 1));

        w = 1.0;
        if (posiOn)
        {
            strcpy(configBase, "Position");
            x = coCoviseConfig::getFloat("x", configEntry, 0.0f);
            y = coCoviseConfig::getFloat("y", configEntry, 0.0f);
            z = coCoviseConfig::getFloat("z", configEntry, 0.0f);
            w = coCoviseConfig::getFloat("w", configEntry, 1.0f);
           // ((osg::Light *)light->getLight())->setPosition(vsg::vec4(x, y, z, w));
        }
        else
        {
          //  ((osg::Light *)light->getLight())->setPosition(vsg::vec4(def.pos.x, def.pos.y, def.pos.z, def.pos.w));
        }

        if (spotOn)
        {
            strcpy(configBase, "Spot");
            x = coCoviseConfig::getFloat("x", configEntry, 0.0f);
            y = coCoviseConfig::getFloat("y", configEntry, 0.0f);
            z = coCoviseConfig::getFloat("z", configEntry, 0.0f);
            expo = coCoviseConfig::getFloat("expo", configEntry, 0.0f);
            angle = coCoviseConfig::getFloat("angle", configEntry, 0.0f);
          /*  ((osg::Light*)light->getLight())->setDirection(vsg::vec3(x, y, z));
            ((osg::Light *)light->getLight())->setSpotExponent(expo);
            ((osg::Light *)light->getLight())->setSpotCutoff(angle);*/
        }
        else
        {
           /* ((osg::Light*)light->getLight())->setDirection(vsg::vec3(def.spot.x, def.spot.y, def.spot.z));
            ((osg::Light *)light->getLight())->setSpotExponent(def.spot.expo);
            ((osg::Light *)light->getLight())->setSpotCutoff(def.spot.angle);*/
        }
    }

    delete[] configEntry;

    return light;
}

int vvLighting::addLight(vsg::ref_ptr<vsg::Light>& ls, vsg::Group *parent, vsg::Node *root, const char *menuName)
{
    // fprintf(stderr, "add light: %p to %p lighting %p (num=%lu)\n",
    //      ls, parent, root, (unsigned long)lightList.size());
    if (!parent)
        parent = vvSceneGraph::instance()->getScene();

    if (!root)
        root = vvSceneGraph::instance()->getScene();
    lightList.push_back(Light(ls, root));

   /* osg::Light* light = ls->getLight();
    if (!light)
    {
        light = new osg::Light();
        ls->setLight(light);
    }*/

    //create LightMenu Switch to disable Light/ group of Lights
    if (menuName != NULL)
    {
        // check if we already have this menuentry
        multimap<string, vsg::Light *>::iterator it;
        it = m.find(menuName);
        if (it == m.end())
        {
            ui::Button *temp = new ui::Button(lightingMenu_, menuName);
            temp->setState(true);
            temp->setCallback([this, menuName](bool state){
                // menuItem must be in map
                // compare menuItem->getName with key in map and enable/disable linked lights
                pair<multimap<string, vsg::Light *>::iterator, multimap<string, vsg::Light *>::iterator> ppp;
                ppp = m.equal_range(menuName);
                for (multimap<string, vsg::Light *>::iterator it = ppp.first; it != ppp.second; ++it)
                {
                    switchLight(((*it).second), state);
                }
            });
        }

        // then add this light to the map
        m.insert(pair<string, vsg::Light *>(menuName, ls));
    }

    if (lightList.size() > 8)
    {
        cerr << "ERROR: ";
        cerr << "Too many lights " << (lightList.size()) << " only 8 are supported" << endl;
    }
   // light->setLightNum(lightList.size() - 1);

    parent->addChild(ls);

    switchLight(ls, true);

    return (int)lightList.size();
}

vsg::Light *vvLighting::removeLight(vsg::Light *ls)
{
    fprintf(stderr, "remove light: %p\n", ls);
    for (unsigned int i = 0; i < lightList.size(); i++)
    {
       /* if (lightList[i].source == ls)
        {
            switchLight(ls, false);
            if (ls->getNumParents())
            {
                vsg::Group *parent = ls->getParent(0);
                if (parent)
                {
                    parent->removeChild(ls);
                }
            }

            if (lightList.size() > i + 1)
            {
                lightList[i] = lightList[lightList.size() - 1];
                lightList.pop_back();
                osg::Light *light = lightList[i].source->getLight();
                light->setLightNum(i);
            }
            else
            {
                lightList.pop_back();
            }

            return ls;
        }*/
    }

    return NULL;
}

void vvLighting::initMenu()
{
    // place the menu inside "view options
    lightingMenu_ = new ui::Menu(vv->viewOptionsMenu, "Lighting");

    auto shadowChoice = new ui::SelectionList(lightingMenu_, "Shadows");
    shadowChoice->append("No shadows");
    shadowChoice->append("Shadow texture");
    shadowChoice->append("Soft shadow map");
    shadowChoice->append("Standard shadow map");
    shadowChoice->append("LightSpacePerspectiveShadowMapVB");
    shadowChoice->append("LightSpacePerspectiveShadowMapCB");
    shadowChoice->append("LightSpacePerspectiveShadowMapDB");
    shadowChoice->append("Shadow map");
    shadowChoice->select(0);
    shadowChoice->setCallback([this](int idx){
        std::string t = "NoShadows";
        switch(idx)
        {
        case 1: t="ShadowTexture"; break;
        case 2: t="SoftShadowMap"; break;
        case 3: t="StandardShadowMap"; break;
        case 4: t="LightSpacePerspectiveShadowMapVB"; break;
        case 5: t="LightSpacePerspectiveShadowMapCB"; break;
        case 6: t="LightSpacePerspectiveShadowMapDB"; break;
        case 7: t="ShadowMap"; break;
        }
        //coVRShadowManager::instance()->setTechnique(t);
    });

    switchHeadlight_ = new ui::Button(lightingMenu_, "Headlight");
    switchHeadlight_->setState(headlightState);
    switchHeadlight_->setCallback([this](bool state){
        headlightState = state;
        switchLight(headlight, headlightState);

        // always enable it for menu
        switchLight(headlight, true, vvSceneGraph::instance()->getMenuGroup());
        switchLight(headlight, true, vvSceneGraph::instance()->getMenuGroup());
        switchLight(headlight, true, vvSceneGraph::instance()->getMenuGroup());
    });

    if (light1 || light2)
    {
        switchOtherlights_ = new ui::Button(lightingMenu_, "OtherLights");
        switchOtherlights_->setText("Other lights");
        switchOtherlights_->setState(otherlightsState);
        switchOtherlights_->setCallback([this](bool state){
            otherlightsState = state;
            switchOtherLights(otherlightsState);
        });
    }

    strengthSpecularlight_ = new ui::Slider(lightingMenu_, "SpecularLight");
    strengthSpecularlight_->setText("Specular light");
    strengthSpecularlight_->setValue(specularlightStrength);
    strengthSpecularlight_->setBounds(0., 2.);
    strengthSpecularlight_->setCallback([this](double value, bool released){
        specularlightStrength = (float)value;
        /*if (headlight)
            setSpecularLight(headlight->getLight(), headlightSpec, value);
        if (spotlight)
            setSpecularLight(spotlight->getLight(), spotlightSpec, value);
        if (light1)
            setSpecularLight(light1->getLight(), light1Spec, value);
        if (light2)
            setSpecularLight(light2->getLight(), light2Spec, value);*/
    });

    switchSpotlight_ = new ui::Button(lightingMenu_, "Spotlight");
    switchSpotlight_->setState(spotlightState);
    switchSpotlight_->setCallback([this](bool state){
        spotlightState = state;
        switchLight(spotlight, spotlightState);
    });
}

void vvLighting::switchOtherLights(bool on)
{
    for (unsigned int i = 0; i < lightList.size(); i++)
    {
        // if light has its own menuentry its in m
        bool in_m = false;
        for (multimap<string, vsg::Light *>::iterator it = m.begin(); it != m.end(); it++)
        {
            if (((*it).second) == lightList[i].source)
                in_m = true;
        }

        if (lightList[i].source != headlight
            && lightList[i].source != spotlight && !in_m
                                                   /*&& lightList[i].source != menulight*/)
            switchLight(lightList[i].source.get(), on);
    }
}

vsg::Light *vvLighting::switchLight(vsg::Light *ls, bool on, vsg::Node *limitToBranch)
{
    for (unsigned int i = 0; i < lightList.size(); i++)
    {
        /*if (lightList[i].source == ls)
        {
            osg::StateAttribute::Values value = osg::StateAttribute::OFF;
            if (on)
            {
                value = osg::StateAttribute::ON;
            }

            if (!limitToBranch)
                lightList[i].on = on;
            vsg::ref_ptr<vsg::Node> root = limitToBranch;
            if (!root.get())
                root = lightList[i].root;
            if (root.get())
            {
                osg::Light *light = ls->getLight();
                osg::StateSet *stateset = root->getOrCreateStateSet();
                stateset->setAttributeAndModes(light, value);

                //	    // disable lights in menu branch except menulight
                //	    if (lightList[i].source != menulight)
                //	    {
                //	    	osg::StateSet *menustateset = vvSceneGraph::instance()->getMenuGroup()->getOrCreateStateSet();
                //			menustateset->setAttributeAndModes(light, osg::StateAttribute::OFF);
                //	    }
            }

            return ls;
        }*/
    }
    return NULL;
}

bool vvLighting::isLightEnabled(size_t ln) const
{
    if (ln >= lightList.size())
        return false;

    return lightList[ln].on;
}

void vvLighting::setShadowLight(vsg::Light *ls)
{
    shadowlight = ls; 
    //coVRShadowManager::instance()->setLight(ls);
}
