 
/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include <fstream>
#include <iostream>
#include <ostream>

#include <GL/glew.h>

#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>

#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>


#include "PointRayTracerPlugin.h"

using namespace osg;
using namespace visionaray;
PointRayTracerPlugin *PointRayTracerPlugin::plugin = NULL;

static FileHandler handlers[] = {
    { NULL,
      PointRayTracerPlugin::sloadPts,
      PointRayTracerPlugin::unloadPts,
      "pts" },
    { NULL,
      PointRayTracerPlugin::sloadPts,
      PointRayTracerPlugin::unloadPts,
      "3d" },
    { NULL,
      PointRayTracerPlugin::sloadPts,
      PointRayTracerPlugin::unloadPts,
      "xyz" },
    { NULL,
      PointRayTracerPlugin::sloadPts,
      PointRayTracerPlugin::unloadPts,
      "ply" }
};

//-----------------------------------------------------------------------------

PointRayTracerPlugin *PointRayTracerPlugin::instance()
{
    return plugin;
}


PointRayTracerPlugin::PointRayTracerPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    //register file handler
    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    coVRFileManager::instance()->registerFileHandler(&handlers[1]);
    coVRFileManager::instance()->registerFileHandler(&handlers[2]);
    coVRFileManager::instance()->registerFileHandler(&handlers[3]);

    //create drawable
    m_drawable = new PointRayTracerDrawable;
}


bool PointRayTracerPlugin::init()
{
    if (cover->debugLevel(1)) fprintf(stderr, "\n    PointRayTracerPlugin::init\n");

    if (PointRayTracerPlugin::plugin != NULL)
        return false;

    PointRayTracerPlugin::plugin = this;


    m_numPointClouds = 0;
    m_currentPointCloud = 0;
    m_currentPointCloud_has_changed = false;

    //TODO: we might want to check if the cached BVH was created
    //with the same point size that is required from the config
    m_pointSize = covise::coCoviseConfig::getFloat("COVER.Plugin.PointRayTracer.PointSize",0.01f);

    //check if loading from cache is enabled
    bool ignore;
    m_useCache = covise::coCoviseConfig::isOn("value", "COVER.Plugin.PointRayTracer.CacheBinaryFile", false, &ignore);

    //check if the files contain UTM data that need to be "cut down" like a dog
    m_cutUTMdata = covise::coCoviseConfig::isOn("value", "COVER.Plugin.PointRayTracer.CutUTMData", false, &ignore);

    //init Menu
    prtSubMenuEntry = new coSubMenuItem("Point Ray Tracer");
    cover->getMenu()->add(prtSubMenuEntry);

    prtMenu = new coRowMenu("Point Ray Tracer", cover->getMenu());
    prtSubMenuEntry->setMenu(prtMenu);

    nextItem = new coButtonMenuItem("Next Point Cloud");
    prtMenu->add(nextItem);
    nextItem->setMenuListener(this);

    prevItem = new coButtonMenuItem("Previous Point Cloud");
    prtMenu->add(prevItem);
    prevItem->setMenuListener(this);

    //ButtonD = forward
    //ButtonE = backward
    interactionNext = new coTrackerButtonInteraction(coInteraction::ButtonB,"Next Point Cloud", coInteraction::Medium);
    interactionPrev = new coTrackerButtonInteraction(coInteraction::ButtonC,"Previous Point Cloud", coInteraction::Medium);
    coInteractionManager::the()->registerInteraction(interactionNext);
    coInteractionManager::the()->registerInteraction(interactionPrev);

    return true;
}


bool PointRayTracerPlugin::init2(){
    if (cover->debugLevel(1)) fprintf(stderr, "\n    PointRayTracerPlugin::init2\n");

    //init geode and add it to the scenegraph
    m_geode = new osg::Geode;
    m_geode->setName("PointRayTracer");
    m_geode->addDrawable(m_drawable);
    opencover::cover->getScene()->addChild(m_geode);

    /*
    m_visibility_has_changed = false;
    for(int i = 0; i < m_bvh_vector.size(); i++){
        m_visibility_vector.push_back(true);
    }
    */

    return true;
}


int PointRayTracerPlugin::sloadPts(const char *filename, osg::Group *loadParent, const char *){
    (void)loadParent;
    return instance()->loadPts(filename);
}


int PointRayTracerPlugin::loadPts(const char *filename){
    if (cover->debugLevel(1)) fprintf(stderr, "\n    PointRayTracerPlugin::loadPts: %s\n", filename);
    if(!PointReader::instance()->readFile(std::string(filename), m_pointSize, m_bvh_vector, m_bbox, m_useCache, m_cutUTMdata)) return 1;

    m_numPointClouds++;
    return 0;
}


int PointRayTracerPlugin::unloadPts(const char *filename, const char *){
    (void)filename;
    return 0;
}


PointRayTracerPlugin::~PointRayTracerPlugin()
{
    if (cover->debugLevel(1))
        fprintf(stderr, "\n    delete PointRayTracerPlugin\n");

    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);
    coInteractionManager::the()->unregisterInteraction(interactionNext);
    coInteractionManager::the()->unregisterInteraction(interactionPrev);

    delete m_reader;

}


void PointRayTracerPlugin::preFrame()
{
    //if (cover->debugLevel(1)) fprintf(stderr, "\n    PointRayTracerPlugin::preFrame\n");

    if(interactionNext->wasStarted())
    {
        std::cout << "Button Next Pressed" << std::endl;
        showNextPointCloud();
    }
    if(interactionPrev->wasStarted()){
        std::cout << "Button Prev Pressed" << std::endl;
        showPreviousPointCloud();
    }

    /*
    static bool takeAction = true;
    int buttonStatus = cover->getPointerButton()->getState();

    switch (buttonStatus)
    {
        case 0:
            //fprintf(stderr, "\n    PointRayTracerPlugin::preFrame  No Button\n");
            takeAction = true;
            break;
        case vruiButtons::ACTION_BUTTON:
            if(takeAction)
            {
                //fprintf(stderr, "\n    PointRayTracerPlugin::preFrame  ACTION_BUTTON\n");
                takeAction = false;
            }            
            break;
        case vruiButtons::FORWARD_BUTTON:
            if(takeAction)
            {
                fprintf(stderr, "\n    PointRayTracerPlugin::preFrame  FORWARD_BUTTON\n");
                showNextPointCloud();
                takeAction = false;
            }            
            break;
        case vruiButtons::BACKWARD_BUTTON:
            if(takeAction)
            {
                fprintf(stderr, "\n    PointRayTracerPlugin::preFrame  BACKWARD_BUTTON\n");
                showPreviousPointCloud();
                takeAction = false;
            }            
            break;
        default:
            //fprintf(stderr, "\n    PointRayTracerPlugin::preFrame some BUTTON: %i\n", buttonStatus);
            break;
    }
    */
   
}


/*
void PointRayTracerPlugin::toggleVisibility(int index){
    //if (cover->debugLevel(1)) std::cout << "PointRayTracerPlugin::toggleVisibility() index: " << index << std::endl;

    if(m_visibility_vector.size() > index){
        m_visibility_vector[index] = !m_visibility_vector[index];
        m_visibility_has_changed = true;
    } else if (cover->debugLevel(1)) {
        std::cout << "PointRayTracerPlugin::toggleVisibility index() too high" << std::endl;
    }
}
*/

 
void PointRayTracerPlugin::showNextPointCloud(){
    m_currentPointCloud++;
    if(m_currentPointCloud >= m_numPointClouds) m_currentPointCloud = 0;
    m_currentPointCloud_has_changed = true;
    std::cout << "PointRayTracerPlugin:: current Point Cloud: " << m_currentPointCloud << std::endl;
}

void PointRayTracerPlugin::showPreviousPointCloud(){
    m_currentPointCloud--;
    if(m_currentPointCloud < 0) m_currentPointCloud = m_numPointClouds - 1;
    m_currentPointCloud_has_changed = true;
    std::cout << "PointRayTracerPlugin:: current Point Cloud: " << m_currentPointCloud << std::endl;
}


void PointRayTracerPlugin::preDraw(osg::RenderInfo &info)
{
    static bool initialized = false;
    if (!initialized)
    {
        if (cover->debugLevel(1)) fprintf(stderr, "\n    PointRayTracerPlugin::preDraw\n");
        m_drawable->initData(m_bvh_vector);
        initialized = true;
    }

    /*
    if(m_visibility_has_changed){
        m_drawable->setVisibility(m_visibility_vector);
        m_visibility_has_changed = false;
    }
    */

    if(m_currentPointCloud_has_changed){
        m_currentPointCloud_has_changed = false;
        m_drawable->setCurrentPointCloud(m_currentPointCloud);
    }
}


void PointRayTracerPlugin::expandBoundingSphere(osg::BoundingSphere &bs)
{
    m_drawable->expandBoundingSphere(bs);
}


void PointRayTracerPlugin::key(int type, int keySym, int mod)
{
    //if (cover->debugLevel(1))
    //fprintf(stderr, "PointRayTracerPlugin::key(type=%d,keySym=%d,mod=%d)\n", type, keySym, mod);

    if(type == osgGA::GUIEventAdapter::KEYUP){

        if(keySym == osgGA::GUIEventAdapter::KEY_Left){
            showPreviousPointCloud();
        }

        if(keySym == osgGA::GUIEventAdapter::KEY_Right){
            showNextPointCloud();
        }

    }

    /*
    if(type == osgGA::GUIEventAdapter::KEYUP && keySym >= osgGA::GUIEventAdapter::KEY_1 && keySym <= osgGA::GUIEventAdapter::KEY_9)
    {
        toggleVisibility(keySym - osgGA::GUIEventAdapter::KEY_1);
    }
    */
}


void PointRayTracerPlugin::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == nextItem)
    {
        showNextPointCloud();
    }

    if(menuItem == prevItem)
    {
        showPreviousPointCloud();
    }
}

COVERPLUGIN(PointRayTracerPlugin)
