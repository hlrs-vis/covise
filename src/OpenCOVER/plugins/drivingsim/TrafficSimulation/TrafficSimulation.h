/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TRAFFICSIMULATION_H
#define _TRAFFICSIMULATION_H
/****************************************************************************\
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: TrafficSimulation Plugin                                    **
 **                                                                          **
 **                                                                          **
 ** Author: Florian Seybold, U.Woessner		                                **
 **                                                                          **
 ** History:  								                                         **
 ** Nov-01  v1	    				       		                                   **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <osg/Texture2D>
#include <osg/Material>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/StateSet>
#include <osg/Material>
#include <vector>
#include <cover/coTabletUI.h>
#include <osg/PositionAttitudeTransform>

#ifdef HAVE_TR1
#ifdef WIN32
#include <random>
#else
#include <tr1/random>
#endif
#else
#include "mtrand.h"
#endif

#include "RoadSystem/RoadSystem.h"
#include "VehicleManager.h"
#include "VehicleFactory.h"
#include "PedestrianManager.h"
#include "PedestrianFactory.h"
//#include "RoadFootprint.h"

#include "UDPBroadcast.h"

#include "Vehicle.h"
using namespace covise;
using namespace opencover;

// forward declarations //
//
class PorscheFFZ;
class TrafficSimulation : public coVRPlugin, public coTUIListener
{
public:
	static TrafficSimulation *instance();
    TrafficSimulation();

    void runSimulation();
    void haltSimulation();

    VehicleManager *getVehicleManager();
    PedestrianManager *getPedestrianManager();

	bool init();

	// this will be called in PreFrame
    void preFrame();

    // key handling
    void key(int type, int keySym, int mod);

    unsigned long getIntegerRandomNumber();
    double getZeroOneRandomNumber();

    static int counter;
    static int createFreq;
    static double min_distance;
    static int min_distance_tui;
    static int max_distance_tui;
    // static int min_distance_50;
    // static int min_distance_100;
    static double delete_at;
    static int delete_delta;
    static int useCarpool;
    static float td_multiplier;
    static float placeholder;
    static int maxVel;
    static int minVel;
    static int maxVehicles;

private:

    RoadSystem *system;
    VehicleManager *manager;
    PedestrianManager *pedestrianManager;
    VehicleFactory *factory;
    PedestrianFactory *pedestrianFactory;
    //osg::Group* roadGroup;
    osg::PositionAttitudeTransform *roadGroup;
    xercesc::DOMElement *rootElement;

    coTUITab *pluginTab;
    coTUIButton *startButton;
    coTUIButton *stopButton;
    coTUIFileBrowserButton *saveButton;
    coTUIFileBrowserButton *openC4DXMLButton;
    coTUIFileBrowserButton *openLandXMLButton;
    coTUIFileBrowserButton *openIntermapRoadButton;
    coTUIFileBrowserButton *exportSceneGraphButton;
    coTUIFileBrowserButton *loadTerrainButton;

    // remove agents //
    coTUIButton *removeAgentsButton;
    coTUISlider *removeAgentsSlider;

    coTUIToggleButton *debugRoadButton;
    int removeAgentsVelocity_;

    // operator map //
    //coTUITab* operatorMapTab;
    //coTUIMap* operatorMap;

    // broadcaster for ffz positions //
    PorscheFFZ *ffzBroadcaster;

    bool runSim;
    bool tessellateRoads;
    bool tessellatePaths;
    bool tessellateBatters;
    bool tessellateObjects;

    osg::ref_ptr<osg::MatrixTransform> sphereTransform;
    osg::ref_ptr<osg::Sphere> sphere;
    osg::ref_ptr<osg::Geode> sphereGeode;
    osg::ref_ptr<osg::StateSet> sphereGeoState;
    osg::ref_ptr<osg::Material> redmtl;
    osg::ref_ptr<osg::Material> greenmtl;

#ifdef HAVE_TR1
    std::tr1::mt19937 mersenneTwisterEngine;
    std::tr1::uniform_real<double> uniformDist;
    std::tr1::variate_generator<std::tr1::mt19937, std::tr1::uniform_real<double> > variGen;
#else
    MTRand_int32 mtGenInt; // 32-bit int generator
    MTRand mtGenDouble; // double in [0, 1) generator, already init
#endif
    std::string xodrDirectory;

    xercesc::DOMElement *getOpenDriveRootElement(std::string);

    //FFZ Tab
    coTUITab *ffzTab;
    //coTUIButton* startButton;
    //coTUIButton* stopButton;

    coTUIFrame *createFrame;

    //coTUIButton* createVehiclesAtMin_Button;
    coTUISlider *createVehiclesAtMin_Slider;
    int createVehiclesAtMin_;
    coTUILabel *createVehiclesAtMinLabel;

    //coTUIButton* minVel_Button;
    coTUISlider *minVel_Slider;
    int minVel_;
    coTUILabel *minVelLabel;

    //coTUIButton* maxVel_Button;
    coTUISlider *maxVel_Slider;
    int maxVel_;
    coTUILabel *maxVelLabel;

    //coTUIButton* createVehiclesAtMax_Button;
    coTUISlider *createVehiclesAtMax_Slider;
    int createVehiclesAtMax_;
    coTUILabel *createVehiclesAtMaxLabel;

    coTUIFrame *removeFrame;
    //coTUIButton* removeVehiclesAtButton;
    coTUISlider *removeVehiclesAtSlider;
    int removeVehiclesDelta_;
    coTUILabel *removeVehiclesAtLabel;

    //coTUIButton* createFreqButton;
    coTUISlider *createFreqSlider;
    int createFreq_;
    coTUILabel *createFreqLabel;

    //coTUILabel* useCarpoolLabel;
    coTUIButton *useCarpoolButton;
    int useCarpool_;
    coTUILabel *toggleCarpoolLabel;
    coTUIFrame *carpoolFrame;
    coTUIEditField *carpoolStateField;

    //coTUIEditIntField* carpoolStateField;

    coTUIFrame *tdFrame;
    //coTUIButton* td_multButton;
    coTUIFloatSlider *td_multSlider;
    float td_mult_;

    coTUIFloatSlider *td_valueSlider;
    float placeholder_;
    coTUILabel *td_valueLabel;

    coTUISlider *maxVehiclesSlider;
    int maxVehicles_;

    //Labels
    coTUILabel *carpoolLabel;
    coTUILabel *createLabel;
    coTUILabel *removeLabel;
    coTUILabel *tdLabel;
    coTUILabel *td_multLabel;
    coTUILabel *maxVehiclesLabel;

    //States
    coTUIEditField *multiField;
    coTUIEditField *tdField;
    coTUIEditField *carpoolField;

    coTUIEditField *tdMultField;
    coTUIEditField *tdValueField;

    coTUIFrame *test;

    void parseOpenDrive(xercesc::DOMElement *);

    bool loadRoadSystem(const char *filename);
    void deleteRoadSystem();

    void tabletEvent(coTUIElement *tUIItem);
    void tabletPressEvent(coTUIElement *tUIItem);
};
#endif
