/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TRAFFICSIMULATION_PLUGIN_H
#define _TRAFFICSIMULATION_PLUGIN_H
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

#include <random>

#include <VehicleUtil/RoadSystem/RoadSystem.h>
#include <TrafficSimulation/VehicleManager.h>
#include <TrafficSimulation/VehicleFactory.h>
#include <TrafficSimulation/PedestrianManager.h>
#include <TrafficSimulation/PedestrianFactory.h>
//#include "RoadFootprint.h"

#include <TrafficSimulation/Vehicle.h>
using namespace covise;
using namespace opencover;

// forward declarations //
//
class PorscheFFZ;
class TrafficSimulationPlugin : public coVRPlugin, public coTUIListener
{
public:
    TrafficSimulationPlugin();
    ~TrafficSimulationPlugin();


    static int loadOD(const char *filename, osg::Group *loadParent, const char *key);
    static int unloadOD(const char *filename, const char *key);

    bool init();

    // key handling
    void key(int type, int keySym, int mod);

    // this will be called in PreFrame
    void preFrame();

    static TrafficSimulationPlugin *plugin;


private:

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
	int removeAgentsVelocity_;

    coTUIToggleButton *debugRoadButton;
    // operator map //
    //coTUITab* operatorMapTab;
    //coTUIMap* operatorMap;



    osg::ref_ptr<osg::MatrixTransform> sphereTransform;
    osg::ref_ptr<osg::Sphere> sphere;
    osg::ref_ptr<osg::Geode> sphereGeode;
    osg::ref_ptr<osg::StateSet> sphereGeoState;
    osg::ref_ptr<osg::Material> redmtl;
    osg::ref_ptr<osg::Material> greenmtl;

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

    void tabletEvent(coTUIElement *tUIItem);
    void tabletPressEvent(coTUIElement *tUIItem);
};
#endif
