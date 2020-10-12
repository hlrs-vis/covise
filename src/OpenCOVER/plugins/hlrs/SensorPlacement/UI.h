#pragma once


#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Group.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Label.h>
#include <cover/ui/Action.h>

#include <cover/coVRPluginSupport.h>

namespace opencover
{
    namespace ui
    {
        class Button;
        class Menu;
        class Group;
        class Slider;
        class Label;
        class Action;
    }
}

using namespace opencover;

class UI : public opencover::ui::Owner
{
public:
    UI();

    static bool m_DeleteStatus;
    static bool m_showAverageUDPPositions;
    static bool m_showShortestUDPPositions;
   
private:
    //Main Menu
    ui::Menu *m_MainMenu;
    ui::Action *m_AddCamera, *m_AddSafetyZonePrio1, *m_AddSafetyZonePrio2, *m_AddSensorZone; 
    ui::Button *m_Delete;

    //Sensor Menu
    ui::Menu *m_SensorProps;
    ui::Button *m_ShowOrientations;
    ui::Button *m_Rotx,*m_Roty,*m_Rotz;
    ui::Slider *m_SliderStepSizeX, *m_SliderStepSizeY, *m_SliderStepSizeZ;


    //Camera Menu
    ui::Menu *m_CameraProps;
    ui::Slider *m_Visibility, *m_FOV;

    //Optimization Menu
    ui::Menu *m_Optimization;
    ui::Action *m_MaxCoverage1, *m_MaxCoverage2;
    ui::Menu *m_Results;
    ui::Label *m_TotalCoverage, *m_Prio1Coverage, *m_Prio2Coverage, *m_Fitness, *m_NbrCameras, *m_NbrControlPoints, *m_OptimizationTime;
    //Max Coverage 1
    ui::Menu *m_MaxCoverage1Menu;
    ui::Slider *m_Penalty;
    ui::Slider *m_WeightingPrio1;
    //Max Coverage 2
    ui::Menu *m_MaxCoverage2Menu;

    //Demonstrator
    ui::Menu *m_Demonstrator;
    ui::Button *m_cameraPositions;


    //UDP Menu
    ui::Menu *m_UDP;
    ui::Button *m_showAverageUDPObjectionPosition; //  calculated the average position of an object from all markers or cameras, which can see the object and show it
    ui::Button *m_showShortestUDPObjectionPosition; // show the calculated position from the marker or camera which is cloesest to the detected object
    ui::Button *m_AverageFrames;                    //calculated the average positions from the last x frames;

};
