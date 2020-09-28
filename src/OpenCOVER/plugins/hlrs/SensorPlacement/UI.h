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
    static bool m_showOrientations;
    static bool m_showAverageUDPPositions;
    static bool m_showShortestUDPPositions;

   
private:
    //Main Menu
    ui::Menu *m_MainMenu;
    ui::Action *m_AddCamera, *m_AddSafetyZone, *m_AddSensorZone; 
    ui::Button *m_Delete;

    //Sensor Menu
    ui::Menu *m_SensorProps;
    ui::Button *m_ShowOrientations;

    //Camera Menu
    ui::Menu *m_CameraProps;
    ui::Slider *m_Visibility;

    //Optimization Menu
    ui::Menu *m_Optimization;
    ui::Action *m_MaxCoverage1, *m_MaxCoverage2;

    //UDP Menu
    ui::Menu *m_UDP;
    ui::Button *m_showAverageUDPObjectionPosition; //  calculated the average position of an object from all markers or cameras, which can see the object and show it
    ui::Button *m_showShortestUDPObjectionPosition; // show the calculated position from the marker or camera which is cloesest to the detected object


};
