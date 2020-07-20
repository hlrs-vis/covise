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


};
