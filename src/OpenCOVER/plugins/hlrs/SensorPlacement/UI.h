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

//Singleton Class
class UI : public opencover::ui::Owner
{
public:
    UI();

    static bool m_DeleteStatus;
    void checkForObstacles()const;
    void checkVisibility()const;

private:
    //Main Menu
    ui::Menu *m_MainMenu;
    ui::Action *m_AddCamera, *m_AddSafetyZone, *m_AddSensorZone; 
    ui::Button *m_Delete;

    //Camera Menu
    ui::Menu *m_CameraProps;
    ui::Slider *m_Visibility;

    //Optimization Menu
    ui::Menu *m_Optimization;
    ui::Action *m_MaxCoverage1, *m_MaxCoverage2;
    

};
