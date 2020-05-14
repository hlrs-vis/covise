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

private:

    ui::Menu* m_MainMenu,*m_CameraProps,*m_Optimization;
    ui::Action* m_AddCamera, *m_AddSafetyZone, *m_MaxCoverage1,*m_MaxCoverage2;
    ui::Slider* m_Visibility;
    ui::Button * m_Delete;

};
