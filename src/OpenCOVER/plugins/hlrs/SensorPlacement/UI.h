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

    bool m_DeleteStatus{false};


private:

    ui::Menu* m_MainMenu,*m_CameraProps;
    ui::Action* m_AddCamera, *m_AddSafetyZone;
    ui::Slider* m_Visibility;
    ui::Button * m_Delete;




};
