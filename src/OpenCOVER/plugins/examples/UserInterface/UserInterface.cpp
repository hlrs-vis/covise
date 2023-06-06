#include "UserInterface.h"

#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Group.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Label.h>

#include <cover/coVRPluginSupport.h>

#include <iostream>

using namespace opencover;

UserInterface::UserInterface()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("UserInterfacePlugin", cover->ui)
{
}

UserInterface::~UserInterface(){
}

bool UserInterface::init()
{
    // create new Tab in TabFolder
    Tab1 = new ui::Menu("User Interface Demo", this);


    // create new ToggleButton in Tab1
    Button1 = new ui::Button(Tab1, "Button1");
    Button1->setCallback([this](bool state){
            Button2->setState(state);
    });

    Button2 = new ui::Button(Tab1, "Button2");
    Button2->setCallback([this](bool state){
            Button1->setState(state);
    });

    // create new ValueRegulator in Tab1
    ValueRegulator1 = new ui::Slider(Tab1, "Slider1");
    ValueRegulator1->setText("ValueRegulator1");
    ValueRegulator1->setBounds(0., 100.);
    ValueRegulator1->setValue(50.);

    // create new Frame-Element
    Frame = new ui::Group(Tab1, "Frame");

    // create new Label-Element
    Label = new ui::Label(Frame, "Label1");
    Label->setText("Label");

    return true;
}

COVERPLUGIN(UserInterface)
