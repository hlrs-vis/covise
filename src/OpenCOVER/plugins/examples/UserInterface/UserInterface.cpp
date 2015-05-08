#include "UserInterface.h"

#include <cover/mui/ToggleButton.h>
#include <cover/mui/Tab.h>
#include <cover/mui/ValueRegulator.h>
#include <cover/mui/Frame.h>
#include <cover/mui/TabFolder.h>
#include <cover/mui/support/ConfigManager.h>
#include <cover/mui/LabelElement.h>
#include <cover/mui/Container.h>

#include <iostream>


UserInterface::UserInterface()
{
    ConfigManager=NULL;
}

UserInterface::~UserInterface(){
}

bool UserInterface::init()
{
    // get Instance of ConfigManager
    ConfigManager=mui::ConfigManager::getInstance();

    // create new TabFolder
    TabFolder1.reset(mui::TabFolder::create("plugins.examples.UserInterface.TabFolder1"));
    TabFolder1->setLabel("TabFolder");

    // create new Tab in TabFolder
    Tab1.reset(mui::Tab::create("plugins.examples.UserInterface.Tab1", TabFolder1.get()));
    Tab1->setLabel("Tab1");


    // create new ToggleButton in Tab1
    Button1.reset(mui::ToggleButton::create("plugins.examples.UserInterface.Button1", Tab1.get()));
    Button1->setLabel("Button1");
    Button1->setEventListener(this);
    Button1->setPos(0,0);                                                                                   // will be ignored, if there exists a positioning instruction in configuration file

    Button2.reset(mui::ToggleButton::create("plugins.examples.UserInterface.Button2", Tab1.get()));
    Button2->setLabel("Button2");
    Button2->setEventListener(this);
    Button2->setPos(1,0);                                                                                   // will be obeyed, if there is no positioning instruction for this element in the configuration file

    // create new ValueRegulator in Tab1
    ValueRegulator1.reset(mui::ValueRegulator::create("plugins.examples.UserInterface.Slider1", Tab1.get(), 0., 100., 50.));
    ValueRegulator1->setLabel("ValueRegulator1");
    ValueRegulator1->setPos(0,1);                                                                                   // will be obeyed, if there is no positioning instruction for this element in the configuration file

    // create new Frame-Element
    Frame.reset(mui::Frame::create("plugins.examples.UserInterface.Frame", Tab1.get()));
    Frame->setLabel("Frame");
    Frame->setPos(0,2);                                                                                     // will be obeyed, if there is no positioning instruction for this element in the configuration file

    // create new Label-Element
    Label.reset(mui::LabelElement::create("plugins.examples.UserInterface.Label1", Frame.get()));
    Label->setLabel("Label");
    Label->setPos(0,3);                                                                                     // will be obeyed, if there is no positioning instruction for this element in the configuration file


    return true;
}

void UserInterface::muiEvent(mui::Element *muiItem)
{
    if (muiItem == Button1.get())
    {
        Button2->setState(Button1->getState());
    }
    if (muiItem == Button2.get())
    {
        Button1->setState(Button2->getState());
    }
}

COVERPLUGIN(UserInterface)
