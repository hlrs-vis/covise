#include "Beispiel.h"
#include <cover/mui/coMUIToggleButton.h>
#include <cover/mui/coMUITab.h>
#include <cover/mui/coMUIPotiSlider.h>
#include <cover/mui/coMUIFrame.h>
#include <cover/mui/support/coMUIConfigManager.h>
#include <cover/mui/coMUILabel.h>
#include <cover/mui/coMUIContainer.h>
#include <iostream>

#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>


Beispiel::Beispiel()
{
    ConfigManager=NULL;
}

Beispiel::~Beispiel(){
}

bool Beispiel::init()
{
    std::cout << std::endl << std::endl;
    // get Instance of ConfigManager
    ConfigManager=coMUIConfigManager::getInstance();
    ConfigManager->setAdress("covise/src/OpenCOVER/plugins/coMUI/Beispiel_soll.xml");

    // create new Tab in TabFolder
    Tab1.reset(new coMUITab("plugins.examples.UserInterface.Tab1", "Tab1"));
    ConfigManager->printElementNames();

    // create new ToggleButton in Tab1
    Button1.reset(new coMUIToggleButton("plugins.examples.UserInterface.Button1", Tab1.get(), "Button1"));
    Button1->setPos(3,3);

    // create new PotiSlider in Tab1
    Slider1.reset(new coMUIPotiSlider("plugins.examples.UserInterface.Slider1", Tab1.get(), 0., 100., 50., "Slider1"));
    Slider1->setPos(1,3);

    // create new Frame-Element
    Frame.reset(new coMUIFrame("plugins.examples.UserInterface.Frame", Tab1.get(), "Frame"));
    Frame->setPos(1,2);

    // create new Label-Element
    Label.reset(new coMUILabel("plugins.examples.UserInterface.Label1", Frame.get(), "Label"));
    Label->setPos(2,1);

    // create new ToggleButton in Frame
    Button2.reset(new coMUIToggleButton("plugins.examples.UserInterface.Button2", Frame.get(), "Button2"));
    // connect Button1 with Button2
    QObject::connect (Button1.get(), SIGNAL(clicked()), Button2.get(), SLOT(click()));

    return true;
}

COVERPLUGIN(Beispiel)
