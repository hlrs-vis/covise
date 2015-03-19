// class, which creates a checkbox with label as VRUI
// creates a ToggleButton as TUI


#include <cover/coVRTui.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include "support/coMUIConfigManager.h"
#include "coMUIContainer.h"
#include "support/coMUISupport.h"
#include "coMUIToggleButton.h"
#include <iostream>

using namespace vrui;
using namespace std;
using namespace opencover;

// constructor:
coMUIToggleButton::coMUIToggleButton(const std::string UniqueIdentifier, coMUIContainer* parent, const std::string label){
    constructor(UniqueIdentifier, parent, label);
}
coMUIToggleButton::coMUIToggleButton(const std::string UniqueIdentifier, coMUIContainer* parent){
    constructor(UniqueIdentifier, parent, UniqueIdentifier);
}

// destructor:
coMUIToggleButton::~coMUIToggleButton(){
    ConfigManager->removeElement(Identifier);
    ConfigManager->deletePos(Identifier);
}

// underlying constructor
void coMUIToggleButton::constructor(const std::string UniqueIdentifier, coMUIContainer* parent, const std::string label){

    Label=label;
    Identifier=UniqueIdentifier;

    ConfigManager = coMUIConfigManager::getInstance();                          // necessary for parsing of configuration file, getting defaultvalues, parameters etc.

    ConfigManager->addElement(UniqueIdentifier, this);    // adds the element to elementlist

    Parent=parent;
    State=false;

    // create defaultvalue or take from constructor:
    // VRUI-CAVE-Element:
    Devices.push_back(device());
    Devices[0].Device=ConfigManager->keywordCAVE();
    Devices[0].UI=ConfigManager->keywordVRUI();
    Devices[0].Identifier=UniqueIdentifier;
    Devices[0].Visible = true;

    Devices[0].Label=ConfigManager->getCorrectLabel(Label, Devices[0].UI, Devices[0].Device, Devices[0].Identifier);

    // create VRUI-Element:
    createVRUIElement(Devices[0].Label);

    // TUI-Tablet-Element:
    Devices.push_back(device());
    Devices[1].Device=ConfigManager->keywordTablet();
    Devices[1].UI=ConfigManager->keywordTUI();
    Devices[1].Identifier = UniqueIdentifier;
    Devices[1].Visible = true;

    Devices[1].Label=ConfigManager->getCorrectLabel(Label, Devices[1].UI, Devices[1].Device, Devices[1].Identifier);
    // create TUI-Element:
    createTUIElement(Devices[1].Label, ConfigManager->getCorrectParent(Parent, Devices[1].UI, Devices[1].Device, Devices[1].Identifier));

    // find and set correct parameter (get them from configuration file, if possible):
    for (size_t i=0; i<Devices.size(); ++i){
        Devices[i].Visible = ConfigManager->getCorrectVisible(Devices[i].Visible, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
        Devices[i].Label   = ConfigManager->getCorrectLabel(Label, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);

        // create UI-Elements:
        if (Devices[i].UI == ConfigManager->keywordTUI()){                      // create TUI-Elements
            int posx=ConfigManager->getCorrectPosX(Devices[i].UI, Devices[i].Device, Devices[i].Identifier, Parent->getUniqueIdentifier());
            int posy=ConfigManager->getCorrectPosY(Devices[i].UI, Devices[i].Device, Devices[i].Identifier, Parent->getUniqueIdentifier());
            ConfigManager->preparePos(posx,posy, Parent->getUniqueIdentifier());
            TUIElement.get()->setPos(posx,posy);
            ConfigManager->addPos(Devices[i].Identifier, posx, posy, Parent->getUniqueIdentifier(), true);
            TUIElement->setHidden(!Devices[i].Visible);
        }else if (Devices[i].UI == ConfigManager->keywordVRUI()){               // create VRUI-Element
            if (Devices[i].Visible){                                            // visible
                ConfigManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->add(VRUIElement.get());
            }else{                                                              // invisible
                ConfigManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->remove(VRUIElement.get());
            }
        }else{
            std::cerr << "coMUIToggleButton::constructor(): Elementtype " << Devices[i].UI << " not found in constructor." << std::endl;
        }
    }
}

// called, if there is an interaction with the tablet
void coMUIToggleButton::tabletEvent(coTUIElement *tUIItem){
    if (tUIItem == TUIElement.get()){                                                 // there is an interaction with the tablet
        VRUIElement->setState(!(VRUIElement->getState()));                      // adjust status of VRUI-Element
        emit clicked();
        State=!State;
    }
}

// called, if there is an interaction with the VRUI
void coMUIToggleButton:: menuEvent(coMenuItem *menuItem){
    if (menuItem == VRUIElement.get()){                                               // there is an interaction with the VRUI
        TUIElement->setState(!(TUIElement->getState()));                        // adjust status of TUI-ELement
        emit clicked();
        State=!State;
    }
}

void coMUIToggleButton::setPos(int posx, int posy){
    for (size_t i=0; i<Devices.size(); ++i){
        if (Devices[i].UI == ConfigManager->keywordTUI()){                      // TUI-Element
            int posX=ConfigManager->getCorrectPosX(posx, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
            int posY=ConfigManager->getCorrectPosY(posy, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
            ConfigManager->preparePos(posx,posy, Parent->getUniqueIdentifier());
            TUIElement->setPos(posX,posY);
        }
    }
}

// sets Label for all UI-elements
void coMUIToggleButton::setLabel(std::string label){
    for (size_t i=0; i<Devices.size(); ++i){
        Devices[i].Label = ConfigManager->getCorrectLabel(label, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
        if (Devices[i].UI == ConfigManager->keywordTUI()){                      // TUI-Element
            TUIElement->setLabel(Devices[i].Label);
        } else if (Devices[i].UI == ConfigManager->keywordVRUI()){              // VRUI-Element
            VRUIElement->setLabel(Devices[i].Label);
        } else{
            std::cerr << "coMUIToggleButton::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string)." << std::endl;
        }
    }
}

// sets Label for named UI-elements
void coMUIToggleButton::setLabel(std::string label, std::string UI){
    for (size_t i=0; i<Devices.size(); ++i){
        if (UI.find(Devices[i].UI) != std::string::npos){                       // Element to be changed
            Devices[i].Label=ConfigManager->getCorrectLabel(label, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
            if (Devices[i].UI == ConfigManager->keywordTUI()){                  // TUI-Element
                TUIElement->setLabel(Devices[i].Label);
            } else if (Devices[i].UI == ConfigManager->keywordVRUI()){          // VRUI-Element
                TUIElement->setLabel(Devices[i].Label);
            } else{
                std::cerr << "coMUIPotiSlider::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string, std::string)." << std::endl;
            }
        }
    }
}

// sets the visible-value for all UI-elements
void coMUIToggleButton::setVisible(bool visible){
    for (size_t i=0; i<Devices.size(); ++i){
        if (Devices[i].Visible != visible){                                     // Value changed
            Devices[i].Visible = ConfigManager->getCorrectVisible(visible, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
            if (Devices[i].UI == ConfigManager->keywordTUI()){                  // TUI-Element
                TUIElement->setHidden(!Devices[i].Visible);
            }
            else if (Devices[i].UI == ConfigManager->keywordVRUI()){            // VRUI-Element
                if (Devices[i].Visible){                                        // Visible
                    ConfigManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->add(VRUIElement.get());
                }
                else{                                                           // Invisible
                    ConfigManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->remove(VRUIElement.get());
                }
            } else{
                std::cerr << "coMUIPotiSlider::setVisible(): Elementtype " << Devices[i].UI << " not found in setVisible(bool)." << std::endl;
            }
        }
    }
}

// sets the visible-value for the named UI-Elements
void coMUIToggleButton::setVisible(bool visible, std::string UI){
    for (size_t i=0; i<Devices.size(); ++i){
        if (UI.find(Devices[i].UI)!=std::string::npos){                         // element shall be changed
            if (Devices[i].Visible != visible){                                 // visible-value changed
                Devices[i].Visible = ConfigManager->getCorrectVisible(visible, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
                if (Devices[i].UI == ConfigManager->keywordTUI()){              // TUI-Element
                    TUIElement->setHidden(!Devices[i].Visible);
                } else if (Devices[i].UI == ConfigManager->keywordVRUI()){      // VRUI-Element
                    if (Devices[i].Visible){
                        ConfigManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->add(VRUIElement.get());
                    }
                } else{
                    ConfigManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->remove(VRUIElement.get());
                }
            } else{
                std::cerr << "coMUIToggleButton::setVisible(): Elementtype " << Devices[i].UI << " not found in setVisible(string, bool)." << std::endl;
            }
        }
    }
}

// returns the parent-element of the toggleButton
coMUIContainer* coMUIToggleButton::getParent(){
    return Parent;
}

// creates the VRUI-Element
void coMUIToggleButton::createVRUIElement(const std::string label){
    VRUIElement.reset(new coCheckboxMenuItem(label, true));
    VRUIElement->setMenuListener(this);                                         // create listener
    VRUIElement->setState(State);
}

// creates the TUI-Element
void coMUIToggleButton::createTUIElement(const std::string label, coMUIContainer* parent){
    TUIElement.reset(new coTUIToggleButton(label, parent->getTUIID()));
    TUIElement->setEventListener(this);                                         // create listener
    TUIElement->setState(State);
}

void coMUIToggleButton::setState(bool stat){
    State=stat;
    for (size_t i=0; i<Devices.size(); ++i){
        if (Devices[i].UI == ConfigManager->keywordVRUI()){                     // VRUI-Element
            VRUIElement->setState(stat);
        }else if(Devices[i].UI == ConfigManager->keywordTUI()){                 // TUI-Element
            TUIElement->setState(stat);
        } else{
            std::cerr << "coMUIToggleButton::setState(): Elementtype " << Devices[i].UI << " not found in setClicked(bool)." << std::endl;
        }
    }
}

bool coMUIToggleButton::getState(){
    return State;
}

std::string coMUIToggleButton::getUniqueIdentifier(){
    return Devices[0].Identifier;
}

//***********************************************************************************
// QT-SLOTS:
//***********************************************************************************
void coMUIToggleButton::activate(){
    if (!State){
        emit clicked();
    }
    State=true;

    for (size_t i=0; i<Devices.size(); ++i){
        if (Devices[i].UI == ConfigManager->keywordTUI()){                      // TUIElement
            TUIElement->setState(true);
        } else if (Devices[i].UI == ConfigManager->keywordVRUI()){
            VRUIElement->setState(true);
        }
    }
}

void coMUIToggleButton::deactivate(){
    if (State){
        emit clicked();
    }
    State=true;

    for (size_t i=0; i<Devices.size(); ++i){
        if (Devices[i].UI == ConfigManager->keywordTUI()){                      // TUIElement
            TUIElement->setState(false);
        } else if (Devices[i].UI == ConfigManager->keywordVRUI()){
            VRUIElement->setState(false);
        }
    }
}

void coMUIToggleButton::click(){
    emit clicked();
    State=!State;
    for (size_t i=0; i<Devices.size(); ++i){
        if (Devices[i].UI == ConfigManager->keywordTUI()){                      // TUIElement
            TUIElement->setState(!TUIElement->getState());
        } else if (Devices[i].UI == ConfigManager->keywordVRUI()){              // VRUIElement
            VRUIElement->setState(!VRUIElement->getState());
        } else{
            std::cerr << "coMUIToggleButton::click(): Elementtyp " << Devices[i].UI << " not found in click()." << std::endl;
        }
    }
}
