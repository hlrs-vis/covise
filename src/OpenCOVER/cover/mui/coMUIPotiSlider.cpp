// class, which creates a Poti as VRUI-Element
// creates a Slider as TUI-Element

#include "coMUIPotiSlider.h"
#include "support/coMUIConfigManager.h"
#include "coMUIContainer.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
#include <OpenVRUI/coMenuItem.h>

#include <iostream>

using namespace vrui;
using namespace opencover;

// constructor:
coMUIPotiSlider::coMUIPotiSlider(const std::string UniqueIdentifier, coMUIContainer* parent, float min, float max, float defaultValue, const std::string label){
    ConfigManager = NULL;
    constructor(UniqueIdentifier,parent, min, max, defaultValue, label);
}
coMUIPotiSlider::coMUIPotiSlider(const std::string UniqueIdentifier, coMUIContainer* parent, float min, float max, float defaultValue){
    ConfigManager = NULL;
    constructor(UniqueIdentifier, parent, min, max, defaultValue, UniqueIdentifier);
}

coMUIPotiSlider::~coMUIPotiSlider(){
    ConfigManager->removeElement(Identifier);
    // Slider allocated two spaces
    ConfigManager->deletePos(Identifier);
    ConfigManager->deletePos(Identifier);
}

// underlying constructor
void coMUIPotiSlider::constructor(const std::string UniqueIdentifier, coMUIContainer* parent, float min, float max, float defaultValue, const std::string label){
    value = defaultValue;
    minVal = min;
    maxVal = max;
    Label=label;
    Identifier=UniqueIdentifier;

    ConfigManager = coMUIConfigManager::getInstance();

    ConfigManager->addElement(UniqueIdentifier, this);

    Parent = parent;

    // create defaultvalue or take from constructor:
    // VRUI-CAVE-Element:
    Devices.push_back(device());
    Devices[0].Device= ConfigManager->keywordCAVE();
    Devices[0].UI= ConfigManager->keywordVRUI();
    Devices[0].Identifier= UniqueIdentifier;
    Devices[0].Visible = true;

    Devices[0].Label= ConfigManager->getCorrectLabel(Label, Devices[0].UI, Devices[0].Device, Devices[0].Identifier);
    // create VRUI-Element:
    createVRUIElement(Devices[0].Label);

    // TUI-Tablet-Element:
    Devices.push_back(device());
    Devices[1].Device= ConfigManager->keywordTablet();
    Devices[1].UI= ConfigManager->keywordTUI();
    Devices[1].Identifier= UniqueIdentifier;
    Devices[1].Visible = true;

    Devices[1].Label= ConfigManager->getCorrectLabel(Label, Devices[1].UI, Devices[1].Device, Devices[1].Identifier);
    // create TUI-Element:
    createTUIElement(Devices[1].Label, ConfigManager->getCorrectParent(Parent, Devices[1].UI, Devices[1].Device, Devices[1].Identifier));

    // find and set correct parameter (get them from configuration file, if possible):
    for (size_t i=0; i<Devices.size(); ++i){
        Devices[i].Visible = ConfigManager->getCorrectVisible(Devices[i].Visible, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
        Devices[i].Label = ConfigManager->getCorrectLabel(Label, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);

        // create UI-Elements:
        if (Devices[i].UI == ConfigManager->keywordTUI()){         // create TUI-Element
            while (true){
                int posx=ConfigManager->getCorrectPosX(Devices[i].UI, Devices[i].Device, Devices[i].Identifier, Parent->getUniqueIdentifier());
                int posy=ConfigManager->getCorrectPosY(Devices[i].UI, Devices[i].Device, Devices[i].Identifier, Parent->getUniqueIdentifier());
                if (!ConfigManager->isPosOccupied(posx+1, posy, Parent->getUniqueIdentifier())){
                    break;
                }
            }
            int posx=ConfigManager->getCorrectPosX(Devices[i].UI, Devices[i].Device, Devices[i].Identifier, Parent->getUniqueIdentifier());
            int posy=ConfigManager->getCorrectPosY(Devices[i].UI, Devices[i].Device, Devices[i].Identifier, Parent->getUniqueIdentifier());
            ConfigManager->preparePos(posx,posy, Parent->getUniqueIdentifier());
            ConfigManager->preparePos(posx+1,posy, Parent->getUniqueIdentifier());
            TUIElement->setPos(posx,posy);
            ConfigManager->addPos(Devices[i].Identifier, posx, posy, Parent->getUniqueIdentifier(), true);
            ConfigManager->addPos(Devices[i].Identifier, posx+1, posy, Parent->getUniqueIdentifier(), true);
            TUIElement->setHidden(!Devices[i].Visible);
        }else if (Devices[i].UI == ConfigManager->keywordVRUI()){  // create VRUI-Element
            if (Devices[i].Visible){                                            // visible
                ConfigManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->add(VRUIElement.get());
            }else{                                                               // invisible
                ConfigManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->remove(VRUIElement.get());
            }
        }else{
            std::cerr << "coMUIPotiSlider::constructor(): Elementtype " << Devices[i].UI << " not found in Constructor." << std::endl;
        }
    }
}

// called, if there is an interaction with the TUI
void coMUIPotiSlider::tabletEvent(coTUIElement *tUIItem){
    if (tUIItem == TUIElement.get()){
        VRUIElement->setValue(TUIElement->getValue());
        if (value!=TUIElement->getValue()){
            value = TUIElement->getValue();
        }
        emit valueChanged();
    }
}

// called, if there is an interaction with the VRUI
void coMUIPotiSlider::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == VRUIElement.get()){
        TUIElement->setValue(VRUIElement->getValue());
        if (value!=VRUIElement->getValue()){
            value = VRUIElement->getValue();
        }
        emit valueChanged();
    }
}

// create VRUI-Element
void coMUIPotiSlider::createVRUIElement(const std::string label){
    VRUIElement.reset(new coPotiMenuItem (label, minVal, maxVal, value));
    VRUIElement->setMenuListener(this);
}

// create TUI-Element
void coMUIPotiSlider::createTUIElement(const std::string label, coMUIContainer* Parent){
    TUIElement.reset(new coTUIFloatSlider(label, Parent->getTUIID()));
    TUIElement->setRange(minVal, maxVal);
    TUIElement->setValue(value);
    TUIElement->setEventListener(this);
}

// returns value
float coMUIPotiSlider::getValue(){
    return value;
}

// sets new value
void coMUIPotiSlider::setValue(float newVal){
    value=newVal;
    TUIElement->setValue(value);
    VRUIElement->setValue(value);
    emit valueChanged();
}

// set position of TUI-Element
void coMUIPotiSlider::setPos(int posx, int posy){
    for (size_t i=0; i<Devices.size(); ++i){
        if (Devices[i].UI == ConfigManager->keywordTUI()){                      // TUI-Element
            while (true){
                int posX=ConfigManager->getCorrectPosX(Devices[i].UI, Devices[i].Device, Devices[i].Identifier, Parent->getUniqueIdentifier());
                int posY=ConfigManager->getCorrectPosY(Devices[i].UI, Devices[i].Device, Devices[i].Identifier, Parent->getUniqueIdentifier());
                if (!ConfigManager->isPosOccupied(posX+1, posY, Parent->getUniqueIdentifier())){
                    break;
                }
            }
            int posX=ConfigManager->getCorrectPosX(posx, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
            int posY=ConfigManager->getCorrectPosY(posy, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
            ConfigManager->preparePos(posx,posy, Parent->getUniqueIdentifier());
            ConfigManager->preparePos(posX+1, posY, Parent->getUniqueIdentifier());
            ConfigManager->deletePos(Devices[i].Identifier);
            ConfigManager->deletePos(Devices[i].Identifier);
            TUIElement->setPos(posX,posY);
            ConfigManager->addPos(Devices[i].Identifier, posx, posy, Parent->getUniqueIdentifier(), false);
            ConfigManager->addPos(Devices[i].Identifier, posx+1, posy, Parent->getUniqueIdentifier(), false);
        }
    }
}


// set visible-value of named Elements
void coMUIPotiSlider::setVisible(bool visible, std::string UI){
    for (size_t i=0; i<Devices.size(); ++i){
        if (UI.find(Devices[i].UI)!=std::string::npos){                             // Element shall be changed
            if (Devices[i].Visible != visible){                                     // visible-value changed
                Devices[i].Visible = ConfigManager->getCorrectVisible(visible, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
                if (Devices[i].UI == ConfigManager->keywordTUI()){                  // TUI-Element
                    TUIElement->setHidden(!Devices[i].Visible);
                } else if (Devices[i].UI == ConfigManager->keywordVRUI()){          // VRUI-Element
                    if (Devices[i].Visible){
                        ConfigManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->add(VRUIElement.get());
                    } else{
                        ConfigManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->remove(VRUIElement.get());
                    }
                } else{
                    std::cerr << "coMUIPotiSlider::setVisible(): Elementtyp " << Devices[i].UI << " not found in setVisible(string, bool, bool)." << std::endl;
                }
            }
        }
    }
}

// set visible-value of all elements
void coMUIPotiSlider::setVisible(bool visible){
    std::string UI;
    UI.append(ConfigManager->keywordTUI() + " ");
    UI.append(ConfigManager->keywordVRUI() + " ");
    setVisible(visible, UI);
}

// set visible-value of named elements
void coMUIPotiSlider::setLabel(std::string label, std::string UI){
    for (size_t i=0; i<Devices.size(); ++i){
        if (UI.find(Devices[i].UI) != std::string::npos){                       // Element to be changed
            Devices[i].Label=ConfigManager->getCorrectLabel(label, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
            if (Devices[i].UI == ConfigManager->keywordTUI()){                  // TUI-Element
                TUIElement->setLabel(Devices[i].Label);
            } else if (Devices[i].UI == ConfigManager->keywordVRUI()){          // VRUI-Element
                VRUIElement->setLabel(Devices[i].Label);
            } else{
                std::cerr << "coMUIPotiSlider::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string, std::string)." << std::endl;
            }
        }
    }
}

// set label for all UI-Elements
void coMUIPotiSlider::setLabel(std::string label){
    std::string UI;
    UI.append(ConfigManager->keywordTUI() + " ");
    UI.append(ConfigManager->keywordVRUI() + " ");
    setLabel(label, UI);
}

// returns the parent of this element
coMUIContainer* coMUIPotiSlider::getParent(){
    return Parent;
}

// returns the UniqueIdentifier of the element
std::string coMUIPotiSlider::getUniqueIdentifier(){
    return Devices[0].Identifier;
}
