// class, which creates a Poti as VRUI-Element
// creates a Slider as TUI-Element

#include "PotiSlider.h"
#include "support/ConfigManager.h"
#include "Container.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRTui.h>
#include <OpenVRUI/coMenuItem.h>

#include <iostream>

using namespace vrui;
using namespace opencover;
using namespace mui;

// constructor:
PotiSlider::PotiSlider(const std::string uniqueIdentifier, Container* parent, float min, float max, float defaultValue, const std::string label)
{
    configManager = NULL;
    constructor(uniqueIdentifier,parent, min, max, defaultValue, label);
}

PotiSlider::PotiSlider(const std::string uniqueIdentifier, Container* parent, float min, float max, float defaultValue)
{
    configManager = NULL;
    constructor(uniqueIdentifier, parent, min, max, defaultValue, uniqueIdentifier);
}

PotiSlider::~PotiSlider()
{
    configManager->removeElement(UniqueIdentifier);
    configManager->deletePosFromPosList(UniqueIdentifier);
}

// underlying constructor
void PotiSlider::constructor(const std::string uniqueIdentifier, Container* parent, float min, float max, float defaultValue, const std::string label)
{
    value = defaultValue;
    minVal = min;
    maxVal = max;
    Label=label;
    UniqueIdentifier=uniqueIdentifier;

    configManager = ConfigManager::getInstance();

    configManager->addElement(UniqueIdentifier, this);

    Parent = parent;

    // create defaultvalue or take from constructor:
    // VRUI-CAVE-Element:
    Devices.push_back(device());
    Devices[0].Device= mui::CAVEEnum;
    Devices[0].UI= mui::VRUIEnum;
    Devices[0].UniqueIdentifier= UniqueIdentifier;
    Devices[0].Visible = true;

    Devices[0].Label= configManager->getCorrectLabel(Label, Devices[0].UI, Devices[0].Device, Devices[0].UniqueIdentifier);
    // create VRUI-Element:
    createVRUIElement(Devices[0].Label);

    // TUI-Tablet-Element:
    Devices.push_back(device());
    Devices[1].Device= mui::TabletEnum;
    Devices[1].UI= mui::TUIEnum;
    Devices[1].UniqueIdentifier= UniqueIdentifier;
    Devices[1].Visible = true;

    Devices[1].Label= configManager->getCorrectLabel(Label, Devices[1].UI, Devices[1].Device, Devices[1].UniqueIdentifier);
    // create TUI-Element:
    createTUIElement(Devices[1].Label, configManager->getCorrectParent(Parent, Devices[1].UI, Devices[1].Device, Devices[1].UniqueIdentifier));

    // find and set correct parameter (get them from configuration file, if possible):
    for (size_t i=0; i<Devices.size(); ++i)
    {
        Devices[i].Visible = configManager->getCorrectVisible(Devices[i].Visible, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
        Devices[i].Label = configManager->getCorrectLabel(Label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);

        // create UI-Elements:
        if (Devices[i].UI == mui::TUIEnum)         // create TUI-Element
        {
            std::pair<int,int> pos=configManager->getCorrectPos(Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier, Parent->getUniqueIdentifier());
            std::pair<int,int> pos2(pos.first+1, pos.second);

            std::vector <std::pair<int,int> > exceptPos;

            while (true)                                        // search for two free positions
            {
                pos=configManager->getCorrectPosExceptOfPos(exceptPos, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier, Parent->getUniqueIdentifier());
                pos2.first = pos.first+1;
                pos2.second= pos2.second;
                if (!configManager->isPosOccupied(std::pair<int,int>(pos2.first, pos2.second), Parent->getUniqueIdentifier()))
                {
                    break;
                }
                else if (configManager->isPosOccupied(std::pair<int,int>(pos.first+1, pos.second), Parent->getUniqueIdentifier()))
                {
                    exceptPos.push_back(pos);
                }
            }
            configManager->preparePos(pos, Parent->getUniqueIdentifier());
            configManager->preparePos(pos2, Parent->getUniqueIdentifier());
            TUIElement->setPos(pos.first,pos.second);
            configManager->addPosToPosList(Devices[i].UniqueIdentifier, pos, Parent->getUniqueIdentifier(), true);
            configManager->addPosToPosList(Devices[i].UniqueIdentifier, pos2, Parent->getUniqueIdentifier(), true);
            TUIElement->setHidden(!Devices[i].Visible);
        }
        else if (Devices[i].UI == mui::VRUIEnum)  // create VRUI-Element
        {
            if (Devices[i].Visible)                                            // visible
            {
                configManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->add(VRUIElement.get());
            }
            else                                                               // invisible
            {
                configManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->remove(VRUIElement.get());
            }
        }
        else
        {
            std::cerr << "PotiSlider::constructor(): Elementtype " << Devices[i].UI << " not found in Constructor." << std::endl;
        }
    }
}


// create VRUI-Element
void PotiSlider::createVRUIElement(const std::string label)
{
    VRUIElement.reset(new coPotiMenuItem (label, minVal, maxVal, value));
    VRUIElement->setMenuListener(this);
}

// create TUI-Element
void PotiSlider::createTUIElement(const std::string label, Container* Parent)
{
    TUIElement.reset(new coTUIFloatSlider(label, Parent->getTUIID()));
    TUIElement->setRange(minVal, maxVal);
    TUIElement->setValue(value);
    TUIElement->setEventListener(this);
}

// returns value
float PotiSlider::getValue()
{
    return value;
}

// sets new value
void PotiSlider::setValue(float newVal)
{
    value=newVal;
    TUIElement->setValue(value);
    VRUIElement->setValue(value);
}

// set position of TUI-Element
void PotiSlider::setPos(int posx, int posy)
{
    std::pair<int,int> pos(posx,posy);
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (Devices[i].UI == mui::TUIEnum)                      // TUI-Element
        {
            if (configManager->getIdentifierByPos(pos, Parent->getUniqueIdentifier()) != Devices[i].UniqueIdentifier)     // if is equal: Element is already at correct position
            {
                pos=configManager->getCorrectPos(pos, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
                configManager->preparePos(pos, Parent->getUniqueIdentifier());
                configManager->preparePos(std::pair<int,int>(pos.first+1, pos.second), Parent->getUniqueIdentifier());
                configManager->deletePosFromPosList(Devices[i].UniqueIdentifier);
                TUIElement->setPos(pos.first, pos.second);
                configManager->addPosToPosList(Devices[i].UniqueIdentifier, pos, Parent->getUniqueIdentifier(), false);
                configManager->addPosToPosList(Devices[i].UniqueIdentifier, std::pair<int,int>(pos.first+1, pos.second), Parent->getUniqueIdentifier(), false);
            }
        }
    }
}

// returns a pointer of TUIElement
coTUIElement* PotiSlider::getTUI()
{
    return TUIElement.get();
}


// set visible-value of named Elements
void PotiSlider::setVisible(bool visible, mui::UITypeEnum UI)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (UI == Devices[i].UI)                             // Element shall be changed
        {
            if (Devices[i].Visible != visible)                                     // visible-value changed
            {
                Devices[i].Visible = configManager->getCorrectVisible(visible, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
                if (Devices[i].UI == mui::TUIEnum){                  // TUI-Element
                    TUIElement->setHidden(!Devices[i].Visible);
                }
                else if (Devices[i].UI == mui::VRUIEnum)          // VRUI-Element
                {
                    if (Devices[i].Visible)
                    {
                        configManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->add(VRUIElement.get());
                    }
                    else
                    {
                        configManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->remove(VRUIElement.get());
                    }
                }
                else
                {
                    std::cerr << "PotiSlider::setVisible(): Elementtyp " << Devices[i].UI << " not found in setVisible(string, bool, bool)." << std::endl;
                }
            }
        }
    }
}

// set visible-value of all elements
void PotiSlider::setVisible(bool visible)
{
   for (size_t i=0; i<Devices.size(); ++i)
   {
       setVisible(visible, Devices[i].UI);
    }
}

// set visible-value of named elements
void PotiSlider::setLabel(std::string label, mui::UITypeEnum UI)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (UI == Devices[i].UI)                       // Element to be changed
        {
            Devices[i].Label=configManager->getCorrectLabel(label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
            if (Devices[i].UI == mui::TUIEnum)                  // TUI-Element
            {
                TUIElement->setLabel(Devices[i].Label);
            }
            else if (Devices[i].UI == mui::VRUIEnum)          // VRUI-Element
            {
                VRUIElement->setLabel(Devices[i].Label);
            }
            else
            {
                std::cerr << "PotiSlider::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string, std::string)." << std::endl;
            }
        }
    }
}

// set label for all UI-Elements
void PotiSlider::setLabel(std::string label)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        setLabel(label, Devices[i].UI);
     }
}

// returns the parent of this element
Container* PotiSlider::getParent()
{
    return Parent;
}

// returns the UniqueIdentifier of the element
std::string PotiSlider::getUniqueIdentifier()
{
    return UniqueIdentifier;
}

// called, if there is an interaction with the TUI
void PotiSlider::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == TUIElement.get())
    {
        VRUIElement->setValue(TUIElement->getValue());
        if (value!=TUIElement->getValue())
        {
            value = TUIElement->getValue();
        }
    }
    if (Listener)
    {
        Listener->muiEvent(this);
        Listener->muiValueChangeEvent(this);
    }
}

// called, if there is an interaction with the VRUI
void PotiSlider::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == VRUIElement.get())
    {
        TUIElement->setValue(VRUIElement->getValue());
        if (value!=VRUIElement->getValue())
        {
            value = VRUIElement->getValue();
        }
    }
    if (Listener)
    {
        Listener->muiEvent(this);
        Listener->muiValueChangeEvent(this);
    }
}
