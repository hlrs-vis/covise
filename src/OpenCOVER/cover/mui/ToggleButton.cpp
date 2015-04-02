// class, which creates a checkbox with label as VRUI
// creates a ToggleButton as TUI


#include <cover/coVRTui.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include "support/ConfigManager.h"
#include "Container.h"
#include "ToggleButton.h"
#include <iostream>

using namespace vrui;
using namespace std;
using namespace opencover;
using namespace mui;

// constructor:
ToggleButton::ToggleButton(const std::string UniqueIdentifier, Container* parent, const std::string label)
{
    constructor(UniqueIdentifier, parent, label);
}
ToggleButton::ToggleButton(const std::string UniqueIdentifier, Container* parent)
{
    constructor(UniqueIdentifier, parent, UniqueIdentifier);
}

// destructor:
ToggleButton::~ToggleButton()
{
    configManager->removeElement(Identifier);
    configManager->deletePosFromPosList(Identifier);
}

// underlying constructor
void ToggleButton::constructor(const std::string UniqueIdentifier, Container* parent, const std::string label)
{

    Label=label;
    Identifier=UniqueIdentifier;

    configManager = ConfigManager::getInstance();                          // necessary for parsing of configuration file, getting defaultvalues, parameters etc.

    configManager->addElement(UniqueIdentifier, this);    // adds the element to elementlist

    Parent=parent;
    State=false;

    // create defaultvalue or take from constructor:
    // VRUI-CAVE-Element:
    Devices.push_back(device());
    Devices[0].Device=mui::CAVEEnum;
    Devices[0].UI=mui::VRUIEnum;
    Devices[0].UniqueIdentifier=UniqueIdentifier;
    Devices[0].Visible = true;

    Devices[0].Label=configManager->getCorrectLabel(Label, Devices[0].UI, Devices[0].Device, Devices[0].UniqueIdentifier);
    Parent= configManager->getCorrectParent(Parent, Devices[0].UI, Devices[0].Device, Devices[0].UniqueIdentifier);

    // create VRUI-Element:
    createVRUIElement(Devices[0].Label);

    // TUI-Tablet-Element:
    Devices.push_back(device());
    Devices[1].Device=mui::TabletEnum;
    Devices[1].UI=mui::TUIEnum;
    Devices[1].UniqueIdentifier = UniqueIdentifier;
    Devices[1].Visible = true;

    Devices[1].Label=configManager->getCorrectLabel(Label, Devices[1].UI, Devices[1].Device, Devices[1].UniqueIdentifier);
    Parent= configManager->getCorrectParent(Parent, Devices[1].UI, Devices[1].Device, Devices[1].UniqueIdentifier);
    // create TUI-Element:
    createTUIElement(Devices[1].Label, configManager->getCorrectParent(Parent, Devices[1].UI, Devices[1].Device, Devices[1].UniqueIdentifier));

    // find and set correct parameter (get them from configuration file, if possible):
    for (size_t i=0; i<Devices.size(); ++i)
    {
        Devices[i].Visible = configManager->getCorrectVisible(Devices[i].Visible, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
        Devices[i].Label   = configManager->getCorrectLabel(Label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);

        // create UI-Elements:
        if (Devices[i].UI == mui::TUIEnum)                      // create TUI-Elements
        {
            std::pair<int,int> pos=configManager->getCorrectPos(Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier, Parent->getUniqueIdentifier());
            configManager->preparePos(pos, Parent->getUniqueIdentifier());
            TUIElement->setPos(pos.first,pos.second);
            if (configManager->existAttributeInConfigFile(Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier, mui::PosXEnum) && configManager->existAttributeInConfigFile(Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier, mui::PosYEnum))
            {
                configManager->addPosToPosList(Devices[i].UniqueIdentifier, pos, Parent->getUniqueIdentifier(), false);
            }
            else
            {
                configManager->addPosToPosList(Devices[i].UniqueIdentifier, pos, Parent->getUniqueIdentifier(), true);
            }
            TUIElement->setHidden(!Devices[i].Visible);
        }
        else if (Devices[i].UI == mui::VRUIEnum)                 // create VRUI-Element
        {
            if (Devices[i].Visible)                                             // visible
            {
                configManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->add(VRUIElement.get());
            }
            else                                                                // invisible
            {
                configManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->remove(VRUIElement.get());
            }
        }
        else
        {
            std::cerr << "ToggleButton::constructor(): Elementtype " << Devices[i].UI << " not found in constructor." << std::endl;
        }
    }
}


void ToggleButton::setPos(int posx, int posy)
{
    std::pair<int,int> pos(posx,posy);
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (Devices[i].UI == mui::TUIEnum)                       // TUI-Element
        {
            pos=configManager->getCorrectPos(pos, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
            if (configManager->getIdentifierByPos(pos, Parent->getUniqueIdentifier()) != Devices[i].UniqueIdentifier)     // if is equal: Element is already at correct position
            {
                configManager->preparePos(pos, Parent->getUniqueIdentifier());
                configManager->deletePosFromPosList(Devices[i].UniqueIdentifier);
                TUIElement->setPos(pos.first,pos.second);
                configManager->addPosToPosList(Devices[i].UniqueIdentifier, pos, Parent->getUniqueIdentifier(), false);
            }
        }
    }
}

// sets Label for all UI-elements
void ToggleButton::setLabel(std::string label)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        Devices[i].Label = configManager->getCorrectLabel(label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
        if (Devices[i].UI == mui::TUIEnum)                       // TUI-Element
        {
            TUIElement->setLabel(Devices[i].Label);
        }
        else if (Devices[i].UI == mui::VRUIEnum)                 // VRUI-Element
        {
            VRUIElement->setLabel(Devices[i].Label);
        }
        else
        {
            std::cerr << "ToggleButton::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string)." << std::endl;
        }
    }
}

// sets Label for named UI-elements
void ToggleButton::setLabel(std::string label, std::string UI)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (UI.find(Devices[i].UI) != std::string::npos)                        // Element to be changed
        {
            Devices[i].Label=configManager->getCorrectLabel(label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
            if (Devices[i].UI == mui::TUIEnum)                   // TUI-Element
            {
                TUIElement->setLabel(Devices[i].Label);
            }
            else if (Devices[i].UI == mui::VRUIEnum)             // VRUI-Element
            {
                TUIElement->setLabel(Devices[i].Label);
            }
            else
            {
                std::cerr << "PotiSlider::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string, std::string)." << std::endl;
            }
        }
    }
}

// sets the visible-value for all UI-elements
void ToggleButton::setVisible(bool visible)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (Devices[i].Visible != visible)                                     // Value changed
        {
            Devices[i].Visible = configManager->getCorrectVisible(visible, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
            if (Devices[i].UI == mui::TUIEnum)                  // TUI-Element
            {
                TUIElement->setHidden(!Devices[i].Visible);
            }
            else if (Devices[i].UI == mui::VRUIEnum)            // VRUI-Element
            {
                if (Devices[i].Visible)                                        // Visible
                {
                    configManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->add(VRUIElement.get());
                }
                else                                                           // Invisible
                {
                    configManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->remove(VRUIElement.get());
                }
            }
            else
            {
                std::cerr << "PotiSlider::setVisible(): Elementtype " << Devices[i].UI << " not found in setVisible(bool)." << std::endl;
            }
        }
    }
}

// sets the visible-value for the named UI-Elements
void ToggleButton::setVisible(bool visible, std::string UI)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (UI.find(Devices[i].UI)!=std::string::npos)                         // element shall be changed
        {
            if (Devices[i].Visible != visible)                                 // visible-value changed
            {
                Devices[i].Visible = configManager->getCorrectVisible(visible, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
                if (Devices[i].UI == mui::TUIEnum)              // TUI-Element
                {
                    TUIElement->setHidden(!Devices[i].Visible);
                }
                else if (Devices[i].UI == mui::VRUIEnum)         // VRUI-Element
                {
                    if (Devices[i].Visible)
                    {
                        configManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->add(VRUIElement.get());
                    }
                }
                else
                {
                    configManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->remove(VRUIElement.get());
                }
            }
            else
            {
                std::cerr << "ToggleButton::setVisible(): Elementtype " << Devices[i].UI << " not found in setVisible(string, bool)." << std::endl;
            }
        }
    }
}

// returns the parent-element of the toggleButton
Container* ToggleButton::getParent()
{
    return Parent;
}

// creates the VRUI-Element
void ToggleButton::createVRUIElement(const std::string label)
{
    VRUIElement.reset(new coCheckboxMenuItem(label, true));
    VRUIElement->setMenuListener(this);                                         // create listener
    VRUIElement->setState(State);
}

// creates the TUI-Element
void ToggleButton::createTUIElement(const std::string label, Container* parent)
{
    TUIElement.reset(new coTUIToggleButton(label, parent->getTUIID()));
    TUIElement->setEventListener(this);                                         // create listener
    TUIElement->setState(State);
}

void ToggleButton::setState(bool stat)
{
    State=stat;
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (Devices[i].UI == mui::VRUIEnum)                     // VRUI-Element
        {
            VRUIElement->setState(stat);
        }
        else if(Devices[i].UI == mui::TUIEnum)                 // TUI-Element
        {
            TUIElement->setState(stat);
        }
        else
        {
            std::cerr << "ToggleButton::setState(): Elementtype " << Devices[i].UI << " not found in setClicked(bool)." << std::endl;
        }
    }
}

bool ToggleButton::getState()
{
    return State;
}

std::string ToggleButton::getUniqueIdentifier()
{
    return Identifier;
}

coTUIElement *ToggleButton::getTUI()
{
    return TUIElement.get();
};

//*****************************************************************************************************************************
// Listener
//*****************************************************************************************************************************


// called, if there is an interaction with the tablet
void ToggleButton::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == TUIElement.get())                                            // there is an interaction with the tablet
    {
        VRUIElement->setState(!(VRUIElement->getState()));                      // adjust status of VRUI-Element
        State=!State;
    }
    if (Listener)
    {
        Listener->muiEvent(this);
    }
}

// called, if there is an interaction with the VRUI
void ToggleButton:: menuEvent(coMenuItem *menuItem)
{
    if (menuItem == VRUIElement.get())                                          // there is an interaction with the VRUI
    {
        TUIElement->setState(!(TUIElement->getState()));                        // adjust status of TUI-ELement
        State=!State;
    }
    if (Listener)
    {
        Listener->muiEvent(this);
    }
}

void ToggleButton::activate()
{
    State=true;

    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (Devices[i].UI == mui::TUIEnum)                      // TUIElement
        {
            TUIElement->setState(true);
        }
        else if (Devices[i].UI == mui::VRUIEnum)
        {
            VRUIElement->setState(true);
        }
    }
}

void ToggleButton::deactivate()
{
    State=false;

    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (Devices[i].UI == mui::TUIEnum)                      // TUIElement
        {
            TUIElement->setState(false);
        }
        else if (Devices[i].UI == mui::VRUIEnum)
        {
            VRUIElement->setState(false);
        }
    }
}

void ToggleButton::click()
{
    State=!State;
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (Devices[i].UI == mui::TUIEnum)
        {                      // TUIElement
            TUIElement->setState(!TUIElement->getState());
        }
        else if (Devices[i].UI == mui::VRUIEnum)              // VRUIElement
        {
            VRUIElement->setState(!VRUIElement->getState());
        }
        else
        {
            std::cerr << "ToggleButton::click(): Elementtyp " << Devices[i].UI << " not found in click()." << std::endl;
        }
    }
}
