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

using namespace mui;

// constructor:
ToggleButton::ToggleButton(const std::string uniqueIdentifier, Container* parent)
    : mui::Element(uniqueIdentifier, parent)
{
    // VRUI:
    VRUIMenuItem.reset(new vrui::coCheckboxMenuItem(storage[mui::VRUIEnum].label, true));
    VRUIMenuItem->setMenuListener(this);                                         // create listener

    // TUI:
    TUIElement.reset(new opencover::coTUIToggleButton(storage[mui::TUIEnum].label, parent->getTUIID()));
    TUIElement->setEventListener(this);

    state = false;
    setState(state);
}

mui::ToggleButton* ToggleButton::create(std::string uniqueIdentifier, Container *parent)
{
    ToggleButton *toggleButton = new ToggleButton(uniqueIdentifier, parent);
    toggleButton->init();
    return toggleButton;
}

// destructor:
ToggleButton::~ToggleButton()
{
}

void ToggleButton::setState(bool stat)
{
    state=stat;
    static_cast<vrui::coCheckboxMenuItem*>(VRUIMenuItem.get())->setState(stat);                           // VRUI-Element
    static_cast<opencover::coTUIToggleButton*>(TUIElement.get())->setState(stat);                             // TUI-Element
}

bool ToggleButton::getState()
{
    return state;
}

//*****************************************************************************************************************************
// Listener
//*****************************************************************************************************************************
void ToggleButton::muiEvent(Element *muiItem)
{
    if (muiItem == this)
    {
        state = !state;
        setState(state);
    }
}

// called, if there is an interaction with the tablet
void ToggleButton::tabletEvent(opencover::coTUIElement *tUIItem)
{
    if (tUIItem == TUIElement.get())                        // there is an interaction with the tablet
    {
        state = static_cast<opencover::coTUIToggleButton*>(TUIElement.get())->getState();
        static_cast<vrui::coCheckboxMenuItem*>(VRUIMenuItem.get())->setState(state);                      // adjust status of VRUI-Element
    }
    if (listener)
    {
        listener->muiEvent(this);
    }
}

// called, if there is an interaction with the VRUI
void ToggleButton::menuEvent(vrui::coMenuItem *menuItem)
{
    if (menuItem == VRUIMenuItem.get())                     // there is an interaction with the VRUI
    {
        state = static_cast<vrui::coCheckboxMenuItem*>(VRUIMenuItem.get())->getState();
        static_cast<opencover::coTUIToggleButton*>(TUIElement.get())->setState(state);                        // adjust status of TUI-ELement
    }
    if (listener)
    {
        listener->muiEvent(this);
    }
}

