// class, which creates a Poti as VRUI-Element
// creates a Slider as TUI-Element

#include "PotiSlider.h"
#include "support/ConfigManager.h"
#include "Container.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRTui.h>
#include <OpenVRUI/coMenuItem.h>

#include <iostream>

using namespace mui;

// constructor:
PotiSlider::PotiSlider(const std::string uniqueIdentifier, Container* parent, float min, float max, float defaultValue)
    : mui::Element(uniqueIdentifier, parent)
{
    minVal = min;
    maxVal = max;
    value = defaultValue;

    // VRUI:
    VRUIMenuItem.reset(new vrui::coPotiMenuItem(storage[mui::VRUIEnum].label, minVal, maxVal, value));
    VRUIMenuItem->setMenuListener(this);

    // TUI:
    TUIElement.reset(new opencover::coTUIFloatSlider(storage[mui::TUIEnum].label, parent->getTUIID()));
    static_cast<opencover::coTUIFloatSlider*>(TUIElement.get())->setRange(minVal, maxVal);
    static_cast<opencover::coTUIFloatSlider*>(TUIElement.get())->setValue(value);
    static_cast<opencover::coTUIFloatSlider*>(TUIElement.get())->setEventListener(this);
}

mui::PotiSlider* PotiSlider::create(std::string uniqueIdentifier, Container *parent, float min, float max, float defaultValue)
{
    PotiSlider *potiSlider = new PotiSlider(uniqueIdentifier, parent, min, max, defaultValue);
    potiSlider->init();
    return potiSlider;
}

// destructor
PotiSlider::~PotiSlider()
{
}

// called, if there is an interaction with the TUI
void PotiSlider::tabletEvent(opencover::coTUIElement *tUIItem)
{
    if (tUIItem == TUIElement.get())
    {
        static_cast<vrui::coPotiMenuItem*>(VRUIMenuItem.get())->setValue(static_cast<opencover::coTUIFloatSlider*>(TUIElement.get())->getValue());
        value = static_cast<opencover::coTUIFloatSlider*>(TUIElement.get())->getValue();
    }
    if (listener)
    {
        listener->muiEvent(this);
        listener->muiValueChangeEvent(this);
    }
}

// called, if there is an interaction with the VRUI
void PotiSlider::menuEvent(vrui::coMenuItem *menuItem)
{
    if (menuItem == VRUIMenuItem.get())
    {
        static_cast<opencover::coTUIFloatSlider*>(TUIElement.get())->setValue(static_cast<vrui::coPotiMenuItem*>(VRUIMenuItem.get())->getValue());
        value = static_cast<vrui::coPotiMenuItem*>(VRUIMenuItem.get())->getValue();
    }
    if (listener)
    {
        listener->muiEvent(this);
        listener->muiValueChangeEvent(this);
    }
}

float PotiSlider::getValue()
{
    return value;
}

void PotiSlider::setValue(float newVal)
{
    value = newVal;
    static_cast<opencover::coTUIFloatSlider*>(TUIElement.get())->setValue(value);
    static_cast<vrui::coPotiMenuItem*>(VRUIMenuItem.get())->setValue(value);
    listener->muiEvent(this);
    listener->muiValueChangeEvent(this);
}
