// class, which creates a label as VRUI and TUI

#include "LabelElement.h"
#include "support/ConfigManager.h"
#include "Container.h"
#include <OpenVRUI/coLabelMenuItem.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coMenu.h>

#include <iostream>

using namespace mui;

// constructor:
LabelElement::LabelElement(std::string uniqueIdentifier, Container* parent)
    : mui::Element(uniqueIdentifier, parent)
{
    // VRUI:
    VRUIMenuItem.reset(new vrui::coLabelMenuItem(storage[mui::VRUIEnum].label));
    VRUIMenuItem->setMenuListener(this);

    // TUI:
    TUIElement.reset(new opencover::coTUILabel(storage[mui::TUIEnum].label, parent->getTUIID()));
    TUIElement->setEventListener(this);
}

mui::LabelElement* LabelElement::create(std::string uniqueIdentifier, Container *parent)
{
    LabelElement *labelElement = new LabelElement(uniqueIdentifier, parent);
    labelElement->init();
    return labelElement;
}

// destructor:
LabelElement::~LabelElement()
{
}
