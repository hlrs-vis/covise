#include "Container.h"
#include "Element.h"
#include <cover/coTabletUI.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenu.h>

using namespace mui;

// constructor
Container::Container(std::string uniqueIdentifier, Container *parent)
    : mui::Element(uniqueIdentifier, parent)
{
}

// destructor
Container::~Container()
{
}

// get ID of TUI-Element
int Container::getTUIID()
{
    return TUIElement->getID();
}

// get Pointer to VR-Parent
vrui::coMenu* Container::getVRUI()
{
    return VRUIContainer.get();
}

// set label of named UI-Elements
void Container::setBackendLabel(std::string label, mui::UITypeEnum UI)
{
    storage[UI].label=configManager->getCorrectLabel(label, UI, storage[UI].device, storage[UI].uniqueIdentifier);
    switch (UI)
    {
    case mui::TUIEnum:                                                  // TUIElement
        TUIElement->setLabel(storage[UI].label);
        break;

    case mui::VRUIEnum:                                                 // VRUIElement
        VRUIMenuItem->setLabel(std::string(storage[UI].label + "..."));
        VRUIContainer->updateTitle(storage[UI].label.c_str());
        break;

    case mui::muiUICounter:
        break;
    }
}

// set label of all UI-Elements
void Container::setLabel(std::string label)
{
    for (size_t i=0; i<storage.size(); ++i)
    {
        setBackendLabel(label, mui::UITypeEnum(i));
    }
}
