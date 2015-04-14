// class, which creates a new menu as VRUI
// creates a Frame as TUI


#include <cover/coVRTui.h>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include "Frame.h"
#include "support/ConfigManager.h"

using namespace mui;

// constructor:
Frame::Frame(const std::string uniqueIdentifier, Container* parent)
    : mui::Container(uniqueIdentifier, parent)
{
    // VRUI:
    VRUIMenuItem.reset(new vrui::coSubMenuItem(uniqueIdentifier));
    VRUIContainer.reset(new vrui::coRowMenu(uniqueIdentifier.c_str()));
    static_cast<vrui::coSubMenuItem*>(VRUIMenuItem.get())->setMenu(VRUIContainer.get());

    // TUI:
    if (parent == NULL)
    {
        TUIElement.reset(new opencover::coTUIFrame(storage[mui::TUIEnum].label, opencover::coVRTui::instance()->mainFolder->getID()));
    }
    else
    {
        TUIElement.reset(new opencover::coTUIFrame(storage[mui::TUIEnum].label, parent->getTUIID()));
    }
}

// destructor:
Frame::~Frame()
{
}

mui::Frame* Frame::create(std::string uniqueIdentifier, Container *parent)
{
    Frame *frame = new Frame(uniqueIdentifier, parent);
    frame->init();
    return frame;
}

