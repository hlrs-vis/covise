// class, which creates a menuentry and a new submenu in VR
// creates a new tab in TUI

#include <cover/coVRTui.h>
#include <OpenVRUI/coRowMenu.h>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coMenuItem.h>
#include "support/ConfigManager.h"
#include "TabFolder.h"

#include <iostream>

using namespace mui;

// constructor:
TabFolder::TabFolder(const std::string uniqueIdentifier, Container* parent)
    : mui::Container(uniqueIdentifier, parent)
{
    // VRUI:
    VRUIMenuItem.reset(new vrui::coSubMenuItem(storage[mui::VRUIEnum].label));
    VRUIContainer.reset(new vrui::coRowMenu(storage[mui::VRUIEnum].label.c_str()));
    static_cast<vrui::coSubMenuItem*>(VRUIMenuItem.get())->setMenu(VRUIContainer.get());

    // TUI:
    if (parent == NULL)
    {
        TUITab.reset(new opencover::coTUITab(storage[mui::TUIEnum].label, opencover::coVRTui::instance()->mainFolder->getID()));
        TUITab->setPos(0,0);
        TUIElement.reset(new opencover::coTUITabFolder(storage[mui::TUIEnum].label, TUITab->getID()));
    }
    else
    {
        TUITab.reset(new opencover::coTUITab(storage[mui::TUIEnum].label, parent->getTUIID()));
        TUITab->setPos(0,0);
        TUIElement.reset(new opencover::coTUITabFolder(storage[mui::TUIEnum].label, TUITab->getID()));
    }
}

mui::TabFolder* TabFolder::create(std::string uniqueIdentifier, Container *parent)
{
    TabFolder *tabFolder = new TabFolder(uniqueIdentifier, parent);
    tabFolder->init();
    return tabFolder;
}

// destructor:
TabFolder::~TabFolder(){
}

