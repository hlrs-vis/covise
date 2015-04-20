// class, which creats a menuentry and a new menu as VRUI-Element
// creates a new Tab as TUI-Element

#include <cover/coVRTui.h>
#include <OpenVRUI/coRowMenu.h>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coMenuItem.h>
#include "support/ConfigManager.h"
#include "support/DefaultValues.h"
#include "Container.h"
#include "Tab.h"

#include <iostream>

using namespace mui;

// constructor:
Tab::Tab(const std::string uniqueIdentifier, Container* parent)
    : mui::Container(uniqueIdentifier, parent)
{
    // VRUI:
    VRUIMenuItem.reset(new vrui::coSubMenuItem(uniqueIdentifier));
    VRUIContainer.reset(new vrui::coRowMenu(uniqueIdentifier.c_str()));
    static_cast<vrui::coSubMenuItem*>(VRUIMenuItem.get())->setMenu(VRUIContainer.get());

    // TUI:
    if (parent == NULL)                         // add Element to mainMenu
    {
        TUIElement.reset(new opencover::coTUITab(storage[mui::TUIEnum].label, opencover::coVRTui::instance()->mainFolder->getID()));
    }
    else
    {
        TUIElement.reset(new opencover::coTUITab(storage[mui::TUIEnum].label, parent->getTUIID()));
    }

    configManager->addElement(uniqueIdentifier, this);
}

/**
 * @brief Tab::create
 * @param uniqueIdentifier
 * @param parent
 * static function to create the tab-element.
 */
mui::Tab* Tab::create(std::string uniqueIdentifier, mui::Container* parent)
{
    Tab *tab = new Tab(uniqueIdentifier, parent);
    tab->init();
    return tab;
}

// destructor:
Tab::~Tab()
{
}

