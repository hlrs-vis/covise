// class, which creates a menuentry and a new submenu in VR
// creates a new tab in TUI

#include <cover/coVRTui.h>
#include <OpenVRUI/coRowMenu.h>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coMenuItem.h>
#include "support/ConfigManager.h"
#include "TabFolder.h"

#include <iostream>

using namespace opencover;
using namespace vrui;
using namespace mui;

// constructors (with parent):
TabFolder::TabFolder(const std::string UniqueIdentifier, Container* parent,  const std::string label)
{
    configManager= NULL;
    Label = label;
    ParentConstructor(UniqueIdentifier, parent);
}
TabFolder::TabFolder(const std::string UniqueIdentifier, Container* parent)
{
    configManager= NULL;
    Label = UniqueIdentifier;
    ParentConstructor(UniqueIdentifier, parent);
}

// constructors (wihtout parents -> new entry in main-menu);
TabFolder::TabFolder(const std::string UniqueIdentifier, std::string label)
{
    std::cout << "TabFolder: 0.0" << std::endl;
    configManager= NULL;
    Label = label;

    std::cout << "TabFolder: 0.1" << std::endl;
    constructor(UniqueIdentifier);

    std::cout << "TabFolder: 0.2" << std::endl;
}

TabFolder::TabFolder(const std::string UniqueIdentifier)
{
    configManager= NULL;
    Label = UniqueIdentifier;
    constructor(UniqueIdentifier);
}

// destructor:
TabFolder::~TabFolder(){
configManager->deletePosFromPosList(Identifier);
configManager->removeElement(Identifier);
}

// underlying constructors:
void TabFolder::ParentConstructor(const std::string UniqueIdentifier,  Container* parent)
{
    configManager= ConfigManager::getInstance();

    Parent=parent;
    Identifier=UniqueIdentifier;

    configManager->addElement(UniqueIdentifier, this);

    // create default-values or tanken from constructor:
    // VRUI-CAVE-Element:
    Devices.push_back(device());
    Devices[0].Device=mui::CAVEEnum;
    Devices[0].UI=mui::VRUIEnum;
    Devices[0].UniqueIdentifier = UniqueIdentifier;
    Devices[0].Visible = true;

    Devices[0].Label= configManager->getCorrectLabel(Label, Devices[0].UI, Devices[0].Device, Devices[0].UniqueIdentifier);
    Parent=configManager->getCorrectParent(Parent, Devices[0].UI, Devices[0].Device, Devices[0].UniqueIdentifier);
    // create VRUI-Element:
    Submenu.reset(new vrui::coRowMenu(Devices[0].Label.c_str()));
    SubmenuItem.reset(new vrui::coSubMenuItem(Devices[0].Label));

    // TUI-Tablet-Element:
    Devices.push_back(device());
    Devices[1].Device=mui::TabletEnum;
    Devices[1].UI=mui::TUIEnum;
    Devices[1].UniqueIdentifier = UniqueIdentifier;
    Devices[1].Visible = true;

    Devices[1].Label = configManager->getCorrectLabel(Label, Devices[1].UI, Devices[1].Device, Devices[1].UniqueIdentifier);
    Parent=configManager->getCorrectParent(Parent, Devices[0].UI, Devices[0].Device, Devices[0].UniqueIdentifier);
    // create TUI-Element:
    TUIElement.reset(new coTUITabFolder(Devices[1].Label, Parent->getTUIID()));


    // find and set correct parameter (get them from configuration file, if possible):
    for (size_t i=0; i<Devices.size(); ++i)
    {
        Devices[i].Visible = configManager->getCorrectVisible(Devices[i].Visible, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
        Devices[i].Label = configManager->getCorrectLabel(Label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);

        // create the UI-Elements
        if  (Devices[i].UI==mui::TUIEnum)                       // create TUI-Elements
        {
            std::pair<int,int> pos=configManager->getCorrectPos(Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier, Parent->getUniqueIdentifier());
            TUIElement->setPos(pos.first,pos.second);
            configManager->addPosToPosList(Devices[i].UniqueIdentifier, pos, Parent->getUniqueIdentifier(), true);
            TUIElement->setHidden(!Devices[i].Visible);
        }
        else if (Devices[i].UI==mui::VRUIEnum)                  // create VRUI-Elements
        {
            SubmenuItem->setMenu(Submenu.get());

            if (Devices[i].Visible)                                            // visible
            {
                configManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->add(SubmenuItem.get());
            }
            else                                                               // invisible
            {
                configManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->remove(SubmenuItem.get());
            }
        }
        else
        {
            std::cerr << "TabFolder::ParentConstructor(): " << Devices[i].UI << " not found in Constructor." << std::endl;
        }
    }
}

void TabFolder::constructor(const std::string identifier)
{

    configManager= ConfigManager::getInstance();
    configManager->addElement(identifier, this);
    Identifier=identifier;

    // create defaultvalue or take from constructor:
    // VRUI-CAVE-Element:
    Devices.push_back(device());
    Devices[0].Device=mui::CAVEEnum;
    Devices[0].UI=mui::VRUIEnum;
    Devices[0].UniqueIdentifier = identifier;

    Devices[0].Label= configManager->getCorrectLabel(Label, Devices[0].UI, Devices[0].Device, Devices[0].UniqueIdentifier);
    Parent = configManager->getCorrectParent(NULL, Devices[0].UI, Devices[0].Device, Devices[0].UniqueIdentifier);
    // create VRUI-Element:
    Submenu.reset(new vrui::coRowMenu(Devices[0].Label.c_str()));
    SubmenuItem.reset(new vrui::coSubMenuItem(Devices[0].Label));

    // TUI-Tablet-Element:
    Devices.push_back(device());
    Devices[1].Device=mui::TabletEnum;
    Devices[1].UI=mui::TUIEnum;
    Devices[1].UniqueIdentifier = identifier;
    Devices[1].Label = configManager->getCorrectLabel(Label, Devices[1].UI, Devices[1].Device, Devices[1].UniqueIdentifier);
    Parent = configManager->getCorrectParent(NULL, Devices[1].UI, Devices[1].Device, Devices[1].UniqueIdentifier);
    // create TUI-Element:

    if (Parent)                                         // parent was declated in configuration file
    {
        TUITab.reset(new coTUITab(Devices[1].Label, Parent->getTUIID()));
        TUIElement.reset(new coTUITabFolder(Devices[1].Label, TUITab->getID()));
    }
    else if (!Parent)                                  // no parent declared->mainmenu
    {
        TUITab.reset(new coTUITab(Devices[1].Label, coVRTui::instance()->mainFolder->getID()));
        TUIElement.reset(new coTUITabFolder(Devices[1].Label, TUITab->getID()));
    }
    else
    {
        std::cerr << "TabFolder::constructor(): Parent not found" << std::endl;
    }

    // find and set correct parameter (get them from configuration file, if possible):
    for (size_t i=0; i<Devices.size(); ++i)
    {
        Devices[i].Visible = configManager->getCorrectVisible(Devices[i].Visible, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
        Devices[i].Label = configManager->getCorrectLabel(Label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);

        // create UI-Elements:
        if  (Devices[i].UI==mui::TUIEnum)  // create TUI-Element
        {
            std::pair<int,int> pos=configManager->getCorrectPos(Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier, mui::MainWindowEnum);
            TUITab->setPos(pos.first,pos.second);
            configManager->addPosToPosList(Devices[i].UniqueIdentifier, pos, mui::MainWindowEnum, true);
            TUITab->setHidden(!Devices[i].Visible);
        }
        else if (Devices[i].UI==mui::VRUIEnum)  // create VRUI-Elemente erstellen
        {
            SubmenuItem->setMenu(Submenu.get());

            if (Devices[i].Visible)                                            // visible
            {
                Parent=configManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
                if (!Parent)
                {
                    cover->getMenu()->add(SubmenuItem.get());
                }
                else if (Parent)
                {
                    Parent->getVRUI()->add(SubmenuItem.get());
                }
                else
                {
                    std::cerr << "TabFolder::constructor: wrong Parent" << std::endl;
                }
            }
            else                                                               // invisible
            {
                Parent=configManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
                if (!Parent)
                {
                    cover->getMenu()->remove(SubmenuItem.get());
                }
                else if (Parent)
                {
                    Parent->getVRUI()->remove(SubmenuItem.get());
                }
                else
                {
                    std::cerr << "TabFolder::constructor: wrong Parent" << std::endl;
                }
            }
        }
        else
        {
            std::cerr << "TabFolder::constructor: " << Devices[i].UI << " not found in Constructor." << std::endl;
        }
    }
}


//  returns the ID of the TUIElements
int TabFolder::getTUIID()
{
    return TUIElement->getID();
}

// returns a pointer to the TUIElement
coTUIElement* TabFolder::getTUI()
{
    return TUIElement.get();
}

// returns the VRUI-Parent
coMenu* TabFolder::getVRUI()
{
    return Submenu.get();
}

// sets the label for all UI-elements
void TabFolder::setLabel(std::string label)
{
    std::string UI;
    for (size_t i=0; i<Devices.size(); ++i)
    {
        UI.append(Devices[i].UI + " ");
    }
    setLabel(label, UI);
}

// sets the Label for all named Elements
void TabFolder::setLabel(std::string label, std::string UI)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (UI.find(Devices[i].UI) != std::string::npos)                       // Element to be changed
        {
            Devices[i].Label=configManager->getCorrectLabel(label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
            if (Devices[i].UI == mui::TUIEnum)                  // TUI-Element
            {
                TUITab->setLabel(Devices[i].Label);
            }
            else if (Devices[i].UI == mui::VRUIEnum)          // VRUI-Element
            {
                SubmenuItem->setLabel(Devices[i].Label);
                Submenu->updateTitle(Devices[i].Label.c_str());
            }
            else
            {
                std::cerr << "MainElement::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string, std::string)." << std::endl;
            }
        }
    }
}

// sets the visibility for all elements
void TabFolder::setVisible(bool visible)
{
    std::string UI;
    for (size_t i=0; i<Devices.size(); ++i)
    {
        UI.append(Devices[i].UI + " ");
    }
    setVisible(visible, UI);
}

// sets the visibility for the named elements
void TabFolder::setVisible(bool visible, std::string UI)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (UI.find(Devices[i].UI)!=std::string::npos)                             // Element to be changed
        {
            if (Devices[i].Visible != visible)
            {
                Devices[i].Visible = configManager->getCorrectVisible(visible, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
                if (Devices[i].UI == mui::TUIEnum)                  // TUI-Element
                {
                    TUITab->setHidden(!Devices[i].Visible);
                }
                else if (Devices[i].UI == mui::VRUIEnum)          // VRUI-Element
                {
                    if (Devices[i].Visible)
                    {
                        cover->getMenu()->add(SubmenuItem.get());
                    }
                    else
                    {
                        cover->getMenu()->remove(SubmenuItem.get());
                    }
                }
                else
                {
                    std::cerr << "TabFolder::setVisible(): Elementtyp " << Devices[i].UI << " wurde in setVisible(string, bool, bool) nicht gefunden." << std::endl;
                }
            }
        }
    }
}

// sets Position
void TabFolder::setPos(int posx, int posy)
{
    std::pair<int,int>pos(posx,posy);
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (Devices[i].UI == mui::TUIEnum)                      // TUI-Element
        {
            if (configManager->getIdentifierByPos(pos, Parent->getUniqueIdentifier()) != Devices[i].UniqueIdentifier)     // if is equal: Element is already at correct position
            {
                pos=configManager->getCorrectPos(pos, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
                configManager->preparePos(pos, Parent->getUniqueIdentifier());
                configManager->deletePosFromPosList(Devices[i].UniqueIdentifier);
                TUIElement->setPos(pos.first,pos.second);
                configManager->addPosToPosList(Devices[i].UniqueIdentifier, pos, Parent->getUniqueIdentifier(), false);
            }
        }
    }
}

std::string TabFolder::getUniqueIdentifier()
{
    return Identifier;
}

Container *TabFolder::getParent()
{
    return Parent;
}
