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

using namespace opencover;
using namespace vrui;
using namespace mui;

// constructors (with parent):
Tab::Tab(const std::string UniqueIdentifier, Container* parent,  const std::string label)
{
    Label = label;
    ParentConstructor(UniqueIdentifier, parent);
}

Tab::Tab(const std::string UniqueIdentifier, Container* parent)
{
    Label = UniqueIdentifier;
    ParentConstructor(UniqueIdentifier, parent);
}

// constructors(without parent => create in mainmenu);
Tab::Tab(const std::string UniqueIdentifier, const std::string label)
{
    Label = label;
    constructor(UniqueIdentifier);
}

Tab::Tab(const std::string UniqueIdentifier)
{
    Label = UniqueIdentifier;
    constructor(UniqueIdentifier);
}


// destructor:
Tab::~Tab()
{
    configManager->removeElement(Identifier);
    configManager->deletePosFromPosList(Identifier);
}

// underlying constructor
void Tab::ParentConstructor(const std::string UniqueIdentifier,  Container* parent)
{
    configManager=ConfigManager::getInstance();
    Identifier= UniqueIdentifier;
    Parent=parent;

    // add element to ElementList
    configManager->addElement(UniqueIdentifier, this);

    // create defaultvalue or take from constructor:
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
    TUIElement.reset(new opencover::coTUITab(Devices[1].Label, Parent->getTUIID()));

    // find and set correct parameter (get them from configuration file, if possible):
    for (size_t i=0; i<Devices.size(); ++i)
    {
        Devices[i].Visible = configManager->getCorrectVisible(Devices[i].Visible, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
        Devices[i].Label = configManager->getCorrectLabel(Label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);

        // create UI-Elements:
        if  (Devices[i].UI==mui::TUIEnum)                           // create TUI-Element
        {
            std::pair<int,int> pos=configManager->getCorrectPos(Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier, Parent->getUniqueIdentifier());
            TUIElement->setPos(pos.first,pos.second);
            configManager->addPosToPosList(Devices[i].UniqueIdentifier, pos, Parent->getUniqueIdentifier(), true);
            TUIElement->setHidden(!Devices[i].Visible);
        }
        else if (Devices[i].UI==mui::VRUIEnum)                      // create VRUI-Elemente
        {
            SubmenuItem->setMenu(Submenu.get());

            if (Devices[i].Visible)                                                // visible
            {
                configManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->add(SubmenuItem.get());
            }
            else                                                                   // invisible
            {
                configManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->remove(SubmenuItem.get());
            }
        }
        else
        {
            std::cerr << "Tab::ParentConstructor(): " << Devices[i].UI << " not found in Constructor." << std::endl;
        }
    }
}

void Tab::constructor(const std::string UniqueIdentifier)
{
    configManager= ConfigManager::getInstance();
    Identifier=UniqueIdentifier;
    configManager->addElement(Identifier, this);
    Parent = NULL;

    // create defaultvalue or take from constructor:
    // VRUI-CAVE-Element:
    Devices.push_back(device());
    Devices[0].Device=mui::CAVEEnum;
    Devices[0].UI=mui::VRUIEnum;
    Devices[0].UniqueIdentifier = UniqueIdentifier;
    Devices[0].Visible=true;

    Devices[0].Label= configManager->getCorrectLabel(Label, Devices[0].UI, Devices[0].Device, Devices[0].UniqueIdentifier);
    Parent=configManager->getCorrectParent(NULL, Devices[0].UI, Devices[0].Device, Devices[0].UniqueIdentifier);

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
    Parent=configManager->getCorrectParent(NULL, Devices[1].UI, Devices[1].Device, Devices[1].UniqueIdentifier);
    // create TUI-Element:
    if (Parent != NULL)                                                         // Parent was dedicated in configuration file
    {
        TUIElement.reset(new opencover::coTUITab(Devices[1].Label, Parent->getTUIID()));
    }
    else if (Parent == NULL)                                                  // Parent is not dedicated -> mainmenu
    {
        TUIElement.reset(new opencover::coTUITab(Devices[1].Label, coVRTui::instance()->mainFolder->getID()));
    }
    else
    {
        std::cerr << "Tab::constructor(): Parent not found" << std::endl;
    }


    // find and set correct parameter (get them from configuration file, if possible):
    for (size_t i=0; i<Devices.size(); ++i)
    {
        Devices[i].Visible = configManager->getCorrectVisible(Devices[i].Visible, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
        Devices[i].Label = configManager->getCorrectLabel(Label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);

        // create UI-Elemente:
        if  (Devices[i].UI==mui::TUIEnum)                           // create TUI-Elements
        {
            std::pair<int,int> pos=configManager->getCorrectPos(Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier, mui::MainWindowEnum);
            TUIElement->setPos(pos.first,pos.second);
            configManager->addPosToPosList(Devices[i].UniqueIdentifier, pos, mui::getKeywordAttribute(mui::MainWindowEnum), true);
            TUIElement->setHidden(!Devices[i].Visible);
        }
        else if (Devices[i].UI==mui::VRUIEnum)                      // create VRUI-Elements
        {
            SubmenuItem->setMenu(Submenu.get());

            if (Devices[i].Visible)                                                // visible
            {
                Parent=configManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
                if (Parent == NULL)
                {
                    cover->getMenu()->add(SubmenuItem.get());
                }
                else if (Parent != NULL)
                {
                    Parent->getVRUI()->add(SubmenuItem.get());
                }
                else
                {
                    std::cerr << "TabFolder::constructor: wrong Parent";
                }
            }
            else
            {                                                                   // invisible
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
                    std::cerr << "TabFolder::constructor: wrong Parent";
                }
            }
        }
        else
        {
            std::cerr << "TabFolder::constructor: " << Devices[i].UI << " not found in Constructor." << std::endl;
        }
    }
}

// returns the ID of the TUIElement
int Tab::getTUIID()
{
    return TUIElement->getID();
}

// returns a pointer to TUIElement
coTUIElement* Tab::getTUI()
{
    return TUIElement.get();
}

// return VRUIElement
coMenu* Tab::getVRUI()
{
    return Submenu.get();
}

// set label of all UI-Elements
void Tab::setLabel(std::string label)
{
    std::string UI;
    for (size_t i=0; i<Devices.size(); ++i)
    {
        UI.append(Devices[i].UI + " ");
    }
    setLabel(label, UI);
}

// set label of named UI-Elements
void Tab::setLabel(std::string label, std::string UI)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (UI.find(Devices[i].UI) != std::string::npos)                           // element to be changed
        {
            Devices[i].Label=configManager->getCorrectLabel(label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
            if (Devices[i].UI == mui::TUIEnum)                      // TUI-Element
            {
                TUIElement->setLabel(Devices[i].Label);
            }
            else if (Devices[i].UI == mui::VRUIEnum)              // VRUI-Element
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

// set visibility of all UI-Elements
void Tab::setVisible(bool visible)
{
    std::string UI;
    for (size_t i=0; i<Devices.size(); ++i)
    {
        UI.append(Devices[i].UI + " ");
    }
    setVisible(visible, UI);
}

// set visibility of named UI-Elements
void Tab::setVisible(bool visible, std::string UI)
{
    for (size_t i=0; i<Devices.size(); ++i){
        if (UI.find(Devices[i].UI)!=std::string::npos)                             // element to be changed
        {
            if (Devices[i].Visible != visible)                                     // visible-value changed
            {
                Devices[i].Visible = configManager->getCorrectVisible(visible, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
                if (Devices[i].UI == mui::TUIEnum)                  // TUI-Element
                {
                    TUIElement->setHidden(!Devices[i].Visible);
                }
                else if (Devices[i].UI == mui::VRUIEnum)          // VRUI-Element
                {
                    if (Devices[i].Visible)
                    {
                        cover->getMenu()->add(SubmenuItem.get());
                    }
                    else{
                        cover->getMenu()->remove(SubmenuItem.get());
                    }
                }
                else
                {
                    std::cerr << "PotiSlider.cpp: Elementtyp " << Devices[i].UI << " wurde in setVisible(string, bool, bool) nicht gefunden." << std::endl;
                }
            }
        }
    }
}

// set position for TUI-Element
void Tab::setPos(int posx, int posy)
{
    std::pair<int,int> pos(posx,posy);
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (Devices[i].UI == mui::TUIEnum)                              // TUI-Element
        {
            pos=configManager->getCorrectPos(pos, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
            if (Parent != NULL)
            {
                if (configManager->getIdentifierByPos(pos, Parent->getUniqueIdentifier()) != Devices[i].UniqueIdentifier)     // if is equal: Element is already at correct position
                {
                    configManager->preparePos(pos, Parent->getUniqueIdentifier());
                    configManager->deletePosFromPosList(Devices[i].UniqueIdentifier);
                    TUIElement->setPos(pos.first,pos.second);
                    configManager->addPosToPosList(Devices[i].UniqueIdentifier, pos, Parent->getUniqueIdentifier(), false);
                }
            }
            else if (Parent == NULL)
            {
                if (configManager->getIdentifierByPos(pos, mui::getKeywordAttribute(mui::MainWindowEnum)) != Devices[i].UniqueIdentifier)
                {
                    configManager->preparePos(pos, mui::getKeywordAttribute(mui::MainWindowEnum));
                    configManager->deletePosFromPosList(Devices[i].UniqueIdentifier);
                    TUIElement->setPos(pos.first, pos.second);
                    configManager->addPosToPosList(Devices[i].UniqueIdentifier, pos, mui::getKeywordAttribute(mui::MainWindowEnum), false);
                }
            }

        }
    }
}

std::string Tab::getUniqueIdentifier()
{
    return Identifier;
}

Container *Tab::getParent()
{
    return Parent;
}
