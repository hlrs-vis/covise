// class, which creates a new menu as VRUI
// creates a Frame as TUI


#include <cover/coVRTui.h>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include "Frame.h"
#include "support/ConfigManager.h"

using namespace opencover;
using namespace vrui;
using namespace mui;

// constructor:
Frame::Frame(const std::string uniqueIdentifier, Container* parent)
{
    configManager = NULL;
    constructor(uniqueIdentifier, parent, uniqueIdentifier);
}

Frame::Frame(const std::string uniqueIdentifier, Container* parent, std::string label)
{
    configManager = NULL;
    constructor(UniqueIdentifier, parent, label);
}

// destructor:
Frame::~Frame()
{
    configManager->removeElement(UniqueIdentifier);
    configManager->deletePosFromPosList(UniqueIdentifier);
}

// underlying constructor:
void Frame::constructor(const std::string UniqueIdentifier, Container* parent, std::string label)
{
    configManager= ConfigManager::getInstance();

    Parent=parent;
    Identifier=UniqueIdentifier;

    configManager->addElement(UniqueIdentifier, this);

    // create defaultvalue or take from constructor:
    // TUI-Tablet-Element:
    Devices.push_back(device());
    Devices[0].Device = mui::TabletEnum;
    Devices[0].UI = mui::TUIEnum;
    Devices[0].UniqueIdentifier = UniqueIdentifier;
    Devices[0].Visible = true;

    Devices[0].Label = configManager->getCorrectLabel(Label, Devices[0].UI, Devices[0].Device, Devices[0].UniqueIdentifier);
    Parent= configManager->getCorrectParent(Parent, Devices[0].UI, Devices[0].Device, Devices[0].UniqueIdentifier);
    // create TUI-Element:
    TUIElement.reset(new coTUIFrame(Devices[0].Label, Parent->getTUIID()));

    // VRUI-CAVE-Element:
    Devices.push_back(device());
    Devices[1].Device= mui::CAVEEnum;
    Devices[1].UI= mui::VRUIEnum;
    Devices[1].UniqueIdentifier = UniqueIdentifier;
    Devices[1].Visible = true;

    Devices[1].Label = configManager->getCorrectLabel(Label, Devices[1].UI, Devices[1].Device, Devices[1].UniqueIdentifier);
    Parent= configManager->getCorrectParent(Parent, Devices[1].UI, Devices[1].Device, Devices[1].UniqueIdentifier);
    // create VRUI-Element:
    Submenu.reset(new vrui::coRowMenu(Devices[1].Label.c_str()));
    SubmenuItem.reset(new vrui::coSubMenuItem(Devices[1].Label));
    SubmenuItem->setMenu(Submenu.get());

    // find and set correct parameter (get them from configuration file, if possible):
    for (size_t i=0; i<Devices.size(); ++i)
    {
        Devices[i].Visible = configManager->getCorrectVisible(Devices[i].Visible, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
        Devices[i].Label = configManager->getCorrectLabel(Label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);

        // create UI-Elements:
        if  (Devices[i].UI==mui::TUIEnum)                       // create TUI-Element
        {
            std::pair<int,int> pos=configManager->getCorrectPos(Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier, Parent->getUniqueIdentifier());
            configManager->preparePos(pos, Parent->getUniqueIdentifier());
            TUIElement->setPos(pos.first, pos.second);
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
            else
            {                                                               // invisible
                configManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->remove(SubmenuItem.get());
            }
        }
        else
        {
            std::cerr << "Tab::ParentConstructor: " << Devices[i].UI << " not found in Constructor." << std::endl;
        }
    }
}


// returns the parent-element
Container* Frame::getParent()
{
    return Parent;
}

// returns ID of TUI-Element
int Frame::getTUIID()
{
    return (TUIElement->getID());
}

// returns VRUI-Element
vrui::coMenu* Frame::getVRUI()
{
    return Submenu.get();
}

// set label for named UI-Elements
void Frame::setLabel(std::string label, std::string UI)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (UI.find(Devices[i].UI) != std::string::npos)                       // element to be changed
        {
            Devices[i].Label=configManager->getCorrectLabel(label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
            if (Devices[i].UI == mui::TUIEnum)                  // TUI-Element
            {
                TUIElement->setLabel(Devices[i].Label);
            }
            else if (Devices[i].UI == mui::VRUIEnum)          // VRUI-Element
            {
                SubmenuItem->setLabel(Devices[i].Label);
                Submenu->updateTitle(Devices[i].Label.c_str());
            }
            else
            {
                std::cerr << "Frame::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string, std::string)." << std::endl;
            }
        }
    }
}

// set label for all UI-Elements
void Frame::setLabel(std::string label)
{
    std::string UI;
    for (size_t i=0; i<Devices.size(); ++i)
    {
        UI.append(Devices[i].UI + " ");
    }
    setLabel(label, UI);
}

// return pointer to TUIElement
coTUIElement* Frame::getTUI()
{
    return TUIElement.get();
}

// set visible-value for named UI-Elements
void Frame::setVisible(bool visible, std::string UI)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (UI.find(Devices[i].UI)!=std::string::npos)                             // element shall be changed
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
                        Parent->getVRUI()->add(SubmenuItem.get());
                    }
                    else{
                        Parent->getVRUI()->remove(SubmenuItem.get());
                    }
                }
                else{
                    std::cerr << "Frame::setVisible(): Elementtyp " << Devices[i].UI << " not found in setVisible(string, bool, bool)." << std::endl;
                }
            }
        }
    }
}

// set visible-value for all UI-Elements
void Frame::setVisible(bool visible)
{
    std::string UI;
    for (size_t i=0; i<Devices.size(); ++i)
    {
        UI.append(Devices[i].UI + " ");
    }
    setVisible(visible, UI);
}

// set position for TUI-Element
void Frame::setPos(int posx, int posy)
{
    std::pair<int,int> pos(posx,posy);
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

std::string Frame::getUniqueIdentifier()
{
    return Devices[0].UniqueIdentifier;
}
