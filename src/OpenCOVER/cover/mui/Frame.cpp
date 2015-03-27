// class, which creates a new menu as VRUI
// creates a Frame as TUI


#include <cover/coVRTui.h>
#include <cover/coTabletUI.h>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include "Frame.h"
#include "support/ConfigManager.h"

using namespace opencover;
using namespace vrui;
using namespace mui;

// constructor:
Frame::Frame(const std::string UniqueIdentifier, Container* parent)
{
    configManager = NULL;
    Label = UniqueIdentifier;
    constructor(UniqueIdentifier, parent, Label);
}

Frame::Frame(const std::string UniqueIdentifier, Container* parent, std::string label)
{
    configManager = NULL;
    Label = label;
    constructor(UniqueIdentifier, parent, Label);
}

// destructor:
Frame::~Frame()
{
    configManager->removeElement(Identifier);
    configManager->deletePosFromPosList(Identifier);
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
    Devices[0].Device = configManager->keywordTablet();
    Devices[0].UI = configManager->keywordTUI();
    Devices[0].Identifier = UniqueIdentifier;
    Devices[0].Visible = true;

    Devices[0].Label = configManager->getCorrectLabel(Label, Devices[0].UI, Devices[0].Device, Devices[0].Identifier);
    Parent= configManager->getCorrectParent(Parent, Devices[0].UI, Devices[0].Device, Devices[0].Identifier);
    // create TUI-Element:
    TUIElement.reset(new coTUIFrame(Devices[0].Label, Parent->getTUIID()));

    // VRUI-CAVE-Element:
    Devices.push_back(device());
    Devices[1].Device= configManager->keywordCAVE();
    Devices[1].UI= configManager->keywordVRUI();
    Devices[1].Identifier = UniqueIdentifier;
    Devices[1].Visible = true;

    Devices[1].Label = configManager->getCorrectLabel(Label, Devices[1].UI, Devices[1].Device, Devices[1].Identifier);
    Parent= configManager->getCorrectParent(Parent, Devices[1].UI, Devices[1].Device, Devices[1].Identifier);
    // create VRUI-Element:
    Submenu.reset(new vrui::coRowMenu(Devices[1].Label.c_str()));
    SubmenuItem.reset(new vrui::coSubMenuItem(Devices[1].Label));
    SubmenuItem->setMenu(Submenu.get());

    // find and set correct parameter (get them from configuration file, if possible):
    for (size_t i=0; i<Devices.size(); ++i)
    {
        Devices[i].Visible = configManager->getCorrectVisible(Devices[i].Visible, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
        Devices[i].Label = configManager->getCorrectLabel(Label, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);

        // create UI-Elements:
        if  (Devices[i].UI==configManager->keywordTUI())                       // create TUI-Element
        {
            std::pair<int,int> pos=configManager->getCorrectPos(Devices[i].UI, Devices[i].Device, Devices[i].Identifier, Parent->getUniqueIdentifier());
            configManager->preparePos(pos, Parent->getUniqueIdentifier());
            TUIElement->setPos(pos.first, pos.second);
            configManager->addPosToPosList(Devices[i].Identifier, pos, Parent->getUniqueIdentifier(), true);
            TUIElement->setHidden(!Devices[i].Visible);
        }
        else if (Devices[i].UI==configManager->keywordVRUI())                  // create VRUI-Elements
        {
            SubmenuItem->setMenu(Submenu.get());

            if (Devices[i].Visible)                                            // visible
            {
                configManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->add(SubmenuItem.get());
            }
            else
            {                                                               // invisible
                configManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->remove(SubmenuItem.get());
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
            Devices[i].Label=configManager->getCorrectLabel(label, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
            if (Devices[i].UI == configManager->keywordTUI())                  // TUI-Element
            {
                TUIElement->setLabel(Devices[i].Label);
            } else if (Devices[i].UI == configManager->keywordVRUI())          // VRUI-Element
            {
                SubmenuItem->setLabel(Devices[i].Label);
                Submenu->updateTitle(Devices[i].Label.c_str());
            } else
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
                Devices[i].Visible = configManager->getCorrectVisible(visible, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
                if (Devices[i].UI == configManager->keywordTUI())                  // TUI-Element
                {
                    TUIElement->setHidden(!Devices[i].Visible);
                } else if (Devices[i].UI == configManager->keywordVRUI())          // VRUI-Element
                {
                    if (Devices[i].Visible)
                    {
                        Parent->getVRUI()->add(SubmenuItem.get());
                    } else{
                        Parent->getVRUI()->remove(SubmenuItem.get());
                    }
                } else{
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
        if (Devices[i].UI == configManager->keywordTUI())                      // TUI-Element
        {
            if (configManager->getIdentifierByPos(pos, Parent->getUniqueIdentifier()) != Devices[i].Identifier)     // if is equal: Element is already at correct position
            {
                pos=configManager->getCorrectPos(pos, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
                configManager->preparePos(pos, Parent->getUniqueIdentifier());
                configManager->deletePosFromPosList(Devices[i].Identifier);
                TUIElement->setPos(pos.first,pos.second);
                configManager->addPosToPosList(Devices[i].Identifier, pos, Parent->getUniqueIdentifier(), false);
            }
        }
    }
}

std::string Frame::getUniqueIdentifier()
{
    return Devices[0].Identifier;
}
