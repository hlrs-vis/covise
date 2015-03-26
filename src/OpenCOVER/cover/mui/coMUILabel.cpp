// class, which creates a label as VRUI and TUI

#include "coMUILabel.h"
#include "support/coMUIConfigManager.h"
#include "coMUIContainer.h"
#include <OpenVRUI/coLabelMenuItem.h>

#include <iostream>

using namespace vrui;
using namespace opencover;

// constructor:
coMUILabel::coMUILabel(std::string UniqueIdentifier, coMUIContainer* parent, std::string label)
{
    constructor(UniqueIdentifier, parent, label);
}
coMUILabel::coMUILabel(std::string UniqueIdentifier, coMUIContainer* parent)
{
    constructor(UniqueIdentifier, parent, UniqueIdentifier);
}

// destructor:
coMUILabel::~coMUILabel()
{
    ConfigManager->deletePosFromPosList(Identifier);
    ConfigManager->removeElement(Identifier);
}


// underlaying constructor:
void coMUILabel::constructor(std::string UniqueIdentifier, coMUIContainer* parent, std::string label)
{

    Identifier = UniqueIdentifier;

    ConfigManager = coMUIConfigManager::getInstance();
    ConfigManager->addElement(UniqueIdentifier, this);

    Label=label;
    Parent=parent;

    // create defaultvalue or take from constructor:
    // VRUI-CAVE-Element:
    Devices.push_back(device());
    Devices[0].Device= ConfigManager->keywordCAVE();
    Devices[0].UI = ConfigManager->keywordVRUI();
    Devices[0].Identifier = UniqueIdentifier;
    Devices[0].Visible = true;

    Devices[0].Label = ConfigManager->getCorrectLabel(Label, Devices[0].UI, Devices[0].Device, Devices[0].Identifier);
    Parent = ConfigManager->getCorrectParent(Parent, Devices[0].UI, Devices[0].Device, Devices[0].Identifier);

    // create VRUI-Element:
    VRUIElement.reset(new coLabelMenuItem(Devices[0].Label));

    // TUI-Tablet-Element:
    Devices.push_back(device());
    Devices[1].Device= ConfigManager->keywordTablet();
    Devices[1].UI = ConfigManager->keywordTUI();
    Devices[1].Identifier = UniqueIdentifier;
    Devices[1].Visible = true;

    Parent = ConfigManager->getCorrectParent(Parent, Devices[1].UI, Devices[1].Device, Devices[1].Identifier);
    Devices[1].Label = ConfigManager->getCorrectLabel(Label, Devices[1].UI, Devices[1].Device, Devices[1].Identifier);
    // create TUI-Element:
    TUIElement.reset(new coTUILabel(Devices[1].Label, Parent->getTUIID()));

    // find and set correct parameter (get them from configuration file, if possible):
    for (size_t  i=0; i<Devices.size(); ++i)
    {
        Devices[i].Visible = ConfigManager->getCorrectVisible(Devices[i].Visible, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
        Devices[i].Label = ConfigManager->getCorrectLabel(Label, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);

        // create UI-Elements:
        if (Devices[i].UI == ConfigManager->keywordTUI())         // create TUI-Element
        {
            std::pair<int,int> pos=ConfigManager->getCorrectPos(Devices[i].UI, Devices[i].Device, Devices[i].Identifier, Parent->getUniqueIdentifier());
            ConfigManager->preparePos(pos, Parent->getUniqueIdentifier());
            TUIElement->setPos(pos.first,pos.second);
            ConfigManager->addPosToPosList(Devices[i].Identifier, pos, Parent->getUniqueIdentifier(), true);
            TUIElement->setHidden(!Devices[i].Visible);
        }else if (Devices[i].UI == ConfigManager->keywordVRUI())  // create VRUI-Element
        {
            if (Devices[i].Visible)                                            // visible
            {
                ConfigManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->add(VRUIElement.get());
            }else                                                               // invisible
            {
                ConfigManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->remove(VRUIElement.get());
            }
        }else
        {
            std::cerr << "coMUIPotiSlider::constructor(): Elementtype " << Devices[i].UI << " not found in Constructor." << std::endl;
        }
    }
}

// return the label
std::string coMUILabel::getLabel()
{
    return Label;
}

// positioning TUI-Element
void coMUILabel::setPos(int posx, int posy)
{
    std::pair<int,int> pos (posx,posy);
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (Devices[i].UI == ConfigManager->keywordTUI())                          // TUI-Element
        {
            if (ConfigManager->getIdentifierByPos(pos, Parent->getUniqueIdentifier()) != Devices[i].Identifier)     // if is equal: Element is already at correct position
            {
                pos=ConfigManager->getCorrectPos(pos, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
                ConfigManager->preparePos(pos, Parent->getUniqueIdentifier());
                ConfigManager->deletePosFromPosList(Devices[i].Identifier);
                TUIElement->setPos(pos.first,pos.second);
                ConfigManager->addPosToPosList(Devices[i].Identifier, pos, Parent->getUniqueIdentifier(), false);
            }
        }
    }
}

// set visible-value of named UI
void coMUILabel::setVisible(bool visible, std::string UI)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (UI.find(Devices[i].UI) != std::string::npos)                           // element shall be changed
        {
            if (Devices[i].Visible != visible)                                     // visible-value changed
            {
                Devices[i].Visible = ConfigManager->getCorrectVisible(visible, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
                if (Devices[i].UI == ConfigManager->keywordTUI())                  // TUI-Element
                {
                    TUIElement->setHidden(!Devices[i].Visible);
                } else if (Devices[i].UI == ConfigManager->keywordVRUI())          // VRUI-Element
                {
                    if (Devices[i].Visible){
                        ConfigManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->add(VRUIElement.get());
                    } else
                    {
                        ConfigManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->remove(VRUIElement.get());
                    }
                } else
                {
                    std::cerr << "coMUIPotiSlider:: Elementtype " << Devices[i].UI << " not found in setVisible(string, bool, bool)." << std::endl;
                }
            }
        }
    }
}

// set visible-value of all UI-Elements
void coMUILabel::setVisible(bool visible)
{
    std::string UI;
    for (size_t i=0; i<Devices.size(); ++i)
    {
        UI.append(Devices[i].UI + " ");
    }
    setVisible(visible, UI);
}

// returns parent of coMUILabel-Element
coMUIContainer* coMUILabel::getParent()
{
    return Parent;
}

std::string coMUILabel::getUniqueIdentifier()
{
    return Devices[0].Identifier;
}

// returns a pointer to TUIElement
coTUIElement* coMUILabel::getTUI()
{
    return TUIElement.get();
}

void coMUILabel::setLabel(std::string label)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        Devices[i].Label = ConfigManager->getCorrectLabel(label, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
        if (Devices[i].UI == ConfigManager->keywordTUI())                          // TUI-Element
        {
            TUIElement->setLabel(Devices[i].Label);
        } else if (Devices[i].UI == ConfigManager->keywordVRUI())                  // VRUI-Element
        {
            VRUIElement->setLabel(Devices[i].Label);
        } else
        {
            std::cerr << "coMUILabel::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string)." << std::endl;
        }
    }
}

// set label for named UI-Elements
void coMUILabel::setLabel(std::string label, std::string UI)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (UI.find(Devices[i].UI) != std::string::npos)                           // element to be changed
        {
            Devices[i].Label=ConfigManager->getCorrectLabel(label, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
            if (Devices[i].UI == ConfigManager->keywordTUI())                      // TUI-Element
            {
                TUIElement->setLabel(Devices[i].Label);
            } else if (Devices[i].UI == ConfigManager->keywordVRUI())              // VRUI-Element
            {
                TUIElement->setLabel(Devices[i].Label);
            } else
            {
                std::cerr << "coMUIPotiSlider::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string, std::string)." << std::endl;
            }
        }
    }
}

void coMUILabel::changeLabel(std::string label)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        Devices[i].Label = label;
        if (Devices[i].UI == ConfigManager->keywordTUI())                          // TUI-Element
        {
            TUIElement->setLabel(Devices[i].Label);
        } else if (Devices[i].UI == ConfigManager->keywordVRUI())                  // VRUI-Element
        {
            VRUIElement->setLabel(Devices[i].Label);
        } else
        {
            std::cerr << "coMUILabel::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string)." << std::endl;
        }
    }
}
