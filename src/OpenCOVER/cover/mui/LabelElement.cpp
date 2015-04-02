// class, which creates a label as VRUI and TUI

#include "LabelElement.h"
#include "support/ConfigManager.h"
#include "Container.h"
#include <OpenVRUI/coLabelMenuItem.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coMenu.h>

#include <iostream>

using namespace vrui;
using namespace opencover;
using namespace mui;

// constructor:
LabelElement::LabelElement(std::string uniqueIdentifier, Container* parent, std::string label)
{
    constructor(uniqueIdentifier, parent, label);
}
LabelElement::LabelElement(std::string uniqueIdentifier, Container* parent)
{
    constructor(uniqueIdentifier, parent, uniqueIdentifier);
}

// destructor:
LabelElement::~LabelElement()
{
    configManager->deletePosFromPosList(Identifier);
    configManager->removeElement(Identifier);
}


// underlaying constructor:
void LabelElement::constructor(std::string uniqueIdentifier, Container* parent, std::string label)
{

    UniqueIdentifier = uniqueIdentifier;

    configManager = ConfigManager::getInstance();
    configManager->addElement(UniqueIdentifier, this);

    Label=label;
    Parent=parent;

    // create defaultvalue or take from constructor:
    // VRUI-CAVE-Element:
    Devices.push_back(device());
    Devices[0].Device = mui::CAVEEnum;
    Devices[0].UI = mui::VRUIEnum;
    Devices[0].UniqueIdentifier = UniqueIdentifier;
    Devices[0].Visible = true;

    Devices[0].Label = configManager->getCorrectLabel(Label, Devices[0].UI, Devices[0].Device, Devices[0].UniqueIdentifier);
    Parent = configManager->getCorrectParent(Parent, Devices[0].UI, Devices[0].Device, Devices[0].UniqueIdentifier);

    // create VRUI-Element:
    VRUIElement.reset(new coLabelMenuItem(Devices[0].Label));

    // TUI-Tablet-Element:
    Devices.push_back(device());
    Devices[1].Device = mui::TabletEnum;
    Devices[1].UI = mui::TUIEnum;
    Devices[1].UniqueIdentifier = UniqueIdentifier;
    Devices[1].Visible = true;

    Parent = configManager->getCorrectParent(Parent, Devices[1].UI, Devices[1].Device, Devices[1].UniqueIdentifier);
    Devices[1].Label = configManager->getCorrectLabel(Label, Devices[1].UI, Devices[1].Device, Devices[1].UniqueIdentifier);
    // create TUI-Element:
    TUIElement.reset(new coTUILabel(Devices[1].Label, Parent->getTUIID()));

    // find and set correct parameter (get them from configuration file, if possible):
    for (size_t  i=0; i<Devices.size(); ++i)
    {
        Devices[i].Visible = configManager->getCorrectVisible(Devices[i].Visible, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
        Devices[i].Label = configManager->getCorrectLabel(Label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);

        // create UI-Elements:
        if (Devices[i].UI == mui::TUIEnum)         // create TUI-Element
        {
            std::pair<int,int> pos=configManager->getCorrectPos(Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier, Parent->getUniqueIdentifier());
            configManager->preparePos(pos, Parent->getUniqueIdentifier());
            TUIElement->setPos(pos.first,pos.second);
            configManager->addPosToPosList(Devices[i].UniqueIdentifier, pos, Parent->getUniqueIdentifier(), true);
            TUIElement->setHidden(!Devices[i].Visible);
        }
        else if (Devices[i].UI == mui::VRUIEnum)  // create VRUI-Element
        {
            if (Devices[i].Visible)                                            // visible
            {
                configManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->add(VRUIElement.get());
            }
            else                                                               // invisible
            {
                configManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->remove(VRUIElement.get());
            }
        }
        else
        {
            std::cerr << "PotiSlider::constructor(): Elementtype " << Devices[i].UI << " not found in Constructor." << std::endl;
        }
    }
}

// return the label
std::string LabelElement::getLabel()
{
    return Label;
}

// positioning TUI-Element
void LabelElement::setPos(int posx, int posy)
{
    std::pair<int,int> pos (posx,posy);
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (Devices[i].UI == mui::TUIEnum)                          // TUI-Element
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

// set visible-value of named UI
void LabelElement::setVisible(bool visible, std::string UI)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        if (UI.find(Devices[i].UI) != std::string::npos)                           // element shall be changed
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
                    if (Devices[i].Visible){
                        configManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->add(VRUIElement.get());
                    }
                    else
                    {
                        configManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier)->getVRUI()->remove(VRUIElement.get());
                    }
                }
                else
                {
                    std::cerr << "PotiSlider:: Elementtype " << Devices[i].UI << " not found in setVisible(string, bool, bool)." << std::endl;
                }
            }
        }
    }
}

// set visible-value of all UI-Elements
void LabelElement::setVisible(bool visible)
{
    std::string UI;
    for (size_t i=0; i<Devices.size(); ++i)
    {
        UI.append(Devices[i].UI + " ");
    }
    setVisible(visible, UI);
}

// returns parent of Label-Element
Container* LabelElement::getParent()
{
    return Parent;
}

std::string LabelElement::getUniqueIdentifier()
{
    return UniqueIdentifier;
}

// returns a pointer to TUIElement
coTUIElement* LabelElement::getTUI()
{
    return TUIElement.get();
}

void LabelElement::setLabel(std::string label)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        Devices[i].Label = configManager->getCorrectLabel(label, Devices[i].UI, Devices[i].Device, Devices[i].UniqueIdentifier);
        if (Devices[i].UI == mui::TUIEnum)                          // TUI-Element
        {
            TUIElement->setLabel(Devices[i].Label);
        }
        else if (Devices[i].UI == mui::VRUIEnum)                  // VRUI-Element
        {
            VRUIElement->setLabel(Devices[i].Label);
        }
        else
        {
            std::cerr << "Label::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string)." << std::endl;
        }
    }
}

// set label for named UI-Elements
void LabelElement::setLabel(std::string label, std::string UI)
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
                TUIElement->setLabel(Devices[i].Label);
            }
            else
            {
                std::cerr << "PotiSlider::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string, std::string)." << std::endl;
            }
        }
    }
}

void LabelElement::changeLabel(std::string label)
{
    for (size_t i=0; i<Devices.size(); ++i)
    {
        Devices[i].Label = label;
        if (Devices[i].UI == mui::TUIEnum)                          // TUI-Element
        {
            TUIElement->setLabel(Devices[i].Label);
        }
        else if (Devices[i].UI == mui::VRUIEnum)                  // VRUI-Element
        {
            VRUIElement->setLabel(Devices[i].Label);
        }
        else
        {
            std::cerr << "Label::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string)." << std::endl;
        }
    }
}
