#include "Element.h"
#include <cover/coTabletUI.h>
#include <cover/coVRPluginSupport.h>

#include <cover/mui/support/EventListener.h>
#include <cover/mui/Container.h>
#include "support/ConfigManager.h"
#include <iostream>


using namespace mui;

// constructor:
Element::Element(std::string uniqueIdentifier, mui::Container *parentParameter)
    : uniqueIdentifier(uniqueIdentifier)
{
    listener=NULL;

    configManager = ConfigManager::getInstance();
    initialiseParent(parentParameter);
    if (parent == NULL)
    {
        parentUniqueIdentifier = "";
    }
    else
    {
        parentUniqueIdentifier = parent->getUniqueIdentifier();
    }

    storage.resize(std::max(mui::VRUIEnum, mui::TUIEnum)+1);

    // Values for CAVE
    storage[mui::VRUIEnum].device = mui::CAVEEnum;

    // Values for Tablet
    storage[mui::TUIEnum].device = mui::TabletEnum;

    // Values for all
    for (size_t i=0; i<storage.size(); ++i)
    {
        storage[i].uniqueIdentifier = uniqueIdentifier;
        storage[i].visible = configManager->getCorrectVisible(true, mui::UITypeEnum(i), storage[i].device, uniqueIdentifier);
        storage[i].label = configManager->getCorrectLabel(uniqueIdentifier, mui::UITypeEnum(i), storage[i].device, uniqueIdentifier);
    }
}

// destructor:
Element::~Element()
{
    configManager->removeElement(uniqueIdentifier);
    configManager->deletePosFromPosList(uniqueIdentifier);
}

void mui::Element::init()
{
    // set position
    for (size_t i=0; i<storage.size(); ++i)
    {
        if (i == mui::TUIEnum)                                  // set Position of TUI-Element
        {
            std::pair<int,int> pos;
            pos = configManager->getCorrectPos(mui::TUIEnum, storage[i].device, storage[i].uniqueIdentifier, parentUniqueIdentifier);
            configManager->preparePos(pos, parentUniqueIdentifier);
            TUIElement->setPos(pos.first, pos.second);

            if (configManager->existAttributeInConfigFile(mui::TUIEnum, storage[i].device, storage[i].uniqueIdentifier, mui::PosXEnum) && configManager->existAttributeInConfigFile(mui::TUIEnum, storage[i].device, storage[i].uniqueIdentifier, mui::PosYEnum))
            {
                configManager->addPosToPosList(storage[i].uniqueIdentifier, pos, parentUniqueIdentifier, false);
            }
            else
            {
                configManager->addPosToPosList(storage[i].uniqueIdentifier, pos, parentUniqueIdentifier, true);
            }
        }
    }

    // actualize visibility
    setVisible(true);
}

/**
 * sets position of mui::elements. Until now only sets the position of TUI-Element. If an elements needs multiple spaces, this function must be overwritten.
 */
void mui::Element::setPos(int posx, int posy)
{
    std::pair<int,int> pos(posx,posy);
    for (size_t i=0; i<storage.size(); ++i)
    {
        if (mui::UITypeEnum(i) == mui::TUIEnum)                          // Element is TUI-Element
        {
            pos = configManager->getCorrectPos(pos, mui::UITypeEnum(i), storage[i].device, storage[i].uniqueIdentifier);
            if (configManager->getIdentifierByPos(pos, parentUniqueIdentifier) != storage[i].uniqueIdentifier)   // element isn't at correct position
            {
                configManager->preparePos(pos, parentUniqueIdentifier);
                configManager->deletePosFromPosList(storage[i].uniqueIdentifier);
                TUIElement->setPos(pos.first, pos.second);
                configManager->addPosToPosList(storage[i].uniqueIdentifier, pos, parentUniqueIdentifier, false);
            }
            else                                                    // element is already at correct position
            {
                configManager->deletePosFromPosList(storage[i].uniqueIdentifier);
                TUIElement->setPos(pos.first, pos.second);
                configManager->addPosToPosList(storage[i].uniqueIdentifier, pos, parentUniqueIdentifier, false);
            }
        }
    }
}

/**
 * sets the visible-value of this mui::element for all User Interfaces. To set visible-value of a special User Interface, use setVisible(bool, mui::UITypeEnum)
 */
void mui::Element::setVisible(bool visible)
{
    for (size_t i=0; i<storage.size(); ++i)
    {
        setBackendVisible(visible, mui::UITypeEnum(i));
    }
}

/**
 * sets the visible-value of this mui::element for one User Interface. To set visible-value of all User Interfaces, use setVisible(bool).
 */
void mui::Element::setBackendVisible(bool visible, mui::UITypeEnum UI)
{
    storage[UI].visible = configManager->getCorrectVisible(visible, UI, storage[UI].device, storage[UI].uniqueIdentifier);
    switch (UI)
    {
    case mui::TUIEnum:              // TUI-Element
        TUIElement->setHidden(!storage[UI].visible);
        break;

    case mui::muiUICounter:
        break;

    case mui::VRUIEnum:             // VRUI-Element
        vrui::coMenu* VRUIParent;
        if (parent == NULL)
        {
            VRUIParent = opencover::cover->getMenu();
        }
        else
        {
            VRUIParent = parent->getVRUI();
        }

        if (storage[UI].visible)
        {
            VRUIParent->add(VRUIMenuItem.get());
        }
        else
        {
            VRUIParent->remove(VRUIMenuItem.get());
        }
        break;
    }
}

/**
 * sets the label of this mui::element for all User Interfaces. To set label of one User Interface, use setLabel(std::string, mui::UITypeEnum). Needs to be overwritten, if creating multiple instances of one UI-type.
 */
void mui::Element::setLabel(std::string label)
{
    for (size_t i=0; i<storage.size(); ++i)
    {
        setBackendLabel(label, mui::UITypeEnum(i));
    }
}

/**
 * sets the label of this mui::element for one User Interface. To set label for all User Interfaces, use setLabel(std::string). Needs to be overwritten, if creating multiple instances of one UI-type.
 */
void mui::Element::setBackendLabel(std::string label, mui::UITypeEnum UI)
{
    storage[UI].label=configManager->getCorrectLabel(label, UI, storage[UI].device, storage[UI].uniqueIdentifier);

    switch (UI)
    {
    case mui::muiUICounter:
        break;

    case mui::TUIEnum:
        TUIElement->setLabel(storage[UI].label);
        break;

    case mui::VRUIEnum:
        VRUIMenuItem->setLabel(storage[UI].label);
        break;
    }
}


/**
 * checks the configuration file for defined parent. Only one valid parent per element is allowed. If multiple different parents are defined for this Element in configuration file, an error will be shown.
 * Else, the parent from configuration (if exists) file will be used. Otherwise the input-parent will be used.
 */
void Element::initialiseParent(mui::Container *parentParameter)
{
    int counter=0;
    mui::Container *newParent=parentParameter;

    for (size_t i=1; i<mui::muiUICounter; ++i)
    {
        for (size_t j=1; j<mui::muiDeviceCounter; ++j)
        {
            parent= configManager->getCorrectParent(newParent, mui::UITypeEnum(i), mui::DeviceTypesEnum(j), uniqueIdentifier);
            if (newParent != parent)            // Parent has changed
            {
                counter++;
            }
            newParent = parent;
        }
    }
    if (counter > 1)
    {
        std::cerr << "mui::Element::initialiseParent(): multiple parents for " << uniqueIdentifier << " found in configuration file." << std::endl;
    }
}

opencover::coTUIElement* mui::Element::getTUI()
{
    return TUIElement.get();
}

std::string mui::Element::getUniqueIdentifier()
{
    return uniqueIdentifier;
}

void mui::Element::setEventListener(EventListener *l)
{
    listener = l;
}
