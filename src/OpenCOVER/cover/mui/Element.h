#ifndef MUIELEMENT_H
#define MUIELEMENT_H

/*! file
 * \brief user interface proxy class
 *
 * \author Andreas Grimm
 * \date 04/2015
 */
#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <cover/mui/support/ConfigManager.h>
#include <cover/mui/support/EventListener.h>
#include <cover/mui/support/DefaultValues.h>
#include <boost/smart_ptr.hpp>


namespace opencover
{
class coTUIElement;
}

/**
 * namespace for all mui-classes, so you can easily type e.q. "mui::ToggleButton", "mui::Tab", etc. which will make it safe and still readable
 */
namespace mui
{
class Container;


/**
 * Base class for mui:: elements
 * All inheritors must overwrite the pure virtual functions createVRUIMenuItem, createTUIElement, tabletEvent and menuEvent
 */
class COVEREXPORT Element: public opencover::coTUIListener, public vrui::coMenuListener, public mui::EventListener
{
public:
    // destructor:
    ~Element();

    // methods:
    virtual void setPos(int posx, int posy);                        ///< sets position of the mui::element
    virtual void setVisible(bool visible);                          ///< sets visibility of the mui::element for all types of UI to visible
    virtual void setBackendVisible(bool visible, mui::UITypeEnum UI);      ///< sets visibility of the mui::element for the type of UI to visible
    virtual void setLabel(std::string label);                       ///< sets label of the mui::element for all types of UI to label
    virtual void setBackendLabel(std::string label, mui::UITypeEnum UI);   ///< sets label of the mui::element for the type of UI to label

    virtual std::string getUniqueIdentifier();
    virtual opencover::coTUIElement* getTUI();                      ///< returns the pointer to the TUI-Element

    virtual void setEventListener(EventListener *listener);         ///< simple eventListener to handle mui::events

protected:
    struct propertyStorage                                          // struct to store all interesting propertys of each element
    {
        mui::DeviceTypesEnum device;
        std::string uniqueIdentifier;
        bool visible;
        std::string label;
    };

    // constructor:
    Element(std::string uniqueIdentifier, mui::Container* parent=NULL);

    // variables:
    std::vector<propertyStorage> storage;                           // vector with length of created elements; each UI-element stores its unique information
    mui::Container* parent;                                         // Parent of this element
    ConfigManager *configManager;                                   // instance of ConfigManager; e.g. needed for auto-positioning and access to the configuration file
    EventListener *listener;                                        // Listener to handle mui::events
    boost::shared_ptr<opencover::coTUIElement> TUIElement;          // instance of TUIElement, which will be created
    boost::shared_ptr<vrui::coMenuItem> VRUIMenuItem;        // instance of VRUIElement, which will be created

    std::string uniqueIdentifier;                                   // UniqueIdentifier of this mui::element; e.g. needet for positioning and matching in configuration file
    std::string parentUniqueIdentifier;
    bool isElementContainer;

    // methods:
    void init();

    void initialiseParent(mui::Container *parent);
};
} // end namespace

#endif
