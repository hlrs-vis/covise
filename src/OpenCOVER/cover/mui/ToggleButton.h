// class, which creats a checkbox with label as VRUI
// creates a ToggleButton as TUI

#ifndef MUITOGGLEBUTTON_H
#define MUITOGGLEBUTTON_H

#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coMenuItem.h>
#include "Widget.h"
#include <boost/smart_ptr.hpp>


namespace vrui
{
class coCheckboxMenuItem;
class coRowMenu;
class coUIElement;
}


namespace mui
{
class Container;
class ConfigManager;



class COVEREXPORT ToggleButton: public Widget, public opencover::coTUIListener, public vrui::coMenuListener
{

public:
    // constructor/destructor:
    ToggleButton(const std::string UniqueIdentifier, Container* parent, const std::string label);
    ToggleButton(const std::string UniqueIdentifier, Container* parent);
    ~ToggleButton();

    // methods:
    void setState(bool);                                                            // set status (pressed or not)
    bool getState();                                                                // get status (pressed or not)
    void setPos(int posx, int posy);                                                // positioning TUI-Element
    void setVisible(bool visible);                                                  // sets visible-vlaue for all UI-elements
    void setVisible(bool visible, std::string UI);                                  // sets visible-value for named UI-elements
    void setLabel(std::string label);                                               // sets Label for all UI-elements
    void setLabel(std::string label, std::string UI);                               // sets Label for named UI-elements
    std::string getUniqueIdentifier();                                              // returns the UniqueIdentifier of this ToggleButtonElement
    Container* getParent();                                                    // returns the parent of this ToggleButtonElement if exists
    opencover::coTUIElement* getTUI();
    void activate();                                                                // changes status to activated/pressed
    void deactivate();                                                              // changes status to deactivated/released
    void click();                                                                   // like a click with mouse

    //void muiEvent(Element *muiItem);

    // variables:

private:
    // methods:
    void constructor(const std::string UniqueIdentifier, Container* parent, const std::string label);
    void createVRUIElement(const std::string label);
    void createTUIElement(const std::string label, Container* parent);

    void tabletEvent(opencover::coTUIElement *tUIItem);
    void menuEvent(vrui::coMenuItem *menuItem);

    // variables:
    boost::shared_ptr<opencover::coTUIToggleButton> TUIElement;                       // instance of TUI-ToggleButton
    boost::shared_ptr<vrui::coCheckboxMenuItem> VRUIElement;                          // instance of VRUI-Checkbox
    ConfigManager *configManager;
    Container* Parent;
    std::vector<device> Devices;
    std::string Label;
    std::string Identifier;
    bool State;

    void clicked();                                                                 // signal, which is emmitted, if the Value is changed by slot or clicked
};
} // end namespace

#endif
