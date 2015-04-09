// class, which creats a checkbox with label as VRUI
// creates a ToggleButton as TUI

#ifndef MUITOGGLEBUTTON_H
#define MUITOGGLEBUTTON_H

#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coMenuItem.h>
#include "Element.h"
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


class COVEREXPORT ToggleButton: public mui::Element
{

public:

    //destructor:
    ~ToggleButton();

    // methods:
    static mui::ToggleButton* create(std::string uniqueIdentifier, mui::Container* parent);

    void setState(bool);                                                            ///< set status of mui::ToggleButton (pressed or not)
    bool getState();                                                                ///< get status of mui::TogglebUTTON (pressed or not)

    void muiEvent(Element *muiItem);

private:
    // methods:
    // constructor:
    ToggleButton(const std::string uniqueIdentifier, Container* parent);

    void tabletEvent(opencover::coTUIElement *tUIItem);
    void menuEvent(vrui::coMenuItem *menuItem);

    // variables:

    bool state;                                                                // signal, which is emmitted, if the Value is changed by slot or clicked
};
} // end namespace

#endif
