#ifndef MUICONTAINER_H
#define MUICONTAINER_H

#include "Element.h"
#include <iostream>
#include <boost/smart_ptr.hpp>

namespace vrui
{
class coMenu;
class coRowMenu;
}

namespace mui
{

/**
 * Baseclass for all mui::elements which shall contain other mui::elements.
 * All inheritors must overwrite the pure virtual methods createVRUIMenuItem, createTUIElement and createVRUIContainer
 */
class COVEREXPORT Container: public Element
{
public:
    // destructor:
    ~Container();

    // methods:
    virtual int getTUIID();                         ///< returns the ID of the TUI-Element
    virtual vrui::coMenu* getVRUI();                ///< returns a pointer to the VRUI-Element
    virtual void setLabel(std::string label);
    virtual void setBackendLabel(std::string label, mui::UITypeEnum UI);


protected:
    // methods:
    // constructor:
    Container(std::string uniqueIdentifier, Container *parent=NULL);

    // variables:
    int TUIID;                                          // ID of TUI-Element
    boost::shared_ptr<vrui::coRowMenu> VRUIContainer;        // VRUI-Element, which can be parent for other VRUI-Elements
};
} // end namespace

#endif
