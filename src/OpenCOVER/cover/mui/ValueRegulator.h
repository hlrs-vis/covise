// class, which creates a Poti as VRUI-Element
// creates a Slider as TUI-Element

#ifndef MUIVALUEREGULATOR_H
#define MUIVALUEREGULATOR_H

#include <cover/coTabletUI.h>
#include <OpenVRUI/coMenuItem.h>
#include "Element.h"
#include <boost/smart_ptr.hpp>

namespace vrui
{
class coPotiMenuItem;
class coRowMenu;
class coUIElement;
class coMenu;
}


namespace mui
{
class Container;

class COVEREXPORT ValueRegulator:public mui::Element
{

public:
    // destructor:
     ~ValueRegulator();

    // methods:
    static mui::ValueRegulator* create(std::string uniqueIdetifier, Container* parent, float min, float max, float defaultValue);


    float getValue();
    void setValue(float newVal);

private:
    // methods:
    // constructor:
    ValueRegulator(const std::string UniqueIdentifier, Container* parent, float min, float max, float defaultValue);

    void tabletEvent(opencover::coTUIElement *tUIItem);
    void menuEvent(vrui::coMenuItem *menuItem);

    // variables:
    float value;
    float minVal;
    float maxVal;
};
} // end namespace
#endif
