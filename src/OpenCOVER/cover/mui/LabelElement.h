// class, which creates a label in VRUI and TUI

#ifndef MUILABELELEMENT_H
#define MUILABELELEMENT_H

#include <cover/coVRPlugin.h>
#include "Element.h"
#include <boost/smart_ptr.hpp>

namespace vrui
{
class coLabelMenuItem;
}
namespace opencover
{
class coTUILabel;
}

namespace mui
{
class Container;

class COVEREXPORT LabelElement:public mui::Element
{

public:
    // destructor:
    ~LabelElement();

    // methods:
    static mui::LabelElement* create(std::string uniqueIdentifier, mui::Container* parent);

private:
    // methods:
    // constructor:
    LabelElement(std::string UniqueIdentifier, Container *parent);
};
} // end namespace

#endif
