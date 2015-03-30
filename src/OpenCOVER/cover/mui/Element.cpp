#include "Element.h"
#include <cover/coTabletUI.h>
#include <cover/mui/support/EventListener.h>

#include "support/ConfigManager.h"
#include <iostream>

using namespace opencover;
using namespace covise;
using namespace mui;

// constructor:
Element::Element()
{

}

// constructor:
Element::Element(const std::string name)
{
    label = name;
}

// destructor:
Element::~Element()
{
}

std::string Element::getUniqueIdentifier()
{
    return UniqueIdentifier;
}

void Element::setEventListener(EventListener *listener)
{
    Listener = listener;
}

EventListener *Element::getMUIListener()
{
    return Listener;
}
