#include "Element.h"
#include <cover/coTabletUI.h>
#include <cover/mui/support/Listener.h>

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
Element::Element(const std::string &n_str)
{
    label_str = n_str;
}

// destructor:
Element::~Element()
{
}

std::string Element::getUniqueIdentifier()
{
    return UniqueIdentifier;
}

void Element::setEventListener(Listener *l)
{
    listener = l;
}

Listener *Element::getMUIListener()
{
    return listener;
}
