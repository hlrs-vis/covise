#include "coMUIElement.h"
#include <cover/coTabletUI.h>

#include "support/coMUIConfigManager.h"
#include <iostream>

using namespace opencover;
using namespace covise;

// constructor:
coMUIElement::coMUIElement()
{

}

// constructor:
coMUIElement::coMUIElement(const std::string &n_str)
{
    label_str = n_str;
}

// destructor:
coMUIElement::~coMUIElement()
{
}

std::string coMUIElement::getUniqueIdentifier()
{
    return UniqueIdentifier;
}
