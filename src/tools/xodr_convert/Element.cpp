/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Element.h"

Element::Element(std::string setid)
    : id(setid)
{
}

std::string Element::getId()
{
    return id;
}

bool Element::operator<(Element *element)
{
    return this->id < element->getId();
}
