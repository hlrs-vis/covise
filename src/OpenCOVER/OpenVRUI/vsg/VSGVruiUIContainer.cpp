/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiUIContainer.h>
#include <OpenVRUI/coUIContainer.h>

namespace vrui
{

VSGVruiUIContainer::VSGVruiUIContainer(coUIContainer *container)
    : VSGVruiUIElement(container)
{

    this->container = container;
}

VSGVruiUIContainer::~VSGVruiUIContainer()
{
}
}
