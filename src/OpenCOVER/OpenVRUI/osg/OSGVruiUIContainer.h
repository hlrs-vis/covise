/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSGVRUIUICONTAINER_H
#define OSGVRUIUICONTAINER_H

#include <OpenVRUI/coUIContainer.h>
#include <OpenVRUI/osg/OSGVruiUIElement.h>

namespace vrui
{

class OSGVruiUIContainer : public OSGVruiUIElement
{

public:
    OSGVruiUIContainer(coUIContainer *container);
    virtual ~OSGVruiUIContainer();

protected:
    coUIContainer *container;
};
}
#endif
