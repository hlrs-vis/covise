/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef SGVRUIUICONTAINER_H
#define SGVRUIUICONTAINER_H

#include <OpenVRUI/coUIContainer.h>
#include <OpenVRUI/opensg/SGVruiUIElement.h>

class SGVruiUIContainer : public SGVruiUIElement
{

public:
    SGVruiUIContainer(coUIContainer *container);
    virtual ~SGVruiUIContainer();

protected:
    coUIContainer *container;
};
#endif
