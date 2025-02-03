/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VSGVruiUICONTAINER_H
#define VSGVruiUICONTAINER_H

#include <OpenVRUI/coUIContainer.h>
#include <OpenVRUI/vsg/VSGVruiUIElement.h>

namespace vrui
{

class VSGVRUIEXPORT VSGVruiUIContainer : public VSGVruiUIElement
{

public:
    VSGVruiUIContainer(coUIContainer *container);
    virtual ~VSGVruiUIContainer();

protected:
    coUIContainer *container;
};
}
#endif
