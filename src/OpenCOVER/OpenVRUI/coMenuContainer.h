/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_MENU_CONTAINER_H
#define CO_MENU_CONTAINER_H

#include <OpenVRUI/coRowContainer.h>

/** Row container for Menu entries, a certain number
 of elements (2 per default) are aligned left, the rest right.
*/
namespace vrui
{

class OPENVRUIEXPORT coMenuContainer : public coRowContainer
{

public:
    coMenuContainer(Orientation orientation = HORIZONTAL);
    virtual ~coMenuContainer();

    virtual void resizeToParent(float, float, float, bool shrink = true);
    void setNumAlignedMin(int number); // number of items aligned aligned left/bottom

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    //virtual void resize();
    /// number of elements aligned left
    int numAlignedMin;
};
}
#endif
