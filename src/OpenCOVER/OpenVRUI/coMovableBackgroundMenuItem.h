/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_MOVABLE_BACKGROUND_MENUITEM_H
#define CO_MOVABLE_BACKGROUND_MENUITEM_H

#include <sys/types.h>
#include "coMenuItem.h"
#include "coTexturedBackground.h"
#include "coFrame.h"

#ifdef WIN32
typedef unsigned int uint;
#endif

namespace vrui
{

class OPENVRUIEXPORT coMovableBackgroundMenuItem : public coMenuItem
{
protected:
    coTexturedBackground *background;
    float aspect_;
    float vsize_;

public:
    coMovableBackgroundMenuItem(const char *name, float aspect, float size = -1);
    coMovableBackgroundMenuItem(const char *name, uint *normalImage, int comp, int ns, int nt, int nr, float aspect, float size = -1);

    virtual ~coMovableBackgroundMenuItem();

    // return the actual UI Element that represents this menu.
    virtual coUIElement *getUIElement();

    /// get the Element's classname
    virtual const char *getClassName() const;

    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    // return scalefactor of movableBackground;
    float getScale();

    // set scalefactor of movable background
    void setScale(float s);

    // set size
    void setSize(float hsize, float vsize);

    // reset scale and position of movable background
    void reset();

    float getAspect();

    float getVSize();
};
}

#endif
