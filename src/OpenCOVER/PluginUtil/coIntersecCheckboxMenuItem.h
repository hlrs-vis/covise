/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_INTERSEC_CHECKBOX_MENUITEM_H_
#define _CO_INTERSEC_CHECKBOX_MENUITEM_H_

#include <OpenVRUI/coCheckboxMenuItem.h>
#include <util/coTypes.h>

namespace covise
{
class coCheckboxGroup;
}

namespace opencover
{
class PLUGIN_UTILEXPORT coIntersecCheckboxMenuItem : public coCheckboxMenuItem
{
public:
    coIntersecCheckboxMenuItem(const char *, bool, coCheckboxGroup * = NULL);
    virtual ~coIntersecCheckboxMenuItem();
    //virtual int hit(osg::Vec3&, pfHit*);
    virtual int hit(vruiHit *hit);
    virtual void miss();
    bool isIntersected() const;

private:
    bool _isIntersected;
};
}
#endif
