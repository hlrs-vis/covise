/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_PARTNERMENUITEM_H
#define CO_PARTNERMENUITEM_H

/*! \file
 \brief menu item that reflects the state of a participant in a collaborative virtual environment

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C)
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#include <OpenVRUI/coCheckboxMenuItem.h>
#include <osg/Vec3>

namespace osgUtil
{
class Hit;
}

namespace opencover
{
/** Menu Item that reflects the state of a participant in a CVE
 */

class COVEREXPORT coPartnerMenuItem : public vrui::coCheckboxMenuItem
{
protected:
    vrui::coButton *viewpoint; ///< actual button which is used for interaction

public:
    coPartnerMenuItem(const char *, bool, vrui::coCheckboxGroup * = NULL);
    virtual ~coPartnerMenuItem();
    int hit(osg::Vec3 &, osgUtil::Hit *);
    void miss();
    virtual void buttonEvent(vrui::coButton *button);

    /// get the Element's classname
    virtual char *getClassName();
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(char *);
};
}
#endif
