/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   05.07.2010
**
**************************************************************************/

#ifndef OSCROADSYSTEMITEM_HPP
#define OSCROADSYSTEMITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class OSCRoadSystemItem : public RoadSystemItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit OSCRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem);
    virtual ~OSCRoadSystemItem();


private:
    OSCRoadSystemItem(); /* not allowed */
    OSCRoadSystemItem(const OSCRoadSystemItem &); /* not allowed */
    OSCRoadSystemItem &operator=(const OSCRoadSystemItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
};

#endif // OSCROADSYSTEMITEM_HPP
