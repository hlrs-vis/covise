/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/11/2010
**
**************************************************************************/

#ifndef JUNCTIONTREEITEM_HPP
#define JUNCTIONTREEITEM_HPP

#include "projecttreeitem.hpp"

class RoadSystemTreeItem;
class RSystemElementJunction;

class JunctionTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionTreeItem(RoadSystemTreeItem *parent, RSystemElementJunction *junction, QTreeWidgetItem *fosterParent);
    virtual ~JunctionTreeItem();

    // Junction //
    //
    RSystemElementJunction *getJunction() const
    {
        return junction_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    JunctionTreeItem(); /* not allowed */
    JunctionTreeItem(const JunctionTreeItem &); /* not allowed */
    JunctionTreeItem &operator=(const JunctionTreeItem &); /* not allowed */

    void init();

    void updateName();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // Parent //
    //
    RoadSystemTreeItem *roadSystemTreeItem_;

    // Junction //
    //
    RSystemElementJunction *junction_;
};

#endif // JUNCTIONTREEITEM_HPP
