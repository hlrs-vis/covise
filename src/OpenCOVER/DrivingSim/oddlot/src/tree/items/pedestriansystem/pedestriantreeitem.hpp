/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/17/2010
**
**************************************************************************/

#ifndef PEDESTRIANTREEITEM_HPP
#define PEDESTRIANTREEITEM_HPP

#include "src/tree/items/projecttreeitem.hpp"

class PedestrianGroupTreeItem;
class Pedestrian;

class PedestrianTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit PedestrianTreeItem(PedestrianGroupTreeItem *parent, Pedestrian *pedestrian, QTreeWidgetItem *fosterParent);
    virtual ~PedestrianTreeItem();

    // Pedestrian //
    //
    Pedestrian *getPedestrian() const
    {
        return pedestrian_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    PedestrianTreeItem(); /* not allowed */
    PedestrianTreeItem(const PedestrianTreeItem &); /* not allowed */
    PedestrianTreeItem &operator=(const PedestrianTreeItem &); /* not allowed */

    void init();
    void updateText();

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
    PedestrianGroupTreeItem *pedestrianGroupTreeItem_;

    // Pedestrian //
    //
    Pedestrian *pedestrian_;
};

#endif // PEDESTRIANTREEITEM_HPP
