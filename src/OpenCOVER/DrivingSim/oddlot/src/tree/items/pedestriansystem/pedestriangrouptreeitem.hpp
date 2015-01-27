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

#ifndef PEDESTRIANGROUPTREEITEM_HPP
#define PEDESTRIANGROUPTREEITEM_HPP

#include "src/tree/items/projecttreeitem.hpp"

class PedestrianSystemTreeItem;
class PedestrianGroup;

class PedestrianGroupTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit PedestrianGroupTreeItem(PedestrianSystemTreeItem *parent, PedestrianGroup *group, QTreeWidgetItem *fosterParent);
    virtual ~PedestrianGroupTreeItem();

    // PedestrianGroup //
    //
    PedestrianGroup *getPedestrianGroup() const
    {
        return group_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    PedestrianGroupTreeItem(); /* not allowed */
    PedestrianGroupTreeItem(const PedestrianGroupTreeItem &); /* not allowed */
    PedestrianGroupTreeItem &operator=(const PedestrianGroupTreeItem &); /* not allowed */

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
    PedestrianSystemTreeItem *pedestrianSystemTreeItem_;

    // PedestrianGroup //
    //
    PedestrianGroup *group_;
};

#endif // PEDESTRIANGROUPTREEITEM_HPP
