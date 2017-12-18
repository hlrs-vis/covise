/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/6/2010
**
**************************************************************************/

#ifndef ROADTREEITEM_HPP
#define ROADTREEITEM_HPP

#include "projecttreeitem.hpp"

class RoadSystemTreeItem;
class RSystemElementRoad;

class RoadTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadTreeItem(RoadSystemTreeItem *parent, RSystemElementRoad *road, QTreeWidgetItem *fosterParent);
    virtual ~RoadTreeItem();

    // Road //
    //
    RSystemElementRoad *getRoad() const
    {
        return road_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    RoadTreeItem(); /* not allowed */
    RoadTreeItem(const RoadTreeItem &); /* not allowed */
    RoadTreeItem &operator=(const RoadTreeItem &); /* not allowed */

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

    // Road //
    //
    RSystemElementRoad *road_;

    // Containers //
    //
    QTreeWidgetItem *typesItem_;
    QTreeWidgetItem *tracksItem_;
    QTreeWidgetItem *objectsItem_;
    QTreeWidgetItem *signalsItem_;
    QTreeWidgetItem *sensorsItem_;
    QTreeWidgetItem *elevationsItem_;
    QTreeWidgetItem *superelevationsItem_;
    QTreeWidgetItem *crossfallsItem_;
	QTreeWidgetItem *shapesItem_;
    QTreeWidgetItem *lanesItem_;
    QTreeWidgetItem *bridgesItem_;
};

#endif // ROADTREEITEM_HPP
