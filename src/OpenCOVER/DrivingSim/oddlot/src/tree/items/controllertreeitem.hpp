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

#ifndef CONTROLLERTREEITEM_HPP
#define CONTROLLERTREEITEM_HPP

#include "projecttreeitem.hpp"

class RoadSystemTreeItem;
class RSystemElementController;

class ControllerTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ControllerTreeItem(RoadSystemTreeItem *parent, RSystemElementController *controller, QTreeWidgetItem *fosterParent);
    virtual ~ControllerTreeItem();

    // Controller //
    //
    RSystemElementController *getController() const
    {
        return controller_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    ControllerTreeItem(); /* not allowed */
    ControllerTreeItem(const ControllerTreeItem &); /* not allowed */
    ControllerTreeItem &operator=(const ControllerTreeItem &); /* not allowed */

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

    // Controller //
    //
    RSystemElementController *controller_;
};

#endif // CONTROLLERTREEITEM_HPP
