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

#ifndef ROADVEHICLETREEITEM_HPP
#define ROADVEHICLETREEITEM_HPP

#include "src/tree/items/projecttreeitem.hpp"

class VehicleGroupTreeItem;
class RoadVehicle;

class RoadVehicleTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadVehicleTreeItem(VehicleGroupTreeItem *parent, RoadVehicle *roadVehicle, QTreeWidgetItem *fosterParent);
    virtual ~RoadVehicleTreeItem();

    // RoadVehicle //
    //
    RoadVehicle *getRoadVehicle() const
    {
        return roadVehicle_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    RoadVehicleTreeItem(); /* not allowed */
    RoadVehicleTreeItem(const RoadVehicleTreeItem &); /* not allowed */
    RoadVehicleTreeItem &operator=(const RoadVehicleTreeItem &); /* not allowed */

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
    VehicleGroupTreeItem *vehicleGroupTreeItem_;

    // RoadVehicle //
    //
    RoadVehicle *roadVehicle_;
};

#endif // ROADVEHICLETREEITEM_HPP
