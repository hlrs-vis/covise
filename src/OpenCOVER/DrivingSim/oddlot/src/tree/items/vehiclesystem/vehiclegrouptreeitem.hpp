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

#ifndef VEHICLEGROUPTREEITEM_HPP
#define VEHICLEGROUPTREEITEM_HPP

#include "src/tree/items/projecttreeitem.hpp"

class VehicleSystemTreeItem;
class VehicleGroup;

class VehicleGroupTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit VehicleGroupTreeItem(VehicleSystemTreeItem *parent, VehicleGroup *group, QTreeWidgetItem *fosterParent);
    virtual ~VehicleGroupTreeItem();

    // VehicleGroup //
    //
    VehicleGroup *getVehicleGroup() const
    {
        return group_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    VehicleGroupTreeItem(); /* not allowed */
    VehicleGroupTreeItem(const VehicleGroupTreeItem &); /* not allowed */
    VehicleGroupTreeItem &operator=(const VehicleGroupTreeItem &); /* not allowed */

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
    VehicleSystemTreeItem *vehicleSystemTreeItem_;

    // VehicleGroup //
    //
    VehicleGroup *group_;
};

#endif // VEHICLEGROUPTREEITEM_HPP
