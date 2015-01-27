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

#ifndef VEHICLESYSTEMTREEITEM_HPP
#define VEHICLESYSTEMTREEITEM_HPP

#include "src/tree/items/projecttreeitem.hpp"

class VehicleSystem;

class VehicleSystemTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit VehicleSystemTreeItem(ProjectTree *projectTree, VehicleSystem *vehicleSystem, QTreeWidgetItem *rootItem);
    virtual ~VehicleSystemTreeItem();

    // Tree //
    //
    virtual ProjectTree *getProjectTree() const
    {
        return projectTree_;
    }

    // VehicleSystem //
    //
    VehicleSystem *getVehicleSystem() const
    {
        return vehicleSystem_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    VehicleSystemTreeItem(); /* not allowed */
    VehicleSystemTreeItem(const VehicleSystemTreeItem &); /* not allowed */
    VehicleSystemTreeItem &operator=(const VehicleSystemTreeItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // Tree //
    //
    ProjectTree *projectTree_;

    // VehicleSystem //
    //
    VehicleSystem *vehicleSystem_;

    // Items //
    //
    QTreeWidgetItem *rootItem_;

    QTreeWidgetItem *vehicleGroupsItem_;
};

#endif // VEHICLESYSTEMTREEITEM_HPP
