/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/13/2010
**
**************************************************************************/

#ifndef SENSORTREEITEM_HPP
#define SENSORTREEITEM_HPP

#include "sectiontreeitem.hpp"

class RoadTreeItem;
class Sensor;

class SensorTreeItem : public SectionTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SensorTreeItem(RoadTreeItem *parent, Sensor *section, QTreeWidgetItem *fosterParent);
    virtual ~SensorTreeItem();

    // SensorSensor //
    //
    Sensor *getSensor() const
    {
        return sensor_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
    virtual void updateName();

private:
    SensorTreeItem(); /* not allowed */
    SensorTreeItem(const SensorTreeItem &); /* not allowed */
    SensorTreeItem &operator=(const SensorTreeItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // Road //
    //
    Sensor *sensor_;
};

#endif // SENSORTREEITEM_HPP
