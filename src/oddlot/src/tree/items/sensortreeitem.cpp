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

#include "sensortreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/sensorobject.hpp"

// Tree //
//
#include "roadtreeitem.hpp"

SensorTreeItem::SensorTreeItem(RoadTreeItem *parent, Sensor *sensor, QTreeWidgetItem *fosterParent)
    : SectionTreeItem(parent, sensor, fosterParent)
    , sensor_(sensor)
{
    init();
}

SensorTreeItem::~SensorTreeItem()
{
}

void
SensorTreeItem::init()
{
    updateName();
}

void
SensorTreeItem::updateName()
{
    QString text = QString("%1").arg(sensor_->getId());
    text.append(QString("%1").arg(sensor_->getS()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
SensorTreeItem::updateObserver()
{

    // Parent //
    //
    SectionTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Sensor //
    //
    int changes = sensor_->getSensorChanges();

    if (changes & Sensor::CEL_ParameterChange)
    {
        updateName();
    }
}
