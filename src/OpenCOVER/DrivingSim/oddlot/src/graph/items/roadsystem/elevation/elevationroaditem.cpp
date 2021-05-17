/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   22.06.2010
 **
 **************************************************************************/

#include "elevationroaditem.hpp"

 // Data //
 //
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/elevationsection.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"

// Items //
//
#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "elevationsectionitem.hpp"

// Editor //
//
#include "src/graph/editors/elevationeditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneHoverEvent>

ElevationRoadItem::ElevationRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road)
    : RoadItem(roadSystemItem, road)
    , elevationEditor_(NULL)
{
    init();
}

ElevationRoadItem::~ElevationRoadItem()
{
    elevationSectionItems_.clear();
}

void
ElevationRoadItem::init()
{
    // ElevationEditor //
    //
    elevationEditor_ = dynamic_cast<ElevationEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());
    if (!elevationEditor_)
    {
        qDebug("Warning 1006241105! ElevationRoadItem not created by an ElevationEditor");
    }

    // SectionItems //
    //
    foreach(ElevationSection * section, getRoad()->getElevationSections())
    {
        elevationSectionItems_.insert(section->getSStart(), new ElevationSectionItem(elevationEditor_, this, section));
    }
}

void
ElevationRoadItem::notifyDeletion()
{
    elevationEditor_->delSelectedRoad(getRoad());
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
* Tells the Elevation Editor, that is has been selected/deselected.
*
*/
void
ElevationRoadItem::updateObserver()
{
    // Parent //
    //
    RoadItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Road //
    //
    int changes = getRoad()->getRoadChanges();
    if (changes & RSystemElementRoad::CRD_ElevationSectionChange)
    {
        // A section has been added.
        //
        QMap<double, ElevationSection *> roadSections = getRoad()->getElevationSections();
        foreach(ElevationSection * section, roadSections)
        {
            int dataElementChanges = section->getDataElementChanges();
            if ((dataElementChanges & DataElement::CDE_DataElementCreated)
                || (dataElementChanges & DataElement::CDE_DataElementAdded))
            {
                elevationSectionItems_.insert(section->getSStart(), new ElevationSectionItem(elevationEditor_, this, section));
            }
        }

        foreach(double s, elevationSectionItems_.keys())
        {
            if (!roadSections.contains(s))
            {
                elevationSectionItems_.remove(s);
            }
        }
    }
}



