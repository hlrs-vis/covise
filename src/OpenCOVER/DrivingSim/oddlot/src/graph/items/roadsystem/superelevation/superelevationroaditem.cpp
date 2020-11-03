/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   16.07.2010
**
**************************************************************************/

#include "superelevationroaditem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/superelevationsection.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"

// Items //
//
#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "superelevationsectionitem.hpp"

// Editor //
//
#include "src/graph/editors/superelevationeditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneHoverEvent>

SuperelevationRoadItem::SuperelevationRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road)
    : RoadItem(roadSystemItem, road)
    , superelevationEditor_(NULL)
{
    init();
}

SuperelevationRoadItem::~SuperelevationRoadItem()
{
	superelevationSectionItems_.clear();
}

void
SuperelevationRoadItem::init()
{
    // SuperelevationEditor //
    //
    superelevationEditor_ = dynamic_cast<SuperelevationEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());
    if (!superelevationEditor_)
    {
        qDebug("Warning 1007141555! SuperelevationRoadItem not created by an SuperelevationEditor");
    }

    // SectionItems //
    //
    foreach (SuperelevationSection *section, getRoad()->getSuperelevationSections())
    {
        superelevationSectionItems_.insert(section->getSStart(), new SuperelevationSectionItem(superelevationEditor_, this, section));
    }

}

void
SuperelevationRoadItem::notifyDeletion()
{
    superelevationEditor_->delSelectedRoad(getRoad());
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
* Tells the SuperelevationEditor, that is has been selected/deselected.
*
*/
void
SuperelevationRoadItem::updateObserver()
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
    if (changes & RSystemElementRoad::CRD_SuperelevationSectionChange)
    {
        // A section has been added.
        //
		QMap<double, SuperelevationSection *> roadSections = getRoad()->getSuperelevationSections();
        foreach (SuperelevationSection *section, roadSections)
        {
            if ((section->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (section->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                superelevationSectionItems_.insert(section->getSStart(), new SuperelevationSectionItem(superelevationEditor_, this, section));
            }
        }

		foreach(double s, superelevationSectionItems_.keys())
		{
			if (!roadSections.contains(s))
			{
				superelevationSectionItems_.remove(s);
			}
		}
    }

}



