/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   05.07.2010
**
**************************************************************************/

#include "roadtyperoaditem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/typesection.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"

// Items //
//
#include "roadtyperoadsystemitem.hpp"
#include "typesectionitem.hpp"

// Editor //
//
#include "src/graph/editors/roadtypeeditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneHoverEvent>

RoadTypeRoadItem::RoadTypeRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road)
    : RoadItem(roadSystemItem, road)
    , roadTypeEditor_(NULL)
{
    init();
}

RoadTypeRoadItem::~RoadTypeRoadItem()
{
}

void
RoadTypeRoadItem::init()
{
    // RoadTypeEditor //
    //
    roadTypeEditor_ = dynamic_cast<RoadTypeEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());
    if (!roadTypeEditor_)
    {
        qDebug("Warning 1007041414! RoadTypeRoadItem not created by an RoadTypeEditor");
    }

    // SectionItems //
    //
    foreach (TypeSection *section, getRoad()->getTypeSections())
    {
        new TypeSectionItem(roadTypeEditor_, this, section);
    }
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
RoadTypeRoadItem::updateObserver()
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

    if (changes & RSystemElementRoad::CRD_TypeSectionChange)
    {
        // A section has been added.
        //
        foreach (TypeSection *section, getRoad()->getTypeSections())
        {
            if ((section->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (section->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new TypeSectionItem(roadTypeEditor_, this, section);
            }
        }
    }
}

//################//
// EVENTS         //
//################//
