/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   14.07.2010
**
**************************************************************************/

#include "shaperoaditem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/shapesection.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"

// Items //
//
#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "shapesectionitem.hpp"

// Editor //
//
#include "src/graph/editors/shapeeditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneHoverEvent>

ShapeRoadItem::ShapeRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road)
    : RoadItem(roadSystemItem, road)
    , shapeEditor_(NULL)
{
    init();
}

ShapeRoadItem::~ShapeRoadItem()
{
}

void
ShapeRoadItem::init()
{
    // ShapeEditor //
    //
	shapeEditor_ = dynamic_cast<ShapeEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());
    if (!shapeEditor_)
    {
        qDebug("Warning 1007141555! ShapeRoadItem not created by an ShapeEditor");
    }

    // SectionItems //
    //
    foreach (ShapeSection *section, getRoad()->getShapeSections())
    {
        new ShapeSectionItem(shapeEditor_, this, section);
    }
}

void
ShapeRoadItem::notifyDeletion()
{
//	shapeEditor_->delSelectedRoad(getRoad());
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
* Tells the ShapeEditor, that is has been selected/deselected.
*
*/
void
ShapeRoadItem::updateObserver()
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
    if (changes & RSystemElementRoad::CRD_ShapeSectionChange)
    {
        // A section has been added.
        //
        foreach (ShapeSection *section, getRoad()->getShapeSections())
        {
            if ((section->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (section->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new ShapeSectionItem(shapeEditor_, this, section);
            }
        }
    }

}
