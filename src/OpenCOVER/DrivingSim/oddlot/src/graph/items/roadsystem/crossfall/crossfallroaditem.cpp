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

#include "crossfallroaditem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/crossfallsection.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"

// Items //
//
#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "crossfallsectionitem.hpp"

// Editor //
//
#include "src/graph/editors/crossfalleditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneHoverEvent>

CrossfallRoadItem::CrossfallRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road)
    : RoadItem(roadSystemItem, road)
    , crossfallEditor_(NULL)
{
    init();
}

CrossfallRoadItem::~CrossfallRoadItem()
{
}

void
CrossfallRoadItem::init()
{
    // CrossfallEditor //
    //
    crossfallEditor_ = dynamic_cast<CrossfallEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());
    if (!crossfallEditor_)
    {
        qDebug("Warning 1007141555! CrossfallRoadItem not created by an CrossfallEditor");
    }

    // SectionItems //
    //
    foreach (CrossfallSection *section, getRoad()->getCrossfallSections())
    {
        new CrossfallSectionItem(crossfallEditor_, this, section);
    }

    // Selection //
    //
    if (getRoad()->isElementSelected())
    {
        crossfallEditor_->addSelectedRoad(getRoad());
    }
    else
    {
        crossfallEditor_->delSelectedRoad(getRoad());
    }
}

void
CrossfallRoadItem::notifyDeletion()
{
    crossfallEditor_->delSelectedRoad(getRoad());
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
* Tells the CrossfallEditor, that is has been selected/deselected.
*
*/
void
CrossfallRoadItem::updateObserver()
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
    if (changes & RSystemElementRoad::CRD_CrossfallSectionChange)
    {
        // A section has been added.
        //
        foreach (CrossfallSection *section, getRoad()->getCrossfallSections())
        {
            if ((section->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (section->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new CrossfallSectionItem(crossfallEditor_, this, section);
            }
        }
    }

    // DataElement //
    //
    int dataElementChanges = getRoad()->getDataElementChanges();
    if ((dataElementChanges & DataElement::CDE_SelectionChange)
        || (dataElementChanges & DataElement::CDE_ChildSelectionChange))
    {
        // Selection //
        //
        if (getRoad()->isElementSelected() || getRoad()->isChildElementSelected())
        {
            crossfallEditor_->addSelectedRoad(getRoad());
        }
        else
        {
            crossfallEditor_->delSelectedRoad(getRoad());
        }
    }
}
