/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.03.2010
**
**************************************************************************/

#include "oscbaseitem.hpp"

// Data //
//
#include "src/data/oscsystem/oscbase.hpp"
#include "src/data/oscsystem/oscelement.hpp"
#include "src/data/commands/osccommands.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphview.hpp"
#include "src/graph/graphscene.hpp"
//#include "src/graph/items/roadsystem/signal/signaltextitem.hpp"
#include "src/graph/items/oscsystem/oscitem.hpp"
#include "src/graph/items/roadsystem/scenario/oscroadsystemitem.hpp"
#include "src/graph/editors/osceditor.hpp"

// Tools //
//
#include "src/gui/tools/toolaction.hpp"
#include "src/gui/tools/zoomtool.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// OpenScenario //
//
#include "oscVehicle.h"
#include "oscObject.h"
#include "oscObjectBase.h"
#include "oscMember.h"
#include "oscPosition.h"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QColor>
#include <QString>
#include <QKeyEvent>

using namespace OpenScenario;

OSCBaseItem::OSCBaseItem(TopviewGraph *topviewGraph, OSCBase *oscBase)
	: GraphElement(NULL, oscBase)
	, topviewGraph_(topviewGraph)
	, oscBase_(oscBase)
{
	roadSystem_ = topviewGraph->getProjectData()->getRoadSystem();

	// OpenScenario Editor
    //
	OpenScenarioEditor *oscEditor = dynamic_cast<OpenScenarioEditor *>(topviewGraph_->getProjectWidget()->getProjectEditor());

	oscRoadSystemItem_ = oscEditor->getRoadSystemItem();

    init();
}

OSCBaseItem::~OSCBaseItem()
{
}

void
OSCBaseItem::init()
{	OpenScenario::OpenScenarioBase *openScenarioBase = oscBase_->getOpenScenarioBase();
	OpenScenario::oscCatalogs *catalogs = openScenarioBase->catalogs.getOrCreateObject();
	entityCatalog_ = catalogs->getCatalog("entityCatalog");

	// Root Base item //
    //

	foreach (OSCElement *element, oscBase_->getOSCElements())
    {
		OpenScenario::oscObject *object = dynamic_cast<OpenScenario::oscObject *>(element->getObject());
		if (object)
		{
			OpenScenario::oscPosition *oscPosition = object->initPosition.getObject();

			if (oscPosition)
			{
				OpenScenario::oscPositionRoad *oscPosRoad = oscPosition->positionRoad.getObject();
				if (oscPosRoad)
				{
					QString roadId = QString::fromStdString(oscPosRoad->roadId.getValue());
					RSystemElementRoad *road = roadSystem_->getRoad(roadId);
					if (road)
					{
						double s = oscPosRoad->s.getValue();
						double t = oscPosRoad->t.getValue();
						new OSCItem(element, this, object, entityCatalog_, road->getGlobalPoint(s, t), roadId);
					}
				}
			}
		}
    }

}

void
OSCBaseItem::appendOSCItem(OSCItem *oscItem)
{
	OSCElement *element = dynamic_cast<OSCElement *>(oscItem->getDataElement());
	QString id = element->getID();
    if (!oscItems_.contains(id))
    {
        oscItems_.insert(id, oscItem);
    }
}

bool
OSCBaseItem::removeOSCItem(OSCItem *oscItem)
{
	OSCElement *element = dynamic_cast<OSCElement *>(oscItem->getDataElement());
    return oscItems_.remove(element->getID());
}


//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
OSCBaseItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Base //
    //
	int changes = oscBase_->getOSCBaseChanges();

	if (changes & OSCBase::OSCBaseChange::COSC_ElementChange)
    {
		QMap<QString, OSCElement *>::const_iterator iter = oscBase_->getOSCElements().constBegin();
         while (iter != oscBase_->getOSCElements().constEnd())
        {
            OSCElement *element = iter.value();
			OpenScenario::oscObject *object = dynamic_cast<OpenScenario::oscObject *>(element->getObject());
			if (object)
			{
				if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
					|| (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
				{
					OpenScenario::oscPosition *oscPosition = object->initPosition.getObject();

					if (oscPosition)
					{
						OpenScenario::oscPositionRoad *oscPosRoad = oscPosition->positionRoad.getObject();
						if (oscPosRoad)
						{
							QString id = QString::fromStdString(oscPosRoad->roadId.getValue());
							RSystemElementRoad *road = roadSystem_->getRoad(id);
							if (road)
							{
								new OSCItem(element, this, object, entityCatalog_, road->getGlobalPoint(oscPosRoad->s.getValue(), oscPosRoad->t.getValue()), id);
							}
						}
					}
				}
			}

            iter++;
        }
    }

}
