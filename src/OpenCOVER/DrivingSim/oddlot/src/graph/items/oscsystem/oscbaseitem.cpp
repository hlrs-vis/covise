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
#include "src/graph/items/oscsystem/oscshapeitem.hpp"
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
#include "schema/oscVehicle.h"
#include "schema/oscObject.h"
#include "oscObjectBase.h"
#include "oscMember.h"
#include "schema/oscCatalogs.h"
#include "schema/oscPosition.h"
#include "schema/oscTrajectory.h"
#include "schema/oscPrivateAction.h"
#include "schema/oscPrivate.h"

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
{	
	OpenScenario::OpenScenarioBase *openScenarioBase = oscBase_->getOpenScenarioBase();
	catalogs_ = openScenarioBase->Catalogs.getOrCreateObject();
	OpenScenario::oscStoryboard *story = openScenarioBase->Storyboard.getOrCreateObject();
	OpenScenario::oscInit *init = story->Init.getOrCreateObject();
	actions_ = init->Actions.getOrCreateObject();
	OpenScenario::oscArrayMember *privateArray = dynamic_cast<OpenScenario::oscArrayMember *>(actions_->getMember("Private"));

	// Root Base item //
    //
	foreach (OSCElement *element, oscBase_->getOSCElements())
	{
		OpenScenario::oscObject *object = dynamic_cast<OpenScenario::oscObject *>(element->getObject());
		if (object)
		{
			std::string objectName = object->name.getValue();
			OpenScenario::oscCatalog *catalog = getCatalog(object);
			if (!catalog)
			{
				continue;
			}

			for(oscArrayMember::iterator it =privateArray->begin();it != privateArray->end();it++)
			{
				OpenScenario::oscPrivate *privateObject = dynamic_cast<OpenScenario::oscPrivate *>(*it);
				if (privateObject->object.getValue() == objectName)
				{
					OpenScenario::oscPrivateAction *privateAction = getPrivateAction(object, privateObject);

					if (!privateAction)
					{
						break;
					}

					OpenScenario::oscPosition *oscPosition = privateAction->Position.getObject();

					if (oscPosition)
					{

						OpenScenario::oscRoad *oscPosRoad = oscPosition->Road.getObject();
						if (oscPosRoad)
						{

							QString roadId = QString::fromStdString(oscPosRoad->roadId.getValue());
							RSystemElementRoad *road = roadSystem_->getRoad(roadId);
							if (road)
							{
								double s = oscPosRoad->s.getValue();
								double t = oscPosRoad->t.getValue();
								new OSCItem(element, this, object, catalog, road->getGlobalPoint(s, t), roadId);
							}
						}
						break;
					}
				}
			}
		}

		else
		{
			OpenScenario::oscTrajectory *trajectory = dynamic_cast<OpenScenario::oscTrajectory *>(element->getObject());
			if (trajectory)
			{
				new OSCShapeItem(element, this, trajectory);
			}
		}
	}

}

OpenScenario::oscCatalog *
OSCBaseItem::getCatalog(OpenScenario::oscObject *object)
{
	OpenScenario::oscCatalog *catalog = NULL;

	OpenScenario::oscCatalogReference *catalogReference = object->CatalogReference.getObject();
	if (!catalogReference)
	{
		return NULL;
	}
	std::string catalogName = catalogReference->catalog.getValue();
	OpenScenario::oscMember *catalogMember = catalogs_->getMember(catalogName);
	if (catalogMember)
	{
		catalog = dynamic_cast<OpenScenario::oscCatalog *>(catalogMember->getObject());
	}

	return catalog;
}


OpenScenario::oscPrivateAction *
OSCBaseItem::getPrivateAction(OpenScenario::oscObject *object, OpenScenario::oscPrivate *privateObject)
{
	OpenScenario::oscPrivateAction *privateAction = NULL;

	OpenScenario::oscArrayMember *privateActionsArray = dynamic_cast<OpenScenario::oscArrayMember *>(privateObject->getMember("Action"));

	for(oscArrayMember::iterator it =privateActionsArray->begin();it != privateActionsArray->end();it++)
	{
		OpenScenario::oscPrivateAction *action = dynamic_cast<OpenScenario::oscPrivateAction *>(*it);

		if (action)
		{
			OpenScenario::oscMember *member = action->getMember("Position");
			if (member->isSelected())
			{
				privateAction = action;
				break;
			}
		}
	}

	return privateAction;
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

void
OSCBaseItem::appendOSCShapeItem(OSCShapeItem *oscShapeItem)
{
	OSCElement *element = dynamic_cast<OSCElement *>(oscShapeItem->getDataElement());
	QString id = element->getID();
    if (!oscShapeItems_.contains(id))
    {
        oscShapeItems_.insert(id, oscShapeItem);
    }
}

bool
OSCBaseItem::removeOSCShapeItem(OSCShapeItem *oscShapeItem)
{
	OSCElement *element = dynamic_cast<OSCElement *>(oscShapeItem->getDataElement());
    return oscShapeItems_.remove(element->getID());
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
				OpenScenario::oscCatalog *catalog = getCatalog(object);
				if (!catalog)
				{
					iter++;
					continue;
				}

				if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
					|| (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
				{
					OpenScenario::oscArrayMember *privateArray = dynamic_cast<OpenScenario::oscArrayMember *>(actions_->getOwnMember());

					// Root Base item //
					//

					OpenScenario::oscPrivate *privateObject = NULL;
					for(oscArrayMember::iterator it =privateArray->begin();it != privateArray->end();it++)
					{
						privateObject = dynamic_cast<OpenScenario::oscPrivate *>(*it);
						if (privateObject->object.getValue() == object->name.getValue())
						{
							break;
						}
					}

					if (!privateObject)
					{
						iter++;
						continue;
					}
					OpenScenario::oscPrivateAction *privateAction = getPrivateAction(object, privateObject);

					if (!privateAction)
					{
						iter++;
						continue;
					}

					OpenScenario::oscPosition *oscPosition = privateAction->Position.getObject();

					if (oscPosition)
					{
						OpenScenario::oscRoad *oscPosRoad = oscPosition->Road.getObject();
						if (oscPosRoad)
						{
							QString id = QString::fromStdString(oscPosRoad->roadId.getValue());
							RSystemElementRoad *road = roadSystem_->getRoad(id);
                            if (road)
                            {
                                new OSCItem(element, this, object, catalog, road->getGlobalPoint(oscPosRoad->s.getValue(), oscPosRoad->t.getValue()), id);
                            }
                        }
                    }
				}
			}
			else
			{

				OpenScenario::oscTrajectory *trajectory = dynamic_cast<OpenScenario::oscTrajectory *>(element->getObject());
				if (trajectory)
				{
					if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
						|| (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
					{
						new OSCShapeItem(element, this, trajectory);
					}
				}

			}
            iter++;
        }
    }

}
