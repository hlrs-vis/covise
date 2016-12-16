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

#include "oscitem.hpp"
#include "osctextitem.hpp"

#include "src/graph/items/roadsystem/scenario/oscroaditem.hpp"

#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"
#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/oscsystem/oscelement.hpp"
#include "src/data/commands/osccommands.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphview.hpp"
//#include "src/graph/items/roadsystem/signal/signaltextitem.hpp"
#include "src/graph/items/roadsystem/scenario/oscroadsystemitem.hpp"
#include "src/graph/items/oscsystem/oscbaseitem.hpp"
#include "src/graph/editors/osceditor.hpp"

// Tools //
//
#include "src/gui/tools/toolaction.hpp"
#include "src/gui/tools/zoomtool.hpp"

// OpenScenario //
//
#include "schema/oscVehicle.h"
#include "schema/oscObject.h"
#include "oscMember.h"
#include "schema/oscCatalogReference.h"
//#include "oscNameRefId.h"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QColor>
#include <QString>
#include <QKeyEvent>

OSCItem::OSCItem(OSCElement *element, OSCBaseItem *oscBaseItem, OpenScenario::oscObject *oscObject, OpenScenario::oscCatalog *catalog, const QPointF &pos, const QString &roadId)
    : GraphElement(oscBaseItem, element)
	, element_(element)
	, oscBaseItem_(oscBaseItem)
    , oscObject_(oscObject)
	, oscPrivateAction_(NULL)
	, catalog_(catalog)
	, selectedObject_(NULL)
	, path_(NULL)
	, pos_(pos)
	, roadID_(roadId)
{

    init();
}

OSCItem::~OSCItem()
{
}

/*!
* Initializes the path 
*/
QPainterPath *
	createVehiclePath(OpenScenario::oscObjectBase *vehicle)
{
	QPainterPath *path = new QPainterPath();
	double width = 10;
	double height = 10;

	oscIntValue *iv = dynamic_cast<oscIntValue *>(vehicle->getMember("category")->getOrCreateValue());
	if (iv)
	{
		switch (iv->getValue())
		{
		case oscVehicle::car:
			{
				QPolygonF polygon;
				polygon << QPointF(0,0) << QPointF(0,2) << QPointF(0,2) << QPointF(3.7,4) << QPointF(6.3,4) << QPointF(7.7,2) << QPointF(9.2,2) << QPointF(9.8,1.2) << QPointF(10,0);
				path->addPolygon(polygon);
				path->closeSubpath();
				path->addEllipse(QPointF(2,-0.1), 0.8, 0.8);
				path->addEllipse(QPointF(8,-0.1), 0.8, 0.8);

				height = 4;
				break;
			}
		default:
			{
				path->addRect(0, 0, 10, 10);
			}

	/*	truck,
		trailer,
		bus,
		motorbike,
		bicycle,
		train,
		tram,*/
		}
	}
	path->translate(-width/2, -height/2);

	return path;

}

void
OSCItem::init()
{
	oscBaseItem_->appendOSCItem(this);
	
    // Hover Events //
    //
    setAcceptHoverEvents(true);
//    setSelectable();
	setFlag(QGraphicsItem::ItemIsSelectable);
	setFlag(ItemIsFocusable);

    // OpenScenario Editor
    //
    oscEditor_ = dynamic_cast<OpenScenarioEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());

    // Context Menu //
    //

    QAction *removeElementAction = getRemoveMenu()->addAction(tr("OpenScenario Object"));
    connect(removeElementAction, SIGNAL(triggered()), this, SLOT(removeElement()));

    if (getTopviewGraph()) // not for profile graph
    {
        // Text //
        //
		QString name = updateName();
        oscTextItem_ = new OSCTextItem(element_, this, name, pos_);
        oscTextItem_->setZValue(1.0); // stack before siblings
    }

	OpenScenario::oscCatalogReference *catalogReference = oscObject_->CatalogReference.getObject();
	if (!catalogReference)
	{
		return;
	}

	std::string catalogFileName = catalogReference->name.getValue();
	oscPrivateAction_ = oscEditor_->getOrCreatePrivateAction(catalogFileName);

	roadSystem_ = getProjectGraph()->getProjectData()->getRoadSystem();
	road_ = roadSystem_->getRoad(roadID_);
	closestRoad_ = road_;
	roadSystemItem_ = oscBaseItem_->getRoadSystemItem();

	doPan_ = false;
	copyPan_ = false;



	// TODO: get type and object from catalog reference //
	//

	OpenScenario::oscObjectBase *catalogObject = catalog_->getCatalogObject(catalogFileName);

	if (!catalogObject)
	{
		catalog_->fullReadCatalogObjectWithName(catalogFileName);
		catalogObject = catalog_->getCatalogObject(catalogFileName);
	}
	
	if (catalogObject)
	{
		OpenScenario::oscArrayMember *objects = dynamic_cast<OpenScenario::oscArrayMember *>(catalogObject->getMember(catalog_->getCatalogName()));
		if (objects && !objects->empty())
		{
			selectedObject_ = objects->at(0);
		}

		if (selectedObject_)
		{
			if (catalog_->getCatalogName() == "Vehicle")
			{
				createPath = createVehiclePath;
				createPath(selectedObject_);

				updateColor(catalog_->getCatalogName());
				updatePosition();
			}
		}
	}
}

QString
    OSCItem::updateName()
{
    QString name = "";
    OpenScenario::oscMemberValue *value =  oscObject_->getMember("name")->getOrCreateValue();
    oscStringValue *sv = dynamic_cast<oscStringValue *>(value);
    if (sv)
    {
        name = QString::fromStdString(sv->getValue());
    }

    return name;
}


/*! \brief Sets the color according to the number of links.
*/
void
OSCItem::updateColor(const std::string &type)
{
	if (type == "Vehicle")
	{
		color_ = Qt::black;
	}

	
//	setBrush(QBrush(color_));
	setPen(QPen(color_));
}


/*
* Update position
*/
void
OSCItem::updatePosition()
{
	path_ = createPath(selectedObject_);
	path_->translate(pos_ );
	setPath(*path_);
}

//*************//
// Delete Item
//*************//

bool
OSCItem::deleteRequest()
{
    if (removeElement())
    {
        return true;
    }

    return false;
}

//################//
// SLOTS          //
//################//

bool
OSCItem::removeElement()
{
	OpenScenario::oscObjectBase *parent = oscObject_->getParentObj();
	OpenScenario::oscArrayMember *arrayMember = dynamic_cast<OpenScenario::oscArrayMember *>(parent->getOwnMember());

	RemoveOSCArrayMemberCommand *command = new RemoveOSCArrayMemberCommand(arrayMember, oscObject_, arrayMember->findObjectIndex(oscObject_), element_);
	getTopviewGraph()->executeCommand(command); 

	return false;
}

//################//
// EVENTS         //
//################//

void
OSCItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{

	setCursor(Qt::OpenHandCursor);
	setFocus();

	// Text //
	//
	oscTextItem_->setVisible(true);
	oscTextItem_->setPos(event->scenePos());

	// Parent //
	//
	GraphElement::hoverEnterEvent(event); // pass to baseclass
}

void
OSCItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::ArrowCursor);
	if (!copyPan_)
	{
		clearFocus();
	}

    // Text //
    //
    oscTextItem_->setVisible(false);

    // Parent //
    //
    GraphElement::hoverLeaveEvent(event); // pass to baseclass
}

void
OSCItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{

    // Parent //
    //
    GraphElement::hoverMoveEvent(event);
}

void
OSCItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    pressPos_ = lastPos_ = event->scenePos();
    ODD::ToolId tool = oscEditor_->getCurrentTool(); // Editor Delete Signal
    if (tool == ODD::TSG_DEL)
    {
        removeElement();
    }
    else 
    {

		doPan_ = true;
		if (copyPan_)
		{
	/*		Signal * newSignal = signal_->getClone();
			AddSignalCommand *command = new AddSignalCommand(newSignal, signal_->getParentRoad(), NULL);
			getProjectGraph()->executeCommand(command); */
		}

        GraphElement::mousePressEvent(event); // pass to baseclass

    }

	oscTextItem_->setVisible(false);
}

void
OSCItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{	
	if (doPan_)
	{

		QPointF newPos = event->scenePos();
		path_->translate(newPos-lastPos_);
		lastPos_ = newPos;
		setPath(*path_);

		QVector2D vec;

		RSystemElementRoad * nearestRoad = roadSystem_->findClosestRoad( newPos, s_, t_, vec);
		if (!nearestRoad)
		{
			nearestRoad = road_;
		}
		if (nearestRoad != closestRoad_)
		{
			RoadItem *nearestRoadItem = roadSystemItem_->getRoadItem(nearestRoad->getID());
			nearestRoadItem->setHighlighting(true);
			setZValue(nearestRoadItem->zValue() + 1);
			roadSystemItem_->getRoadItem(closestRoad_->getID())->setHighlighting(false);
			closestRoad_ = nearestRoad;
		}

		GraphElement::mouseMoveEvent(event);
	}
}

void
OSCItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{

    if (doPan_)
    {
		double diff = (lastPos_ - pressPos_).manhattanLength();
		if (diff > 0.01) // otherwise item has not been moved by intention
		{
			bool parentChanged = oscEditor_->translateObject(oscPrivateAction_, closestRoad_->getID(), s_, t_);
			pos_ += lastPos_ - pressPos_;

		}
		else
		{
			pos_ = lastPos_;
		}

		doPan_ = false;
    }

	oscTextItem_->setVisible(true);

	GraphElement::mouseReleaseEvent(event);
}

/*! \brief Key events for panning, etc.
*
*/
void
OSCItem::keyPressEvent(QKeyEvent *event)
{
    // TODO: This will not notice a key pressed, when the view is not active
    switch (event->key())
    {
	case Qt::Key_Shift:
        copyPan_ = true;
        break;

    default:
        QGraphicsItem::keyPressEvent(event);
    }
}

/*! \brief Key events for panning, etc.
*
*/
void
OSCItem::keyReleaseEvent(QKeyEvent *event)
{
    switch (event->key())
    {
    case Qt::Key_Shift:
        copyPan_ = false;
		if (!isHovered())
		{
			clearFocus();
		}
        break;


    default:
        QGraphicsItem::keyReleaseEvent(event);
    }
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
OSCItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Get change flags //
    //
	int changes = element_->getOSCElementChanges();

    if (changes & OSCElement::COE_ParameterChange)
    {
        // Text //
        //
        if (oscTextItem_)
        {
            oscTextItem_->updateText(updateName());
        }
    }

    // Signal //
    //
 /*   int changes = signal_->getSignalChanges();

    if ((changes & Signal::CEL_TypeChange))
    {
        updatePosition();
    }
    else if ((changes & Signal::CEL_ParameterChange))
    {
        updatePosition();
    } */
}
