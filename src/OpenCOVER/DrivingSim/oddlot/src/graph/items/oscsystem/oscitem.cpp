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
#include "src/graph/items/oscsystem/svgitem.hpp"
#include "src/graph/editors/osceditor.hpp"

// Tools //
//
#include "src/gui/tools/toolaction.hpp"
#include "src/gui/tools/zoomtool.hpp"

// OpenScenario //
//
#include <OpenScenario/schema/oscVehicle.h>
#include <OpenScenario/schema/oscObject.h>
#include <OpenScenario/oscMember.h>
#include <OpenScenario/schema/oscCatalogReference.h>
//#include <OpenScenario/oscNameRefId.h>

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QColor>
#include <QString>
#include <QKeyEvent>
#include <QTransform>


OSCItem::OSCItem(OSCElement *element, OSCBaseItem *oscBaseItem, OpenScenario::oscObject *oscObject, OpenScenario::oscCatalog *catalog, OpenScenario::oscRoad *oscRoad)
    : GraphElement(oscBaseItem, element)
	, element_(element)
	, oscBaseItem_(oscBaseItem)
    , oscObject_(oscObject)
	, oscRoad_(oscRoad)
	, catalog_(catalog)
	, angle_(0)
{

    init();
}

OSCItem::~OSCItem()
{
	delete svgItem_;
}

void OSCItem::updateIcon()
{
	OpenScenario::oscVehicle *vehicle = dynamic_cast<OpenScenario::oscVehicle *>(oscObject_);
	double widthBoundBox = vehicle->BoundingBox->Dimension->width;
	double lengthBoundBox = vehicle->BoundingBox->Dimension->length;
	double heightBoundBox = vehicle->BoundingBox->Dimension->height;
	if (widthBoundBox < 1.0)
		widthBoundBox = 1.8;
	if (lengthBoundBox < 1.0)
		lengthBoundBox = 3;
	if (heightBoundBox < 1.0)
		heightBoundBox = 1.5;
	if (vehicle)
	{
		//svgItem->setElementId((int)vehicle->category.getValue())
		switch (vehicle->category.getValue())
		{
		case oscVehicle::car:
		{
			break;
		}
		case oscVehicle::truck:
		{
			break;
		}
		case oscVehicle::bus:
		{
			break;
		}
		default:
		{
			//path.addRect(0, 0, lengthBoundBox, heightBoundBox);
		}

		//	trailer,
		//	motorbike,
		//	bicycle,
		//		train,
		//	tram,
		}
	}
}
/*!
* Initializes the path 

QPainterPath 
	createVehiclePath(OpenScenario::oscObjectBase *object, RSystemElementRoad *road)
{
	QPainterPath path = QPainterPath();
	
	OpenScenario::oscVehicle *vehicle = dynamic_cast<OpenScenario::oscVehicle *>(object);
	double widthBoundBox = vehicle->BoundingBox->Dimension->width;
	double lengthBoundBox = vehicle->BoundingBox->Dimension->length;
	double heightBoundBox = vehicle->BoundingBox->Dimension->height;
	if (widthBoundBox < 1.0)
		widthBoundBox = 1.8;
	if (lengthBoundBox < 1.0)
		lengthBoundBox = 3;
	if (heightBoundBox < 1.0)
		heightBoundBox = 1.5;
	if(vehicle)
	{
		switch (vehicle->category.getValue())
		{
		case oscVehicle::car:
			{
				QPolygonF polygon;
				polygon << QPointF(0,0) << QPointF(0, widthBoundBox /2) << QPointF(lengthBoundBox/3, widthBoundBox) << QPointF(lengthBoundBox/1.5, widthBoundBox) << QPointF(lengthBoundBox/1.2, widthBoundBox /2) << QPointF(lengthBoundBox/1.086, widthBoundBox /2) << QPointF(lengthBoundBox/1.02, widthBoundBox /3.33) << QPointF(lengthBoundBox,0);
				path.addPolygon(polygon);
				path.closeSubpath();
				path.addEllipse(QPointF(lengthBoundBox/5,-widthBoundBox /40), lengthBoundBox/12.5,lengthBoundBox/12.5);
				path.addEllipse(QPointF(lengthBoundBox/1.25,-widthBoundBox /40), lengthBoundBox/12.5,lengthBoundBox/12.5);

				
				break;
			}
		case oscVehicle::truck:
			{
				QPolygonF polygon;
				polygon << QPointF(0,0) << QPointF(0, heightBoundBox) << QPointF(lengthBoundBox/1.88, heightBoundBox) << QPointF(lengthBoundBox/1.88, heightBoundBox/1.5) << QPointF(lengthBoundBox/1.58, heightBoundBox/1.5) << QPointF(lengthBoundBox/1.3, heightBoundBox/3) << QPointF(lengthBoundBox/1.1,heightBoundBox/3) << QPointF(lengthBoundBox/1.08,heightBoundBox/5) << QPointF(lengthBoundBox,0);
				path.addPolygon(polygon);
				path.closeSubpath();
				path.addEllipse(QPointF(lengthBoundBox/5,-heightBoundBox/40), lengthBoundBox/12.5,lengthBoundBox/12.5);
				path.addEllipse(QPointF(lengthBoundBox/1.25,-heightBoundBox/40), lengthBoundBox/12.5,lengthBoundBox/12.5);

				break;
			}
		case oscVehicle::bus:
			{
				QPolygonF polygon;
				polygon << QPointF(0,0) << QPointF(0, heightBoundBox) << QPointF(lengthBoundBox, heightBoundBox) << QPointF(lengthBoundBox, 0);
				path.addPolygon(polygon);
				path.closeSubpath();
				path.addEllipse(QPointF(lengthBoundBox/5,-heightBoundBox/40), lengthBoundBox/12.5,lengthBoundBox/12.5);
				path.addEllipse(QPointF(lengthBoundBox/1.25,-heightBoundBox/40), lengthBoundBox/12.5,lengthBoundBox/12.5);

				break;
			}
		default:
			{
				path.addRect( 0,0 , lengthBoundBox, heightBoundBox);
			}

	//	trailer,
	//	motorbike,
	//	bicycle,
//		train,
	//	tram,
		}
	}
	path.translate(-lengthBoundBox/2, -heightBoundBox/2);
	return path;

}*/

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

	std::string catalogName = catalogReference->catalogName.getValue();
	std::string entryName = catalogReference->entryName.getValue();


	roadSystem_ = getProjectGraph()->getProjectData()->getRoadSystem();

	odrID roadID(atoi(oscRoad_->roadId.getValue().c_str()), 0, "", odrID::ID_Road);
	road_ = roadSystem_->getRoad(roadID);
	closestRoad_ = road_;
	roadSystemItem_ = oscBaseItem_->getRoadSystemItem();
	s_ = oscRoad_->s.getValue();
	t_ = oscRoad_->t.getValue();

	doPan_ = false;
	copyPan_ = false;
	lastPos_ = pos_;



	// TODO: get type and object from catalog reference //
	//

	OpenScenario::oscObjectBase *catalogObject = catalog_->getCatalogObject(catalogName,entryName);

/*	OpenScenario::oscMember *categoryMember = catalogObject->getMember("category");
	if (categoryMember)
	{
		OpenScenario::oscEnum *oscVar = dynamic_cast<OpenScenario::oscEnum *>(categoryMember);
		oscIntValue *iv = dynamic_cast<oscIntValue *>(categoryMember->getValue()); 
		qDebug() << "Category: " << oscVar->getValueAsStr(iv->getValue()).c_str();
	} */


#ifdef WIN32
	char *pValue;
	size_t len;
	errno_t err = _dupenv_s(&pValue, &len, "ODDLOTDIR");
	if (err || pValue == NULL || strlen(pValue) == 0)
		err = _dupenv_s(&pValue, &len, "COVISEDIR");
	if (err)
		return;
	covisedir_ = pValue;
#else
	covisedir_ = getenv("ODDLOTDIR");
	if (covisedir_ == "")
		covisedir_ = getenv("COVISEDIR");
#endif
	std::string fn = covisedir_+"/share/covise/icons/svg/car.svg";
	svgItem_ = new SVGItem(this, fn);


	if (catalogObject)
	{

		if (catalog_->getCatalogName() == "Vehicle")
		{
			createPath = NULL;
			updateColor(catalog_->getCatalogName());
			//path_ = createPath(catalogObject, road_);
			pos_ = road_->getGlobalPoint(s_, t_);
			lastPos_ = pos_;
			doPan_ = false;
			updatePosition();
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
		QPen pen;
		pen.setBrush(Qt::black);
		pen.setWidthF(0.2);
		setPen(pen);
	}
}


/*
* Update position
*/
void
OSCItem::updatePosition()
{
	QTransform transform;
//	double s = road_->getSFromGlobalPoint(pos_);
	double heading = road_->getGlobalHeading(s_);
	transform.rotate(road_->getGlobalHeading(s_));
	transform.translate(pos_.x(),pos_.y());
	QRectF rect = svgItem_->boundingRect();
//	qDebug() << rect.width() << " " << rect.height();
	transform.scale(0.1, 0.1);
	svgItem_->setTransform(transform);

/*	path_ = transform.map(path_);
	setPath(path_); */

	angle_ = heading;

	oscTextItem_->setPos(pos_);
}

/* Move
*/
void
OSCItem::move(QPointF &diff)
{
	path_.translate(diff); 
	setPath(path_);
	pos_ += diff;
	updatePosition();
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
	OpenScenario::oscArrayMember *arrayMember = dynamic_cast<OpenScenario::oscArrayMember *>(parent->getMember("Object"));

	if (arrayMember)
	{
		QUndoStack *undoStack = getProjectData()->getUndoStack();
		undoStack->beginMacro(QObject::tr("Delete Object"));

		RemoveOSCArrayMemberCommand *command = new RemoveOSCArrayMemberCommand(arrayMember, arrayMember->findObjectIndex(oscObject_), element_);
		getTopviewGraph()->executeCommand(command);

		OpenScenario::oscPrivateAction *oscPrivateAction = oscEditor_->getOrCreatePrivateAction(oscObject_->name.getValue());
		OpenScenario::oscPrivate *oscPrivate = static_cast<OpenScenario::oscPrivate *>(oscPrivateAction->getParentObj());
		arrayMember = dynamic_cast<OpenScenario::oscArrayMember *>(oscPrivate->getOwnMember());
		if (arrayMember)
		{
			RemoveOSCArrayMemberCommand *command = new RemoveOSCArrayMemberCommand(arrayMember, arrayMember->findObjectIndex(oscPrivate), NULL);
			getTopviewGraph()->executeCommand(command);
		}

		undoStack->endMacro();
		
	}

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
	closestRoad_ = road_;

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
			oscEditor_->cloneEntity(element_, oscObject_);
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
		QPointF diff = newPos - lastPos_;
		oscEditor_->move(diff);
		lastPos_ = newPos;

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
			QPointF diff = lastPos_ - pressPos_;
			oscEditor_->translate(diff);
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
		odrID roadID(atoi(oscRoad_->roadId.getValue().c_str()), 0, "",odrID::ID_Road);
		road_ = roadSystem_->getRoad(roadID);
		path_.translate(-pos_);
		s_ = oscRoad_->s.getValue();
		t_ = oscRoad_->t.getValue();
		updatePosition();
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
