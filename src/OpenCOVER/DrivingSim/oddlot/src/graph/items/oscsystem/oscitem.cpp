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
#include <QFile>


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
	road_ = roadSystem_->getRoad(QString::fromStdString(oscRoad_->roadId.getValue()));
	closestRoad_ = road_;
	roadSystemItem_ = oscBaseItem_->getRoadSystemItem();
	s_ = oscRoad_->s.getValue();
	t_ = oscRoad_->t.getValue();

	doPan_ = false;
	copyPan_ = false;
	lastPos_ = pos_;

	std::string categoryName;
	OpenScenario::oscObjectBase *catalogObject = catalog_->getCatalogObject(catalogName,entryName);
	if (!catalogObject)
	{
		return;		//nothing to draw
	}

	OpenScenario::oscMember *categoryMember = catalogObject->getMember("category");

	if (categoryMember) 
	{
		OpenScenario::oscEnum *categoryEnums = dynamic_cast<OpenScenario::oscEnum*>(categoryMember);
		OpenScenario::oscIntValue *categoryValue = dynamic_cast<OpenScenario::oscIntValue*>(categoryMember->getValue());
		categoryName = categoryEnums->getValueAsStr(categoryValue->getValue());
	}

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

	updateIcon(catalogObject, catalogName, categoryName, entryName);

	if (catalogObject)
	{
			createPath = NULL;
			updateColor(catalog_->getCatalogName());
			pos_ = road_->getGlobalPoint(s_, t_);
			lastPos_ = pos_;
			doPan_ = false;
			updatePosition();
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
*	update icon
*/
void OSCItem::updateIcon(OpenScenario::oscObjectBase *catalogObject, std::string catalogName, std::string categoryName, std::string entryName)
{
	QFile file;
	QString dir = ":/svgIcons/";
	std::string fn;

	QString qCatalogName = QString::fromStdString(catalogName);
	QString qCategoryName = QString::fromStdString(categoryName);
	QString qEntryName = QString::fromStdString(entryName);

	OpenScenario::oscMember *boundingBoxMember = catalogObject->getMember("BoundingBox");

	if (boundingBoxMember)
	{
		oscBoundingBox *BoundingBox = dynamic_cast<OpenScenario::oscBoundingBox*>(boundingBoxMember->getObjectBase());

		if (BoundingBox)
		{
			double widthBoundBox = BoundingBox->Dimension->width;
			double lengthBoundBox = BoundingBox->Dimension->length;
			double heightBoundBox = BoundingBox->Dimension->height;
			if (widthBoundBox < 1.0)
				widthBoundBox = 1.8;
			if (lengthBoundBox < 1.0)
				lengthBoundBox = 3;
			if (heightBoundBox < 1.0)
				heightBoundBox = 1.5;

			if (file.exists(dir + qCatalogName + "_" + qCategoryName + "_" + qEntryName + ".svg"))
			{
				fn = ":/svgIcons/" + catalogName + "_" + categoryName + "_" + entryName + ".svg";
			}
			else if (file.exists(dir + qCatalogName + "_" + qCategoryName + ".svg"))
			{
				fn = ":/svgIcons/" + catalogName + "_" + categoryName + ".svg";
			}
			else
			{
				fn = ":/svgIcons/" + catalogName + ".svg";
			}

			svgItem_ = new SVGItem(this, fn);

			QRectF svgRect = svgItem_->boundingRect();
			iconScaleX_ = (1 / svgRect.width()) * lengthBoundBox;
			iconScaleY_ = (1 / svgRect.height()) * heightBoundBox;
			svgCenter_ = QPointF(lengthBoundBox / 2, heightBoundBox / 2);
		}
		else
		{
			fn = ":/svgIcons/default.svg";
			svgItem_ = new SVGItem(this, fn);

			QRectF svgRect = svgItem_->boundingRect();
			iconScaleX_ = (1 / svgRect.width()) * 2;
			iconScaleY_ = (1 / svgRect.height()) * 2;
		}
	}
}

/*
* Update position
*/
void
OSCItem::updatePosition()
{
	QTransform tR;
	QTransform tS;
	QTransform tT;
	QTransform tM;
	QTransform tR2;

	double s = road_->getSFromGlobalPoint(pos_);
	QVector2D vec = QVector2D(pos_ - road_->getGlobalPoint(s));
	double t = vec.length();
	QVector2D normal = road_->getGlobalNormal(s);
	double heading = road_->getGlobalHeading(s);

	if (QVector2D::dotProduct(normal, vec) > 0)
	{
		t = -t;
	}

	if (t + svgCenter_.y() > 0)
	{
		heading += 180;
	}

	tM.translate(-svgCenter_.x(), -svgCenter_.y());
	tR2.rotate(heading);

	tR = tM * tR2 * tM.inverted();
	tT.translate(pos_.x(), pos_.y());
	tS.scale(iconScaleX_, iconScaleY_);

	svgItem_->setPos(QPointF(0, 0));
	svgItem_->setTransform(tS*tR*tT);

	angle_ = heading;

	oscTextItem_->setPos(pos_);
}

/* Move
*/
void
OSCItem::move(QPointF &diff)
{
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
    mousePressPos_ = mouseLastPos_ = event->scenePos();
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
		//	oscObject * oscObjectClone = oscObject_->getClone();
		/*	Signal * newSignal = signal_->getClone();
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
		QPointF mouseNewPos = event->scenePos();
		QPointF diff = mouseNewPos - mouseLastPos_;
		oscEditor_->move(diff);
		mouseLastPos_ = mouseNewPos;

		QVector2D vec;

		RSystemElementRoad * nearestRoad = roadSystem_->findClosestRoad( mouseNewPos, s_, t_, vec);
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
		double diff = (mouseLastPos_ - mousePressPos_).manhattanLength();
		if (diff > 0.01) // otherwise item has not been moved by intention
		{
			QPointF diff = mouseLastPos_ - mousePressPos_;
			oscEditor_->translate(diff);
		}
		else
		{
			//pos_ = mouseLastPos_;
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

		road_ = roadSystem_->getRoad(QString::fromStdString(oscRoad_->roadId.getValue()));
		s_ = oscRoad_->s.getValue();
		t_ = oscRoad_->t.getValue();
		pos_ = road_->getGlobalPoint(s_, t_);
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
