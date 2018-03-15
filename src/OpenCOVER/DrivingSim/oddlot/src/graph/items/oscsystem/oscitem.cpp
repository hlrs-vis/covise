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
#include "src/data/commands/dataelementcommands.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphview.hpp"
#include "src/graph/graphscene.hpp"
//#include "src/graph/items/roadsystem/signal/signaltextitem.hpp"
#include "src/graph/items/roadsystem/scenario/oscroadsystemitem.hpp"
#include "src/graph/items/oscsystem/oscbaseitem.hpp"
#include "src/graph/editors/osceditor.hpp"
#include "src/graph/items/svgelement.hpp"

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

#include "QtSvg/qsvgrenderer.h"


OSCItem::OSCItem(OSCElement *element, OSCBaseItem *oscBaseItem, OpenScenario::oscObject *oscObject, OpenScenario::oscCatalog *catalog, OpenScenario::oscRoad *oscRoad)
    : SVGElement(oscBaseItem, element)
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

} 

void
OSCItem::init()
{
	oscBaseItem_->appendOSCItem(this);
	
    // Hover Events //
    //
    setAcceptHoverEvents(true);
    setSelectable();
//	setFlag(QGraphicsItem::ItemIsSelectable);
	setFlag(ItemIsFocusable); 

    // OpenScenario Editor
    //
    oscEditor_ = dynamic_cast<OpenScenarioEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());


    if (getTopviewGraph()) // not for profile graph
    {
        // Text //
        //
		QString name = updateName();
        oscTextItem_ = new OSCTextSVGItem(element_, this, name, pos_);
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
//	odrID roadID(atoi(oscRoad_->roadId.getValue().c_str()),0," ",odrID::ID_Road);
	odrID roadID(QString::fromStdString(oscRoad_->roadId.getValue()));
	road_ = roadSystem_->getRoad(roadID);
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
				fn_ = ":/svgIcons/" + catalogName + "_" + categoryName + "_" + entryName + ".svg";
			}
			else if (file.exists(dir + qCatalogName + "_" + qCategoryName + ".svg"))
			{
				fn_ = ":/svgIcons/" + catalogName + "_" + categoryName + ".svg";
			}
			else
			{
				fn_ = ":/svgIcons/" + catalogName + ".svg";
			}

			renderer_ = new QSvgRenderer(QString::fromStdString(fn_));
			setSharedRenderer(renderer_);

			QRectF svgRect = boundingRect();
			iconScaleX_ = (1 / svgRect.width()) * lengthBoundBox;
			iconScaleY_ = (1 / svgRect.height()) * heightBoundBox;
			svgCenter_ = QPointF(lengthBoundBox / 2, heightBoundBox / 2);
		}
		else
		{
			fn_ = ":/svgIcons/default.svg";
			renderer_ = new QSvgRenderer(QString::fromStdString(fn_));
			setSharedRenderer(renderer_);

			QRectF svgRect = boundingRect();
			iconScaleX_ = (1 / svgRect.width()) * 2;
			iconScaleY_ = (1 / svgRect.height()) * 2;
		}
	}

	// Context Menu //
	//

	QAction *removeElementAction = getRemoveMenu()->addAction(tr("OpenScenario Object"));
	connect(removeElementAction, SIGNAL(triggered()), this, SLOT(removeElement()));
}

/*
* Update position
*/
void
OSCItem::updatePosition()
{
	/*QTransform tR;
	QTransform tS;
	QTransform tT;*/
	tR_.reset();
	tS_.reset();
	tT_.reset();

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

	tR_ = tM * tR2 * tM.inverted();
	tT_.translate(pos_.x(), pos_.y());
	tS_.scale(iconScaleX_, iconScaleY_);

	setPos(QPointF(0, 0));
	setTransform(tS_*tR_*tT_);

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
	SVGElement::hoverEnterEvent(event); // pass to baseclass
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
    SVGElement::hoverLeaveEvent(event); // pass to baseclass
}

void
OSCItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{

    // Parent //
    //
    SVGElement::hoverMoveEvent(event);
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
		if (event->modifiers() & Qt::ShiftModifier)
		{
			copyPan_ = true;

			cloneSvgItem_ = new QGraphicsSvgItem();
			cloneSvgItem_->setSharedRenderer(renderer_);
			cloneSvgItem_->setPos(QPointF(0, 0));
			cloneSvgItem_->setTransform(tS_*tR_*tT_); 
			getTopviewGraph()->getScene()->addItem(cloneSvgItem_);
		}

        SVGElement::mousePressEvent(event); // pass to baseclass

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

		SVGElement::mouseMoveEvent(event);
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
			QPointF diffPoint = mouseLastPos_ - mousePressPos_;
			if (copyPan_)
			{
				cloneElement_ = oscEditor_->cloneEntity(element_, oscObject_);
				pos_ -= diffPoint;
				updatePosition();

				OpenScenario::oscObject *cloneObject = dynamic_cast<OpenScenario::oscObject *>(cloneElement_->getObject());

				oscEditor_->translateObject(cloneObject, diffPoint);

				DeselectDataElementCommand *command = new DeselectDataElementCommand(element_);
				getProjectGraph()->executeCommand(command); 
				SelectDataElementCommand *selectCommand = new SelectDataElementCommand(cloneElement_);
				getProjectGraph()->executeCommand(selectCommand);
				
				getTopviewGraph()->getScene()->removeItem(cloneSvgItem_);
				delete cloneSvgItem_;
				copyPan_ = false;
			}
			else
			{
				oscEditor_->translate(diffPoint);
			}
		}
		else
		{
			//pos_ = mouseLastPos_;
		}

		doPan_ = false;
    }

	oscTextItem_->setVisible(true);

	SVGElement::mouseReleaseEvent(event);
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
    SVGElement::updateObserver();
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

//		odrID roadID(atoi(oscRoad_->roadId.getValue().c_str()), 0, " ", odrID::ID_Road);
		odrID roadID(QString::fromStdString(oscRoad_->roadId.getValue()));
		road_ = roadSystem_->getRoad(roadID);
		s_ = oscRoad_->s.getValue();
		t_ = oscRoad_->t.getValue();
		pos_ = road_->getGlobalPoint(s_, t_);
		updatePosition();
	}

	int dataChanges = element_->getDataElementChanges();
	if (dataChanges & DataElement::CDE_SelectionChange)
	{
		// Selection //
		//
		if (isSelected() != element_->isElementSelected())
		{
			setSelected(element_->isElementSelected());
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
