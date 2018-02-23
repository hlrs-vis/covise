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

#include "oscshapeitem.hpp"
#include "osctextitem.hpp"


#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"
#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/oscsystem/oscelement.hpp"
#include "src/data/commands/osccommands.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphview.hpp"
#include "src/graph/items/oscsystem/oscbaseitem.hpp"
#include "src/graph/editors/osceditor.hpp"

// Tools //
//
#include "src/gui/tools/toolaction.hpp"
#include "src/gui/tools/zoomtool.hpp"

// OpenScenario //
//
#include <OpenScenario/schema/oscTrajectory.h>
#include <OpenScenario/schema/oscPosition.h>
#include <OpenScenario/schema/oscWorld.h>
#include <OpenScenario/oscVariables.h>

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QColor>
#include <QString>
#include <QKeyEvent>
#include <QVector>
#include <QPainterPath>

OSCShapeItem::OSCShapeItem(OSCElement *element, OSCBaseShapeItem *oscBaseShapeItem, OpenScenario::oscTrajectory *trajectory)
    : GraphElement(oscBaseShapeItem, element)
	, element_(element)
	, oscBaseShapeItem_(oscBaseShapeItem)
    , trajectory_(trajectory)
	, path_(NULL)
{

    init();
}

OSCShapeItem::~OSCShapeItem()
{
}

/*!
* Initializes the path 
*/
void
OSCShapeItem::createPath()
{

    if (path_)
    {
        delete path_;
        path_ = NULL;
    }

    if (controlPoints_.size() > 1)
    {

        path_ = new QPainterPath();

        int pointSize = 1.0;
        QPointF startPoint = controlPoints_.at(0);
        pos_ = QPointF(0, 0);
        if (!startPoint.isNull())
        {
            path_->addRect(QRectF(startPoint.x() - pointSize,
                startPoint.y() - pointSize,
                pointSize * 2, pointSize * 2));

            QPointF endPoint = controlPoints_.at(controlPoints_.size() - 1);
            if (!endPoint.isNull())
            {
                path_->addRect(QRectF(endPoint.x() - pointSize,
                    endPoint.y() - pointSize,
                    pointSize * 2, pointSize * 2));
            }
        }



        QPen penCubic(QBrush(Qt::blue), 0.2);
        setPen(penCubic);

        for (int i = 0; i < controlPoints_.size() - 1; i += 3) {
            QPainterPath pathCubic;
            QPointF p0;

            p0 = controlPoints_.at(i);

            pathCubic.moveTo(p0);

            QPointF p1 = controlPoints_.at(i + 1);
            QPointF p2 = controlPoints_.at(i + 2);
            QPointF p3 = controlPoints_.at(i + 3);
            pathCubic.cubicTo(p1, p2, p3);
            path_->addPath(pathCubic);
        }

        //  view_->setSplineControlPoints(controlPoints_);
    }

}

void
OSCShapeItem::init()
{
	oscBaseShapeItem_->appendOSCShapeItem(this);
	
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

    QAction *removeElementAction = getRemoveMenu()->addAction(tr("OpenScenario Spline"));
    connect(removeElementAction, SIGNAL(triggered()), this, SLOT(removeElement()));

    if (getTopviewGraph()) // not for profile graph
    {
        // Text //
        //
        QString name = updateName();
        oscTextItem_ = new OSCTextItem(element_, this, name, pos_);
        oscTextItem_->setZValue(1.0); // stack before siblings
    }


    doPan_ = false;
    copyPan_ = false;

    createControlPoints();
	if (element_->isElementSelected())
	{
		setSelected(true);
		getTopviewGraph()->getView()->setSplineControlPoints(controlPoints_);
	}
    createPath();
    updatePosition();
}

void
OSCShapeItem::createControlPoints()
{
    controlPoints_.clear();

    OpenScenario::oscArrayMember *vertexArray = dynamic_cast<OpenScenario::oscArrayMember *>(trajectory_->getMember("Vertex"));
    for (int i = 0; i < vertexArray->size(); i++)
    {
        OpenScenario::oscVertex *vertex = dynamic_cast<OpenScenario::oscVertex *>(vertexArray->at(i));
        OpenScenario::oscPosition *position = vertex->Position.getObject();
        if (!position)
        {
            continue;
        }
        OpenScenario::oscWorld *posWorld = position->World.getObject();
        if (!posWorld)
        {
            continue;
        }

        QPointF p0(posWorld->x.getValue(), posWorld->y.getValue());

        OpenScenario::oscShape *shape = vertex ->Shape.getObject();
        if (!shape)
        {
            continue;
        }

        OpenScenario::oscSpline *spline = shape->Spline.getObject();
        if (!spline)
        {
            continue;
        }

        OpenScenario::oscControlPoint1 *controlPoint1 = spline->ControlPoint1.getObject();
        if (controlPoint1)
        {
			std::string s = controlPoint1->status.getValue();
            double x, y;
            sscanf(s.c_str(), "%lf %lf", &x, &y);
            QPointF p1(p0.x() + x, p0.y() + y);

            controlPoints_.push_back(p1);
        }

        controlPoints_.push_back(p0);

        OpenScenario::oscControlPoint2 *controlPoint2 = spline->ControlPoint2.getObject();
        if (controlPoint2)
        {
            std::string s = controlPoint2->status.getValue();
            double x, y;
            sscanf(s.c_str(), "%lf %lf", &x, &y);
            QPointF p1(p0.x() + x, p0.y() + y);

            controlPoints_.push_back(p1);
        }

    }
}




/*
* Update position
*/
void
OSCShapeItem::updatePosition()
{
    if (path_)
    {
        path_->translate(pos_ );
        setPath(*path_);
    }
}

QString
OSCShapeItem::updateName()
{
    QString name = "Spline";
 /*   OpenScenario::oscMemberValue *value =  oscObject_->getMember("name")->getOrCreateValue();
    oscStringValue *sv = dynamic_cast<oscStringValue *>(value);
    if (sv)
    {
        name = QString::fromStdString(sv->getValue());
    } */

    return name;
}

//*************//
// Delete Item
//*************//

bool
OSCShapeItem::deleteRequest()
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
OSCShapeItem::removeElement()
{

	RemoveOSCObjectCommand *command = new RemoveOSCObjectCommand(element_);
	getTopviewGraph()->executeCommand(command); 

	return false;
}

//################//
// EVENTS         //
//################//

void
OSCShapeItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
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
OSCShapeItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
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
OSCShapeItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{

    // Parent //
    //
    GraphElement::hoverMoveEvent(event);
}

void
OSCShapeItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
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
OSCShapeItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{	
	if (doPan_)
	{

		QPointF newPos = event->scenePos();
		path_->translate(newPos-lastPos_);
		lastPos_ = newPos;
		setPath(*path_);

		GraphElement::mouseMoveEvent(event);
	}
}

void
OSCShapeItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{

    if (doPan_)
    {
		double diff = (lastPos_ - pressPos_).manhattanLength();
		if (diff > 0.01) // otherwise item has not been moved by intention
		{
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
OSCShapeItem::keyPressEvent(QKeyEvent *event)
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
OSCShapeItem::keyReleaseEvent(QKeyEvent *event)
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
OSCShapeItem::updateObserver()
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
        createControlPoints();
        createPath();
        updatePosition();

        // Text //
        //
        if (oscTextItem_)
        {
            oscTextItem_->updateText(updateName());
        }
    }
    else if (changes & OSCElement::COE_SettingChanged)
    {
         getTopviewGraph()->getView()->setSplineControlPoints(controlPoints_);
    }

    else
    {
        changes = element_->getDataElementChanges();

        if (changes & OSCElement::CDE_SelectionChange)
        {
            if (element_->isElementSelected())
            {
                getTopviewGraph()->getView()->setSplineControlPoints(controlPoints_);
            }
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
