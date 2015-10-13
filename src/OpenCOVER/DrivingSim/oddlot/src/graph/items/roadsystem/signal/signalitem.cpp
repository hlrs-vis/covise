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

#include "signalitem.hpp"

#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"
#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/commands/signalcommands.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/rsystemelementcontroller.hpp"
#include "src/data/commands/controllercommands.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphview.hpp"
#include "src/graph/items/roadsystem/signal/signaltextitem.hpp"
#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "src/graph/editors/signaleditor.hpp"

// Manager //
//
#include "src/data/signalmanager.hpp" 

// Tools //
//
#include "src/gui/tools/toolaction.hpp"
#include "src/gui/tools/zoomtool.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QColor>
#include <QString>

SignalItem::SignalItem(RoadSystemItem *roadSystemItem, Signal *signal, QPointF pos)
    : GraphElement(roadSystemItem, signal)
    , signal_(signal)
    , pos_(pos)
    , pixmapItem_(NULL)
{
    init();
}

SignalItem::~SignalItem()
{
}

void
SignalItem::init()
{
    // Hover Events //
    //
    setAcceptHoverEvents(true);
    setSelectable();

    // Signal Editor
    //
    signalEditor_ = dynamic_cast<SignalEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());

	// Signal Manager
	//
	signalManager_ = getProjectData()->getProjectWidget()->getMainWindow()->getSignalManager();

	// Category Size
	//
	categorySize_ = signalManager_->getCategoriesSize();

    // Context Menu //
    //

    QAction *removeRoadAction = getRemoveMenu()->addAction(tr("Signal"));
    connect(removeRoadAction, SIGNAL(triggered()), this, SLOT(removeSignal()));

    if (getTopviewGraph()) // not for profile graph
    {
        // Text //
        //
        signalTextItem_ = new SignalTextItem(this);
        signalTextItem_->setZValue(1.0); // stack before siblings
    }

    // value for pixmap representation
    //
    lodThreshold_ = 5.0;
    size_ = 8.0;
    halfsize_ = size_ / 2.0;

    pos_ = signal_->getParentRoad()->getGlobalPoint(signal_->getSStart(), signal_->getT());
    updateCategory();
    updatePosition();
    createPath();
}


/*! \brief Sets the color according to the number of links.
*/
void
SignalItem::updateCategory()
{
    if ((signal_->getType() == 294) || (signal_->getType() == 293) || (signal_->getType() == 341))
    {
        if (pixmapItem_)
        {
            delete pixmapItem_;
            pixmapItem_ = NULL;
        }
    }
    else
    {
        SignalContainer *signalContainer = signalManager_->getSignalContainer(signal_->getType(), signal_->getTypeSubclass(), signal_->getSubtype());
        if (signalContainer)
        {
            QString category = signalContainer->getsignalCategory();
            int i = 360 / categorySize_;
            outerColor_.setHsv(signalManager_->getCategoryNumber(category) * i, 255, 255, 255);


            QIcon icon = signalContainer->getSignalIcon();
            pixmap_ = icon.pixmap(icon.availableSizes().first());
            if (pixmap_.isNull())
            {
                qDebug("ERROR 1006111429! Pixmap could not be loaded!");
            }
            else
            {
                // Pixmap //
                //
                if (pixmapItem_)
                {
                    delete pixmapItem_;
                }
                pixmapItem_ = new QGraphicsPixmapItem(pixmap_);
                pixmapItem_->setParentItem(this);

                // Transformation //
                //
                // Note: The y-Axis must be flipped so the image is not mirrored

                QTransform trafo;
                if (pixmap_.width() > pixmap_.height())
                {
                    scale_ = size_/pixmap_.width();
                    width_ = size_;
                    height_ = scale_ * pixmap_.height();
                    x_ = pos_.x() - halfsize_;
                    double h = height_ / 2.0;
                    trafo.translate(x_, pos_.y() + h);
                    y_ = pos_.y() - h;   // Pixmap and drawing coordinate system differ
                }
                else
                {
                    scale_ = size_/pixmap_.height();
                    width_ = scale_ * pixmap_.width();
                    height_ = size_;
                    x_ = pos_.x() - width_ / 2.0;
                    trafo.translate(x_, pos_.y() + halfsize_);
                    y_ = pos_.y() - halfsize_;
                }
                trafo.rotate(180, Qt::XAxis);
                trafo.scale(scale_, scale_);

                pixmapItem_->setTransform(trafo);
                if (!showPixmap_)
                {
                    pixmapItem_->hide();
                }
            }
        }
        else
        {
            showPixmap_ = false;
            outerColor_.setRgb(80, 80, 80);
        }
    }
}


/*! \brief Sets the color according to the number of links.
*/
void
SignalItem::updateColor()
{
    if (pixmapItem_)
    {
        double scaling = getTopviewGraph()->getView()->getScaling();
        if ((scaling < lodThreshold_) &&  showPixmap_)
        {
            showPixmap_ = false;
            pixmapItem_->hide();
        }
        else if ((scaling > lodThreshold_) && !showPixmap_)
        {
            showPixmap_ = true;
            pixmapItem_->show();
        }
    }
}

/*!
* Initializes the path (only once).
*/
void
SignalItem::createPath()
{
    QPainterPath path;

    // Stopp line
    //
    if (signal_->getType() == 294)
    {
        setPen(QPen(QColor(255, 255, 255), 2, Qt::SolidLine, Qt::FlatCap, Qt::RoundJoin));

        LaneSection *laneSection = signal_->getParentRoad()->getLaneSection(signal_->getSStart());

        if (signal_->getT() > 0)
        {
            double width = laneSection->getLaneSpanWidth(0, signal_->getValidFromLane() - 1, signal_->getSStart());
            path.moveTo(signal_->getParentRoad()->getGlobalPoint(signal_->getSStart(), width));
            width = laneSection->getLaneSpanWidth(0, signal_->getValidToLane(), signal_->getSStart());
            path.lineTo(signal_->getParentRoad()->getGlobalPoint(signal_->getSStart(), width));
        }
        else
        {
            double width = laneSection->getLaneSpanWidth(0, signal_->getValidFromLane() + 1, signal_->getSStart());
            path.moveTo(signal_->getParentRoad()->getGlobalPoint(signal_->getSStart(), -width));
            width = laneSection->getLaneSpanWidth(0, signal_->getValidToLane(), signal_->getSStart());
            path.lineTo(signal_->getParentRoad()->getGlobalPoint(signal_->getSStart(), -width));
        }
    }
    else if (signal_->getType() == 293)
    {
        setPen(QPen(QColor(255, 255, 255), 0.2, Qt::SolidLine, Qt::FlatCap, Qt::RoundJoin));

        LaneSection *laneSection = signal_->getParentRoad()->getLaneSection(signal_->getSStart());

        if (signal_->getValidFromLane() > 0)
        {
            double width = laneSection->getLaneSpanWidth(0, signal_->getValidFromLane(), signal_->getSStart());
            if (signal_->getValidToLane() >= 0)
            {
                while (width >= laneSection->getLaneSpanWidth(0, signal_->getValidToLane(), signal_->getSStart()))
                {
                    path.moveTo(signal_->getParentRoad()->getGlobalPoint(signal_->getSStart(), width));
                    path.lineTo(signal_->getParentRoad()->getGlobalPoint(signal_->getSStart() + signal_->getValue(), width));
                    width -= 1;
                }
            }
            else
            {
                while (width >= -laneSection->getLaneSpanWidth(0, signal_->getValidToLane(), signal_->getSStart()))
                {
                    path.moveTo(signal_->getParentRoad()->getGlobalPoint(signal_->getSStart(), width));
                    path.lineTo(signal_->getParentRoad()->getGlobalPoint(signal_->getSStart() + signal_->getValue(), width));
                    width -= 1;
                }
            }
        }
        else
        {
            double width = laneSection->getLaneSpanWidth(0, signal_->getValidFromLane(), signal_->getSStart());
            while (width <= laneSection->getLaneSpanWidth(0, signal_->getValidToLane(), signal_->getSStart()))
            {
                path.moveTo(signal_->getParentRoad()->getGlobalPoint(signal_->getSStart(), -width));
                path.lineTo(signal_->getParentRoad()->getGlobalPoint(signal_->getSStart() + signal_->getValue(), -width));
                width += 1;
            }
        }
    }
    else
    {
        if (pixmapItem_ && showPixmap_)
        {
            setBrush(QBrush(QColor(0,0,0,0)));
            setPen(QPen(QColor(0,0,0,0)));
            path.addRect(x_, y_, width_, height_);
        }
        else
        {
            double length = halfsize_/2.0;
            setBrush(QBrush(outerColor_));
            setPen(QPen(outerColor_));

            path.addEllipse(pos_, halfsize_, halfsize_);

            setPen(QPen(QColor(255, 255, 255)));
            path.moveTo(pos_.x() - length, pos_.y());
            path.lineTo(pos_.x() + length, pos_.y());

            path.moveTo(pos_.x(), pos_.y() - length);
            path.lineTo(pos_.x(), pos_.y() + length);
        }
    }

    setPath(path);
    
}

/*
* Update position
*/
void
SignalItem::updatePosition()
{

    pos_ = signal_->getParentRoad()->getGlobalPoint(signal_->getSStart(), signal_->getT());

    updateColor();
    createPath();
}

//*************//
// Delete Item
//*************//

bool
SignalItem::deleteRequest()
{
    if (removeSignal())
    {
        return true;
    }

    return false;
}

//################//
// SLOTS          //
//################//

bool
SignalItem::removeSignal()
{
    RemoveSignalCommand *command = new RemoveSignalCommand(signal_, signal_->getParentRoad());
    return getProjectGraph()->executeCommand(command);
}

/*! \brief .
*
*/
void
SignalItem::zoomAction()
{
    // Zoom //
    //

    updateColor();
    createPath();

}

//################//
// EVENTS         //
//################//

void
SignalItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    if (signalEditor_->getCurrentTool() == ODD::TSG_MOVE)
    {
        setCursor(Qt::OpenHandCursor);
    }
    else
    {
        // Text //
        //
        getSignalTextItem()->setVisible(true);
        getSignalTextItem()->setPos(event->scenePos());
    }

    // Parent //
    //
    //GraphElement::hoverEnterEvent(event); // pass to baseclass
}

void
SignalItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::ArrowCursor);

    // Text //
    //
    getSignalTextItem()->setVisible(false);

    // Parent //
    //
    //GraphElement::hoverLeaveEvent(event); // pass to baseclass
}

void
SignalItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{

    // Parent //
    //
    //GraphElement::hoverMoveEvent(event);
}

void
SignalItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    pressPos_ = event->pos();
    ODD::ToolId tool = signalEditor_->getCurrentTool(); // Editor Delete Signal
    if (tool == ODD::TSG_DEL)
    {
        removeSignal();
    }
    else if (tool == ODD::TSG_ADD_CONTROL_ENTRY)
    {
        QList<DataElement *>selectedElements = getProjectData()->getSelectedElements();
        QList<RSystemElementController *>selectedControllers;
        foreach (DataElement *element, selectedElements)
        {
            RSystemElementController * controller = dynamic_cast<RSystemElementController *>(element);
            if (controller)
            {
                selectedControllers.append(controller);
            }
        }

        if (selectedControllers.size() > 0)
        {
            // Macro Command //
            //
            int numberOfSelectedControllers = selectedControllers.size();
            if(numberOfSelectedControllers > 1)
            {
                getProjectData()->getUndoStack()->beginMacro(QObject::tr("Add Control Entry"));
            }
            for (int i = 0; i < selectedControllers.size(); i++)
            {

                ControlEntry * controlEntry = new ControlEntry(signal_->getId(), QString::number(signal_->getType()));
                AddControlEntryCommand *addControlEntryCommand = new AddControlEntryCommand(selectedControllers.at(i), controlEntry, signal_);
                getProjectGraph()->executeCommand(addControlEntryCommand);
            }

            // Macro Command //
            //
            if (numberOfSelectedControllers > 1)
            {
                getProjectData()->getUndoStack()->endMacro();
            }
        }
    }
    else if (tool == ODD::TSG_REMOVE_CONTROL_ENTRY)
    {
        QList<DataElement *>selectedElements = getProjectData()->getSelectedElements();
        QList<RSystemElementController *>selectedControllers;
        foreach (DataElement *element, selectedElements)
        {
            RSystemElementController * controller = dynamic_cast<RSystemElementController *>(element);
            if (controller)
            {
                selectedControllers.append(controller);
            }
        }

        if (selectedControllers.size() > 0)
        {
            // Macro Command //
            //
            int numberOfSelectedControllers = selectedControllers.size();
            if(numberOfSelectedControllers > 1)
            {
                getProjectData()->getUndoStack()->beginMacro(QObject::tr("Remove Control Entry"));
            }
            for (int i = 0; i < selectedControllers.size(); i++)
            {
                RSystemElementController * controller = selectedControllers.at(i);
                foreach (ControlEntry * controlEntry, controller->getControlEntries())
                {
                    if (controlEntry->getSignalId() == signal_->getId())
                    {
                        DelControlEntryCommand *delControlEntryCommand = new DelControlEntryCommand(selectedControllers.at(i), controlEntry, signal_);
                        getProjectGraph()->executeCommand(delControlEntryCommand);
                        break;
                    }
                }
            }

            // Macro Command //
            //
            if (numberOfSelectedControllers > 1)
            {
                getProjectData()->getUndoStack()->endMacro();
            }
        }
    }
    else if (tool == ODD::TSG_MOVE)
    {
        if (pixmapItem_ && showPixmap_)
        {
            pixmapItem_->hide();
            showPixmap_ = false;
            createPath();
        }
    }
    else
    {
        GraphElement::mousePressEvent(event); // pass to baseclass
    }
}

void
SignalItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = signalEditor_->getCurrentTool();
    if (tool == ODD::TSG_MOVE)
    {
        pos_ = event->scenePos();
        createPath();
    }
}

void
SignalItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = signalEditor_->getCurrentTool();
    if (tool == ODD::TSG_MOVE)
    {
        bool parentChanged = signalEditor_->translateSignal(signal_, event->scenePos());

        if (!parentChanged && pixmapItem_)
        {
            QTransform trafo;
            if (pixmap_.width() > pixmap_.height())
            {
                x_ = pos_.x() - halfsize_;
                double h = height_ / 2.0;
                trafo.translate(x_, pos_.y() + h);
                y_ = pos_.y() - h;   // Pixmap and drawing coordinate system differ
            }
            else
            {
                x_ = pos_.x() - width_ / 2.0;
                trafo.translate(x_, pos_.y() + halfsize_);
                y_ = pos_.y() - halfsize_;
            }
            trafo.rotate(180, Qt::XAxis);
            trafo.scale(scale_, scale_);
            pixmapItem_->setTransform(trafo);

            updateColor();
            createPath();
        }
    }
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
SignalItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Signal //
    //
    int changes = signal_->getSignalChanges();

    if ((changes & Signal::CEL_TypeChange))
    {
        updateCategory();
        updatePosition();
    }
    else if ((changes & Signal::CEL_ParameterChange))
    {
        updatePosition();
    }
}
