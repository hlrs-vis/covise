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

#include "src/graph/items/roadsystem/signal/signaltextitem.hpp"
#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "src/graph/editors/signaleditor.hpp"

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

    updateColor();
    updatePosition();
    createPath();
}

/*! \brief Sets the color according to the number of links.
*/
void
SignalItem::updateColor()
{

    if (signal_->getCountry() == "Germany")
    {
        if (signal_->getType() < 201) // Gefahrzeichen
        {
            outerColor_.setRgb(255, 0, 0);
        }
        else if (signal_->getType() < 298) // Vorschriftzeichen
        {
            outerColor_.setRgb(0, 255, 0);
        }
        else if (signal_->getType() < 532) // Richtzeichen
        {
            outerColor_.setRgb(0, 0, 255);
        }
        else
        {
            outerColor_.setRgb(80, 80, 80);
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
        setBrush(QBrush(outerColor_));
        setPen(QPen(outerColor_));
        double size = 4.0;
        double length = 2.0;

        path.addEllipse(pos_, size, size);

        setPen(QPen(QColor(255, 255, 255)));
        path.moveTo(pos_.x() - length, pos_.y());
        path.lineTo(pos_.x() + length, pos_.y());

        path.moveTo(pos_.x(), pos_.y() - length);
        path.lineTo(pos_.x(), pos_.y() + length);
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

//################//
// EVENTS         //
//################//

void
SignalItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{

    // Text //
    //
    getSignalTextItem()->setVisible(true);
    getSignalTextItem()->setPos(event->scenePos());

    // Parent //
    //
    //GraphElement::hoverEnterEvent(event); // pass to baseclass
}

void
SignalItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{

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
    else
    {
        GraphElement::mousePressEvent(event); // pass to baseclass
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

    if ((changes & Signal::CEL_ParameterChange))
    {
        updatePosition();
    }
}
