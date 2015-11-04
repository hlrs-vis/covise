/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   15.03.2010
**
**************************************************************************/

#include "signaleditor.hpp"

// Project //
//
#include "src/gui/projectwidget.hpp"

//MainWindow //
//
#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/signalmanager.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementcontroller.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"

// Commands //
//
#include "src/data/commands/controllercommands.hpp"
#include "src/data/commands/signalcommands.hpp"
#include "src/data/commands/roadsectioncommands.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/graphview.hpp"

#include "src/graph/items/roadsystem/signal/signalroadsystemitem.hpp"
#include "src/graph/items/roadsystem/signal/signalroaditem.hpp"
#include "src/graph/items/roadsystem/signal/signalitem.hpp"
#include "src/graph/items/roadsystem/signal/objectitem.hpp"
#include "src/graph/items/roadsystem/signal/bridgeitem.hpp"

#include "src/graph/items/roadsystem/signal/signalhandle.hpp"

// Tools //
//
#include "src/gui/tools/signaleditortool.hpp"
#include "src/gui/mouseaction.hpp"

// Tree //
//
#include "src/tree/signaltreewidget.hpp"

// Visitor //
//
//#include "src/graph/visitors/roadmarkvisitor.hpp"

// Qt //
//
#include <QGraphicsItem>
#include <QUndoStack>

//################//
// CONSTRUCTORS   //
//################//

SignalEditor::SignalEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph)
    : ProjectEditor(projectWidget, projectData, topviewGraph)
    , signalRoadSystemItem_(NULL)
    , insertSignalHandle_(NULL)
    , lastSelectedSignalItem_(NULL)
    , lastSelectedObjectItem_(NULL)
    , lastSelectedBridgeItem_(NULL)
{
	MainWindow * mainWindow = projectWidget->getMainWindow();
	signalTree_ = mainWindow->getSignalTree();
	signalManager_ = mainWindow->getSignalManager();
}

SignalEditor::~SignalEditor()
{
    kill();
}

//################//
// FUNCTIONS      //
//################//

/**
*
*/
void
SignalEditor::init()
{
    if (!signalRoadSystemItem_)
    {
        // Root item //
        //
        signalRoadSystemItem_ = new SignalRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
        getTopviewGraph()->getScene()->addItem(signalRoadSystemItem_);

        // Signal Handle //
        //
        insertSignalHandle_ = new SignalHandle(signalRoadSystemItem_);
        insertSignalHandle_->hide();
    }

    lastTool_ = getCurrentTool();

	// Raise Signal Tree //
	//
	signalTree_->setSignalEditor(this);
}

/*!
*/
void
SignalEditor::kill()
{
    delete signalRoadSystemItem_;
    signalRoadSystemItem_ = NULL;
	signalTree_->setSignalEditor(NULL);
}

SignalHandle *
SignalEditor::getInsertSignalHandle() const
{
    if (!insertSignalHandle_)
    {
        qDebug("ERROR 1003281422! SignalEditor not yet initialized.");
    }
    return insertSignalHandle_;
}

// Move Signal //
//
RSystemElementRoad *
SignalEditor::findClosestRoad(const QPointF &to, double &s, double &t, QVector2D &vec)
{
	RoadSystem * roadSystem = getProjectData()->getRoadSystem();
	QMap<QString, RSystemElementRoad *>::const_iterator it = roadSystem->getRoads().constBegin();
	RSystemElementRoad *road = it.value();
	s = road->getSFromGlobalPoint(to, 0.0, road->getLength());
	vec = QVector2D(road->getGlobalPoint(s) - to);
	t = vec.length();

	while (++it != roadSystem->getRoads().constEnd())
	{
		RSystemElementRoad *newRoad = it.value();
		double newS = newRoad->getSFromGlobalPoint(to, 0.0, newRoad->getLength());
		QVector2D newVec = QVector2D(newRoad->getGlobalPoint(s) - to);
		double dist = newVec.length();

		if (dist < t)
		{
			road = newRoad;
			t = dist;
			s = newS;
			vec = newVec;
		}
	}

	QVector2D normal = road->getGlobalNormal(s);

	if (QVector2D::dotProduct(normal, vec) < 0)
	{
		t = -t;
	}

	return road;
}

bool 
SignalEditor::translateSignal(Signal * signal, RSystemElementRoad *newRoad, QPointF &to)
{
    RSystemElementRoad * road = signal->getParentRoad();

    getProjectData()->getUndoStack()->beginMacro(QObject::tr("Move Signal"));
    bool parentChanged = false;
    if (newRoad != road)
    {
        RemoveSignalCommand * removeSignalCommand = new RemoveSignalCommand(signal, road);
        getProjectGraph()->executeCommand(removeSignalCommand);

        AddSignalCommand * addSignalCommand = new AddSignalCommand(signal, newRoad);
        getProjectGraph()->executeCommand(addSignalCommand);
		signal->setElementSelected(false);

        road = newRoad;
        parentChanged = true;
    }  

	double s = road->getSFromGlobalPoint(to, 0.0, road->getLength());
	QVector2D vec = QVector2D(road->getGlobalPoint(s) - to);
	double t = vec.length();

	QVector2D normal = road->getGlobalNormal(s);

	if (QVector2D::dotProduct(normal, vec) < 0)
	{
		t = -t;
	}

	LaneSection *laneSection = road->getLaneSection(s);
    int validToLane;
    if (t < 0)
    {
        validToLane = laneSection->getRightmostLaneId();
	}
    else
    {
        validToLane = laneSection->getLeftmostLaneId();
    }

    SetSignalPropertiesCommand * signalPropertiesCommand = new SetSignalPropertiesCommand(signal, signal->getId(), signal->getName(), t, signal->getDynamic(), signal->getOrientation(), signal->getValue(), signal->getCountry(), signal->getType(), signal->getTypeSubclass(), signal->getSubtype(), signal->getValue(), signal->getZOffset(), signal->getPitch(), signal->getRoll(), signal->getPole(), signal->getSize(), 0, validToLane);
    getProjectGraph()->executeCommand(signalPropertiesCommand);
    MoveRoadSectionCommand * moveSectionCommand = new MoveRoadSectionCommand(signal, s, RSystemElementRoad::DRS_SignalSection);
    getProjectGraph()->executeCommand(moveSectionCommand);

    getProjectData()->getUndoStack()->endMacro();

    return parentChanged;
}

Signal *
SignalEditor::addSignalToRoad(RSystemElementRoad *road, double s, double t)
{
	int validToLane = 0;	// make a new signal //

	LaneSection *laneSection = road->getLaneSection(s);
	if (t < 0)
	{
		validToLane = laneSection->getRightmostLaneId();
	}
	else
	{
		validToLane = laneSection->getLeftmostLaneId();
	}
	QList<UserData *> userData;
	SignalContainer *lastSignal = signalManager_->getSelectedSignalContainer();
	Signal *newSignal = NULL;

	if (lastSignal)
	{
		if (t < 0)
		{
			t -= lastSignal->getSignalDistance();
		}
		else
		{
			t += lastSignal->getSignalDistance();
		}
		newSignal = new Signal("signal", "", s, t, false, Signal::NEGATIVE_TRACK_DIRECTION, 0.0, signalManager_->getCountry(lastSignal), lastSignal->getSignalType(), lastSignal->getSignalTypeSubclass(), lastSignal->getSignalSubType(), lastSignal->getSignalValue(), lastSignal->getSignalHeight(), 0.0, 0.0, true, 2, 0, validToLane);
		AddSignalCommand *command = new AddSignalCommand(newSignal, road, NULL);
		getProjectGraph()->executeCommand(command);
	}
	else
	{
		newSignal = new Signal("signal", "", s, t, false, Signal::NEGATIVE_TRACK_DIRECTION, 0.0, "Germany", -1, "", -1, 0.0, 0.0, 0.0, 0.0, true, 2, 0, validToLane);
		AddSignalCommand *command = new AddSignalCommand(newSignal, road, NULL);
		getProjectGraph()->executeCommand(command);
	}

	return newSignal;
}

/*
void
	SignalEditor
	::setCurrentRoadType(TypeSection::RoadType roadType)
{
	currentRoadType_ = roadType;
}*/

//################//
// MOUSE & KEY    //
//################//

/*! \brief .
*
*/
void
SignalEditor::mouseAction(MouseAction *mouseAction)
{

    QGraphicsSceneMouseEvent *mouseEvent = mouseAction->getEvent();
    ProjectEditor::mouseAction(mouseAction);

    // SELECT //
    //

    if (getCurrentTool() == ODD::TSG_SELECT)
    {
        QPointF mousePoint = mouseAction->getEvent()->scenePos();

        if (mouseAction->getMouseActionType() == MouseAction::ATM_PRESS)
        {
            if (mouseEvent->button() == Qt::LeftButton)
            {
                if (lastSelectedSignalItem_ && obscuredSignalItems_.empty())
                {
                    QList<QGraphicsItem *> underMouseItems = getTopviewGraph()->getScene()->items(mousePoint);
                    foreach (QGraphicsItem *item, underMouseItems)
                    {
                        SignalItem *signalItem = dynamic_cast<SignalItem *>(item);
                        if (signalItem)
                        {
                            signalItem->setZValue(signalItem->zValue() + 10.0); // Initialize zValues to ensure that the signals are not obscured by the road
                            obscuredSignalItems_.insert(signalItem->zValue(), signalItem);
                        }
                    }
                }
                else if (lastSelectedObjectItem_ && obscuredObjectItems_.empty())
                {
                    QList<QGraphicsItem *> underMouseItems = getTopviewGraph()->getScene()->items(mousePoint);
                    foreach (QGraphicsItem *item, underMouseItems)
                    {
                        ObjectItem *objectItem = dynamic_cast<ObjectItem *>(item);
                        if (objectItem)
                        {
                            objectItem->setZValue(objectItem->zValue() + 10.0); // Initialize zValues to ensure that the signals are not obscured by the road
                            obscuredObjectItems_.insert(objectItem->zValue(), objectItem);
                        }
                    }
                }
                else if (lastSelectedBridgeItem_ && obscuredBridgeItems_.empty())
                {
                    QList<QGraphicsItem *> underMouseItems = getTopviewGraph()->getScene()->items(mousePoint);
                    foreach (QGraphicsItem *item, underMouseItems)
                    {
                        BridgeItem *bridgeItem = dynamic_cast<BridgeItem *>(item);
                        if (bridgeItem)
                        {
                            bridgeItem->setZValue(bridgeItem->zValue() + 10.0); // Initialize zValues to ensure that the signals are not obscured by the road
                            obscuredBridgeItems_.insert(bridgeItem->zValue(), bridgeItem);
                        }
                    }
                }
                else
                {
                    QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();
                    int signalCount = 0;
                    int objectCount = 0;
                    int bridgeCount = 0;
                    SignalItem *signalItem = NULL;
                    ObjectItem *objectItem = NULL;
                    BridgeItem *bridgeItem = NULL;
                    foreach (QGraphicsItem *item, selectedItems)
                    {
                        signalItem = dynamic_cast<SignalItem *>(item);
                        if (signalItem)
                        {
                            signalCount++;
                        }
                        else
                        {
                            objectItem = dynamic_cast<ObjectItem *>(item);
                            if (objectItem)
                            {
                                objectCount++;
                            }
                            else
                            {
                                bridgeItem = dynamic_cast<BridgeItem *>(item);
                                if (bridgeItem)
                                {
                                    bridgeCount++;
                                }
                            }
                        }
                    }
                    if (signalItem && (signalCount == 1))
                    {
                        lastSelectedSignalItem_ = signalItem;
                    }
                    else if (objectItem && (objectCount == 1))
                    {
                        lastSelectedObjectItem_ = objectItem;
                    }
                    else if (bridgeItem && (bridgeCount == 1))
                    {
                        lastSelectedBridgeItem_ = bridgeItem;
                    }
                }

                if (obscuredSignalItems_.count() > 1)
                {
                    obscuredSignalItems_.remove(lastSelectedSignalItem_->zValue(), lastSelectedSignalItem_);
                    lastSelectedSignalItem_->setZValue(obscuredSignalItems_.constBegin().key() - 0.1);
                    obscuredSignalItems_.insert(lastSelectedSignalItem_->zValue(), lastSelectedSignalItem_);

                    lastSelectedSignalItem_ = obscuredSignalItems_.constEnd()--.value();
                }
                else if (obscuredObjectItems_.count() > 1)
                {
                    obscuredObjectItems_.remove(lastSelectedObjectItem_->zValue(), lastSelectedObjectItem_);
                    lastSelectedObjectItem_->setZValue(obscuredObjectItems_.constBegin().key() - 0.1);
                    obscuredObjectItems_.insert(lastSelectedObjectItem_->zValue(), lastSelectedObjectItem_);

                    lastSelectedObjectItem_ = obscuredObjectItems_.constEnd()--.value();
                }
                else if (obscuredBridgeItems_.count() > 1)
                {
                    obscuredBridgeItems_.remove(lastSelectedBridgeItem_->zValue(), lastSelectedBridgeItem_);
                    lastSelectedBridgeItem_->setZValue(obscuredBridgeItems_.constBegin().key() - 0.1);
                    obscuredBridgeItems_.insert(lastSelectedBridgeItem_->zValue(), lastSelectedBridgeItem_);

                    lastSelectedBridgeItem_ = obscuredBridgeItems_.constEnd()--.value();
                }
            }
        }

        else if (mouseAction->getMouseActionType() == MouseAction::ATM_MOVE)
        {
            if (lastSelectedSignalItem_)
            {
                lastSelectedSignalItem_ = NULL;
                obscuredSignalItems_.clear();
            }
            else if (lastSelectedObjectItem_)
            {
                lastSelectedObjectItem_ = NULL;
                obscuredObjectItems_.clear();
            }
            else if (lastSelectedBridgeItem_)
            {
                lastSelectedBridgeItem_ = NULL;
                obscuredBridgeItems_.clear();
            }
        }

        else if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
        {
        }
    }
	else if (getCurrentTool() == ODD::TSG_SIGNAL)
	{
		QPointF mousePoint = mouseAction->getEvent()->scenePos();

        if (mouseAction->getMouseActionType() == MouseAction::ATM_PRESS)
		{
			if (mouseEvent->button() == Qt::LeftButton)
			{
				QList<QGraphicsItem *> underMouseItems = getTopviewGraph()->getScene()->items(mousePoint);

				if (underMouseItems.count() == 0)		// find the closest road //
				{
					double s;
					double t;
					QVector2D vec;
					RSystemElementRoad * road = findClosestRoad(mousePoint, s, t, vec);

					addSignalToRoad(road, s, t);    
				}
			}
		}
	}

    //	ProjectEditor::mouseAction(mouseAction);
}

//################//
// TOOL           //
//################//

/*! \brief .
*
*/
void
SignalEditor::toolAction(ToolAction *toolAction)
{
    // Parent //
    //
    ProjectEditor::toolAction(toolAction);

    ODD::ToolId currentTool = getCurrentTool();
    if (currentTool != lastTool_)
    {
        if (currentTool == ODD::TSG_SELECT)
        {
            if ((lastTool_ = ODD::TSG_ADD_CONTROL_ENTRY) || (lastTool_ = ODD::TSG_REMOVE_CONTROL_ENTRY))
            {
                foreach (QGraphicsItem *item, getTopviewGraph()->getScene()->items())
                {
                    SignalItem * signalItem = dynamic_cast<SignalItem *>(item);
                    if (signalItem)
                    {
                        signalItem->setFlag(QGraphicsItem::ItemIsSelectable, true);
                    }
                }
            }
        }
        else if ((currentTool == ODD::TSG_SIGNAL) || (currentTool == ODD::TSG_OBJECT) 
            || (currentTool == ODD::TSG_BRIDGE))
        {
            foreach (QGraphicsItem *item, getTopviewGraph()->getScene()->selectedItems())
            {
                item->setSelected(false);
            }
            // does nothing //
            // Problem: The ToolAction is resent, after a warning message has been clicked away. (Due to resend on getting the
        }
        else if (currentTool == ODD::TSG_CONTROLLER)
        {
            QList<ControlEntry *>controlEntryList;
            RSystemElementController *newController = new RSystemElementController("unnamed", "", 0,"", 0.0, controlEntryList);
            AddControllerCommand *command = new AddControllerCommand(newController, getProjectData()->getRoadSystem(), NULL);

            getProjectGraph()->executeCommand(command);
        }
        else if ((currentTool == ODD::TSG_ADD_CONTROL_ENTRY) || (currentTool == ODD::TSG_REMOVE_CONTROL_ENTRY))
        {

            foreach (QGraphicsItem *item, getTopviewGraph()->getScene()->items())
            {
                SignalItem * signalItem = dynamic_cast<SignalItem *>(item);
                if (signalItem)
                {
                    signalItem->setFlag(QGraphicsItem::ItemIsSelectable, false);
                }
            }
        }
        else if (currentTool == ODD::TSG_DEL)
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
                    getProjectData()->getUndoStack()->beginMacro(QObject::tr("Set Road Type"));
                }
                for (int i = 0; i < selectedControllers.size(); i++)
                {
                    RemoveControllerCommand * delControllerCommand = new RemoveControllerCommand(selectedControllers.at(i), selectedControllers.at(i)->getRoadSystem());
                    getProjectGraph()->executeCommand(delControllerCommand);
                }

                // Macro Command //
                //
                if (numberOfSelectedControllers > 1)
                {
                    getProjectData()->getUndoStack()->endMacro();
                }
            }
        }

        lastTool_ = currentTool;

    }
    // RoadType //
    //
    /*	TypeEditorToolAction * typeEditorToolAction = dynamic_cast<TypeEditorToolAction *>(toolAction);
	if(typeEditorToolAction)
	{
		// Set RoadType //
		//
		TypeSection::RoadType roadType = typeEditorToolAction->getRoadType();
		if(roadType != TypeSection::RTP_NONE)
		{
			if(typeEditorToolAction->isApplyingRoadType())
			{
				QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();

				// Macro Command //
				//
				int numberOfSelectedItems = selectedItems.size();
				if(numberOfSelectedItems > 1)
				{
					getProjectData()->getUndoStack()->beginMacro(QObject::tr("Set Road Type"));
				}

				// Change types of selected items //
				//
				foreach(QGraphicsItem * item, selectedItems)
				{
					TypeSectionItem * typeSectionItem = dynamic_cast<TypeSectionItem *>(item);
					if(typeSectionItem)
					{
						typeSectionItem->changeRoadType(roadType);
					}
				}

				// Macro Command //
				//
				if(numberOfSelectedItems > 1)
				{
					getProjectData()->getUndoStack()->endMacro();
				}
			}
			else
			{
				setCurrentRoadType(roadType);
			}
		}
	}*/
}
