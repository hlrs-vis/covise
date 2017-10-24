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
	, lastTool_(ODD::TSG_NONE)
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

void
SignalEditor::duplicate()
{
	QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();

	foreach (QGraphicsItem *item, selectedItems)
	{
		SignalItem *signalItem = dynamic_cast<SignalItem *>(item);
		if (signalItem)
		{
			signalItem->duplicate();
		}
		else
		{
			ObjectItem *objectItem = dynamic_cast<ObjectItem *>(item);
			if (objectItem)
			{
				objectItem->duplicate();
			}
			else
			{
				BridgeItem *bridgeItem = dynamic_cast<BridgeItem *>(item);
				if (bridgeItem)
				{
					bridgeItem->duplicate();
				}
			}
		}
	}

}

void
SignalEditor::move(QPointF &diff)
{
	QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();

	foreach (QGraphicsItem *item, selectedItems)
	{
		SignalItem *signalItem = dynamic_cast<SignalItem *>(item);
		if (signalItem)
		{
			signalItem->move(diff);
		}
		else
		{
			ObjectItem *objectItem = dynamic_cast<ObjectItem *>(item);
			if (objectItem)
			{
				objectItem->move(diff);
			}
			else
			{
				BridgeItem *bridgeItem = dynamic_cast<BridgeItem *>(item);
				if (bridgeItem)
				{
					bridgeItem->move(diff);
				}
			}
		}
	}

}

void 
	SignalEditor::translateSignal(SignalItem *signalItem, QPointF &diff)
{
	Signal *signal = signalItem->getSignal();
	RSystemElementRoad * road = signal->getParentRoad();
	double s;
	QVector2D vec;
	double dist;
	QPointF to = road->getGlobalPoint(signal->getSStart(), signal->getT()) + diff;
	RSystemElementRoad * newRoad = getProjectData()->getRoadSystem()->findClosestRoad( to, s, dist, vec);

	if (newRoad != road)
	{
		RemoveSignalCommand * removeSignalCommand = new RemoveSignalCommand(signal, road);
		getProjectGraph()->executeCommand(removeSignalCommand);

		AddSignalCommand * addSignalCommand = new AddSignalCommand(signal, newRoad);
		getProjectGraph()->executeCommand(addSignalCommand);
		signal->setElementSelected(false);

		road = newRoad;
	}  

	LaneSection *laneSection = road->getLaneSection(s);
	int validToLane = 0;
	int validFromLane = 0;
	if (dist < 0)
	{
		validToLane = laneSection->getRightmostLaneId();
	}
	else
	{
		validFromLane = laneSection->getLeftmostLaneId();
	}

	SetSignalPropertiesCommand * signalPropertiesCommand = new SetSignalPropertiesCommand(signal, signal->getId(), signal->getName(), dist, signal->getDynamic(), signal->getOrientation(), signal->getZOffset(), signal->getCountry(), signal->getType(), signal->getTypeSubclass(), signal->getSubtype(), signal->getValue(), signal->getHeading(), signal->getPitch(), signal->getRoll(), signal->getUnit(), signal->getText(), signal->getWidth(), signal->getHeight(),  signal->getPole(), signal->getSize(), validFromLane, validToLane, signal->getCrossingProbability(), signal->getResetTime());
	getProjectGraph()->executeCommand(signalPropertiesCommand);
	MoveRoadSectionCommand * moveSectionCommand = new MoveRoadSectionCommand(signal, s, RSystemElementRoad::DRS_SignalSection);
	getProjectGraph()->executeCommand(moveSectionCommand);
}

void
SignalEditor::translate(QPointF &diff)
{

	getProjectData()->getUndoStack()->beginMacro(QObject::tr("Move Signal"));

	QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();

	foreach (QGraphicsItem *item, selectedItems)
	{
		SignalItem *signalItem = dynamic_cast<SignalItem *>(item);
		if (signalItem)
		{
			translateSignal(signalItem, diff);
		}
		else
		{
			ObjectItem *objectItem = dynamic_cast<ObjectItem *>(item);
			if (objectItem)
			{
				translateObject(objectItem, diff);
			}
			else
			{
				BridgeItem *bridgeItem = dynamic_cast<BridgeItem *>(item);
				if (bridgeItem)
				{
					translateBridge(bridgeItem, diff);
				}
			}
		}
	}

	getProjectData()->getUndoStack()->endMacro();
}

Signal *
SignalEditor::addSignalToRoad(RSystemElementRoad *road, double s, double t)
{
	int validToLane = 0;	// make a new signal //
	int validFromLane = 0;

	LaneSection *laneSection = road->getLaneSection(s);
	if (t < 0)
	{
		validToLane = laneSection->getRightmostLaneId();
	}
	else
	{
		validFromLane = laneSection->getLeftmostLaneId();
	}

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
		newSignal = new Signal("signal", "", s, t, false, Signal::NEGATIVE_TRACK_DIRECTION, lastSignal->getSignalheightOffset(), signalManager_->getCountry(lastSignal), lastSignal->getSignalType(), lastSignal->getSignalTypeSubclass(), lastSignal->getSignalSubType(), lastSignal->getSignalValue(), 0.0, 0.0, 0.0, lastSignal->getSignalUnit(), lastSignal->getSignalText(),lastSignal->getSignalWidth(), lastSignal->getSignalHeight(), true, 2, validFromLane, validToLane);
		AddSignalCommand *command = new AddSignalCommand(newSignal, road, NULL);
		getProjectGraph()->executeCommand(command);
	}
	else
	{
		newSignal = new Signal("signal", "", s, t, false, Signal::NEGATIVE_TRACK_DIRECTION, 0.0, "Germany", -1, "", -1, 0.0, 0.0, 0.0, 0.0, "hm/h", "", 0.0, 0.0, true, 2, validFromLane, validToLane);
		AddSignalCommand *command = new AddSignalCommand(newSignal, road, NULL);
		getProjectGraph()->executeCommand(command);
	}

	return newSignal;
}

void 
SignalEditor::translateObject(ObjectItem * objectItem, QPointF &diff)
{
	Object *object = objectItem->getObject();
    RSystemElementRoad * road = object->getParentRoad();
	double s;
	QVector2D vec;
	double dist;
	QPointF to = road->getGlobalPoint(object->getSStart(), object->getT()) + diff;
	RSystemElementRoad * newRoad = getProjectData()->getRoadSystem()->findClosestRoad( to, s, dist, vec);

    if (newRoad != road)
    {
        RemoveObjectCommand * removeObjectCommand = new RemoveObjectCommand(object, road);
        getProjectGraph()->executeCommand(removeObjectCommand);

        AddObjectCommand * addObjectCommand = new AddObjectCommand(object, newRoad);
        getProjectGraph()->executeCommand(addObjectCommand);
		object->setElementSelected(false);

        road = newRoad;
    }  

	Object::ObjectProperties objectProps = object->getProperties();
	objectProps.t = dist;
	Object::ObjectRepeatRecord repeatProps = object->getRepeatProperties();
	repeatProps.s = s;
	SetObjectPropertiesCommand * objectPropertiesCommand = new SetObjectPropertiesCommand(object, object->getId(), object->getName(), objectProps, repeatProps, object->getTextureFileName());
    getProjectGraph()->executeCommand(objectPropertiesCommand);
    MoveRoadSectionCommand * moveSectionCommand = new MoveRoadSectionCommand(object, s, RSystemElementRoad::DRS_ObjectSection);
    getProjectGraph()->executeCommand(moveSectionCommand);
}

Object *
SignalEditor::addObjectToRoad(RSystemElementRoad *road, double s, double t)
{
	ObjectContainer *lastObject = signalManager_->getSelectedObjectContainer();
	Object *newObject = NULL;

	if (lastObject)
	{
		if (t < 0)
		{
			t -= lastObject->getObjectDistance();
		}
		else
		{
			t += lastObject->getObjectDistance();
		}

		Object::ObjectProperties objectProps{ t, Object::NEGATIVE_TRACK_DIRECTION, 0.0, lastObject->getObjectType(), 0.0, lastObject->getObjectLength(), lastObject->getObjectWidth(),
			lastObject->getObjectRadius(), lastObject->getObjectHeight(), lastObject->getObjectHeading(),
			0.0, 0.0, false };

		Object::ObjectRepeatRecord repeatProps{ s, 0.0, lastObject->getObjectRepeatDistance(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };  // TODO: add properties to container
		newObject = new Object("object", "", s, objectProps, repeatProps, lastObject->getObjectFile());
/*		newObject = new Object("object", "", lastObject->getObjectType(), s, t, 0.0, 0.0, Object::NEGATIVE_TRACK_DIRECTION, lastObject->getObjectLength(), 
			lastObject->getObjectWidth(), lastObject->getObjectRadius(), lastObject->getObjectHeight(), lastObject->getObjectHeading(),
					0.0, 0.0, false, s, 0.0, lastObject->getObjectRepeatDistance(), lastObject->getObjectFile()); */
		AddObjectCommand *command = new AddObjectCommand(newObject, road, NULL);
		getProjectGraph()->executeCommand(command);
	}
	else
	{
		Object::ObjectProperties objectProps{ t, Object::NEGATIVE_TRACK_DIRECTION, 0.0, "", 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, false };
		Object::ObjectRepeatRecord repeatProps{ s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		Object *newObject = new Object("object", "", s, objectProps, repeatProps, "");
		AddObjectCommand *command = new AddObjectCommand(newObject, road, NULL);
		getProjectGraph()->executeCommand(command);
	}

	return newObject;
}

void 
SignalEditor::translateBridge(BridgeItem * bridgeItem, QPointF &diff)
{
	Bridge *bridge = bridgeItem->getBridge();
    RSystemElementRoad * road = bridge->getParentRoad();

	double s;
	QVector2D vec;
	double dist;
	QPointF to = road->getGlobalPoint(bridge->getSStart()) + diff;
	RSystemElementRoad * newRoad = getProjectData()->getRoadSystem()->findClosestRoad( to, s, dist, vec);


    if (newRoad != road)
    {
        RemoveBridgeCommand * removeBridgeCommand = new RemoveBridgeCommand(bridge, road);
        getProjectGraph()->executeCommand(removeBridgeCommand);

        AddBridgeCommand * addBridgeCommand = new AddBridgeCommand(bridge, newRoad);
        getProjectGraph()->executeCommand(addBridgeCommand);
		bridge->setElementSelected(false);

        road = newRoad;
    }  

	SetBridgePropertiesCommand * bridgePropertiesCommand = new SetBridgePropertiesCommand(bridge, bridge->getId(), bridge->getFileName(), bridge->getName(), bridge->getType(),bridge->getLength());
    getProjectGraph()->executeCommand(bridgePropertiesCommand);
    MoveRoadSectionCommand * moveSectionCommand = new MoveRoadSectionCommand(bridge, s, RSystemElementRoad::DRS_BridgeSection);
    getProjectGraph()->executeCommand(moveSectionCommand);
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
	ODD::ToolId currentTool = getCurrentTool();
    if (currentTool == ODD::TSG_SELECT)
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
	else if ((currentTool == ODD::TSG_SIGNAL) || (currentTool == ODD::TSG_OBJECT))
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
					RSystemElementRoad * road = getProjectData()->getRoadSystem()->findClosestRoad(mousePoint, s, t, vec);
					if (road)
					{
						if (currentTool == ODD::TSG_SIGNAL)
						{
							addSignalToRoad(road, s, t);   
						}
						else
						{
							addObjectToRoad(road, s, t);
						}
					}
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
			signalTree_->clearSelection();
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
			|| (currentTool == ODD::TSG_BRIDGE) || (currentTool = ODD::TSG_TUNNEL))
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
