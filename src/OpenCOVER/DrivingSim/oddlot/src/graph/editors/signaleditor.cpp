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
#include "src/data/commands/dataelementcommands.hpp"

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
#include "src/graph/items/roadsystem/controlleritem.hpp"

// Tools //
//
#include "src/gui/tools/signaleditortool.hpp"
#include "src/gui/mouseaction.hpp"

// Tree //
//
#include "src/tree/signaltreewidget.hpp"

// GUI //
//
#include "src/gui/parameters/toolvalue.hpp"
#include "src/gui/parameters/toolparametersettings.hpp"

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
	, controller_(NULL)
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
		newSignal = new Signal(odrID::invalidID(), "signal", s, t, false, Signal::NEGATIVE_TRACK_DIRECTION, lastSignal->getSignalheightOffset(), signalManager_->getCountry(lastSignal), lastSignal->getSignalType(), lastSignal->getSignalTypeSubclass(), lastSignal->getSignalSubType(), lastSignal->getSignalValue(), 0.0, 0.0, 0.0, lastSignal->getSignalUnit(), lastSignal->getSignalText(),lastSignal->getSignalWidth(), lastSignal->getSignalHeight(), true, 2, validFromLane, validToLane);
		AddSignalCommand *command = new AddSignalCommand(newSignal, road, NULL);
		getProjectGraph()->executeCommand(command);
	}
	else
	{
		newSignal = new Signal(odrID::invalidID(), "signal", s, t, false, Signal::NEGATIVE_TRACK_DIRECTION, 0.0, "Germany", "-1", "", "-1", 0.0, 0.0, 0.0, 0.0, "hm/h", "", 0.0, 0.0, true, 2, validFromLane, validToLane);
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

		Object::ObjectProperties objectProps{ t, Signal::NEGATIVE_TRACK_DIRECTION, 0.0, lastObject->getObjectType(), 0.0, lastObject->getObjectLength(), lastObject->getObjectWidth(),
			lastObject->getObjectRadius(), lastObject->getObjectHeight(), lastObject->getObjectHeading(),
			0.0, 0.0, false };

		Object::ObjectRepeatRecord repeatProps{ s, 0.0, lastObject->getObjectRepeatDistance(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };  // TODO: add properties to container
		newObject = new Object(odrID::invalidID(), "object", s, objectProps, repeatProps, lastObject->getObjectFile());
/*		newObject = new Object("object", "", lastObject->getObjectType(), s, t, 0.0, 0.0, Object::NEGATIVE_TRACK_DIRECTION, lastObject->getObjectLength(), 
			lastObject->getObjectWidth(), lastObject->getObjectRadius(), lastObject->getObjectHeight(), lastObject->getObjectHeading(),
					0.0, 0.0, false, s, 0.0, lastObject->getObjectRepeatDistance(), lastObject->getObjectFile()); */
		AddObjectCommand *command = new AddObjectCommand(newObject, road, NULL);
		getProjectGraph()->executeCommand(command);
	}
	else
	{
		Object::ObjectProperties objectProps{ t, Signal::NEGATIVE_TRACK_DIRECTION, 0.0, "", 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, false };
		Object::ObjectRepeatRecord repeatProps{ s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		Object *newObject = new Object(odrID::invalidID(), "object",  s, objectProps, repeatProps, "");
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
	static QList<QGraphicsItem *> oldSelectedItems;

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
	else if ((currentTool == ODD::TSG_SIGNAL) || (currentTool == ODD::TSG_OBJECT) || (currentTool == ODD::TSG_BRIDGE) || (currentTool == ODD::TSG_TUNNEL))
    {
        if (mouseAction->getMouseActionType() == MouseAction::ATM_DROP)
        {
            QGraphicsSceneDragDropEvent *mouseEvent = mouseAction->getDragDropEvent();
            QPointF mousePoint = mouseEvent->scenePos();

			QList<QGraphicsItem *> underMouseItems = getTopviewGraph()->getScene()->items(mousePoint);

			RSystemElementRoad * road = NULL;
			double s, t;
			for (int i = 0; i < underMouseItems.size(); i++)
			{
				SignalRoadItem *roadItem = dynamic_cast<SignalRoadItem *>(underMouseItems.at(i));
				if (roadItem)
				{
					road = roadItem->getRoad();
					s = road->getSFromGlobalPoint(mousePoint);
					t = road->getTFromGlobalPoint(mousePoint, s);

					break;
				}
			}

			if (!road)		// find the closest road //
			{
				QVector2D vec;
				road = getProjectData()->getRoadSystem()->findClosestRoad(mousePoint, s, t, vec);
			}

			if (road)
			{
				switch (currentTool)
				{
				case ODD::TSG_SIGNAL:
					addSignalToRoad(road, s, t);
					break;
				case ODD::TSG_OBJECT:
					addObjectToRoad(road, s, t);
					break;
				case ODD::TSG_BRIDGE: {
					// Add new bridge //
					//
					Bridge *newBridge = new Bridge(getProjectData()->getRoadSystem()->getID("bridge", odrID::ID_Bridge), "", "", Bridge::BT_CONCRETE, s, 100.0);
					AddBridgeCommand *command = new AddBridgeCommand(newBridge, road, NULL);

					getProjectGraph()->executeCommand(command);
					break; }
				case ODD::TSG_TUNNEL: {
					// Add new tunnel //
					//
					Tunnel *newTunnel = new Tunnel(getProjectData()->getRoadSystem()->getID("tunnel", odrID::ID_Bridge), "", "", Tunnel::TT_STANDARD, s, 100.0, 0.0, 0.0);
					AddBridgeCommand *command = new AddBridgeCommand(newTunnel, road, NULL);

					getProjectGraph()->executeCommand(command);
					break; }
                default:
                    break;
				}
			}

		}
        else if (mouseAction->getMouseActionType() == MouseAction::ATM_DOUBLECLICK)
        {
            //opens the ui for shieldeditor

            /*QMessageBox msg;
            msg.setText("HELLO!");
            msg.exec();*/
        }
	}
	else if ((getCurrentTool() == ODD::TSG_CONTROLLER) || (getCurrentTool() == ODD::TSG_ADD_CONTROL_ENTRY) || (getCurrentTool() == ODD::TSG_REMOVE_CONTROL_ENTRY))
	{
		if (getCurrentParameterTool() == ODD::TPARAM_SELECT)
		{
			if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
			{
				if (mouseAction->getEvent()->button() == Qt::LeftButton)
				{
					if (selectedSignals_.empty())
					{
						oldSelectedItems.clear();
					}

					QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();
					QList<Signal *>selectionChangedSignals;
					QMultiMap<Signal *, QGraphicsItem *>graphicSignalItems;

					for (int i = 0; i < selectedItems.size();)
					{
						QGraphicsItem *item = selectedItems.at(i);
						SignalItem *signalItem = dynamic_cast<SignalItem *>(item);
						if (signalItem)
						{
							Signal *signal = signalItem->getSignal();
							if (!oldSelectedItems.contains(item))
							{
								if (!selectionChangedSignals.contains(signal))
								{
									if (!selectedSignals_.contains(signal))
									{
										createToolParameters<Signal>(signal);
										selectedSignals_.append(signal);

										item->setSelected(true);
									}
									else
									{
										item->setSelected(false);
										graphicSignalItems.insert(signal, item);

										removeToolParameters<Signal>(signal);
										selectedSignals_.removeOne(signal);
									}
									selectionChangedSignals.append(signal);
								}
								else if (!selectedSignals_.contains(signal))
								{
									graphicSignalItems.insert(signal, item);
								}
							}
							else
							{
								int j = oldSelectedItems.indexOf(item);
								oldSelectedItems.takeAt(j);
								graphicSignalItems.insert(signal, item);
							}
							i++;
						}
						else
						{
							item->setSelected(false);
							selectedItems.removeAt(i);
						}
					}

					for (int i = 0; i < selectionChangedSignals.size(); i++)
					{
						Signal *signal = selectionChangedSignals.at(i);

						if (!selectedSignals_.contains(signal))
						{
							QGraphicsItem *signalItem = graphicSignalItems.value(signal);
							selectedItems.removeOne(signalItem);
							oldSelectedItems.removeOne(signalItem);
							graphicSignalItems.remove(signal);
						}

					}

					for (int i = 0; i < oldSelectedItems.size(); i++)
					{
						QGraphicsItem *item = oldSelectedItems.at(i);
						SignalItem *signalItem = dynamic_cast<SignalItem *>(item);
						if (signalItem)
						{
							Signal *signal = signalItem->getSignal();
							if (!selectionChangedSignals.contains(signal))
							{
								item->setSelected(false);

								removeToolParameters<Signal>(signal);
								selectedSignals_.removeOne(signal);

								selectionChangedSignals.append(signal);
							}
						}
					}

					for (int i = 0; i < selectionChangedSignals.size(); i++)
					{
						Signal *signal = selectionChangedSignals.at(i);
						if (!selectedSignals_.contains(signal))
						{
							QGraphicsItem *signalItem = graphicSignalItems.value(signal);
							selectedItems.removeOne(signalItem);
							graphicSignalItems.remove(signal);
						}
					}
					oldSelectedItems = selectedItems;
					mouseAction->intercept();

					// verify if apply can be displayed //

					int objectCount = tool_->getObjectCount(getCurrentTool(), getCurrentParameterTool());
					if ((objectCount >= applyCount_) && (((getCurrentTool() != ODD::TSG_ADD_CONTROL_ENTRY) && (getCurrentTool() != ODD::TSG_REMOVE_CONTROL_ENTRY)) || controller_))
					{
						settingsApplyBox_->setApplyButtonVisible(true);
					}
				}
			}
		}
	}
	else if (getCurrentTool() == ODD::TSG_SELECT_CONTROLLER)
	{
		if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
		{
			if (getCurrentParameterTool() == ODD::TPARAM_SELECT)
			{
				if (mouseAction->getEvent()->button() == Qt::LeftButton)
				{

					QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();
					foreach(QGraphicsItem *item, selectedItems)
					{
						ControllerItem *controllerItem = dynamic_cast<ControllerItem *>(item);
						if (controllerItem)
						{
							controller_ = controllerItem->getController();
							setToolValue<RSystemElementController>(controller_, controller_->getIdName());
						}
						else //if (!oldSelectedItems.contains(item))
						{
							item->setSelected(false);
						}
					}

					// verify if apply can be displayed //

					int objectCount = tool_->getObjectCount(tool_->getToolId(), getCurrentParameterTool());
					if ((objectCount >= applyCount_) && controller_)
					{
						settingsApplyBox_->setApplyButtonVisible(true);
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

	SignalEditorToolAction *signalEditorToolAction = dynamic_cast<SignalEditorToolAction *>(toolAction);
	if (signalEditorToolAction)
	{
		// Create Controller //
			//
		if (signalEditorToolAction->getToolId() == ODD::TSG_CONTROLLER)
		{
			ODD::ToolId paramTool = getCurrentParameterTool();

			if ((paramTool == ODD::TNO_TOOL) && !tool_)
			{
				getTopviewGraph()->getScene()->deselectAll();

				ToolValue<Signal> *param = new ToolValue<Signal>(ODD::TSG_CONTROLLER, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT_LIST, "Select/Remove");
				tool_ = new Tool(ODD::TSG_CONTROLLER, 4);
				tool_->readParams(param);

				createToolParameterSettingsApplyBox(tool_, ODD::ESG);
				ODD::mainWindow()->showParameterDialog(true, "Create controller from signals", "SELECT/DESELECT signals and press APPLY to create Controller");

				applyCount_ = 1;

			}
		}

		else if (signalEditorToolAction->getToolId() == ODD::TSG_REMOVE_CONTROL_ENTRY)
		{
			ODD::ToolId paramTool = getCurrentParameterTool();

			if ((paramTool == ODD::TNO_TOOL) && !tool_)
			{
				getTopviewGraph()->getScene()->deselectAll();

				ToolValue<RSystemElementController> *param = new ToolValue<RSystemElementController>(ODD::TSG_SELECT_CONTROLLER, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select Controller");
				tool_ = new Tool(ODD::TSG_REMOVE_CONTROL_ENTRY, 4);
				tool_->readParams(param);
				ToolValue<Signal> *signalParam = new ToolValue<Signal>(ODD::TSG_REMOVE_CONTROL_ENTRY, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT_LIST, "Select/Remove");
				tool_->readParams(signalParam);

				createToolParameterSettingsApplyBox(tool_, ODD::ESG);
				ODD::mainWindow()->showParameterDialog(true, "Remove signals from controller", "SELECT a controller, SELECT/DESELECT signals and press APPLY");

				applyCount_ = 1;

			}
		}
		else if (signalEditorToolAction->getToolId() == ODD::TSG_ADD_CONTROL_ENTRY)
		{
			ODD::ToolId paramTool = getCurrentParameterTool();

			if ((paramTool == ODD::TNO_TOOL) && !tool_)
			{
				getTopviewGraph()->getScene()->deselectAll();

				ToolValue<RSystemElementController> *param = new ToolValue<RSystemElementController>(ODD::TSG_SELECT_CONTROLLER, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select Controller");
				tool_ = new Tool(ODD::TSG_ADD_CONTROL_ENTRY, 4);
				tool_->readParams(param);
				ToolValue<Signal> *signalParam = new ToolValue<Signal>(ODD::TSG_ADD_CONTROL_ENTRY, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT_LIST, "Select/Remove");
				tool_->readParams(signalParam);

				createToolParameterSettingsApplyBox(tool_, ODD::ESG);
				ODD::mainWindow()->showParameterDialog(true, "Add signals to controller", "SELECT a controller, SELECT/DESELECT signals and press APPLY");

				applyCount_ = 1;

			}
		}
	}
	else
	{
		ParameterToolAction *action = dynamic_cast<ParameterToolAction *>(toolAction);
		if (action)
		{
			if ((action->getToolId() == ODD::TSG_CONTROLLER) || (action->getToolId() == ODD::TSG_ADD_CONTROL_ENTRY) || (action->getToolId() == ODD::TSG_REMOVE_CONTROL_ENTRY))
			{
				if (action->getParamToolId() == ODD::TPARAM_SELECT)
				{
					if (!action->getState())
					{

						QList<Signal *> signalList = tool_->removeToolParameters<Signal>(action->getParamId());
						foreach(Signal *signal, signalList)
						{
							DeselectDataElementCommand *command = new DeselectDataElementCommand(signal, NULL);
							getProjectGraph()->executeCommand(command);
							selectedSignals_.removeOne(signal);
						}

						// verify if apply has to be hidden //
						if (tool_->getObjectCount(getCurrentTool(), getCurrentParameterTool()) <= applyCount_)
						{
							settingsApplyBox_->setApplyButtonVisible(false);
						}
					}
				}
				else if ((action->getParamToolId() == ODD::TNO_TOOL) && !tool_)
				{
					if (action->getToolId() == ODD::TSG_CONTROLLER)
					{
						ToolValue<Signal> *param = new ToolValue<Signal>(ODD::TSG_CONTROLLER, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT_LIST, "Select/Remove");
						tool_ = new Tool(ODD::TSG_CONTROLLER, 4);
						tool_->readParams(param);

						generateToolParameterUI(tool_);
					}

					else if (action->getToolId() == ODD::TSG_ADD_CONTROL_ENTRY)
					{
						controller_ = NULL;

						ToolValue<RSystemElementController> *param = new ToolValue<RSystemElementController>(ODD::TSG_SELECT_CONTROLLER, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select Controller");
						tool_ = new Tool(ODD::TSG_ADD_CONTROL_ENTRY, 4);
						tool_->readParams(param);
						ToolValue<Signal> *signalParam = new ToolValue<Signal>(ODD::TSG_ADD_CONTROL_ENTRY, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT_LIST, "Select/Remove");
						tool_->readParams(signalParam);

						generateToolParameterUI(tool_);
					}

					else if (action->getToolId() == ODD::TSG_REMOVE_CONTROL_ENTRY)
					{
						controller_ = NULL;

						ToolValue<RSystemElementController> *param = new ToolValue<RSystemElementController>(ODD::TSG_SELECT_CONTROLLER, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT, "Select Controller");
						tool_ = new Tool(ODD::TSG_ADD_CONTROL_ENTRY, 4);
						tool_->readParams(param);
						ToolValue<Signal> *signalParam = new ToolValue<Signal>(ODD::TSG_ADD_CONTROL_ENTRY, ODD::TPARAM_SELECT, 1, ToolParameter::ParameterTypes::OBJECT_LIST, "Select/Remove");
						tool_->readParams(signalParam);

						generateToolParameterUI(tool_);
					}
				}
			}
		}
	}


    if (currentTool != lastTool_)
    {
        if ((currentTool == ODD::TSG_SIGNAL) || (currentTool == ODD::TSG_OBJECT) 
			|| (currentTool == ODD::TSG_BRIDGE) || (currentTool == ODD::TSG_TUNNEL))
        {
            foreach (QGraphicsItem *item, getTopviewGraph()->getScene()->selectedItems())
            {
                item->setSelected(false);
            }
            // does nothing //
            // Problem: The ToolAction is resent, after a warning message has been clicked away. (Due to resend on getting the
        }


        lastTool_ = currentTool;

    } 

   
}

//################//
// SLOTS          //
//################//

void
SignalEditor::apply()
{
	ODD::ToolId toolId = tool_->getToolId();
	if (toolId == ODD::TSG_CONTROLLER)
	{
		getProjectGraph()->beginMacro("Add Controller");

		QList<ControlEntry *>controlEntryList;
		RSystemElementController *newController = new RSystemElementController("controller", getProjectData()->getRoadSystem()->getID(odrID::ID_Controller), 0, "", 0.0, controlEntryList);
		AddControllerCommand *command = new AddControllerCommand(newController, getProjectData()->getRoadSystem(), NULL);

		getProjectGraph()->executeCommand(command);

		foreach(Signal *signal, selectedSignals_)
		{
			ControlEntry * controlEntry = new ControlEntry(signal->getId(), signal->getType());
			AddControlEntryCommand *addControlEntryCommand = new AddControlEntryCommand(newController, controlEntry, signal);
			getProjectGraph()->executeCommand(addControlEntryCommand);
		}

		getProjectGraph()->endMacro();
	}
	else if (toolId == ODD::TSG_ADD_CONTROL_ENTRY)
	{
		getProjectGraph()->beginMacro("Add ControlEntry");

		foreach(Signal *signal, selectedSignals_)
		{
			ControlEntry * controlEntry = new ControlEntry(signal->getId(), signal->getType());
			AddControlEntryCommand *addControlEntryCommand = new AddControlEntryCommand(controller_, controlEntry, signal);
			getProjectGraph()->executeCommand(addControlEntryCommand);
		}

		getProjectGraph()->endMacro();
	}
	else if (toolId == ODD::TSG_REMOVE_CONTROL_ENTRY)
	{
		getProjectGraph()->beginMacro("Remove ControlEntry");

		foreach(Signal *signal, selectedSignals_)
		{
			foreach(ControlEntry * controlEntry, controller_->getControlEntries())
			{
				if (controlEntry->getSignalId() == signal->getId())
				{
					DelControlEntryCommand *delControlEntryCommand = new DelControlEntryCommand(controller_, controlEntry, signal);
					getProjectGraph()->executeCommand(delControlEntryCommand);
					break;
				}
			}
		}

		getProjectGraph()->endMacro();
	}
	
}


void
SignalEditor::clearToolObjectSelection()
{
	// deselect all //

	foreach(Signal *signal, selectedSignals_)
	{
		DeselectDataElementCommand *command = new DeselectDataElementCommand(signal, NULL);
		getProjectGraph()->executeCommand(command);
	}

	selectedSignals_.clear();
}

void
SignalEditor::reset()
{
	ODD::ToolId toolId = tool_->getToolId();
	clearToolObjectSelection();
	delToolParameters();
}

void SignalEditor::reject()
{
	ProjectEditor::reject();

	clearToolObjectSelection();
	deleteToolParameterSettings();
	ODD::mainWindow()->showParameterDialog(false);
}
