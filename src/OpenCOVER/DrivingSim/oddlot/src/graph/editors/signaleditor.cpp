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

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/roadsystem.hpp"

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
}

/*!
*/
void
SignalEditor::kill()
{
    delete signalRoadSystemItem_;
    signalRoadSystemItem_ = NULL;
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

    if (getCurrentTool() == ODD::TSG_SELECT)
    {
        // does nothing //
    }
    else if ((getCurrentTool() == ODD::TSG_SIGNAL) || (getCurrentTool() == ODD::TSG_OBJECT) || (getCurrentTool() == ODD::TSG_BRIDGE))
    {
        foreach (QGraphicsItem *item, getTopviewGraph()->getScene()->selectedItems())
        {
            item->setSelected(false);
        }
        // does nothing //
        // Problem: The ToolAction is resent, after a warning message has been clicked away. (Due to resend on getting the
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
