/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/15/2010
**
**************************************************************************/

#include "laneeditor.hpp"

#include <qundostack.h>

// Project //
//
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/commands/lanesectioncommands.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/graphview.hpp"
#include "src/graph/items/handles/lanemovehandle.hpp"

#include "src/graph/items/roadsystem/lanes/laneroadsystemitem.hpp"
#include "src/graph/items/roadsystem/lanes/lanesectionitem.hpp"
#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"

// Tools //
//
#include "src/gui/tools/laneeditortool.hpp"
#include "src/gui/mouseaction.hpp"

// Qt //
//
#include <QGraphicsItem>

//################//
// CONSTRUCTORS   //
//################//

LaneEditor::LaneEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph)
    : ProjectEditor(projectWidget, projectData, topviewGraph)
    , roadSystemItem_(NULL)
    , insertSectionHandle_(NULL)
	, pointHandle_(NULL)
{
}

LaneEditor::~LaneEditor()
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
LaneEditor::init()
{
    if (!roadSystemItem_)
    {
        // Root item //
        //
        roadSystemItem_ = new LaneRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
        getTopviewGraph()->getScene()->addItem(roadSystemItem_);

        // Section Handle //
        //
        insertSectionHandle_ = new SectionHandle(roadSystemItem_);
        insertSectionHandle_->hide();

		// Width Add Handle //
		//
		pointHandle_ = new PointHandle(roadSystemItem_);
		pointHandle_->hide();
    }
}

/*!
*/
void
LaneEditor::kill()
{
    delete roadSystemItem_;
    roadSystemItem_ = NULL;
}

SectionHandle *
LaneEditor::getInsertSectionHandle() const
{
    if (!insertSectionHandle_)
    {
        qDebug("ERROR 1010151634! LaneEditor not yet initialized.");
    }
    return insertSectionHandle_;
}

PointHandle *
LaneEditor::getAddWidthHandle() const
{
	if (!pointHandle_)
	{
		qDebug("ERROR 1010151634! LaneEditor not yet initialized.");
	}
	return pointHandle_;
}

//################//
// TOOL           //
//################//

/*! \brief .
*
*/
void
LaneEditor::toolAction(ToolAction *toolAction)
{
    // Parent //
    //
    ProjectEditor::toolAction(toolAction);

	if (getCurrentTool() == ODD::TLE_SELECT)
	{
		LaneEditorToolAction *laneEditorToolAction = dynamic_cast<LaneEditorToolAction *>(toolAction);
		if (laneEditorToolAction)
		{
			LaneEditorToolAction::ActionType type = laneEditorToolAction->getType();
			if (selectedLaneMoveHandles_.size() > 0 && (type == LaneEditorToolAction::Width))
			{
				translateLaneBorder(QPointF(0, 0), QPointF(0, 0), laneEditorToolAction->getWidth(), true);
			}
		}
	}
}

void
LaneEditor::mouseAction(MouseAction *mouseAction)
{
	QGraphicsSceneMouseEvent *mouseEvent = mouseAction->getEvent();

	if (getCurrentTool() == ODD::TTE_ROAD_NEW)
	{
		QPointF mousePoint = mouseAction->getEvent()->scenePos();
	}
}


//################//
// LaneMoveHandles    //
//################//

void
LaneEditor::registerMoveHandle(BaseLaneMoveHandle *handle)
{
	selectedLaneMoveHandles_.append(handle);
}

int
LaneEditor::unregisterMoveHandle(BaseLaneMoveHandle *handle)
{
	return selectedLaneMoveHandles_.removeOne(handle);
} 

bool
LaneEditor::translateLaneBorder(const QPointF &pressPos, const QPointF &mousePosConst, double width, bool setWidth)
{

	QMap<RSystemElementRoad *, QMap<double, LaneMoveProperties *>> selectedLaneMoveProps;
	foreach (BaseLaneMoveHandle *baseMoveHandle, selectedLaneMoveHandles_)
	{
		LaneMoveProperties *props = new LaneMoveProperties();
		LaneWidth *lowSlot = baseMoveHandle->getLowSlot();
		RSystemElementRoad *road;
		double s;
		if (lowSlot)
		{
			props->lowSlot = lowSlot;
			road = lowSlot->getParentLane()->getParentLaneSection()->getParentRoad();
			s = lowSlot->getSSectionEnd();
		}
		LaneWidth *highSlot = baseMoveHandle->getHighSlot();
		if (highSlot)
		{
			props->highSlot = highSlot;
			if (!lowSlot)
			{
				road = highSlot->getParentLane()->getParentLaneSection()->getParentRoad();
				s = highSlot->getSSectionStartAbs();
			}
		}

		QMap<double, LaneMoveProperties *> propsMap;
		if (selectedLaneMoveProps.find(road) != selectedLaneMoveProps.end())
		{
			propsMap = selectedLaneMoveProps.value(road);
		}

		propsMap.insert(s, props);
		selectedLaneMoveProps.insert(road, propsMap);
	}
	
	if (setWidth)
	{
		TranslateLaneBorderCommand *command = new TranslateLaneBorderCommand(selectedLaneMoveProps, width, QPointF(0,0), NULL);
		return getProjectGraph()->executeCommand(command);
	}
	else
	{
		QPointF mousePos = mousePosConst;

		// Snap to MoveHandle //
		//
		foreach(QGraphicsItem *item, getTopviewGraph()->getScene()->items(mousePos, Qt::IntersectsItemShape, Qt::AscendingOrder, getTopviewGraph()->getView()->viewportTransform()))
		{
			BaseLaneMoveHandle *handle = dynamic_cast<BaseLaneMoveHandle *>(item);

			if (handle)
			{
				mousePos = handle->pos();
				break;
			}
		}

		if ((mousePos - pressPos).manhattanLength() < NUMERICAL_ZERO6)
		{
			return false;
		}

		// DeltaPos  //
		//
		QPointF dPos = mousePos - pressPos;

		TranslateLaneBorderCommand *command = new TranslateLaneBorderCommand(selectedLaneMoveProps, width, dPos, NULL);
		return getProjectGraph()->executeCommand(command);
	}

}
