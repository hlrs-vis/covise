/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   31.03.2010
**
**************************************************************************/

#include "toolmanager.hpp"

// Tools //
//
#include "tool.hpp"
#include "toolwidget.hpp"

#include "zoomtool.hpp"
#include "selectiontool.hpp"

#include "roadlinkeditortool.hpp"
#include "typeeditortool.hpp"
#include "trackeditortool.hpp"
#include "elevationeditortool.hpp"
#include "superelevationeditortool.hpp"
#include "crossfalleditortool.hpp"
#include "shapeeditortool.hpp"
#include "laneeditortool.hpp"
#include "junctioneditortool.hpp"
#include "signaleditortool.hpp"
#include "osceditortool.hpp"

#include "maptool.hpp"

#include "src/gui/projectwidget.hpp"

// Qt //
//
#include <QToolBox>

//################//
// CONSTRUCTOR    //
//################//

ToolManager::ToolManager(PrototypeManager *prototypeManager, QObject *parent)
    : QObject(parent)
    , prototypeManager_(prototypeManager)
    , lastToolAction_(NULL)
	, currentProject_(NULL)
{
    initTools();
}

//	ToolManager
//	::~ToolManager()
//{
//	// TODO
//}

void
ToolManager::addMenu(QMenu *menu)
{
    menus_.append(menu);
}

void
ToolManager::addToolBoxWidget(ToolWidget *widget, const QString &title)
{
    int index = toolBox_->addItem(widget, title);
    widget->setToolBoxIndex(index);
    connect(toolBox_, SIGNAL(currentChanged(int)), widget, SLOT(activateWidget(int)));
    // parentship of the widget is not set? TODO (toolmanager is no widget)
}

void
ToolManager::addRibbonWidget(ToolWidget *widget, const QString &title)
{
    int index = ribbon_->addTab(widget,title);
    widget->setToolBoxIndex(index);
    connect(ribbon_, SIGNAL(currentChanged(int)), widget, SLOT(activateWidget(int)));
}
void
ToolManager::initTools()
{
    // Tool Box //
    //
    toolBox_ = new QToolBox();
    toolBox_->setMinimumWidth(190);
    
    // Ribbon based Toolbox
    //
    ribbon_ = new QTabWidget;
    ribbon_->setMinimumHeight(100);
    ribbon_->setMaximumHeight(100);


    // Zoom //
    //
    zoomTool_ = new ZoomTool(this);

    // Selection //
    //
 //   selectionTool_ = new SelectionTool(this);

    // RoadLinkEditor //
    //
    RoadLinkEditorTool *defaultEditor = new RoadLinkEditorTool(this);

    // TypeEditor //
    //
    new TypeEditorTool(this);

    // TrackEditor //
    //
    new TrackEditorTool(prototypeManager_, this);

    // ElevationEditor //
    //
    new ElevationEditorTool(this);

    // SuperelevationEditor //
    //
    new SuperelevationEditorTool(this);

    // CrossfallEditor //
    //
    new CrossfallEditorTool(this);

	// RoadShapeEditor //
	//
	new ShapeEditorTool(this);

    // LaneEditor //
    //
    new LaneEditorTool(this);

    // JunctionEditor //
    new JunctionEditorTool(this);

    // Signal and Object Editor
    signalEditorTool_ = new SignalEditorTool(this);

    // Map //
    //
    new MapTool(this);

	// OpenScenario //
	//
	oscEditorTool_ = new OpenScenarioEditorTool(this);

    // Default //
    //
    defaultEditor->activateEditor();
}

/*! Resends a toolAction with the last EditorId and ToolId.
*/
void
ToolManager::resendCurrentTool(ProjectWidget *project)
{
	currentProject_ = project;
	lastToolAction_ = getProjectEditingState(project);
    if (lastToolAction_)
    {
		ribbon_->setCurrentIndex(lastToolAction_->getEditorId());

		emit(pressButton(lastToolAction_->getToolId()));
		emit(toolAction(lastToolAction_));
    }
}


void
ToolManager::enableOSCEditorToolButton(bool state)
{
    oscEditorTool_->enableGraphEdit(state);
}

void
ToolManager::activateSignalSelection(bool state)
{
    signalEditorTool_->signalSelection(state);
}

void
ToolManager::activateOSCObjectSelection(bool state)
{
	oscEditorTool_->objectSelection(state);
}

void
ToolManager::setPushButtonColor(const QString &name, QColor color)
{
	oscEditorTool_->setButtonColor(name, color);
}

void
ToolManager::addProjectEditingState(ProjectWidget *project)
{
	if (!editingStates_.contains(project))
	{
		QList<ToolAction *> toolList;
		lastToolAction_ = new ToolAction(ODD::ERL, ODD::TRL_SELECT);
		toolList.append(lastToolAction_);
		editingStates_.insert(project, toolList);
	}
}

void
ToolManager::setProjectEditingState(ProjectWidget *project, ToolAction *toolAction)
{
	QMap<ProjectWidget *, QList<ToolAction *>>::iterator it = editingStates_.find(project);
	if (it != editingStates_.end())
	{
		QList<ToolAction *> *toolList = &it.value();
		if (toolList->contains(toolAction))
		{
			return;
		}

		int editorId = toolAction->getEditorId();
		foreach(ToolAction *tool, *toolList)
		{
			if (tool->getEditorId() == editorId)
			{
				toolList->removeOne(tool);
				toolList->prepend(toolAction);
				return;
			}
		}
		toolList->prepend(toolAction);
	}
	else
	{
		addProjectEditingState(project);
	}
}

// Gets the toolId of Editor editorId and inserts the related ToolAction as first element of the list
//
ODD::ToolId 
ToolManager::getProjectEditingState(ProjectWidget *project, ODD::EditorId editorId)
{
	QMap<ProjectWidget *, QList<ToolAction *>>::iterator it = editingStates_.find(project);
	if (it != editingStates_.end())
	{
		QList<ToolAction *> *toolList = &it.value();
		foreach(ToolAction *tool, *toolList)
		{
			if (tool->getEditorId() == editorId)
			{
				toolList->removeOne(tool);
				toolList->prepend(tool);
				return tool->getToolId();
			}
		}
	}

	return ODD::TNO_TOOL;
}


ToolAction *
ToolManager::getProjectEditingState(ProjectWidget *project)
{
	QMap<ProjectWidget *, QList<ToolAction *>>::iterator it = editingStates_.find(project);
	if (it != editingStates_.end())
	{
		return it.value().first();
	}

	return NULL;
}

//################//
// SLOTS          //
//################//

void
ToolManager::toolActionSlot(ToolAction *action)
{
	static ODD::EditorId lastEditor = ODD::ENO_EDITOR;

	ODD::EditorId editorId = action->getEditorId();
	ODD::ToolId toolId = action->getToolId();
	if (editorId != ODD::ENO_EDITOR)
	{
		if (currentProject_ && (editorId != lastEditor))
		{
			ODD::ToolId lastToolId = getProjectEditingState(currentProject_, editorId);
			if (lastToolId != ODD::TNO_TOOL)
			{
				emit(pressButton(toolId));
			}
			else
			{
				lastToolAction_ = new ToolAction(editorId, toolId);
				setProjectEditingState(currentProject_, lastToolAction_);
			}
		}
		else if (toolId != ODD::TNO_TOOL)
		{
			lastToolAction_ = new ToolAction(editorId, toolId);
			if (currentProject_)
			{
				setProjectEditingState(currentProject_, lastToolAction_);
			}
		}
		lastEditor = editorId;
	}
	emit(toolAction(action));
}
