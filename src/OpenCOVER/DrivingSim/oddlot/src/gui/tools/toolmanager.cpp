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
#include "laneeditortool.hpp"
#include "junctioneditortool.hpp"
#include "signaleditortool.hpp"
#include "osceditortool.hpp"

#include "maptool.hpp"

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
ToolManager::resendCurrentTool()
{
    if (lastToolAction_)
    {
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

//################//
// SLOTS          //
//################//

void
ToolManager::toolActionSlot(ToolAction *action)
{
    if (action->getEditorId() != ODD::ENO_EDITOR && action->getToolId() != ODD::TNO_TOOL)
    {
        lastToolAction_ = new ToolAction(action->getEditorId(), action->getToolId());
    }
    emit(toolAction(action));
}
