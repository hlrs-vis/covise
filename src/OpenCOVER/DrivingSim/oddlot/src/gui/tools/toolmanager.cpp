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
ToolManager::initTools()
{
    // Tool Box //
    //
    toolBox_ = new QToolBox();
    toolBox_->setMinimumWidth(190);

    // Zoom //
    //
    zoomTool_ = new ZoomTool(this);

    // Selection //
    //
    selectionTool_ = new SelectionTool(this);

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
    new SignalEditorTool(this);

    // Map //
    //
    new MapTool(this);

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
