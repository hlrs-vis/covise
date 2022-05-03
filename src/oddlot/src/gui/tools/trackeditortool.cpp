/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   06.04.2010
 **
 **************************************************************************/

#include "trackeditortool.hpp"

#include "toolmanager.hpp"
#include "toolwidget.hpp"

#include "ui_TrackRibbon.h"


 //################//
 //                //
 // TrackEditorTool //
 //                //
 //################//

 /*! \todo Ownership/destructor
 */
TrackEditorTool::TrackEditorTool(ToolManager *toolManager)
    : EditorTool(toolManager)
    , toolId_(ODD::TTE_ROAD_MOVE_ROTATE)
{
    // Connect //
    //
    connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));

    // Tool Bar //
    //
    initToolBar();
    initToolWidget();
}

void
TrackEditorTool::initToolWidget()
{
    ToolWidget *ribbonWidget = new ToolWidget();
    Ui::TrackRibbon *ui = new Ui::TrackRibbon();
    ui->setupUi(ribbonWidget);

    ribbonToolGroup_ = new QButtonGroup(toolManager_);
#if (QT_VERSION >= QT_VERSION_CHECK(6, 0, 0))
    connect(ribbonToolGroup_, SIGNAL(idClicked(int)), this, SLOT(handleToolClick(int)));
#else
    connect(ribbonToolGroup_, SIGNAL(buttonClicked(int)), this, SLOT(handleToolClick(int)));
#endif
    ribbonToolGroup_->addButton(ui->trackModify, ODD::TTE_MOVE_ROTATE);
    ribbonToolGroup_->addButton(ui->trackAppend, ODD::TTE_ADD);
    ribbonToolGroup_->addButton(ui->trackAddPrototype, ODD::TTE_ADD_PROTO);
    ribbonToolGroup_->addButton(ui->trackDelete, ODD::TTE_DELETE);
    ribbonToolGroup_->addButton(ui->trackSplit, ODD::TTE_TRACK_SPLIT);

    ribbonToolGroup_->addButton(ui->roadModify, ODD::TTE_ROAD_MOVE_ROTATE);
    ribbonToolGroup_->addButton(ui->roadNew, ODD::TTE_ROAD_NEW);
    ribbonToolGroup_->addButton(ui->roadAddPrototype, ODD::TTE_ROADSYSTEM_ADD);
    ribbonToolGroup_->addButton(ui->roadDelete, ODD::TTE_ROAD_DELETE);
    ribbonToolGroup_->addButton(ui->roadSplit, ODD::TTE_ROAD_SPLIT);
    ribbonToolGroup_->addButton(ui->roadMerge, ODD::TTE_ROAD_MERGE);
    ribbonToolGroup_->addButton(ui->roadSnap, ODD::TTE_ROAD_SNAP);
    ribbonToolGroup_->addButton(ui->roadCut, ODD::TTE_TRACK_ROAD_SPLIT);
    ribbonToolGroup_->addButton(ui->roadCircle, ODD::TTE_ROAD_CIRCLE);

    ribbonToolGroup_->addButton(ui->tileMove, ODD::TTE_TILE_MOVE);
    ribbonToolGroup_->addButton(ui->tileNew, ODD::TTE_TILE_NEW);
    ribbonToolGroup_->addButton(ui->tileDelete, ODD::TTE_TILE_DELETE);

    toolManager_->addRibbonWidget(ribbonWidget, tr("Track"), ODD::ETE);
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateRibbonEditor()));

}

void
TrackEditorTool::initToolBar()
{
    // no tool bar for me
}

//################//
// SLOTS          //
//################//

/*! \brief Creates a ToolAction and sends it.
*
*/
void
TrackEditorTool::sendToolAction()
{
    TrackEditorToolAction *action = new TrackEditorToolAction(toolId_);
    emit toolAction(action);
    //   delete action;
}

/*! \brief Is called by the toolmanager to initialize the UI */
/* UI sets the values of the current project */
void
TrackEditorTool::activateRibbonEditor()
{
    ToolAction *action = toolManager_->getLastToolAction(ODD::ETE);

    ribbonToolGroup_->button(action->getToolId())->click();

}

/*! \brief Gets called when a tool button has been selected.
*
*/
void
TrackEditorTool::handleToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Send //
    //
    sendToolAction();
}


//################//
//                //
// TrackEditorToolAction //
//                //
//################//

TrackEditorToolAction::TrackEditorToolAction(ODD::ToolId toolId)
    : ToolAction(ODD::ETE, toolId)
{
}
