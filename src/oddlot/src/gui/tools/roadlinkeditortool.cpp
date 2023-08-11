/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   11/2/2010
 **
 **************************************************************************/

#include "roadlinkeditortool.hpp"

#include "toolmanager.hpp"
#include "toolwidget.hpp"

#include "src/mainwindow.hpp"


 // Qt //
 //

#include <QButtonGroup>
#include <QtGlobal>

//################//
//                //
// RoadLinkEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
RoadLinkEditorTool::RoadLinkEditorTool(ToolManager *toolManager)
    : EditorTool(toolManager)
    , toolId_(ODD::TRL_SELECT)
{
    // Connect emitted ToolActions to ToolManager //
    //
    connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));

    // Tool Bar //
    //
    initToolBar();
    initToolWidget();
}

void
RoadLinkEditorTool::initToolWidget()
{

    // Ribbon //
    //

    ToolWidget *ribbonWidget = new ToolWidget();

    ui = new Ui::RoadLinkRibbon();
    ui->setupUi(ribbonWidget);

    ribbonToolGroup_ = new QButtonGroup(toolManager_);
#if (QT_VERSION >= QT_VERSION_CHECK(6, 0, 0))
    connect(ribbonToolGroup_, SIGNAL(idClicked(int)), this, SLOT(handleRibbonToolClick(int)));
#else
    connect(ribbonToolGroup_, SIGNAL(buttonClicked(int)), this, SLOT(handleRibbonToolClick(int)));
#endif

    ribbonToolGroup_->addButton(ui->roadlUnlink, ODD::TRL_UNLINK);
    ribbonToolGroup_->addButton(ui->roadLink, ODD::TRL_ROADLINK);
    ribbonToolGroup_->addButton(ui->roadLinkHandles, ODD::TRL_LINK);
    ribbonToolGroup_->addButton(ui->select, ODD::TRL_SELECT);

    toolManager_->addRibbonWidget(ribbonWidget, tr("Road Link"), ODD::ERL);
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateRibbonEditor()));
}

void
RoadLinkEditorTool::initToolBar()
{
    // no toolbar for me //
}

//################//
// SLOTS          //
//################//

/*! \brief Is called by the toolmanager to initialize the UI */
/* UI sets the values of the current project */
void
RoadLinkEditorTool::activateRibbonEditor()
{
    ToolAction *action = toolManager_->getLastToolAction(ODD::ERL);

    ribbonToolGroup_->button(action->getToolId())->click();

}


/*! \brief Gets called when a tool has been selected.
*/
void
RoadLinkEditorTool::handleRibbonToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Set a tool //
    //
    RoadLinkEditorToolAction *action = new RoadLinkEditorToolAction(toolId_);
    emit toolAction(action);
    //   delete action;
}

/*! \brief Gets called when a tool has been selected.
*/
void
RoadLinkEditorTool::setThreshold()
{
    ODD::ToolId toolId_ = (ODD::ToolId)ribbonToolGroup_->checkedId();

    // Set a tool //
    //
    RoadLinkEditorToolAction *action = new RoadLinkEditorToolAction(toolId_);
    emit toolAction(action);
    delete action;
}

//################//
//                //
// RoadLinkEditorToolAction //
//                //
//################//

RoadLinkEditorToolAction::RoadLinkEditorToolAction(ODD::ToolId toolId, ODD::ToolId paramToolId)
    : ToolAction(ODD::ERL, toolId, paramToolId)
{
}

