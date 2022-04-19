/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   14.07.2010
 **
 **************************************************************************/

#include "crossfalleditortool.hpp"

#include "toolmanager.hpp"
#include "toolwidget.hpp"


 // Qt //
 //
#include <QButtonGroup>

//################//
//                //
// CrossfallEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
CrossfallEditorTool::CrossfallEditorTool(ToolManager *toolManager)
    : EditorTool(toolManager)
    , toolId_(ODD::TCF_SELECT)
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
CrossfallEditorTool::initToolWidget()
{

    // Ribbon //
    //

    ToolWidget *ribbonWidget = new ToolWidget();
    ui_ = new Ui::CrossfallRibbon();
    ui_->setupUi(ribbonWidget);

    ribbonToolGroup_ = new QButtonGroup(toolManager_);
    connect(ribbonToolGroup_, SIGNAL(buttonClicked(int)), this, SLOT(handleRibbonToolClick(int)));


    ribbonToolGroup_->addButton(ui_->select, ODD::TCF_SELECT);
    ribbonToolGroup_->addButton(ui_->crossfallAdd, ODD::TCF_ADD);
    ribbonToolGroup_->addButton(ui_->crossfallDelete, ODD::TCF_DEL);
    //ribbonToolGroup->addButton(ui->elevationSmooth, ODD::TSE_SMOOTH);

    connect(ui_->radiusEdit, SIGNAL(editingFinished()), this, SLOT(setRibbonRadius()));

    toolManager_->addRibbonWidget(ribbonWidget, tr("Crossfall"), ODD::ECF);
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateRibbonEditor()));
}

void
CrossfallEditorTool::initToolBar()
{
    // no toolbar for me //
}

//################//
// SLOTS          //
//################//

/*! \brief Is called by the toolmanager to initialize the UI */
/* UI sets the values of the current project */
void
CrossfallEditorTool::activateRibbonEditor()
{
    ToolAction *action = toolManager_->getLastToolAction(ODD::ECF);
    CrossfallEditorToolAction *crossfallEditorToolAction = dynamic_cast<CrossfallEditorToolAction *>(action);

    if (crossfallEditorToolAction->getRadius() != ui_->radiusEdit->value())
    {
        ui_->radiusEdit->blockSignals(true);
        ui_->radiusEdit->setValue(crossfallEditorToolAction->getRadius());
        ui_->radiusEdit->blockSignals(false);
    }

    ribbonToolGroup_->button(action->getToolId())->click();
}

void
CrossfallEditorTool::handleRibbonToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Set a tool //
    //
    CrossfallEditorToolAction *action = new CrossfallEditorToolAction(toolId_, ODD::TNO_TOOL, ui_->radiusEdit->value());
    emit toolAction(action);
    // delete action;
}

void
CrossfallEditorTool::setRibbonRadius()
{
    ODD::ToolId toolId = (ODD::ToolId)ribbonToolGroup_->checkedId();

    CrossfallEditorToolAction *action = new CrossfallEditorToolAction(ODD::TCF_SELECT, ODD::TCF_RADIUS, ui_->radiusEdit->value());
    emit toolAction(action);
    // delete action;

    ribbonToolGroup_->button(toolId_)->click();
}

//################//
//                //
// CrossfallEditorToolAction //
//                //
//################//

CrossfallEditorToolAction::CrossfallEditorToolAction(ODD::ToolId toolId, ODD::ToolId paramToolId, double radius)
    : ToolAction(ODD::ECF, toolId, paramToolId)
    , radius_(radius)
{
}

void
CrossfallEditorToolAction::setRadius(double radius)
{
    radius_ = radius;
}
