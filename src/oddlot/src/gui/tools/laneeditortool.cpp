/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   10/18/2010
 **
 **************************************************************************/

#include "laneeditortool.hpp"

#include "toolmanager.hpp"
#include "toolwidget.hpp"

 // Qt //
 //
#include <QButtonGroup>

//################//
//                //
// LaneEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
LaneEditorTool::LaneEditorTool(ToolManager *toolManager)
    : EditorTool(toolManager)
    , toolId_(ODD::TLE_SELECT)
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
LaneEditorTool::initToolWidget()
{

    // Ribbon //
    //

    ToolWidget *ribbonWidget = new ToolWidget();
    //ribbonWidget->
    ui = new Ui::LaneRibbon();
    ui->setupUi(ribbonWidget);

    ribbonToolGroup_ = new QButtonGroup(toolManager_);
    connect(ribbonToolGroup_, SIGNAL(buttonClicked(int)), this, SLOT(handleRibbonToolClick(int)));


    ribbonToolGroup_->addButton(ui->select, ODD::TLE_SELECT);
    ribbonToolGroup_->addButton(ui->laneAdd, ODD::TLE_ADD);
    ribbonToolGroup_->addButton(ui->laneDelete, ODD::TLE_DEL);
    ribbonToolGroup_->addButton(ui->laneAddWidth, ODD::TLE_ADD_WIDTH);
    ribbonToolGroup_->addButton(ui->insertButton, ODD::TLE_INSERT);

    connect(ui->handleCheckBox, SIGNAL(stateChanged(int)), this, SLOT(onCheckBoxStateChanged(int)));
    connect(ui->widthEdit, SIGNAL(editingFinished()), this, SLOT(setRibbonWidth()));

    toolManager_->addRibbonWidget(ribbonWidget, tr("Lane"), ODD::ELN);
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateRibbonEditor()));
}

void
LaneEditorTool::initToolBar()
{
    // no toolbar for me //
}

//################//
// SLOTS          //
//################//

/*! \brief Is called by the toolmanager to initialize the UI */
/* UI sets the values of the current project */
void
LaneEditorTool::activateRibbonEditor()
{
    ToolAction *action = toolManager_->getLastToolAction(ODD::ELN);

    if (action->getToolId() == ODD::TLE_SELECT_ALL)
    {
        ribbonToolGroup_->button(ODD::TLE_SELECT)->click();
        ui->handleCheckBox->setCheckState(Qt::CheckState::Unchecked);
    }
    else if (action->getToolId() == ODD::TLE_SELECT_CONTROLS)
    {
        ribbonToolGroup_->button(ODD::TLE_SELECT)->click();
        ui->handleCheckBox->setCheckState(Qt::CheckState::Checked);
    }
    else
    {
        LaneEditorToolAction *laneEditorToolAction = dynamic_cast<LaneEditorToolAction *>(action);

        if (action->getToolId() == ODD::TLE_SET_WIDTH)
        {
            emit toolAction(laneEditorToolAction);
        }
        else
        {
            ribbonToolGroup_->button(action->getToolId())->click();
        }
    }

}

void
LaneEditorTool::handleRibbonToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Set a tool //
    //
    LaneEditorToolAction *action = new LaneEditorToolAction(toolId_, ui->widthEdit->value());
    emit toolAction(action);
    //   delete action;
}

void
LaneEditorTool::setRibbonWidth()
{
    LaneEditorToolAction *action = new LaneEditorToolAction(ODD::TLE_SET_WIDTH, ui->widthEdit->value());
    emit toolAction(action);
    //  delete action;

    QWidget *focusWidget = QApplication::focusWidget();
    if (focusWidget)
    {
        focusWidget->clearFocus();
    }
    ui->widthEdit->setValue(0.0);

    ribbonToolGroup_->button(toolId_)->click();
}

void
LaneEditorTool::onCheckBoxStateChanged(int state)
{
    if (state == Qt::CheckState::Unchecked)
    {
        toolId_ = ODD::TLE_SELECT_ALL;

    }
    else if (state == Qt::CheckState::Checked)
    {
        toolId_ = ODD::TLE_SELECT_CONTROLS;
    }

    LaneEditorToolAction *laneEditorToolAction = new LaneEditorToolAction(toolId_, ui->widthEdit->value());
    emit toolAction(laneEditorToolAction);
}

//################//
//                //
// LaneEditorToolAction //
//                //
//################//

LaneEditorToolAction::LaneEditorToolAction(ODD::ToolId toolId, double width)
    : ToolAction(ODD::ELN, toolId),
    width_(width)
{
}
