/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   21.06.2010
 **
 **************************************************************************/

#include "elevationeditortool.hpp"

#include "toolmanager.hpp"
#include "toolwidget.hpp"

#include <cmath>
 // Qt //
 //
#include <QButtonGroup>

//################//
//                //
// ElevationEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
ElevationEditorTool::ElevationEditorTool(ToolManager *toolManager)
    : EditorTool(toolManager)
    , toolId_(ODD::TEL_SELECT)
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
ElevationEditorTool::initToolWidget()
{

    // Ribbon //
    //

    ToolWidget *ribbonWidget = new ToolWidget();
    //ribbonWidget->
    ui = new Ui::ElevationRibbon();
    ui->setupUi(ribbonWidget);

    ribbonToolGroup_ = new QButtonGroup(toolManager_);
    connect(ribbonToolGroup_, SIGNAL(buttonClicked(int)), this, SLOT(handleRibbonToolClick(int)));

    ribbonToolGroup_->addButton(ui->select, ODD::TEL_SELECT);
    ribbonToolGroup_->addButton(ui->elevationAdd, ODD::TEL_ADD);
    ribbonToolGroup_->addButton(ui->elevationDelete, ODD::TEL_DEL);
    ribbonToolGroup_->addButton(ui->elevationSmooth, ODD::TEL_SMOOTH);
    ribbonToolGroup_->addButton(ui->slopeEdit, ODD::TEL_SLOPE);

    connect(ui->heightEdit, SIGNAL(editingFinished()), this, SLOT(setRHeight()));
    connect(ui->iHeightEdit, SIGNAL(editingFinished()), this, SLOT(setRIHeight()));
    connect(ui->radiusEdit, SIGNAL(editingFinished()), this, SLOT(setRRadius()));
    connect(ui->startEdit, SIGNAL(editingFinished()), this, SLOT(setSectionStart()));

    toolManager_->addRibbonWidget(ribbonWidget, tr("Elevation"), ODD::EEL);
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateRibbonEditor()));
}

void
ElevationEditorTool::initToolBar()
{
    // no toolbar for me //
}

//################//
// SLOTS          //
//################//

/*! \brief Is called by the toolmanager to initialize the UI */
/* UI sets the values of the current project */
void
ElevationEditorTool::activateRibbonEditor()
{
    ToolAction *action = toolManager_->getLastToolAction(ODD::EEL);
    ElevationEditorToolAction *elevationEditorToolAction = dynamic_cast<ElevationEditorToolAction *>(action);

    if (elevationEditorToolAction)
    {
        if (elevationEditorToolAction->getRadius() != ui->radiusEdit->value())
        {
            ui->radiusEdit->blockSignals(true);
            ui->radiusEdit->setValue(elevationEditorToolAction->getRadius());
            ui->radiusEdit->blockSignals(false);
        }
    }

    ribbonToolGroup_->button(action->getToolId())->click();

}


/*! \brief Gets called when a tool has been selected.
*
* Sends a ToolAction with the current ToolId and Radius.
*/
void
ElevationEditorTool::handleRibbonToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    ElevationEditorToolAction *action = new ElevationEditorToolAction(toolId_, ODD::TNO_TOOL, ui->radiusEdit->value(), ui->heightEdit->value(), ui->iHeightEdit->value(), ui->startEdit->value());
    emit toolAction(action);

}

/*! \brief Gets called when the radius has been changed.
*
* Sends a ToolAction with the current ToolId and Height.
*/
void
ElevationEditorTool::setRRadius()
{
    ToolAction *lastAction = toolManager_->getLastToolAction(ODD::EEL);
    ElevationEditorToolAction *elevationEditorToolAction = dynamic_cast<ElevationEditorToolAction *>(lastAction);
    ElevationEditorToolAction *action = new ElevationEditorToolAction(elevationEditorToolAction->getToolId(), ODD::TEL_RADIUS, ui->radiusEdit->value(), ui->heightEdit->value(), ui->iHeightEdit->value(), ui->startEdit->value());
    emit toolAction(action);
}

/*! \brief Gets called when the height has been changed.
*
* Sends a ToolAction with the current ToolId and Height.
*/
void
ElevationEditorTool::setRHeight()
{
    ToolAction *lastAction = toolManager_->getLastToolAction(ODD::EEL);
    ElevationEditorToolAction *elevationEditorToolAction = dynamic_cast<ElevationEditorToolAction *>(lastAction);
    ElevationEditorToolAction *action = new ElevationEditorToolAction(elevationEditorToolAction->getToolId(), ODD::TEL_HEIGHT, ui->radiusEdit->value(), ui->heightEdit->value(), ui->iHeightEdit->value(), ui->startEdit->value());
    emit toolAction(action);

    ui->heightEdit->blockSignals(true);
    ui->heightEdit->setValue(0.0);
    ui->heightEdit->clearFocus();
    ui->heightEdit->blockSignals(false);

}

/*! \brief Gets called when the height has been changed.
*
* Sends a ToolAction with the current ToolId and iHeight.
*/
void
ElevationEditorTool::setRIHeight()
{
    if (fabs(ui->iHeightEdit->value()) > NUMERICAL_ZERO3)
    {
        ToolAction *lastAction = toolManager_->getLastToolAction(ODD::EEL);
        ElevationEditorToolAction *elevationEditorToolAction = dynamic_cast<ElevationEditorToolAction *>(lastAction);
        ElevationEditorToolAction *action = new ElevationEditorToolAction(elevationEditorToolAction->getToolId(), ODD::TEL_IHEIGHT, ui->radiusEdit->value(), ui->heightEdit->value(), ui->iHeightEdit->value(), ui->startEdit->value());
        emit toolAction(action);

        ui->iHeightEdit->blockSignals(true);
        ui->iHeightEdit->setValue(0.0);
        ui->heightEdit->clearFocus();
        ui->iHeightEdit->blockSignals(false);
    }
}

/*! \brief Gets called when the start of the section has been changed.
*
* Sends a ToolAction with the current ToolId and Height.
*/
void
ElevationEditorTool::setSectionStart()
{

    if (std::abs(ui->startEdit->value()) > NUMERICAL_ZERO3)
    {
        ToolAction *lastAction = toolManager_->getLastToolAction(ODD::EEL);
        ElevationEditorToolAction *elevationEditorToolAction = dynamic_cast<ElevationEditorToolAction *>(lastAction);
        ElevationEditorToolAction *action = new ElevationEditorToolAction(elevationEditorToolAction->getToolId(), ODD::TEL_MOVE, ui->radiusEdit->value(), ui->heightEdit->value(), ui->iHeightEdit->value(), ui->startEdit->value());
        emit toolAction(action);

        ui->startEdit->blockSignals(true);
        ui->startEdit->setValue(0.0);
        ui->heightEdit->clearFocus();
        ui->startEdit->blockSignals(false);

    }
}


//################//
//                //
// ElevationEditorToolAction //
//                //
//################//

ElevationEditorToolAction::ElevationEditorToolAction(ODD::ToolId toolId, ODD::ToolId paramToolId, double radius, double height, double iHeight, double sectionStart)
    : ToolAction(ODD::EEL, toolId, paramToolId),
    radius_(radius),
    height_(height),
    iHeight_(iHeight),
    start_(sectionStart)
{
}

void
ElevationEditorToolAction::setRadius(double radius)
{
    radius_ = radius;
}

void
ElevationEditorToolAction::setHeight(double h)
{
    height_ = h;
}

void
ElevationEditorToolAction::setIHeight(double h)
{
    iHeight_ = h;
}

void
ElevationEditorToolAction::setSectionStart(double sectionStart)
{
    start_ = sectionStart;
}