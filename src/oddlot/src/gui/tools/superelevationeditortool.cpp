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

#include "superelevationeditortool.hpp"

#include "toolmanager.hpp"
#include "toolwidget.hpp"

 // Qt //
 //
#include <QButtonGroup>
#include <QtGlobal>


//################//
//                //
// SuperelevationEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
SuperelevationEditorTool::SuperelevationEditorTool(ToolManager *toolManager)
    : EditorTool(toolManager)
    , toolId_(ODD::TSE_SELECT)
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
SuperelevationEditorTool::initToolWidget()
{
    // Ribbon //
    //

    ToolWidget *ribbonWidget = new ToolWidget();
    //ribbonWidget->
    ui_ = new Ui::SuperelevationRibbon();
    ui_->setupUi(ribbonWidget);

    ribbonToolGroup_ = new QButtonGroup(toolManager_);
#if (QT_VERSION >= QT_VERSION_CHECK(6, 0, 0))
    connect(ribbonToolGroup_, SIGNAL(idClicked(int)), this, SLOT(handleRibbonToolClick(int)));
#else
    connect(ribbonToolGroup_, SIGNAL(buttonClicked(int)), this, SLOT(handleRibbonToolClick(int)));
#endif


    ribbonToolGroup_->addButton(ui_->select, ODD::TSE_SELECT);
    ribbonToolGroup_->addButton(ui_->elevationAdd, ODD::TSE_ADD);
    ribbonToolGroup_->addButton(ui_->elevationDelete, ODD::TSE_DEL);
    //ribbonToolGroup->addButton(ui->elevationSmooth, ODD::TSE_SMOOTH);

    connect(ui_->radiusEdit, SIGNAL(editingFinished()), this, SLOT(setRibbonRadius()));

    toolManager_->addRibbonWidget(ribbonWidget, tr("Superelevation"), ODD::ESE);
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateRibbonEditor()));
}

void
SuperelevationEditorTool::initToolBar()
{
    // no toolbar for me //
}

//################//
// SLOTS          //
//################//

/*! \brief Is called by the toolmanager to initialize the UI */
/* UI sets the values of the current project */
void
SuperelevationEditorTool::activateRibbonEditor()
{
    ToolAction *action = toolManager_->getLastToolAction(ODD::ESE);
    SuperelevationEditorToolAction *superelevationEditorToolAction = dynamic_cast<SuperelevationEditorToolAction *>(action);

    if (superelevationEditorToolAction->getRadius() != ui_->radiusEdit->value())
    {
        ui_->radiusEdit->blockSignals(true);
        ui_->radiusEdit->setValue(superelevationEditorToolAction->getRadius());
        ui_->radiusEdit->blockSignals(false);
    }

    ribbonToolGroup_->button(action->getToolId())->click();
}

void
SuperelevationEditorTool::handleRibbonToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Set a tool //
    //
    SuperelevationEditorToolAction *action = new SuperelevationEditorToolAction(toolId_, ODD::TNO_TOOL, ui_->radiusEdit->value());
    emit toolAction(action);
    // delete action;
}

void
SuperelevationEditorTool::setRibbonRadius()
{
    ODD::ToolId toolId = (ODD::ToolId)ribbonToolGroup_->checkedId();

    SuperelevationEditorToolAction *action = new SuperelevationEditorToolAction(ODD::TSE_SELECT, ODD::TSE_RADIUS, ui_->radiusEdit->value());
    emit toolAction(action);
    // delete action;

    ribbonToolGroup_->button(toolId_)->click();
}

//################//
//                //
// SuperelevationEditorToolAction //
//                //
//################//

SuperelevationEditorToolAction::SuperelevationEditorToolAction(ODD::ToolId toolId, ODD::ToolId paramToolId, double radius)
    : ToolAction(ODD::ESE, toolId, paramToolId)
    , radius_(radius)
{
}

void
SuperelevationEditorToolAction::setRadius(double radius)
{
    radius_ = radius;
}
