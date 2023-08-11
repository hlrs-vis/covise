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

#include "typeeditortool.hpp"

#include "toolmanager.hpp"
#include "toolwidget.hpp"

#include "ui_TypeRibbon.h"

 // Qt //
 //
#include <QButtonGroup>
#include <QtGlobal>

//################//
//                //
// TypeEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
TypeEditorTool::TypeEditorTool(ToolManager *toolManager)
    : EditorTool(toolManager)
    , toolId_(ODD::TRT_MOVE)
    , active_(false)
{
    // Connect //
    //
    connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));

    initToolWidget();
}

void
TypeEditorTool::initToolWidget()
{

    ToolWidget *ribbonWidget = new ToolWidget();
    ribbonWidget->setObjectName("Ribbon");

    Ui::TypeRibbon *ui = new Ui::TypeRibbon();
    ui->setupUi(ribbonWidget);

    ribbonToolGroup_ = new QButtonGroup(toolManager_);
#if (QT_VERSION >= QT_VERSION_CHECK(6, 0, 0))
    connect(ribbonToolGroup_, SIGNAL(idClicked(int)), this, SLOT(handleToolClick(int)));
#else
    connect(ribbonToolGroup_, SIGNAL(buttonClicked(int)), this, SLOT(handleToolClick(int)));
#endif

    // move also selects ribbonToolGroup->addButton(ui->typeSelect, ODD::TRT_SELECT);
    ribbonToolGroup_->addButton(ui->select, ODD::TRT_MOVE);
    ui->select->setChecked(true);
    ribbonToolGroup_->addButton(ui->typeAdd, ODD::TRT_ADD);
    ribbonToolGroup_->addButton(ui->typeDelete, ODD::TRT_DEL);


    toolManager_->addRibbonWidget(ribbonWidget, tr("Road Type"), ODD::ERT);
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateRibbonEditor()));
}

//################//
// SLOTS          //
//################//

/*! \brief.
*/
void
TypeEditorTool::activateProject(bool active)
{
    active_ = active;
}

/*! \brief Is called by the toolmanager to initialize the UI */
/* UI sets the values of the current project */
void
TypeEditorTool::activateRibbonEditor()
{
    ToolAction *action = toolManager_->getLastToolAction(ODD::ERT);

    TypeEditorToolAction *typeEditorToolAction = dynamic_cast<TypeEditorToolAction *>(action);

    ribbonToolGroup_->button(action->getToolId())->click();
}

/*! \brief
*
*/
void
TypeEditorTool::handleToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;


    // Set a tool //
    //
    TypeEditorToolAction *action = new TypeEditorToolAction(toolId_);
    emit toolAction(action);
    //   delete action;
}

//################//
//                //
// TypeEditorToolAction //
//                //
//################//

TypeEditorToolAction::TypeEditorToolAction(ODD::ToolId toolId)
    : ToolAction(ODD::ERT, toolId)

{
}
