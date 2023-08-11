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

#include "signaleditortool.hpp"

#include "toolmanager.hpp"
#include "toolwidget.hpp"

#include "src/mainwindow.hpp"

// Qt //
//
#include <QGridLayout>
#include <QPushButton>
#include <QButtonGroup>
#include <QGroupBox>
#include <QtGlobal>

// Utils //
//
#include "src/util/odd.hpp"

//################//
//                //
// SignalEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
SignalEditorTool::SignalEditorTool(ToolManager *toolManager)
    : EditorTool(toolManager)
    , toolId_(ODD::TSG_SELECT)
    , active_(false)
    , ui(new Ui::SignalRibbon)
{
    // Connect //
    //
    connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));

    initToolWidget();
}

void
SignalEditorTool::initToolWidget()
{

    ToolWidget *ribbonWidget = new ToolWidget();
    ui->setupUi(ribbonWidget);


    ribbonToolGroup_ = new QButtonGroup(toolManager_);
#if (QT_VERSION >= QT_VERSION_CHECK(6, 0, 0))
    connect(ribbonToolGroup_, SIGNAL(idClicked(int)), this, SLOT(handleToolClick(int)));
#else
    connect(ribbonToolGroup_, SIGNAL(buttonClicked(int)), this, SLOT(handleToolClick(int)));
#endif

    // move also selects ribbonToolGroup->addButton(ui->typeSelect, ODD::TRT_SELECT);
    ribbonToolGroup_->addButton(ui->newController, ODD::TSG_CONTROLLER);
    ribbonToolGroup_->addButton(ui->addSignal, ODD::TSG_ADD_CONTROL_ENTRY);
    ribbonToolGroup_->addButton(ui->removeSignal, ODD::TSG_REMOVE_CONTROL_ENTRY);
    ribbonToolGroup_->addButton(ui->select, ODD::TSG_SELECT);
    ribbonToolGroup_->addButton(ui->invisibleButton, ODD::TSG_NONE);
    ui->invisibleButton->hide();

    toolManager_->addRibbonWidget(ribbonWidget, tr("Signals and Objects"), ODD::ESG);
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateRibbonEditor()));
}

void
SignalEditorTool::signalSelection(bool state)
{
    ui->invisibleButton->setChecked(!state);
}

//################//
// SLOTS          //
//################//

/*! \brief.
*/
void
SignalEditorTool::activateProject(bool active)
{
    active_ = active;
}


/*! \brief Is called by the toolmanager to initialize the UI */
/* UI sets the values of the current project */
void
SignalEditorTool::activateRibbonEditor()
{
    ToolAction *action = toolManager_->getLastToolAction(ODD::ESG);

    QAbstractButton *currentButton = ribbonToolGroup_->button(action->getToolId());
    if (currentButton)
    {
        currentButton->click();
    }

}

/*! \brief
*
*/
void
SignalEditorTool::handleToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Set a tool //
    //
    SignalEditorToolAction *action = new SignalEditorToolAction(toolId_);
    emit toolAction(action);
    //   delete action;

}


//################//
//                //
// SignalEditorToolAction //
//                //
//################//

SignalEditorToolAction::SignalEditorToolAction(ODD::ToolId toolId)
    : ToolAction(ODD::ESG, toolId)
{
}
