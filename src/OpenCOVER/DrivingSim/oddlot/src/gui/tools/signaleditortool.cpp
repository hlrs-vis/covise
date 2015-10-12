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
#include "src/util/roadtypecombobox.h"

#include "toolmanager.hpp"
#include "toolwidget.hpp"

#include "src/mainwindow.hpp"

// Qt //
//
#include <QGridLayout>
#include <QPushButton>
#include <QButtonGroup>
#include <QGroupBox>

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
    : Tool(toolManager)
    , toolId_(ODD::TSG_SELECT)
    , active_(false)
{
    // Connect //
    //
    connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));

    initToolWidget();
}

void
SignalEditorTool::initToolWidget()
{

    QGridLayout *toolLayout = new QGridLayout;

    // ButtonGroup //
    //
    // A button group so only one button can be checked at a time
    QButtonGroup *toolGroup = new QButtonGroup;
    connect(toolGroup, SIGNAL(buttonClicked(int)), this, SLOT(handleToolClick(int)));

    // Tools //
    //
    QPushButton *toolButton;
    int row = -1; // button row

    toolButton = new QPushButton(tr("Select"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TSG_SELECT); // button, id
    toolButton->setChecked(true);

    toolButton = new QPushButton(tr("New Signal"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TSG_SIGNAL); // button, id

    toolButton = new QPushButton(tr("New Object"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TSG_OBJECT); // button, id

    toolButton = new QPushButton(tr("New Bridge"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TSG_BRIDGE); // button, id

    toolButton = new QPushButton(tr("New Controller"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TSG_CONTROLLER); // button, id

    toolButton = new QPushButton(tr("Add Controlled Signal"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TSG_ADD_CONTROL_ENTRY); // button, id

    toolButton = new QPushButton(tr("Remove Controlled Signal"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TSG_REMOVE_CONTROL_ENTRY); // button, id

    toolButton = new QPushButton(tr("Delete"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TSG_DEL); // button, id

    // Finish Layout //
    //
    toolLayout->setRowStretch(++row, 1); // row 3 fills the rest of the availlable space
    toolLayout->setColumnStretch(1, 1); // column 1 fills the rest of the availlable space

    // Widget/Layout //
    //
    ToolWidget *toolWidget = new ToolWidget();
    toolWidget->setLayout(toolLayout);
    toolManager_->addToolBoxWidget(toolWidget, tr("Signal and Object Editor"));
    connect(toolWidget, SIGNAL(activated()), this, SLOT(activateEditor()));
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

/*! \brief Gets called when this widget (tab) is activated.
*
*/
void
SignalEditorTool::activateEditor()
{
    SignalEditorToolAction *action = new SignalEditorToolAction(toolId_);
    emit toolAction(action);
    delete action;
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
    delete action;

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
