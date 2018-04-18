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

#include "shapeeditortool.hpp"

#include "toolmanager.hpp"
#include "toolwidget.hpp"

#include "src/mainwindow.hpp"


// Qt //
//
#include <QGridLayout>
#include <QPushButton>
#include <QButtonGroup>


//################//
//                //
// ShapeEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
ShapeEditorTool::ShapeEditorTool(ToolManager *toolManager)
    : Tool(toolManager)
    , toolId_(ODD::TRS_SELECT)
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
ShapeEditorTool::initToolWidget()
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
    toolGroup->addButton(toolButton, ODD::TRS_SELECT); // button, id
    toolButton->setChecked(true);


    toolButton = new QPushButton(tr("Add Section"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TRS_ADD); // button, id

    toolButton = new QPushButton(tr("Del Section"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TRS_DEL); // button, id

    // Finish Layout //
    //
    toolLayout->setRowStretch(++row, 1); // row 3 fills the rest of the availlable space
    toolLayout->setColumnStretch(1, 1); // column 1 fills the rest of the availlable space

    // Widget/Layout //
    //
    ToolWidget *toolWidget = new ToolWidget();
    toolWidget->setLayout(toolLayout);
    toolManager_->addToolBoxWidget(toolWidget, tr("RoadShape Editor"));
    connect(toolWidget, SIGNAL(activated()), this, SLOT(activateEditor()));

    // Ribbon //
    //

    ToolWidget *ribbonWidget = new ToolWidget();
    //ribbonWidget->
    ui_ = new Ui::ShapeRibbon();
    ui_->setupUi(ribbonWidget);
    
    QButtonGroup *ribbonToolGroup = new QButtonGroup;
    connect(ribbonToolGroup, SIGNAL(buttonClicked(int)), this, SLOT(handleRibbonToolClick(int)));
    
    
    ribbonToolGroup->addButton(ui_->select, ODD::TRS_SELECT);
    ribbonToolGroup->addButton(ui_->shapeAdd, ODD::TRS_ADD);
    ribbonToolGroup->addButton(ui_->shapeDelete, ODD::TRS_DEL);

    toolManager_->addRibbonWidget(ribbonWidget, tr("RoadShape"));
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateRibbonEditor()));
}

void
ShapeEditorTool::initToolBar()
{
    // no toolbar for me //
}

//################//
// SLOTS          //
//################//

/*! \brief Gets called when this widget (tab) has been activated.
*
* Sends a ToolAction with the current ToolId and Radius.
*/
void
ShapeEditorTool::activateEditor()
{
    ShapeEditorToolAction *action = new ShapeEditorToolAction(toolId_);
    emit toolAction(action);
    delete action;
}

void
ShapeEditorTool::activateRibbonEditor()
{
	ShapeEditorToolAction *action = new ShapeEditorToolAction(toolId_);
	emit toolAction(action);
	delete action;
}

/*! \brief Gets called when a tool has been selected.
*
* Sends a ToolAction with the current ToolId and Radius.
*/
void
ShapeEditorTool::handleToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Set a tool //
    //
    ShapeEditorToolAction *action = new ShapeEditorToolAction(toolId_);
    emit toolAction(action);
    delete action;
}

void
ShapeEditorTool::handleRibbonToolClick(int id)
{
	toolId_ = (ODD::ToolId)id;

	// Set a tool //
	//
	ShapeEditorToolAction *action = new ShapeEditorToolAction(toolId_);
	emit toolAction(action);
	delete action;
}


//################//
//                //
// ShapeEditorToolAction //
//                //
//################//

ShapeEditorToolAction::ShapeEditorToolAction(ODD::ToolId toolId)
    : ToolAction(ODD::ERS, toolId)
{
}

