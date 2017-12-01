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

#include "src/mainwindow.hpp"


// Qt //
//
#include <QGridLayout>
#include <QPushButton>
#include <QButtonGroup>
#include <QGroupBox>
#include <QComboBox>
#include <QToolBar>
#include <QToolButton>
#include <QMenu>
#include <QLabel>
#include <QDoubleSpinBox>

//################//
//                //
// CrossfallEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
CrossfallEditorTool::CrossfallEditorTool(ToolManager *toolManager)
    : Tool(toolManager)
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
    toolGroup->addButton(toolButton, ODD::TCF_SELECT); // button, id
    toolButton->setChecked(true);

    //	toolButton = new QPushButton(tr("Move Section"));
    //	toolButton->setCheckable(true);
    //	toolLayout->addWidget(toolButton, ++row, 0);
    //	toolGroup->addButton(toolButton, ODD::TCF_MOVE); // button, id

    toolButton = new QPushButton(tr("Add Section"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TCF_ADD); // button, id

    toolButton = new QPushButton(tr("Del Section"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TCF_DEL); // button, id

    QLabel *radiusLabel = new QLabel("Smooth Radius:");
    radiusEdit_ = new QDoubleSpinBox();
    radiusEdit_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    radiusEdit_->setRange(0.01, 1000000.0);
    radiusEdit_->setValue(2000.0);
    radiusEdit_->setMinimumWidth(80.0);
    radiusEdit_->setMaximumWidth(80.0);

    toolLayout->addWidget(radiusLabel, ++row, 0);
    toolLayout->addWidget(radiusEdit_, ++row, 0);

    connect(radiusEdit_, SIGNAL(editingFinished()), this, SLOT(setRadius()));

    // Finish Layout //
    //
    toolLayout->setRowStretch(++row, 1); // row 3 fills the rest of the availlable space
    toolLayout->setColumnStretch(1, 1); // column 1 fills the rest of the availlable space

    // Widget/Layout //
    //
    ToolWidget *toolWidget = new ToolWidget();
    toolWidget->setLayout(toolLayout);
    toolManager_->addToolBoxWidget(toolWidget, tr("Crossfall Editor"));
    connect(toolWidget, SIGNAL(activated()), this, SLOT(activateEditor()));

    // Ribbon //
    //

    ToolWidget *ribbonWidget = new ToolWidget();
    //ribbonWidget->
    ui_ = new Ui::CrossfallRibbon();
    ui_->setupUi(ribbonWidget);
    
    QButtonGroup *ribbonToolGroup = new QButtonGroup;
    connect(ribbonToolGroup, SIGNAL(buttonClicked(int)), this, SLOT(handleRibbonToolClick(int)));
    
    
    ribbonToolGroup->addButton(ui_->select, ODD::TCF_SELECT);
    ribbonToolGroup->addButton(ui_->crossfallAdd, ODD::TCF_ADD);
    ribbonToolGroup->addButton(ui_->crossfallDelete, ODD::TCF_DEL);
    //ribbonToolGroup->addButton(ui->elevationSmooth, ODD::TSE_SMOOTH);
    
    connect(ui_->radiusEdit, SIGNAL(editingFinished()), this, SLOT(setRibbonRadius()));

    toolManager_->addRibbonWidget(ribbonWidget, tr("Crossfall"));
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

/*! \brief Gets called when this widget (tab) has been activated.
*
* Sends a ToolAction with the current ToolId and Radius.
*/
void
CrossfallEditorTool::activateEditor()
{
    CrossfallEditorToolAction *action = new CrossfallEditorToolAction(toolId_, radiusEdit_->value());
    emit toolAction(action);
    delete action;
}

void
CrossfallEditorTool::activateRibbonEditor()
{
	CrossfallEditorToolAction *action = new CrossfallEditorToolAction(toolId_, ui_->radiusEdit->value());
	emit toolAction(action);
	delete action;
}

/*! \brief Gets called when a tool has been selected.
*
* Sends a ToolAction with the current ToolId and Radius.
*/
void
CrossfallEditorTool::handleToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Set a tool //
    //
    CrossfallEditorToolAction *action = new CrossfallEditorToolAction(toolId_, radiusEdit_->value());
    emit toolAction(action);
    delete action;
}

void
CrossfallEditorTool::handleRibbonToolClick(int id)
{
	toolId_ = (ODD::ToolId)id;

	// Set a tool //
	//
	CrossfallEditorToolAction *action = new CrossfallEditorToolAction(toolId_, ui_->radiusEdit->value());
	emit toolAction(action);
	delete action;
}

/*! \brief Gets called when the radius has been changed.
*
* Sends a ToolAction with the current ToolId and Radius.
*/
void
CrossfallEditorTool::setRadius()
{
    CrossfallEditorToolAction *action = new CrossfallEditorToolAction(ODD::TCF_SELECT, ui_->radiusEdit->value());
    emit toolAction(action);
    delete action;
}

void
CrossfallEditorTool::setRibbonRadius()
{
	CrossfallEditorToolAction *action = new CrossfallEditorToolAction(ODD::TCF_SELECT, ui_->radiusEdit->value());
	emit toolAction(action);
	delete action;
}

//################//
//                //
// CrossfallEditorToolAction //
//                //
//################//

CrossfallEditorToolAction::CrossfallEditorToolAction(ODD::ToolId toolId, double radius)
    : ToolAction(ODD::ECF, toolId)
    , radius_(radius)
{
}

void
CrossfallEditorToolAction::setRadius(double radius)
{
    radius_ = radius;
}
