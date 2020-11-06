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
    toolGroup->addButton(toolButton, ODD::TEL_SELECT); // button, id
    toolButton->setChecked(true);

    //	toolButton = new QPushButton(tr("Move Section"));
    //	toolButton->setCheckable(true);
    //	toolLayout->addWidget(toolButton, ++row, 0);
    //	toolGroup->addButton(toolButton, ODD::TEL_MOVE); // button, id

    toolButton = new QPushButton(tr("Add Section"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TEL_ADD); // button, id

    toolButton = new QPushButton(tr("Del Section"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TEL_DEL); // button, id

    toolButton = new QPushButton(tr("Smooth Roads"));
    toolButton->setCheckable(false);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TEL_SMOOTH); // button, id

    QLabel *radiusLabel = new QLabel("Smooth Radius:");
    radiusEdit_ = new QDoubleSpinBox();
    radiusEdit_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    radiusEdit_->setRange(0.01, 1000000.0);
    radiusEdit_->setValue(900.0);
    radiusEdit_->setMinimumWidth(80.0);
    radiusEdit_->setMaximumWidth(80.0);

    toolLayout->addWidget(radiusLabel, ++row, 0);
    toolLayout->addWidget(radiusEdit_, ++row, 0);

    connect(radiusEdit_, SIGNAL(editingFinished()), this, SLOT(setRadius()));

    QLabel *heightLabel = new QLabel("Height:");
    heightEdit_ = new QDoubleSpinBox();
    heightEdit_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    heightEdit_->setRange(-1000.0, 100000.0);
    heightEdit_->setValue(00.0);
    heightEdit_->setMinimumWidth(80.0);
    heightEdit_->setMaximumWidth(80.0);

    toolLayout->addWidget(heightLabel, ++row, 0);
    toolLayout->addWidget(heightEdit_, ++row, 0);

    connect(heightEdit_, SIGNAL(editingFinished()), this, SLOT(setHeight()));

    QLabel *iHeightLabel = new QLabel("Inc. Height:");
    iHeightEdit_ = new QDoubleSpinBox();
    iHeightEdit_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    iHeightEdit_->setRange(-1000.0, 100000.0);
    iHeightEdit_->setValue(00.0);
    iHeightEdit_->setMinimumWidth(80.0);
    iHeightEdit_->setMaximumWidth(80.0);

    toolLayout->addWidget(iHeightLabel, row - 1, 1);
    toolLayout->addWidget(iHeightEdit_, row, 1);

    connect(iHeightEdit_, SIGNAL(editingFinished()), this, SLOT(setIHeight()));

    // Finish Layout //
    //
    toolLayout->setRowStretch(++row, 1); // row 3 fills the rest of the availlable space
    toolLayout->setColumnStretch(1, 1); // column 1 fills the rest of the availlable space

    // Widget/Layout //
    //
    ToolWidget *toolWidget = new ToolWidget();
    toolWidget->setLayout(toolLayout);
    toolManager_->addToolBoxWidget(toolWidget, tr("Elevation Editor"));
    connect(toolWidget, SIGNAL(activated()), this, SLOT(activateEditor()));

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

/*! \brief Gets called when this widget (tab) has been activated.
*
* Sends a ToolAction with the current ToolId and Radius.
*/
void
ElevationEditorTool::activateEditor()
{
    ElevationEditorToolAction *action = new ElevationEditorToolAction(ODD::TEL_SELECT, toolId_, radiusEdit_->value(), heightEdit_->value(), iHeightEdit_->value());
    emit toolAction(action);
    delete action;
}

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
ElevationEditorTool::handleToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;
    ElevationEditorToolAction *action;

    // Set a tool //
    //
	switch (toolId_)
	{
	case ODD::TEL_SMOOTH:
		action = new ElevationEditorToolAction(ODD::TEL_SELECT, toolId_, radiusEdit_->value(), heightEdit_->value(), iHeightEdit_->value());
		emit toolAction(action);
		delete action;

		toolId_ = ODD::TEL_SELECT;
		action = new ElevationEditorToolAction(ODD::TEL_SELECT, toolId_, radiusEdit_->value(), heightEdit_->value(), iHeightEdit_->value());
		break;
	default:
		action = new ElevationEditorToolAction(toolId_, ODD::TNO_TOOL, radiusEdit_->value(), heightEdit_->value(), iHeightEdit_->value());
	}


    emit toolAction(action);
    delete action;
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
* Sends a ToolAction with the current ToolId and Radius.
*/
void
ElevationEditorTool::setRadius()
{
    ElevationEditorToolAction *action = new ElevationEditorToolAction(ODD::TEL_SELECT, ODD::TEL_RADIUS, radiusEdit_->value(), heightEdit_->value(), iHeightEdit_->value());
    emit toolAction(action);
    delete action;
}

/*! \brief Gets called when the height has been changed.
*
* Sends a ToolAction with the current ToolId and Height.
*/
void
ElevationEditorTool::setHeight()
{
    ElevationEditorToolAction *action = new ElevationEditorToolAction(ODD::TEL_SELECT, ODD::TEL_HEIGHT, radiusEdit_->value(), heightEdit_->value(), iHeightEdit_->value());
    emit toolAction(action);
    delete action;
}

/*! \brief Gets called when the height has been changed.
*
* Sends a ToolAction with the current ToolId and iHeight.
*/
void
ElevationEditorTool::setIHeight()
{
    ElevationEditorToolAction *action = new ElevationEditorToolAction(ODD::TEL_SELECT, ODD::TEL_IHEIGHT, radiusEdit_->value(), heightEdit_->value(), iHeightEdit_->value());
    emit toolAction(action);
    delete action;
    iHeightEdit_->setValue(0.0);
}

/*! \brief Gets called when the radius has been changed.
*
* Sends a ToolAction with the current ToolId and Height.
*/
void
ElevationEditorTool::setRRadius()
{
    ElevationEditorToolAction *action = new ElevationEditorToolAction(ODD::TEL_SELECT, ODD::TEL_RADIUS, ui->radiusEdit->value(), ui->heightEdit->value(), ui->iHeightEdit->value(), ui->startEdit->value());
    emit toolAction(action);
 //   delete action;

    if (toolId_ != ODD::TEL_SELECT)
    {
        ribbonToolGroup_->button(toolId_)->click();
    }
}

/*! \brief Gets called when the height has been changed.
*
* Sends a ToolAction with the current ToolId and Height.
*/
void
ElevationEditorTool::setRHeight()
{

	ElevationEditorToolAction *action = new ElevationEditorToolAction(ODD::TEL_SELECT, ODD::TEL_HEIGHT, ui->radiusEdit->value(), ui->heightEdit->value(), ui->iHeightEdit->value(), ui->startEdit->value());
	emit toolAction(action);
	//   delete action;

	ui->heightEdit->blockSignals(true);
	ui->heightEdit->setValue(0.0);
    ui->heightEdit->clearFocus();
	ui->heightEdit->blockSignals(false);

    if (toolId_ != ODD::TEL_SELECT)
    {
        ribbonToolGroup_->button(toolId_)->click();
    } 
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
		ElevationEditorToolAction *action = new ElevationEditorToolAction(ODD::TEL_SELECT, ODD::TEL_IHEIGHT, ui->radiusEdit->value(), ui->heightEdit->value(), ui->iHeightEdit->value(), ui->startEdit->value());
		emit toolAction(action);
		//   delete action;
		ui->iHeightEdit->blockSignals(true);
		ui->iHeightEdit->setValue(0.0);
        ui->heightEdit->clearFocus();
		ui->iHeightEdit->blockSignals(false);

        if (toolId_ != ODD::TEL_SELECT)
        {
            ribbonToolGroup_->button(toolId_)->click();
        }
	}
}

/*! \brief Gets called when the start of the section has been changed.
*
* Sends a ToolAction with the current ToolId and Height.
*/
void
ElevationEditorTool::setSectionStart()
{

	if (fabs(ui->startEdit->value()) > NUMERICAL_ZERO3)
	{
		ElevationEditorToolAction *action = new ElevationEditorToolAction(ODD::TEL_SELECT, ODD::TEL_MOVE, ui->radiusEdit->value(), ui->heightEdit->value(), ui->iHeightEdit->value(), ui->startEdit->value());
		emit toolAction(action);
		//   delete action;

		ui->startEdit->blockSignals(true);
		ui->startEdit->setValue(0.0);
        ui->heightEdit->clearFocus();
		ui->startEdit->blockSignals(false);

        if (toolId_ != ODD::TEL_SELECT)
        {
            ribbonToolGroup_->button(toolId_)->click();
        }
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