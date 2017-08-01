/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/2/2010
**
**************************************************************************/

#include "roadlinkeditortool.hpp"

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
// RoadLinkEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
RoadLinkEditorTool::RoadLinkEditorTool(ToolManager *toolManager)
    : Tool(toolManager)
    , toolId_(ODD::TRL_SELECT)
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
RoadLinkEditorTool::initToolWidget()
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

    // Link Roads by Handles//
    //
    toolButton = new QPushButton(tr("Link Handles"));
    toolButton->setCheckable(false);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TRL_LINK); // button, id

    // Link Roads //
    //
    toolButton = new QPushButton(tr("Link Roads"));
    toolButton->setCheckable(false);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TRL_ROADLINK); // button, id

    // Threshold //
    //
    QLabel *thresholdLabel = new QLabel("Threshold:");
    thresholdEdit_ = new QDoubleSpinBox();
    thresholdEdit_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    thresholdEdit_->setRange(0.0, 1000.0);
    thresholdEdit_->setValue(10.0);
    thresholdEdit_->setMinimumWidth(80.0);
    thresholdEdit_->setMaximumWidth(80.0);

    toolLayout->addWidget(thresholdLabel, ++row, 0);
    toolLayout->addWidget(thresholdEdit_, ++row, 0);

    connect(thresholdEdit_, SIGNAL(editingFinished()), this, SLOT(setThreshold()));

    // Unlink Roads //
    //
    toolButton = new QPushButton(tr("Unlink Roads"));
    toolButton->setCheckable(false);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TRL_UNLINK); // button, id

    // Finish Layout //
    //
    toolLayout->setRowStretch(++row, 1); // last row fills the rest of the availlable space
    toolLayout->setColumnStretch(1, 1); // column 1 fills the rest of the availlable space

    // Ribbon //
    //
    ToolWidget *toolWidget = new ToolWidget();
    toolWidget->setLayout(toolLayout);
    toolManager_->addToolBoxWidget(toolWidget, tr("RoadLink Editor"));
    connect(toolWidget, SIGNAL(activated()), this, SLOT(activateEditor()));

    ToolWidget *ribbonWidget = new ToolWidget();
    //ribbonWidget->
    ui = new Ui::RoadLinkRibbon();
    ui->setupUi(ribbonWidget);
    
    QButtonGroup *ribbonToolGroup = new QButtonGroup;
    connect(ribbonToolGroup, SIGNAL(buttonClicked(int)), this, SLOT(handleRibbonToolClick(int)));
    
    ribbonToolGroup->addButton(ui->roadlUnlink, ODD::TRL_UNLINK);
    ribbonToolGroup->addButton(ui->roadLink, ODD::TRL_ROADLINK);
    ribbonToolGroup->addButton(ui->roadLinkHandles, ODD::TRL_LINK);
    
    connect(ui->thresholdSpinBox, SIGNAL(editingFinished()), this, SLOT(setRibbonThreshold()));

    toolManager_->addRibbonWidget(ribbonWidget, tr("Road Link"));
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateRibbonEditor()));
}

void
RoadLinkEditorTool::initToolBar()
{
    // no toolbar for me //
}

//################//
// SLOTS          //
//################//

/*! \brief Gets called when this widget (tab) has been activated.
*/
void
RoadLinkEditorTool::activateEditor()
{
    RoadLinkEditorToolAction *action = new RoadLinkEditorToolAction(toolId_, thresholdEdit_->value());
    emit toolAction(action);
    delete action;
}

/*! \brief Gets called when this widget (tab) has been activated.
*/
void
RoadLinkEditorTool::activateRibbonEditor()
{
    RoadLinkEditorToolAction *action = new RoadLinkEditorToolAction(toolId_, ui->thresholdSpinBox->value());
    emit toolAction(action);
    delete action;
}

/*! \brief Gets called when a tool has been selected.
*/
void
RoadLinkEditorTool::handleToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Set a tool //
    //
    RoadLinkEditorToolAction *action = new RoadLinkEditorToolAction(toolId_, thresholdEdit_->value());
    emit toolAction(action);
    delete action;
}

/*! \brief Gets called when a tool has been selected.
*/
void
RoadLinkEditorTool::handleRibbonToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Set a tool //
    //
    RoadLinkEditorToolAction *action = new RoadLinkEditorToolAction(toolId_, ui->thresholdSpinBox->value());
    emit toolAction(action);
    delete action;
}

/*! \brief Gets called when a tool has been selected.
*/
void
RoadLinkEditorTool::setThreshold()
{

    // Set a tool //
    //
    RoadLinkEditorToolAction *action = new RoadLinkEditorToolAction(ODD::TRL_SELECT, thresholdEdit_->value());
    emit toolAction(action);
    delete action;
}

/*! \brief Gets called when a tool has been selected.
*/
void
RoadLinkEditorTool::setRibbonThreshold()
{

    // Set a tool //
    //
    RoadLinkEditorToolAction *action = new RoadLinkEditorToolAction(ODD::TRL_SELECT, ui->thresholdSpinBox->value());
    emit toolAction(action);
    delete action;
}

//################//
//                //
// RoadLinkEditorToolAction //
//                //
//################//

RoadLinkEditorToolAction::RoadLinkEditorToolAction(ODD::ToolId toolId, double threshold)
    : ToolAction(ODD::ERL, toolId)
    , threshold_(threshold)
{
}

void RoadLinkEditorToolAction::setThreshold(double threshold)
{
    threshold_ = threshold;
}
