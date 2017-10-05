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
#include "src/util/roadtypecombobox.h"

#include "toolmanager.hpp"
#include "toolwidget.hpp"

#include "src/mainwindow.hpp"

#include "ui_TypeRibbon.h"

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

// Utils //
//
#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"

//################//
//                //
// TypeEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
TypeEditorTool::TypeEditorTool(ToolManager *toolManager)
    : Tool(toolManager)
    , toolId_(ODD::TRT_MOVE)
    , roadType_(TypeSection::RTP_UNKNOWN)
    , // default, note: should be the same as in the comboBox
    active_(false)
{
    // Connect //
    //
    connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));

    // List //
    //
    roadTypes_.insert(tr("unknown"), TypeSection::RTP_UNKNOWN);
    roadTypes_.insert(tr("pedestrian"), TypeSection::RTP_PEDESTRIAN);
    roadTypes_.insert(tr("lowSpeed"), TypeSection::RTP_LOWSPEED);
    roadTypes_.insert(tr("town"), TypeSection::RTP_TOWN);
    roadTypes_.insert(tr("rural"), TypeSection::RTP_RURAL);
    roadTypes_.insert(tr("motorway"), TypeSection::RTP_MOTORWAY);

    initToolWidget();
}

void
TypeEditorTool::initToolWidget()
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
    toolGroup->addButton(toolButton, ODD::TRT_SELECT); // button, id
    toolButton->setChecked(true);

    toolButton = new QPushButton(tr("Move Section"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TRT_MOVE); // button, id

    toolButton = new QPushButton(tr("Add Section"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TRT_ADD); // button, id

    toolButton = new QPushButton(tr("Del Section"));
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TRT_DEL); // button, id

    RoadTypeComboBox *roadTypeBox_ = new RoadTypeComboBox();
    selectGroupBox_ = new QGroupBox(tr("Add Settings"));
    selectGroupBox_->setLayout(roadTypeBox_->getGridLayout());
    selectGroupBox_->setEnabled(false);
    selectGroupBox_->setVisible(false);

    toolLayout->addWidget(selectGroupBox_, ++row, 0);

    connect(roadTypeBox_->getComboBox(), SIGNAL(currentIndexChanged(int)), this, SLOT(handleRoadTypeSelection(int)));

    // Finish Layout //
    //
    toolLayout->setRowStretch(++row, 1); // row 3 fills the rest of the availlable space
    toolLayout->setColumnStretch(1, 1); // column 1 fills the rest of the availlable space

    // Widget/Layout //
    //
    ToolWidget *toolWidget = new ToolWidget();
    toolWidget->setLayout(toolLayout);
    toolManager_->addToolBoxWidget(toolWidget, tr("Road Type Editor"));
    connect(toolWidget, SIGNAL(activated()), this, SLOT(activateEditor()));

    
    ToolWidget *ribbonWidget = new ToolWidget();
    ribbonWidget->setObjectName("Ribbon");
    //ribbonWidget->
    Ui::TypeRibbon *ui = new Ui::TypeRibbon();
    ui->setupUi(ribbonWidget);
    ribbonWidget->setObjectName("Ribbon");
    
    QButtonGroup *ribbonToolGroup = new QButtonGroup;
    connect(ribbonToolGroup, SIGNAL(buttonClicked(int)), this, SLOT(handleToolClick(int)));

    // move also selects ribbonToolGroup->addButton(ui->typeSelect, ODD::TRT_SELECT);
    ribbonToolGroup->addButton(ui->select, ODD::TRT_MOVE);
    ui->select->setChecked(true);
    ribbonToolGroup->addButton(ui->typeAdd, ODD::TRT_ADD);
    ribbonToolGroup->addButton(ui->typeDelete, ODD::TRT_DEL);
    
    RoadTypeComboBox *ribbonRoadTypeBox_ = new RoadTypeComboBox();
    selectGroupBox2_ = new QGroupBox(tr("Add Settings"));
    selectGroupBox2_->setLayout(ribbonRoadTypeBox_->getGridLayout());
    selectGroupBox2_->setEnabled(false);
    selectGroupBox2_->setVisible(false);
    ui->horizontalLayout->insertWidget(2,selectGroupBox2_);

    connect(ribbonRoadTypeBox_->getComboBox(), SIGNAL(currentIndexChanged(int)), this, SLOT(handleRoadTypeSelection(int)));

    toolManager_->addRibbonWidget(ribbonWidget, tr("Road Type"));
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateEditor()));
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

    // Enable/Disable //
    //
    roadTypeToolButton_->setEnabled(active_);
}

/*! \brief Gets called when this widget (tab) is activated.
*
*/
void
TypeEditorTool::activateEditor()
{
    TypeEditorToolAction *action = new TypeEditorToolAction(toolId_);
    emit toolAction(action);
    delete action;
}

/*! \brief
*
*/
void
TypeEditorTool::handleToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Settings GUI //
    //
    if (toolId_ == ODD::TRT_ADD)
    {
        selectGroupBox_->setEnabled(true);
        selectGroupBox2_->setEnabled(true);
        selectGroupBox_->setVisible(true);
        selectGroupBox2_->setVisible(true);
    }
    else
    {
        selectGroupBox_->setEnabled(false);
        selectGroupBox2_->setEnabled(false);
        selectGroupBox_->setVisible(false);
        selectGroupBox2_->setVisible(false);
    }

    // Set a tool //
    //
    TypeEditorToolAction *action = new TypeEditorToolAction(toolId_);
    emit toolAction(action);
    delete action;
}

/*! \brief
*
*/
void
TypeEditorTool::handleRoadTypeToolButton()
{
    handleRoadTypeAction(roadTypeToolButtonMenu_->defaultAction());
}

/*! \brief
*
*/
void
TypeEditorTool::handleRoadTypeAction(QAction *action)
{
    roadTypeToolButtonMenu_->setDefaultAction(action);
    roadTypeToolButton_->setIcon(roadTypeToolButtonMenu_->defaultAction()->icon());

    // Apply RoadType //
    //
    TypeSection::RoadType roadType = roadTypes_.value(action->data().value<QString>());
    TypeEditorToolAction *toolAct = new TypeEditorToolAction(ODD::TNO_TOOL, roadType, true);
    emit toolAction(toolAct);
    delete toolAct;
}

/*! \brief
*
*/
void
TypeEditorTool::handleRoadTypeSelection(int id)
{
    if (id == 0)
    {
        roadType_ = TypeSection::RTP_UNKNOWN;
    }
    else if (id == 1)
    {
        roadType_ = TypeSection::RTP_RURAL;
    }
    else if (id == 2)
    {
        roadType_ = TypeSection::RTP_MOTORWAY;
    }
    else if (id == 3)
    {
        roadType_ = TypeSection::RTP_TOWN;
    }
    else if (id == 4)
    {
        roadType_ = TypeSection::RTP_LOWSPEED;
    }
    else
    {
        roadType_ = TypeSection::RTP_PEDESTRIAN;
    }

    // Set RoadType //
    //
    TypeEditorToolAction *action = new TypeEditorToolAction(toolId_, roadType_);
    emit toolAction(action);
    delete action;
}

//################//
//                //
// TypeEditorToolAction //
//                //
//################//

TypeEditorToolAction::TypeEditorToolAction(ODD::ToolId toolId, TypeSection::RoadType roadType, bool applyRoadType)
    : ToolAction(ODD::ERT, toolId)
    , roadType_(roadType)
    , applyRoadType_(applyRoadType)
{
}
