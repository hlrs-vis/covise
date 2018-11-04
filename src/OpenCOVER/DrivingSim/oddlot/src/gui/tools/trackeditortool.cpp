/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   06.04.2010
**
**************************************************************************/

#include "trackeditortool.hpp"

#include "toolmanager.hpp"
#include "toolwidget.hpp"

#include "ui_TrackRibbon.h"

// Data //
//
#include "src/data/prototypemanager.hpp"

// Qt //
//
#include <QGridLayout>
#include <QPushButton>
#include <QButtonGroup>
#include <QGroupBox>
//#include <QComboBox>
#include <QToolBar>
#include <QToolButton>
#include <QMenu>
#include <QLabel>
#include <QComboBox>
#include <QListWidget>
#include <QListWidgetItem>

#define ColumnCount 2

//################//
//                //
// TrackEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
TrackEditorTool::TrackEditorTool(PrototypeManager *prototypeManager, ToolManager *toolManager)
    : Tool(toolManager)
    , prototypeManager_(prototypeManager)
    , toolId_(ODD::TTE_ROAD_MOVE_ROTATE)
    , currentRoadSystemPrototype_(NULL)
{
    // Connect //
    //
    connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));

    // Tool Bar //
    //
    initToolBar();
    initToolWidget();
}

void
TrackEditorTool::initToolWidget()
{
    //	prototypeListWidget->setMaximumWidth(156);

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

    QLabel *trackLabel = new QLabel(tr("Track Tools"));
    toolLayout->addWidget(trackLabel, ++row, 0, 1, ColumnCount);

    //	toolButton = new QPushButton("Select");
    //	toolButton->setCheckable(true);
    //	toolLayout->addWidget(toolButton, ++row, 0, 1, ColumnCount);
    //	toolGroup->addButton(toolButton, ODD::TTE_SELECT); // button, id

    toolButton = new QPushButton("Move Rotate");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TTE_MOVE_ROTATE); // button, id

    toolButton = new QPushButton("Add Line");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TTE_ADD_LINE); // button, id

    toolButton = new QPushButton("Add Curve");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, row, 1);
    toolGroup->addButton(toolButton, ODD::TTE_ADD_CURVE); // button, id
    //	toolButton->setChecked(true);

    toolButton = new QPushButton("Add Prototype");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0, 1, ColumnCount);
    toolGroup->addButton(toolButton, ODD::TTE_ADD); // button, id

    toolButton = new QPushButton("Delete Tracks");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0, 1, ColumnCount);
    toolGroup->addButton(toolButton, ODD::TTE_DELETE); // button, id

    toolButton = new QPushButton("Split Track");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0, 1, ColumnCount);
    toolGroup->addButton(toolButton, ODD::TTE_TRACK_SPLIT); // button, id

    QFrame *line = new QFrame();
    line->setFrameStyle(QFrame::HLine | QFrame::Sunken);
    line->setLineWidth(1);
    toolLayout->addWidget(line, ++row, 0, 1, ColumnCount);

    QLabel *roadLabel = new QLabel(tr("Road Tools"));
    toolLayout->addWidget(roadLabel, ++row, 0, 1, ColumnCount);

    toolButton = new QPushButton("Modify");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TTE_ROAD_MOVE_ROTATE); // button, id

    toolButton = new QPushButton("New Road");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0, 1, ColumnCount);
    toolGroup->addButton(toolButton, ODD::TTE_ROAD_NEW); // button, id

    toolButton = new QPushButton("Add Prototype");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0, 1, ColumnCount);
    toolGroup->addButton(toolButton, ODD::TTE_ROADSYSTEM_ADD); // button, id

    toolButton = new QPushButton("Delete Road");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0, 1, ColumnCount);
    toolGroup->addButton(toolButton, ODD::TTE_ROAD_DELETE); // button, id

    toolButton = new QPushButton("Split Road");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0, 1, ColumnCount);
    toolGroup->addButton(toolButton, ODD::TTE_ROAD_SPLIT); // button, id

    toolButton = new QPushButton("Merge");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TTE_ROAD_MERGE); // button, id

    toolButton = new QPushButton("Snap to");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, row, 1);
    toolGroup->addButton(toolButton, ODD::TTE_ROAD_SNAP); // button, id

	toolButton = new QPushButton("New Circle");
	toolButton->setCheckable(true);
	toolLayout->addWidget(toolButton, ++row, 0, 1, ColumnCount);
	toolGroup->addButton(toolButton, ODD::TTE_ROAD_CIRCLE); // button, id

    line = new QFrame();
    line->setFrameStyle(QFrame::HLine | QFrame::Sunken);
    line->setLineWidth(1);
    toolLayout->addWidget(line, ++row, 0, 1, ColumnCount);

    toolButton = new QPushButton("Split Track and Road");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0, 1, ColumnCount);
    toolGroup->addButton(toolButton, ODD::TTE_TRACK_ROAD_SPLIT); // button, id

    // Tiles
    //

    line = new QFrame();
    line->setFrameStyle(QFrame::HLine | QFrame::Sunken);
    line->setLineWidth(1);
    toolLayout->addWidget(line, ++row, 0, 1, ColumnCount);

    roadLabel = new QLabel(tr("Tile Tools"));
    toolLayout->addWidget(roadLabel, ++row, 0, 1, ColumnCount);

    toolButton = new QPushButton("Move Tile");
    toolButton->setCheckable(true);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TTE_TILE_MOVE); // button, id

    toolButton = new QPushButton("New Tile");
    toolButton->setCheckable(false);
    toolLayout->addWidget(toolButton, ++row, 0, 1, ColumnCount);
    toolGroup->addButton(toolButton, ODD::TTE_TILE_NEW); // button, id

    toolButton = new QPushButton("Delete Tile");
    toolButton->setCheckable(false);
    toolLayout->addWidget(toolButton, ++row, 0, 1, ColumnCount);
    toolGroup->addButton(toolButton, ODD::TTE_TILE_DELETE); // button, id

    // Prototypes //
    //
    //
    QGridLayout *groupBoxLayout;
    QLabel *label;
    QComboBox *comboBox;

    // Track Prototypes //
    //
    //
    groupBoxLayout = new QGridLayout;

    trackPrototypesGroupBox_ = new QGroupBox(tr("Track Settings"));
    trackPrototypesGroupBox_->setLayout(groupBoxLayout);
    trackPrototypesGroupBox_->setEnabled(false);
    toolLayout->addWidget(trackPrototypesGroupBox_, ++row, 0, 1, ColumnCount);

    label = new QLabel(tr("Track Prototype"));
    comboBox = new QComboBox;
    foreach (const PrototypeContainer<RSystemElementRoad *> *container, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_TrackPrototype))
    {
        comboBox->addItem(container->getPrototypeIcon(), container->getPrototypeName());
    }
    comboBox->setIconSize(QSize(16, 16));
    connect(comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(handleTrackSelection(int)));
    comboBox->setCurrentIndex(0); // this doesn't trigger an event...
    handleTrackSelection(0); // ... so do it yourself
    groupBoxLayout->addWidget(label, 0, 0);
    groupBoxLayout->addWidget(comboBox, 1, 0);

    // Section Prototypes //
    //
    //
    groupBoxLayout = new QGridLayout;

    sectionPrototypesGroupBox_ = new QGroupBox(tr("Section Settings"));
    sectionPrototypesGroupBox_->setLayout(groupBoxLayout);
    sectionPrototypesGroupBox_->setEnabled(false);
    toolLayout->addWidget(sectionPrototypesGroupBox_, ++row, 0, 1, ColumnCount);

    int groupBoxLayoutRow = 0;

    // LaneSection Prototype //
    //
    label = new QLabel(tr("LaneSection Prototype"));
    comboBox = new QComboBox;
    foreach (const PrototypeContainer<RSystemElementRoad *> *container, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_LaneSectionPrototype))
    {
        comboBox->addItem(container->getPrototypeIcon(), container->getPrototypeName());
    }
    comboBox->setIconSize(QSize(16, 16));
    connect(comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(handleLaneSectionSelection(int)));
    comboBox->setCurrentIndex(0);
    handleLaneSectionSelection(0);
    groupBoxLayout->addWidget(label, groupBoxLayoutRow++, 0);
    groupBoxLayout->addWidget(comboBox, groupBoxLayoutRow++, 0);

    // RoadType Prototype //
    //
    label = new QLabel(tr("RoadType Prototype"));
    comboBox = new QComboBox;
    foreach (const PrototypeContainer<RSystemElementRoad *> *container, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_RoadTypePrototype))
    {
        comboBox->addItem(container->getPrototypeIcon(), container->getPrototypeName());
    }
    comboBox->setIconSize(QSize(16, 16));
    connect(comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(handleRoadTypeSelection(int)));
    comboBox->setCurrentIndex(0);
    handleRoadTypeSelection(0);
    groupBoxLayout->addWidget(label, groupBoxLayoutRow++, 0);
    groupBoxLayout->addWidget(comboBox, groupBoxLayoutRow++, 0);

    // Elevation Prototype //
    //
    label = new QLabel(tr("Elevation Prototype"));
    comboBox = new QComboBox;
    foreach (const PrototypeContainer<RSystemElementRoad *> *container, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_ElevationPrototype))
    {
        comboBox->addItem(container->getPrototypeIcon(), container->getPrototypeName());
    }
    comboBox->setIconSize(QSize(16, 16));
    connect(comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(handleElevationSelection(int)));
    comboBox->setCurrentIndex(0);
    handleElevationSelection(0);
    groupBoxLayout->addWidget(label, groupBoxLayoutRow++, 0);
    groupBoxLayout->addWidget(comboBox, groupBoxLayoutRow++, 0);

    // Superelevation Prototype //
    //
    label = new QLabel(tr("Superelevation Prototype"));
    comboBox = new QComboBox;
    foreach (const PrototypeContainer<RSystemElementRoad *> *container, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_SuperelevationPrototype))
    {
        comboBox->addItem(container->getPrototypeIcon(), container->getPrototypeName());
    }
    comboBox->setIconSize(QSize(16, 16));
    connect(comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(handleSuperelevationSelection(int)));
    comboBox->setCurrentIndex(0);
    handleSuperelevationSelection(0);
    groupBoxLayout->addWidget(label, groupBoxLayoutRow++, 0);
    groupBoxLayout->addWidget(comboBox, groupBoxLayoutRow++, 0);

    // Crossfall Prototype //
    //
    label = new QLabel(tr("Crossfall Prototype"));
    comboBox = new QComboBox;
    foreach (const PrototypeContainer<RSystemElementRoad *> *container, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_CrossfallPrototype))
    {
        comboBox->addItem(container->getPrototypeIcon(), container->getPrototypeName());
    }
    comboBox->setIconSize(QSize(16, 16));
    connect(comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(handleCrossfallSelection(int)));
    comboBox->setCurrentIndex(0);
    handleCrossfallSelection(0);
    groupBoxLayout->addWidget(label, groupBoxLayoutRow++, 0);
    groupBoxLayout->addWidget(comboBox, groupBoxLayoutRow++, 0);

	// RoadShape Prototype //
	//
	label = new QLabel(tr("RoadShape Prototype"));
	comboBox = new QComboBox;
	foreach(const PrototypeContainer<RSystemElementRoad *> *container, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_RoadShapePrototype))
	{
		comboBox->addItem(container->getPrototypeIcon(), container->getPrototypeName());
	}
	comboBox->setIconSize(QSize(16, 16));
	connect(comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(handleShapeSelection(int)));
	comboBox->setCurrentIndex(0);
	handleShapeSelection(0);
	groupBoxLayout->addWidget(label, groupBoxLayoutRow++, 0);
	groupBoxLayout->addWidget(comboBox, groupBoxLayoutRow++, 0);

    // RoadSystem Prototypes //
    //
    //
    groupBoxLayout = new QGridLayout;

    roadSystemPrototypesGroupBox_ = new QGroupBox(tr("RoadSystem Settings"));
    roadSystemPrototypesGroupBox_->setLayout(groupBoxLayout);
    roadSystemPrototypesGroupBox_->setEnabled(false);
    toolLayout->addWidget(roadSystemPrototypesGroupBox_, ++row, 0, 1, ColumnCount);

    label = new QLabel(tr("RoadSystem Prototype"));
    comboBox = new QComboBox;
    foreach (const PrototypeContainer<RoadSystem *> *container, prototypeManager_->getRoadSystemPrototypes())
    {
        comboBox->addItem(container->getPrototypeIcon(), container->getPrototypeName());
    }
    comboBox->setIconSize(QSize(16, 16));
    connect(comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(handleRoadSystemSelection(int)));
    comboBox->setCurrentIndex(0);
    handleRoadSystemSelection(0);
    groupBoxLayout->addWidget(label, 0, 0);
    groupBoxLayout->addWidget(comboBox, 1, 0);

    // Finish Layout //
    //
    toolLayout->setRowStretch(++row, 1); // row x fills the rest of the availlable space
    toolLayout->setColumnStretch(ColumnCount, 1); // column 2 fills the rest of the availlable space

    // Widget/Layout //
    //
    ToolWidget *toolWidget = new ToolWidget();
    toolWidget->setLayout(toolLayout);
    toolManager_->addToolBoxWidget(toolWidget, tr("Track Editor"));
    connect(toolWidget, SIGNAL(activated()), this, SLOT(activateEditor()));
    
    ToolWidget *ribbonWidget = new ToolWidget();
    //ribbonWidget->
    Ui::TrackRibbon *ui = new Ui::TrackRibbon();
    ui->setupUi(ribbonWidget);
    
    QButtonGroup *ribbonToolGroup = new QButtonGroup;
    connect(ribbonToolGroup, SIGNAL(buttonClicked(int)), this, SLOT(handleToolClick(int)));
    ribbonToolGroup->addButton(ui->trackModify, ODD::TTE_MOVE_ROTATE);
    ribbonToolGroup->addButton(ui->trackNewLine, ODD::TTE_ADD_LINE);
    ribbonToolGroup->addButton(ui->trackNewCurve, ODD::TTE_ADD_CURVE);
	ribbonToolGroup->addButton(ui->trackNewPoly, ODD::TTE_ADD_POLY);
    ribbonToolGroup->addButton(ui->trackAddPrototype, ODD::TTE_ADD);
    ribbonToolGroup->addButton(ui->trackDelete, ODD::TTE_DELETE);
    ribbonToolGroup->addButton(ui->trackSplit, ODD::TTE_TRACK_SPLIT);

    ribbonToolGroup->addButton(ui->roadModify, ODD::TTE_ROAD_MOVE_ROTATE);
    ribbonToolGroup->addButton(ui->roadNew, ODD::TTE_ROAD_NEW);
    ribbonToolGroup->addButton(ui->roadAddPrototype, ODD::TTE_ROADSYSTEM_ADD);
    ribbonToolGroup->addButton(ui->roadDelete, ODD::TTE_ROAD_DELETE);
    ribbonToolGroup->addButton(ui->roadSplit, ODD::TTE_ROAD_SPLIT);
    ribbonToolGroup->addButton(ui->roadMerge, ODD::TTE_ROAD_MERGE);
    ribbonToolGroup->addButton(ui->roadSnap, ODD::TTE_ROAD_SNAP);
	ribbonToolGroup->addButton(ui->roadCut, ODD::TTE_TRACK_ROAD_SPLIT);
	ribbonToolGroup->addButton(ui->roadCircle, ODD::TTE_ROAD_CIRCLE);
    
    ribbonToolGroup->addButton(ui->tileMove, ODD::TTE_TILE_MOVE);
    ribbonToolGroup->addButton(ui->tileNew, ODD::TTE_TILE_NEW);
    ribbonToolGroup->addButton(ui->tileDelete, ODD::TTE_TILE_DELETE);

    toolManager_->addRibbonWidget(ribbonWidget, tr("Track"));
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateEditor()));


    // Default Settings //
    //
    sectionPrototypesGroupBox_->setVisible(false);
    trackPrototypesGroupBox_->setVisible(false);
    roadSystemPrototypesGroupBox_->setVisible(false);
}

void
TrackEditorTool::initToolBar()
{
    // no tool bar for me
}

//################//
// SLOTS          //
//################//

/*! \brief Creates a ToolAction and sends it.
*
*/
void
TrackEditorTool::sendToolAction()
{
    TrackEditorToolAction *action = new TrackEditorToolAction(toolId_, currentPrototypes_, currentRoadSystemPrototype_);
    emit toolAction(action);
    delete action;
}

/*! \brief Gets called when this widget (tab) has been activated.
*
*/
void
TrackEditorTool::activateEditor()
{
    // Send //
    //
    if (toolId_ != ODD::TTE_TILE_NEW)
    {
        sendToolAction();
    }
}

/*! \brief Gets called when a tool button has been selected.
*
*/
void
TrackEditorTool::handleToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Settings GUI //
    //
    sectionPrototypesGroupBox_->setVisible(false);
    trackPrototypesGroupBox_->setVisible(false);
    roadSystemPrototypesGroupBox_->setVisible(false);

    if (toolId_ == ODD::TTE_ADD)
    {
        sectionPrototypesGroupBox_->setVisible(true);
        sectionPrototypesGroupBox_->setEnabled(true);

        trackPrototypesGroupBox_->setVisible(true);
        trackPrototypesGroupBox_->setEnabled(true);
    }
    else if (toolId_ == ODD::TTE_ROAD_NEW)
    {
        sectionPrototypesGroupBox_->setVisible(true);
        sectionPrototypesGroupBox_->setEnabled(true);
    }
    else if (toolId_ == ODD::TTE_ADD_LINE)
    {
        sectionPrototypesGroupBox_->setVisible(true);
        sectionPrototypesGroupBox_->setEnabled(true);
    }
    else if (toolId_ == ODD::TTE_ADD_CURVE)
    {
        sectionPrototypesGroupBox_->setVisible(true);
        sectionPrototypesGroupBox_->setEnabled(true);
    }
	else if (toolId_ == ODD::TTE_ADD_POLY)
	{
		sectionPrototypesGroupBox_->setVisible(true);
		sectionPrototypesGroupBox_->setEnabled(true);
	}
    else if (toolId_ == ODD::TTE_ROADSYSTEM_ADD)
    {
        roadSystemPrototypesGroupBox_->setVisible(true);
        roadSystemPrototypesGroupBox_->setEnabled(true);
    }

    // Send //
    //
    sendToolAction();
}

/*! \brief Gets called when a prototype has been selected.
*
*/
void
TrackEditorTool::handleRoadTypeSelection(int id)
{
    currentPrototypes_.insert(PrototypeManager::PTP_RoadTypePrototype, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_RoadTypePrototype).at(id)->getPrototype());
    sendToolAction();
}

/*! \brief Gets called when a prototype has been selected.
*
*/
void
TrackEditorTool::handleTrackSelection(int id)
{
    currentPrototypes_.insert(PrototypeManager::PTP_TrackPrototype, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_TrackPrototype).at(id)->getPrototype());
    sendToolAction();
}

/*! \brief Gets called when a prototype has been selected.
*
*/
void
TrackEditorTool::handleElevationSelection(int id)
{
    currentPrototypes_.insert(PrototypeManager::PTP_ElevationPrototype, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_ElevationPrototype).at(id)->getPrototype());
    sendToolAction();
}

/*! \brief Gets called when a prototype has been selected.
*
*/
void
TrackEditorTool::handleSuperelevationSelection(int id)
{
    currentPrototypes_.insert(PrototypeManager::PTP_SuperelevationPrototype, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_SuperelevationPrototype).at(id)->getPrototype());
    sendToolAction();
}

/*! \brief Gets called when a prototype has been selected.
*
*/
void
TrackEditorTool::handleCrossfallSelection(int id)
{
    currentPrototypes_.insert(PrototypeManager::PTP_CrossfallPrototype, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_CrossfallPrototype).at(id)->getPrototype());
    sendToolAction();
}

/*! \brief Gets called when a prototype has been selected.
*
*/
void
TrackEditorTool::handleShapeSelection(int id)
{
	currentPrototypes_.insert(PrototypeManager::PTP_RoadShapePrototype, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_RoadShapePrototype).at(id)->getPrototype());
	sendToolAction();
}

/*! \brief Gets called when a prototype has been selected.
*
*/
void
TrackEditorTool::handleLaneSectionSelection(int id)
{
    currentPrototypes_.insert(PrototypeManager::PTP_LaneSectionPrototype, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_LaneSectionPrototype).at(id)->getPrototype());
    sendToolAction();
}

/*! \brief Gets called when a prototype has been selected.
*
*/
void
TrackEditorTool::handleRoadSystemSelection(int id)
{
    currentRoadSystemPrototype_ = prototypeManager_->getRoadSystemPrototypes().at(id)->getPrototype();
    sendToolAction();
}

//################//
//                //
// TrackEditorToolAction //
//                //
//################//

TrackEditorToolAction::TrackEditorToolAction(ODD::ToolId toolId, QMap<PrototypeManager::PrototypeType, RSystemElementRoad *> prototypes, RoadSystem *roadSystemPrototype)
    : ToolAction(ODD::ETE, toolId)
    , prototypes_(prototypes)
    , roadSystemPrototype_(roadSystemPrototype)
{
}
