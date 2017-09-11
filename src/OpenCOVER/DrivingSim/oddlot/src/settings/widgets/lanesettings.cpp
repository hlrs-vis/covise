/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/23/2010
**
**************************************************************************/

#include "lanesettings.hpp"
#include "ui_lanesettings.h"
#include "src/graph/profilegraphscene.hpp"
#include "src/graph/profilegraphview.hpp"
#include "src/graph/items/roadsystem/lanes/lanewidthroadsystemitem.hpp"
#include "src/graph/items/roadsystem/lanes/lanesectionwidthitem.hpp"
#include "src/graph/items/roadsystem/lanes/lanewidthmovehandle.hpp"

#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"

#include "src/settings/projectsettings.hpp"
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/commands/lanesectioncommands.hpp"

// Editor //
//
#include "src/graph/editors/laneeditor.hpp"

//################//
// CONSTRUCTOR    //
//################//

LaneSettings::LaneSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, Lane *lane)
    : SettingsElement(projectSettings, parentSettingsElement, lane)
    , ui(new Ui::LaneSettings)
    , lane_(lane)
    , init_(false)
    , roadSystemItemPolyGraph_(NULL)
    , insertWidthSectionHandle_(NULL)
    , lswItem_(NULL)
{
    ui->setupUi(this);

	activateWidthGroupBox(false);
	activateInsertGroupBox(false);
	connect(ui->insertPushButton, SIGNAL(clicked(bool)), this, SLOT(activateInsertGroupBox(bool)));
	connect(ui->editPushButton, SIGNAL(clicked(bool)), this, SLOT(activateWidthGroupBox(bool)));

    // List //
    //
    QStringList typeNames;
    typeNames << Lane::parseLaneTypeBack(Lane::LT_NONE)
              << Lane::parseLaneTypeBack(Lane::LT_DRIVING)
              << Lane::parseLaneTypeBack(Lane::LT_STOP)
              << Lane::parseLaneTypeBack(Lane::LT_SHOULDER)
              << Lane::parseLaneTypeBack(Lane::LT_BIKING)
              << Lane::parseLaneTypeBack(Lane::LT_SIDEWALK)
              << Lane::parseLaneTypeBack(Lane::LT_BORDER)
              << Lane::parseLaneTypeBack(Lane::LT_RESTRICTED)
              << Lane::parseLaneTypeBack(Lane::LT_PARKING)
              << Lane::parseLaneTypeBack(Lane::LT_MWYENTRY)
              << Lane::parseLaneTypeBack(Lane::LT_MWYEXIT)
              << Lane::parseLaneTypeBack(Lane::LT_SPECIAL1)
              << Lane::parseLaneTypeBack(Lane::LT_SPECIAL2)
              << Lane::parseLaneTypeBack(Lane::LT_SPECIAL3);
    ui->typeBox->addItems(typeNames);

    /*heightGraph_ = new ProfileGraph(projectSettings->getProjectWidget(), projectSettings->getProjectData());
	heightGraph_->setParent(ui->widthGroup);
	ui->horizontalLayout_3->insertWidget(0,heightGraph_);
	//heightGraph_->getView()->setDragMode(QGraphicsView::ScrollHandDrag);
	//QGraphicsScene* pScene = new NoDeselectScene(this); 
    //heightGraph_->getView()->setScene(pScene);
	heightGraph_->getScene()->doDeselect(false);// we don't want to deselect ourselves if we click on the background, otherwise we delete ourselves--> chrash and it would not be practical anyway
	*/

    heightGraph_ = projectSettings->getProjectWidget()->getHeightGraph();

    laneEditor_ = dynamic_cast<LaneEditor *>(projectSettings->getProjectWidget()->getProjectEditor());
    if (!laneEditor_)
    {
        return; // another editor is active
    }

    roadSystemItemPolyGraph_ = new LaneWidthRoadSystemItem(heightGraph_, projectSettings->getProjectData()->getRoadSystem());
    heightGraph_->getScene()->addItem(roadSystemItemPolyGraph_);

    // Section Handle //
    //
    //insertWidthSectionHandle_ = new SectionHandle(roadSystemItemPolyGraph_);
    //insertWidthSectionHandle_->hide();
    roadSystemItemPolyGraph_->setSettings(this);
    roadSystemItemPolyGraph_->setAcceptHoverEvents(true);
    // Activate Road in ProfileGraph //
    //
    lswItem_ = new LaneSectionWidthItem(roadSystemItemPolyGraph_, lane_);
    //selectedElevationRoadItems_.insert(road, roadItem);

    // Fit View //
    //
    QRectF boundingBox = lswItem_->boundingRect();
    if (boundingBox.width() < 15.0)
    {
        boundingBox.setWidth(15.0);
    }
    if (boundingBox.height() < 10.0)
    {
        boundingBox.setHeight(10.0);
    }

    heightGraph_->getView()->fitInView(boundingBox);
    heightGraph_->getView()->zoomOut(Qt::Horizontal | Qt::Vertical);

    //ui->horizontalLayout_3->insertWidget(0,heightGraph_);
    //heightGraph_->show();
    // Initial Values //
    //
    updateId();
    updateType();
    updateLevel();
    updatePredecessor();
    updateSuccessor();
    updateWidth();

    // Done //
    //
    init_ = true;
}

LaneSettings::~LaneSettings()
{
    //heightGraph_->getScene()->removeItem(roadSystemItemPolyGraph_);

    //heightGraph_->getScene()->clearSelection();
    //delete insertWidthSectionHandle_;

    if (lswItem_)
    {
//        delete lswItem_;
    }

    if (roadSystemItemPolyGraph_)
    {
        roadSystemItemPolyGraph_->registerForDeletion();
    }
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
LaneSettings::updateId()
{
    ui->idBox->setValue(lane_->getId());

    ui->newIdBox->setMinimum(lane_->getParentLaneSection()->getRightmostLaneId() - 1);
    ui->newIdBox->setMaximum(lane_->getParentLaneSection()->getLeftmostLaneId() + 1);
    ui->newIdBox->setValue(lane_->getId());
}

void
LaneSettings::updateType()
{
    ui->typeBox->setCurrentIndex(ui->typeBox->findText(Lane::parseLaneTypeBack(lane_->getLaneType())));
}

void
LaneSettings::updateLevel()
{
    ui->levelBox->setChecked(lane_->getLevel());
    if (lane_->getLevel())
    {
        ui->levelBox->setText(tr("true"));
    }
    else
    {
        ui->levelBox->setText(tr("false"));
    }
}

void
LaneSettings::updatePredecessor()
{
    ui->predecessorBox->setValue(lane_->getPredecessor());

    if (lane_->getPredecessor() == Lane::NOLANE)
    {
        ui->predecessorCheckBox->setChecked(false);
        ui->predecessorBox->setEnabled(false);
    }
    else
    {
        ui->predecessorCheckBox->setChecked(true);
        ui->predecessorBox->setEnabled(true);
    }
}

void
LaneSettings::updateSuccessor()
{
    ui->successorBox->setValue(lane_->getSuccessor());

    if (lane_->getSuccessor() == Lane::NOLANE)
    {
        ui->successorCheckBox->setChecked(false);
        ui->successorBox->setEnabled(false);
    }
    else
    {
        ui->successorCheckBox->setChecked(true);
        ui->successorBox->setEnabled(true);
    }
}

void
LaneSettings::updateWidth()
{
    LaneWidthMoveHandle *laneWidthMoveHandle = getFirstSelectedLaneWidthHandle();

    if (laneWidthMoveHandle)
    {
        LaneWidth *laneWidth = laneWidthMoveHandle->getLowSlot();

        if (laneWidth)
        {
            double w = laneWidth->getWidth(laneWidth->getSSectionEnd() - laneWidth->getParentLane()->getParentLaneSection()->getSStart());
            ui->widthSpinBox->setValue(w);
        }

        laneWidth = laneWidthMoveHandle->getHighSlot();
        if (laneWidth)
        {
            double w = laneWidth->getWidth(0.0);
            ui->widthSpinBox->setValue(w);
        }
    }

}

LaneWidthMoveHandle *
LaneSettings::getFirstSelectedLaneWidthHandle()
{
    QList<QGraphicsItem *> selectList = getProjectSettings()->getProjectWidget()->getHeightGraph()->getScene()->selectedItems();

    foreach (QGraphicsItem *item, selectList)
    {
        LaneWidthMoveHandle *laneWidthMoveHandle = dynamic_cast<LaneWidthMoveHandle *>(item);
        if (laneWidthMoveHandle)
        {
            return laneWidthMoveHandle;
        }
    }

    return NULL;
}

//################//
// SLOTS          //
//################//

void
LaneSettings::on_typeBox_currentIndexChanged(const QString &text)
{
    if (init_ && text != Lane::parseLaneTypeBack(lane_->getLaneType()))
    {
        SetLaneTypeCommand *command = new SetLaneTypeCommand(lane_, Lane::parseLaneType(text));
        getProjectSettings()->executeCommand(command);
    }
}

void
LaneSettings::on_levelBox_stateChanged(int state)
{
    bool bState = false;
    if (state != 0)
        bState = true;
    if (init_ && (bState != lane_->getLevel()))
    {
        SetLaneLevelCommand *command = new SetLaneLevelCommand(lane_, ui->levelBox->isChecked());
        getProjectSettings()->executeCommand(command);
    }
}

void
LaneSettings::on_predecessorCheckBox_stateChanged(int /*state*/)
{
    if (!init_)
    {
        return;
    }

    if (ui->predecessorCheckBox->isChecked())
    {
        SetLanePredecessorIdCommand *command = new SetLanePredecessorIdCommand(lane_, lane_->getId());
        getProjectSettings()->executeCommand(command);
    }
    else
    {
        SetLanePredecessorIdCommand *command = new SetLanePredecessorIdCommand(lane_, Lane::NOLANE);
        getProjectSettings()->executeCommand(command);
    }
}

void
LaneSettings::on_predecessorBox_valueChanged(int i)
{
    if (init_ && i != lane_->getId())
    {
        SetLanePredecessorIdCommand *command = new SetLanePredecessorIdCommand(lane_, i);
        getProjectSettings()->executeCommand(command);
    }
}

void
LaneSettings::on_successorCheckBox_stateChanged(int /*state*/)
{
    if (!init_)
    {
        return;
    }

    if (ui->successorCheckBox->isChecked())
    {
        SetLaneSuccessorIdCommand *command = new SetLaneSuccessorIdCommand(lane_, lane_->getId());
        getProjectSettings()->executeCommand(command);
    }
    else
    {
        SetLaneSuccessorIdCommand *command = new SetLaneSuccessorIdCommand(lane_, Lane::NOLANE);
        getProjectSettings()->executeCommand(command);
    }
}

void
LaneSettings::on_successorBox_valueChanged(int i)
{
    if (init_ && i != lane_->getId())
    {
        SetLaneSuccessorIdCommand *command = new SetLaneSuccessorIdCommand(lane_, i);
        getProjectSettings()->executeCommand(command);
    }
}

void
LaneSettings::on_addButton_released()
{
    Lane *newLane = new Lane(ui->newIdBox->value(), Lane::LT_DRIVING);

    LaneWidth *width = new LaneWidth(0.0, ui->widthBox->value(), 0.0, 0.0, 0.0);
    newLane->addWidthEntry(width);

    LaneRoadMark *roadMark = new LaneRoadMark(0.0, LaneRoadMark::RMT_SOLID, LaneRoadMark::RMW_STANDARD, LaneRoadMark::RMC_STANDARD, 0.12);
    newLane->addRoadMarkEntry(roadMark);

    InsertLaneCommand *command = new InsertLaneCommand(lane_->getParentLaneSection(), newLane);
    getProjectSettings()->executeCommand(command);
}

void
LaneSettings::on_addWidthButton_released()
{

    //insertWidthSectionHandle_->show();

    double s;
    LaneWidth *laneWidth = lane_->getWidthEntry(0);
    double endWidth = lane_->getWidth(laneWidth->getSSectionEnd());
    s = laneWidth->getLength() / 2;
    double startWidth = lane_->getWidth(s);
    double slope = (endWidth - startWidth) / (s);
    LaneWidth *newLaneWidth = new LaneWidth(s, startWidth, slope, 0.0, 0.0);

    InsertLaneWidthCommand *command = new InsertLaneWidthCommand(lane_, newLaneWidth);
    getProjectSettings()->executeCommand(command);
}

void LaneSettings::on_widthSpinBox_valueChanged(double w)
{
  //  laneEditor_->setWidth(w);
}

void
LaneSettings::activateInsertGroupBox(bool activ)
{
	ui->insertGroupBox->setVisible(activ);
	double y;
	if (activ)
	{
		y = ui->insertFrame->height() + ui->insertFrame->geometry().y();
	}
	else
	{
		y = ui->insertPushButton->height() + ui->insertFrame->geometry().y();
	}
	QRect geometry = ui->editFrame->geometry();
	geometry.setY(y + 6);
	ui->editFrame->setGeometry(geometry);
	ui->widthGroupBox->updateGeometry();
}

void
LaneSettings::activateWidthGroupBox(bool activ)
{
	ui->widthGroupBox->setVisible(activ);
}

//##################//
// Observer Pattern //
//##################//

void
LaneSettings::updateObserver()
{

    // Parent //
    //
    SettingsElement::updateObserver();
    if (isInGarbage())
    {
        return; // no need to go on
    }

    // Get change flags //
    //
    int changes = lane_->getLaneChanges();

    if ((changes & Lane::CLN_WidthsChanged) || (changes & DataElement::CDE_SelectionChange))
    {
        updateId();
        updateType();
        updateLevel();
        updatePredecessor();
        updateSuccessor();
        updateWidth();
    }

    if ((changes & Lane::CLN_IdChanged))
    {
        updateId();
    }

    if ((changes & Lane::CLN_TypeChanged))
    {
        updateType();
    }

    if ((changes & Lane::CLN_LevelChanged))
    {
        updateLevel();
    }

    if ((changes & Lane::CLN_PredecessorChanged))
    {
        updatePredecessor();
    }

    if ((changes & Lane::CLN_SuccessorChanged))
    {
        updateSuccessor();
    }
}
