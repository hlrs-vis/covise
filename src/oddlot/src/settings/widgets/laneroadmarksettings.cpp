/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/29/2010
**
**************************************************************************/

#include "laneroadmarksettings.hpp"
#include "ui_laneroadmarksettings.h"

// Data //
//
#include "src/data/roadsystem/sections/laneroadmark.hpp"
#include "src/data/commands/lanesectioncommands.hpp"

// Qt //
//
#include <QFileDialog>

// Utils //
//
#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

LaneRoadMarkSettings::LaneRoadMarkSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, LaneRoadMark *laneRoadMark)
    : SettingsElement(projectSettings, parentSettingsElement, laneRoadMark)
    , ui(new Ui::LaneRoadMarkSettings)
    , roadMark_(laneRoadMark)
    , init_(false)
{
    ui->setupUi(this);

    // Lists //
    //
    QStringList typeNames;
    typeNames << LaneRoadMark::parseRoadMarkTypeBack(LaneRoadMark::RMT_NONE)
              << LaneRoadMark::parseRoadMarkTypeBack(LaneRoadMark::RMT_SOLID)
              << LaneRoadMark::parseRoadMarkTypeBack(LaneRoadMark::RMT_BROKEN)
              << LaneRoadMark::parseRoadMarkTypeBack(LaneRoadMark::RMT_SOLID_SOLID)
              << LaneRoadMark::parseRoadMarkTypeBack(LaneRoadMark::RMT_SOLID_BROKEN)
              << LaneRoadMark::parseRoadMarkTypeBack(LaneRoadMark::RMT_BROKEN_SOLID);
    ui->typeBox->addItems(typeNames);

    QStringList weightNames;
    weightNames << LaneRoadMark::parseRoadMarkWeightBack(LaneRoadMark::RMW_STANDARD)
                << LaneRoadMark::parseRoadMarkWeightBack(LaneRoadMark::RMW_BOLD);
    ui->weightBox->addItems(weightNames);

    QStringList colorNames;
    colorNames << LaneRoadMark::parseRoadMarkColorBack(LaneRoadMark::RMC_STANDARD)
               << LaneRoadMark::parseRoadMarkColorBack(LaneRoadMark::RMC_YELLOW);
    ui->colorBox->addItems(colorNames);

    QStringList laneChangeNames;
    laneChangeNames << LaneRoadMark::parseRoadMarkLaneChangeBack(LaneRoadMark::RMLC_INCREASE)
                    << LaneRoadMark::parseRoadMarkLaneChangeBack(LaneRoadMark::RMLC_DECREASE)
                    << LaneRoadMark::parseRoadMarkLaneChangeBack(LaneRoadMark::RMLC_BOTH)
                    << LaneRoadMark::parseRoadMarkLaneChangeBack(LaneRoadMark::RMLC_NONE);
    ui->laneChangeBox->addItems(laneChangeNames);

    // Initial Values //
    //
    updateSOffset();
    updateType();
    updateWeight();
    updateColor();
    updateWidth();
    updateLaneChange();

    init_ = true;
}

LaneRoadMarkSettings::~LaneRoadMarkSettings()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
LaneRoadMarkSettings::updateSOffset()
{
    ui->offsetBox->setValue(roadMark_->getSOffset());
    if (roadMark_->getSOffset() == 0.0)
    {
        ui->offsetBox->setEnabled(false);
    }
}

void
LaneRoadMarkSettings::updateType()
{
    ui->typeBox->setCurrentIndex(ui->typeBox->findText(LaneRoadMark::parseRoadMarkTypeBack(roadMark_->getRoadMarkType())));
}

void
LaneRoadMarkSettings::updateWeight()
{
    ui->weightBox->setCurrentIndex(ui->weightBox->findText(LaneRoadMark::parseRoadMarkWeightBack(roadMark_->getRoadMarkWeight())));
}

void
LaneRoadMarkSettings::updateColor()
{
    ui->colorBox->setCurrentIndex(ui->colorBox->findText(LaneRoadMark::parseRoadMarkColorBack(roadMark_->getRoadMarkColor())));
}

void
LaneRoadMarkSettings::updateWidth()
{
    ui->widthBox->setValue(roadMark_->getRoadMarkWidth());
}

void
LaneRoadMarkSettings::updateLaneChange()
{
    ui->laneChangeBox->setCurrentIndex(ui->laneChangeBox->findText(LaneRoadMark::parseRoadMarkLaneChangeBack(roadMark_->getRoadMarkLaneChange())));
}

//################//
// SLOTS          //
//################//

void
LaneRoadMarkSettings::on_offsetBox_editingFinished()
{
    double newValue = ui->offsetBox->value();
    if (init_ && newValue != roadMark_->getSOffset())
    {
        SetLaneRoadMarkSOffsetCommand *command = new SetLaneRoadMarkSOffsetCommand(roadMark_, newValue, NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
LaneRoadMarkSettings::on_typeBox_currentIndexChanged(const QString &text)
{
    if (init_ && text != LaneRoadMark::parseRoadMarkTypeBack(roadMark_->getRoadMarkType()))
    {
        SetLaneRoadMarkTypeCommand *command = new SetLaneRoadMarkTypeCommand(roadMark_, LaneRoadMark::parseRoadMarkType(text));
        getProjectSettings()->executeCommand(command);
    }
}

void
LaneRoadMarkSettings::on_weightBox_currentIndexChanged(const QString &text)
{
    if (init_ && text != LaneRoadMark::parseRoadMarkWeightBack(roadMark_->getRoadMarkWeight()))
    {
        SetLaneRoadMarkWeightCommand *command = new SetLaneRoadMarkWeightCommand(roadMark_, LaneRoadMark::parseRoadMarkWeight(text));
        getProjectSettings()->executeCommand(command);
    }
}

void
LaneRoadMarkSettings::on_colorBox_currentIndexChanged(const QString &text)
{
    if (init_ && text != LaneRoadMark::parseRoadMarkColorBack(roadMark_->getRoadMarkColor()))
    {
        SetLaneRoadMarkColorCommand *command = new SetLaneRoadMarkColorCommand(roadMark_, LaneRoadMark::parseRoadMarkColor(text));
        getProjectSettings()->executeCommand(command);
    }
}

void
LaneRoadMarkSettings::on_widthBox_editingFinished()
{
    double newValue = ui->widthBox->value();
    if (init_ && newValue != roadMark_->getRoadMarkWidth())
    {
        SetLaneRoadMarkWidthCommand *command = new SetLaneRoadMarkWidthCommand(roadMark_, newValue, NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
LaneRoadMarkSettings::on_laneChangeBox_currentIndexChanged(const QString &text)
{
    if (init_ && text != LaneRoadMark::parseRoadMarkLaneChangeBack(roadMark_->getRoadMarkLaneChange()))
    {
        SetLaneRoadMarkLaneChangeCommand *command = new SetLaneRoadMarkLaneChangeCommand(roadMark_, LaneRoadMark::parseRoadMarkLaneChange(text));
        getProjectSettings()->executeCommand(command);
    }
}

//##################//
// Observer Pattern //
//##################//

void
LaneRoadMarkSettings::updateObserver()
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
    int changes = roadMark_->getRoadMarkChanges();

    // LaneRoadMark //
    //
    if ((changes & LaneRoadMark::CLR_OffsetChanged))
    {
        updateSOffset();
    }

    if ((changes & LaneRoadMark::CLR_TypeChanged))
    {
        updateType();
    }

    if ((changes & LaneRoadMark::CLR_WeightChanged))
    {
        updateWeight();
    }

    if ((changes & LaneRoadMark::CLR_ColorChanged))
    {
        updateColor();
    }

    if ((changes & LaneRoadMark::CLR_WidthChanged))
    {
        updateWidth();
    }

    if ((changes & LaneRoadMark::CLR_LaneChangeChanged))
    {
        updateLaneChange();
    }
}
