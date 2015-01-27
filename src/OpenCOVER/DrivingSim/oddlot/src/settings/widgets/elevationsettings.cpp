/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/19/2010
**
**************************************************************************/

#include "elevationsettings.hpp"
#include "ui_elevationsettings.h"

// Data //
//

#include "src/data/commands/elevationsectioncommands.hpp"
#include "src/data/roadsystem/sections/elevationsection.hpp"
#include "src/graph/items/roadsystem/elevation/elevationmovehandle.hpp"
#include "src/graph/profilegraph.hpp"
#include "src/graph/profilegraphscene.hpp"
#include "src/gui/projectwidget.hpp"

// Commands //
//
#include "src/data/commands/roadsectioncommands.hpp"

// Qt //
//
#include <QFileDialog>

// Utils //
//
#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

ElevationSettings::ElevationSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, ElevationSection *elevationSection)
    : SettingsElement(projectSettings, parentSettingsElement, elevationSection)
    , ui(new Ui::ElevationSettings)
    , elevationSection_(elevationSection)
{
    ui->setupUi(this);

    // Initial Values //
    //
    updateProperties();
    updateHeight();

    //	projectSettings->getProjectWidget()->getTopviewGraph()->getScene()->is
}

ElevationSettings::~ElevationSettings()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//
void
ElevationSettings::enableElevationSettingParams(bool value)
{
    ui->sLabel->setEnabled(value);
    ui->sSpinBox->setEnabled(value);
    ui->slopeLabel->setEnabled(value);
    ui->slopeSpinBox->setEnabled(value);
}

void
ElevationSettings::updateProperties()
{
    if (elevationSection_)
    {
        ui->sSpinBox->setValue(elevationSection_->getSStart());
    }
}

void
ElevationSettings::updateHeight()
{
    ElevationMoveHandle *elevationMoveHandle = getFirstSelectedMoveHandle();

    if (elevationMoveHandle)
    {
        ElevationSection *elevationSection = elevationMoveHandle->getLowSlot();

        if (elevationSection)
        {
            double h = elevationSection->getElevation(elevationSection->getSEnd());
            ui->heightBox->setValue(h);
        }

        elevationSection = elevationMoveHandle->getHighSlot();
        if (elevationSection)
        {
            double h = elevationSection->getElevation(elevationSection->getSStart());
            ui->heightBox->setValue(h);
        }

        if (getProjectSettings()->getProjectWidget()->getProfileGraph()->getScene()->selectedItems().count() <= 1)
        {
            enableElevationSettingParams(true);
        }

        return;
    }

    enableElevationSettingParams(false);
}

void
ElevationSettings::on_heightBox_editingFinished()
{
    /*  double newValue = ui->heightBox->value();
   editor->setHeight(newValue);*/
}

void
ElevationSettings::on_sSpinBox_editingFinished()
{
    if (elevationSection_->getDegree() > 1)
    {
        updateProperties();
        return;
    }

    QList<ElevationSection *> endPointSections;
    QList<ElevationSection *> startPointSections;
    startPointSections.append(elevationSection_);
    ElevationSection *sectionBefore = elevationSection_->getParentRoad()->getElevationSectionBefore(elevationSection_->getSStart());

    if (sectionBefore)
    {
        if (sectionBefore->getDegree() > 1)
        {
            updateProperties();
            return;
        }
        else
        {
            endPointSections.append(sectionBefore);
        }
    }

    // Command //
    //
    QPointF dPos = QPointF(ui->sSpinBox->value() - elevationSection_->getSStart(), 0.0);
    ElevationMovePointsCommand *command = new ElevationMovePointsCommand(endPointSections, startPointSections, dPos, NULL);

    if (command->isValid())
    {
        getProjectData()->getUndoStack()->push(command);
    }
    else
    {
        delete command;
    }
}

void
ElevationSettings::on_slopeSpinBox_editingFinished()
{
    if (elevationSection_->getDegree() > 1)
    {
        updateProperties();
        return;
    }

    QList<ElevationSection *> endPointSections;
    QList<ElevationSection *> startPointSections;
    endPointSections.append(elevationSection_);
    ElevationSection *sectionNext = elevationSection_->getParentRoad()->getElevationSectionNext(elevationSection_->getSStart());
    if (sectionNext)
    {
        if (sectionNext->getDegree() > 1)
        {
            return;
        }
        else
        {
            startPointSections.append(sectionNext);
        }
    }

    // Command //
    //
    double s = 100 * abs(elevationSection_->getElevation(elevationSection_->getSStart()) - elevationSection_->getElevation(elevationSection_->getSEnd())) / ui->slopeSpinBox->value() + elevationSection_->getSStart();
    if (s < elevationSection_->getParentRoad()->getLength())
    {
        QPointF dPos = QPointF(s - sectionNext->getSStart(), 0.0);
        ElevationMovePointsCommand *command = new ElevationMovePointsCommand(endPointSections, startPointSections, dPos, NULL);

        if (command->isValid())
        {
            getProjectData()->getUndoStack()->push(command);
        }
        else
        {
            delete command;
        }
    }
}

ElevationMoveHandle *
ElevationSettings::
    getFirstSelectedMoveHandle()
{
    QList<QGraphicsItem *> selectList = getProjectSettings()->getProjectWidget()->getProfileGraph()->getScene()->selectedItems();

    foreach (QGraphicsItem *item, selectList)
    {
        ElevationMoveHandle *elevationMoveHandle = dynamic_cast<ElevationMoveHandle *>(item);
        if (elevationMoveHandle)
        {
            return elevationMoveHandle;
        }
    }

    return NULL;
}

//##################//
// Observer Pattern //
//##################//

void
ElevationSettings::updateObserver()
{

    // Get change flags //
    //
    int changes = elevationSection_->getElevationSectionChanges();

    // Elevation //
    //
    if ((changes & ElevationSection::CEL_ParameterChange) || (changes & DataElement::CDE_SelectionChange))
    {
        updateProperties();
        updateHeight();
    }

    // Parent //
    //
    SettingsElement::updateObserver();
}
