/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/25/2010
**
**************************************************************************/

#include "projectdatasettings.hpp"
#include "ui_projectdatasettings.h"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/commands/projectdatacommands.hpp"

#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/scenerysystem/scenerysystem.hpp"
#include "src/data/visitors/boundingboxvisitor.hpp"

// Qt //
//
#include <QInputDialog>

// Utils //
//
#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

ProjectDataSettings::ProjectDataSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, ProjectData *projectData)
    : SettingsElement(projectSettings, parentSettingsElement, projectData)
    , ui(new Ui::ProjectDataSettings)
    , projectData_(projectData)
{
    ui->setupUi(this);

    updateName();
    updateVersion();
    updateDate();
    updateSize();
}

ProjectDataSettings::~ProjectDataSettings()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
ProjectDataSettings::updateName()
{
    ui->nameBox->setText(projectData_->getName());
}

void
ProjectDataSettings::updateVersion()
{
    ui->versionBox->setValue(projectData_->getVersion());
}

void
ProjectDataSettings::updateDate()
{
    ui->dateBox->setText(projectData_->getDate());
}

void
ProjectDataSettings::updateSize()
{
    ui->northBox->setValue(projectData_->getNorth());
    ui->southBox->setValue(projectData_->getSouth());
    ui->eastBox->setValue(projectData_->getEast());
    ui->westBox->setValue(projectData_->getWest());
}

//################//
// SLOTS          //
//################//

void
ProjectDataSettings::on_nameButton_released()
{
    // Open a small dialog an ask for a new name //
    //
    bool ok = false;
    QString newValue = QInputDialog::getText(this, tr("ODD: Project Name"), tr("Please enter a new project name:"), QLineEdit::Normal, projectData_->getName(), &ok);
    if (ok && !newValue.isEmpty() && newValue != projectData_->getName())
    {
        SetProjectNameCommand *command = new SetProjectNameCommand(projectData_, newValue, NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
ProjectDataSettings::on_autoSizeMeButton_released()
{
    BoundingBoxVisitor *visitor = new BoundingBoxVisitor();
    projectData_->getRoadSystem()->accept(visitor);
    projectData_->getScenerySystem()->accept(visitor);
    QRectF box = visitor->getBoundingBox();
    SetProjectDimensionsCommand *command = new SetProjectDimensionsCommand(projectData_, box.bottom() + 0.1 * box.height(), box.top() - 0.1 * box.height(), box.right() + 0.1 * box.width(), box.left() - 0.1 * box.width());
    getProjectSettings()->executeCommand(command);
}

void
ProjectDataSettings::on_versionBox_editingFinished()
{
    double newValue = ui->versionBox->value();
    if (newValue != projectData_->getVersion())
    {
        SetProjectVersionCommand *command = new SetProjectVersionCommand(projectData_, newValue, NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
ProjectDataSettings::on_northBox_editingFinished()
{
    double newValue = ui->northBox->value();
    if (newValue != projectData_->getNorth())
    {
        SetProjectDimensionsCommand *command = new SetProjectDimensionsCommand(projectData_, newValue, projectData_->getSouth(), projectData_->getEast(), projectData_->getWest(), NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
ProjectDataSettings::on_southBox_editingFinished()
{
    double newValue = ui->southBox->value();
    if (newValue != projectData_->getSouth())
    {
        SetProjectDimensionsCommand *command = new SetProjectDimensionsCommand(projectData_, projectData_->getNorth(), newValue, projectData_->getEast(), projectData_->getWest(), NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
ProjectDataSettings::on_eastBox_editingFinished()
{
    double newValue = ui->eastBox->value();
    if (newValue != projectData_->getEast())
    {
        SetProjectDimensionsCommand *command = new SetProjectDimensionsCommand(projectData_, projectData_->getNorth(), projectData_->getSouth(), newValue, projectData_->getWest(), NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
ProjectDataSettings::on_westBox_editingFinished()
{
    double newValue = ui->westBox->value();
    if (newValue != projectData_->getWest())
    {
        SetProjectDimensionsCommand *command = new SetProjectDimensionsCommand(projectData_, projectData_->getNorth(), projectData_->getSouth(), projectData_->getEast(), newValue, NULL);
        getProjectSettings()->executeCommand(command);
    }
}

//##################//
// Observer Pattern //
//##################//

void
ProjectDataSettings::updateObserver()
{

    // Get change flags //
    //
    int changes = projectData_->getProjectDataChanges();

    // ProjectData //
    //
    if ((changes & ProjectData::CPD_NameChange))
    {
        updateName();
    }

    if ((changes & ProjectData::CPD_VersionChange))
    {
        updateVersion();
    }

    if ((changes & ProjectData::CPD_DateChange))
    {
        updateDate();
    }

    if ((changes & ProjectData::CPD_SizeChange))
    {
        updateSize();
    }

    // Parent //
    //
    SettingsElement::updateObserver();
}
