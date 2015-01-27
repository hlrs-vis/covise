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

#include "scenerymapsettings.hpp"
#include "ui_scenerymapsettings.h"

// Data //
//
#include "src/data/scenerysystem/scenerymap.hpp"
#include "src/data/scenerysystem/heightmap.hpp"
#include "src/data/commands/scenerycommands.hpp"

// Qt //
//
#include <QFileDialog>

// Utils //
//
#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

SceneryMapSettings::SceneryMapSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, SceneryMap *sceneryMap)
    : SettingsElement(projectSettings, parentSettingsElement, sceneryMap)
    , ui(new Ui::SceneryMapSettings)
    , sceneryMap_(sceneryMap)
    , heightmap_(NULL)
{
    ui->setupUi(this);

    if (sceneryMap_->getMapType() == SceneryMap::DMT_Aerial)
    {
        ui->dataLabel->setVisible(false);
        ui->dataText->setVisible(false);
        ui->changeDataButton->setVisible(false);
    }
    else
    {
        ui->dataLabel->setVisible(true);
        ui->dataText->setVisible(true);
        ui->changeDataButton->setVisible(true);
        heightmap_ = dynamic_cast<Heightmap *>(sceneryMap_);
        updateDataFilename();
    }

    updatePosition();
    updateSize();
    updateOpacity();
    updateFilename();
}

SceneryMapSettings::~SceneryMapSettings()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
SceneryMapSettings::updatePosition()
{
    ui->xBox->setValue(sceneryMap_->getX());
    ui->yBox->setValue(sceneryMap_->getY());
}

void
SceneryMapSettings::updateSize()
{
    ui->widthBox->setValue(sceneryMap_->getWidth());
    ui->heightBox->setValue(sceneryMap_->getHeight());
}

void
SceneryMapSettings::updateOpacity()
{
    ui->opacityBox->setValue(sceneryMap_->getOpacity() * 100);
}

void
SceneryMapSettings::updateFilename()
{
    ui->filenameText->setText(sceneryMap_->getFilename());
}

void
SceneryMapSettings::updateDataFilename()
{
    ui->dataText->setText(heightmap_->getHeightmapDataFilename());
}

//################//
// SLOTS          //
//################//

void
SceneryMapSettings::on_changeFileButton_released()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Open Image File"));
    if (filename.isEmpty())
    {
        return;
    }

    QPixmap pixmap(filename); // this pixmap is only temporary
    if (pixmap.isNull())
    {
        qDebug("ERROR 1010251155! SceneryMapSettings: Pixmap could not be loaded!");
        return;
    }

    SetMapFilenameCommand *command = new SetMapFilenameCommand(sceneryMap_, filename);
    getProjectSettings()->executeCommand(command);
}

void
SceneryMapSettings::on_changeDataButton_released()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Open Data File"));
    if (filename.isEmpty())
    {
        return;
    }

    SetHeightmapDataFilenameCommand *command = new SetHeightmapDataFilenameCommand(heightmap_, filename);
    getProjectSettings()->executeCommand(command);
}

void
SceneryMapSettings::on_xBox_editingFinished()
{
    double newValue = ui->xBox->value();
    if (newValue != sceneryMap_->getX())
    {
        SetMapPositionCommand *command = new SetMapPositionCommand(sceneryMap_, newValue, sceneryMap_->getY(), NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
SceneryMapSettings::on_yBox_editingFinished()
{
    double newValue = ui->yBox->value();
    if (newValue != sceneryMap_->getY())
    {
        SetMapPositionCommand *command = new SetMapPositionCommand(sceneryMap_, sceneryMap_->getX(), newValue, NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
SceneryMapSettings::on_widthBox_editingFinished()
{
    double newValue = ui->widthBox->value();
    if (newValue != sceneryMap_->getWidth())
    {
        double ratio = newValue / sceneryMap_->getWidth();
        SetMapSizeCommand *command = new SetMapSizeCommand(sceneryMap_, newValue, sceneryMap_->getHeight() * ratio, NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
SceneryMapSettings::on_heightBox_editingFinished()
{
    double newValue = ui->heightBox->value();
    if (newValue != sceneryMap_->getHeight())
    {
        double ratio = newValue / sceneryMap_->getHeight();
        SetMapSizeCommand *command = new SetMapSizeCommand(sceneryMap_, sceneryMap_->getWidth() * ratio, newValue, NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
SceneryMapSettings::on_opacityBox_editingFinished()
{
    int newValue = ui->opacityBox->value();
    if (fabs(newValue - sceneryMap_->getOpacity() * 100) > NUMERICAL_ZERO6)
    {
        SetMapOpacityCommand *command = new SetMapOpacityCommand(sceneryMap_, newValue / 100.0, NULL);
        getProjectSettings()->executeCommand(command);
    }
}

//##################//
// Observer Pattern //
//##################//

void
SceneryMapSettings::updateObserver()
{

    // Get change flags //
    //
    int changes = sceneryMap_->getSceneryMapChanges();

    // SceneryMap //
    //
    if ((changes & SceneryMap::CSM_X)
        || (changes & SceneryMap::CSM_Y))
    {
        updatePosition();
    }

    if ((changes & SceneryMap::CSM_Width)
        || (changes & SceneryMap::CSM_Height))
    {
        updateSize();
        updatePosition(); // update too since Qt images have a different origin
    }

    if (changes & SceneryMap::CSM_Opacity)
    {
        updateOpacity();
    }

    if (changes & SceneryMap::CSM_Filename)
    {
        updateFilename();
    }

    if (changes & SceneryMap::CSM_Id)
    {
        // TODO, but should not be scenerymap specific anyway (SystemElement)
    }

    if (heightmap_)
    {
        int changes = heightmap_->getHeightmapChanges();
        if (changes & Heightmap::CHM_DataFileChanged)
        {
            updateDataFilename();
        }
    }

    // Parent //
    //
    SettingsElement::updateObserver();
}
