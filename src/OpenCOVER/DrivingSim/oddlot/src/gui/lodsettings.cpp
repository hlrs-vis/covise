/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Uwe Woessner (c) 2013
**   <woessner@hlrs.de.de>
**   03/2013
**
**************************************************************************/

#include "lodsettings.hpp"
#include "ui_lodsettings.h"

// Data //

LODSettings *LODSettings::inst = NULL;

void LODSettings::okPressed()
{
    TopViewEditorPointsPerMeter = ui->LODTopViewEditorSpin->value();
    HeightEditorPointsPerMeter = ui->LODHeightEditorSpin->value();
    SignalEditorScalingLevel = ui->LODSignalEditorSpin->value();
}
//################//
// CONSTRUCTOR    //
//################//

LODSettings::LODSettings()
    : ui(new Ui::LODSettings)
{
    inst = this;
    ui->setupUi(this);

    connect(this, SIGNAL(accepted()), this, SLOT(okPressed()));

    ui->LODTopViewEditorSpin->setDecimals(10);
    ui->LODTopViewEditorSpin->setMaximum(100);
    ui->LODTopViewEditorSpin->setMinimum(-100);
    ui->LODTopViewEditorSpin->setValue(1.0);
    ui->LODHeightEditorSpin->setDecimals(10);
    ui->LODHeightEditorSpin->setMaximum(100);
    ui->LODHeightEditorSpin->setMinimum(-100);
    ui->LODHeightEditorSpin->setValue(2.0);
    ui->LODSignalEditorSpin->setDecimals(10);
    ui->LODSignalEditorSpin->setMaximum(100);
    ui->LODSignalEditorSpin->setMinimum(0.1);
    ui->LODSignalEditorSpin->setValue(3.0);

    TopViewEditorPointsPerMeter = ui->LODTopViewEditorSpin->value();
    HeightEditorPointsPerMeter = ui->LODHeightEditorSpin->value();
    SignalEditorScalingLevel = ui->LODSignalEditorSpin->value();

    inst = this;
}

LODSettings::~LODSettings()
{
    delete ui;
}
