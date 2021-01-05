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

#include "importsettings.hpp"
#include "ui_importsettings.h"

// Data //

ImportSettings *ImportSettings::inst = NULL;

void ImportSettings::okPressed()
{

}
//################//
// CONSTRUCTOR    //
//################//

ImportSettings::ImportSettings()
    : ui(new Ui::ImportSettings)
{
    inst = this;
    ui->setupUi(this);

    //connect(this, SIGNAL(accepted()), this, SLOT(okPressed()));

    ui->LinearErrorSpin->setDecimals(10);
    ui->LinearErrorSpin->setMaximum(100);
    ui->LinearErrorSpin->setMinimum(0);
    ui->LinearErrorSpin->setValue(0.4);
    ui->CurveErrorSpin->setDecimals(10);
    ui->CurveErrorSpin->setMaximum(100);
    ui->CurveErrorSpin->setMinimum(0);
    ui->CurveErrorSpin->setValue(1.4);

    inst = this;
}
double ImportSettings::LinearError()
{
    return ui->LinearErrorSpin->value();
}
double ImportSettings::CurveError()
{
    return ui->CurveErrorSpin->value();
}

bool ImportSettings::importPrimary()
{
    return ui->primary_cb->isChecked();
}
bool ImportSettings::importSecondary()
{
    return ui->secondary_cb->isChecked();
}
bool ImportSettings::importTertiary()
{
    return ui->tertiary_cb->isChecked();
}
bool ImportSettings::importMotorway()
{
    return ui->motorway_cb->isChecked();
}
bool ImportSettings::importService()
{
    return ui->service_cb->isChecked();
}
bool ImportSettings::importPath()
{
    return ui->path_cb->isChecked();
}
bool ImportSettings::importSteps()
{
    return ui->steps_cb->isChecked();
}
bool ImportSettings::importTrack()
{
    return ui->track_cb->isChecked();
}
bool ImportSettings::importFootway()
{
    return ui->footway_cb->isChecked();
}
bool ImportSettings::importResidential()
{
    return ui->residential_cb->isChecked();
}
bool ImportSettings::importLiving_street()
{
    return ui->living_street_cb->isChecked();
}
bool ImportSettings::importCycleway()
{
    return ui->cycleway_cb->isChecked();
}
bool ImportSettings::importTurning_circle()
{
    return ui->turning_circle_cb->isChecked();
}
bool ImportSettings::importPedestrian()
{
    return ui->pedestrian_cb->isChecked();
}
bool ImportSettings::importUnclassified()
{
    return ui->unclassified_cb->isChecked();
}
bool ImportSettings::maximizeCurveRadius()
{
    return ui->maximize_curve_radius_cb->isChecked();
}

ImportSettings::~ImportSettings()
{
    delete ui;
}
