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

    LinearError = ui->LinearErrorSpin->value();
    CurveError = ui->CurveErrorSpin->value();
}
//################//
// CONSTRUCTOR    //
//################//

ImportSettings::ImportSettings()
    : ui(new Ui::ImportSettings)
{
    inst = this;
    ui->setupUi(this);

    connect(this, SIGNAL(accepted()), this, SLOT(okPressed()));

    ui->LinearErrorSpin->setDecimals(10);
    ui->LinearErrorSpin->setMaximum(100);
    ui->LinearErrorSpin->setMinimum(0);
    ui->LinearErrorSpin->setValue(0.4);
    ui->CurveErrorSpin->setDecimals(10);
    ui->CurveErrorSpin->setMaximum(100);
    ui->CurveErrorSpin->setMinimum(0);
    ui->CurveErrorSpin->setValue(1.4);
    LinearError = ui->LinearErrorSpin->value();
    CurveError = ui->CurveErrorSpin->value();

    inst = this;
}

ImportSettings::~ImportSettings()
{
    delete ui;
}
