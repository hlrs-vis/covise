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

#include "apisettings.hpp"
#include "ui_apisettings.h"
#include "src/mainwindow.hpp"

// Data //

APISettings *APISettings::inst = NULL;

void APISettings::okPressed()
{
    GoogleAPIKey = ui->GoogleAPIKeyLineEdit->text();
    BINGAPIKey = ui->BINGMapsLineEdit->text();
    mainWindow->appSettings.setValue("apiKey/google", GoogleAPIKey);
    mainWindow->appSettings.sync();
}

//################//
// CONSTRUCTOR    //
//################//

APISettings::APISettings(MainWindow *mw)
    : ui(new Ui::APISettings)
{
    mainWindow = mw;
    inst = this;
    ui->setupUi(this);

    //connect(this, SIGNAL(accepted()), this, SLOT(okPressed()));

    

    ui->GoogleAPIKeyLineEdit->setText(mainWindow->appSettings.value("apiKey/google").toString());
    ui->BINGMapsLineEdit->setText("");


    GoogleAPIKey = ui->GoogleAPIKeyLineEdit->text();
    BINGAPIKey = ui->BINGMapsLineEdit->text();
    inst = this;
}

APISettings::~APISettings()
{
    delete ui;
}
