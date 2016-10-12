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

#include "oscsettings.hpp"
#include "ui_oscsettings.h"

// Data //

OSCSettings *OSCSettings::inst = NULL;

void OSCSettings::okPressed()
{
	bool currentValidationValue = ui->readValidationCheckBox->isChecked();
	if (currentValidationValue != validation)
	{
		validation = currentValidationValue;
		emit readValidationChanged(ui->readValidationCheckBox->isChecked());
	}
}
//################//
// CONSTRUCTOR    //
//################//

OSCSettings::OSCSettings()
    : ui(new Ui::OSCSettings)
	, validation(false)
{
    inst = this;
    ui->setupUi(this);

    connect(this, SIGNAL(accepted()), this, SLOT(okPressed()));

    inst = this;
}

bool OSCSettings::readValidation()
{
    return validation;
}
bool OSCSettings::loadDefaults()
{
    return ui->defaultValuesCheckBox->isChecked();
}

OSCSettings::~OSCSettings()
{
    delete ui;
}
