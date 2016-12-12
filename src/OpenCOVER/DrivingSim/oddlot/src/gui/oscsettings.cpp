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

#include <QFileDialog>

// Data //

OSCSettings *OSCSettings::inst = NULL;

void OSCSettings::okPressed()
{
	bool currentValidationValue = ui->readValidationCheckBox->isChecked();
	if (currentValidationValue != validation_)
	{
		validation_ = currentValidationValue;
		emit readValidationChanged(ui->readValidationCheckBox->isChecked());
	}

	catalogDir_ = ui->dirLabel->text();
}
//################//
// CONSTRUCTOR    //
//################//

OSCSettings::OSCSettings(const QString &dir)
    : ui(new Ui::OSCSettings)
	, validation_(false)
	, catalogDir_(dir)
{
    inst = this;
    ui->setupUi(this);
	ui->dirLabel->setText(catalogDir_);

    connect(this, SIGNAL(accepted()), this, SLOT(okPressed()));
	connect(ui->dirPushButton, SIGNAL(pressed()), this, SLOT(dirPushButtonPressed()));

    inst = this;
}

bool OSCSettings::readValidation()
{
    return validation_;
}
bool OSCSettings::loadDefaults()
{
    return ui->defaultValuesCheckBox->isChecked();
}

QString OSCSettings::getCatalogDir()
{
	return catalogDir_;
}

void OSCSettings::dirPushButtonPressed()
{
	QFileDialog dialog (this);
	dialog.setFileMode(QFileDialog::Directory);
	QString dir = dialog.getExistingDirectory(this, "", "", QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks) + "/";
	ui->dirLabel->setText(dir);
}

OSCSettings::~OSCSettings()
{
    delete ui;
}
