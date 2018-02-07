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

#include "exportsettings.hpp"
#include "ui_exportsettings.h"

// Data //

ExportSettings *ExportSettings::inst = NULL;

void ExportSettings::okPressed()
{

}
//################//
// CONSTRUCTOR    //
//################//

ExportSettings::ExportSettings()
    : ui(new Ui::ExportSettings)
{
    inst = this;
    ui->setupUi(this);

    connect(this, SIGNAL(accepted()), this, SLOT(okPressed()));

    //ui->exportIDOptions->

    inst = this;
}
ExportSettings::ExportIDVariants ExportSettings::ExportIDVariant()
{
	return (ExportSettings::ExportIDVariants)ui->exportIDOptions->currentIndex();
}

ExportSettings::~ExportSettings()
{
    delete ui;
}
