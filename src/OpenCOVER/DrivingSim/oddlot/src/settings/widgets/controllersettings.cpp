/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/2/2010
**
**************************************************************************/

#include "controllersettings.hpp"
#include "ui_controllersettings.h"

#include "src/mainwindow.hpp"

#include "src/settings/projectsettings.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/rsystemelementcontroller.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/commands/controllercommands.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"

#include "src/graph/items/roadsystem/signal/signalitem.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Qt //
//
#include <QInputDialog>
#include <QStringList>

// Utils //
//
#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

ControllerSettings::ControllerSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, RSystemElementController *controller)
    : SettingsElement(projectSettings, parentSettingsElement, controller)
    , ui(new Ui::ControllerSettings)
    , controller_(controller)
    , init_(false)
    , valueChanged_(true)
{
    ui->setupUi(this);

    // Initial Values //
    //
    updateProperties();
    updateControlEntries();

    connect(ui->nameBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->nameBox, SIGNAL(textChanged(const QString&)), this, SLOT(onValueChanged()));
    connect(ui->sequenceSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onEditingFinished(int)));
    connect(ui->scriptLineEdit, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->scriptLineEdit, SIGNAL(textChanged(const QString&)), this, SLOT(onValueChanged()));
    connect(ui->cycleTimeSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onEditingFinished(double)));

    init_ = true;
}

ControllerSettings::~ControllerSettings()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
ControllerSettings::updateProperties()
{
    if (controller_)
    {
        ui->nameBox->setText(controller_->getName());
        ui->idLabel->setText(controller_->getID().speakingName());
        ui->sequenceSpinBox->setValue(controller_->getSequence());
        ui->scriptLineEdit->setText(controller_->getScript());
        ui->cycleTimeSpinBox->setValue(controller_->getCycleTime());

    }
}

void
ControllerSettings::updateControlEntries()
{
    ui->controlEntryTableWidget->clear();

    // Entries //
    //
    QStringList header;
    header << "signal id"
           << "type"
;
    ui->controlEntryTableWidget->setHorizontalHeaderLabels(header);


    QList<ControlEntry *> controlEntries = controller_->getControlEntries();
    ui->controlEntryTableWidget->setRowCount(controlEntries.size());
    int row = 0;
    foreach (ControlEntry *element, controlEntries)
    {
        ui->controlEntryTableWidget->setItem(row, 0, new QTableWidgetItem(element->getSignalId().speakingName()));
        ui->controlEntryTableWidget->setItem(row, 1, new QTableWidgetItem(element->getType()));
        ++row;
    }
}

//################//
// SLOTS          //
//################//
void
ControllerSettings::onEditingFinished(int i)
{
    if (ui->sequenceSpinBox->value() != controller_->getSequence())
    {
        valueChanged_ = true;
        onEditingFinished();
    }
}

void
ControllerSettings::onEditingFinished(double d)
{
    if (ui->sequenceSpinBox->value() != controller_->getSequence())
    {
        valueChanged_ = true;
        onEditingFinished();
    }
}

void
ControllerSettings::onEditingFinished()
{
    if (valueChanged_)
    {
        QString filename = ui->nameBox->text();
        odrID newId = controller_->getID();
		newId.setName(filename);
    

        SetControllerPropertiesCommand *command = new SetControllerPropertiesCommand(controller_, newId, filename, ui->sequenceSpinBox->value(), ui->scriptLineEdit->text(), ui->cycleTimeSpinBox->value());
        getProjectSettings()->executeCommand(command);

        valueChanged_ = false;
        QWidget * focusWidget = QApplication::focusWidget();
        if (focusWidget)
        {
            focusWidget->clearFocus();
        }
    }
}


void
ControllerSettings::onValueChanged()
{
    valueChanged_ = true;
}

//##################//
// Observer Pattern //
//##################//

void
ControllerSettings::updateObserver()
{

    // Parent //
    //
    SettingsElement::updateObserver();
    if (isInGarbage())
    {
        return; // no need to go on
    }

    // Controller //
    //
    int changes = controller_->getControllerChanges();

    if ((changes & RSystemElementController::CRC_ParameterChange))
    {
        updateProperties();
    }

    if ((changes & RSystemElementController::CRC_EntryChange))
    {
        updateControlEntries();
    }


}
