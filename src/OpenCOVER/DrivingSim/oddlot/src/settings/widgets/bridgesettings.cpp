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

#include "bridgesettings.hpp"
#include "ui_bridgesettings.h"
#include "src/data/roadsystem/odrID.hpp"

#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/roadsystem/sections/bridgeobject.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/commands/signalcommands.hpp"
#include "src/data/commands/roadsectioncommands.hpp"


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

BridgeSettings::BridgeSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, Bridge *bridge)
    : SettingsElement(projectSettings, parentSettingsElement, bridge)
    , ui(new Ui::BridgeSettings)
    , bridge_(bridge)
    , init_(false)
    , valueChanged_(true)
{

    ui->setupUi(this);

    // Initial Values //
    //
    updateProperties();

	connect(ui->sSpinBox, SIGNAL(editingFinished()), this, SLOT(on_sSpinBox_editingFinished()));
    connect(ui->sSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->nameBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->nameBox, SIGNAL(textChanged(const QString&)), this, SLOT(onValueChanged()));
    connect(ui->typeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onEditingFinished(int)));
    connect(ui->lengthSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->lengthSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));

    init_ = true;
}

BridgeSettings::~BridgeSettings()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
BridgeSettings::updateProperties()
{
    if (bridge_)
    {
        ui->nameBox->setText(bridge_->getName());
        ui->idLabel->setText(bridge_->getId().speakingName());
        ui->sSpinBox->setValue(bridge_->getSStart());
        ui->typeComboBox->setCurrentIndex(bridge_->getType());

        ui->lengthSpinBox->setValue(bridge_->getLength());
    }
}

//################//
// SLOTS          //
//################//
void
BridgeSettings::onEditingFinished(int i)
{
    if (ui->typeComboBox->currentIndex() != bridge_->getType())
    {
        valueChanged_ = true;
        onEditingFinished();
    }
}

void
BridgeSettings::onEditingFinished()
{
    if (valueChanged_)
    {
        QString filename = ui->nameBox->text();
        odrID newId = bridge_->getId();
		newId.setName(filename);
    

        SetBridgePropertiesCommand *command = new SetBridgePropertiesCommand(bridge_, newId, filename, ui->nameBox->text(), ui->typeComboBox->currentIndex(), ui->lengthSpinBox->value());
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
BridgeSettings::on_sSpinBox_editingFinished()
{
    if (valueChanged_)
    {
        MoveRoadSectionCommand *moveSectionCommand = new MoveRoadSectionCommand(bridge_, ui->sSpinBox->value(), RSystemElementRoad::DRS_BridgeSection);
        if (moveSectionCommand->isValid())
        {
            getProjectData()->getUndoStack()->beginMacro(QObject::tr("Change Start Value"));
            getProjectSettings()->executeCommand(moveSectionCommand);

            SetBridgePropertiesCommand *setPropertiesCommand = new SetBridgePropertiesCommand(bridge_, bridge_->getId(), bridge_->getFileName(), bridge_->getName(), bridge_->getType(), bridge_->getLength());
            getProjectSettings()->executeCommand(setPropertiesCommand);

            // Macro Command //
            //

            getProjectData()->getUndoStack()->endMacro();
        }

        valueChanged_ = false;

        QWidget * focusWidget = QApplication::focusWidget();
        if (focusWidget)
        {
            focusWidget->clearFocus();
        }
    }
}

void
    BridgeSettings::onValueChanged()
{
    valueChanged_ = true;
}

//##################//
// Observer Pattern //
//##################//

void
BridgeSettings::updateObserver()
{

    // Parent //
    //
    SettingsElement::updateObserver();
    if (isInGarbage())
    {
        return; // no need to go on
    }

    // Bridge //
    //
    int changes = bridge_->getBridgeChanges();

    if ((changes & Bridge::CEL_ParameterChange))
    {
        updateProperties();
    }
}
