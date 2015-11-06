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

#include "tunnelsettings.hpp"
#include "ui_tunnelsettings.h"

#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/roadsystem/sections/tunnelobject.hpp"
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

TunnelSettings::TunnelSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, Tunnel *tunnel)
    : SettingsElement(projectSettings, parentSettingsElement, tunnel)
    , ui(new Ui::TunnelSettings)
    , tunnel_(tunnel)
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
	connect(ui->lightingSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->lightingSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
	connect(ui->daylightSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->daylightSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));

    init_ = true;
}

TunnelSettings::~TunnelSettings()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
TunnelSettings::updateProperties()
{
    if (tunnel_)
    {
        ui->nameBox->setText(tunnel_->getName());
        ui->idBox->setText(tunnel_->getId());
        ui->sSpinBox->setValue(tunnel_->getSStart());
        ui->typeComboBox->setCurrentIndex(tunnel_->getType());

        ui->lengthSpinBox->setValue(tunnel_->getLength());
		ui->lightingSpinBox->setValue(tunnel_->getLighting());
		ui->daylightSpinBox->setValue(tunnel_->getDaylight());
    }
}

//################//
// SLOTS          //
//################//
void
TunnelSettings::onEditingFinished(int i)
{
    if (ui->typeComboBox->currentIndex() != tunnel_->getType())
    {
        valueChanged_ = true;
        onEditingFinished();
    }
}

void
TunnelSettings::onEditingFinished()
{
    if (valueChanged_)
    {
        QString filename = ui->nameBox->text();
        QString newId = tunnel_->getId();
        if (filename != tunnel_->getName())
        {
            QStringList parts = tunnel_->getId().split("_");

            if (parts.size() > 2)
            {
                newId = QString("%1_%2_%3").arg(parts.at(0)).arg(parts.at(1)).arg(filename); 
            }
            else
            {
                newId = tunnel_->getParentRoad()->getRoadSystem()->getUniqueId(tunnel_->getId(), filename);
            }
        }
    

        SetTunnelPropertiesCommand *command = new SetTunnelPropertiesCommand(tunnel_, newId, filename, ui->nameBox->text(), ui->typeComboBox->currentIndex(), ui->lengthSpinBox->value(), ui->lightingSpinBox->value(), ui->daylightSpinBox->value());
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
TunnelSettings::on_sSpinBox_editingFinished()
{
    if (valueChanged_)
    {
        MoveRoadSectionCommand *moveSectionCommand = new MoveRoadSectionCommand(tunnel_, ui->sSpinBox->value(), RSystemElementRoad::DRS_BridgeSection);
        if (moveSectionCommand->isValid())
        {
            getProjectData()->getUndoStack()->beginMacro(QObject::tr("Change Start Value"));
            getProjectSettings()->executeCommand(moveSectionCommand);

            SetTunnelPropertiesCommand *setPropertiesCommand = new SetTunnelPropertiesCommand(tunnel_, tunnel_->getId(), tunnel_->getFileName(), tunnel_->getName(), tunnel_->getType(), tunnel_->getLength(), tunnel_->getLighting(), tunnel_->getDaylight());
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
    TunnelSettings::onValueChanged()
{
    valueChanged_ = true;
}

//##################//
// Observer Pattern //
//##################//

void
TunnelSettings::updateObserver()
{

    // Parent //
    //
    SettingsElement::updateObserver();
    if (isInGarbage())
    {
        return; // no need to go on
    }

    // Tunnel //
    //
    int changes = tunnel_->getTunnelChanges();

    if ((changes & Tunnel::CEL_ParameterChange))
    {
        updateProperties();
    }
}
