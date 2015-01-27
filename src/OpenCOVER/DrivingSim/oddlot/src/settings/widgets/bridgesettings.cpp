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

#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/roadsystem/sections/bridgeobject.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/roadlink.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/commands/roadcommands.hpp"
#include "src/data/commands/roadsystemcommands.hpp"
#include "src/data/commands/signalcommands.hpp"
#include "src/data/commands/roadsectioncommands.hpp"
#include "src/data/signalmanager.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"

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
{
    ui->setupUi(this);

    // Initial Values //
    //
    updateProperties();

    connect(ui->nameBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    //	connect(ui->sSpinBox, SIGNAL(editingFinished()), this, SLOT(onSEditingFinished()));
    connect(ui->typeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onEditingFinished()));
    connect(ui->lengthSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));

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
        ui->idBox->setText(bridge_->getId());
        ui->sSpinBox->setValue(bridge_->getSStart());
        ui->typeComboBox->setCurrentIndex(bridge_->getType());
        //	ui->typeComboBox->setCurrentIndex(bridge_->getType()-100001);

        ui->lengthSpinBox->setValue(bridge_->getLength());
    }
}

//################//
// SLOTS          //
//################//

void
BridgeSettings::onEditingFinished()
{
    QString filename = ui->nameBox->text();
    QString newId = bridge_->getNewId(filename);
    bridge_->setId(newId);

    SetBridgePropertiesCommand *command = new SetBridgePropertiesCommand(bridge_, bridge_->getId(), filename, ui->nameBox->text(), ui->typeComboBox->currentIndex(), ui->lengthSpinBox->value());
    getProjectSettings()->executeCommand(command);
}

void
BridgeSettings::on_sSpinBox_editingFinished()
{
    QList<DataElement *> selectedElements = getProjectData()->getSelectedElements();
    int numberOfSelectedElements = selectedElements.size();

    getProjectData()->getUndoStack()->beginMacro(QObject::tr("Change Signal Type"));

    double s = ui->sSpinBox->value();

    foreach (DataElement *element, selectedElements)
    {

        Bridge *bridge = dynamic_cast<Bridge *>(element);
        if (bridge)
        {
            MoveRoadSectionCommand *moveSectionCommand = new MoveRoadSectionCommand(bridge_, s, RSystemElementRoad::DRS_BridgeSection);
            getProjectSettings()->executeCommand(moveSectionCommand);

            SetBridgePropertiesCommand *setPropertiesCommand = new SetBridgePropertiesCommand(bridge, bridge->getId(), bridge->getFileName(), bridge->getName(), bridge->getType(), bridge->getLength());
            getProjectSettings()->executeCommand(setPropertiesCommand);
        }
    }

    // Macro Command //
    //

    getProjectData()->getUndoStack()->endMacro();
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
