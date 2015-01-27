/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/11/2010
**
**************************************************************************/

#include "junctionsettings.hpp"
#include "ui_junctionsettings.h"

// Data //
//
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/roadlink.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"
//#include "src/data/commands/junction_commands.hpp"
#include "src/data/commands/roadsystemcommands.hpp"

// Qt //
//
#include <QInputDialog>

// Utils //
//
#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

JunctionSettings::JunctionSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, RSystemElementJunction *junction)
    : SettingsElement(projectSettings, parentSettingsElement, junction)
    , ui(new Ui::JunctionSettings)
    , junction_(junction)
{
    ui->setupUi(this);

    // Initial Values //
    //
    updateProperties();
    updateConnections();
}

JunctionSettings::~JunctionSettings()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
JunctionSettings::updateProperties()
{
    ui->nameBox->setText(junction_->getName());
    ui->idBox->setText(junction_->getID());
}

void
JunctionSettings::updateConnections()
{
    ui->connectionTableWidget->clear();

    // Connections //
    //
    QStringList header;
    header << "id"
           << "inc.Rd."
           << "connect.Rd."
           << "contactPoint";
    ui->connectionTableWidget->setHorizontalHeaderLabels(header);

    QMultiMap<QString, JunctionConnection *> connections = junction_->getConnections();
    ui->connectionTableWidget->setRowCount(connections.size());
    int row = 0;
    foreach (JunctionConnection *element, connections)
    {
        ui->connectionTableWidget->setItem(row, 0, new QTableWidgetItem(element->getId()));
        ui->connectionTableWidget->setItem(row, 1, new QTableWidgetItem(element->getIncomingRoad()));
        ui->connectionTableWidget->setItem(row, 2, new QTableWidgetItem(element->getConnectingRoad()));
        ui->connectionTableWidget->setItem(row, 3, new QTableWidgetItem(element->getContactPoint()));
        ++row;
    }
}

//################//
// SLOTS          //
//################//

void
JunctionSettings::on_nameButton_released()
{
    // Open a small dialog an ask for a new name //
    //
    bool ok = false;
    QString newValue = QInputDialog::getText(this, tr("ODD: Junction Name"), tr("Please enter a new junction name:"), QLineEdit::Normal, junction_->getName(), &ok);
    if (ok && !newValue.isEmpty() && newValue != junction_->getName())
    {
        QString newId = junction_->getNewId(junction_, newValue);
        SetRSystemElementIdCommand *command = new SetRSystemElementIdCommand(junction_->getRoadSystem(), junction_, newId, NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
JunctionSettings::on_cleanConnectionsButton_released()
{

    QMultiMap<QString, JunctionConnection *> connections = junction_->getConnections();
    foreach (JunctionConnection *connection, connections)
    {
        RSystemElementRoad *incommingRoad = junction_->getRoadSystem()->getRoad(connection->getIncomingRoad());
        if (incommingRoad)
        {
            RoadLink *rl1 = incommingRoad->getPredecessor();
            if (rl1 && rl1->getElementId() == junction_->getID())
                continue;
            rl1 = incommingRoad->getSuccessor();
            if (rl1 && rl1->getElementId() == junction_->getID())
                continue;
            connections.remove(connection->getIncomingRoad(), connection);
        }
    }
}

//##################//
// Observer Pattern //
//##################//

void
JunctionSettings::updateObserver()
{

    // Parent //
    //
    SettingsElement::updateObserver();
    if (isInGarbage())
    {
        return; // no need to go on
    }

    // RSystemElement //
    //
    int changes = junction_->getRSystemElementChanges();

    // Road //
    //
    if ((changes & RSystemElement::CRE_NameChange)
        || (changes & RSystemElement::CRE_IdChange))
    {
        updateProperties();
    }

    // RSystemElementJunction //
    //
    changes = junction_->getJunctionChanges();

    // Road //
    //
    if ((changes & RSystemElementJunction::CJN_ConnectionChanged)
        || (changes & RSystemElementRoad::CRD_JunctionChange))
    {
        updateConnections();
    }
}
