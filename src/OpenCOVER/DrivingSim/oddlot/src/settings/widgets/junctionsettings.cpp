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

    connect(ui->nameBox, SIGNAL(editingFinished()), this, SLOT(on_editingFinished()));
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
    ui->idLabel->setText(junction_->getID());
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
JunctionSettings::on_editingFinished()
{

    QString filename = ui->nameBox->text();
    if (filename != junction_->getName())
    {
        QString newId;
        QStringList parts = junction_->getID().split("_");

        if (parts.size() > 2)
        {
            newId = QString("%1_%2_%3").arg(parts.at(0)).arg(parts.at(1)).arg(filename); 
        }
        else
        {
            newId = junction_->getRoadSystem()->getUniqueId(junction_->getID(), filename);
        }
        SetRSystemElementIdCommand *command = new SetRSystemElementIdCommand(junction_->getRoadSystem(), junction_, newId, filename, NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
JunctionSettings::on_cleanConnectionsButton_released()
{
    QList<QTableWidgetItem *> selectedTableEntries = ui->connectionTableWidget->selectedItems();
    if (selectedTableEntries.empty())
    {
        junction_->delConnections();
    }
    else
    {
        QMultiMap<QString, JunctionConnection *>::ConstIterator iter = junction_->getConnections().constBegin();
        QList<JunctionConnection *> deleteList;
        for (int i = 0; i < selectedTableEntries.size(); i++)
        {
            iter += selectedTableEntries.at(i)->row();
            if (iter != junction_->getConnections().constEnd())
            {
                deleteList.append(iter.value());
            }
        }

        for (int i = 0; i < deleteList.size(); i++)
        {
            junction_->delConnection(deleteList.at(i));
        }
    }

 /*   QMultiMap<QString, JunctionConnection *> connections = junction_->getConnections();
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
    }*/
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
