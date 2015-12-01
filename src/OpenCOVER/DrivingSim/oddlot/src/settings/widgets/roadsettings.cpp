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

#include "roadsettings.hpp"
#include "ui_roadsettings.h"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/roadlink.hpp"
#include "src/data/commands/roadcommands.hpp"
#include "src/data/commands/roadsystemcommands.hpp"
#include "src/data/prototypemanager.hpp"
#include "src/gui/projectwidget.hpp"
#include "src/mainwindow.hpp"

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

RoadSettings::RoadSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, RSystemElementRoad *road)
    : SettingsElement(projectSettings, parentSettingsElement, road)
    , ui(new Ui::RoadSettings)
    , road_(road)
    , init_(false)
{
    prototypeManager_ = getProjectSettings()->getProjectWidget()->getMainWindow()->getPrototypeManager();

    ui->setupUi(this);
    addLaneSectionPrototypes();

    // Initial Values //
    //
    updateProperties();
    updateRoadLinks();
    //	updateSectionCount();

    connect(ui->nameBox, SIGNAL(editingFinished()), this, SLOT(on_editingFinished()));
    init_ = true;
}

RoadSettings::~RoadSettings()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//
void
RoadSettings::addLaneSectionPrototypes()
{
    foreach (const PrototypeContainer<RSystemElementRoad *> *container, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_LaneSectionPrototype))
    {
        ui->laneSectionComboBox->addItem(container->getPrototypeIcon(), container->getPrototypeName());
    }
    foreach (const PrototypeContainer<RSystemElementRoad *> *container, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_RoadTypePrototype))
    {
        ui->roadTypeComboBox->addItem(container->getPrototypeIcon(), container->getPrototypeName());
    }
    foreach (const PrototypeContainer<RSystemElementRoad *> *container, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_ElevationPrototype))
    {
        ui->elevationComboBox->addItem(container->getPrototypeIcon(), container->getPrototypeName());
    }
    foreach (const PrototypeContainer<RSystemElementRoad *> *container, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_SuperelevationPrototype))
    {
        ui->superelevationComboBox->addItem(container->getPrototypeIcon(), container->getPrototypeName());
    }
    foreach (const PrototypeContainer<RSystemElementRoad *> *container, prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_CrossfallPrototype))
    {
        ui->crossfallComboBox->addItem(container->getPrototypeIcon(), container->getPrototypeName());
    }
}

void
RoadSettings::updateProperties()
{
    ui->nameBox->setText(road_->getName());
    ui->idLabel->setText(road_->getID());
    ui->lengthBox->setValue(road_->getLength());
    ui->junctionBox->setText(road_->getJunction());
    if (road_->getJunction() == QString("-1") || road_->getJunction() != QString(""))
    {
        ui->junctionBox->setEnabled(false);
    }
    else
    {
        ui->junctionBox->setEnabled(true);
    }
}

void
RoadSettings::updateRoadLinks()
{
    if (road_->getPredecessor())
    {
        RoadLink *link = road_->getPredecessor();
        QString text = link->getElementType().append(" ").append(link->getElementId()).append(" ").append(link->getContactPoint());
        ui->predecessorBox->setText(text);
        ui->predecessorBox->setEnabled(true);
    }
    else
    {
        ui->predecessorBox->setText(tr("none"));
        ui->predecessorBox->setEnabled(false);
    }
    if (road_->getSuccessor())
    {
        RoadLink *link = road_->getSuccessor();
        QString text = link->getElementType().append(" ").append(link->getElementId()).append(" ").append(link->getContactPoint());
        ui->successorBox->setText(text);
        ui->successorBox->setEnabled(true);
    }
    else
    {
        ui->successorBox->setText(tr("none"));
        ui->successorBox->setEnabled(false);
    }
}

//void
//	RoadSettings
//	::updateSectionCount()
//{
//}

//################//
// SLOTS          //
//################//

void
RoadSettings::on_editingFinished()
{
    QString filename = ui->nameBox->text();
    if (filename != road_->getName())
    {
        QString newId;
        QStringList parts = road_->getID().split("_");

        if (parts.size() > 2)
        {
            newId = QString("%1_%2_%3").arg(parts.at(0)).arg(parts.at(1)).arg(filename); 
        }
        else
        {
            newId = road_->getRoadSystem()->getUniqueId(road_->getID(), filename);
        }
        SetRSystemElementIdCommand *command = new SetRSystemElementIdCommand(road_->getRoadSystem(), road_, newId, filename, NULL);
        getProjectSettings()->executeCommand(command);
    }

 
}
void
RoadSettings::on_addButton_released()
{
    // Open a dialog asking for the junction //
    //
    QStringList junctions;
    foreach (RSystemElementJunction *junction, getProjectData()->getRoadSystem()->getJunctions())
    {
        junctions.append(junction->getID());
    }
    bool ok = false;
    QString newValue = QInputDialog::getItem(this, tr("ODD: Junction"), tr("Please select a junction:"), junctions, junctions.size() - 1, false, &ok);
    if (ok && !newValue.isEmpty())
    {
        RSystemElementJunction *junction = getProjectData()->getRoadSystem()->getJunction(newValue);
        AddToJunctionCommand *command = new AddToJunctionCommand(getProjectData()->getRoadSystem(), road_, junction, NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
RoadSettings::on_newButton_released()
{

    AddToJunctionCommand *command = new AddToJunctionCommand(getProjectData()->getRoadSystem(), road_, NULL, NULL);
    getProjectSettings()->executeCommand(command);
}

void
RoadSettings::on_laneSectionComboBox_activated(int id)
{

    QList<PrototypeContainer<RSystemElementRoad *> *> prototypeList = prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_LaneSectionPrototype);
    RSystemElementRoad *lanePrototype = prototypeList.at(id)->getPrototype();

    QList<DataElement *> selectedElements = getProjectData()->getSelectedElements();

    // Macro Command //
    //
    int numberOfSelectedElements = selectedElements.size();
    if (numberOfSelectedElements > 1)
    {
        getProjectData()->getUndoStack()->beginMacro(QObject::tr("Change Lane Prototype"));
    }

    // Change types of selected items //
    //
    foreach (DataElement *element, selectedElements)
    {
        RSystemElementRoad *road = dynamic_cast<RSystemElementRoad *>(element);
        if (road)
        {
            ChangeLanePrototypeCommand *command = new ChangeLanePrototypeCommand(road, lanePrototype, NULL);
            getProjectSettings()->executeCommand(command);
        }
    }

    // Macro Command //
    //
    if (numberOfSelectedElements > 1)
    {
        getProjectData()->getUndoStack()->endMacro();
    }
}

void
RoadSettings::on_roadTypeComboBox_activated(int id)
{

    QList<PrototypeContainer<RSystemElementRoad *> *> prototypeList = prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_RoadTypePrototype);
    RSystemElementRoad *roadTypePrototype = prototypeList.at(id)->getPrototype();

    QList<DataElement *> selectedElements = getProjectData()->getSelectedElements();

    // Macro Command //
    //
    int numberOfSelectedElements = selectedElements.size();
    if (numberOfSelectedElements > 1)
    {
        getProjectData()->getUndoStack()->beginMacro(QObject::tr("Change RoadType Prototype"));
    }

    // Change types of selected items //
    //
    foreach (DataElement *element, selectedElements)
    {
        RSystemElementRoad *road = dynamic_cast<RSystemElementRoad *>(element);
        if (road)
        {
            ChangeRoadTypePrototypeCommand *command = new ChangeRoadTypePrototypeCommand(road, roadTypePrototype, NULL);
            getProjectSettings()->executeCommand(command);
        }
    }

    // Macro Command //
    //
    if (numberOfSelectedElements > 1)
    {
        getProjectData()->getUndoStack()->endMacro();
    }
}

void
RoadSettings::on_elevationComboBox_activated(int id)
{

    QList<PrototypeContainer<RSystemElementRoad *> *> prototypeList = prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_ElevationPrototype);
    RSystemElementRoad *elevationPrototype = prototypeList.at(id)->getPrototype();

    QList<DataElement *> selectedElements = getProjectData()->getSelectedElements();

    // Macro Command //
    //
    int numberOfSelectedElements = selectedElements.size();
    if (numberOfSelectedElements > 1)
    {
        getProjectData()->getUndoStack()->beginMacro(QObject::tr("Change Elevation Prototype"));
    }

    // Change types of selected items //
    //
    foreach (DataElement *element, selectedElements)
    {
        RSystemElementRoad *road = dynamic_cast<RSystemElementRoad *>(element);
        if (road)
        {
            ChangeElevationPrototypeCommand *command = new ChangeElevationPrototypeCommand(road, elevationPrototype, NULL);
            getProjectSettings()->executeCommand(command);
        }
    }

    // Macro Command //
    //
    if (numberOfSelectedElements > 1)
    {
        getProjectData()->getUndoStack()->endMacro();
    }
}

void
RoadSettings::on_superelevationComboBox_activated(int id)
{

    QList<PrototypeContainer<RSystemElementRoad *> *> prototypeList = prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_SuperelevationPrototype);
    RSystemElementRoad *superelevationPrototype = prototypeList.at(id)->getPrototype();

    QList<DataElement *> selectedElements = getProjectData()->getSelectedElements();

    // Macro Command //
    //
    int numberOfSelectedElements = selectedElements.size();
    if (numberOfSelectedElements > 1)
    {
        getProjectData()->getUndoStack()->beginMacro(QObject::tr("Change Superelevation Prototype"));
    }

    // Change types of selected items //
    //
    foreach (DataElement *element, selectedElements)
    {
        RSystemElementRoad *road = dynamic_cast<RSystemElementRoad *>(element);
        if (road)
        {
            ChangeSuperelevationPrototypeCommand *command = new ChangeSuperelevationPrototypeCommand(road, superelevationPrototype, NULL);
            getProjectSettings()->executeCommand(command);
        }
    }

    // Macro Command //
    //
    if (numberOfSelectedElements > 1)
    {
        getProjectData()->getUndoStack()->endMacro();
    }
}

void
RoadSettings::on_crossfallComboBox_activated(int id)
{

    QList<PrototypeContainer<RSystemElementRoad *> *> prototypeList = prototypeManager_->getRoadPrototypes(PrototypeManager::PTP_CrossfallPrototype);
    RSystemElementRoad *crossfallPrototype = prototypeList.at(id)->getPrototype();

    QList<DataElement *> selectedElements = getProjectData()->getSelectedElements();

    // Macro Command //
    //
    int numberOfSelectedElements = selectedElements.size();
    if (numberOfSelectedElements > 1)
    {
        getProjectData()->getUndoStack()->beginMacro(QObject::tr("Change Crossfall Prototype"));
    }

    // Change types of selected items //
    //
    foreach (DataElement *element, selectedElements)
    {
        RSystemElementRoad *road = dynamic_cast<RSystemElementRoad *>(element);
        if (road)
        {
            ChangeCrossfallPrototypeCommand *command = new ChangeCrossfallPrototypeCommand(road, crossfallPrototype, NULL);
            getProjectSettings()->executeCommand(command);
        }
    }

    // Macro Command //
    //
    if (numberOfSelectedElements > 1)
    {
        getProjectData()->getUndoStack()->endMacro();
    }
}

//##################//
// Observer Pattern //
//##################//

void
RoadSettings::updateObserver()
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
    int changes = road_->getRSystemElementChanges();

    // Road //
    //
    if ((changes & RSystemElement::CRE_NameChange)
        || (changes & RSystemElement::CRE_IdChange))
    {
        updateProperties();
    }

    // RSystemElementRoad //
    //
    changes = road_->getRoadChanges();

    // Road //
    //
    if ((changes & RSystemElementRoad::CRD_LengthChange)
        || (changes & RSystemElementRoad::CRD_JunctionChange))
    {
        updateProperties();
    }

    if ((changes & RSystemElementRoad::CRD_PredecessorChange)
        || (changes & RSystemElementRoad::CRD_SuccessorChange))
    {
        updateRoadLinks();
    }
}
