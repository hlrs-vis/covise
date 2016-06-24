/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/8/2010
**
**************************************************************************/

#include "superelevationwizard.hpp"
#include "ui_superelevationwizard.h"

// Data //
//
#include "src/data/projectdata.hpp"

#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/elevationsection.hpp"

#include "src/data/scenerysystem/scenerysystem.hpp"
#include "src/data/scenerysystem/heightmap.hpp"

#include "src/data/commands/superelevationsectioncommands.hpp"

#include "src/cover/coverconnection.hpp"

//################//
// CONSTRUCTOR    //
//################//

SuperelevationWizard::SuperelevationWizard(ProjectData *projectData, QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::SuperelevationWizard)
    , projectData_(projectData)
{
    ui->setupUi(this);

    init();
}

SuperelevationWizard::~SuperelevationWizard()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
SuperelevationWizard::init()
{
    // Signals //
    //
    connect(ui->selectAllRoads, SIGNAL(released()), this, SLOT(selectAllRoads()));
    connect(ui->deselectAllRoads, SIGNAL(released()), this, SLOT(deselectAllRoads()));

    connect(ui->selectAllMaps, SIGNAL(released()), this, SLOT(selectAllMaps()));
    connect(ui->deselectAllMaps, SIGNAL(released()), this, SLOT(deselectAllMaps()));

    connect(ui->useCubicBox, SIGNAL(stateChanged(int)), this, SLOT(approximationMethod(int)));

    connect(ui->roadsList, SIGNAL(itemSelectionChanged()), this, SLOT(validateRunButton()));
    connect(ui->heightmapsList, SIGNAL(itemSelectionChanged()), this, SLOT(validateRunButton()));
    validateRunButton(); // run

    connect(ui->buttonBox->button(QDialogButtonBox::Apply), SIGNAL(released()), this, SLOT(runCalculation()));

    // Roads //
    //
    ui->roadsList->setSelectionMode(QAbstractItemView::ExtendedSelection);

    foreach (RSystemElementRoad *road, projectData_->getRoadSystem()->getRoads())
    {
        QListWidgetItem *item = new QListWidgetItem(road->getIdName());
        item->setData(Qt::UserRole, QVariant::fromValue((void *)road));
        ui->roadsList->addItem(item);
        if (road->isElementSelected())
        {
            item->setSelected(true);
        }
    }

    // Maps //
    //
    ui->heightmapsList->setSelectionMode(QAbstractItemView::ExtendedSelection);

    foreach (Heightmap *map, projectData_->getScenerySystem()->getHeightmaps())
    {
        QListWidgetItem *item = new QListWidgetItem(map->getId().append(" (").append(map->getFilename()).append(")"));
        item->setData(Qt::UserRole, qVariantFromValue((void *)map));
        ui->heightmapsList->addItem(item);
        if (map->isElementSelected())
        {
            item->setSelected(true);
        }
    }
}

//################//
// EVENTS         //
//################//

//################//
// SLOTS          //
//################//

void
SuperelevationWizard::selectAllRoads()
{
    ui->roadsList->selectAll();
}

void
SuperelevationWizard::selectAllMaps()
{
    ui->heightmapsList->selectAll();
}

void
SuperelevationWizard::deselectAllRoads()
{
    ui->roadsList->clearSelection();
}

void
SuperelevationWizard::deselectAllMaps()
{
    ui->heightmapsList->clearSelection();
}

void
SuperelevationWizard::approximationMethod(int state)
{
    if (state == 0)
    {
        ui->smoothEdgesLabel->setEnabled(true);
        ui->smoothEdges->setEnabled(true);
        ui->smoothEdges->setCheckState(Qt::Checked);
        ui->radiusLabel->setEnabled(true);
        ui->radius->setEnabled(true);
    }
    else
    {
        ui->smoothEdgesLabel->setEnabled(false);
        ui->smoothEdges->setEnabled(false);
        ui->smoothEdges->setCheckState(Qt::Unchecked);
        ui->radiusLabel->setEnabled(false);
        ui->radius->setEnabled(false);
    }
}

void
SuperelevationWizard::validateRunButton()
{
    // Enable the apply button only if there are selected roads and maps //
    //
    if (ui->roadsList->selectedItems().isEmpty() || (ui->heightmapsList->selectedItems().isEmpty() && !COVERConnection::instance()->isConnected()))
    {
        ui->buttonBox->button(QDialogButtonBox::Apply)->setEnabled(false);
    }
    else
    {
        ui->buttonBox->button(QDialogButtonBox::Apply)->setEnabled(true);
    }
}

void
SuperelevationWizard::runCalculation()
{
    QList<Heightmap *> maps;

    foreach (QListWidgetItem *item, ui->heightmapsList->selectedItems())
    {
        // Parse back to map pointer //
        //
        void *pointer = item->data(Qt::UserRole).value<void *>();
        Heightmap *map = static_cast<Heightmap *>(pointer);
        maps.append(map);
    }

    // Macro Command //
    //
    projectData_->getUndoStack()->beginMacro(tr("Apply Superelevation Heightmaps"));

    // Roads: Commands //
    //
    foreach (QListWidgetItem *item, ui->roadsList->selectedItems())
    {
        // Parse back to road pointer //
        //
        void *pointer = item->data(Qt::UserRole).value<void *>();
        RSystemElementRoad *road = static_cast<RSystemElementRoad *>(pointer);

        ApplyHeightMapSuperelevationCommand *command = new ApplyHeightMapSuperelevationCommand(road, maps, ui->heightBox->value(), ui->sampleBox->value(), ui->deviationBox->value(), ui->filterBox->value(), ui->useCubicBox->isChecked(), ui->smoothEdges->isChecked(), ui->radius->value());
        if (command->isValid())
        {
            projectData_->getUndoStack()->push(command);
        }
        else
        {
            delete command;
        }
    }

    // Macro Command //
    //
    projectData_->getUndoStack()->endMacro();

    // Quit //
    //
    done(0);
}
