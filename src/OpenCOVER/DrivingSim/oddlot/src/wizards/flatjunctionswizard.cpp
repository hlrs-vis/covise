/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/3/2010
**
**************************************************************************/

#include "flatjunctionswizard.hpp"
#include "ui_flatjunctionswizard.h"

// Data //
//
#include "src/data/projectdata.hpp"

#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/sections/elevationsection.hpp"

#include "src/data/commands/elevationsectioncommands.hpp"

//################//
// CONSTRUCTOR    //
//################//

FlatJunctionsWizard::FlatJunctionsWizard(ProjectData *projectData, QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::FlatJunctionsWizard)
    , projectData_(projectData)
{
    ui->setupUi(this);

    init();
}

FlatJunctionsWizard::~FlatJunctionsWizard()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
FlatJunctionsWizard::init()
{
    validateRunButton(); // check for the first time

    // Signals //
    //
    connect(ui->buttonBox->button(QDialogButtonBox::Apply), SIGNAL(released()), this, SLOT(runCalculation()));
    connect(ui->junctionsList, SIGNAL(itemSelectionChanged()), this, SLOT(validateRunButton()));

    // Junctions //
    //
    ui->junctionsList->setSelectionMode(QAbstractItemView::ExtendedSelection);

    foreach (RSystemElementJunction *element, projectData_->getRoadSystem()->getJunctions())
    {
        QListWidgetItem *item = new QListWidgetItem(element->getIdName());
        item->setData(Qt::UserRole, QVariant::fromValue((void *)element));
        ui->junctionsList->addItem(item);
        if (element->isElementSelected())
        {
            item->setSelected(true);
        }
    }
}

//################//
// SLOTS          //
//################//

void
FlatJunctionsWizard::on_selectAll_released()
{
    ui->junctionsList->selectAll();
}

void
FlatJunctionsWizard::on_deselectAll_released()
{
    ui->junctionsList->clearSelection();
}

void
FlatJunctionsWizard::validateRunButton()
{
    // Enable the apply button only if there are selected junctions //
    //
    if (ui->junctionsList->selectedItems().isEmpty())
    {
        ui->buttonBox->button(QDialogButtonBox::Apply)->setEnabled(false);
    }
    else
    {
        ui->buttonBox->button(QDialogButtonBox::Apply)->setEnabled(true);
    }
}

void
FlatJunctionsWizard::runCalculation()
{
    // Macro Command //
    //
    projectData_->getUndoStack()->beginMacro(tr("Flatten Junctions"));

    // Roads: Commands //
    //
    foreach (QListWidgetItem *item, ui->junctionsList->selectedItems())
    {
        // Parse back to road pointer //
        //
        void *pointer = item->data(Qt::UserRole).value<void *>();
        RSystemElementJunction *junction = static_cast<RSystemElementJunction *>(pointer);

        FlatJunctionsElevationCommand *command = new FlatJunctionsElevationCommand(junction, ui->lengthBox->value());
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
