/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/29/2010
**
**************************************************************************/

#include "roadlinkwizard.hpp"
#include "ui_roadlinkwizard.h"

// Data //
//
#include "src/data/projectdata.hpp"

#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Qt //
//
#include <QPointF>

//################//
// CONSTRUCTOR    //
//################//

RoadLinkWizard::RoadLinkWizard(ProjectData *projectData, QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::RoadLinkWizard)
    , projectData_(projectData)
{
    ui->setupUi(this);

    init();
}

RoadLinkWizard::~RoadLinkWizard()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
RoadLinkWizard::init()
{
    // Signals //
    //
    connect(ui->selectAllRoads, SIGNAL(released()), this, SLOT(selectAllRoads()));
    connect(ui->deselectAllRoads, SIGNAL(released()), this, SLOT(deselectAllRoads()));

    connect(ui->roadsList, SIGNAL(itemSelectionChanged()), this, SLOT(validateRunButton()));
    validateRunButton(); // run for the first time

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
}

//################//
// EVENTS         //
//################//

//################//
// SLOTS          //
//################//

void
RoadLinkWizard::selectAllRoads()
{
    ui->roadsList->selectAll();
}

void
RoadLinkWizard::deselectAllRoads()
{
    ui->roadsList->clearSelection();
}

void
RoadLinkWizard::validateRunButton()
{
    // Enable the apply button only if there are selected roads //
    //
    if (ui->roadsList->selectedItems().isEmpty())
    {
        ui->buttonBox->button(QDialogButtonBox::Apply)->setEnabled(false);
    }
    else
    {
        ui->buttonBox->button(QDialogButtonBox::Apply)->setEnabled(true);
    }
}

void
RoadLinkWizard::runCalculation()
{
    //	bool keepExisting = ui->keepExistingBox->isChecked();
    //	double maxDistance = ui->distanceBox->value();

    //	QMultiMap<double, RSystemElementRoad *> roadXCoords;

    //	foreach(QListWidgetItem * item, ui->roadsList->selectedItems())
    //	{
    //		// Parse back to road pointer //
    //		//
    //		void * pointer = item->data(Qt::UserRole).value<void *>();
    //		RSystemElementRoad * road = static_cast<RSystemElementRoad *>(pointer);

    //		roadXCoords.insert(road->getGlobalPoint(0.0).x(), road);
    //		roadXCoords.insert(road->getGlobalPoint(road->getLength()).x(), road);
    //	}

    //	QMultiMap<double, RSystemElementRoad *>::const_iterator i = roadXCoords.constBegin();
    //	RSystemElementRoad * roadOne = NULL;
    //	RSystemElementRoad * roadTwo = NULL;
    //	while (i != roadXCoords.constEnd())
    //	{
    //		RSystemElementRoad * road = i.value();

    //		// todo road == roadone/two

    //		if(roadTwo && roadOne)
    //		{

    //		}

    //		else if(roadOne)
    //		{
    //			if()
    //		}

    //		else
    //		{
    //			roadOne = road;
    //		}

    //		qDebug() << i.key();
    //		++i;
    //	}

    // Quit //
    //
    done(0);
}
