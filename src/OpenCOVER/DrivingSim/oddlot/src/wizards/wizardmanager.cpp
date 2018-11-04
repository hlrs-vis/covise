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

#include "wizardmanager.hpp"

#include "src/mainwindow.hpp"
#include "src/gui/projectwidget.hpp"
#include "src/data/projectdata.hpp"

// Wizards //
//
#include "elevationwizard.hpp"
#include "superelevationwizard.hpp"
#include "flatjunctionswizard.hpp"
#include "circlewizard.hpp"
#include "roadlinkwizard.hpp"

// Qt //
//
#include <QMenu>

//################//
// CONSTRUCTOR    //
//################//

WizardManager::WizardManager(MainWindow *mainWindow)
    : QObject(mainWindow)
    , mainWindow_(mainWindow)
{
    init();
}

WizardManager::~WizardManager()
{
}

//################//
// FUNCTIONS      //
//################//

void
WizardManager::init()
{
    // Elevation wizard //
    //
    elevatioWizardAction_ = new QAction(tr("&Elevation wizard"), this);
    elevatioWizardAction_->setStatusTip(tr("Use heightmaps to set the road elevation."));
    connect(elevatioWizardAction_, SIGNAL(triggered()), this, SLOT(runElevationWizard()));
    connect(mainWindow_, SIGNAL(hasActiveProject(bool)), elevatioWizardAction_, SLOT(setEnabled(bool)));

    mainWindow_->getWizardsMenu()->addAction(elevatioWizardAction_);

    // Superelevation wizard //
    //
    supereelevatioWizardAction_ = new QAction(tr("&Superelevation wizard"), this);
    supereelevatioWizardAction_->setStatusTip(tr("Use heightmaps to set the road elevation."));
    connect(supereelevatioWizardAction_, SIGNAL(triggered()), this, SLOT(runSuperelevationWizard()));
    connect(mainWindow_, SIGNAL(hasActiveProject(bool)), supereelevatioWizardAction_, SLOT(setEnabled(bool)));

    mainWindow_->getWizardsMenu()->addAction(supereelevatioWizardAction_);

    // Flat junctions wizard //
    //
    QAction *flatJunctionsWizardAction = new QAction(tr("&Flat junctions wizard"), this);
    flatJunctionsWizardAction->setStatusTip(tr("Flattens the elevation of junctions."));
    connect(flatJunctionsWizardAction, SIGNAL(triggered()), this, SLOT(runFlatJunctionsWizard()));
    connect(mainWindow_, SIGNAL(hasActiveProject(bool)), flatJunctionsWizardAction, SLOT(setEnabled(bool)));

    mainWindow_->getWizardsMenu()->addAction(flatJunctionsWizardAction);
	
	// Circle wizard //
	//
	QAction *circleWizardAction = new QAction(tr("&Circle wizard"), this);
	flatJunctionsWizardAction->setStatusTip(tr("creates Circular tracks."));
	connect(circleWizardAction, SIGNAL(triggered()), this, SLOT(runCircleWizard()));
	connect(mainWindow_, SIGNAL(hasActiveProject(bool)), circleWizardAction, SLOT(setEnabled(bool)));

	mainWindow_->getWizardsMenu()->addAction(circleWizardAction);

    // Road link wizard //
    //
    QAction *roadLinkWizardAction = new QAction(tr("&Road link wizard"), this);
    roadLinkWizardAction->setStatusTip(tr("Automatically links roads."));
    connect(roadLinkWizardAction, SIGNAL(triggered()), this, SLOT(runRoadLinkWizard()));
    connect(mainWindow_, SIGNAL(hasActiveProject(bool)), roadLinkWizardAction, SLOT(setEnabled(bool)));

    //	mainWindow_->getWizardsMenu()->addAction(roadLinkWizardAction);
}

//################//
// EVENTS         //
//################//

//################//
// SLOTS          //
//################//

void
WizardManager::runElevationWizard()
{
    if (mainWindow_->getActiveProject())
    {
        ElevationWizard *wizard = new ElevationWizard(mainWindow_->getActiveProject()->getProjectData(), mainWindow_);
        wizard->exec();
    }
}

void
WizardManager::runSuperelevationWizard()
{
    if (mainWindow_->getActiveProject())
    {
        SuperelevationWizard *wizard = new SuperelevationWizard(mainWindow_->getActiveProject()->getProjectData(), mainWindow_);
        wizard->exec();
    }
}

void
WizardManager::runFlatJunctionsWizard()
{
    if (mainWindow_->getActiveProject())
    {
        FlatJunctionsWizard *wizard = new FlatJunctionsWizard(mainWindow_->getActiveProject()->getProjectData(), mainWindow_);
        wizard->exec();
    }
}

void
WizardManager::runCircleWizard()
{
	if (mainWindow_->getActiveProject())
	{
		CircleWizard *wizard = new CircleWizard(mainWindow_->getActiveProject()->getProjectData(), mainWindow_);
		wizard->exec();
	}
}

void
WizardManager::runRoadLinkWizard()
{
    if (mainWindow_->getActiveProject())
    {
        RoadLinkWizard *wizard = new RoadLinkWizard(mainWindow_->getActiveProject()->getProjectData(), mainWindow_);
        wizard->exec();
    }
}
