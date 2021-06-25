/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   04.03.2010
**
**************************************************************************/

#include "odd.hpp"

#include "src/mainwindow.hpp"
#include "src/gui/projectwidget.hpp"
#include "src/data/projectdata.hpp"
#include "colorpalette.hpp"

ODD *ODD::instance_ = NULL;
MainWindow *ODD::mainWindow_ = NULL;
const QList<std::string> ODD::CATALOGLIST = QList<std::string>() << "VehicleCatalog" << "DriverCatalog" << "PedestrianCatalog" << "PedestrianControllerCatalog" << "MiscObjectCatalog" << "EnvironmentCatalog" << "ManeuverCatalog" << "TrajectoryCatalog" << "RouteCatalog";

ODD::ODD()
{
    colorPalette_ = new ColorPalette();
}

ODD *
ODD::instance()
{
    if (!instance_)
    {
        instance_ = new ODD();
    }
    return instance_;
}

/*! \brief Initializes the ODD with a MainWindow. This can only be done once.
*/
void
ODD::init(MainWindow *mainWindow)
{
    if (mainWindow_)
    {
        qDebug("Error 1003041242. ODD can not be initialized twice!");
    }
    if (!mainWindow)
    {
        qDebug("Error 1003041243. ODD can not be initialized with NULL!");
    }
    mainWindow_ = mainWindow;
}

/*! \brief Tells ODD that the MainWindow has been shut down.
*/
void
ODD::kill()
{
    mainWindow_ = NULL;
}

/*! \brief Returns the currently active project (ProjectWidget).
*/
ProjectWidget *
ODD::project()
{
    if (!mainWindow_)
    {
        //		qDebug("Error 1003041241. ODD not initialized!");
        return NULL;
    }

    return mainWindow_->getActiveProject();
}

/*! \brief Returns the currently active project (ProjectWidget).
*/
ProjectWidget *
ODD::getProjectWidget()
{
    if (!mainWindow_)
    {
        return NULL;
        //		qDebug("Error 1003041241. ODD not initialized!");
    }

    return mainWindow_->getActiveProject();
}

/*! \brief
*/
ProjectData *
ODD::getProjectData()
{
    if (getProjectWidget())
    {
        return getProjectWidget()->getProjectData();
    }
    else
    {
        return NULL;
    }
}

/*! \brief
*/
ChangeManager *
ODD::getChangeManager()
{
    if (getProjectData())
    {
        return getProjectData()->getChangeManager();
    }
    else
    {
        return NULL;
    }
}
