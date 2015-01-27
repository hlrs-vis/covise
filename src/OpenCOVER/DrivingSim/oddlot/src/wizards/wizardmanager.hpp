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

#ifndef WIZARDMANAGER_HPP
#define WIZARDMANAGER_HPP

#include <QObject>

class MainWindow;

#include <QAction>

class WizardManager : public QObject
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit WizardManager(MainWindow *mainWindow);
    virtual ~WizardManager();

protected:
private:
    WizardManager(); /* not allowed */
    WizardManager(const WizardManager &); /* not allowed */
    WizardManager &operator=(const WizardManager &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // SLOTS          //
    //################//

public slots:
    void runElevationWizard();
    void runSuperelevationWizard();
    void runFlatJunctionsWizard();
    void runRoadLinkWizard();

    //################//
    // PROPERTIES     //
    //################//

private:
    MainWindow *mainWindow_;

    QAction *elevatioWizardAction_;
    QAction *supereelevatioWizardAction_;
};

#endif // WIZARDMANAGER_HPP
