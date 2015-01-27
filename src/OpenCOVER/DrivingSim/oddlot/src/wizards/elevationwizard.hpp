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

#ifndef ELEVATIONWIZARD_HPP
#define ELEVATIONWIZARD_HPP

#include <QDialog>

class ProjectData;

namespace Ui
{
class ElevationWizard;
}

class ElevationWizard : public QDialog
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ElevationWizard(ProjectData *projectData, QWidget *parent = 0);
    virtual ~ElevationWizard();

protected:
private:
    ElevationWizard(); /* not allowed */
    ElevationWizard(const ElevationWizard &); /* not allowed */
    ElevationWizard &operator=(const ElevationWizard &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // SLOTS          //
    //################//

public slots:
    void selectAllRoads();
    void selectAllMaps();
    void deselectAllRoads();
    void deselectAllMaps();
    void approximationMethod(int state);
    void validateRunButton();
    void runCalculation();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::ElevationWizard *ui;

    ProjectData *projectData_;
};

#endif // ELEVATIONWIZARD_HPP
