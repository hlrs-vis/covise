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

#ifndef ROADLINKWIZARD_HPP
#define ROADLINKWIZARD_HPP

#include <QDialog>

class ProjectData;

namespace Ui
{
class RoadLinkWizard;
}

class RoadLinkWizard : public QDialog
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadLinkWizard(ProjectData *projectData, QWidget *parent = 0);
    virtual ~RoadLinkWizard();

protected:
private:
    RoadLinkWizard(); /* not allowed */
    RoadLinkWizard(const RoadLinkWizard &); /* not allowed */
    RoadLinkWizard &operator=(const RoadLinkWizard &); /* not allowed */

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
    void deselectAllRoads();
    void validateRunButton();
    void runCalculation();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::RoadLinkWizard *ui;

    ProjectData *projectData_;
};

#endif // ROADLINKWIZARD_HPP
