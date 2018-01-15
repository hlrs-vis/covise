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

#ifndef CIRCLEWIZARD_HPP
#define CIRCLEWIZARD_HPP

#include <QDialog>

class ProjectData;

namespace Ui
{
class CircleWizard;
}

class CircleWizard : public QDialog
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CircleWizard(ProjectData *projectData, QWidget *parent = 0);
    virtual ~CircleWizard();

protected:
private:
    CircleWizard(); /* not allowed */
    CircleWizard(const CircleWizard &); /* not allowed */
    CircleWizard &operator=(const CircleWizard &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // SLOTS          //
    //################//

public slots:
    void runCalculation();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::CircleWizard *ui;

    ProjectData *projectData_;
};

#endif // CIRCLEWIZARD_HPP
