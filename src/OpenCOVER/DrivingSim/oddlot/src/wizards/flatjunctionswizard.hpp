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

#ifndef FLATJUNCTIONSWIZARD_HPP
#define FLATJUNCTIONSWIZARD_HPP

#include <QDialog>

namespace Ui
{
class FlatJunctionsWizard;
}

class ProjectData;

class FlatJunctionsWizard : public QDialog
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit FlatJunctionsWizard(ProjectData *projectData, QWidget *parent = 0);
    virtual ~FlatJunctionsWizard();

protected:
private:
    FlatJunctionsWizard(); /* not allowed */
    FlatJunctionsWizard(const FlatJunctionsWizard &); /* not allowed */
    FlatJunctionsWizard &operator=(const FlatJunctionsWizard &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // SLOTS          //
    //################//

public slots:
    void on_selectAll_released();
    void on_deselectAll_released();

    void validateRunButton();
    void runCalculation();

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::FlatJunctionsWizard *ui;

    ProjectData *projectData_;
};

#endif // FLATJUNCTIONSWIZARD_HPP
