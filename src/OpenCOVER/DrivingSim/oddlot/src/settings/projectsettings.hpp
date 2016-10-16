/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/12/2010
**
**************************************************************************/

#ifndef PROJECTSETTINGS_HPP
#define PROJECTSETTINGS_HPP

#include <QWidget>
#include "src/data/observer.hpp"

#include "src/data/commands/datacommand.hpp"

namespace Ui
{
class ErrorMessageTree;
}

class ProjectWidget;

class ProjectData;

class ProjectSettingsVisitor;

class SettingsElement;

class QVBoxLayout;

class ProjectSettings : public QWidget, public Observer
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProjectSettings(ProjectWidget *projectWidget, ProjectData *projectData);
    virtual ~ProjectSettings();

    ProjectWidget *getProjectWidget() const
    {
        return projectWidget_;
    }
    ProjectData *getProjectData() const
    {
        return projectData_;
    }

    void addToGarbage(SettingsElement *element);

    // Commands //
    //
    bool executeCommand(DataCommand *command);

    // Obsever Pattern //
    //
    virtual void updateObserver();

	// Error Messages //
	//
	void printErrorMessage(const QString &text);

protected:
private:
    ProjectSettings(); /* not allowed */
    ProjectSettings(const ProjectSettings &); /* not allowed */
    ProjectSettings &operator=(const ProjectSettings &); /* not allowed */

    void updateWidget();

    //################//
    // EVENTS         //
    //################//

public:
//################//
// SIGNALS        //
//################//

signals:

    //################//
    // SLOTS          //
    //################//

public slots:

    void projectActivated(bool active);

    void garbageDisposal();

    //################//
    // PROPERTIES     //
    //################//

private:
    ProjectWidget *projectWidget_; // Project, linked

    ProjectData *projectData_; // Model, linked

    SettingsElement *settingsElement_; // owned

    QList<SettingsElement *> garbageList_;

    QVBoxLayout *settingsLayout_;

	Ui::ErrorMessageTree *ui;

};

#endif // PROJECTSETTINGS_HPP
