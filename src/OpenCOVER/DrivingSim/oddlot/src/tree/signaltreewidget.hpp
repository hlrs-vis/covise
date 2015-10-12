/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/11/2010
**
**************************************************************************/

#ifndef SIGNALTREEWIDGET_HPP
#define SIGNALTREEWIDGET_HPP

#include <QTreeWidget>

class ProjectData;
class ProjectWidget;
class SignalManager;
//class SetSignalPropertiesCommand;
//class SettingsElement;
class MainWindow;

class SignalTreeWidget : public QTreeWidget
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit SignalTreeWidget(SignalManager *signalManager, MainWindow *mainWindow);
    virtual ~SignalTreeWidget();

	void setActiveProject(ProjectWidget *projectWidget)
	{
		projectWidget_ = projectWidget ;
	}

protected:
private:
    SignalTreeWidget(); /* not allowed */
    SignalTreeWidget(const SignalTreeWidget &); /* not allowed */
    SignalTreeWidget &operator=(const SignalTreeWidget &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    void selectionChanged(const QItemSelection &selected, const QItemSelection &deselected);

    //################//
    // PROPERTIES     //
    //################//

private:
	ProjectWidget *projectWidget_;
    ProjectData *projectData_; // Model, linked
	SignalManager *signalManager_;
	MainWindow *mainWindow_;
};

#endif // PROJECTTREEWIDGET_HPP
