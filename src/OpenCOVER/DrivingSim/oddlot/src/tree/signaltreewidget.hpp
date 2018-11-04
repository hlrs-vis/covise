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

#include "src/util/odd.hpp"

#include <QTreeWidget>
#include <QDrag>

class ProjectData;
class ProjectWidget;
class SignalManager;
class ToolManager;
class MainWindow;
class SignalEditor;
class ToolAction;

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

	void setSignalEditor(SignalEditor *signalEditor);

protected:

    virtual void mouseMoveEvent(QMouseEvent *event);
private:
    SignalTreeWidget(); /* not allowed */
    SignalTreeWidget(const SignalTreeWidget &); /* not allowed */
    SignalTreeWidget &operator=(const SignalTreeWidget &); /* not allowed */

    void PrepareDrag(const QIcon &icon);
    void init();

    //################//
    // EVENTS         //
    //################//

public:
    void selectionChanged(const QItemSelection &selected, const QItemSelection &deselected);

	//################//
	// SIGNALS        //
	//################//

signals:
    void toolAction(ToolAction *);  // This widget has to behave like a toolEditor and send the selected tool //

    //################//
    // PROPERTIES     //
    //################//

private:
	ProjectWidget *projectWidget_;
    ProjectData *projectData_; // Model, linked
	SignalManager *signalManager_;
	ToolManager *toolManager_;
	MainWindow *mainWindow_;
	SignalEditor *signalEditor_;
	ODD::ToolId currentTool_;

    QPointF dragStartPosition_;
};

#endif // PROJECTTREEWIDGET_HPP
