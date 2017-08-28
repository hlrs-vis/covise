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

#ifndef CATALOGTREEWIDGET_HPP
#define CATALOGTREEWIDGET_HPP

#include "src/util/odd.hpp"
#include "src/data/observer.hpp"

#include "oscCatalog.h"

#include <QTreeWidget>

class ProjectData;
class ProjectWidget;
class MainWindow;
class ToolAction;
class ToolManager;
class OpenScenarioEditor;
class OSCBase;
class OSCElement;
class QTreeWidgetItem;

namespace OpenScenario
{
class oscObject;
class oscObjectBase;
class OpenScenarioBase;
class oscMember;
class oscCatalog;
}

class CatalogTreeWidget : public QTreeWidget, public Observer
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit CatalogTreeWidget(MainWindow *mainWindow, OpenScenario::oscCatalog *catalog);
    virtual ~CatalogTreeWidget();

/*	void setActiveProject(ProjectWidget *projectWidget)
	{
		projectWidget_ = projectWidget ;
	} */

	void setOpenScenarioEditor(OpenScenarioEditor *oscEditor);


	// Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    CatalogTreeWidget(); /* not allowed */
    CatalogTreeWidget(const CatalogTreeWidget &); /* not allowed */
    CatalogTreeWidget &operator=(const CatalogTreeWidget &); /* not allowed */

    void init();

	void createTree();
	QTreeWidgetItem *getItem(const QString &name);

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
    // SLOTS          //
    //################//

public slots:
	void onVisibilityChanged(bool);

    //################//
    // PROPERTIES     //
    //################//

private:
	ProjectWidget *projectWidget_;
    ProjectData *projectData_; // Model, linked
	MainWindow *mainWindow_;
	OpenScenarioEditor *oscEditor_;
	ToolManager *toolManager_;

	// OpenScenario Base //
	//
	OSCBase *base_;
	OpenScenario::OpenScenarioBase *openScenarioBase_;

	OpenScenario::oscObjectBase *objectBase_;
	// temporary: test base
	OSCElement *testBase_;
	std::string catalogName_; //catalog name
	std::string catalogType_;
	OpenScenario::oscCatalog *catalog_;
	QString directoryPath_;

	OSCElement *oscElement_;

	OpenScenario::oscMember *currentMember_;

	ODD::ToolId currentTool_;
};

#endif // CATALOGTREEWIDGET_HPP
