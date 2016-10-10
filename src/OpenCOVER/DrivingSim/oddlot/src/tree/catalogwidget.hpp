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

#ifndef CATALOGWIDGET_HPP
#define CATALOGWIDGET_HPP

#include "src/util/odd.hpp"
#include "src/util/droparea.hpp"
#include "src/gui/tools/toolaction.hpp"

#include "oscCatalog.h"

#include <QWidget>

class MainWindow;
class OSCBase;
class OSCElement;
class CatalogTreeWidget;

class QDragEnterEvent;
class QDragLeaveEvent;
class QDragMoveEvent;
class QDropEvent;


namespace OpenScenario
{
class oscObjectBase;
}

class CatalogWidget : public QWidget
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit CatalogWidget(MainWindow *mainWindow, OpenScenario::oscCatalog *catalog, const QString &name);
    virtual ~CatalogWidget();

/*	void setActiveProject(ProjectWidget *projectWidget)
	{
		projectWidget_ = projectWidget ;
	} */


	CatalogTreeWidget *getCatalogTreeWidget()
	{
		return catalogTreeWidget_;
	}

	void onDeleteCatalogItem();

protected:
private:
    CatalogWidget(); /* not allowed */
    CatalogWidget(const CatalogWidget &); /* not allowed */
    CatalogWidget &operator=(const CatalogWidget &); /* not allowed */

    void init();

//################//
// SIGNALS        //
//################//

signals:
    void toolAction(ToolAction *);	// This widget has to behave like a toolEditor and send the selected tool //

	//################//
    // SLOTS          //
    //################//

public slots:
	void handleToolClick();

    //################//
    // PROPERTIES     //
    //################//

private:
    ProjectData *projectData_; // Model, linked
	MainWindow *mainWindow_;

	CatalogTreeWidget *catalogTreeWidget_;
	OpenScenario::oscCatalog *catalog_;

	// OpenScenario Base //
	//
	OSCBase *base_;

	const QString name_;	// catalog type

	OpenScenario::oscObjectBase *object_;
	OSCElement *oscElement_;

};

class CatalogDropArea : public DropArea
{
	//################//
    // FUNCTIONS      //
    //################//

public:
	explicit CatalogDropArea(CatalogWidget *catalogWidget, QPixmap *pixmap);

private:
    CatalogDropArea(); /* not allowed */
    CatalogDropArea(const CatalogWidget &, QPixmap *pixmap); /* not allowed */
    CatalogDropArea &operator=(const CatalogWidget &); /* not allowed */

	//################//
    // SLOTS          //
    //################//
protected:
    void dropEvent(QDropEvent *event);

private:
	CatalogWidget *catalogWidget_;
};


#endif // CATALOGWIDGET_HPP
