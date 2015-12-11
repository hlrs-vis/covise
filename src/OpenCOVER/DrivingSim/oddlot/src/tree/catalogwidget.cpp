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

#include "catalogwidget.hpp"

// Data //
//
#include "src/data/oscsystem/oscelement.hpp"
#include "src/data/oscsystem/oscbase.hpp"
#include "src/data/projectdata.hpp"

// Trees //
//
#include "catalogtreewidget.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"
#include "src/gui/tools/osceditortool.hpp"
#include "src/gui/tools/toolmanager.hpp"

// MainWindow//
//
#include "src/mainwindow.hpp"

//Settings//
//
#include "src/settings/projectsettings.hpp"

// Editor //
//
#include "src/graph/editors/osceditor.hpp"

// Commands //
//
#include "src/data/commands/osccommands.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"

// OpenScenario //
//
#include "oscObject.h"
#include "oscMember.h"

#include <QWidget>
#include <QDragEnterEvent>
#include <QDragLeaveEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QMimeData>

using namespace OpenScenario;

//################//
// CONSTRUCTOR    //
//################//

CatalogWidget::CatalogWidget(MainWindow *mainWindow, OpenScenario::oscObject *object)
	: QWidget()
	, mainWindow_(mainWindow)
	, projectWidget_(NULL)
	, object_(object)
	, oscElement_(NULL)
{
    init();
}

CatalogWidget::~CatalogWidget()
{
    
}

//################//
// FUNCTIONS      //
//################//

void
CatalogWidget::init()
{
	projectData_ = mainWindow_->getActiveProject()->getProjectData();

	const QPixmap recycleIcon(":/icons/recycle.png");
	DropArea *recycleArea = new DropArea(recycleIcon, this);

	catalogTreeWidget_ = new CatalogTreeWidget(mainWindow_, object_);
}

void 
CatalogWidget::onDeleteCatalogItem()
{
	OSCBase *base = projectData_->getOSCBase();
	OpenScenario::oscObjectBase *oscBase = base->getOSCObjectBase();

	bool deletedSomething = false;
	do
	{
		deletedSomething = false;
		QList<DataElement *> selectedElements = projectData_->getSelectedElements();
		foreach (DataElement * element, selectedElements)
		{
			OSCElement *oscElement = dynamic_cast<OSCElement *>(element);
			if (oscElement)
			{
	/*			RemoveOSCObjectCommand *command = new RemoveOSCObjectCommand(oscBase, oscElement->getObject());
				projectData_->getProjectWidget()->getTopviewGraph()->executeCommand(command);

				if (command->isValid())*/
				{
					base->delOSCElement(oscElement);
					deletedSomething = true;
					break;
				}
			}
		}
	}while(deletedSomething);
}
 

//#######################################################//
// DropArea for the recycle bin of the catalog widget //
//
//#######################################################//
DropArea::DropArea(const QPixmap &pixmap, CatalogWidget *parent)
    : QLabel(parent)
	, parent_(parent)
{
    setMinimumSize(200, 200);
    setFrameStyle(QFrame::Sunken | QFrame::StyledPanel);
    setAlignment(Qt::AlignCenter);
    setAcceptDrops(true);
    setAutoFillBackground(true);
	recycleLabel_->setPixmap(pixmap);
}

//################//
// EVENTS         //
//################//

void 
DropArea::dragEnterEvent(QDragEnterEvent *event)
{
    setBackgroundRole(QPalette::Highlight);

    event->acceptProposedAction();
}

void 
DropArea::dragMoveEvent(QDragMoveEvent *event)
{
    event->acceptProposedAction();
}

void 
DropArea::dropEvent(QDropEvent *event)
{
	parent_->onDeleteCatalogItem();

	setBackgroundRole(QPalette::Dark);
    event->acceptProposedAction();
}

void 
DropArea::dragLeaveEvent(QDragLeaveEvent *event)
{
    clear();
    event->accept();
}