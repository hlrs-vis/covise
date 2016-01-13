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
#include "oscObjectBase.h"
#include "oscMember.h"

#include <QWidget>
#include <QDragEnterEvent>
#include <QDragLeaveEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QGridLayout>
#include <QMenu>


using namespace OpenScenario;

//################//
// CONSTRUCTOR    //
//################//

CatalogWidget::CatalogWidget(MainWindow *mainWindow, OSCElement *element, const QString &type)
	: QWidget()
	, mainWindow_(mainWindow)
	, projectWidget_(NULL)
	, oscElement_(element)
	, type_(type)
{
	object_ = oscElement_->getObject();
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

	// Widget/Layout //
    //
	QGridLayout *toolLayout = new QGridLayout;
	QPixmap recycleIcon(":/icons/recycle.png");
	DropArea *recycleArea = new DropArea(recycleIcon, this);
	recycleArea->setPixmap(recycleIcon);
	toolLayout->addWidget(recycleArea, 0, 2);


	catalogTreeWidget_ = new CatalogTreeWidget(mainWindow_, object_, type_);
	toolLayout->addWidget(catalogTreeWidget_, 0, 0);

	this->setLayout(toolLayout);
}

void 
CatalogWidget::onDeleteCatalogItem()
{
	OSCBase *base = oscElement_->getOSCBase();

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
				RemoveOSCObjectCommand *command = new RemoveOSCObjectCommand(oscElement);
				projectData_->getProjectWidget()->getTopviewGraph()->executeCommand(command);

				if (command->isValid())
				{
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
    setMaximumSize(20, 20);
    setFrameStyle(QFrame::Sunken | QFrame::StyledPanel);
    setAlignment(Qt::AlignCenter);
    setAcceptDrops(true);
    setAutoFillBackground(true);
	
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

