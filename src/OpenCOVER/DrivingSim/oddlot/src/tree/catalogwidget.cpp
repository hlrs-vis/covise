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
#include "src/util/droparea.hpp"

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
#include <OpenScenario/oscObjectBase.h>
#include <OpenScenario/oscMember.h>

#include <QWidget>
#include <QDropEvent>
#include <QGridLayout>
#include <QMenu>


using namespace OpenScenario;

//################//
// CONSTRUCTOR    //
//################//

CatalogWidget::CatalogWidget(MainWindow *mainWindow, OpenScenario::oscCatalog *catalog, const QString &name)
	: QWidget()
	, mainWindow_(mainWindow)
	, catalog_(catalog)
	, name_(name)
	, catalogTreeWidget_(NULL)
{
    init();
}

CatalogWidget::~CatalogWidget()
{
	delete catalogTreeWidget_;
}

//################//
// FUNCTIONS      //
//################//

void
CatalogWidget::init()
{
	projectData_ = mainWindow_->getActiveProject()->getProjectData();
	base_ = projectData_->getOSCBase();

	// Widget/Layout //
    //
	QGridLayout *toolLayout = new QGridLayout;
	QPixmap recycleIcon(":/icons/recycle.png");

	CatalogDropArea *recycleArea = new CatalogDropArea(this, &recycleIcon);
	toolLayout->addWidget(recycleArea, 0, 2);

	catalogTreeWidget_ = new CatalogTreeWidget(mainWindow_, catalog_);
	toolLayout->addWidget(catalogTreeWidget_, 0, 0); 

    int row = -1; // button row

    // Link Roads by Handles//
    //
    QPushButton * toolButton = new QPushButton(tr("Save"));
    toolButton->setCheckable(false);
    toolLayout->addWidget(toolButton, 1, 0);
	connect(toolButton, SIGNAL(clicked()), this, SLOT(handleToolClick()));

	this->setLayout(toolLayout);

	// Connect with the ToolManager to send the selected signal or object //
    //
	ToolManager *toolManager = mainWindow_->getToolManager();
	if (toolManager)
	{
		connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));
	} 
}

void 
CatalogWidget::onDeleteCatalogItem()
{
	bool deletedSomething = false;
	do
	{
		deletedSomething = false;

		QList<QTreeWidgetItem *> selectedItems = catalogTreeWidget_->selectedItems();
		for ( int i = 0; i < selectedItems.size(); i++)
		{
			QString text = selectedItems.at(i)->text(0);
			std::string refId = text.split("(")[1].remove(")").toStdString();
			OSCElement *element = base_->getOSCElement(catalog_->getCatalogObject(refId));

			RemoveOSCCatalogObjectCommand *command = new RemoveOSCCatalogObjectCommand(catalog_, refId, element);

			if (command->isValid())
			{
				if (!element)
				{
					projectData_->getProjectWidget()->getTopviewGraph()->executeCommand(command);
					catalogTreeWidget_->createTree();
				}
				else
				{
					projectData_->getProjectWidget()->getTopviewGraph()->executeCommand(command);
				}
				deletedSomething = true;
				break;
			}
		}
	}while(deletedSomething);
}

//################//
// SLOTS          //
//################//

/*! \brief Gets called when a tool has been selected.
*/
void
	CatalogWidget::handleToolClick()
{

    // Set a tool //
    //
    OpenScenarioEditorToolAction *action = new OpenScenarioEditorToolAction(ODD::TOS_SAVE_CATALOG, name_);
    emit toolAction(action);
    delete action;
}

//###############################//
// DropArea for the recycle bin //
//
//#############################//
CatalogDropArea::CatalogDropArea(CatalogWidget *catalogWidget, QPixmap *pixmap)
    : DropArea(pixmap)
	, catalogWidget_(catalogWidget)
{
}

//################//
// EVENTS         //
//################//

void 
CatalogDropArea::dropEvent(QDropEvent *event)
{
	catalogWidget_->onDeleteCatalogItem();

	DropArea::dropEvent(event);
}


 
