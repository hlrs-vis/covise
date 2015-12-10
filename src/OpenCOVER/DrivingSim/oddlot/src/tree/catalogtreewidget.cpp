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

#include "catalogtreewidget.hpp"

// Data //
//
#include "src/data/oscsystem/oscelement.hpp"

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

// OpenScenario //
//
#include "oscObjectBase.h"
#include "oscMember.h"

#include <QWidget>

using namespace OpenScenario;

//################//
// CONSTRUCTOR    //
//################//

CatalogTreeWidget::CatalogTreeWidget(MainWindow *mainWindow, OpenScenario::oscObjectBase *object)
	: QTreeWidget()
	, mainWindow_(mainWindow)
	, projectWidget_(NULL)
	, oscEditor_(NULL)
	, currentTool_(ODD::TNO_TOOL)
	, object_(object)
{
    init();
}

CatalogTreeWidget::~CatalogTreeWidget()
{
    
}

//################//
// FUNCTIONS      //
//################//

void
CatalogTreeWidget::init()
{
	// Connect with the ToolManager to send the selected signal or object //
    //
	ToolManager *toolManager = mainWindow_->getToolManager();
	if (toolManager)
	{
		connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));
	}

    setSelectionMode(QAbstractItemView::ExtendedSelection);
    setUniformRowHeights(true);
	setIndentation(6);

	// Signals Widget //
    //
	setColumnCount(3);
	setColumnWidth(0, 180);
	setColumnWidth(1, 30);

	setHeaderHidden(true);
	QList<QTreeWidgetItem *> rootList;

	// emtpy item to create new elements //
	//
	QTreeWidgetItem *item = new QTreeWidgetItem();
	item->setText(0, "New Element");
	rootList.append(item);

	// add all catalog members //
	//
	if (object_)
	{
//		OpenScenario::oscObjectBase::MemberMap *members = object_->getMembers();
/*		for(MemberMap::iterator it = members.begin();it != members.end();it++)
		{
			std::string elementName = it->first();

			QTreeWidgetItem *item = new QTreeWidgetItem();
			item->setText(0,QString(elementName.c_str()));

			rootList.append(item);
		}*/
	}

	insertTopLevelItems(0,rootList);
}


void 
CatalogTreeWidget::setOpenScenarioEditor(OpenScenarioEditor *oscEditor)
{
	oscEditor_ = oscEditor;

	if (!oscEditor)
	{
		clearSelection();
	}
}

  
//################//
// EVENTS         //
//################//
void
CatalogTreeWidget::selectionChanged(const QItemSelection &selected, const QItemSelection &deselected)
{
	if (oscEditor_)
	{
		QTreeWidgetItem *item = selectedItems().at(0);
		const QString text = item->text(0);

		if (text == "New Element")
		{
			currentTool_ = ODD::TOS_OBJECT;
			OpenScenario::oscObject *newObject;
			/*		OpenScenario::oscObject *newObject = object->addObject("catalog element");
			AddOSCObjectCommand *command = new AddOSCObjectCommand(base_, &objectName, NULL);

			getProjectGraph()->executeCommand(command);
			*/	
			OSCElement *oscElement = new OSCElement(newObject);
			oscElement->setElementSelected(true);

			QTreeWidgetItem *item = new QTreeWidgetItem();
		//	item->setText(0, newObject->getName());
			insertTopLevelItem(1, item);
		}
		else
		{
			currentTool_ = ODD::TOS_SELECT;
		//		OpenScenario::oscMember *member = object_->getMember(text);
		}



		// Set a tool //
		//
		OpenScenarioEditorToolAction *action = new OpenScenarioEditorToolAction(currentTool_, text);
		emit toolAction(action);
		delete action;


		QTreeWidget::selectionChanged(selected, deselected);
	}
	else
	{
		clearSelection();
		clearFocus();
	}

}

