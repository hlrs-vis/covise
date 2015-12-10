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
#include "src/data/oscsystem/oscbase.hpp"
#include "src/data/projectdata.hpp"

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

using namespace OpenScenario;

//################//
// CONSTRUCTOR    //
//################//

CatalogTreeWidget::CatalogTreeWidget(MainWindow *mainWindow, OpenScenario::oscObject *object)
	: QTreeWidget()
	, mainWindow_(mainWindow)
	, projectWidget_(NULL)
	, oscEditor_(NULL)
	, currentTool_(ODD::TNO_TOOL)
	, object_(object)
	, oscElement_(NULL)
	, currentSelectedItem_(NULL)
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
	projectWidget_ = mainWindow_->getActiveProject();

	// OpenScenario Element base //
	//
	base_ = projectWidget_->getProjectData()->getOSCBase();
		
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
		currentSelectedItem_ = selectedItems().at(0);
		const QString text = currentSelectedItem_->text(0);

		if (text == "New Element")
		{
			currentTool_ = ODD::TOS_OBJECT;
			OpenScenario::oscObject *newObject;
			/*		OpenScenario::oscObject *newObject = object->addObject("catalog element");
			AddOSCObjectCommand *command = new AddOSCObjectCommand(base_, oscObject->getName(), NULL);

			projectWidget_->getTopviewGraph()->executeCommand(command);
			
			if (command->isValid()) 
			{*/
			oscElement_ = new OSCElement("prototype", newObject);
			if (oscElement_)
			{
				base_->addOSCElement(oscElement_);
		//		oscElement_->setElementSelected(true);

				QTreeWidgetItem *item = new QTreeWidgetItem(); // should be called by observer
				//	item->setText(0, newObject->getName());
				insertTopLevelItem(1, item);
				item->setSelected(true);
			}
		//	}
		}
//		else
		{
			currentTool_ = ODD::TOS_SELECT;
		/*		OpenScenario::oscMember *member = object_->getMember(text);
			oscElement_ = base_->getOSCElement(member);
			if (oscElement_)
			{
				oscElement_->setElementSelected(true);
			}*/
			

		}

		// Set a tool //
		//
	/*	OpenScenarioEditorToolAction *action = new OpenScenarioEditorToolAction(currentTool_, text);
		emit toolAction(action);
		delete action;*/


		QTreeWidget::selectionChanged(selected, deselected);
	}
	else
	{
		clearSelection();
		clearFocus();
	}

}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
CatalogTreeWidget::updateObserver()
{
/*    if (isInGarbage())
    {
        return; // will be deleted anyway
    }*/

    // Object name //
    //
    int changes = oscElement_->getOSCElementChanges();

	if (changes & DataElement::CDE_ChildChange)
    {
/*		if (currentSelectedItem_->text(0) != object_->getName())
		{
			currentSelectedItem_->setText(object_->getName());
		}*/
    }
}
