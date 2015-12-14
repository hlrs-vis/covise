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
#include "oscObjectBase.h"
#include "oscMember.h"

#include <QWidget>

using namespace OpenScenario;

//################//
// CONSTRUCTOR    //
//################//

CatalogTreeWidget::CatalogTreeWidget(MainWindow *mainWindow, const OpenScenario::oscObjectBase *object, const QString &type)
	: QTreeWidget()
	, mainWindow_(mainWindow)
	, projectWidget_(NULL)
	, oscEditor_(NULL)
	, currentTool_(ODD::TNO_TOOL)
	, objectBase_(object)
	, oscElement_(NULL)
	, currentSelectedItem_(NULL)
	, type_(type)
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
	openScenarioBase_ = objectBase_->getBase();
		
	// Connect with the ToolManager to send the selected signal or object //
    //
	ToolManager *toolManager = mainWindow_->getToolManager();
	if (toolManager)
	{
		connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));
	}

    setSelectionMode(QAbstractItemView::ExtendedSelection);
	setDragEnabled(true);
    setUniformRowHeights(true);
	setIndentation(6);

	// Signals Widget //
    //
	setColumnCount(3);
	setColumnWidth(0, 180);
	setColumnWidth(1, 30);

	setHeaderHidden(true);
	QList<QTreeWidgetItem *> rootList;

	// temporary: make test base
	if (!openScenarioBase_->test.exists())
	{
		testBase_ = new OSCElement("test");

		AddOSCObjectCommand *command = new AddOSCObjectCommand(openScenarioBase_, base_, "test", testBase_, NULL);
		projectWidget_->getTopviewGraph()->executeCommand(command);
	}

	type_ = type_.remove("Catalog");

	createTree();
}

void
CatalogTreeWidget::createTree()
{
	QList<QTreeWidgetItem *> rootList;

	// emtpy item to create new elements //
	//
	QTreeWidgetItem *item = new QTreeWidgetItem();
	item->setText(0, "New Element");
	rootList.append(item);

	// add all catalog members //
	//
	if (objectBase_)
	{
		OpenScenario::oscObjectBase::MemberMap members = objectBase_->getMembers();
		for(OpenScenario::oscObjectBase::MemberMap::iterator it = members.begin();it != members.end();it++)
		{
			std::string elementName = it->first;

			if ((elementName != "directory") && (elementName != "userData"))
			{
				QTreeWidgetItem *item = new QTreeWidgetItem();
				item->setText(0,QString(elementName.c_str()));
				item->setFlags(Qt::ItemIsDragEnabled);

				rootList.append(item);
			}
		}
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
//	if (oscEditor_)
	{
		currentSelectedItem_ = selectedItems().at(0);
		const QString text = currentSelectedItem_->text(0);

		if (text == "New Element")
		{
			currentTool_ = ODD::TOS_OBJECT;
			oscElement_ = new OSCElement(type_);
	
//			testBase_ only temporary
			AddOSCObjectCommand *command = new AddOSCObjectCommand(testBase_->getObject(), base_,type_.toStdString(), oscElement_, NULL);
			projectWidget_->getTopviewGraph()->executeCommand(command);
			
			if (command->isValid()) 
			{
				currentMember_ = testBase_->getObject()->getMembers().at(type_.toStdString());
				oscElement_->setElementSelected(true);

				QTreeWidgetItem *item = new QTreeWidgetItem(); // should be called by observer
				item->setText(0, type_);
				insertTopLevelItem(1, item);
//				item->setSelected(true);
			}
		}
		else
		{
			currentTool_ = ODD::TOS_SELECT;
			currentMember_ = testBase_->getObject()->getMembers().at(text.toStdString());
			oscElement_ = base_->getOSCElement(currentMember_->getObject());
			if (oscElement_)
			{
				oscElement_->setElementSelected(true);
			}
			

		}

		// Set a tool //
		//
	/*	OpenScenarioEditorToolAction *action = new OpenScenarioEditorToolAction(currentTool_, text);
		emit toolAction(action);
		delete action;*/


		QTreeWidget::selectionChanged(selected, deselected);
	}
/*	else
	{
		clearSelection();
		clearFocus();
	}*/

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
		if (currentSelectedItem_->text(0).toStdString() != currentMember_->getName())
		{
			currentSelectedItem_->setText(1, QString::fromStdString(currentMember_->getName()));
		}
    }
	else if ((changes & DataElement::CDE_DataElementAdded) || (changes & DataElement::CDE_DataElementDeleted))
	{
		createTree();
	}
}
