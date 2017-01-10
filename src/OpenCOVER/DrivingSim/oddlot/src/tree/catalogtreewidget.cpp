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
#include "src/data/changemanager.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"
#include "src/gui/tools/osceditortool.hpp"
#include "src/gui/tools/toolmanager.hpp"
#include "src/gui/oscsettings.hpp"

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
#include "src/data/commands/dataelementcommands.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"

// OpenScenario //
//
#include "schema/oscObject.h"
#include "oscObjectBase.h"
#include "oscMember.h"
#include "oscMemberValue.h"
#include "oscCatalog.h"

#include <QWidget>
#include <QDockWidget>

using namespace OpenScenario;

//################//
// CONSTRUCTOR    //
//################//

CatalogTreeWidget::CatalogTreeWidget(MainWindow *mainWindow, OpenScenario::oscCatalog *catalog)
	: QTreeWidget()
	, mainWindow_(mainWindow)
	, oscEditor_(NULL)
	, currentTool_(ODD::TNO_TOOL)
	, oscElement_(NULL)
	, catalog_(catalog)
	, currentMember_(NULL)
{
    init();
}

CatalogTreeWidget::~CatalogTreeWidget()
{
    if (oscElement_)
	{
		oscElement_->detachObserver(this);
	}
}

//################//
// FUNCTIONS      //
//################//

void
CatalogTreeWidget::init()
{
	// Connect to DockWidget to receive raise signal//
	//

	projectWidget_ = mainWindow_->getActiveProject();
	projectData_ = projectWidget_->getProjectData();

	// OpenScenario Element base //
	//
	base_ = projectData_->getOSCBase();
	openScenarioBase_ = catalog_->getBase();
	directoryPath_ = QString::fromStdString(catalog_->Directory->path.getValue());
		
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

	catalogName_ = catalog_->getCatalogName();
	catalogType_ = "osc" + catalogName_;

	//get all catalog object filenames
	std::vector<bf::path> filenames = catalog_->getXoscFilesFromDirectory(directoryPath_.toStdString());

	//parse all files
	//store object name and filename in map
	catalog_->fastReadCatalogObjects(filenames);

	createTree();
}

void
CatalogTreeWidget::createTree()
{
	clear();

	QList<QTreeWidgetItem *> rootList;

	// emtpy item to create new elements //
	//
	QTreeWidgetItem *item = new QTreeWidgetItem();
	item->setText(0, "New Element");
	rootList.append(item);

	// add all catalog members //
	//
	if (catalog_)
	{

			OpenScenario::oscCatalog::ObjectsMap objects = catalog_->getObjectsMap();
			for(OpenScenario::oscCatalog::ObjectsMap::iterator it = objects.begin();it != objects.end();it++)
			{
				QString elementName = QString::fromStdString(it->first);
				OpenScenario::oscObjectBase *obj = it->second.object;
				if (obj)
				{
			/*		OpenScenario::oscMember *member = obj->getMember("name");
					if (member->exists())
					{
						oscStringValue *sv = dynamic_cast<oscStringValue *>(member->getOrCreateValue());
						QString text = QString::fromStdString(sv->getValue());
						elementName = text + "(" + elementName + ")";
					} */
					elementName = "Loaded(" + elementName + ")";
				}
				else
				{
					elementName = "NotLoaded(" + elementName + ")";
				}

				QTreeWidgetItem *item = new QTreeWidgetItem();
				item->setText(0,elementName);
				item->setFlags(Qt::ItemIsDragEnabled|Qt::ItemIsSelectable|Qt::ItemIsEnabled);

				rootList.append(item);
			}
	}

	insertTopLevelItems(0,rootList);
}

QTreeWidgetItem *CatalogTreeWidget::getItem(const QString &name)
{
	QTreeWidgetItemIterator it(this);
	while (*it) {
		if ((*it)->text(0).contains(name))
            return (*it);
        ++it;
    }

	return NULL;
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

		if (selectedItems().count() > 0)
		{
			const QString text = selectedItems().at(0)->text(0);
			currentTool_ = ODD::TOS_ELEMENT;

			if (text == "New Element")
			{
				// Group undo commands
				//
				projectData_->getUndoStack()->beginMacro(QObject::tr("New Catalog Object"));

				if (oscElement_ && oscElement_->isElementSelected())
				{
					DeselectDataElementCommand *command = new DeselectDataElementCommand(oscElement_, NULL);
					projectWidget_->getTopviewGraph()->executeCommand(command);
				}

				oscElement_ = new OSCElement(text);

				if (oscElement_)
				{
					oscElement_->attachObserver(this);


					// refid vergeben, prüfen ob Datei schon vorhanden, path von vehicleCatalog?, neue Basis für catalog?
					// Element anlegen
					QString filePath;
					std::string refId;
					int i = 1;
					do
					{
						refId = catalog_->generateRefId(i++);
						filePath = directoryPath_ + "/" + QString::fromStdString(catalogName_) + QString::fromStdString(refId) + ".xosc";
					} while (bf::exists(filePath.toStdString())); // test if file exists

					OpenScenario::oscObjectBase *obj = NULL;
					if (OSCSettings::instance()->loadDefaults())
					{
						obj = catalog_->readDefaultXMLObject( filePath.toStdString(), catalogName_, catalogType_);
					}

					if (!obj)
					{
						OpenScenario::oscSourceFile *oscSourceFile = openScenarioBase_->createSource(filePath.toStdString(), catalog_->getType(catalogName_));

						AddOSCObjectCommand *command = new AddOSCObjectCommand(catalog_, base_, catalog_->getCatalogType(), oscElement_, oscSourceFile);
						if (command->isValid())
						{
							projectWidget_->getTopviewGraph()->executeCommand(command);
						}

			/*			AddOSCObjectCommand *command = new AddOSCObjectCommand(catalog_, base_, catalogType_, oscElement_, oscSourceFile);
						if (command->isValid())
						{
							projectWidget_->getTopviewGraph()->executeCommand(command);
						} */

						obj = oscElement_->getObject();
					}

					AddOSCCatalogObjectCommand *addCatalogObjectCommand = new AddOSCCatalogObjectCommand(catalog_, refId, obj, filePath.toStdString(), base_, oscElement_);

					if (addCatalogObjectCommand->isValid())
					{
						projectWidget_->getTopviewGraph()->executeCommand(addCatalogObjectCommand);

						if (obj)
						{
/*						std::string name = "name";
							SetOSCValuePropertiesCommand<std::string> *setPropertyCommand = new SetOSCValuePropertiesCommand<std::string>(oscElement_, obj, name, text.toStdString());
							projectWidget_->getTopviewGraph()->executeCommand(setPropertyCommand); */

							obj->writeToDisk();
						}
					}
				}

				projectData_->getUndoStack()->endMacro();
			}
			else
			{
				// Group undo commands
				//
				projectData_->getUndoStack()->beginMacro(QObject::tr("Load Catalog Object"));

				if (oscElement_ && oscElement_->isElementSelected())
				{
					DeselectDataElementCommand *command = new DeselectDataElementCommand(oscElement_, NULL);
					projectWidget_->getTopviewGraph()->executeCommand(command);
				}

			
				std::string refId = text.split("(")[1].remove(")").toStdString();

				OpenScenario::oscObjectBase *oscObject = catalog_->getCatalogObject(refId);
				if (oscObject)
				{
					oscElement_ = base_->getOrCreateOSCElement(oscObject);
				}
				else
				{
					oscElement_ = new OSCElement(text);

					if (oscElement_)
					{
						oscElement_->attachObserver(this);

						LoadOSCCatalogObjectCommand *command = new LoadOSCCatalogObjectCommand(catalog_, refId, base_, oscElement_);
						projectWidget_->getTopviewGraph()->executeCommand(command);
					}
				}

				if (oscElement_)
				{
					SelectDataElementCommand *command = new SelectDataElementCommand(oscElement_, NULL);
					projectWidget_->getTopviewGraph()->executeCommand(command); 					
				}

				projectData_->getUndoStack()->endMacro();

				// Set a tool //
				//
				OpenScenarioEditorToolAction *action = new OpenScenarioEditorToolAction(currentTool_, text);
				emit toolAction(action);
				delete action;

			}
		}


		QTreeWidget::selectionChanged(selected, deselected);
	}
/*	else
	{
		clearSelection();
		clearFocus();
	}*/

}


//################//
// SLOTS          //
//################//
void
CatalogTreeWidget::onVisibilityChanged(bool visible)
{
	if (visible && oscEditor_)
	{
		oscEditor_->catalogChanged(catalog_);
	}

	clearSelection();

	if (oscElement_ && oscElement_->isElementSelected())
	{
		DeselectDataElementCommand *command = new DeselectDataElementCommand(oscElement_, NULL);
		projectWidget_->getTopviewGraph()->executeCommand(command);
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

	if (changes & OSCElement::COE_ParameterChange)
    {
		OpenScenario::oscObjectBase *obj = oscElement_->getObject();
		
		OpenScenario::oscMember *member = obj->getMember("name");
		if (member->exists())
		{
			oscStringValue *sv = dynamic_cast<oscStringValue *>(member->getOrCreateValue());
			QString text = QString::fromStdString(sv->getValue());

			QTreeWidgetItem *currentEditedItem = selectedItems().at(0);
			if (currentEditedItem && (text != currentEditedItem->text(0)))
			{
				currentEditedItem->setText(0, text);

				// Update Editor //
				//
				OpenScenarioEditorToolAction *action = new OpenScenarioEditorToolAction(currentTool_, text);
				emit toolAction(action);
				delete action;
			}
		}
    }
	else if (changes & OSCBase::COSC_ElementChange)
	{
		createTree();
	}

	changes = oscElement_->getDataElementChanges();
	if ((changes & DataElement::CDE_DataElementAdded) || (changes & DataElement::CDE_DataElementRemoved))
	{
		createTree();
	}
	else if (changes & DataElement::CDE_SelectionChange)
	{
/*		OpenScenario::oscObjectBase *obj = oscElement_->getObject();
		
		OpenScenario::oscMember *member = obj->getMember("name");
		if (member->exists())
		{
			QTreeWidgetItem *currentEditedItem = selectedItems().at(0);

			currentEditedItem->setSelected(oscElement_->isElementSelected());
		} */
	}

}
