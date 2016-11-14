/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   15.03.2010
**
**************************************************************************/

#include "osceditor.hpp"

// Project //
//
#include "src/gui/projectwidget.hpp"

//MainWindow //
//
#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/signalmanager.hpp"
#include "src/data/oscsystem/oscelement.hpp"
#include "src/data/oscsystem/oscbase.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/changemanager.hpp"

// Commands //
//
#include "src/data/commands/osccommands.hpp"
#include "src/data/commands/signalcommands.hpp"
#include "src/data/commands/roadsectioncommands.hpp"
#include "src/data/commands/dataelementcommands.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/graphview.hpp"

#include "src/graph/items/roadsystem/scenario/oscroadsystemitem.hpp"
#include "src/graph/items/oscsystem/oscitem.hpp"
#include "src/graph/items/oscsystem/oscbaseitem.hpp"

// Tools //
//
#include "src/gui/tools/osceditortool.hpp"
#include "src/gui/mouseaction.hpp"

// Tree //
//
#include "src/tree/catalogtreewidget.hpp"

// Visitor //
//
//#include "src/graph/visitors/roadmarkvisitor.hpp"

// OpenScenario //
//
#include "oscObjectBase.h"
#include "oscObject.h"
#include "oscCatalogs.h"
#include "oscCatalog.h"
#include "oscPosition.h"
#include "oscArrayMember.h"
#include "oscCatalogReferenceTypeA.h"

// Boost //
//
#include <boost/filesystem.hpp>


// Qt //
//
#include <QGraphicsItem>
#include <QUndoStack>
#include <QStatusBar>

using namespace OpenScenario;

//################//
// CONSTRUCTORS   //
//################//

OpenScenarioEditor::OpenScenarioEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph)
    : ProjectEditor(projectWidget, projectData, topviewGraph)
	, oscBaseItem_(NULL)
	, oscBase_(NULL)
	, oscCatalog_(NULL)
{
	mainWindow_ = projectWidget->getMainWindow();
	oscBase_ = projectData->getOSCBase();
	openScenarioBase_ = oscBase_->getOpenScenarioBase();
    oscRoadSystemItem_ = NULL;
}

OpenScenarioEditor::~OpenScenarioEditor()
{
    kill();
}

//################//
// FUNCTIONS      //
//################//

/**
*
*/
void
OpenScenarioEditor::init()
{
	OpenScenario::oscEntities *entitiesObject = openScenarioBase_->entities.getOrCreateObject();
	OpenScenario::oscArrayMember *oscObjectArray = dynamic_cast<OpenScenario::oscArrayMember *>(entitiesObject->getMember("objects"));
	
	for (int i = 0; i < oscObjectArray->size(); i++)
	{
		OpenScenario::oscObject *object = dynamic_cast<OpenScenario::oscObject *>(oscObjectArray->at(i));
		OpenScenario::oscCatalogReferenceTypeA *catalogReference = object->catalogReference.getOrCreateObject();

		if (!oscBase_->getOSCElement(object))
		{
			OSCElement *element = new OSCElement(QString::fromStdString(catalogReference->catalogId.getValue()), object);
			oscBase_->addOSCElement(element);
		}
	}

	if (!oscRoadSystemItem_)
	{
		// Root item //
        //
        oscRoadSystemItem_ = new OSCRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
        getTopviewGraph()->getScene()->addItem(oscRoadSystemItem_);
	}

	// Reset change //
    //

    getProjectData()->getChangeManager()->notifyObservers();


	if (!oscBaseItem_ && openScenarioBase_->getSource())   // OpenScenario Editor graphischee Elemente
    {
		// Root OSC item //
		//
		oscBaseItem_ = new OSCBaseItem(getTopviewGraph(), oscBase_);
		getTopviewGraph()->getScene()->addItem(oscBaseItem_);
	}


        // Signal Handle //
        //
 /*       insertOSCHandle_ = new OSCHandle(signalRoadSystemItem_);
        insertOSCHandle_->hide();*/


    lastTool_ = getCurrentTool();

}

/*!
*/
void
OpenScenarioEditor::kill()
{
	if (oscRoadSystemItem_)
	{
		delete oscRoadSystemItem_;
		oscRoadSystemItem_ = NULL;
	}

	if (oscBaseItem_)
	{
		delete oscBaseItem_;
		oscBaseItem_ = NULL;
	}

}

/*OSCHandle *
OpenScenarioEditor::getOSCHandle() const
{
    if (!insertOSCHandle_)
    {
        qDebug("ERROR 1003281422! OpenScenarioEditor not yet initialized.");
    }
    return insertOSCHandle_;
}*/

// Move Object //
//


bool 
OpenScenarioEditor::translateObject(OpenScenario::oscObject * object, const QString &newRoadId, double s, double t)
{
	OSCElement *oscElement = oscBase_->getOrCreateOSCElement(object);
	OpenScenario::oscPosition *oscPosition = object->initPosition.getOrCreateObject();
	OpenScenario::oscPositionRoad *oscPosRoad = oscPosition->positionRoad.getOrCreateObject();
	oscStringValue *roadId = dynamic_cast<oscStringValue *>(oscPosRoad->roadId.getOrCreateValue());

	getProjectData()->getUndoStack()->beginMacro(QObject::tr("Move Object"));

	bool parentChanged = false;
	if (QString::fromStdString(roadId->getValue()) != newRoadId)
	{
		SetOSCValuePropertiesCommand<std::string> *command = new SetOSCValuePropertiesCommand<std::string>(oscElement, oscPosRoad, "roadId", newRoadId.toStdString());
		getProjectGraph()->executeCommand(command);

		parentChanged = true;
		//		object->setElementSelected(false);
	}


	SetOSCValuePropertiesCommand<double> *command = new SetOSCValuePropertiesCommand<double>(oscElement, oscPosRoad, "s", s);
	getProjectGraph()->executeCommand(command);
	command = new SetOSCValuePropertiesCommand<double>(oscElement, oscPosRoad, "t", t);
	getProjectGraph()->executeCommand(command);

    getProjectData()->getUndoStack()->endMacro();

    return parentChanged;
}

/*Object *
OpenScenarioEditor::addObjectToRoad(RSystemElementRoad *road, double s, double t)
{
	ObjectContainer *lastObject = signalManager_->getSelectedObjectContainer();
	Object *newObject = NULL;

	if (lastObject)
	{
		if (t < 0)
		{
			t -= lastObject->getObjectDistance();
		}
		else
		{
			t += lastObject->getObjectDistance();
		}
		newObject = new Object("object", "", lastObject->getObjectType(), s, t, 0.0, 0.0, Object::NEGATIVE_TRACK_DIRECTION, lastObject->getObjectLength(), 
			lastObject->getObjectWidth(), lastObject->getObjectRadius(), lastObject->getObjectHeight(), lastObject->getObjectHeading(),
					0.0, 0.0, false, 0.0, 0.0, lastObject->getObjectRepeatDistance(), lastObject->getObjectFile());
		AddObjectCommand *command = new AddObjectCommand(newObject, road, NULL);
		getProjectGraph()->executeCommand(command);
	}
	else
	{
		Object *newObject = new Object("object", "", "", s, t, 0.0, 0.0, Object::NEGATIVE_TRACK_DIRECTION, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, false, s, 0.0, 0.0, "");
		AddObjectCommand *command = new AddObjectCommand(newObject, road, NULL);
		getProjectGraph()->executeCommand(command);
	}

	return newObject;
}
*/

void 
OpenScenarioEditor::catalogChanged(OpenScenario::oscCatalog * member)
{
	oscCatalog_ = member;
}

OpenScenario::oscCatalog *
OpenScenarioEditor::getCatalog(std::string name)
{

	OpenScenario::oscCatalogs *catalogs = openScenarioBase_->catalogs.getOrCreateObject();
	OpenScenario::oscCatalog *catalog = catalogs->getCatalog(name);
	QString catalogDir = OSCSettings::instance()->getCatalogDir();
	std::string catalogsDir = catalogDir.toStdString(); 
	if (!bf::exists(bf::path(catalogsDir)))
	{
		bf::create_directory(catalogsDir);
	}

	OpenScenario::oscDirectory *dir = dynamic_cast<oscDirectory *>(catalog->getObjectByName("directory"));
	if (dir)
	{
		std::string dirName = catalog->directory->path.getValue();
		if (dirName == "")
		{
			dirName = catalog->directory->path = catalogsDir + name;
			if (!bf::exists(bf::path(dirName)))
			{
				bf::create_directory(dirName);
			}
		}
	}


	name = name.erase(name.find("Catalog"));
	catalog->setCatalogName(name);

	OSCElement *oscElement = oscBase_->getOrCreateOSCElement(catalog);

	return catalog;
}


//################//
// MOUSE & KEY    //
//################//

/*! \brief .
*
*/
void
OpenScenarioEditor::mouseAction(MouseAction *mouseAction)
{

    QGraphicsSceneMouseEvent *mouseEvent = mouseAction->getEvent();
    ProjectEditor::mouseAction(mouseAction);

    // SELECT //
    //
	ODD::ToolId currentTool = getCurrentTool();
    if (currentTool == ODD::TSG_SELECT)
    {
        QPointF mousePoint = mouseAction->getEvent()->scenePos();

        if (mouseAction->getMouseActionType() == MouseAction::ATM_PRESS)
        {
            if (mouseEvent->button() == Qt::LeftButton)
            {
  
            }
        }

    }
	else if ((currentTool == ODD::TOS_ELEMENT) && oscCatalog_ && (oscCatalog_->getCatalogName() == "entity"))
	{
		QPointF mousePoint = mouseAction->getEvent()->scenePos();

        if (mouseAction->getMouseActionType() == MouseAction::ATM_PRESS)
		{
			if (mouseEvent->button() == Qt::LeftButton)
			{
				QList<QGraphicsItem *> underMouseItems = getTopviewGraph()->getScene()->items(mousePoint);

				if (underMouseItems.count() == 0)		// find the closest road //
				{
					double s;
					double t;
					QVector2D vec;

					RSystemElementRoad * road = getProjectData()->getRoadSystem()->findClosestRoad(mousePoint, s, t, vec);
					if (road)
					{
						// Create new object //
						OpenScenario::oscCatalog *entityCatalog = getCatalog("entityCatalog");
						OpenScenario::oscEntity *oscEntity;

						QList<DataElement *> elements = getProjectData()->getSelectedElements();
						for (int i = 0; i < elements.size(); i++)
						{
							OSCElement *element = dynamic_cast<OSCElement *>(elements.at(i));
							if (element)
							{
								OpenScenario::oscObjectBase * objectBase = element->getObject();
								if (objectBase)
								{
									oscEntity = dynamic_cast<OpenScenario::oscEntity *>(objectBase);
								}
							}
						}

						OpenScenario::oscEntities *entitiesObject = openScenarioBase_->entities.getOrCreateObject();
						OpenScenario::oscMember *objectsMember = entitiesObject->getMember("objects");
						OpenScenario::oscObjectBase *objects = objectsMember->getOrCreateObject();

						OpenScenario::oscArrayMember *oscObjectArray = dynamic_cast<OpenScenario::oscArrayMember *>(objectsMember);

						OSCElement *oscElement = new OSCElement("object");
						if (oscElement)
						{
							AddOSCArrayMemberCommand *command = new AddOSCArrayMemberCommand(oscObjectArray, objects, NULL, "object", oscBase_, oscElement);
							getProjectGraph()->executeCommand(command);

							OpenScenario::oscObject *oscObject = static_cast<OpenScenario::oscObject *>(oscElement->getObject());
							OpenScenario::oscCatalogReferenceTypeA *catalogReference = oscObject->catalogReference.getOrCreateObject();
							catalogReference->catalogId.setValue(std::to_string(oscEntity->refId.getValue()));

							translateObject(oscObject, road->getID(), s, t);

							OSCItem *oscItem = new OSCItem(oscElement, oscBaseItem_, oscObject, entityCatalog, mousePoint, road->getID());  
						}
					}
				}
			}
		}
	}

    //	ProjectEditor::mouseAction(mouseAction);
}

//################//
// TOOL           //
//################//

/*! \brief .
*
*/
void
OpenScenarioEditor::toolAction(ToolAction *toolAction)
{
    // Parent //
    //
    ProjectEditor::toolAction(toolAction);

    ODD::ToolId currentTool = getCurrentTool();

	if (currentTool == ODD::TOS_CREATE_CATALOG)
	{
		OpenScenarioEditorToolAction *action = dynamic_cast<OpenScenarioEditorToolAction *>(toolAction);
		if (action)
		{
			// Create new catalog //
			QString objectName = action->getText();
			
			if (objectName != "")
			{
				OpenScenario::oscCatalog *oscCatalog = getCatalog(objectName.toStdString());

				catalogTree_ = getProjectWidget()->addCatalogTree(objectName, oscCatalog);   
				catalogTree_->setOpenScenarioEditor(this);  

			}
		}
	}
	else if (currentTool == ODD::TOS_ELEMENT)
	{
		OpenScenarioEditorToolAction *action = dynamic_cast<OpenScenarioEditorToolAction *>(toolAction);
		if (action && ((currentTool != lastTool_) || (action->getText() != catalogElement_)))
		{
			// Create new object //
			catalogElement_ = action->getText();
		}
	}
	else if (currentTool == ODD::TOS_BASE)
	{
		OpenScenarioEditorToolAction *action = dynamic_cast<OpenScenarioEditorToolAction *>(toolAction);

		if (action)
		{
			QString oscObjectName = action->getText();
	//		if ((currentTool != lastTool_) || (oscObjectName != lastOSCObjectName_))
			{
				OpenScenario::oscMember *member = openScenarioBase_->getMember(oscObjectName.toStdString());
				if (member)
				{
					OpenScenario::oscObjectBase *object = member->getOrCreateObject();

					OSCElement *memberElement = oscBase_->getOrCreateOSCElement(object);
					if (memberElement)
					{
						// Group undo commands
						//
						getProjectData()->getUndoStack()->beginMacro(QObject::tr("Base Element selected"));

						foreach (DataElement *element, getProjectData()->getSelectedElements())
						{
							DeselectDataElementCommand *command = new DeselectDataElementCommand(element, NULL);
							getProjectGraph()->executeCommand(command); 

						}

						SelectDataElementCommand *command = new SelectDataElementCommand(memberElement, NULL);
						getProjectGraph()->executeCommand(command);		

						getProjectData()->getUndoStack()->endMacro();
					}


					lastOSCObjectName_ = oscObjectName;
				}
			}
		}
	}
	else
	{		
		if (currentTool == ODD::TOS_SAVE_CATALOG)
		{
			// Save catalog //
			//			if (catalog_ && mainWindow_->getActiveProject()->saveCatalogAs())
			/*if (mainWindow_->getActiveProject()->saveAs())
			{
			printStatusBarMsg(tr("File has been saved."), 2000);
			} */


			OpenScenarioEditorToolAction *action = dynamic_cast<OpenScenarioEditorToolAction *>(toolAction);
			if (action)
			{
				// Save catalog //
				oscCatalog_->writeCatalogToDOM();
				oscCatalog_->writeCatalogToDisk();
				oscCatalog_->clearDOM();


/*				QString type = action->getText();

				if (type != "")
				{
					oscCatalog *catalogMember = dynamic_cast<oscCatalog *>(oscCatalog_->getOwnMember());
					if (catalogMember->getName().compare(type.toStdString()) != 0)
					{
						OSCElement *oscElement = getCatalog(type.toStdString());
						oscCatalog_ = oscElement->getObject();
					}
					OpenScenario::OpenScenarioBase *openScenarioBase = openScenarioBase_();
	OpenScenario::oscObjectBase *objectCatalog = openScenarioBase->getMember("catalogs")->getObject();
	objectCatalog = objectCatalog->getMember("objectCatalog")->getObject(); */
	
/*	dynamic_cast<OpenScenario::oscCatalog*>(objectCatalog->getMember(type.toStdString()))->writeCatalogToDOM();
				} */
			}
		}



		/*        if (currentTool == ODD::TSG_SELECT)
		{
		if ((lastTool_ = ODD::TSG_ADD_CONTROL_ENTRY) || (lastTool_ = ODD::TSG_REMOVE_CONTROL_ENTRY))
		{
		foreach (QGraphicsItem *item, getTopviewGraph()->getScene()->items())
		{
		SignalItem * signalItem = dynamic_cast<SignalItem *>(item);
		if (signalItem)
		{
		signalItem->setFlag(QGraphicsItem::ItemIsSelectable, true);
		}
		}
		}
		}
		else if ((currentTool == ODD::TSG_SIGNAL) || (currentTool == ODD::TSG_OBJECT) 
		|| (currentTool == ODD::TSG_BRIDGE) || (currentTool = ODD::TSG_TUNNEL))
		{
		foreach (QGraphicsItem *item, getTopviewGraph()->getScene()->selectedItems())
		{
		item->setSelected(false);
		}
		// does nothing //
		// Problem: The ToolAction is resent, after a warning message has been clicked away. (Due to resend on getting the
		}
		else if (currentTool == ODD::TSG_CONTROLLER)
		{
		QList<ControlEntry *>controlEntryList;
		RSystemElementController *newController = new RSystemElementController("unnamed", "", 0,"", 0.0, controlEntryList);
		AddControllerCommand *command = new AddControllerCommand(newController, getProjectData()->getRoadSystem(), NULL);

		getProjectGraph()->executeCommand(command);
		}
		else if ((currentTool == ODD::TSG_ADD_CONTROL_ENTRY) || (currentTool == ODD::TSG_REMOVE_CONTROL_ENTRY))
		{

		foreach (QGraphicsItem *item, getTopviewGraph()->getScene()->items())
		{
		SignalItem * signalItem = dynamic_cast<SignalItem *>(item);
		if (signalItem)
		{
		signalItem->setFlag(QGraphicsItem::ItemIsSelectable, false);
		}
		}
		}
		else if (currentTool == ODD::TSG_DEL)
		{
		QList<DataElement *>selectedElements = getProjectData()->getSelectedElements();
		QList<RSystemElementController *>selectedControllers;
		foreach (DataElement *element, selectedElements)
		{
		RSystemElementController * controller = dynamic_cast<RSystemElementController *>(element);
		if (controller)
		{
		selectedControllers.append(controller);
		}
		}
		if (selectedControllers.size() > 0)
		{
		// Macro Command //
		//
		int numberOfSelectedControllers = selectedControllers.size();
		if(numberOfSelectedControllers > 1)
		{
		getProjectData()->getUndoStack()->beginMacro(QObject::tr("Set Road Type"));
		}
		for (int i = 0; i < selectedControllers.size(); i++)
		{
		RemoveControllerCommand * delControllerCommand = new RemoveControllerCommand(selectedControllers.at(i), selectedControllers.at(i)->getRoadSystem());
		getProjectGraph()->executeCommand(delControllerCommand);
		}

		// Macro Command //
		//
		if (numberOfSelectedControllers > 1)
		{
		getProjectData()->getUndoStack()->endMacro();
		}
		}*/
	}

	lastTool_ = currentTool;
}
