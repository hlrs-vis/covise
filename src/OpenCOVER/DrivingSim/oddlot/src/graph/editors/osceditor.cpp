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

// Commands //
//
#include "src/data/commands/osccommands.hpp"
#include "src/data/commands/signalcommands.hpp"
#include "src/data/commands/roadsectioncommands.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/graphview.hpp"

#include "src/graph/items/roadsystem/scenario/oscroadsystemitem.hpp"
#include "src/graph/items/oscsystem/oscitem.hpp"

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
	, oscRoadSystemItem_(NULL)
	, oscBase_(NULL)
	, oscCatalog_(NULL)
{
	mainWindow_ = projectWidget->getMainWindow();
	oscBase_ = projectData->getOSCBase();
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
    if (!oscRoadSystemItem_)   // Signaleditor graphischee Elemente
    {
        // Root item //
        //
        oscRoadSystemItem_ = new OSCRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
        getTopviewGraph()->getScene()->addItem(oscRoadSystemItem_);

        // Signal Handle //
        //
 /*       insertOSCHandle_ = new OSCHandle(signalRoadSystemItem_);
        insertOSCHandle_->hide();*/
    }

    lastTool_ = getCurrentTool();

}

/*!
*/
void
OpenScenarioEditor::kill()
{
    delete oscRoadSystemItem_;
    oscRoadSystemItem_ = NULL;

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
RSystemElementRoad *
OpenScenarioEditor::findClosestRoad(const QPointF &to, double &s, double &t, QVector2D &vec)
{
	RoadSystem * roadSystem = getProjectData()->getRoadSystem();
	QMap<QString, RSystemElementRoad *> roads = getProjectData()->getRoadSystem()->getRoads();

	if (roads.count() < 1)
	{
		return NULL;
	}

	QMap<QString, RSystemElementRoad *>::const_iterator it = roads.constBegin();
	RSystemElementRoad *road = it.value();
	s = road->getSFromGlobalPoint(to, 0.0, road->getLength());
	vec = QVector2D(road->getGlobalPoint(s) - to);
	t = vec.length();

	while (++it != roads.constEnd())
	{
		RSystemElementRoad *newRoad = it.value();
		double newS = newRoad->getSFromGlobalPoint(to, 0.0, newRoad->getLength());
		QVector2D newVec = QVector2D(newRoad->getGlobalPoint(s) - to);
		double dist = newVec.length();

		if (dist < t)
		{
			road = newRoad;
			t = dist;
			s = newS;
			vec = newVec;
		}
	}

	QVector2D normal = road->getGlobalNormal(s);

	if (QVector2D::dotProduct(normal, vec) < 0)
	{
		t = -t;
	}

	return road;
}


bool 
OpenScenarioEditor::translateObject(OpenScenario::oscObject * object, RSystemElementRoad *newRoad, QPointF &to)
{
 //   RSystemElementRoad * road = object->getParentRoad();
	RSystemElementRoad * road;

    getProjectData()->getUndoStack()->beginMacro(QObject::tr("Move Object"));
    bool parentChanged = false;
    if (newRoad != road)
    {
    //    RemoveObjectCommand * removeObjectCommand = new RemoveObjectCommand(object, road);
     //   getProjectGraph()->executeCommand(removeObjectCommand);

 //       AddObjectCommand * AddObjectCommand = new AddObjectCommand(object, newRoad);
 //       getProjectGraph()->executeCommand(AddObjectCommand);
//		object->setElementSelected(false);

        road = newRoad;
        parentChanged = true;
    }  

	double s = road->getSFromGlobalPoint(to, 0.0, road->getLength());
	QVector2D vec = QVector2D(road->getGlobalPoint(s) - to);
	double t = vec.length();

	QVector2D normal = road->getGlobalNormal(s);

	if (QVector2D::dotProduct(normal, vec) < 0)
	{
		t = -t;
	}

 /*   SetObjectPropertiesCommand * objectPropertiesCommand = new SetObjectPropertiesCommand(object, object->getId(), object->getName(), object->getType(), t, object->getzOffset(), object->getValidLength(), object->getOrientation(), object->getLength(), object->getWidth(), object->getRadius(), object->getHeight(), object->getHeading(), object->getPitch(), object->getRoll(), object->getPole(), s, object->getRepeatLength(), object->getRepeatDistance(), object->getTextureFileName());
    getProjectGraph()->executeCommand(objectPropertiesCommand);
    MoveRoadSectionCommand * moveSectionCommand = new MoveRoadSectionCommand(object, s, RSystemElementRoad::DRS_ObjectSection);
    getProjectGraph()->executeCommand(moveSectionCommand);*/

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
OpenScenarioEditor::catalogChanged(OpenScenario::oscObjectBase *object)
{
	oscCatalog_ = object;
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
	else if (currentTool == ODD::TOS_ELEMENT)
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
					RSystemElementRoad * road = findClosestRoad(mousePoint, s, t, vec);
					if (road)
					{
						// Create new object //
						OpenScenario::OpenScenarioBase *openScenarioBase = oscBase_->getOpenScenarioBase();
						OpenScenario::oscObjectBase *entitiesObject = openScenarioBase->getMember("entities")->getGenerateObject();

						OpenScenario::oscObjectBase *entities = entitiesObject->getMember("objects")->getGenerateObject();
						OpenScenario::oscObject *entity = dynamic_cast<oscObject *>(entities->getMember("object")->getGenerateObject());
						
						OpenScenario::oscObjectBase *oscPosition = entity->getMember("initPosition")->getGenerateObject();
						OpenScenario::oscObjectBase *oscPosRoad = oscPosition->getMember("positionRoad")->getGenerateObject();
						oscPosRoad->getMember("roadId")->getGenerateValue()->setValue(road->getID().toStdString());
						oscPosRoad->getMember("s")->getGenerateValue()->setValue(s);
						oscPosRoad->getMember("t")->getGenerateValue()->setValue(t);

	//					OpenScenario::oscObjectBase *selectedObject = oscCatalog_->getMember(catalogElement_.toStdString())->getGenerateObject();
						
	/*					std::string type = oscCatalog_->getOwnMember()->getTypeName();
						std::size_t found = type.find("Catalog");
						type = type.substr(0, found-1); */
						std::string type("vehicle");

	//					OpenScenario::oscObjectBase *selectedObject = oscCatalog_->getMember(catalogElement_.toStdString())->getGenerateObject();
						OpenScenario::oscObjectBase *selectedObject = oscCatalog_->getMember(type)->getGenerateObject();
						OSCItem *oscItem = new OSCItem(oscRoadSystemItem_, entity, selectedObject, mousePoint);  
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
				OpenScenario::OpenScenarioBase *openScenarioBase = oscBase_->getOpenScenarioBase();
				OpenScenario::oscObjectBase *catalogObject = openScenarioBase->getMember("catalogs")->getGenerateObject();

				OpenScenario::oscMember *catalogMember = catalogObject->getMember(objectName.toStdString());
				if (!catalogMember)
				{
					catalogObject = catalogObject->getMember("objectCatalog")->getGenerateObject();
					catalogMember = catalogObject->getMember(objectName.toStdString());
				}
				OpenScenario::oscObjectBase *catalog = catalogMember->getGenerateObject();
				oscBase_->getOSCElement(catalogObject);
				OSCElement *oscElement = oscBase_->getOSCElement(catalog);

				catalogTree_ = getProjectWidget()->addCatalogTree(objectName, oscElement);   
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
	else if (currentTool != lastTool_)
	{		
		if (currentTool == ODD::TOS_SAVE_CATALOG)
		{
			// Save catalog //
			//			if (catalog_ && mainWindow_->getActiveProject()->saveCatalogAs())
			if (mainWindow_->getActiveProject()->saveAs())
			{
				printStatusBarMsg(tr("File has been saved."), 2000);
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
