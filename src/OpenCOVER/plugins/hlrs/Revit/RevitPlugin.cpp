/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /****************************************************************************\
 **                                                            (C)2009 HLRS  **
 **                                                                          **
 ** Description: Revit Plugin (connection to Autodesk Revit Architecture)    **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** Mar-09  v1	    				       		                                   **
 **                                                                          **
 **                                                                          **
 \****************************************************************************/

#include "RevitPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/RenderObject.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>
#include <cover/coVRSelectionManager.h>
#include <cover/coVRTui.h>
#include <cover/coVRShader.h>
#include <cover/OpenCOVER.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/osg/OSGVruiUserDataCollection.h>
#include <OpenVRUI/osg/mathUtils.h>

#include <PluginUtil/PluginMessageTypes.h>


#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Array>
#include <osg/CullFace>
#include <osg/MatrixTransform>

#include <osgDB/ReadFile>
#include "GenNormals.h"

#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <net/tokenbuffer.h>
#include <config/CoviseConfig.h>
#include <util/unixcompat.h>

using covise::TokenBuffer;
using covise::coCoviseConfig;

int ElementInfo::yPos = 3;

static void matrix2array(const osg::Matrix &m, osg::Matrix::value_type *a)
{
	for (unsigned y = 0; y < 4; ++y)
		for (unsigned x = 0; x < 4; ++x)
		{
			a[y * 4 + x] = m(x, y);
		}
}

static void array2matrix(osg::Matrix &m, const osg::Matrix::value_type *a)
{
	for (unsigned y = 0; y < 4; ++y)
		for (unsigned x = 0; x < 4; ++x)
		{
			m(x, y) = a[y * 4 + x];
		}
}

RevitInfo::RevitInfo()
{
}
RevitInfo::~RevitInfo()
{
}

ElementInfo::ElementInfo()
{
	frame = NULL;
};

ElementInfo::~ElementInfo()
{
	for (std::list<RevitParameter *>::iterator it = parameters.begin();
		it != parameters.end(); it++)
	{
		delete *it;
	}
	delete frame;
};
void ElementInfo::addParameter(RevitParameter *p)
{
	if (frame == NULL)
	{
		frame = new coTUIFrame(name, RevitPlugin::instance()->revitTab->getID());
		frame->setPos(0, yPos);
	}
	if (yPos > 200)
	{
		return;
	}
	yPos++;
	p->createTUI(frame, parameters.size());
	parameters.push_back(p);
}

RevitParameter::~RevitParameter()
{
	delete tuiLabel;
	delete tuiElement;
}

void RevitParameter::tabletEvent(coTUIElement *tUIItem)
{
	TokenBuffer tb;
	tb << element->ID;
	tb << ID;
	switch (StorageType)
	{
	case RevitPlugin::Double:
	{
		coTUIEditFloatField *ef = (coTUIEditFloatField *)tUIItem;
		tb << (double)ef->getValue();
	}
	break;
	case RevitPlugin::ElementId:
	{
		coTUIEditIntField *ef = (coTUIEditIntField *)tUIItem;
		tb << ef->getValue();
	}
	break;
	case RevitPlugin::Integer:
	{
		coTUIEditIntField *ef = (coTUIEditIntField *)tUIItem;
		tb << ef->getValue();
	}
	break;
	case RevitPlugin::String:
	{
		coTUIEditField *ef = (coTUIEditField *)tUIItem;
		tb << ef->getText();
	}
	break;
	default:
	{
		coTUIEditField *ef = (coTUIEditField *)tUIItem;
		tb << ef->getText();
	}
	break;
	}
	Message m(tb);
	m.type = (int)RevitPlugin::MSG_SetParameter;
	RevitPlugin::instance()->sendMessage(m);
}
void RevitParameter::createTUI(coTUIFrame *frame, int pos)
{
	tuiLabel = new coTUILabel(name, frame->getID());
	tuiLabel->setLabel(name);
	tuiLabel->setPos(0, pos);
	tuiElement = NULL;
	switch (StorageType)
	{
	case RevitPlugin::Double:
	{
		coTUIEditFloatField *ef = new coTUIEditFloatField(name + "ef", frame->getID());
		ef->setPos(1, pos);
		ef->setValue(d);
		tuiElement = ef;
	}
	break;
	case RevitPlugin::ElementId:
	{
		coTUIEditIntField *ef = new coTUIEditIntField(name + "ei", frame->getID());
		ef->setPos(1, pos);
		ef->setValue(i);
		tuiElement = ef;
	}
	break;
	case RevitPlugin::Integer:
	{
		coTUIEditIntField *ef = new coTUIEditIntField(name + "ei", frame->getID());
		ef->setPos(1, pos);
		ef->setValue(i);
		tuiElement = ef;
	}
	break;
	case RevitPlugin::String:
	{
		coTUIEditField *ef = new coTUIEditField(name + "e", frame->getID());
		ef->setPos(1, pos);
		ef->setText(s);
		tuiElement = ef;
	}
	break;
	default:
	{
		coTUIEditField *ef = new coTUIEditField(name + "e", frame->getID());
		ef->setPos(1, pos);
		ef->setText(s);
		tuiElement = ef;
	}
	break;
	}

	tuiElement->setEventListener(this);
}

RevitViewpointEntry::RevitViewpointEntry(osg::Vec3 pos, osg::Vec3 dir, osg::Vec3 up, RevitPlugin *plugin, std::string n, int id, coCheckboxMenuItem *me)
	: menuItem(NULL)
{
	myPlugin = plugin;
	name = n;
	entryNumber = plugin->maxEntryNumber++;
	eyePosition = pos;
	viewDirection = -dir;
	upDirection = up;
	ID = id;
	menuEntry = me;
	isActive = false;

	tuiItem = new coTUIToggleButton(name.c_str(), plugin->revitTab->getID());
	tuiItem->setEventListener(plugin);
	tuiItem->setPos((int)(entryNumber / 10.0) + 1, entryNumber % 10);
}

void RevitViewpointEntry::setValues(osg::Vec3 pos, osg::Vec3 dir, osg::Vec3 up, std::string n)
{
	name = n;
	eyePosition = pos;
	viewDirection = -dir;
	upDirection = up;
}

RevitViewpointEntry::~RevitViewpointEntry()
{
	delete menuItem;
}

void RevitViewpointEntry::setMenuItem(coCheckboxMenuItem *aButton)
{
	menuItem = aButton;
}
void RevitViewpointEntry::deactivate()
{
	menuEntry->setState(false);
	tuiItem->setState(false);
	isActive = false;
}

void RevitViewpointEntry::activate()
{
	RevitPlugin::instance()->deactivateAllViewpoints();

	tuiItem->setState(true);
	menuEntry->setState(true);
	isActive = true;
	osg::Matrix mat, rotMat;
	mat.makeTranslate(-eyePosition[0] * REVIT_FEET_TO_M, -eyePosition[1] * REVIT_FEET_TO_M, -eyePosition[2] * REVIT_FEET_TO_M);
	//rotMat.makeRotate(-ori[3], Vec3(ori[0],ori[1],ori[2]));
	rotMat.makeIdentity();
	osg::Vec3 xDir = viewDirection ^ upDirection;

	rotMat(0, 0) = xDir[0];
	rotMat(0, 1) = xDir[1];
	rotMat(0, 2) = xDir[2];
	rotMat(1, 0) = viewDirection[0];
	rotMat(1, 1) = viewDirection[1];
	rotMat(1, 2) = viewDirection[2];
	rotMat(2, 0) = upDirection[0];
	rotMat(2, 1) = upDirection[1];
	rotMat(2, 2) = upDirection[2];
	osg::Matrix irotMat;
	irotMat.invert(rotMat);
	mat.postMult(irotMat);

	osg::Matrix scMat;
	osg::Matrix iscMat;
	float scaleFactor = 1000;
	cover->setScale(scaleFactor);
	scMat.makeScale(scaleFactor, scaleFactor, scaleFactor);
	iscMat.makeScale(1.0 / scaleFactor, 1.0 / scaleFactor, 1.0 / scaleFactor);
	mat.postMult(scMat);
	mat.preMult(iscMat);
	osg::Matrix viewerTrans;
	viewerTrans.makeTranslate(cover->getViewerMat().getTrans());
	mat.postMult(viewerTrans);
	cover->setXformMat(mat);
}

void RevitViewpointEntry::updateCamera()
{
	osg::Matrix m;
	std::string path;
	TokenBuffer stb;
	stb << ID;

	osg::Matrix mat = cover->getXformMat();
	osg::Matrix viewerTrans;
	viewerTrans.makeTranslate(cover->getViewerMat().getTrans());
	osg::Matrix itransMat;
	itransMat.invert(viewerTrans);
	mat.postMult(itransMat);


	osg::Matrix scMat;
	osg::Matrix iscMat;
	float scaleFactor = cover->getScale();
	scMat.makeScale(scaleFactor, scaleFactor, scaleFactor);
	iscMat.makeScale(1.0 / scaleFactor, 1.0 / scaleFactor, 1.0 / scaleFactor);
	mat.postMult(iscMat);
	mat.preMult(scMat);

	osg::Matrix irotMat = mat;
	irotMat.setTrans(0, 0, 0);

	osg::Matrix rotMat;
	rotMat.invert(irotMat);
	mat.postMult(rotMat);
	osg::Vec3 eyePos = mat.getTrans();
	eyePosition[0] = -eyePos[0] / REVIT_FEET_TO_M;
	eyePosition[1] = -eyePos[1] / REVIT_FEET_TO_M;
	eyePosition[2] = -eyePos[2] / REVIT_FEET_TO_M;

	viewDirection[0] = rotMat(1, 0);
	viewDirection[1] = rotMat(1, 1);
	viewDirection[2] = rotMat(1, 2);
	upDirection[0] = rotMat(2, 0);
	upDirection[1] = rotMat(2, 1);
	upDirection[2] = rotMat(2, 2);

	stb << (double)eyePosition[0];
	stb << (double)eyePosition[1];
	stb << (double)eyePosition[2];
	stb << (double)viewDirection[0];
	stb << (double)viewDirection[1];
	stb << (double)viewDirection[2];
	stb << (double)upDirection[0];
	stb << (double)upDirection[1];
	stb << (double)upDirection[2];

	Message message(stb);
	message.type = (int)RevitPlugin::MSG_UpdateView;
	RevitPlugin::instance()->sendMessage(message);
}


void RevitViewpointEntry::menuEvent(coMenuItem *aButton)
{
	if (((coCheckboxMenuItem *)aButton)->getState())
	{
		activate();
	}
}

void RevitPlugin::createMenu()
{

	maxEntryNumber = 0;
	cbg = new coCheckboxGroup();
	viewpointMenu = new coRowMenu("Revit Viewpoints");

	REVITButton = new coSubMenuItem("Revit");
	REVITButton->setMenu(viewpointMenu);

	roomInfoMenu = new coRowMenu("Room Information");

	roomInfoButton = new coSubMenuItem("Room Info");
	roomInfoButton->setMenu(roomInfoMenu);
	viewpointMenu->add(roomInfoButton);
	label1 = new coLabelMenuItem("No Room");
	roomInfoMenu->add(label1);
	addCameraButton = new coButtonMenuItem("Add Camera");
	addCameraButton->setMenuListener(this);
	viewpointMenu->add(addCameraButton);
	updateCameraButton = new coButtonMenuItem("UpdateCamera");
	updateCameraButton->setMenuListener(this);
	viewpointMenu->add(updateCameraButton);

	cover->getMenu()->add(REVITButton);

	revitTab = new coTUITab("Revit", coVRTui::instance()->mainFolder->getID());
	revitTab->setPos(0, 0);

	updateCameraTUIButton = new coTUIButton("Update Camera", revitTab->getID());
	updateCameraTUIButton->setEventListener(this);
	updateCameraTUIButton->setPos(0, 0);


	addCameraTUIButton = new coTUIButton("Add Camera", revitTab->getID());
	addCameraTUIButton->setEventListener(this);
	addCameraTUIButton->setPos(0, 1);

	viewsCombo = new coTUIComboBox("views", revitTab->getID());
	viewsCombo->setEventListener(this);
	viewsCombo->setPos(0, 2);
}

void RevitPlugin::destroyMenu()
{
	delete roomInfoButton;
	delete roomInfoMenu;
	delete label1;
	delete viewpointMenu;
	delete REVITButton;
	delete cbg;

	delete addCameraTUIButton;
	delete updateCameraTUIButton;
	delete viewsCombo;
	delete revitTab;
}

RevitPlugin::RevitPlugin()
{
	fprintf(stderr, "RevitPlugin::RevitPlugin\n");
	plugin = this;
	MoveFinished = true;
    setViewpoint = true;
	int port = coCoviseConfig::getInt("port", "COVER.Plugin.Revit.Server", 31821);
    textureDir = coCoviseConfig::getEntry("textures", "COVER.Plugin.Revit", "C:/Program Files (x86)/Common Files/Autodesk Shared/Materials/Textures");
    localTextureDir = coCoviseConfig::getEntry("localTextures", "COVER.Plugin.Revit", "c:/tmp");
	toRevit = NULL;
    serverConn = NULL;
    if (coVRMSController::instance()->isMaster())
    {
        serverConn = new ServerConnection(port, 1234, Message::UNDEFINED);
        if (!serverConn->getSocket())
        {
            cout << "tried to open server Port " << port << endl;
            cout << "Creation of server failed!" << endl;
            cout << "Port-Binding failed! Port already bound?" << endl;
            delete serverConn;
            serverConn = NULL;
        }
        else
        {
            cover->watchFileDescriptor(serverConn->getSocket()->get_id());
        }
    }

	struct linger linger;
	linger.l_onoff = 0;
	linger.l_linger = 0;
	cout << "Set socket options..." << endl;
	if (serverConn)
	{
		setsockopt(serverConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

		cout << "Set server to listen mode..." << endl;
		serverConn->listen();
		if (!serverConn->is_connected()) // could not open server port
		{
			fprintf(stderr, "Could not open server port %d\n", port);
			delete serverConn;
			serverConn = NULL;
		}
	}
	msg = new Message;

}

bool RevitPlugin::init()
{
	cover->addPlugin("Annotation"); // we would like to have the Annotation plugin
	cover->addPlugin("Move"); // we would like to have the Move plugin
	globalmtl = new osg::Material;
	globalmtl->ref();
	globalmtl->setColorMode(osg::Material::OFF);
	globalmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0));
	globalmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
	globalmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
	globalmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
	globalmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
	revitGroup = new osg::MatrixTransform();
	revitGroup->setName("RevitGeometry");
	scaleFactor = REVIT_FEET_TO_M; // Revit internal units are always feet
	revitGroup->setMatrix(osg::Matrix::scale(scaleFactor, scaleFactor, scaleFactor));
	cover->setScale(1000.0);
	currentGroup.push(revitGroup.get());
	cover->getObjectsRoot()->addChild(revitGroup.get());
	createMenu();
	return true;
}
// this is called if the plugin is removed at runtime
RevitPlugin::~RevitPlugin()
{
	destroyMenu();
	while (currentGroup.size() > 1)
		currentGroup.pop();

    if (revitGroup)
    {
        revitGroup->removeChild(0, revitGroup->getNumChildren());
        cover->getObjectsRoot()->removeChild(revitGroup.get());
    }

    if (serverConn && serverConn->getSocket())
        cover->unwatchFileDescriptor(serverConn->getSocket()->get_id());
	delete serverConn;
	serverConn = NULL;

    if (toRevit && toRevit->getSocket())
        cover->unwatchFileDescriptor(toRevit->getSocket()->get_id());
	delete toRevit;
	delete msg;
	toRevit = NULL;
}

void RevitPlugin::menuEvent(coMenuItem *aButton)
{
	if (aButton == updateCameraButton)
	{
		for (list<RevitViewpointEntry *>::iterator it = viewpointEntries.begin();
			it != viewpointEntries.end(); it++)
		{
			if ((*it)->isActive)
			{
				(*it)->updateCamera();
			}
		}
	}
	if (aButton == addCameraButton)
	{
	}
}
void RevitPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
	if (tUIItem == updateCameraTUIButton)
	{
		for (list<RevitViewpointEntry *>::iterator it = viewpointEntries.begin();
			it != viewpointEntries.end(); it++)
		{
			if ((*it)->isActive)
			{
				(*it)->updateCamera();
			}
		}
	}
}

void RevitPlugin::tabletEvent(coTUIElement *tUIItem)
{
	if (tUIItem == addCameraTUIButton)
	{
	}
	else if (tUIItem == viewsCombo)
	{
		TokenBuffer tb;
		tb << viewsCombo->getSelectedEntry();
		Message message(tb);
		message.type = (int)RevitPlugin::MSG_SetView;
		RevitPlugin::instance()->sendMessage(message);
		message.type = (int)RevitPlugin::MSG_Resend;
		RevitPlugin::instance()->sendMessage(message);
	}
	else
	{
		for (list<RevitViewpointEntry *>::iterator it = viewpointEntries.begin();
			it != viewpointEntries.end(); it++)
		{
			if ((*it)->getTUIItem() == tUIItem)
			{
				(*it)->activate();
				break;
			}
		}
	}
}

void RevitPlugin::deactivateAllViewpoints()
{
	for (list<RevitViewpointEntry *>::iterator it = viewpointEntries.begin();
		it != viewpointEntries.end(); it++)
	{
		(*it)->deactivate();
	}
}

void RevitPlugin::setDefaultMaterial(osg::StateSet *geoState)
{
	geoState->setAttributeAndModes(globalmtl.get(), osg::StateAttribute::ON);
}

bool RevitPlugin::sendMessage(Message &m)
{
	if (toRevit) // false on slaves
	{
        if (toRevit->send_msg(&m) > 0)
            return true;
	}
    return false;
}


void RevitPlugin::message(int toWhom, int type, int len, const void *buf)
{
	if (type == PluginMessageTypes::MoveAddMoveNode)
	{
	}
	else if (type == PluginMessageTypes::MoveMoveNodeFinished)
	{
		MoveFinished = true;
		std::string path;
		TokenBuffer tb((const char *)buf, len);
		tb >> path;
		tb >> path;

		osg::Node *selectedNode = coVRSelectionManager::validPath(path);
		if (selectedNode)
		{
			info = dynamic_cast<RevitInfo *>(OSGVruiUserDataCollection::getUserData(selectedNode, "RevitInfo"));
			if (info)
			{
				osg::Matrix m;
				for (int i = 0; i < 4; i++)
					for (int j = 0; j < 4; j++)
						tb >> m(i, j);
				osg::Matrix currentBaseMat, dcsMat;
				osg::Group* currentNode = selectedNode->getParent(0);
				currentBaseMat.makeIdentity();
				while (currentNode != NULL && currentNode != revitGroup)
				{
					if (dynamic_cast<osg::MatrixTransform *>(currentNode))
					{
						dcsMat = ((osg::MatrixTransform *)currentNode)->getMatrix();
						currentBaseMat.postMult(dcsMat);
					}
					if (currentNode->getNumParents() > 0)
						currentNode = currentNode->getParent(0);
					else
						currentNode = NULL;
				}
				currentBaseMat(3, 0) = 0;
				currentBaseMat(3, 1) = 0;
				currentBaseMat(3, 2) = 0;
				m = m* currentBaseMat;
				TokenBuffer stb;
				stb << MovedID;
				stb << (double)m.getTrans().x();
				stb << (double)m.getTrans().y();
				stb << (double)m.getTrans().z();

				Message message(stb);
				message.type = (int)RevitPlugin::MSG_SetTransform;
				RevitPlugin::instance()->sendMessage(message);

			}
		}

	}
	else if (type == PluginMessageTypes::AnnotationMessage) // An AnnotationMessage has been received
	{
		AnnotationMessage *mm = (AnnotationMessage *)buf;

		switch (mm->token)
		{
		case ANNOTATION_MESSAGE_TOKEN_MOVEADD: // MOVE/ADD
		{

			int revitID = getRevitAnnotationID(mm->id);
			if (revitID > 0)
			{
				changeAnnotation(revitID, mm);
			}
			else if (revitID == -1)
			{
				createNewAnnotation(mm->id, mm);
			}
			break;
		} // case moveadd

		case ANNOTATION_MESSAGE_TOKEN_REMOVE: // Remove an annotation
		{
			int revitID = getRevitAnnotationID(mm->id);
			if (revitID > 0)
			{
				TokenBuffer stb;
				stb << revitID;

				Message message(stb);
				message.type = (int)RevitPlugin::MSG_DeleteObject;
				RevitPlugin::instance()->sendMessage(message);
			}
			else if (revitID == -1)
			{
			}
			break;
		} // case remove

		case ANNOTATION_MESSAGE_TOKEN_SELECT: // annotation selected (right-clicked)
		{

			break;
		} // case select

		case ANNOTATION_MESSAGE_TOKEN_COLOR: // Change annotation color
		{
			break;
		} // case color

		case ANNOTATION_MESSAGE_TOKEN_DELETEALL: // Deletes all Annotations
		{
			break;
		} // case deleteall

		// Release current lock on a specific annotation
		// TODO: Possibly remove this, as unlock all
		// does what this is supposed to do
		case ANNOTATION_MESSAGE_TOKEN_UNLOCK:
		{
			break;
		} //case unlock

		case ANNOTATION_MESSAGE_TOKEN_SCALE: // scale an annotation
		{

			break;
		} //case scale

		case ANNOTATION_MESSAGE_TOKEN_SCALEALL: //scale all Annotations
		{

			break;
		} //case scaleall

		case ANNOTATION_MESSAGE_TOKEN_COLORALL: //change all annotation's colors
		{
			break;
		} //case colorall

		// release lock on all annotations that are owned by sender
		case ANNOTATION_MESSAGE_TOKEN_UNLOCKALL:
		{
			break;
		} //case unlockall

		case ANNOTATION_MESSAGE_TOKEN_FORCEUNLOCK:
		{
			break;
		}

		case ANNOTATION_MESSAGE_TOKEN_HIDE: //hide an annotation
		{
			break;
		} //case hide

		case ANNOTATION_MESSAGE_TOKEN_HIDEALL: //hide all annotations
		{
			break;
		} //case hideall

		default:
			std::cerr
				<< "Annotation: Error: Bogus Annotation message with Token "
				<< (int)mm->token << std::endl;
		} //switch mm->token
	} //if type == ann_message
	else if (type == PluginMessageTypes::AnnotationTextMessage)
	{
		std::string text;
		TokenBuffer tb((const char *)buf, len);
		int id, owner;
		char *ctext;
		tb >> id;
		tb >> owner;
		tb >> ctext;
		text = ctext;
		int AnnotationID = getRevitAnnotationID(id);
		if (AnnotationID > 0)
		{
			TokenBuffer stb;
			stb << AnnotationID;
			stb << text;

			Message message(stb);
			message.type = (int)RevitPlugin::MSG_ChangeAnnotationText;
			RevitPlugin::instance()->sendMessage(message);
		}
	}
	else if (type == PluginMessageTypes::MoveMoveNode)
	{
		osg::Matrix m;
		std::string path;
		TokenBuffer tb((const char *)buf, len);
		tb >> path;
		tb >> path;
		osg::Node *selectedNode = coVRSelectionManager::validPath(path);
		if (selectedNode)
		{
			info = dynamic_cast<RevitInfo *>(OSGVruiUserDataCollection::getUserData(selectedNode, "RevitInfo"));
			if (info)
			{
				for (int i = 0; i < 4; i++)
					for (int j = 0; j < 4; j++)
						tb >> m(i, j);
				if (MovedID != info->ObjectID)
				{
					MoveFinished = true;
				}

				lastMoveMat = invStartMoveMat*m;
				//invStartMoveMat.invert(m);

				if (MoveFinished)
				{
					MoveFinished = false;
					MovedID = info->ObjectID;
					invStartMoveMat.invert(m);

					/*   TokenBuffer stb;
					   stb << info->ObjectID;
					   stb << (double)lastMoveMat.getTrans().x();
					   stb << (double)lastMoveMat.getTrans().y();
					   stb << (double)lastMoveMat.getTrans().z();

					   Message message(stb);
					   message.type = (int)RevitPlugin::MSG_SetTransform;
					   RevitPlugin::instance()->sendMessage(message);*/
				}
			}
		}
	}
	else if (type >= PluginMessageTypes::HLRS_Revit_Message && type <= (PluginMessageTypes::HLRS_Revit_Message + 100))
	{
        Message m{ type - PluginMessageTypes::HLRS_Revit_Message + MSG_NewObject , DataHandle{(char*)buf, len, false} };
		handleMessage(&m);
	}

}

RevitPlugin *RevitPlugin::plugin = NULL;
void
RevitPlugin::handleMessage(Message *m)
{
	//cerr << "got Message" << endl;
	//m->print();
	enum MessageTypes type = (enum MessageTypes)m->type;

	switch (type)
	{
	case MSG_Views:
	{
		TokenBuffer tb(m);
		int numViews;
		tb >> numViews;
		viewsCombo->clear();
		for (int i = 0; i < numViews; i++)
		{
			std::string ViewName;
			tb >> ViewName;
			viewsCombo->addEntry(ViewName);
		}
	}
	break;
    case MSG_Finished:
    {
        for (auto Mat = MaterialInfos.begin(); Mat != MaterialInfos.end(); Mat++)
        {
            if (Mat->second->diffuseTexture->requestTexture)
            {
                requestTexture(Mat->second->ID, Mat->second->diffuseTexture);
            }
            if (Mat->second->bumpTexture->requestTexture)
            {
                requestTexture(Mat->second->ID, Mat->second->bumpTexture);
            }
        }
    }
    break;
	case MSG_RoomInfo:
	{
		TokenBuffer tb(m);
		double area;
		char *roomNumber;
		char *roomName;
		char *levelName;
		tb >> roomNumber;
		tb >> roomName;
		tb >> area;
		tb >> levelName;
		char info[1000];
		sprintf(info, "Nr.: %s\n%s\nArea: %3.7lfm^2\nLevel: %s", roomNumber, roomName, area / 10.0, levelName);
		label1->setLabel(info);
		//fprintf(stderr,"Room %s %s Area: %lf Level: %s\n", roomNumber,roomName,area,levelName);
	}
	break;
	case MSG_NewParameter:
	{
		TokenBuffer tb(m);
		int ID;
		tb >> ID;
		int numParams;
		tb >> numParams;
		std::map<int, ElementInfo *>::iterator it = ElementIDMap.find(ID);
		if (it != ElementIDMap.end())
		{
			for (int i = 0; i < numParams; i++)
			{
				fprintf(stderr, "PFound: %d\n", ID);
				int pID;
				tb >> pID;
				char *name;
				tb >> name;
				int StorageType;
				tb >> StorageType;
				int ParameterType;
				tb >> ParameterType;
				ElementInfo *ei = it->second;
				RevitParameter *p = new RevitParameter(pID, std::string(name), StorageType, ParameterType, (int)ei->parameters.size(), ei);
				switch (StorageType)
				{
				case Double:
					tb >> p->d;
					break;
				case ElementId:
					tb >> p->ElementReferenceID;
					break;
				case Integer:
					tb >> p->i;
					break;
				case String:
					tb >> p->s;
					break;
				default:
					tb >> p->s;
					break;
				}
				ei->addParameter(p);
			}
		}
	}
	break;
	case MSG_NewGroup:
	{
		TokenBuffer tb(m);
		int ID;
		tb >> ID;
		char *name;
		tb >> name;
		osg::Group *newGroup = new osg::Group();
		newGroup->setName(name);
		currentGroup.top()->addChild(newGroup);

		RevitInfo *info = new RevitInfo();
		info->ObjectID = ID;
		OSGVruiUserDataCollection::setUserData(newGroup, "RevitInfo", info);
		currentGroup.push(newGroup);
	}
	break;
	case MSG_NewDoorGroup:
	{
		TokenBuffer tb(m);
		int ID;
		tb >> ID;
		char *name;
		tb >> name;
		osg::MatrixTransform *newTrans = new osg::MatrixTransform();
		newTrans->setName(name);
		currentGroup.top()->addChild(newTrans);
		doors.push_back(new DoorInfo(ID, name, newTrans, tb));
		RevitInfo *info = new RevitInfo();
		info->ObjectID = ID;
		OSGVruiUserDataCollection::setUserData(newTrans, "RevitInfo", info);
		currentGroup.push(newTrans);
	}
	break;
	case MSG_DeleteElement:
	{
		TokenBuffer tb(m);
		int ID;
		tb >> ID;
		std::map<int, ElementInfo *>::iterator it = ElementIDMap.find(ID);
		if (it != ElementIDMap.end())
		{
			//fprintf(stderr, "DFound: %d\n", ID);
			ElementInfo *ei = it->second;
			for (std::list<osg::Node *>::iterator nodesIt = ei->nodes.begin(); nodesIt != ei->nodes.end(); nodesIt++)
			{
				osg::Node *n = *nodesIt;
				const MatrixTransform *mt = dynamic_cast<const MatrixTransform *>(n);
				if (mt != NULL)
				{// this could be a door

					for (std::list<DoorInfo *>::iterator it = activeDoors.begin(); it != activeDoors.end();)
					{
						if ((*it)->transformNode == mt)
						{
							activeDoors.erase(it);
							break;
						}
					}
					for (std::list<DoorInfo *>::iterator it = doors.begin(); it != doors.end(); it++)
					{
						if ((*it)->transformNode == mt)
						{
							DoorInfo *di = *it;
							doors.erase(it);
							delete di;
							break;
						}
					}
				}
				while (n->getNumParents())
				{
					n->getParent(0)->removeChild(n);
				}
				//fprintf(stderr, "DeleteID: %d\n", ID);
			}
			delete ei;
			ElementIDMap.erase(it);
		}
	}
	break;
	case MSG_NewTransform:
	{
		TokenBuffer tb(m);
		int ID;
		tb >> ID;
		char *name;
		tb >> name;
		osg::MatrixTransform *newTrans = new osg::MatrixTransform();
		osg::Matrix m;
		m.makeIdentity();
		float x, y, z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(0, 0) = x;
		m(0, 1) = y;
		m(0, 2) = z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(1, 0) = x;
		m(1, 1) = y;
		m(1, 2) = z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(2, 0) = x;
		m(2, 1) = y;
		m(2, 2) = z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(3, 0) = x;
		m(3, 1) = y;
		m(3, 2) = z;
		newTrans->setMatrix(m);
		newTrans->setName(name);
		currentGroup.top()->addChild(newTrans);
		currentGroup.push(newTrans);
	}
	break;
	case MSG_EndGroup:
	{
		currentGroup.pop();
	}
	break;
	case MSG_NewInstance:
	{
		TokenBuffer tb(m);
		int ID;
		tb >> ID;
		char *name;
		tb >> name;
		osg::MatrixTransform *newTrans = new osg::MatrixTransform();
		osg::Matrix m;
		m.makeIdentity();
		float x, y, z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(0, 0) = x;
		m(0, 1) = y;
		m(0, 2) = z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(1, 0) = x;
		m(1, 1) = y;
		m(1, 2) = z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(2, 0) = x;
		m(2, 1) = y;
		m(2, 2) = z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(3, 0) = x;
		m(3, 1) = y;
		m(3, 2) = z;
		newTrans->setMatrix(m);
		newTrans->setName(name);
		currentGroup.top()->addChild(newTrans);
		currentGroup.push(newTrans);
	}
	break;
	case MSG_EndInstance:
	{
		currentGroup.pop();
	}
	break;
	case MSG_ClearAll:
	{
		while (currentGroup.size() > 1)
			currentGroup.pop();

		revitGroup->removeChild(0, revitGroup->getNumChildren());

		// remove viewpoints
		maxEntryNumber = 0;
		for (std::list<RevitViewpointEntry *>::iterator it = viewpointEntries.begin();
			it != viewpointEntries.end(); it++)
		{
			delete *it;
		}
		viewpointEntries.clear();
		for (std::map<int, ElementInfo *>::iterator it = ElementIDMap.begin(); it != ElementIDMap.end(); it++)
		{
			delete (it->second);
		}
		ElementIDMap.clear();
		for (std::map<int, MaterialInfo *>::iterator it = MaterialInfos.begin(); it != MaterialInfos.end(); it++)
		{
			delete (it->second);
		}
		MaterialInfos.clear();

		for (std::list<DoorInfo *>::iterator it = doors.begin(); it != doors.end(); it++)
		{
			delete *it;
		}
		doors.clear();
		activeDoors.clear();

	}
	break;
	case MSG_NewAnnotation:
	{
		TokenBuffer tb(m);
		int ID;
		float x, y, z;
		char *text;
		tb >> ID;
		tb >> x;
		tb >> y;
		tb >> z;
		tb >> text;

		int AID = getAnnotationID(ID);
		AnnotationMessage am;
		am.token = ANNOTATION_MESSAGE_TOKEN_MOVEADD;
		am.id = AID;
		am.sender = 101;
		am.color = 0.4;
		osg::Matrix trans;
		osg::Matrix orientation;
		trans.makeTranslate(x*scaleFactor, y*scaleFactor, z*scaleFactor); // get rid of scale part
		orientation.makeRotate(3.0, -1, 0, 0);
		matrix2array(trans, am.translation());
		matrix2array(orientation, am.orientation());
		cover->sendMessage(this, "Annotation",
			PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);

		TokenBuffer tb3;
		tb3 << AID;
		tb3 << 101; // owner
		tb3 << text;
		cover->sendMessage(this, "Annotation",
			PluginMessageTypes::AnnotationTextMessage, tb3.getData().length(), tb3.getData().data());
		break;
	}
    case MSG_DocumentInfo:
    {
        TokenBuffer tb(m);
        char *fileName;
        tb >> fileName;
        if (fileName != currentRevitFile)
        {
            setViewpoint = true;
            currentRevitFile = fileName;
        }
        else
        {
            setViewpoint = false;
        }
        break;
    }
	case MSG_AddView:
	{
		TokenBuffer tb(m);
		int ID;
		tb >> ID;
		char *name;
		tb >> name;
		float x, y, z;
		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 pos(x, y, z);
		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 dir(x, y, z);
		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 up(x, y, z);

		bool foundIt = false;

		for (list<RevitViewpointEntry *>::iterator it = viewpointEntries.begin();
			it != viewpointEntries.end(); it++)
		{
			if ((*it)->ID == ID)
			{
				foundIt = true;
				RevitViewpointEntry *vpe = (*it);
				if (vpe->isActive)
				{
					vpe->setValues(pos, dir, up, name);
					vpe->activate();
				}
                if (setViewpoint && strncasecmp("Start",vpe->getName().c_str(),5)==0)
                {
                    vpe->activate();
                }
				break;
			}
		}
		if (!foundIt)
		{

			coCheckboxMenuItem *menuEntry;

			menuEntry = new coCheckboxMenuItem(name, false, cbg);
			// add viewpoint to menu
			RevitViewpointEntry *vpe = new RevitViewpointEntry(pos, dir, up, this, name, ID, menuEntry);
			menuEntry->setMenuListener(vpe);
			viewpointMenu->add(menuEntry);
			vpe->setMenuItem(menuEntry);
			viewpointEntries.push_back(vpe);

            if (setViewpoint && strncasecmp("Start", vpe->getName().c_str(),5) == 0)
                vpe->activate();
		}
	}
	break;
	case MSG_NewAnnotationID:
	{
		TokenBuffer tb(m);
		int annotationID;
		int ID;
		tb >> annotationID;
		tb >> ID;
		while (annotationIDs.size() <= annotationID)
			annotationIDs.push_back(-1);
		annotationIDs[annotationID] = ID;
		// check if we have cached changes for this Annotation and send it to Revit.
	}
	case MSG_NewMaterial:
	{
		TokenBuffer tb(m);
		MaterialInfo*mi = new MaterialInfo(tb);
		MaterialInfos[mi->ID] = mi;

		mi->geoState = new osg::StateSet;
		setDefaultMaterial(mi->geoState);
		mi->geoState->setMode(GL_LIGHTING, osg::StateAttribute::ON);
		osg::Material *localmtl = new osg::Material;
		localmtl->setColorMode(osg::Material::OFF);
		localmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(mi->r / 255.0f, mi->g / 255.0f, mi->b / 255.0f, mi->a / 255.0f));
		localmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(mi->r / 255.0f, mi->g / 255.0f, mi->b / 255.0f, mi->a / 255.0f));
		localmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
		localmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
		localmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

		mi->geoState->setAttributeAndModes(localmtl, osg::StateAttribute::ON);
		if (mi->diffuseTexture->texturePath != "")
		{
            std::string fileName = mi->diffuseTexture->texturePath;
            osg::ref_ptr<osg::Image> diffuseImage = readImage(fileName);

            if (diffuseImage.valid())
			{
                mi->updateTexture(TextureInfo::diffuse, diffuseImage.get());
			}
            else
            {
                mi->diffuseTexture->requestTexture = true;
            }
		}

		if (mi->bumpTexture->texturePath != "")
		{
			std::string fileName = mi->bumpTexture->texturePath;
			osg::ref_ptr<osg::Image> bumpImage = readImage(fileName);
			if (!bumpImage.valid())
			{
				//osg::notify(osg::ALWAYS) << "Can't open image file" << fileName << " couldn't get it from remote" << std::endl;
                mi->bumpTexture->requestTexture = true;
			}
			else
			{

                mi->updateTexture(TextureInfo::bump,bumpImage.get());
			}
		}
		break;
	}
	case MSG_NewObject:
	{
		TokenBuffer tb(m);
		int ID;
		int GeometryType;
		tb >> ID;
		ElementInfo *ei;
		std::map<int, ElementInfo *>::iterator it = ElementIDMap.find(ID);
		if (it != ElementIDMap.end())
		{
			ei = it->second;
			//fprintf(stderr, "NFound: %d\n", ID);
		}
		else
		{
			ei = new ElementInfo();
			ElementIDMap[ID] = ei;
			//fprintf(stderr, "NewID: %d\n", ID);
		}
		char *name;
		tb >> name;
		ei->name = name;
		ei->ID = ID;
		tb >> GeometryType;
		if (GeometryType == OBJ_TYPE_Mesh)
		{

			osg::Geode *geode = new osg::Geode();
			geode->setName(name);
			osg::Geometry *geom = new osg::Geometry();
			cover->setRenderStrategy(geom);
			geode->addDrawable(geom);

			// set up geometry
			bool isTwoSided = false;
			char tmpChar;
			tb >> tmpChar;
			if (tmpChar != '\0')
				isTwoSided = true;

			int numTriangles;
			tb >> numTriangles;

			osg::Vec3Array *vert = new osg::Vec3Array;
			osg::DrawArrays *triangles = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, numTriangles * 3);
			for (int i = 0; i < numTriangles; i++)
			{
				float x, y, z;
				tb >> x;
				tb >> y;
				tb >> z;
				vert->push_back(osg::Vec3(x, y, z));
				tb >> x;
				tb >> y;
				tb >> z;
				vert->push_back(osg::Vec3(x, y, z));
				tb >> x;
				tb >> y;
				tb >> z;
				vert->push_back(osg::Vec3(x, y, z));
			}
			unsigned char r, g, b, a;
			int MaterialID;
			tb >> r;
			tb >> g;
			tb >> b;
			tb >> a;
			tb >> MaterialID;

			geom->setVertexArray(vert);
			geom->addPrimitiveSet(triangles);
			GenNormalsVisitor *sv = new GenNormalsVisitor(45.0);
			sv->apply(*geode);
			ei->nodes.push_back(geode);

			osg::StateSet *geoState;
			MaterialInfo *mi = getMaterial(MaterialID);
			if (mi && mi->shader != NULL)
			{
				geoState = mi->geoState;
				geode->setStateSet(geoState);
				mi->shader->apply(geode, geom); // generates tangents if required
			}
			else
			{
				geoState = geode->getOrCreateStateSet();
				setDefaultMaterial(geoState);
				geoState->setMode(GL_LIGHTING, osg::StateAttribute::ON);
				geode->setStateSet(geoState);
				osg::Material *localmtl = new osg::Material;
				localmtl->setColorMode(osg::Material::OFF);
				localmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f));
				localmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f));
				localmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
				localmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
				localmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

				geoState->setAttributeAndModes(localmtl, osg::StateAttribute::ON);
			}

			if (a < 250)
			{
				geoState->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
				geoState->setMode(GL_BLEND, osg::StateAttribute::ON);
				geoState->setNestRenderBins(false);
			}
			else
			{
				geoState->setRenderingHint(osg::StateSet::OPAQUE_BIN);
				geoState->setMode(GL_BLEND, osg::StateAttribute::OFF);
				geoState->setNestRenderBins(false);
			}

			if (!isTwoSided)
			{
				osg::CullFace *cullFace = new osg::CullFace();
				cullFace->setMode(osg::CullFace::BACK);
				geoState->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
			}


			RevitInfo *info = new RevitInfo();
			info->ObjectID = ID;
			OSGVruiUserDataCollection::setUserData(geode, "RevitInfo", info);
			currentGroup.top()->addChild(geode);
		}
		else if (GeometryType == OBJ_TYPE_Inline)
		{
			std::string url;
			tb >> url;
			osg::MatrixTransform *mt = new osg::MatrixTransform();
			mt->setMatrix(osg::Matrix::scale(REVIT_M_TO_FEET, REVIT_M_TO_FEET, REVIT_M_TO_FEET));
			char buffer[333];
			sprintf(buffer, "Default %d", ID);
			mt->setName(buffer);
			currentGroup.top()->addChild(mt);
			osg::Node *inlineNode = NULL;
			auto it = inlineNodes.find(url);
			if (it != inlineNodes.end())
			{
				inlineNode = it->second;
			}
			else
			{
				osg::Group *g=new osg::Group();
				inlineNode = coVRFileManager::instance()->loadFile(url.c_str(),NULL,g);
				if (inlineNode)
				{
					inlineNodes[url] = inlineNode;
				}
				g->removeChild(inlineNode);
			}
			if (inlineNode)
			{
				mt->addChild(inlineNode);
			}

			ei->nodes.push_back(mt);
			RevitInfo *info = new RevitInfo();
			info->ObjectID = ID;
			OSGVruiUserDataCollection::setUserData(mt, "RevitInfo", info);
		}

	}
	break;
    case MSG_File:
    {
        TokenBuffer tb(msg);
        const char *buf;
        int numBytes;
        std::string fileName;
        int MatID;
        tb >> MatID;
        tb >> fileName;
        localTextureFile = localTextureDir + "/" + fileName;
        tb >> numBytes;
        buf = tb.getBinary(numBytes);
        if (numBytes > 0)
        {
#ifdef _WIN32
            int fd = open(localTextureFile.c_str(), O_RDWR | O_CREAT | O_BINARY, 0777);
#else
            int fd = open(localTextureFile.c_str(), O_RDWR | O_CREAT, 0777);
#endif
            if (fd != -1)
            {
                if (write(fd, buf, numBytes) != numBytes)
                {
                    osg::notify(osg::ALWAYS) << "remoteFetch: " << localTextureFile << " write error" << std::endl;
                }
                close(fd);
            }
            else
            {

                osg::notify(osg::ALWAYS) << "remoteFetch: " << localTextureFile << " can't open File" << std::endl;;
            }

            osg::Image * image = osgDB::readImageFile(localTextureFile);
            if (image == NULL)
            {
                
                    osg::notify(osg::ALWAYS) << "Can't open image file" << fileName << " could not get it from remote" << std::endl;
            }

        }

        break;
    }
	/*   case MSG_NewPolyMesh:
	   {
		   cerr << "not used anymore" << endl;
		   TokenBuffer tb(m);

		   int numPoints;
		   int numTriangles;
		   int numNormals;
		   int numUVs;
		   tb >> numPoints;
		   tb >> numTriangles;
		   tb >> numNormals;
		   tb >> numUVs;
		   osg::Geode *geode = new osg::Geode();
		   //geode->setName(name);
		   osg::Geometry *geom = new osg::Geometry();
		   geom->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
		   geom->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());
		   geode->addDrawable(geom);

		   // set up geometry

		   osg::Vec3Array *points = new osg::Vec3Array;
		   points->resize(numPoints);
		   for (int i = 0; i < numPoints; i++)
		   {
			   tb >> (*points)[i][0];
			   tb >> (*points)[i][1];
			   tb >> (*points)[i][2];
		   }
		   osg::Vec3Array *norms = new osg::Vec3Array;
		   norms->resize(numNormals);
		   for (int i = 0; i < numNormals; i++)
		   {
			   tb >> (*norms)[i][0];
			   tb >> (*norms)[i][1];
			   tb >> (*norms)[i][2];
		   }
		   osg::Vec2Array *UVs = new osg::Vec2Array;
		   UVs->resize(numUVs);
		   for (int i = 0; i < numUVs; i++)
		   {
			   tb >> (*UVs)[i][0];
			   tb >> (*UVs)[i][1];
		   }

		   osg::Vec3Array *vert = new osg::Vec3Array;
		   vert->reserve(numTriangles*3);

		   osg::Vec3Array *normals = NULL;
		   if(numNormals == numPoints)
		   {
			   normals = new osg::Vec3Array;
			   normals->reserve(numTriangles*3);
		   }
		   if(numNormals == 1)
		   {
			   normals = new osg::Vec3Array;
			   normals->push_back((*norms)[0]);
			   normals->push_back((*norms)[0]);
			   normals->push_back((*norms)[0]);
		   }
		   osg::Vec2Array *texcoords = NULL;
		   if(numUVs == numPoints)
		   {
			   texcoords = new osg::Vec2Array;
			   texcoords->reserve(numTriangles*3);
		   }
		   osg::DrawArrays *triangles = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, numTriangles * 3);
		   for (int i = 0; i < numTriangles; i++)
		   {
			   int v1,v2,v3;
			   tb >> v1;
			   tb >> v2;
			   tb >> v3;
			   vert->push_back((*points)[v1]);
			   vert->push_back((*points)[v2]);
			   vert->push_back((*points)[v3]);
			   if(numUVs == numPoints)
			   {
				   texcoords->push_back((*UVs)[v1]);
				   texcoords->push_back((*UVs)[v2]);
				   texcoords->push_back((*UVs)[v3]);
			   }
			   if(numNormals == numPoints)
			   {
				   normals->push_back((*norms)[v1]);
				   normals->push_back((*norms)[v2]);
				   normals->push_back((*norms)[v3]);
			   }
		   }

		   unsigned char r, g, b, a;
		   int MaterialID;
		   tb >> r;
		   tb >> g;
		   tb >> b;
		   tb >> a;
		   tb >> MaterialID;

		   osg::StateSet *geoState;
		   MaterialInfo *mi = getMaterial(MaterialID);
		   if (mi)
		   {
			   geoState = mi->geoState;
			   geode->setStateSet(geoState);
		   }
		   else
		   {
			   geoState = geode->getOrCreateStateSet();
			   setDefaultMaterial(geoState);
			   geoState->setMode(GL_LIGHTING, osg::StateAttribute::ON);
			   geode->setStateSet(geoState);
			   osg::Material *localmtl = new osg::Material;
			   localmtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
			   localmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f));
			   localmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f));
			   localmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
			   localmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
			   localmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

			   geoState->setAttributeAndModes(localmtl, osg::StateAttribute::ON);
		   }

		   if (a < 250)
		   {
			   geoState->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
			   geoState->setMode(GL_BLEND, osg::StateAttribute::ON);
			   geoState->setNestRenderBins(false);
		   }
		   else
		   {
			   geoState->setRenderingHint(osg::StateSet::OPAQUE_BIN);
			   geoState->setMode(GL_BLEND, osg::StateAttribute::OFF);
			   geoState->setNestRenderBins(false);
		   }


		   geom->setVertexArray(vert);
		   if(numNormals == numPoints)
		   {
			   geom->setNormalArray(normals);
			   geom->getNormalArray()->setBinding(osg::Array::BIND_PER_VERTEX);
		   }
		   else if ( numNormals == 1)
		   {
			   geom->setNormalArray(normals);
			   geom->getNormalArray()->setBinding(osg::Array::BIND_OVERALL);
		   }
		   if(texcoords !=NULL)
		   {
			   geom->setTexCoordArray(0,texcoords);
		   }
		   geom->addPrimitiveSet(triangles);
		   //GenNormalsVisitor *sv = new GenNormalsVisitor(45.0);
		   //sv->apply(*geode);
		   //ei->nodes.push_back(geode);
		   currentGroup.top()->addChild(geode);

	   }
	   break;*/
	default:
		switch (m->type)
		{
		case Message::SOCKET_CLOSED:
		case Message::CLOSE_SOCKET:

            cover->unwatchFileDescriptor(toRevit->getSocket()->get_id());
			delete toRevit;
			toRevit = NULL;

			cerr << "connection to Revit closed" << endl;
			break;
		default:
			cerr << "Unknown message [" << m->type << "] "  << endl;
			break;
		}
	}
}

osg::Image *RevitPlugin::readImage(std::string fileName)
{

    std::size_t found = fileName.find_first_of("|", 0);
    if (found != std::string::npos)
    {
        fileName = fileName.substr(0, found);
    }
    std::replace( fileName.begin(), fileName.end(), '\\', '/');

    localTextureFile = localTextureDir + "/" + fileName;
    found = fileName.find_last_of('\\');
    std::string fn;
    if (found != std::string::npos)
    {
        fn = fileName.substr(found+1);
    }
    else
    {
        found = fileName.find_last_of('/');
        if (found != std::string::npos)
        {
            fn = fileName.substr(found+1);
        }
    }
    std::string localTextureFileOnly = textureDir + "/" + fn;
    std::string texFile = textureDir + "/" + fileName;

    osg::Image *diffuseImage = osgDB::readImageFile(texFile);
    if (diffuseImage==NULL)
    {
        diffuseImage = osgDB::readImageFile(localTextureFile);
        if (diffuseImage == NULL)
        {
            diffuseImage = osgDB::readImageFile(fileName);
            if (diffuseImage == NULL)
            {
                diffuseImage = osgDB::readImageFile(localTextureFileOnly);
                if (diffuseImage == NULL)
                {
			cerr << "did not find it under any of its names, even not " << localTextureFileOnly<< endl;
                    return NULL;
                }
            }
        }
    }
    return diffuseImage;
}

bool
RevitPlugin::checkDoors()
{
    bool needUpdate = false;

	osg::Matrix headmat = cover->getViewerMat();
	headmat *= cover->getInvBaseMat();
	osg::Vec3 ViewerPosition = headmat.getTrans();
	for (std::list<DoorInfo *>::iterator it = doors.begin(); it != doors.end(); it++)
	{
		(*it)->checkStart(ViewerPosition);
	}
	if (activeDoors.size() > 0)
	{
        needUpdate = true;
	}
	for (std::list<DoorInfo *>::iterator it = activeDoors.begin(); it != activeDoors.end();)
	{
		if ((*it)->update(ViewerPosition))
		{
			it++;
		}
		else
		{
			it = activeDoors.erase(it);
		}
	}

    return needUpdate;
}

bool
RevitPlugin::update()
{
    return checkDoors();
}

void
RevitPlugin::preFrame()
{
	if (serverConn && serverConn->is_connected() && serverConn->check_for_input()) // we have a server and received a connect
	{
		//   std::cout << "Trying serverConn..." << std::endl;
		toRevit = serverConn->spawn_connection();
		if (toRevit && toRevit->is_connected())
		{
			fprintf(stderr, "Connected to Revit\n");
            cover->watchFileDescriptor(toRevit->getSocket()->get_id());
		}
	}
	char gotMsg = '\0';
	if (coVRMSController::instance()->isMaster())
	{
		if (toRevit)
		{
			static double lastTime = 0;
			if (cover->frameTime() > lastTime + 4)
			{
				TokenBuffer stb;

				osg::Matrix mat = cover->getXformMat();
				osg::Matrix viewerTrans;
				viewerTrans.makeTranslate(cover->getViewerMat().getTrans());
				osg::Matrix itransMat;
				itransMat.invert(viewerTrans);
				mat.postMult(itransMat);


				osg::Matrix scMat;
				osg::Matrix iscMat;
				float scaleFactor = cover->getScale();
				scMat.makeScale(scaleFactor, scaleFactor, scaleFactor);
				iscMat.makeScale(1.0 / scaleFactor, 1.0 / scaleFactor, 1.0 / scaleFactor);
				mat.postMult(iscMat);
				mat.preMult(scMat);

				osg::Matrix irotMat = mat;
				irotMat.setTrans(0, 0, 0);

				osg::Matrix rotMat;
				rotMat.invert(irotMat);
				mat.postMult(rotMat);
				osg::Vec3 eyePos = mat.getTrans();
				static osg::Vec3 oldEyePos(0, 0, 0);
				if ((eyePos - oldEyePos).length() > 0.3) // if distance to old pos > 30cm
				{
					oldEyePos = eyePos;
					lastTime = cover->frameTime();

					double eyePosition[3];
					double viewDirection[3];
					eyePosition[0] = -eyePos[0] / REVIT_FEET_TO_M;
					eyePosition[1] = -eyePos[1] / REVIT_FEET_TO_M;
					eyePosition[2] = -eyePos[2] / REVIT_FEET_TO_M;

					viewDirection[0] = rotMat(1, 0);
					viewDirection[1] = rotMat(1, 1);
					viewDirection[2] = rotMat(1, 2);

					stb << (double)eyePosition[0];
					stb << (double)eyePosition[1];
					stb << (double)eyePosition[2];
					stb << (double)viewDirection[0];
					stb << (double)viewDirection[1];
					stb << (double)viewDirection[2];

					Message message(stb);
					message.type = (int)RevitPlugin::MSG_AvatarPosition;
					RevitPlugin::instance()->sendMessage(message);
				}
			}
		}
		while (toRevit && toRevit->check_for_input())
		{
			toRevit->recv_msg(msg);
			if (msg)
			{
				gotMsg = '\1';
				coVRMSController::instance()->sendSlaves(&gotMsg, sizeof(char));
				coVRMSController::instance()->sendSlaves(msg);

				cover->sendMessage(this, coVRPluginSupport::TO_SAME_OTHERS, PluginMessageTypes::HLRS_Revit_Message + msg->type - MSG_NewObject, msg->data.length(), msg->data.data());
				handleMessage(msg);
			}
			else
			{
				gotMsg = '\0';
				cerr << "could not read message" << endl;
				break;
			}
		}
		gotMsg = '\0';
		coVRMSController::instance()->sendSlaves(&gotMsg, sizeof(char));
	}
	else
	{
		do
		{
			coVRMSController::instance()->readMaster(&gotMsg, sizeof(char));
			if (gotMsg != '\0')
			{
				coVRMSController::instance()->readMaster(msg);
				handleMessage(msg);
			}
		} while (gotMsg != '\0');
	}
}

int RevitPlugin::getRevitAnnotationID(int ai)
{
	if (ai >= annotationIDs.size())
		return -1; // never seen this Annotation (-2 == has already been created but revitID has not yet been received)
	else return annotationIDs[ai];
}
int RevitPlugin::getAnnotationID(int revitID)
{
	for (int i = 0; i < annotationIDs.size(); i++)
	{
		if (annotationIDs[i] == revitID)
			return i;
	}
	int newID = annotationIDs.size();
	annotationIDs.push_back(revitID);
	return newID;
}


void RevitPlugin::createNewAnnotation(int id, AnnotationMessage *am)
{
	while (annotationIDs.size() <= id)
		annotationIDs.push_back(-1);
	annotationIDs[id] = -2;
	osg::Matrix trans;
	osg::Matrix ori;
	for (unsigned y = 0; y < 4; ++y)
	{
		for (unsigned x = 0; x < 4; ++x)
		{
			ori(x, y) = am->_orientation[y * 4 + x];
			trans(x, y) = am->_translation[y * 4 + x];
		}
	}
	coCoord orientation(ori);
	TokenBuffer stb;
	stb << id;
	stb << (double)trans.getTrans()[0] / scaleFactor;
	stb << (double)trans.getTrans()[1] / scaleFactor;
	stb << (double)trans.getTrans()[2] / scaleFactor;
	stb << (double)orientation.hpr[0];
	stb << (double)orientation.hpr[1];
	stb << (double)orientation.hpr[2];
	char tmpText[100];
	sprintf(tmpText, "Annotation %d", id);
	stb << tmpText;

	Message message(stb);
	message.type = (int)RevitPlugin::MSG_NewAnnotation;
	RevitPlugin::instance()->sendMessage(message);
}
void RevitPlugin::changeAnnotation(int id, AnnotationMessage *am)
{
	osg::Matrix trans;
	osg::Matrix ori;
	for (unsigned y = 0; y < 4; ++y)
	{
		for (unsigned x = 0; x < 4; ++x)
		{
			ori(x, y) = am->_orientation[y * 4 + x];
			trans(x, y) = am->_translation[y * 4 + x];
		}
	}
	coCoord orientation(ori);
	TokenBuffer stb;
	stb << id;
	stb << (double)trans.getTrans()[0] / scaleFactor;
	stb << (double)trans.getTrans()[1] / scaleFactor;
	stb << (double)trans.getTrans()[2] / scaleFactor;
	stb << (double)orientation.hpr[0];
	stb << (double)orientation.hpr[1];
	stb << (double)orientation.hpr[2];

	Message message(stb);
	message.type = (int)RevitPlugin::MSG_ChangeAnnotation;
	RevitPlugin::instance()->sendMessage(message);
}
MaterialInfo * RevitPlugin::getMaterial(int revitID)
{
	std::map<int, MaterialInfo *>::iterator it = MaterialInfos.find(revitID);
	if (it != MaterialInfos.end())
	{
		return(it->second);
	}
	return NULL;


}



// pretend types, something like this
struct pixel
{
	uint8_t red;
	uint8_t green;
	uint8_t blue;
	uint8_t alpha;
};


// determine intensity of pixel, from 0 - 1
const double intensity(const pixel& pPixel)
{
	const double r = static_cast<double>(pPixel.red);
	const double g = static_cast<double>(pPixel.green);
	const double b = static_cast<double>(pPixel.blue);

	const double average = (r + g + b) / 3.0;

	return average / 255.0;
}

const int repeat(int pX, int pMax)
{
	if (pX > pMax)
	{
		return 0;
	}
	else if (pX < 0)
	{
		return pMax;
	}
	else
	{
		return pX;
	}
}

// transform -1 - 1 to 0 - 255
const uint8_t map_component(double pX)
{
	return (pX + 1.0) * (255.0 / 2.0);
}

void RevitPlugin::requestTexture(int matID, TextureInfo * texture)
{
    std::string filePathName = texture->texturePath;
    std::size_t found = filePathName.find_first_of("|", 0);
    if (found != std::string::npos)
    {
        filePathName = filePathName.substr(0, found);
    }
    found = filePathName.find_last_of('\\');
    std::string fileName;
    if (found != std::string::npos)
    {
        fileName = filePathName.substr(found+1);
    }
    else
    {
        found = filePathName.find_last_of('/');
        if (found != std::string::npos)
        {
            fileName = filePathName.substr(found+1);
        }
    }

    char gotMsg = '\1';
    if (coVRMSController::instance()->isMaster())
    {
        TokenBuffer stb;
        stb << matID;
        stb << filePathName;
        stb << fileName;
        Message message(stb);
        message.type = (int)RevitPlugin::MSG_File;
        if (RevitPlugin::instance()->sendMessage(message) == false)
        {
            gotMsg = '\0';
			coVRMSController::instance()->sendSlaves(&gotMsg, sizeof(char));
        }
        else
        {
            while (toRevit)
            {
                toRevit->recv_msg(msg);
                if (msg)
                {
                    gotMsg = '\1';
                    coVRMSController::instance()->sendSlaves(&gotMsg, sizeof(char));
                    coVRMSController::instance()->sendSlaves(msg);
                    cover->sendMessage(this, coVRPluginSupport::TO_SAME_OTHERS, PluginMessageTypes::HLRS_Revit_Message + msg->type - MSG_NewObject, msg->data.length(), msg->data.data());
                    handleMessage(msg);
                    if (msg->type == MSG_File)
                        break; // done
                }
                else
                {
                    gotMsg = '\0';
                    cerr << "could not read message" << endl;
                    break;
                }
            }
            gotMsg = '\0';
            coVRMSController::instance()->sendSlaves(&gotMsg, sizeof(char));
        }
    }
    else
    {
        do
        {
            coVRMSController::instance()->readMaster(&gotMsg, sizeof(char));
            if (gotMsg != '\0')
            {
                coVRMSController::instance()->readMaster(msg);
                handleMessage(msg);
            }
        } while (gotMsg != '\0');
    }

}

osg::Image *MaterialInfo::createNormalMap(osg::Image *srcImage, double pStrength)
{
	// assume square texture, not necessarily true in real code
	osg::Image *result = new osg::Image();
	int width = srcImage->s();
	int height = srcImage->t();
	result->allocateImage(width, height, 1, GL_RGB, srcImage->getDataType());
	if (srcImage->getPixelFormat() == GL_LUMINANCE || srcImage->getPixelFormat() == GL_INTENSITY || srcImage->getPixelFormat() == GL_ALPHA)
	{
		for (size_t row = 0; row < height; ++row)
		{
			for (size_t column = 0; column < width; ++column)
			{
				// surrounding pixels
				const pixel *topLeft = (const pixel*)srcImage->data(repeat(column - 1, height), repeat(row - 1, width));
				const pixel *top = (const pixel*)srcImage->data(repeat(column, height), repeat(row - 1, width));
				const pixel *topRight = (const pixel*)srcImage->data(repeat(column + 1, height), repeat(row - 1, width));
				const pixel *right = (const pixel*)srcImage->data(repeat(column + 1, height), repeat(row, width));
				const pixel *bottomRight = (const pixel*)srcImage->data(repeat(column + 1, height), repeat(row + 1, width));
				const pixel *bottom = (const pixel*)srcImage->data(repeat(column, height), repeat(row + 1, width));
				const pixel *bottomLeft = (const pixel*)srcImage->data(repeat(column - 1, height), repeat(row + 1, width));
				const pixel *left = (const pixel*)srcImage->data(repeat(column - 1, height), repeat(row, width));

				// their intensities
				const double tl = topLeft->red / 255.0;
				const double t = top->red / 255.0;
				const double tr = topRight->red / 255.0;
				const double r = right->red / 255.0;
				const double br = bottomRight->red / 255.0;
				const double b = bottom->red / 255.0;
				const double bl = bottomLeft->red / 255.0;
				const double l = left->red / 255.0;

				// sobel filter
				const double dX = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
				const double dY = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
				const double dZ = 1.0 / pStrength;

				osg::Vec3d v(dX, dY, dZ);
				v.normalize();

				// convert to rgb
				unsigned char* dst_pixel = result->data(column, row, 0);
				*(dst_pixel++) = map_component(v.x());
				*(dst_pixel++) = map_component(v.y());
				*(dst_pixel++) = map_component(v.z());
			}
		}
	}
	else
	{
		for (size_t row = 0; row < height; ++row)
		{
			for (size_t column = 0; column < width; ++column)
			{
				// surrounding pixels
				const pixel *topLeft = (const pixel*)srcImage->data(repeat(column - 1, height), repeat(row - 1, width));
				const pixel *top = (const pixel*)srcImage->data(repeat(column, height), repeat(row - 1, width));
				const pixel *topRight = (const pixel*)srcImage->data(repeat(column + 1, height), repeat(row - 1, width));
				const pixel *right = (const pixel*)srcImage->data(repeat(column + 1, height), repeat(row, width));
				const pixel *bottomRight = (const pixel*)srcImage->data(repeat(column + 1, height), repeat(row + 1, width));
				const pixel *bottom = (const pixel*)srcImage->data(repeat(column, height), repeat(row + 1, width));
				const pixel *bottomLeft = (const pixel*)srcImage->data(repeat(column - 1, height), repeat(row + 1, width));
				const pixel *left = (const pixel*)srcImage->data(repeat(column - 1, height), repeat(row, width));

				// their intensities
				const double tl = intensity(*topLeft);
				const double t = intensity(*top);
				const double tr = intensity(*topRight);
				const double r = intensity(*right);
				const double br = intensity(*bottomRight);
				const double b = intensity(*bottom);
				const double bl = intensity(*bottomLeft);
				const double l = intensity(*left);

				// sobel filter
				const double dX = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
				const double dY = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
				const double dZ = 1.0 / pStrength;

				osg::Vec3d v(dX, dY, dZ);
				v.normalize();

				// convert to rgb
				unsigned char* dst_pixel = result->data(column, row, 0);
				*(dst_pixel++) = map_component(v.x());
				*(dst_pixel++) = map_component(v.y());
				*(dst_pixel++) = map_component(v.z());
			}
		}
	}

	return result;
}



COVERPLUGIN(RevitPlugin)

TextureInfo::TextureInfo(TokenBuffer & tb)
{
	tb >> texturePath;
	tb >> sx;
	tb >> sy;
	tb >> ox;
	tb >> oy;
	tb >> angle;
	tb >> r;
	tb >> g;
	tb >> b;
	amount = 1.0;
	requestTexture=false;
}

MaterialInfo::MaterialInfo(TokenBuffer & tb)
{
	tb >> ID;
	diffuseTexture = new TextureInfo(tb);
    diffuseTexture->type = TextureInfo::diffuse;
	bumpTexture = new TextureInfo(tb);
    bumpTexture->type = TextureInfo::bump;
	tb >> bumpTexture->amount;
	r = diffuseTexture->r;
	g = diffuseTexture->g;
	b = diffuseTexture->b;
	a = 255;
	shader = NULL;
}

void MaterialInfo::updateTexture(TextureInfo::textureType type, osg::Image * image)
{
    int textureUnit=0;
    if (type == TextureInfo::bump && diffuseTexture->image != NULL)
        textureUnit = 1;

    osg::Texture2D *texture = new osg::Texture2D(image);
    texture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
    texture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
    texture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::REPEAT);
    texture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::REPEAT);
    geoState->setTextureAttribute(textureUnit, texture);
    textureUnit++;

    if (type == TextureInfo::diffuse)
    {
        osg::Uniform *revitSX = new osg::Uniform("revitSX", (float)diffuseTexture->sx);
        osg::Uniform *revitSY = new osg::Uniform("revitSY", (float)diffuseTexture->sy);
        osg::Uniform *revitOX = new osg::Uniform("revitOX", (float)diffuseTexture->ox);
        osg::Uniform *revitOY = new osg::Uniform("revitOY", (float)diffuseTexture->oy);
        osg::Uniform *revitAngle = new osg::Uniform("revitAngle", (float)diffuseTexture->angle);
        geoState->addUniform(revitSX);
        geoState->addUniform(revitSY);
        geoState->addUniform(revitOX);
        geoState->addUniform(revitOY);
        geoState->addUniform(revitAngle);
        if(bumpTexture->image == NULL)
            shader = coVRShaderList::instance()->get("RevitDiffuse");
        diffuseTexture->image = image;
    }
    else if (type == TextureInfo::bump)
    {


        osg::Image *ni = createNormalMap(image, bumpTexture->amount);
        osg::Texture2D *normalTexture = new osg::Texture2D(ni);
        normalTexture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
        normalTexture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
        normalTexture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::REPEAT);
        normalTexture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::REPEAT);
        geoState->setTextureAttribute(textureUnit, normalTexture);
        textureUnit++;

        osg::Uniform *revitSX = new osg::Uniform("revitSX", (float)bumpTexture->sx);
        osg::Uniform *revitSY = new osg::Uniform("revitSY", (float)bumpTexture->sy);
        osg::Uniform *revitOX = new osg::Uniform("revitOX", (float)bumpTexture->ox);
        osg::Uniform *revitOY = new osg::Uniform("revitOY", (float)bumpTexture->oy);
        osg::Uniform *revitAngle = new osg::Uniform("revitAngle", (float)bumpTexture->angle);
        geoState->addUniform(revitSX);
        geoState->addUniform(revitSY);
        geoState->addUniform(revitOX);
        geoState->addUniform(revitOY);
        geoState->addUniform(revitAngle);
        if (diffuseTexture->image != NULL)
        {
            shader = coVRShaderList::instance()->get("RevitDiffuseBump");
        }
        else
        {
            shader = coVRShaderList::instance()->get("RevitBumpOnly");
        }
        bumpTexture->image = image;
    }
}


DoorInfo::DoorInfo(int id, const char *Name, osg::MatrixTransform *tn, TokenBuffer &tb)
{
	ID = id;
	name = Name;
	transformNode = tn;
	unsigned char boolValue;
	tb >> boolValue;
	HandFlipped = boolValue;
	tb >> HandOrientation;
	tb >> boolValue;
	FaceFlipped = boolValue;
	tb >> FaceOrientation;
	tb >> Origin;
	tb >> boolValue;
	isSliding = boolValue;
	osg::Vec3 BBMin;
	osg::Vec3 BBMax;
	tb >> BBMin;
	tb >> BBMax;
	boundingBox.set(BBMin, BBMax);
	if (isSliding)
	{
		maxDistance = boundingBox.xMax() - boundingBox.xMin();
		if (maxDistance == 0)
			maxDistance = 1;
		HandOrientation.normalize(); // HandOrientation is in Revit World coordinates so either transform this back to local coordinates orjust use X for now.
		Direction = osg::Vec3(-1, 0, 0)*maxDistance;
	//	if (!HandFlipped)
	//	{
	//		Direction *= -1;
	//	}
	}
	else
	{
		Direction = HandOrientation ^ FaceOrientation;
		Direction.normalize();
		maxDistance = M_PI_2;
		fprintf(stderr, "HandFlipped %d\n", (int)HandFlipped);
		fprintf(stderr, "FaceFlipped %d\n", (int)FaceFlipped);

		Direction *= -1;
		if(Origin.x() == 10000)
		{
			Origin = boundingBox._max;
			if (strncmp(Name, "DoorMovingParts_Right", 21) == 0)
			{
				Origin = boundingBox._min;
			}
		}
		if (strncmp(Name, "DoorMovingParts_Right", 21) == 0)
		{
			Direction *= -1;
		}
	}

	Center = boundingBox.center();
	// transform Center to Object Coordinates
	Matrix tr;
	tr.makeIdentity();
	//cerr << "LocalToVRML: hitPoint: "<<hitPoint[0]<<' '<<hitPoint[1]<<' '<<hitPoint[2]<<endl;
	const Node *parent = transformNode->getParent(0);
	while (parent != NULL && parent != cover->getObjectsRoot())
	{
		Matrix dcsMat;
		const MatrixTransform *mtParent = dynamic_cast<const MatrixTransform *>(parent);
		if (mtParent)
		{
			dcsMat = mtParent->getMatrix();
			tr.postMult(dcsMat);
		}
		if (parent->getNumParents())
			parent = parent->getParent(0);
		else
			parent = NULL;
	}
	Center = Center * tr;
	activationDistance2 = 3; // three m
	activationDistance2 *= activationDistance2; // squared
	animationTime = 1.0; // one second
	isActive = false;
	left = false;
	entered = false;
}

void DoorInfo::checkStart(osg::Vec3 &viewerPosition)
{
	if (!isActive)
	{
		if ((Center - viewerPosition).length2() < activationDistance2)
		{
			isActive = true;
			RevitPlugin::instance()->activeDoors.push_back(this);
			entered = true;
			left = false;
			startTime = cover->frameTime();
		}
	}
}

void DoorInfo::translateDoor(float fraction)
{
	if (isSliding)
	{
		transformNode->setMatrix(osg::Matrix::translate(Direction*maxDistance*fraction));
	}
	else
	{
		transformNode->setMatrix(osg::Matrix::translate(-Origin) * osg::Matrix::rotate(maxDistance*fraction, Direction)*osg::Matrix::translate(Origin));
	}
}

bool DoorInfo::update(osg::Vec3 &viewerPosition)
{
	if (!left && ((Center - viewerPosition).length2() > activationDistance2))
	{
		left = true;
		startTime = cover->frameTime();
	}
	if (left)
	{
		if (cover->frameTime() < startTime + animationTime)
		{
			float fraction = 1.0 - ((cover->frameTime() - startTime) / animationTime);
			translateDoor(fraction);
		}
		else
		{
			translateDoor(0.0);
			left = false;
			isActive = false;
			return false;
		}
	}
	if (entered)
	{
		if (cover->frameTime() < startTime + animationTime)
		{
			float fraction = (cover->frameTime() - startTime) / animationTime;
			translateDoor(fraction);
		}
		else
		{
			translateDoor(1.0);
			entered = false;
		}
	}
	return true;
}

