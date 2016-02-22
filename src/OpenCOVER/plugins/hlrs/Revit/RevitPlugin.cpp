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
#include <cover/RenderObject.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>
#include <cover/coVRSelectionManager.h>
#include "cover/coVRTui.h"
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
#include "GenNormals.h"

#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <net/tokenbuffer.h>
#include <config/CoviseConfig.h>

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

RevitViewpointEntry::RevitViewpointEntry(osg::Vec3 pos, osg::Vec3 dir, osg::Vec3 up, RevitPlugin *plugin, std::string n, int id,coCheckboxMenuItem *me)
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
    mat.makeTranslate(-eyePosition[0] * 0.3048, -eyePosition[1] * 0.3048, -eyePosition[2] * 0.3048);
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
    float scaleFactor = 304.8;
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
    irotMat.setTrans(0,0,0);
    
    osg::Matrix rotMat;
    rotMat.invert(irotMat);
    mat.postMult(rotMat);
    osg::Vec3 eyePos = mat.getTrans();
    eyePosition[0] = -eyePos[0]/0.3048;
    eyePosition[1] = -eyePos[1]/0.3048;
    eyePosition[2] = -eyePos[2]/0.3048;
    
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
    delete revitTab;
}

RevitPlugin::RevitPlugin()
{
    fprintf(stderr, "RevitPlugin::RevitPlugin\n");
    plugin = this;
    MoveFinished = true;
    int port = coCoviseConfig::getInt("port", "COVER.Plugin.Revit.Server", 31821);
    toRevit = NULL;
    serverConn = new ServerConnection(port, 1234, Message::UNDEFINED);
    if (!serverConn->getSocket())
    {
        cout << "tried to open server Port " << port << endl;
        cout << "Creation of server failed!" << endl;
        cout << "Port-Binding failed! Port already bound?" << endl;
        delete serverConn;
        serverConn = NULL;
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
    globalmtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    globalmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0));
    globalmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    globalmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    globalmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
    globalmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    revitGroup = new osg::MatrixTransform();
    revitGroup->setName("RevitGeometry");
    scaleFactor = 0.3048;
    revitGroup->setMatrix(osg::Matrix::scale(scaleFactor, scaleFactor, scaleFactor));
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

    revitGroup->removeChild(0, revitGroup->getNumChildren());
    cover->getObjectsRoot()->removeChild(revitGroup.get());

    delete serverConn;
    serverConn = NULL;
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
            if((*it)->isActive)
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
            if((*it)->isActive)
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

void RevitPlugin::sendMessage(Message &m)
{
    if(toRevit) // false on slaves
    {
        toRevit->send_msg(&m);
    }
}


void RevitPlugin::message(int type, int len, const void *buf)
{
    if (type == PluginMessageTypes::MoveAddMoveNode)
    {
    }
    else if (type == PluginMessageTypes::MoveMoveNodeFinished)
    {
        MoveFinished=true;
	std::string path;
        TokenBuffer tb((const char *)buf, len);
        tb >> path;
        tb >> path;
                TokenBuffer stb;
                stb << MovedID;
                stb << -(double)lastMoveMat.getTrans().x();
                stb << (double)lastMoveMat.getTrans().y();
                stb << (double)lastMoveMat.getTrans().z();

                Message message(stb);
                message.type = (int)RevitPlugin::MSG_SetTransform;
                RevitPlugin::instance()->sendMessage(message);
    }
    else if (type == PluginMessageTypes::AnnotationMessage) // An AnnotationMessage has been received
    {
        AnnotationMessage *mm = (AnnotationMessage *)buf;

        switch (mm->token)
        {
        case ANNOTATION_MESSAGE_TOKEN_MOVEADD: // MOVE/ADD
            {

                int revitID = getRevitAnnotationID(mm->id);
                if(revitID > 0)
                {
                    changeAnnotation(revitID,mm);
                }
                else if(revitID == -1)
                {
                    createNewAnnotation(mm->id,mm);
                }
                break;
            } // case moveadd

        case ANNOTATION_MESSAGE_TOKEN_REMOVE: // Remove an annotation
        {
            int revitID = getRevitAnnotationID(mm->id);
            if(revitID > 0)
            {
                TokenBuffer stb;
                stb << revitID;

                Message message(stb);
                message.type = (int)RevitPlugin::MSG_DeleteObject;
                RevitPlugin::instance()->sendMessage(message);
            }
            else if(revitID == -1)
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
        int id,owner;
        char *ctext;
        tb >> id;
        tb >> owner;
        tb >> ctext;
        text = ctext;
        int AnnotationID = getRevitAnnotationID(id);
        if(AnnotationID > 0)
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
        if(selectedNode)
        {
            info = dynamic_cast<RevitInfo *>(OSGVruiUserDataCollection::getUserData(selectedNode, "RevitInfo"));
            if(info)
            {
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        tb >> m(i, j);
                if(MovedID!=info->ObjectID)
                {
                    MoveFinished=true;
                }
                
                lastMoveMat = invStartMoveMat*m;
                //invStartMoveMat.invert(m);
		
                if(MoveFinished)
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
    else if(type >= PluginMessageTypes::HLRS_Revit_Message && type <= (PluginMessageTypes::HLRS_Revit_Message+100))
    {
        Message m;
        m.type = type-PluginMessageTypes::HLRS_Revit_Message + MSG_NewObject;
        m.length = len;
        m.data = (char *)buf;
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
            sprintf(info,"Nr.: %s\n%s\nArea: %3.7lfm^2\nLevel: %s",roomNumber,roomName,area/10.0,levelName);
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
    }
    break;
    case MSG_NewAnnotation:
    {
        TokenBuffer tb(m);
        int ID;
        float x,y,z;
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
        trans.makeTranslate(x*scaleFactor,y*scaleFactor,z*scaleFactor); // get rid of scale part
        orientation.makeRotate(3.0,-1,0,0);
        matrix2array(trans, am.translation());
        matrix2array(orientation, am.orientation());
        cover->sendMessage(this, "Annotation",
            PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
              
        TokenBuffer tb3;
        tb3 << AID;
        tb3 << 101; // owner
        tb3 << text;
        cover->sendMessage(this, "Annotation",
            PluginMessageTypes::AnnotationTextMessage, tb3.get_length(), tb3.get_data());
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

        bool foundIt=false;

        for (list<RevitViewpointEntry *>::iterator it = viewpointEntries.begin();
             it != viewpointEntries.end(); it++)
        {
            if ((*it)->ID == ID)
            {
                foundIt = true;
                RevitViewpointEntry *vpe = (*it);
                if(vpe->isActive)
                {
                    vpe->setValues(pos, dir, up,name);
                    vpe->activate();
                }
                break;
            }
        }
        if(!foundIt)
        {

        coCheckboxMenuItem *menuEntry;

        menuEntry = new coCheckboxMenuItem(name, false, cbg);
        // add viewpoint to menu
        RevitViewpointEntry *vpe = new RevitViewpointEntry(pos, dir, up, this, name,ID,menuEntry);
        menuEntry->setMenuListener(vpe);
        viewpointMenu->add(menuEntry);
        vpe->setMenuItem(menuEntry);
        viewpointEntries.push_back(vpe);
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
            while(annotationIDs.size() <= annotationID)
                annotationIDs.push_back(-1);
            annotationIDs[annotationID]=ID;
            // check if we have cached changes for this Annotation and send it to Revit.
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
            geom->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
            geom->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());
            geode->addDrawable(geom);
            osg::StateSet *geoState = geode->getOrCreateStateSet();
            setDefaultMaterial(geoState);
            geoState->setMode(GL_LIGHTING, osg::StateAttribute::ON);
            geode->setStateSet(geoState);

            // set up geometry
            bool isTwoSided = false;
            char tmpChar;
            tb >> tmpChar;
            if (tmpChar != '\0')
                isTwoSided = true;
            if (!isTwoSided)
            {
                osg::CullFace *cullFace = new osg::CullFace();
                cullFace->setMode(osg::CullFace::BACK);
                geoState->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
            }

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

            osg::Material *localmtl = new osg::Material;
            localmtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
            localmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f));
            localmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f));
            localmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
            localmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
            localmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

            geoState->setAttributeAndModes(localmtl, osg::StateAttribute::ON);

            geom->setVertexArray(vert);
            geom->addPrimitiveSet(triangles);
            GenNormalsVisitor *sv = new GenNormalsVisitor(45.0);
            sv->apply(*geode);
            ei->nodes.push_back(geode);
            
            RevitInfo *info = new RevitInfo();
            info->ObjectID = ID;
            OSGVruiUserDataCollection::setUserData(geode, "RevitInfo", info);
            currentGroup.top()->addChild(geode);
        }

    }
    break;
    
    case MSG_NewPolyMesh:
    {
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
        osg::StateSet *geoState = geode->getOrCreateStateSet();
        setDefaultMaterial(geoState);
        geoState->setMode(GL_LIGHTING, osg::StateAttribute::ON);
        geode->setStateSet(geoState);

        // set up geometry
      /*  bool isTwoSided = false;
        char tmpChar;
        tb >> tmpChar;
        if (tmpChar != '\0')
            isTwoSided = true;
        if (!isTwoSided)
        {
            osg::CullFace *cullFace = new osg::CullFace();
            cullFace->setMode(osg::CullFace::BACK);
            geoState->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
        }*/
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
        /*unsigned char r, g, b, a;
        int MaterialID;
        tb >> r;
        tb >> g;
        tb >> b;
        tb >> a;
        tb >> MaterialID;
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
        }*/

        osg::Material *localmtl = new osg::Material;
        localmtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        /*
        localmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f));
        localmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f));
        */
        localmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
        localmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
        localmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

        geoState->setAttributeAndModes(localmtl, osg::StateAttribute::ON);
        
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
        /*GenNormalsVisitor *sv = new GenNormalsVisitor(45.0);
        sv->apply(*geode);
        ei->nodes.push_back(geode);*/
        currentGroup.top()->addChild(geode);

    }
    break;
    default:
        switch (m->type)
        {
        case Message::SOCKET_CLOSED:
        case Message::CLOSE_SOCKET:
            delete toRevit;
            toRevit = NULL;

            cerr << "connection to Revit closed" << endl;
            break;
        default:
            cerr << "Unknown message [" << MSG_NewObject << "] " << m->type << endl;
            break;
        }
    }
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
        }
    }
    char gotMsg = '\0';
    if (coVRMSController::instance()->isMaster())
    {
        if(toRevit)
        {
            static double lastTime = 0;
            if(cover->frameTime() > lastTime+4)
            {
                lastTime = cover->frameTime();
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
                irotMat.setTrans(0,0,0);

                osg::Matrix rotMat;
                rotMat.invert(irotMat);
                mat.postMult(rotMat);
                osg::Vec3 eyePos = mat.getTrans();
                double eyePosition[3];
                double viewDirection[3];
                eyePosition[0] = -eyePos[0]/0.3048;
                eyePosition[1] = -eyePos[1]/0.3048;
                eyePosition[2] = -eyePos[2]/0.3048;

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
        while (toRevit && toRevit->check_for_input())
        {
            toRevit->recv_msg(msg);
            if (msg)
            {
                gotMsg = '\1';
                coVRMSController::instance()->sendSlaves(&gotMsg, sizeof(char));
                coVRMSController::instance()->sendSlaves(msg);
                
                cover->sendMessage(this, coVRPluginSupport::TO_SAME_OTHERS,PluginMessageTypes::HLRS_Revit_Message+msg->type-MSG_NewObject,msg->length, msg->data);
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
    if(ai >= annotationIDs.size())
        return -1; // never seen this Annotation (-2 == has already been created but revitID has not yet been received)
    else return annotationIDs[ai];
}
int RevitPlugin::getAnnotationID(int revitID)
{
    for(int i=0;i<annotationIDs.size();i++)
    {
        if(annotationIDs[i]== revitID)
            return i;
    }
    int newID = annotationIDs.size();
    annotationIDs.push_back(revitID);
    return newID;
}


void RevitPlugin::createNewAnnotation(int id, AnnotationMessage *am)
{
    while(annotationIDs.size() <= id)
        annotationIDs.push_back(-1);
    annotationIDs[id]=-2;
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
    stb << (double)trans.getTrans()[0]/scaleFactor;
    stb << (double)trans.getTrans()[1]/scaleFactor;
    stb << (double)trans.getTrans()[2]/scaleFactor;
    stb << (double)orientation.hpr[0];
    stb << (double)orientation.hpr[1];
    stb << (double)orientation.hpr[2];
    char tmpText[100];
    sprintf(tmpText,"Annotation %d",id);
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
    stb << (double)trans.getTrans()[0]/scaleFactor;
    stb << (double)trans.getTrans()[1]/scaleFactor;
    stb << (double)trans.getTrans()[2]/scaleFactor;
    stb << (double)orientation.hpr[0];
    stb << (double)orientation.hpr[1];
    stb << (double)orientation.hpr[2];

    Message message(stb);
    message.type = (int)RevitPlugin::MSG_ChangeAnnotation;
    RevitPlugin::instance()->sendMessage(message);
}


COVERPLUGIN(RevitPlugin)
