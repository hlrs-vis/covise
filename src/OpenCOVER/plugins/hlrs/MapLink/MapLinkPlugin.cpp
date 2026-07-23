/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MapLinkPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>
#include <cover/coVRSelectionManager.h>
#include "cover/coVRTui.h"
#include <cover/coVRRenderer.h>
#include <cover/VRViewer.h>
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
#include <osg/LineSegment>
#include <cover/coIntersection.h>


#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <net/tokenbuffer.h>
#include <config/CoviseConfig.h>

using covise::TokenBuffer;
using covise::coCoviseConfig;


void DrawCallback::operator()(const osg::Camera &cam) const
{

        plugin->sendImage();
        
}

void MapLinkPlugin::createMenu()
{

   /* cbg = new coCheckboxGroup();
    viewpointMenu = new coRowMenu("MapLink Viewpoints");

    REVITButton = new coSubMenuItem("MapLink");
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

    cover->getMenu()->add(REVITButton);*/

    MapLinkTab = new coTUITab("MapLink", coVRTui::instance()->mainFolder->getID());
    MapLinkTab->setPos(0, 0);

   /* updateCameraTUIButton = new coTUIButton("Update Camera", revitTab->getID());
    updateCameraTUIButton->setEventListener(this);
    updateCameraTUIButton->setPos(0, 0);

    addCameraTUIButton = new coTUIButton("Add Camera", revitTab->getID());
    addCameraTUIButton->setEventListener(this);
    addCameraTUIButton->setPos(0, 1);*/
}

void MapLinkPlugin::destroyMenu()
{
  /*  delete roomInfoButton;
    delete roomInfoMenu;
    delete label1;
    delete viewpointMenu;
    delete REVITButton;
    delete cbg;

    delete addCameraTUIButton;
    delete updateCameraTUIButton;*/
    delete MapLinkTab;
}


osg::Matrixd MapLinkPlugin::computeLeftEyeProjection(const osg::Matrixd &projection) const
{
	(void)projection;
	return projMat;
}

osg::Matrixd MapLinkPlugin::computeLeftEyeView(const osg::Matrixd &view) const
{
	(void)view;
	return viewMat;
}

osg::Matrixd MapLinkPlugin::computeRightEyeProjection(const osg::Matrixd &projection) const
{
	(void)projection;
	return projMat;
}

osg::Matrixd MapLinkPlugin::computeRightEyeView(const osg::Matrixd &view) const
{
	(void)view;
	return viewMat;
}

MapLinkPlugin::MapLinkPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "MapLinkPlugin::MapLinkPlugin\n");
    fprintf(stderr, "MEINE NEUE PLUGIN VERSION WIRD GELADEN\n");
    plugin = this;
	width = 0;
    int port = coCoviseConfig::getInt("port", "COVER.Plugin.MapLink.Server", 31822);
    toMapLink = NULL;
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
            cover->unwatchFileDescriptor(serverConn->getSocket()->get_id());
            delete serverConn;
            serverConn = NULL;

        }
    }
    msg = new Message;

}

void MapLinkPlugin::sendImage()
{
    if(width > 0)
    {
        TokenBuffer rtb;
        rtb << MSG_GetMap;
        rtb << x;
        rtb << y;
        rtb << width;
        rtb << height;
        rtb << xRes;
        rtb << yRes;
        rtb.addBinary((char *)image->getDataPointer(),xRes*yRes*4);
        Message m(rtb);
        m.type = PluginMessageTypes::HLRS_MapLink_Message;
        sendMessage(m);
    }
    width = 0;

}

bool MapLinkPlugin::init()
{
    //cover->addPlugin("Annotation"); // we would like to have the Annotation plugin
    createMenu();
    createCamera();
    return true;
}
// this is called if the plugin is removed at runtime
MapLinkPlugin::~MapLinkPlugin()
{
    destroyMenu();
    if (serverConn && serverConn->getSocket())
        cover->unwatchFileDescriptor(serverConn->getSocket()->get_id());
    delete serverConn;
    serverConn = NULL;
    if (toMapLink && toMapLink->getSocket())
        cover->unwatchFileDescriptor(toMapLink->getSocket()->get_id());
    delete msg;
    
    if (camera.get())
    {
        camera->detach(osg::Camera::COLOR_BUFFER);
        camera->setGraphicsContext(NULL);
        VRViewer::instance()->removeCamera(camera.get());
    }
    toMapLink = NULL;
}

void MapLinkPlugin::setProjection(float xPos, float yPos, float width, float height)
{
    float hw = width/2.0;
    float hh = height/2.0;
    // ProjectionMatrix //
    //
    projMat = osg::Matrix::ortho(-hw, hw, -hh, hh, 10000.0, 4000000.0);

    // ViewMatrix //
    //
    
    //osg::Matrix viewMat = cover->getInvBaseMat();
    //viewMat.postMult(osg::Matrix::lookAt(osg::Vec3d(xPos+hw, yPos+hh, 1800000.0), osg::Vec3d(xPos+hw, yPos+hh, -1000000.0), osg::Vec3d(0.0, 1.0, 0.0)));
    osg::Matrix tmpMat = osg::Matrix::lookAt(osg::Vec3d(xPos+hw, yPos+hh, 1800000.0), osg::Vec3d(xPos+hw, yPos+hh, -1000000.0), osg::Vec3d(0.0, 1.0, 0.0));
    viewMat = cover->getInvBaseMat() *osg::Matrix::translate(-(xPos+hw), -(yPos+hh), -1800000.0);


    camera->setProjectionMatrix(projMat);
    camera->setViewMatrix(viewMat);
    
    //VRViewer::instance()->addCamera(camera.get());

}
void MapLinkPlugin::createCamera()
{
    resX=1024;
    resY=768;
    
    drawCallback = new DrawCallback(this);

    image = new osg::Image();
    image.get()->allocateImage(resX,resY, 1, GL_RGBA, GL_UNSIGNED_BYTE);
    

    osg::Camera *cam = dynamic_cast<osg::Camera *>(coVRConfig::instance()->channels[0].camera.get());
    camera = new osg::Camera();

    camera->setViewport(0, 0, resX,resY);
    camera->setRenderOrder(osg::Camera::PRE_RENDER);
    camera->setRenderTargetImplementation((osg::Camera::RenderTargetImplementation)(osg::Camera::FRAME_BUFFER_OBJECT));
    camera->setClearColor(osg::Vec4(0, 0, 0, 0));
    camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    camera->setView(cam->getView());

    camera->setCullMask(~0 & ~(Isect::Collision|Isect::Intersection|Isect::NoMirror|Isect::Pick|Isect::Walk|Isect::Touch)); // cull everything that is visible
    camera->setCullMaskLeft(~0 & ~(Isect::Right|Isect::Collision|Isect::Intersection|Isect::NoMirror|Isect::Pick|Isect::Walk|Isect::Touch)); // cull everything that is visible and not right
    camera->setCullMaskRight(~0 & ~(Isect::Left|Isect::Collision|Isect::Intersection|Isect::NoMirror|Isect::Pick|Isect::Walk|Isect::Touch)); // cull everything that is visible and not Left


    osgViewer::Renderer *renderer = new coVRRenderer(camera.get(), 0);
    camera->setRenderer(renderer);
    camera->setGraphicsContext(cam->getGraphicsContext());
    camera->attach(osg::Camera::COLOR_BUFFER, image.get());
    //pBufferCamera->setNearFarRatio(coVRConfig::instance()->nearClip()/coVRConfig::instance()->farClip());
    camera->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
    camera->setPostDrawCallback(drawCallback.get());
    camera->setLODScale(0.0); // always highest LOD
    renderer->getSceneView(0)->setSceneData(cover->getScene());
    renderer->getSceneView(1)->setSceneData(cover->getScene());

	renderer->getSceneView(0)->setComputeStereoMatricesCallback(this);
	renderer->getSceneView(1)->setComputeStereoMatricesCallback(this);
}

void MapLinkPlugin::menuEvent(coMenuItem *aButton)
{
    
}
void MapLinkPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
}

void MapLinkPlugin::tabletEvent(coTUIElement *tUIItem)
{
}


void MapLinkPlugin::sendMessage(Message &m)
{
    if(toMapLink) // false on slaves
    {
        toMapLink->sendMessage(&m);
    }
}


void MapLinkPlugin::message(int toWhom, int type, int len, const void *buf)
{
    if (type == PluginMessageTypes::MoveAddMoveNode)
    {
    }
    else if(type >= PluginMessageTypes::HLRS_MapLink_Message && type <= (PluginMessageTypes::HLRS_MapLink_Message+100))
    {
        Message m{ type - PluginMessageTypes::HLRS_MapLink_Message + MSG_GetHeight , covise::DataHandle{(char *)buf, len, false} };
        handleMessage(&m);
    }

}

MapLinkPlugin *MapLinkPlugin::plugin = NULL;
void
MapLinkPlugin::handleMessage(Message *m)
{
    //cerr << "got Message" << endl;
    //m->print();
    enum PluginMessageTypes::Type type = (enum PluginMessageTypes::Type)m->type;
    
    switch (type)
    {
        
        case opencover::PluginMessageTypes::HLRS_MapLink_Message:
        {
            TokenBuffer tb(m);
            int t;
            tb >> t;
            std::cerr << "t from payload: " << t << std::endl;
            float _scale = cover->getScale();
            switch(t)
            {
            case MSG_GetHeight:
            {
                    std::cerr << "entered MSG_GetHeight" << std::endl;
                    osg::Matrix oldXformMat=cover->getXformMat();
                    cover->setXformMat(osg::Matrix());
                    int numPoints;
                    std::cerr << "before reading numPoints" << std::endl;
                    tb >> numPoints;
                    std::cerr << "after reading numPoints" << std::endl;
                    std::cerr << "numPoints: " << numPoints << std::endl;
                    TokenBuffer rtb;
                    rtb << MSG_GetHeight;
                    rtb << numPoints;
                    for(int i=0;i<numPoints;i++)
                    {
                        float x; float y;
                        tb >> x;
                        tb >> y;

                        std::cerr << "point " << i << ": x=" << x << " y=" << y << std::endl;

                        x*=(1000.0);
                        y*=(1000.0);

                        double minHeightValue = 100000000.0;
                        double maxHeightValue = -100000000.0;


                        osg::Vec3 rayP = osg::Vec3(x, y, 9999999);
                        osg::Vec3 rayQ = osg::Vec3(x, y, -9999999);

                        coIntersector* isect = coIntersection::instance()->newIntersector(rayP, rayQ);
                        osgUtil::IntersectionVisitor visitor(isect);
                        visitor.setTraversalMask(Isect::Collision);

                        cover->getObjectsXform()->accept(visitor);

                        //std::cerr << "Hits ray num: " << num1 << ", down (" << ray->start()[0] << ", " << ray->start()[1] <<  ", " << ray->start()[2] << "), up (" << ray->end()[0] << ", " << ray->end()[1] <<  ", " << ray->end()[2] << ")" <<  std::endl;
                        if (!isect->containsIntersections())
                        {
                            rtb << 0.0f;
                        }
                        else
                        {
                            auto results = isect->getFirstIntersection();

                            osg::Vec3d terrainHeight = results.getWorldIntersectPoint();

                            double height = terrainHeight.z() / 1000.0;
                            if (height < minHeightValue)
                            {
                                minHeightValue = height;
                            }
                            if (height > maxHeightValue)
                            {
                                maxHeightValue = height;
                            }

                            rtb << (float)height;
                        }
                    }
                    Message m(rtb);
                    m.type = PluginMessageTypes::HLRS_MapLink_Message;
                    std::cerr << "sending height response" << std::endl;
                    sendMessage(m);
                    cover->setXformMat(oldXformMat);
                }
                break;
            case MSG_GetMap:
                {
                    tb >> x;
                    tb >> y;
                    tb >> width;
                    tb >> height;
                    tb >> xRes;
                    tb >> yRes;
                    fprintf(stderr," x: %f  y: %f width: %f height: %f\n",x,y,width,height);
                    setProjection(x,y,width,height);
                }
                break;

            default:
                cerr << "Unknown MapLink to COVER message " << t << endl;
                break;
            }
        }
        break;
        
    
        
    default:
        switch (m->type)
        {
        case Message::SOCKET_CLOSED:
        case Message::CLOSE_SOCKET:
            cover->unwatchFileDescriptor(toMapLink->getSocket()->get_id());
            toMapLink.reset(nullptr);

            cerr << "connection to MapLink closed" << endl;
            break;
        default:
            cerr << "Unknown MapLink message " << m->type << endl;
            break;
        }
    }
}

void
MapLinkPlugin::preFrame()
{
}

bool MapLinkPlugin::update()
{

    if (serverConn && serverConn->is_connected() && serverConn->check_for_input()) // we have a server and received a connect
    {
        //   std::cout << "Trying serverConn..." << std::endl;
        toMapLink = serverConn->spawn_connection();
        if (toMapLink && toMapLink->is_connected())
        {
            fprintf(stderr, "Connected to MapLink\n");
            //int testit = 4321;
            //toMapLink->send(&testit, 4);
            cover->watchFileDescriptor(toMapLink->getSocket()->get_id());
        }
    }
    char gotMsg = '\0';
    if (coVRMSController::instance()->isMaster())
    {
        if(toMapLink)
        {
            static double lastTime = 0;
            if(abs(cover->frameTime() - lastTime )> +4)
            {
                lastTime = cover->frameTime();
                
            }
        }
        while (toMapLink && toMapLink->check_for_input())
        {
            // Testcode zum direkten Lesen von 4 Byte.
            // Wurde nur zur Prüfung der Raw-TCP-Verbindung verwendet.
            //cerr << "received data" << endl;
            //int buf;
            //toMapLink->receive(&buf, 4);
            //cerr << "received" << buf << endl;
           
            toMapLink->recv_msg(msg);


            if (msg)
            {
                std::cerr << "msg received" << std::endl;
                std::cerr << "msg->type: " << msg->type << std::endl;
                std::cerr << "msg->data.length(): " << msg->data.length() << std::endl;
                gotMsg = '\1';
                coVRMSController::instance()->sendSlaves(&gotMsg, sizeof(char));
                coVRMSController::instance()->sendSlaves(msg);
                
                //cover->sendMessage(this, coVRPluginSupport::TO_SAME_OTHERS,PluginMessageTypes::HLRS_MapLink_Message+msg->type-MSG_GetHeight,msg->data.length(), msg->data.data());
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
    return true;
}

COVERPLUGIN(MapLinkPlugin)
