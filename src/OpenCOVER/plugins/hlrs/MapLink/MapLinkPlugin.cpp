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
#include <cover/coIntersection.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/osg/OSGVruiUserDataCollection.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <geodata/GeoData.h>
#include <iostream>
#include <limits>
#include <osg/Material>
#include <osg/StateSet>

#include <osg/NodeVisitor>
#include <string>


#include <PluginUtil/PluginMessageTypes.h>

#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Array>
#include <osg/CullFace>
#include <osg/MatrixTransform>
#include <osg/LineSegment>
#include <osg/Node>
#include <osg/Vec3d>
#include <osg/ref_ptr>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/LineWidth>
#include <osg/StateSet>
#include <osg/ComputeBoundsVisitor>
#include <osg/Matrix>
#include <osgDB/ReadFile>
#include <osg/Group>


#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <net/tokenbuffer.h>
#include <config/CoviseConfig.h>
#include <array>
#include <unordered_map>


using covise::TokenBuffer;
using covise::coCoviseConfig;

// Parameter für die Transformation für das .wrl Modell
namespace
{
struct ModelTransformConfig
{
    double modelEasting;
    double modelNorthing;
    double scale;
    double worldOffsetX;
    double worldOffsetY;
};

const ModelTransformConfig modelTransform {
    507297.0,
    5398513.0,
    1000.0,
    225900.0,
    87640.1
};
}

//Funktion für die Trafo von GeoData-Koordinaten zum tats. 3D-Modell
osg::Vec2d referenceToModelWorldXY(const osg::Vec3d &referencePosition)
{
    const double modelX = referencePosition.x() - modelTransform.modelEasting;

    const double modelY = referencePosition.y() - modelTransform.modelNorthing;

    const double worldX = modelX * modelTransform.scale + modelTransform.worldOffsetX;

    const double worldY = modelY * modelTransform.scale + modelTransform.worldOffsetY;

    return osg::Vec2d(worldX, worldY);
}

void printMatrix(const char *name, const osg::Matrix &m)
{
    std::cerr << name << std::endl;

    for (int row = 0; row < 4; ++row)
    {
        std::cerr << "  ";

        for (int col = 0; col < 4; ++col)
        {
            std::cerr << m(row, col) << " ";
        }

        std::cerr << std::endl;
    }
}


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
    m_selectModulesButton = new coCheckboxMenuItem(
        "PV-Module auswaehlen",
        false);

    m_selectModulesButton->setMenuListener(this);

    cover->getMenu()->add(
        m_selectModulesButton);


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
    MapLinkTab = nullptr;
}

/*
void MapLinkPlugin::showLocationMarker(double x, double y, const osg::Vec4 &color)
{
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry();

    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
    vertices->push_back(osg::Vec3(x, y, 400000.0));
    vertices->push_back(osg::Vec3(x, y, 520000.0));

    geometry->setVertexArray(vertices.get());
    geometry->addPrimitiveSet(
        new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 2));

    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();
    colors->push_back(color);

    geometry->setColorArray(colors.get());
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);

    osg::StateSet *stateSet = geometry->getOrCreateStateSet();
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::ref_ptr<osg::LineWidth> lineWidth = new osg::LineWidth();
    lineWidth->setWidth(8.0f);
    stateSet->setAttributeAndModes(
        lineWidth.get(),
        osg::StateAttribute::ON);

    geode->addDrawable(geometry.get());
    if (m_cityModelParent.valid())
    {
        osg::ref_ptr<osg::MatrixTransform> markerTransform = new osg::MatrixTransform();

        markerTransform->setMatrix(m_worldToCityParent);
        markerTransform->addChild(geode.get());

        m_cityModelParent->addChild(markerTransform.get());
    }

    printMatrix(
        "ObjectsXform BEIM EINFUEGEN:",
        cover->getObjectsXform()->getMatrix());

    std::cerr << "LOCATION MARKER added:"
              << " x=" << x
              << " y=" << y
              << std::endl;

    osg::Matrixd objectsMatrix = cover->getObjectsXform()->getMatrix();

    osg::Vec3d markerPosition(x, y, 462000.0);

    osg::Vec3d transformedPosition = markerPosition * objectsMatrix;

    std::cerr << "----- MARKER TRANSFORM -----" << std::endl;

    std::cerr << "Marker lokal: "
              << markerPosition.x() << ", "
              << markerPosition.y() << ", "
              << markerPosition.z() << std::endl;

    std::cerr << "Marker transformiert: "
              << transformedPosition.x() << ", "
              << transformedPosition.y() << ", "
              << transformedPosition.z() << std::endl;
}*/

void MapLinkPlugin::createModule(
    int moduleId,
    const std::array<osg::Vec3d, 4> &corners,
    osg::Node *pvModel)
{
    if (!pvModel || !m_pvModuleGroup.valid())
        return;

    // Lokale X-Achse: P1 -> P2
    osg::Vec3d xAxis = corners[1] - corners[0];
    xAxis.normalize();

    // Vorläufige Y-Achse: P1 -> P4
    osg::Vec3d yDirection = corners[3] - corners[0];

    // Normale der Modulfläche berechnen
    osg::Vec3d zAxis = xAxis ^ yDirection;
    zAxis.normalize();

    // Sicherstellen, dass die Moduloberseite nach oben zeigt
    if (zAxis.z() < 0.0)
    {
        zAxis = -zAxis;
    }

    // Rechtwinklige Y-Achse berechnen
    osg::Vec3d yAxis = zAxis ^ xAxis;
    yAxis.normalize();

    // Mittelpunkt der vier Eckpunkte
    osg::Vec3d center = (corners[0] + corners[1] + corners[2] + corners[3]) / 4.0;

    // Rotationsmatrix aus den drei lokalen Achsen
    osg::Matrixd rotation(
        xAxis.x(), xAxis.y(), xAxis.z(), 0.0,
        yAxis.x(), yAxis.y(), yAxis.z(), 0.0,
        zAxis.x(), zAxis.y(), zAxis.z(), 0.0,
        0.0, 0.0, 0.0, 1.0);

    // Blender-Meter -> OpenCOVER-Millimeter
    osg::Matrixd transform = osg::Matrixd::scale(1000.0, 1000.0, 1000.0) * rotation * osg::Matrixd::translate(center);

    osg::ref_ptr<osg::MatrixTransform> moduleTransform = new osg::MatrixTransform();

    moduleTransform->setName(
        "PV_Module_" + std::to_string(moduleId));

    moduleTransform->setMatrix(transform);
    moduleTransform->addChild(pvModel);

    // Existiert dieses Modul bereits?
    auto existing = m_moduleNodes.find(moduleId);

    if (existing != m_moduleNodes.end())
    {
        // Alten 3D-Knoten entfernen
        if (existing->second.valid())
        {
            m_pvModuleGroup->removeChild(
                existing->second.get());
        }

        m_moduleNodes.erase(existing);
    }

    // Neuen 3D-Knoten einfügen
    m_pvModuleGroup->addChild(moduleTransform.get());

    // Modul anhand seiner ID speichern
    m_moduleNodes[moduleId] = moduleTransform;
}

void MapLinkPlugin::deleteModule(int moduleId)
{
    auto it = m_moduleNodes.find(moduleId);

    if (it == m_moduleNodes.end())
        return;

    if (it->second.valid() && m_pvModuleGroup.valid())
    {
        m_pvModuleGroup->removeChild(
            it->second.get());
    }

    m_moduleNodes.erase(it);
    m_modules.erase(moduleId);

    std::cerr
        << "MapLink: Modul "
        << moduleId
        << " geloescht."
        << std::endl;
}

void MapLinkPlugin::clearAllModules()
{
    if (m_pvModuleGroup.valid())
    {
        m_pvModuleGroup->removeChildren(
            0,
            m_pvModuleGroup->getNumChildren());
    }

    m_moduleNodes.clear();
    m_modules.clear();

    std::cerr
        << "MapLink: Alle PV-Module geloescht."
        << std::endl;
}

void MapLinkPlugin::sendDeleteModuleRequest(int moduleId)
{
    covise::TokenBuffer tb;

    tb << MSG_DeleteModuleRequest;
    tb << moduleId;

    Message m(tb);
    m.type = PluginMessageTypes::HLRS_MapLink_Message;

    std::cerr
        << "MapLink: Sende Loeschanforderung fuer Modul "
        << moduleId
        << std::endl;

    sendMessage(m);
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

    // Interaktion zur Auswahl eines PV-Moduls in OpenCOVER
    m_selectInteraction = new coTrackerButtonInteraction(
        coInteraction::ButtonA,
        "MapLinkModuleSelection");

    // Gruppe für die PV-Module erstellen
    m_pvModuleGroup = new osg::Group();
    m_pvModuleGroup->setName("PV_Modules");
    // PV-Module von der Hoehenabfrage ausschliessen
    m_pvModuleGroup->setNodeMask(0xFFFFFFFF);

    std::cerr << "MapLink: PV-Modulgruppe erstellt." << std::endl;
    return true;
}
// this is called if the plugin is removed at runtime
MapLinkPlugin::~MapLinkPlugin()
{
    if (serverConn && serverConn->getSocket())
        cover->unwatchFileDescriptor(serverConn->getSocket()->get_id());

    if (toMapLink && toMapLink->getSocket())
        cover->unwatchFileDescriptor(toMapLink->getSocket()->get_id());

    destroyMenu();

    delete serverConn;
    serverConn = nullptr;

    delete msg;
    msg = nullptr;

    if (camera.get())
    {
        camera->detach(osg::Camera::COLOR_BUFFER);
        camera->setGraphicsContext(nullptr);
        VRViewer::instance()->removeCamera(camera.get());
    }

    toMapLink.reset();
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
    if (aButton == m_selectModulesButton)
    {
        if (m_selectModulesButton->getState())
        {
            // Auswahlmodus einschalten
            coInteractionManager::the()->registerInteraction(
                m_selectInteraction);

            std::cerr
                << "MapLink: PV-Modulauswahl aktiviert."
                << std::endl;
        }
        else
        {
            // Auswahlmodus ausschalten
            coInteractionManager::the()->unregisterInteraction(
                m_selectInteraction);

            m_selectedModuleId = -1;

            std::cerr
                << "MapLink: PV-Modulauswahl deaktiviert."
                << std::endl;
        }
    }
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
                std::cerr << "MSG_GetHeight received" << std::endl;

                const osg::Matrix oldXformMat = cover->getXformMat();
                cover->setXformMat(osg::Matrix());

                int numPoints;
                tb >> numPoints;

                TokenBuffer rtb;
                rtb << MSG_GetHeight;
                rtb << numPoints;

                // Production_OSG-Georeferenzierung aus praesentation_26.wrl
                constexpr double MODEL_EASTING = 507297.0;
                constexpr double MODEL_NORTHING = 5398513.0;

                // Production_OSG -> OpenCOVER, aus osg::computeLocalToWorld ermittelt
                constexpr double MODEL_SCALE = 1000.0;
                constexpr double MODEL_WORLD_OFFSET_X = 225900.0;
                constexpr double MODEL_WORLD_OFFSET_Y = 87640.1;


                for (int i = 0; i < numPoints; ++i)
                {
                    double longitude;
                    double latitude;

                    tb >> longitude;
                    tb >> latitude;

                    // Eingang: EPSG:4326
                    const osg::Vec3d globalPosition(
                        static_cast<double>(longitude),
                        static_cast<double>(latitude),
                        0.0);

                    // EPSG:4326 -> EPSG:25832
                    const osg::Vec3d referencePosition = GeoData::instance()->globalToReference(globalPosition);

                    // Nur zum Vergleich mit dem bisherigen GeoData-Projektraum
                    const osg::Vec3d projectPosition = GeoData::instance()->globalToProject(globalPosition);

                    // UTM -> lokale Koordinaten des Production_OSG
                    const osg::Vec2d modelWorldPosition = referenceToModelWorldXY(referencePosition);

                    const double rayX = modelWorldPosition.x();
                    const double rayY = modelWorldPosition.y();


                    std::cerr
                        << "Point " << i
                        << " | LonLat=(" << longitude << ", " << latitude << ")"
                        << " | UTM=(" << referencePosition.x()
                        << ", " << referencePosition.y() << ")"
                        << " | GeoDataProject=(" << projectPosition.x()
                        << ", " << projectPosition.y() << ")"
                        << " | Ray=(" << rayX
                        << ", " << rayY << ")"
                        << std::endl;

          
                    const osg::Vec3 rayP(
                        rayX,
                        rayY,
                        9999999.0);

                    const osg::Vec3 rayQ(
                        rayX,
                        rayY,
                        -9999999.0);

                    std::cerr
                        << "ObjectsXform mask: "
                        << cover->getObjectsXform()->getNodeMask()
                        << " | children: "
                        << cover->getObjectsXform()->getNumChildren()
                        << std::endl;

                    for (unsigned int j = 0;
                        j < cover->getObjectsXform()->getNumChildren();
                        ++j)
                    {
                        osg::Node *child = cover->getObjectsXform()->getChild(j);

                        std::cerr
                            << "Child " << j
                            << " | Name: " << child->getName()
                            << " | Mask: " << child->getNodeMask()
                            << std::endl;
                    }

                    coIntersector *isect = coIntersection::instance()->newIntersector(rayP, rayQ);

                    osgUtil::IntersectionVisitor visitor(isect);
                    visitor.setTraversalMask(~0u);

                    // Bereits gesetzte PV-Module vorübergehend
                    // von der Höhenabfrage ausschließen.
                    osg::Node::NodeMask previousPvMask = 0;

                    if (m_pvModuleGroup.valid())
                    {
                        previousPvMask = m_pvModuleGroup->getNodeMask();
                        m_pvModuleGroup->setNodeMask(0u);
                    }

                    // Höhenabfrage durchführen
                    cover->getObjectsXform()->accept(visitor);

                    // Ursprüngliche Maske wiederherstellen,
                    // damit die PV-Module sichtbar bleiben.
                    if (m_pvModuleGroup.valid())
                    {
                        m_pvModuleGroup->setNodeMask(previousPvMask);
                    }

                    if (!isect->containsIntersections())
                    {
                        std::cerr << "  -> NO INTERSECTION: height unavailable" << std::endl;
                        rtb << std::numeric_limits<float>::quiet_NaN();
                        continue;
                    }

                    const auto result = isect->getFirstIntersection();

                    if (i == 0)
                    {
                        const osg::NodePath &path = result.nodePath;

                        for (std::size_t j = 0; j < path.size(); ++j)
                        {
                            if (path[j]->getName() == "VRMLRoot")
                            {
                                osg::Group *parent = path[j]->asGroup();

                                if (parent)
                                {
                                    m_cityModelParent = parent;

                                    osg::NodePath parentPath(
                                        path.begin(),
                                        path.begin() + j + 1);

                                    m_worldToCityParent = osg::computeWorldToLocal(parentPath);

                                    if (m_pvModuleGroup.valid() && m_pvModuleGroup->getNumParents() == 0)
                                    {
                                        osg::ref_ptr<osg::MatrixTransform> pvTransform = new osg::MatrixTransform();

                                        pvTransform->setName("PV_Modules_Transform");
                                        pvTransform->setMatrix(m_worldToCityParent);
                                        pvTransform->addChild(m_pvModuleGroup.get());

                                        parent->addChild(pvTransform.get());
                                    }
                                }

                                break;
                            }
                        }
                    }

                    if (i == 0)
                    {
                        std::cerr << "----- DACH: SZENENGRAPH-PFAD -----"
                                  << std::endl;

                        for (osg::Node *node : result.nodePath)
                        {
                            if (node)
                            {
                                std::cerr
                                    << node->className()
                                    << " | "
                                    << node->getName()
                                    << std::endl;
                            }
                        }
                    }
                    const osg::Vec3d worldPoint = result.getWorldIntersectPoint();

                    const double height = worldPoint.z() / 1000.0;


                    std::cerr
                        << "  -> HIT=("
                        << worldPoint.x() << ", "
                        << worldPoint.y() << ", "
                        << worldPoint.z() << ")"
                        << " | height=" << height << " m"
                        << std::endl;

                    rtb << static_cast<float>(height);
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

            case MSG_SetModules:
            {
                // OBJ-Modell aus Blender laden
                const std::string modelPath = "C:/src/covise/src/OpenCOVER/plugins/hlrs/MapLink/models/PV_kompakt_hoch.obj";

                osg::ref_ptr<osg::Node> pvModel = osgDB::readNodeFile(modelPath);

                if (pvModel.valid())
                {
                    std::cerr
                        << "MapLink: OBJ-Modell erfolgreich geladen!"
                        << std::endl;
                }
                else
                {
                    std::cerr
                        << "MapLink: FEHLER - OBJ-Modell konnte nicht geladen werden!"
                        << std::endl;
                }

                // Anzahl der empfangenen Module
                int numModules;
                tb >> numModules;

                std::cerr
                    << "MSG_SetModules: "
                    << numModules
                    << " modules received"
                    << std::endl;

                // Alle empfangenen Module verarbeiten
                for (int moduleIndex = 0;
                    moduleIndex < numModules;
                    ++moduleIndex)
                {
                    int moduleId;
                    tb >> moduleId;

                    std::array<osg::Vec3d, 4> worldCorners;

                    // Vier Eckpunkte jedes Moduls empfangen
                    for (int cornerIndex = 0;
                        cornerIndex < 4;
                        ++cornerIndex)
                    {
                        double longitude, latitude, height;

                        tb >> longitude;
                        tb >> latitude;
                        tb >> height;

                        // EPSG:4326
                        const osg::Vec3d globalCorner(
                            longitude,
                            latitude,
                            0.0);

                        // EPSG:4326 -> EPSG:25832
                        const osg::Vec3d referenceCorner = GeoData::instance()->globalToReference(
                            globalCorner);

                        // EPSG:25832 -> OpenCOVER-Modellkoordinaten
                        const osg::Vec2d modelWorldXY = referenceToModelWorldXY(referenceCorner);

                        // Modulecken werden nach oben gesetzt
                        constexpr double MODULE_HEIGHT_OFFSET_MM = 100.0;

                        // Höhe von Metern in Millimeter umrechnen
                        const double worldZ = height * 1000.0 + MODULE_HEIGHT_OFFSET_MM;

                        worldCorners[cornerIndex] = osg::Vec3d(
                            modelWorldXY.x(),
                            modelWorldXY.y(),
                            worldZ);

                        std::cerr
                            << "Module " << moduleId
                            << ", corner " << cornerIndex
                            << " -> world=("
                            << worldCorners[cornerIndex].x() << ", "
                            << worldCorners[cornerIndex].y() << ", "
                            << worldCorners[cornerIndex].z() << ")"
                            << std::endl;
                    }

                    // Eckpunkte speichern
                    m_modules[moduleId] = worldCorners;

                    // Tatsächliches 3D-Modul erstellen
                    if (pvModel.valid())
                    {
                        createModule(
                            moduleId,
                            worldCorners,
                            pvModel.get());
                    }
                }

                std::cerr
                    << "Stored modules: "
                    << m_modules.size()
                    << std::endl;

                // Antwort an den Server
                TokenBuffer rtb;
                rtb << MSG_SetModules;
                rtb << numModules;

                Message response(rtb);
                response.type = PluginMessageTypes::HLRS_MapLink_Message;

                sendMessage(response);

                break;
            }

            case MSG_DeleteModules:
            {
                int numModules;
                tb >> numModules;

                std::cerr << "MSG_DeleteModules received: "
                          << numModules << " modules" << std::endl;

                for (int i = 0; i < numModules; ++i)
                {
                    int moduleId;
                    tb >> moduleId;

                    deleteModule(moduleId);
                }

                std::cerr << "Remaining modules: "
                          << m_modules.size() << std::endl;

                break;
            }

            case MSG_ClearAllModules:
            {
                std::cerr << "MSG_ClearAllModules received"
                          << std::endl;

                clearAllModules();

                std::cerr << "All modules cleared. Remaining modules: "
                          << m_modules.size() << std::endl;

                break;
            }

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

void MapLinkPlugin::preFrame()
{
    if (m_selectInteraction && m_selectInteraction->wasStarted())
    {
        std::cerr
            << "MapLink: Klick erkannt."
            << std::endl;

        osg::Matrix mouseMat = cover->getMouseMat();

        osg::Vec3d rayStart(
            0.0,
            0.0,
            0.0);

        osg::Vec3d rayEnd(
            0.0,
            10000000.0,
            0.0);

        rayStart = mouseMat.preMult(rayStart);

        rayEnd = mouseMat.preMult(rayEnd);

        std::cerr
            << "MapLink: Ray berechnet."
            << std::endl;

        coIntersector *isect = coIntersection::instance()->newIntersector(
            rayStart,
            rayEnd);

        std::cerr
            << "MapLink: Intersector erzeugt."
            << std::endl;

        if (!m_pvModuleGroup)
        {
            std::cerr
                << "MapLink: FEHLER - PV-Modulgruppe existiert nicht."
                << std::endl;

            return;
        }

        std::cerr
            << "MapLink: PV-Gruppe hat "
            << m_pvModuleGroup->getNumChildren()
            << " Kinder."
            << std::endl;

        osgUtil::IntersectionVisitor visitor(isect);

        std::cerr
            << "MapLink: Starte Traversierung."
            << std::endl;

        cover->getObjectsXform()->accept(visitor);

        std::cerr
            << "MapLink: Traversierung beendet."
            << std::endl;

        if (isect->containsIntersections())
        {
            const auto result = isect->getFirstIntersection();

            bool moduleFound = false;

            for (osg::Node *node : result.nodePath)
            {
                if (!node)
                    continue;

                const std::string &name = node->getName();

                std::cerr
                    << "Trefferpfad: "
                    << name
                    << std::endl;

                const std::string prefix = "PV_Module_";

                if (name.rfind(prefix, 0) == 0)
                {
                    try
                    {
                        const int moduleId = std::stoi(name.substr(prefix.length()));

                        m_selectedModuleId = moduleId;
                        moduleFound = true;

                        std::cerr
                            << "MapLink: PV-Modul "
                            << moduleId
                            << " ausgewaehlt."
                            << std::endl;

                        break;
                    }
                    catch (...)
                    {
                        std::cerr
                            << "MapLink: Ungueltige Modul-ID in Knotenname: "
                            << name
                            << std::endl;
                    }
                }
            }

            if (!moduleFound)
            {
                m_selectedModuleId = -1;

                std::cerr
                    << "MapLink: Treffer ist kein PV-Modul."
                    << std::endl;
            }
        }
        else
        {
            m_selectedModuleId = -1;

            std::cerr
                << "MapLink: Kein Treffer."
                << std::endl;
        }
    }

    static int frameCounter = 0;

    if (++frameCounter % 300 == 0)
    {
        printMatrix(
            "ObjectsXform:",
            cover->getObjectsXform()->getMatrix());
    }
}

void MapLinkPlugin::key(int type, int keySym, int mod)
{
    if (type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        if (keySym == osgGA::GUIEventAdapter::KEY_Delete)
        {
            if (m_selectedModuleId >= 0)
            {
                std::cerr
                    << "MapLink: Loeschen angefordert fuer Modul "
                    << m_selectedModuleId
                    << std::endl;

                sendDeleteModuleRequest(m_selectedModuleId);
            }
            else
            {
                std::cerr
                    << "MapLink: Entf gedrueckt, aber kein Modul ausgewaehlt."
                    << std::endl;
            }
        }
    }
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
