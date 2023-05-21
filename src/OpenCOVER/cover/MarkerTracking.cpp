/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <OpenVRUI/osg/mathUtils.h>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#endif

#include <util/common.h>

#include <sysdep/opengl.h>
#include <config/CoviseConfig.h>
#include "MarkerTracking.h"
#include "VRViewer.h"
#include "coVRConfig.h"
#include "coVRPluginSupport.h"
#include "coVRPluginList.h"
#include "coVRMSController.h"
#include "coVRCollaboration.h"
#include "coVRTui.h"

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Vec3>
#include <osg/StateSet>

#include <osgDB/ReadFile>
#include <osgDB/Registry>

#include <set>
#include <sstream>

using namespace covise;
using namespace opencover;

osg::Matrix OpenGLToOSGMatrix;
osg::Matrix PfToOpenGLMatrix;


MarkerTracking *MarkerTracking::art = NULL;
MarkerTracking::MarkerTracking()
{
    assert(!art);
    art = this;
    artTab = new coTUITab("MarkerTracking", coVRTui::instance()->mainFolder->getID());
    artTab->setHidden(true); // hide until a marker tracking plugin is loaded
    m_markerDatabase = cover->configFile("markerdatabase"); 
    m_markerDatabase->setSaveOnExit(true);
    OpenGLToOSGMatrix.makeRotate(M_PI / 2.0, 1, 0, 0);
    PfToOpenGLMatrix.makeRotate(-M_PI / 2.0, 1, 0, 0);
    // auto g = new coTUIGroupBox("configure new marker", artTab->getID());
    m_buttonsFrame = new coTUIFrame("config", artTab->getID());
    m_buttonsFrame->setPos(0, 0);
    m_trackingFrame = new coTUIFrame("tracking", artTab->getID());
    m_trackingFrame->setPos(1, 0);
    
    m_configureMarkerBtn = new coTUIButton("add new markers", m_buttonsFrame->getID());
    m_configureMarkerBtn->setEventListener(this);
    m_configureMarkerBtn->setPos(0, 0);
    m_saveBtn = new coTUIButton("save", m_buttonsFrame->getID());
    m_saveBtn->setEventListener(this);
    m_saveBtn->setPos(1, 0);

}

void MarkerTracking::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == m_configureMarkerBtn && arInterface)
    {
        arInterface->createUnconfiguredTrackedMarkers();
    }
    if (tUIItem == m_saveBtn)
    {
        m_markerDatabase->save();
    }
}


MarkerTrackingNode::MarkerTrackingNode(std::string MarkerTrackingVariant)
{
    theNode = this;
    setSupportsDisplayList(false);
    std::string configPath = "COVER.Plugin.";
    configPath += MarkerTrackingVariant;
    std::string entry = configPath + "DisplayVideo";
    displayVideo = coCoviseConfig::isOn(entry, true);
    entry = configPath + "RenderTextures";
    renderTextures = coCoviseConfig::isOn(entry, true);
    m_MarkerTrackingVariant = MarkerTrackingVariant;
}

MarkerTrackingNode::~MarkerTrackingNode()
{
    theNode = NULL;
}

MarkerTrackingNode *MarkerTrackingNode::theNode = NULL;

/** Clone the type of an object, with Object* return type.
Must be defined by derived classes.*/
osg::Object *MarkerTrackingNode::cloneType() const
{
    return new MarkerTrackingNode(m_MarkerTrackingVariant);
}

/** Clone the an object, with Object* return type.
Must be defined by derived classes.*/
osg::Object *MarkerTrackingNode::clone(const osg::CopyOp &) const
{
    return new MarkerTrackingNode(m_MarkerTrackingVariant);
}

void MarkerTrackingNode::drawImplementation(osg::RenderInfo &renderInfo) const
{
    static bool firstTime = true;
    static GLuint texHandle = 0;
    if (firstTime)
    {
        glGenTextures(1, &texHandle);
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, texHandle);
        glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
        firstTime = false;
    }
    bool rightVideo = false;
    if (osg::View *view = renderInfo.getView())
    {
        if (osg::State *state = renderInfo.getState())
        {
            if (const osg::DisplaySettings *ds = state->getDisplaySettings())
            {
                switch (ds->getStereoMode())
                {
                case osg::DisplaySettings::HORIZONTAL_INTERLACE:
                case osg::DisplaySettings::VERTICAL_INTERLACE:
                case osg::DisplaySettings::CHECKERBOARD:
                case osg::DisplaySettings::ANAGLYPHIC:
                    /* TODO */
                    break;
                case osg::DisplaySettings::HORIZONTAL_SPLIT:
                case osg::DisplaySettings::VERTICAL_SPLIT:
                    if (osg::Camera *cam = view->getCamera())
                    {
                        for (int i = 0; i < coVRConfig::instance()->numScreens(); ++i)
                        {
                            if (coVRConfig::instance()->channels[i].camera.get() == cam)
                            {
                                rightVideo = coVRConfig::instance()->channels[i].stereoMode == osg::DisplaySettings::RIGHT_EYE;
                                break;
                            }
                        }
                    }
                    break;
                case osg::DisplaySettings::LEFT_EYE:
                    break;
                case osg::DisplaySettings::RIGHT_EYE:
                    rightVideo = true;
                    break;
                case osg::DisplaySettings::QUAD_BUFFER:
                    if (osg::Camera *cam = view->getCamera())
                        rightVideo = (cam->getDrawBuffer() == GL_BACK_RIGHT || cam->getDrawBuffer() == GL_FRONT_RIGHT);
                    break;
                default:
                    cerr << "MarkerTrackingNode::drawImplementation: unknown stereo mode" << endl;
                    break;
                }
            }
        }
    }

    GLint viewport[4]; // OpenGL viewport information (position and size)
    if (MarkerTracking::instance()->testImage)
    {

        if (osg::View *view = renderInfo.getView())
        {
            osg::Camera *cam = view->getCamera();
            if (cam)
            {

                static bool firstTime = true;
                static GLuint texHandle2 = 0;
                static osg::Image *image = NULL;
                if (firstTime)
                {
                    osgDB::ReaderWriter::Options *options = 0;

                    std::string testimage = coCoviseConfig::getEntry("COVER.TestImage");
                    image = osgDB::readImageFile(testimage.c_str(), options);
                }

                if (image)
                {
                    glMatrixMode(GL_MODELVIEW);
                    glPushMatrix();
                    glMatrixMode(GL_PROJECTION);
                    glPushMatrix();
                    glGetIntegerv(GL_VIEWPORT, viewport);
                    glDepthMask(false);

                    glMatrixMode(GL_MODELVIEW);
                    glLoadIdentity();
                    glMatrixMode(GL_PROJECTION);
                    glLoadIdentity();
                    gluOrtho2D(-1, 1, -1, 1);

                    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, texHandle2);

                    glEnable(GL_TEXTURE_RECTANGLE_ARB); //
                    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE); //

                    float xPos = 1.0;
                    float yPos = 1.0;

                    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA8, image->s(), image->t(), 0, image->getPixelFormat(), GL_UNSIGNED_BYTE, image->data());
                    glBegin(GL_QUADS);
                    {
                        /*glTexCoord2f(0, image->t());
                        glVertex2f(-xPos, -yPos);
                        glTexCoord2f(image->s(), image->t());
                        glVertex2f(xPos, -yPos);
                        glTexCoord2f(image->s(), 0);
                        glVertex2f(xPos, yPos);
                        glTexCoord2f(0, 0);
                        glVertex2f(-xPos, yPos);*/
						glTexCoord2f(0, 0);
						glVertex2f(-xPos, -yPos);
						glTexCoord2f(image->s(), 0);
						glVertex2f(xPos, -yPos);
						glTexCoord2f(image->s(), image->t());
						glVertex2f(xPos, yPos);
						glTexCoord2f(0, image->t());
						glVertex2f(-xPos, yPos);
                    }
                    glEnd();

                    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
                    glDisable(GL_TEXTURE_RECTANGLE_ARB);

                    glDepthMask(true);
                    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
                    glMatrixMode(GL_PROJECTION);
                    glPopMatrix();
                    glMatrixMode(GL_MODELVIEW);
                    glPopMatrix();

                    firstTime = false;
                }
            }
        }
    }
    if (displayVideo)
    {
        if (MarkerTracking::instance()->videoWidth > 0)
        {
            // Save OpenGL state:
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glGetIntegerv(GL_VIEWPORT, viewport);
            glDepthMask(false);

            float xsize;
            float ysize;

            if ((coVRConfig::instance()->viewports[0].viewportXMax - coVRConfig::instance()->viewports[0].viewportXMin) == 0)
            {
                xsize = coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sx;
                ysize = coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sy;
            }
            else
            {
                xsize = coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sx * (coVRConfig::instance()->viewports[0].viewportXMax - coVRConfig::instance()->viewports[0].viewportXMin);
                ysize = coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sy * (coVRConfig::instance()->viewports[0].viewportYMax - coVRConfig::instance()->viewports[0].viewportYMin);
            }

            if (renderTextures) // textures
            {

                // DISPLAY TEXTURE //
                glMatrixMode(GL_MODELVIEW);
                glLoadIdentity();
                glMatrixMode(GL_PROJECTION);
                glLoadIdentity();
                gluOrtho2D(-1, 1, -1, 1);

                glBindTexture(GL_TEXTURE_RECTANGLE_ARB, texHandle);

                glEnable(GL_TEXTURE_RECTANGLE_ARB); //
                glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE); //

                float xPos = 1.0;
                float yPos = 1.0;

                if (MarkerTracking::instance()->flipH)
                {
                    xPos *= -1;
                }

                if ((MarkerTracking::instance()->stereoVideo) && (rightVideo))
                {
                    if (MarkerTracking::instance()->videoDataRight)
                    {
                        glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA8, MarkerTracking::instance()->videoWidth, MarkerTracking::instance()->videoHeight, 0, MarkerTracking::instance()->videoMode, GL_UNSIGNED_BYTE, MarkerTracking::instance()->videoDataRight);
                        if (MarkerTracking::instance()->videoMirrorRight)
                        {
                            xPos *= -1;
                            yPos *= -1;
                        }
                    }
                }
                else
                {
                    if (MarkerTracking::instance()->videoData)
                    {
                        glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA8, MarkerTracking::instance()->videoWidth, MarkerTracking::instance()->videoHeight, 0, MarkerTracking::instance()->videoMode, GL_UNSIGNED_BYTE, MarkerTracking::instance()->videoData);
                        if (MarkerTracking::instance()->videoMirrorLeft)
                        {
                            xPos *= -1;
                            yPos *= -1;
                        }
                    }
                }

                glBegin(GL_QUADS);
                {
                    glTexCoord2f(0, MarkerTracking::instance()->videoHeight);
                    glVertex2f(-xPos, -yPos);
                    glTexCoord2f(MarkerTracking::instance()->videoWidth, MarkerTracking::instance()->videoHeight);
                    glVertex2f(xPos, -yPos);
                    glTexCoord2f(MarkerTracking::instance()->videoWidth, 0);
                    glVertex2f(xPos, yPos);
                    glTexCoord2f(0, 0);
                    glVertex2f(-xPos, yPos);
                }
                glEnd();

                glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
                glDisable(GL_TEXTURE_RECTANGLE_ARB);
            }
            else //glDrawPixels
            {

                // Draw:
                glMatrixMode(GL_MODELVIEW);
                glLoadIdentity();
                glMatrixMode(GL_PROJECTION);
                glLoadIdentity();

                //        glViewport(0, 0, 1, 1);
                float yPos = 1.0;
                float xPos = -1.0;
                if (MarkerTracking::instance()->flipH)
                {
                    ysize *= -1;
                    yPos *= -1;
                }
                glPixelZoom(xsize / MarkerTracking::instance()->videoWidth, -ysize / MarkerTracking::instance()->videoHeight);
                if ((MarkerTracking::instance()->stereoVideo) && (rightVideo))
                {
                    if (MarkerTracking::instance()->videoDataRight)
                    {
                        if (MarkerTracking::instance()->videoMirrorRight)
                        {
                            glPixelZoom(-xsize / MarkerTracking::instance()->videoWidth, ysize / MarkerTracking::instance()->videoHeight);
                            yPos *= -1;
                            xPos *= -1;
                        }
                        glRasterPos2f(xPos, yPos);
                        glDrawPixels(MarkerTracking::instance()->videoWidth, MarkerTracking::instance()->videoHeight, MarkerTracking::instance()->videoMode, GL_UNSIGNED_BYTE, MarkerTracking::instance()->videoDataRight);
                        if (MarkerTracking::instance()->videoMirrorRight)
                        {
                            glPixelZoom(xsize / MarkerTracking::instance()->videoWidth, -ysize / MarkerTracking::instance()->videoHeight);
                        }
                    }
                }
                else
                {
                    if (MarkerTracking::instance()->videoData)
                    {
                        if (MarkerTracking::instance()->videoMirrorLeft)
                        {
                            glPixelZoom(-xsize / MarkerTracking::instance()->videoWidth, ysize / MarkerTracking::instance()->videoHeight);
                            yPos *= -1;
                            xPos *= -1;
                        }
                        glRasterPos2f(xPos, yPos);
                        //if(MarkerTracking::instance()->videoData)
                        //cerr << "x " << (long long)MarkerTracking::instance()->videoData << " content: " << (int)MarkerTracking::instance()->videoData[100] << endl;
                        glDrawPixels(MarkerTracking::instance()->videoWidth, MarkerTracking::instance()->videoHeight, MarkerTracking::instance()->videoMode, GL_UNSIGNED_BYTE, MarkerTracking::instance()->videoData);
                    }
                }
            }

            // Restore state:
            glDepthMask(true);
            glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glMatrixMode(GL_MODELVIEW);
            glPopMatrix();
        }
    }
    coVRPluginList::instance()->preDraw(renderInfo);
}

MarkerTracking *MarkerTracking::instance()
{
    if (art == NULL)
        art = new MarkerTracking();
    return art;
}

MarkerTrackingMarker *MarkerTracking::getMarker(const std::string &name)
{
    auto m = markers.find(name);
    if(m == markers.end())
        return nullptr;
    return m->second.get();
}

MarkerTrackingMarker *MarkerTracking::getOrCreateMarker(const std::string &name, const std::string &pattern, double size, const osg::Matrix &offset, bool vrml, bool isObjectMarker)
{
    auto m = getMarker(name);
    if(m) //update existing marker
    {
        if(m->getPattern() != pattern)
        {
            std::cerr << "Changing pattern of existing marker " << name << " from " << m->getPattern() << " to " << pattern << " is not supported" << std::endl;
        } else {
            m->updateData(size, offset, vrml);
            m->setObjectMarker(isObjectMarker);
        }
        return m;
    } else { //create a new marker
        auto mtm = std::unique_ptr<MarkerTrackingMarker>(new MarkerTrackingMarker(name, pattern, size, offset, vrml));
        m = markers.emplace(std::make_pair(name, std::move(mtm))).first->second.get();
        m->setObjectMarker(isObjectMarker);
        return m;
    }
}

void MarkerTracking::config()
{
    osg::Geode *geodevideo = new osg::Geode;
    MarkerTrackingNode *artnode;
    artnode = new MarkerTrackingNode(m_MarkerTrackingVariant);
    osg::StateSet *statesetBackgroundBin = new osg::StateSet();
    statesetBackgroundBin->setRenderBinDetails(-2, "RenderBin");
    statesetBackgroundBin->setNestRenderBins(false);
    artnode->setStateSet(statesetBackgroundBin);
    geodevideo->addDrawable(artnode);
    cover->getScene()->addChild(geodevideo);
    for(const auto &section : m_markerDatabase->sections())
    {
        auto pattId = m_markerDatabase->value<std::string>(section, "pattern"); 
        if(pattId)
        {
            auto mtm = std::unique_ptr<MarkerTrackingMarker>(new MarkerTrackingMarker(section));
            auto m = markers.emplace(std::make_pair(section, std::move(mtm))).first->second.get();
            
            if(m->isObjectMarker())
                objectMarkers.push_back(m);
        }
    }
}

MarkerTracking::~MarkerTracking()
{
    delete artTab;
    art = NULL;
}

MarkerTrackingMarker::~MarkerTrackingMarker()
{
    std::cerr << "destoying marker" << std::endl;
}

void MarkerTracking::update()
{
    artTab->setHidden(!isRunning());

    if (isRunning())
    {
        for(auto &m : markers)
        {
            float s = 1.0 / cover->getScale();
            MarkerTrackingMarker *marker = m.second.get();
            if (marker->displayQuad && marker->displayQuad->getState())
            {
                marker->markerQuad->setMatrix(osg::Matrix::scale(s, s, s));
                if (marker->lastVisible != marker->isVisible())
                {
                    if (marker->isVisible())
                    {
                        marker->setColor(0, 1, 0);
                    }
                    else
                    {
                        marker->setColor(1, 0, 0);
                    }
                    marker->lastVisible = marker->isVisible();
                }
            }
        }
        int numVisible = 0;
        bool doCallibration = false;
        osg::Matrix tmpMat;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                tmpMat(i, j) = 0;
        for(auto currentMarker : objectMarkers)
        {
            assert(currentMarker);
            if (currentMarker->isVisible())
            {
                if (currentMarker->calibrate && currentMarker->calibrate->getState())
                {
                    doCallibration = true;
                }
                else
                {
                    numVisible++;
                    osg::Matrix MarkerPos; // marker position in camera coordinate system
                    MarkerPos = currentMarker->getMarkerTrans();
                    osg::Matrix tmpMat2, leftCameraTrans;
                    leftCameraTrans = VRViewer::instance()->getViewerMat();
                    if (coVRConfig::instance()->stereoState())
                    {
                        leftCameraTrans.preMult(osg::Matrix::translate(-(VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                    }
                    else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_LEFT)
                    {
                        leftCameraTrans.preMult(osg::Matrix::translate(-(VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                    }
                    else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_RIGHT)
                    {
                        leftCameraTrans.preMult(osg::Matrix::translate((VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                    }
                    tmpMat = MarkerPos * leftCameraTrans;
                }
                //break;
            }
            if (numVisible)
            {
                if (doCallibration)
                {
                    for (auto currentMarker : objectMarkers)
                    {
                        if (currentMarker && currentMarker->isVisible() && currentMarker->calibrate->getState())
                        {
                            if (currentMarker->numCalibSamples < 100)
                            {
                                if (currentMarker->numCalibSamples == 0)
                                {
                                    for (int i = 0; i < 4; i++)
                                        for (int j = 0; j < 4; j++)
                                            currentMarker->matrixSumm(i, j) = 0;
                                }
                                currentMarker->numCalibSamples++;

                                osg::Matrix MarkerPos; // marker position in camera coordinate system
                                MarkerPos = currentMarker->getMarkerTrans();
                                osg::Matrix tmpMat2, leftCameraTrans;
                                leftCameraTrans = VRViewer::instance()->getViewerMat();
                                if (coVRConfig::instance()->stereoState())
                                {
                                    leftCameraTrans.preMult(osg::Matrix::translate(-(VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                                }
                                else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_LEFT)
                                {
                                    leftCameraTrans.preMult(osg::Matrix::translate(-(VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                                }
                                else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_RIGHT)
                                {
                                    leftCameraTrans.preMult(osg::Matrix::translate((VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                                }
                                tmpMat2 = MarkerPos * osg::Matrix::inverse(tmpMat * osg::Matrix::inverse(leftCameraTrans));
                                //OT = Inv(Ctrans *offset) * leftCameraTrans

                                //Inv(Ctrans) * Inv(OT * Inv(leftCameraTrans)) = offset

                                for (int i = 0; i < 4; i++)
                                    for (int j = 0; j < 4; j++)
                                        currentMarker->matrixSumm(i, j) += (tmpMat2(i, j) / 100.0);
                            }
                            else
                            {
                                currentMarker->stopCalibration();
                                currentMarker->numCalibSamples = 0;
                                currentMarker->setOffset(currentMarker->matrixSumm);
                                if (MarkerTracking::instance()->arInterface)
                                {
                                    MarkerTracking::instance()->arInterface->updateMarkerParams();
                                }
                            }
                        }
                    }
                }

                cover->getObjectsXform()->setMatrix(tmpMat);
                coVRCollaboration::instance()->SyncXform();
            }
            if (coVRMSController::instance()->isCluster())
            {
                osg::Matrix tmpMat;
                if (coVRMSController::instance()->isMaster())
                {
                    tmpMat = cover->getObjectsXform()->getMatrix();
                    coVRMSController::instance()->syncData((char *)&tmpMat, sizeof(tmpMat));
                }
                else
                {

                    coVRMSController::instance()->syncData((char *)&tmpMat, sizeof(tmpMat));
                    cover->getObjectsXform()->setMatrix(tmpMat);
                }
            }
        }
    }
    if (remoteAR && remoteAR->usesIRMOS() && remoteAR->isReceiver())
    {
        //Check if we are slave or master
        if (coVRMSController::instance()->isMaster())
        {
            //Receive package
            ClientConnection *client = this->remoteAR->getIRMOSClient();
            if (client && client->check_for_input())
            {
                //We have a frame message
                //Read it and send frame data to RemoteAR portion of
                //MarkerTracking plugin

                //client->recv_msg_fast(&msg);
                client->recv_msg(&msg);
                //std::cerr << "MarkerTracking::update(): Size of received message: " << msg.length << std::endl;

                //Message received we need to distribute it to the slaves
                coVRMSController::instance()->sendSlaves(&msg);
            }
        }
        else
        {
            //We are a slave, we need to read the message from the master
            coVRMSController::instance()->readMaster(&msg);
        }

        //Need to strip of the message type first
        //Better to check if it really is for us
        if (msg.data.data() != nullptr)
        {
            char *datablock = &msg.data.accessData()[strlen(&msg.data.data()[1]) + 2];
            if (strcmp(&(msg.data.data()[1]), "AR_VIDEO_FRAME") == 0)
            {
                remoteAR->receiveImage(datablock);
            }
        }
    }
}

void MarkerTracking::changeObjectMarker(MarkerTrackingMarker *m)
{
    auto om = std::find_if(objectMarkers.begin(), objectMarkers.end(), [m](MarkerTrackingMarker *otherM){
        return m == otherM;
    });
    if(m->isObjectMarker() && om == objectMarkers.end())
    {
        objectMarkers.push_back(m);
    }
    else if(!m->isObjectMarker() && om != objectMarkers.end())
    {
        objectMarkers.erase(om);
    }
}
bool MarkerTracking::isRunning()
{
    return running;
}

opencover::config::File &MarkerTracking::markerDatabase()
{
    return *m_markerDatabase;
}

void MarkerTrackingMarker::matToEuler(const osg::Matrix &mat)
{
    m_offset = mat;
    coCoord offsetCoord(m_offset);
    std::array<double, 3> xyz, hpr;
    for (size_t i = 0; i < 3; i++)
    {
            xyz[i] = offsetCoord.xyz[i];
            hpr[i] = offsetCoord.hpr[i];
    }
    m_xyz->setValue(xyz);
    m_hpr->setValue(hpr);
}

osg::Matrix MarkerTrackingMarker::eulerToMat() const
{
    osg::Matrix offMat;
    coCoord offsetCoord;
    auto xyz = m_xyz->getValue();
    auto hpr = m_hpr->getValue();
    for (size_t i = 0; i < 3; i++)
    {
        offsetCoord.xyz[i] = xyz[i];
        offsetCoord.hpr[i] = hpr[i];
    }
    offsetCoord.makeMat(offMat);
    return offMat;
}

void MarkerTrackingMarker::setOffset(const osg::Matrix &mat)
{
    matToEuler(mat);
    m_vrmlToPf->setValue(false);

    osg::Matrix tmpMat;
    tmpMat.makeScale(getSize(), getSize(), getSize());
    posSize->setMatrix(tmpMat * m_offset);
}

void MarkerTrackingMarker::stopCalibration()
{
    calibrate->setState(false);
    tabletEvent(calibrate);
}

void MarkerTrackingMarker::updateData(double markerSize, const osg::Matrix& m, bool vrmlToOsg)
{
    matToEuler(m);
	m_size->setValue(markerSize);

	if (m_vrmlToPf->getValue())
		m_offset.preMult(PfToOpenGLMatrix);
	osg::Matrix mat;
	mat.makeScale(getSize(), getSize(), getSize());
	mat = mat * m_offset;

    m_vrmlToPf->setValue(false);
    posSize->setMatrix(mat);

	if (MarkerTracking::instance()->arInterface)
	{
		MarkerTracking::instance()->arInterface->updateMarkerParams();
	}
}

MarkerTrackingMarker::MarkerTrackingMarker(const std::string &configName, const std::string &pattern, double markerSize, const osg::Matrix& m, bool vrmlToOsg)
{
    createUiandConfigValues(configName);
    m_pattID->setValue(pattern);
    m_size->setValue(markerSize);
    m_vrmlToPf->setValue(vrmlToOsg);
    matToEuler(m);
    init();
}

MarkerTrackingMarker::MarkerTrackingMarker(const std::string &name)
{
    createUiandConfigValues(name);
    m_offset = eulerToMat();
    init();
}

void MarkerTrackingMarker::init()
{
    m_cameraTransform.makeIdentity();
	if (m_vrmlToPf->getValue())
		m_offset.preMult(PfToOpenGLMatrix);
	osg::Matrix mat;
	mat.makeScale(getSize(), getSize(), getSize());
	mat = mat * m_offset;

	string pattern = getPattern();
	if (cover->debugLevel(3))
		cerr << "MarkerTrackingMarker::MarkerTrackingMarker(): Loading pattern with ID = " << pattern.c_str() << endl;
	if (MarkerTracking::instance()->arInterface)
	{
        pattern = MarkerTracking::instance()->arInterface->loadPattern(pattern);
        if (!pattern.empty())
        {
            m_pattID->setValue(pattern);
        }
	}

	
    float ZPOS = 0.0f;
    float WIDTH = 1.0f;
    float HEIGHT = 1.0f;

    geom = new osg::Geometry();

    osg::Vec3Array* vertices = new osg::Vec3Array(4);
    // bottom left
    (*vertices)[0].set(-WIDTH / 2.0, ZPOS, -HEIGHT / 2.0);
    // bottom right
    (*vertices)[1].set(WIDTH / 2.0, ZPOS, -HEIGHT / 2.0);
    // top right
    (*vertices)[2].set(WIDTH / 2.0, ZPOS, HEIGHT / 2.0);
    // top left
    (*vertices)[3].set(-WIDTH / 2.0, ZPOS, HEIGHT / 2.0);
    geom->setVertexArray(vertices);

    osg::Vec3Array* normals = new osg::Vec3Array(1);
    (*normals)[0].set(0.0f, 0.0f, 1.0f);
    geom->setNormalArray(normals);
    geom->setNormalBinding(osg::Geometry::BIND_OVERALL);

    colors = new osg::Vec4Array(1);
    (*colors)[0].set(1.0, 0.0, 0.0, 1.0);
    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 4));

    osg::StateSet* stateset = geom->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    quadGeode = new osg::Geode();
    quadGeode->addDrawable(geom);
    posSize = new osg::MatrixTransform();
    posSize->addChild(quadGeode);
    posSize->setMatrix(mat);
    markerQuad = new osg::MatrixTransform();
    markerQuad->addChild(posSize);
    numCalibSamples = 0;

	if (MarkerTracking::instance()->arInterface)
	{
		cerr << "MarkerTrackingMarker::MarkerTrackingMarker(): init size pattern with ID = " << getPattern() << endl;
		MarkerTracking::instance()->arInterface->updateMarkerParams();
	}
}

void MarkerTrackingMarker::createUiandConfigValues(const std::string &configName)
{
    int pos = MarkerTracking::instance()->markers.size() + 1;
    
    m_toggleConfigOff = new coTUIButton(configName,  MarkerTracking::instance()->artTab->getID());
    m_toggleConfigOff->setPos(0, pos);
    m_toggleConfigOff->setEventListener(this);

    m_layoutGroup = new coTUIFrame(configName,  MarkerTracking::instance()->artTab->getID());
    m_layoutGroup->setPos(1, pos+1);
    m_layoutGroup->setHidden(true);
    m_configLabel = std::make_unique<coTUILabel>(configName,m_layoutGroup->getID());
    m_configLabel->setLabel(configName);
    m_configLabel->setPos(0,0);

    m_toggleConfigOn = new coTUIButton("hide",  m_layoutGroup->getID());
    m_toggleConfigOn->setEventListener(this);
    m_toggleConfigOn->setPos(0, 1);

    coTUIGroupBox *line1 = new coTUIGroupBox("",  m_layoutGroup->getID());
    line1->setPos(0, 2);
    coTUIGroupBox *line2 = new coTUIGroupBox("",  m_layoutGroup->getID());
    line2->setPos(0, 3);

    m_pattID = std::make_unique<covTUIEditField>(MarkerTracking::instance()->markerDatabase(), configName, "pattern",
                                                 line1, "");
    m_pattID->ui()->setPos(0, 0);
    m_pattID->ui()->setEnabled(false);

    m_size = std::make_unique<covTUIEditFloatField>(MarkerTracking::instance()->markerDatabase(), configName, "size", line1, 80);
    m_size->ui()->setPos(1, 0);
    m_size->ui()->setLabel("size");
    m_size->setUpdater([this](){updateMatrices();});

    m_pattGroup = std::make_unique<covTUIEditIntField>(MarkerTracking::instance()->markerDatabase(), configName, "group", line1, noMarkerGroup);
    m_pattGroup->ui()->setPos(2,0);
    m_pattGroup->ui()->setLabel("group");
    m_pattGroup->setUpdater([this](){updateMatrices();});

    m_objectMarker = std::make_unique<covTUIToggleButton>(MarkerTracking::instance()->markerDatabase(), configName, "objectMarker", line2, false);
    m_objectMarker->setUpdater([this](){
        MarkerTracking::instance()->changeObjectMarker(this);
        updateMatrices();});
    m_objectMarker->ui()->setPos(0, 0);


    m_vrmlToPf = std::make_unique<covTUIToggleButton>(MarkerTracking::instance()->markerDatabase(), configName, "vrml", line2, false);
    m_vrmlToPf->ui()->setPos(1, 0);
    m_vrmlToPf->setUpdater([this](){updateMatrices();});

    displayQuad = new coTUIToggleButton("displayQuad", line2->getID());
    displayQuad->setPos(2, 0);
    displayQuad->setEventListener(this);

    calibrate = new coTUIToggleButton("calibrate", line2->getID());
    calibrate->setPos(3, 0);
    calibrate->setEventListener(this);
    calibrate->setState(false);

    m_xyz = std::make_unique<covTUIEditFloatFieldVec3>(MarkerTracking::instance()->markerDatabase(), configName, "xyz", m_layoutGroup, std::array<double, 3>{0.0, 0.0, 0.0});
    m_xyz->box()->setPos(0, 4);
    m_xyz->setUpdater([this](){
            updateMatrices();
        });

    m_hpr = std::make_unique<covTUIEditFloatFieldVec3>(MarkerTracking::instance()->markerDatabase(), configName, "hpr", m_layoutGroup, std::array<double, 3>{0.0, 0.0, 0.0});
    m_hpr->box()->setPos(0, 5);
    m_hpr->setUpdater([this](){
            updateMatrices();
        });
    
}


void MarkerTrackingMarker::setColor(float r, float g, float b)
{
    (*colors)[0].set(r, g, b, 1.0);
	colors->dirty();
    geom->dirtyDisplayList();
}

double MarkerTrackingMarker::getSize() const
{
    return m_size->getValue();
}

std::string MarkerTrackingMarker::getPattern() const
{
    return m_pattID->getValue();
}

void MarkerTrackingMarker::updateMatrices()
{
    m_offset = eulerToMat();
    if (m_vrmlToPf->getValue()) 
        m_offset.preMult(PfToOpenGLMatrix);

    osg::Matrix mat;
    mat.makeScale(getSize(), getSize(), getSize());
    mat = mat * m_offset;
    if(posSize)
        posSize->setMatrix(mat);
    if (MarkerTracking::instance()->arInterface)
    {
        MarkerTracking::instance()->arInterface->updateMarkerParams();
    }
}

void MarkerTrackingMarker::tabletEvent(coTUIElement *tUIItem)
{
    if(tUIItem == m_toggleConfigOff || tUIItem == m_toggleConfigOn)
    {
        m_toggleConfigOff->setHidden(tUIItem == m_toggleConfigOff);
        m_layoutGroup->setHidden(tUIItem == m_toggleConfigOn);
        return;
    }
    else if (tUIItem == displayQuad)
    {
        if (displayQuad->getState())
            cover->getObjectsRoot()->addChild(markerQuad.get());
        else
            cover->getObjectsRoot()->removeChild(markerQuad.get());
    }
    else if (tUIItem == calibrate)
    {
        if(calibrate->getState())
        {
            m_oldpattGroup = m_pattGroup->getValue();        
            m_pattGroup->setValue(noMarkerGroup);
            osg::Matrix m;
            m.identity();
            setOffset(m);
        }
        else{
           m_pattGroup->setValue(m_oldpattGroup);
        }
    }
    updateMatrices();
}

const osg::Matrix &MarkerTrackingMarker::getCameraTrans()
{
    if (MarkerTracking::instance()->isRunning())
    {

        if ((MarkerTracking::instance()->arInterface) && !getPattern().empty() && MarkerTracking::instance()->arInterface->isVisible(this))
        {
            m_cameraTransform = MarkerTracking::instance()->arInterface->getMat(this);
            if (MarkerTracking::instance()->arInterface->isMarkerTracking())
            {
                osg::Vec3 trans;
                trans = m_cameraTransform.getTrans();
                m_cameraTransform.setTrans(0, 0, 0);

                osg::Matrix rotMat;
                rotMat.makeIdentity();
                rotMat(0, 0) = -1;
                rotMat(1, 1) = 0;
                rotMat(2, 2) = 0; // probably have to change this too (remove it)
                rotMat(1, 2) = 1;
                rotMat(2, 1) = -1;

                m_cameraTransform = rotMat * m_cameraTransform;

                osg::Matrix tmp;
                tmp = m_cameraTransform;
                m_cameraTransform.invert(tmp);

                tmp = m_cameraTransform;
                m_cameraTransform.preMult(osg::Matrix::translate(trans[0], trans[1], trans[2]));
                m_cameraTransform = m_cameraTransform * m_offset;
            }
            else
            {
                osg::Matrix CameraToOrigin;
                CameraToOrigin.makeTranslate(VRViewer::instance()->getViewerPos());
                m_cameraTransform.invert(m_cameraTransform); //*CameraToOrigin
                // offset is handeled by MultiMarker class this might have to be removed if multimarker class is used again
                m_cameraTransform = m_cameraTransform * m_offset;
            }

            //Ctrans.print(1,1,"Ctrans: ",stderr);
        }
    }
    return m_cameraTransform;
}

osg::Matrix MarkerTrackingMarker::getMarkerTrans()
{
    return osg::Matrix::inverse(getCameraTrans());
}

int MarkerTrackingMarker::getMarkerGroup() const
{
    return m_pattGroup->getValue();
}

bool MarkerTrackingMarker::isVisible() const
{
    if (MarkerTracking::instance()->isRunning() && MarkerTracking::instance()->arInterface)
        return MarkerTracking::instance()->arInterface->isVisible(this);
    return false;
}

bool MarkerTrackingMarker::isObjectMarker() const
{
    return m_objectMarker->getValue();
}

void MarkerTrackingMarker::setObjectMarker(bool o)
{
    m_objectMarker->setValue(o);
    MarkerTracking::instance()->changeObjectMarker(this);
}
