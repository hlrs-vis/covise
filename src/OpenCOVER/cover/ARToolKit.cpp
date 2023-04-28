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
#include "ARToolKit.h"
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

ARToolKit *ARToolKit::art = NULL;
ARToolKit::ARToolKit()
{
    assert(!art);
    art = this;
    artTab = new coTUITab("ARToolKit", coVRTui::instance()->mainFolder->getID());
    artTab->setHidden(true); // hide until a marker is added

}

ARToolKitNode::ARToolKitNode(std::string artoolkitVariant)
{
    theNode = this;
    setSupportsDisplayList(false);
    std::string configPath = "COVER.Plugin.";
    configPath += artoolkitVariant;
    std::string entry = configPath + "DisplayVideo";
    displayVideo = coCoviseConfig::isOn(entry, true);
    entry = configPath + "RenderTextures";
    renderTextures = coCoviseConfig::isOn(entry, true);
    m_artoolkitVariant = artoolkitVariant;
}

ARToolKitNode::~ARToolKitNode()
{
    theNode = NULL;
}

ARToolKitNode *ARToolKitNode::theNode = NULL;

/** Clone the type of an object, with Object* return type.
Must be defined by derived classes.*/
osg::Object *ARToolKitNode::cloneType() const
{
    return new ARToolKitNode(m_artoolkitVariant);
}

/** Clone the an object, with Object* return type.
Must be defined by derived classes.*/
osg::Object *ARToolKitNode::clone(const osg::CopyOp &) const
{
    return new ARToolKitNode(m_artoolkitVariant);
}

void ARToolKitNode::drawImplementation(osg::RenderInfo &renderInfo) const
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
                    cerr << "ARToolKitNode::drawImplementation: unknown stereo mode" << endl;
                    break;
                }
            }
        }
    }

    GLint viewport[4]; // OpenGL viewport information (position and size)
    if (ARToolKit::instance()->testImage)
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
        if (ARToolKit::instance()->videoWidth > 0)
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

                if (ARToolKit::instance()->flipH)
                {
                    xPos *= -1;
                }

                if ((ARToolKit::instance()->stereoVideo) && (rightVideo))
                {
                    if (ARToolKit::instance()->videoDataRight)
                    {
                        glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA8, ARToolKit::instance()->videoWidth, ARToolKit::instance()->videoHeight, 0, ARToolKit::instance()->videoMode, GL_UNSIGNED_BYTE, ARToolKit::instance()->videoDataRight);
                        if (ARToolKit::instance()->videoMirrorRight)
                        {
                            xPos *= -1;
                            yPos *= -1;
                        }
                    }
                }
                else
                {
                    if (ARToolKit::instance()->videoData)
                    {
                        glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA8, ARToolKit::instance()->videoWidth, ARToolKit::instance()->videoHeight, 0, ARToolKit::instance()->videoMode, GL_UNSIGNED_BYTE, ARToolKit::instance()->videoData);
                        if (ARToolKit::instance()->videoMirrorLeft)
                        {
                            xPos *= -1;
                            yPos *= -1;
                        }
                    }
                }

                glBegin(GL_QUADS);
                {
                    glTexCoord2f(0, ARToolKit::instance()->videoHeight);
                    glVertex2f(-xPos, -yPos);
                    glTexCoord2f(ARToolKit::instance()->videoWidth, ARToolKit::instance()->videoHeight);
                    glVertex2f(xPos, -yPos);
                    glTexCoord2f(ARToolKit::instance()->videoWidth, 0);
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
                if (ARToolKit::instance()->flipH)
                {
                    ysize *= -1;
                    yPos *= -1;
                }
                glPixelZoom(xsize / ARToolKit::instance()->videoWidth, -ysize / ARToolKit::instance()->videoHeight);
                if ((ARToolKit::instance()->stereoVideo) && (rightVideo))
                {
                    if (ARToolKit::instance()->videoDataRight)
                    {
                        if (ARToolKit::instance()->videoMirrorRight)
                        {
                            glPixelZoom(-xsize / ARToolKit::instance()->videoWidth, ysize / ARToolKit::instance()->videoHeight);
                            yPos *= -1;
                            xPos *= -1;
                        }
                        glRasterPos2f(xPos, yPos);
                        glDrawPixels(ARToolKit::instance()->videoWidth, ARToolKit::instance()->videoHeight, ARToolKit::instance()->videoMode, GL_UNSIGNED_BYTE, ARToolKit::instance()->videoDataRight);
                        if (ARToolKit::instance()->videoMirrorRight)
                        {
                            glPixelZoom(xsize / ARToolKit::instance()->videoWidth, -ysize / ARToolKit::instance()->videoHeight);
                        }
                    }
                }
                else
                {
                    if (ARToolKit::instance()->videoData)
                    {
                        if (ARToolKit::instance()->videoMirrorLeft)
                        {
                            glPixelZoom(-xsize / ARToolKit::instance()->videoWidth, ysize / ARToolKit::instance()->videoHeight);
                            yPos *= -1;
                            xPos *= -1;
                        }
                        glRasterPos2f(xPos, yPos);
                        //if(ARToolKit::instance()->videoData)
                        //cerr << "x " << (long long)ARToolKit::instance()->videoData << " content: " << (int)ARToolKit::instance()->videoData[100] << endl;
                        glDrawPixels(ARToolKit::instance()->videoWidth, ARToolKit::instance()->videoHeight, ARToolKit::instance()->videoMode, GL_UNSIGNED_BYTE, ARToolKit::instance()->videoData);
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

ARToolKit *ARToolKit::instance()
{
    if (art == NULL)
        art = new ARToolKit();
    return art;
}

void ARToolKit::config()
{

    osg::Geode *geodevideo = new osg::Geode;
    ARToolKitNode *artnode;
    artnode = new ARToolKitNode(m_artoolkitVariant);
    osg::StateSet *statesetBackgroundBin = new osg::StateSet();
    statesetBackgroundBin->setRenderBinDetails(-2, "RenderBin");
    statesetBackgroundBin->setNestRenderBins(false);
    artnode->setStateSet(statesetBackgroundBin);
    geodevideo->addDrawable(artnode);
    cover->getScene()->addChild(geodevideo);
    doMerge = coCoviseConfig::isOn("COVER.Plugin.ARToolKit.MergeMarkers", false);
    if (coCoviseConfig::isOn("COVER.Plugin.ARToolKit.TrackObjects", false))
    {
        objTracking = true;
        char configName[100];
        std::string pattern;
        ARToolKitMarker *objMarker = NULL;
        do
        {

            sprintf(configName, "ObjectMarker%d", (int)objectMarkers.size());

            std::string entry = std::string("COVER.Plugin.ARToolKit.Marker:") + configName + std::string(".Pattern");
            pattern = coCoviseConfig::getEntry(entry);
            if (!pattern.empty())
            {
                objMarker = new ARToolKitMarker(configName);
                objMarker->setObjectMarker(true);
                objectMarkers.push_back(objMarker);
            }
        } while (!pattern.empty());

        if (objectMarkers.size() == 0)
        {
            pattern = coCoviseConfig::getEntry("COVER.Plugin.ARToolKit.Marker:ObjectMarker.Pattern");
            if (!pattern.empty())
            {
                objMarker = new ARToolKitMarker("ObjectMarker");
                objMarker->setObjectMarker(true);
                if (objMarker)
                {
                    objectMarkers.push_back(objMarker);
                }
            }
        }
    }
}

ARToolKit::~ARToolKit()
{
    delete artTab;
    art = NULL;
}

void ARToolKit::update()
{
    artTab->setHidden(markers.empty());

    if (isRunning())
    {
        std::list<ARToolKitMarker *>::iterator it;
        for (it = markers.begin(); it != markers.end(); it++)
        {
            float s = 1.0 / cover->getScale();
            ARToolKitMarker *marker = (*it);
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
        if (objTracking)
        {
            int numVisible = 0;
            bool doCallibration = false;
            osg::Matrix tmpMat;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    tmpMat(i, j) = 0;
            for (list<ARToolKitMarker *>::const_iterator it = objectMarkers.begin();
                 it != objectMarkers.end();
                 it++)
            {
                ARToolKitMarker *currentMarker = *it;
                if (currentMarker)
                {
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
                            if (doMerge)
                            {
                                tmpMat2 = MarkerPos * leftCameraTrans;
                                for (int i = 0; i < 4; i++)
                                    for (int j = 0; j < 4; j++)
                                        tmpMat(i, j) += tmpMat2(i, j);
                            }
                            else
                            {
                                tmpMat = MarkerPos * leftCameraTrans;
                            }
                        }
                        //break;
                    }
                }
            }
            if (numVisible)
            {
                if (doMerge)
                {

                    for (int i = 0; i < 4; i++)
                        for (int j = 0; j < 4; j++)
                            tmpMat(i, j) /= numVisible;
                }
                if (doCallibration)
                {
                    for (list<ARToolKitMarker *>::const_iterator it = objectMarkers.begin();
                         it != objectMarkers.end();
                         it++)
                    {
                        ARToolKitMarker *currentMarker = *it;
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
                                if (ARToolKit::instance()->arInterface)
                                {
                                    ARToolKit::instance()->arInterface->updateMarkerParams();
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
                //ARToolkit plugin

                //client->recv_msg_fast(&msg);
                client->recv_msg(&msg);
                //std::cerr << "ARToolKit::update(): Size of received message: " << msg.length << std::endl;

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

void ARToolKit::addMarker(ARToolKitMarker *m)
{
    if(m->isObjectMarker())
        objectMarkers.push_back(m);
}
bool ARToolKit::isRunning()
{
    return running;
}

void ARToolKitMarker::matToEuler(const osg::Matrix &mat)
{
    offset = mat;
    coCoord offsetCoord(offset);
    for (size_t i = 0; i < m_transform.size(); i++)
    {
        if(i < 3)
            m_transform[i].edit->setValue(offsetCoord.xyz[i]);
        else
            m_transform[i].edit->setValue(offsetCoord.hpr[i - 3]);
    }
}

osg::Matrix ARToolKitMarker::eulerToMat() const
{
    osg::Matrix offMat;
    coCoord offsetCoord;
    for (size_t i = 0; i < m_transform.size(); i++)
    {
        i < 3 ? offsetCoord.xyz[i] = m_transform[i].edit->getValue() : offsetCoord.hpr[i-3] = m_transform[i].edit->getValue();
    }
    offsetCoord.makeMat(offMat);
    return offMat;
}

void ARToolKitMarker::setOffset(osg::Matrix &mat)
{
    matToEuler(mat);
    vrmlToPf->setState(false);

    osg::Matrix tmpMat;
    tmpMat.makeScale(getSize(), getSize(), getSize());
    posSize->setMatrix(tmpMat * offset);
}

void ARToolKitMarker::stopCalibration()
{
    calibrate->setState(false);
    tabletEvent(calibrate);
}

void ARToolKitMarker::updateData(double markerSize, osg::Matrix& m, osg::Matrix& hostMat, bool vrmlToOsg)
{
	ARToolKitMarker* existingMarker = nullptr;
	for (const auto& i : ARToolKit::instance()->markers)
	{
		if (i!=this && i->getPattern() == getPattern())
		{
			existingMarker = i;
			break;
		}
	}
    matToEuler(m);
	size->setValue(markerSize);

	if (vrmlToPf->getState())
		offset.preMult(PfToOpenGLMatrix);
	osg::Matrix mat;
	mat.makeScale(getSize(), getSize(), getSize());
	mat = mat * offset;

	if (!existingMarker)
	{
		vrmlToPf->setState(false);
	}
	else
	{
		existingMarker->m_transform = m_transform;
		existingMarker->vrmlToPf->setState(vrmlToPf->getState());
		existingMarker->size->setValue(getSize());
		existingMarker->offset = offset;

		existingMarker->posSize->setMatrix(mat);
	}

	if (ARToolKit::instance()->arInterface)
	{
		ARToolKit::instance()->arInterface->updateMarkerParams();
	}
}

int generateMarkerGroupFromName(const std::string& name)
{
    static std::set<std::string> groups;
    size_t last_index = name.find_last_not_of("0123456789");
    string groupName = name.substr(0, last_index + 1);
    if(groupName != "ObjectMarker") //for now only use ObjectMarker as multimarker
        groupName = name;
    auto result = groups.insert(groupName);
    return std::distance(groups.begin(), result.first);
}

ARToolKitMarker::Coord::Coord(const std::string &name,  coTUIGroupBox* group)
:name(name)
, edit(new coTUIEditFloatField(name, group->getID()))
{}

float ARToolKitMarker::Coord::value() const
{
    return edit->getValue();
}

ARToolKitMarker::Coord &ARToolKitMarker::Coord::operator=(const ARToolKitMarker::Coord &other)
{
    if(!name.empty())
    {
        assert(other.name == name);
        edit->setValue(other.edit->getValue());

    } else  
    {
        edit = other.edit;
        name = other.name;
    }
    return *this;
}

std::string getConfigEntryName(const std::string &name, const std::string &val)
{
    std::stringstream entry;
    std::string arToolKitVariant = ARToolKit::instance()->m_artoolkitVariant;
    entry << "COVER.Plugin." << arToolKitVariant << ".Marker:" << name << "." << val;
    return entry.str();
}

int getPatternIdFromConfig(const std::string &name)
{
    auto pattern = coCoviseConfig::getEntry("value", getConfigEntryName(name, "Pattern"), name);
    return std::stoi(pattern);
}

double getMarkerSizeFromConfig(const std::string &name)
{
    return coCoviseConfig::getFloat(getConfigEntryName(name, "Size"), 80.0);
}

osg::Matrix getMarkerOffsetFromConfig(const std::string &name)
{
    auto e = getConfigEntryName(name, "Offset");
    std::string line = coCoviseConfig::getEntry("x", e);
    osg::Matrix offset;
    if (line.empty())
    {
        std::cerr << e << " configuration missing" << std::endl;
        offset.makeIdentity();
    }
    else
    {
        coCoord offsetCoord;
        const std::array<const char*, 6> coordNames = {"x","y", "z","h", "p","r"};
        for (size_t i = 0; i < coordNames.size(); i++)
        {
            auto val = coCoviseConfig::getFloat(coordNames[i], e, 0.0f);
            i < 3 ? offsetCoord.xyz[i] = val : offsetCoord.hpr[i-3] = val;
        }
        //offset.makeCoord(&offsetCoord);
        offsetCoord.makeMat(offset);
    }
    return offset;
}

float getVrmlToOsgFlagFromConfig(const std::string &name)
{
    return coCoviseConfig::isOn(getConfigEntryName(name, "VrmlToPf"), false);
}

ARToolKitMarker::ARToolKitMarker(const std::string &configName,int MarkerID, double markerSize, const osg::Matrix& m, bool vrmlToOsg)
{
	pattGroup = generateMarkerGroupFromName(configName);
    ARToolKitMarker* existingMarker = nullptr;
	for (const auto &i : ARToolKit::instance()->markers)
	{
		if (i->getPattern() == MarkerID)
		{
			existingMarker = i;
			break;
		}
	}
    createUi(configName);
    size->setValue(markerSize);
    ARToolKit::instance()->markers.push_back(this);
    matToEuler(m);

	Ctrans.makeIdentity();
	string pattern = std::to_string(MarkerID);


	OpenGLToOSGMatrix.makeRotate(M_PI / 2.0, 1, 0, 0);
	PfToOpenGLMatrix.makeRotate(-M_PI / 2.0, 1, 0, 0);

	vrmlToPf->setState(vrmlToOsg);
	if (vrmlToPf->getState())
		offset.preMult(PfToOpenGLMatrix);
	osg::Matrix mat;
	mat.makeScale(getSize(), getSize(), getSize());
	mat = mat * offset;

	if (cover->debugLevel(3))
		cerr << "ARToolKitMarker::ARToolKitMarker(): Loading pattern with ID = " << pattern.c_str() << endl;
	if (ARToolKit::instance()->arInterface)
	{
		pattID->setValue(ARToolKit::instance()->arInterface->loadPattern(pattern.c_str()));
	}
	if (getPattern() < 0)
	{
		pattID->setValue(atoi(pattern.c_str()));
		if (getPattern() < 0)
		{
            std::cerr << "pattern load error for "  << pattern << "!!" << std::endl;
            pattID->setValue(-1);

		}
	}
	if (!existingMarker)
	{
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
	}
	else
	{
        setObjectMarker(existingMarker->isObjectMarker());
        existingMarker->m_transform = m_transform;
        existingMarker->vrmlToPf->setState(vrmlToPf->getState());
        existingMarker->size->setValue(getSize());
        existingMarker->posSize->setMatrix(mat);
	}

	if (ARToolKit::instance()->arInterface)
	{
		cerr << "ARToolKitMarker::ARToolKitMarker(): init size pattern with ID = " << pattern.c_str() << endl;
		ARToolKit::instance()->arInterface->updateMarkerParams();
	}
}
ARToolKitMarker::ARToolKitMarker(const std::string &name)
:ARToolKitMarker(
    name,
    getPatternIdFromConfig(name), 
    getMarkerSizeFromConfig(name), 
    getMarkerOffsetFromConfig(name), 
    getVrmlToOsgFlagFromConfig(name))
{
}

void ARToolKitMarker::createUi(const std::string &configName)
{
    int pos = ARToolKit::instance()->markers.size();
    coTUIGroupBox *box = new coTUIGroupBox(configName,  ARToolKit::instance()->artTab->getID());
    box->setPos(0, pos);
    pattID = new coTUIEditFloatField(configName + "_pattern", box->getID(), -1);
    std::array<std::string, 6> coords = { "posX" , "posY", "posZ", "rotH", "rotP", "rotR"};
    for (size_t i = 0; i < m_transform.size(); i++)
    {
        m_transform[i] = Coord(coords[i], box);
        m_transform[i].edit->setEventListener(this);
        i < 3 ? m_transform[i].edit->setPos(i, 1) : m_transform[i].edit->setPos(i - 3, 2);
    }
    
    vrmlToPf = new coTUIToggleButton("VRML", box->getID());
    size = new coTUIEditFloatField("size", box->getID());
    displayQuad = new coTUIToggleButton("displayQuad", box->getID());
    calibrate = new coTUIToggleButton("calibrate", box->getID());
    size->setEventListener(this);
    vrmlToPf->setEventListener(this);
    displayQuad->setEventListener(this);
    calibrate->setEventListener(this);

    pattID->setPos(0, 0);
    size->setPos(1, 0);
    vrmlToPf->setPos(2, 0);
    displayQuad->setPos(3, 0);
    calibrate->setPos(4, 0);
    calibrate->setState(false);
}


void ARToolKitMarker::setColor(float r, float g, float b)
{
    (*colors)[0].set(r, g, b, 1.0);
	colors->dirty();
    geom->dirtyDisplayList();
}

double ARToolKitMarker::getSize() CONST
{
    return size->getValue();
}

int ARToolKitMarker::getPattern() const
{
    return pattID->getValue();
}

void ARToolKitMarker::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == displayQuad)
    {
        if (displayQuad->getState())
            cover->getObjectsRoot()->addChild(markerQuad.get());
        else
            cover->getObjectsRoot()->removeChild(markerQuad.get());
    }
    if (tUIItem == calibrate)
    {
        if(calibrate->getState())
        {
            oldpattGroup = pattGroup;        
            pattGroup = generateMarkerGroupFromName("calibrationInternal");
            osg::Matrix m;
            m.identity();
            setOffset(m);
        }
        else{
           pattGroup = oldpattGroup;
        }
    }
    offset = eulerToMat();
    if (vrmlToPf->getState()) 
        offset.preMult(PfToOpenGLMatrix);

    osg::Matrix mat;
    mat.makeScale(getSize(), getSize(), getSize());
    mat = mat * offset;
    posSize->setMatrix(mat);
    if (ARToolKit::instance()->arInterface)
    {
        ARToolKit::instance()->arInterface->updateMarkerParams();
    }
}

ARToolKitMarker::~ARToolKitMarker()
{
}

osg::Matrix &ARToolKitMarker::getCameraTrans()
{
    if (ARToolKit::instance()->isRunning())
    {

        if ((ARToolKit::instance()->arInterface) && (getPattern() >= 0) && ARToolKit::instance()->arInterface->isVisible(getPattern()))
        {
            Ctrans = ARToolKit::instance()->arInterface->getMat(getPattern(), pattCenter, getSize(), pattTrans);
            if (ARToolKit::instance()->arInterface->isARToolKit())
            {
                osg::Vec3 trans;
                trans = Ctrans.getTrans();
                Ctrans.setTrans(0, 0, 0);

                osg::Matrix rotMat;
                rotMat.makeIdentity();
                rotMat(0, 0) = -1;
                rotMat(1, 1) = 0;
                rotMat(2, 2) = 0; // probably have to change this too (remove it)
                rotMat(1, 2) = 1;
                rotMat(2, 1) = -1;

                Ctrans = rotMat * Ctrans;

                osg::Matrix tmp;
                tmp = Ctrans;
                Ctrans.invert(tmp);

                tmp = Ctrans;
                Ctrans.preMult(osg::Matrix::translate(trans[0], trans[1], trans[2]));
                Ctrans = Ctrans * offset;
            }
            else
            {
                osg::Matrix CameraToOrigin;
                CameraToOrigin.makeTranslate(VRViewer::instance()->getViewerPos());
                Ctrans.invert(Ctrans); //*CameraToOrigin
                // offset is handeled by MultiMarker class this might have to be removed if multimarker class is used again
                Ctrans = Ctrans * offset;
            }

            //Ctrans.print(1,1,"Ctrans: ",stderr);
        }
    }
    return Ctrans;
}

osg::Matrix &ARToolKitMarker::getMarkerTrans()
{
    Mtrans.invert(getCameraTrans());

    //Mtrans.print(1,1,"Mtrans: ",stderr);

    return Mtrans;
}

int ARToolKitMarker::getMarkerGroup() const
{
    return pattGroup;
}

bool ARToolKitMarker::isVisible() const
{
    if (ARToolKit::instance()->isRunning())
    {

        if (ARToolKit::instance()->arInterface)
        {
            if (getPattern() >= 0)
                return ARToolKit::instance()->arInterface->isVisible(getPattern());
            else
                return false;
        }
    }
    return false;
}
