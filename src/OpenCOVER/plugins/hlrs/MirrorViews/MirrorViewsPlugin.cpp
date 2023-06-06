/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: MirrorViews Plugin (does nothing)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** Nov-01  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "MirrorViewsPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <osgViewer/Viewer>
#include <osg/Matrix>
#include <osg/MatrixTransform>

#include <OpenVRUI/osg/mathUtils.h>
#include <config/CoviseConfig.h>

#include "ip/IpEndpointName.h"

MirrorViewsPlugin::MirrorViewsPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "MirrorViewsPlugin::MirrorViewsPlugin\n");

    tuiMirrorTab = new coTUITab("MirrorPos", coVRTui::instance()->mainFolder->getID());
    tuiMirrorTab->setPos(0, 0);

    tuiPosX = new coTUIEditFloatField("tuiPosX", tuiMirrorTab->getID());
    tuiPosX->setValue(0.0);
    tuiPosX->setEventListener(this);
    tuiPosX->setPos(0, 0);

    tuiPosY = new coTUIEditFloatField("tuiPosY", tuiMirrorTab->getID());
    tuiPosY->setValue(0.0);
    tuiPosY->setEventListener(this);
    tuiPosY->setPos(1, 0);

    tuiPosZ = new coTUIEditFloatField("tuiPosZ", tuiMirrorTab->getID());
    tuiPosZ->setValue(0.0);
    tuiPosZ->setEventListener(this);
    tuiPosZ->setPos(2, 0);

    tuiOriH = new coTUIEditFloatField("tuiOriH", tuiMirrorTab->getID());
    tuiOriH->setValue(0.0);
    tuiOriH->setEventListener(this);
    tuiOriH->setPos(0, 1);

    tuiOriP = new coTUIEditFloatField("tuiOriP", tuiMirrorTab->getID());
    tuiOriP->setValue(0.0);
    tuiOriP->setEventListener(this);
    tuiOriP->setPos(1, 1);

    tuiOriR = new coTUIEditFloatField("tuiOriR", tuiMirrorTab->getID());
    tuiOriR->setValue(0.0);
    tuiOriR->setEventListener(this);
    tuiOriR->setPos(2, 1);

    mi.enabled = false;
    bool exists;
    if (covise::coCoviseConfig::isOn("enabled", "COVER.Plugin.MirrorViews", false, &exists))
    {
        mi.enabled = true;
        mi.pos[0] = covise::coCoviseConfig::getFloat("posX", "COVER.Plugin.MirrorViews", 0.0);
        mi.pos[1] = covise::coCoviseConfig::getFloat("posY", "COVER.Plugin.MirrorViews", 0.0);
        mi.pos[2] = covise::coCoviseConfig::getFloat("posZ", "COVER.Plugin.MirrorViews", 0.0);
        mi.ori[0] = covise::coCoviseConfig::getFloat("oriH", "COVER.Plugin.MirrorViews", 0.0);
        mi.ori[1] = covise::coCoviseConfig::getFloat("oriP", "COVER.Plugin.MirrorViews", 0.0);
        mi.ori[2] = covise::coCoviseConfig::getFloat("oriR", "COVER.Plugin.MirrorViews", 0.0);
    }

    VRViewer::instance()->overwriteViewAndProjectionMatrix(mi.enabled);
    numSlaves = coVRMSController::instance()->getNumSlaves();
    numScreens = 0;
    screens.reserve(numSlaves);
    mirrors = new mirrorInfo[numSlaves];
    for (int i = 0; i < numSlaves; i++)
    {
        if (coVRMSController::instance()->isMaster())
        {
            int nS;
            coVRMSController::instance()->readSlave(i, &nS, sizeof(nS));
            for (int n = 0; n < nS; n++)
            {
                screenStruct *s = new screenStruct;
                coVRMSController::instance()->readSlave(i, s, sizeof(screenStruct));
                screens.push_back(s);
            }
            coVRMSController::instance()->readSlave(i, &mirrors[i], sizeof(mirrorInfo));
            numScreens += nS;
        }
        else
        {
            int nS = coVRConfig::instance()->numScreens();
            coVRMSController::instance()->sendMaster(&nS, sizeof(nS));
            for (int n = 0; n < nS; n++)
            {
                coVRMSController::instance()->sendMaster(&(coVRConfig::instance()->screens[n]), sizeof(screenStruct));
            }
            coVRMSController::instance()->sendMaster(&mi, sizeof(mirrorInfo));
        }
    }
    packet = new osc::OutboundPacketStream(buffer, IP_MTU_SIZE);
    std::string hostname = covise::coCoviseConfig::getEntry("host", "COVER.Plugin.MirrorViews", "192.168.2.12", &exists);
    cerr << hostname << endl;
    udpSocket = new UdpTransmitSocket(IpEndpointName(hostname.c_str(), 11999));
}

// this is called if the plugin is removed at runtime
MirrorViewsPlugin::~MirrorViewsPlugin()
{
    fprintf(stderr, "MirrorViewsPlugin::~MirrorViewsPlugin\n");
    if (mi.enabled)
    {
        VRViewer::instance()->overwriteViewAndProjectionMatrix(false);
    }
    delete udpSocket;
    delete packet;
}

void MirrorViewsPlugin::tabletEvent(coTUIElement * /*tUIItem*/)
{
    mi.pos[0] = tuiPosX->getValue();
    mi.pos[1] = tuiPosY->getValue();
    mi.pos[2] = tuiPosZ->getValue();
    mi.ori[0] = tuiOriH->getValue();
    mi.ori[1] = tuiOriP->getValue();
    mi.ori[2] = tuiOriR->getValue();
}

void MirrorViewsPlugin::tabletPressEvent(coTUIElement * /*tUIItem*/)
{
    mi.pos[0] = tuiPosX->getValue();
    mi.pos[1] = tuiPosY->getValue();
    mi.pos[2] = tuiPosZ->getValue();
    mi.ori[0] = tuiOriH->getValue();
    mi.ori[1] = tuiOriP->getValue();
    mi.ori[2] = tuiOriR->getValue();
}

void MirrorViewsPlugin::tabletReleaseEvent(coTUIElement * /*tUIItem*/)
{
}

void
MirrorViewsPlugin::computeFrustum(osg::Vec3 viewerPos, float &h, float &p, float &r, float &top, float &left, float &right, float &bottom, screenStruct *screen)
{
    osg::Matrixf offsetMatRight;
    osg::Matrixf offsetMatLeft;
    osg::Matrix trans;
    osg::Matrix euler;
    osg::Matrix mat;
    //osg::Vec3 rightEye;

    float dx = screen->hsize;
    float dz = screen->vsize;

    osg::Vec3 hpr = screen->hpr;
    osg::Vec3 xyz = screen->xyz;

    // transform the screen to fit the xz-plane
    trans.makeTranslate(-xyz[0], -xyz[1], -xyz[2]);

    MAKE_EULER_MAT_VEC(euler, hpr);
    euler.invert(euler);

    mat.mult(trans, euler);

    euler.makeRotate(-coVRConfig::instance()->worldAngle(), osg::X_AXIS);
    //euler.invertN(euler);
    mat.mult(euler, mat);
    //cerr << "test" << endl;

    osg::Matrix viewMat = VRViewer::instance()->getViewerMat();
    osg::Matrix mirrorPosition;
    mirrorPosition.makeTranslate(mi.pos[0], mi.pos[1], mi.pos[2]);
    osg::Matrix mirrorOrientation;
    MAKE_EULER_MAT_VEC(mirrorOrientation, mi.ori);
    osg::Matrix mirrorTransform;
    mirrorTransform = mirrorOrientation * mirrorPosition;
    osg::Matrix invMirrorTransform;
    invMirrorTransform.invert(mirrorTransform);
    osg::Matrix mirror;
    mirror.makeIdentity();
    mirror(1, 1) = -1;
    osg::Matrix MirrorMat = invMirrorTransform * mirror * mirrorTransform;

    // mirror the eye positions on the mirror plane
    //viewerPos = MirrorMat.preMult(viewerPos);

    viewerPos = mat.preMult(viewerPos);

    // dist of right channel eye to screen (absolute)
    float rc_dist = -viewerPos[1];

    // parameter of right channel
    float rc_right = (dx / 2.0 - viewerPos[0]) / rc_dist;
    float rc_left = (dx / 2.0 + viewerPos[0]) / -rc_dist;
    float rc_top = (dz / 2.0 - viewerPos[2]) / rc_dist;
    float rc_bottom = (dz / 2.0 + viewerPos[2]) / -rc_dist;

    right = (atan(rc_right) / M_PI) * 180.0;
    left = -(atan(rc_left) / M_PI) * 180.0;
    top = (atan(rc_top) / M_PI) * 180.0;
    bottom = -(atan(rc_bottom) / M_PI) * 180.0;
    h = -hpr[0];
    p = hpr[1];
    r = hpr[2];
}

void
MirrorViewsPlugin::preFrame()
{

    if (mi.enabled)
    {
        osg::Vec3 xyz; // center position of the screen
        osg::Vec3 hpr; // orientation of the screen
        osg::Matrix mat, trans, euler; // xform screencenter - world origin
        osg::Matrixf offsetMat;
        osg::Vec3 leftEye, rightEye, middleEye; // transformed eye position
        float rc_dist, lc_dist, mc_dist; // dist from eye to screen for left&right chan
        float rc_left, rc_right, rc_bottom, rc_top; // parameter of right frustum
        float lc_left, lc_right, lc_bottom, lc_top; // parameter of left frustum
        float mc_left, mc_right, mc_bottom, mc_top; // parameter of middle frustum
        float n_over_d; // near over dist -> Strahlensatz
        float dx, dz; // size of screen

        //othEyesDirOffset; == hpr
        //osg::Vec3  rightEyePosOffset(0.0,0.0,0.0), leftEyePosOffset(0.0,0.0,0.0);

        osg::Matrixf offsetMatRight;
        osg::Matrixf offsetMatLeft;

        dx = coVRConfig::instance()->screens[0].hsize;
        dz = coVRConfig::instance()->screens[0].vsize;

        hpr = coVRConfig::instance()->screens[0].hpr;
        xyz = coVRConfig::instance()->screens[0].xyz;

        // transform the screen to fit the xz-plane
        trans.makeTranslate(-xyz[0], -xyz[1], -xyz[2]);

        MAKE_EULER_MAT_VEC(euler, hpr);
        euler.invert(euler);

        mat.mult(trans, euler);

        euler.makeRotate(-coVRConfig::instance()->worldAngle(), osg::X_AXIS);
        //euler.invertN(euler);
        mat.mult(euler, mat);
        //cerr << "test" << endl;

        rightEye.set(VRViewer::instance()->getSeparation() / 2.0, 0.0, 0.0);
        leftEye.set(-(VRViewer::instance()->getSeparation() / 2.0), 0.0, 0.0);
        middleEye.set(0.0, 0.0, 0.0);
        VRViewer::instance()->getViewerMat();
        osg::Matrix viewMat = VRViewer::instance()->getViewerMat();
        osg::Matrix mirrorPosition;
        mirrorPosition.makeTranslate(mi.pos[0], mi.pos[1], mi.pos[2]);
        osg::Matrix mirrorOrientation;
        MAKE_EULER_MAT_VEC(mirrorOrientation, mi.ori);
        osg::Matrix mirrorTransform;
        mirrorTransform = mirrorOrientation * mirrorPosition;
        osg::Matrix invMirrorTransform;
        invMirrorTransform.invert(mirrorTransform);
        osg::Matrix mirror;
        mirror.makeIdentity();
        mirror(1, 1) = -1;
        osg::Matrix MirrorMat = invMirrorTransform * mirror * mirrorTransform;

        // transform the left and right eye with this matrix
        rightEye = viewMat.preMult(rightEye);
        leftEye = viewMat.preMult(leftEye);
        middleEye = viewMat.preMult(middleEye);

        // mirror the eye positions on the mirror plane
        rightEye = MirrorMat.preMult(rightEye);
        leftEye = MirrorMat.preMult(leftEye);
        middleEye = MirrorMat.preMult(middleEye);

        rightEye = mat.preMult(rightEye);
        leftEye = mat.preMult(leftEye);
        middleEye = mat.preMult(middleEye);

        offsetMat = mat;

        // compute right frustum

        // dist of right channel eye to screen (absolute)
        rc_dist = -rightEye[1];
        lc_dist = -leftEye[1];
        mc_dist = -middleEye[1];

        // relation near plane to screen plane
        n_over_d = coVRConfig::instance()->nearClip() / rc_dist;

        // parameter of right channel
        rc_right = n_over_d * (dx / 2.0 - rightEye[0]);
        rc_left = -n_over_d * (dx / 2.0 + rightEye[0]);
        rc_top = n_over_d * (dz / 2.0 - rightEye[2]);
        rc_bottom = -n_over_d * (dz / 2.0 + rightEye[2]);

        // compute left frustum
        n_over_d = coVRConfig::instance()->nearClip() / lc_dist;
        lc_right = n_over_d * (dx / 2.0 - leftEye[0]);
        lc_left = -n_over_d * (dx / 2.0 + leftEye[0]);
        lc_top = n_over_d * (dz / 2.0 - leftEye[2]);
        lc_bottom = -n_over_d * (dz / 2.0 + leftEye[2]);

        // compute left frustum
        n_over_d = coVRConfig::instance()->nearClip() / mc_dist;
        mc_right = n_over_d * (dx / 2.0 - middleEye[0]);
        mc_left = -n_over_d * (dx / 2.0 + middleEye[0]);
        mc_top = n_over_d * (dz / 2.0 - middleEye[2]);
        mc_bottom = -n_over_d * (dz / 2.0 + middleEye[2]);

        coVRConfig::instance()->screens[0].rightProj.makeFrustum(rc_left, rc_right, rc_bottom, rc_top, coVRConfig::instance()->nearClip(), coVRConfig::instance()->farClip());
        coVRConfig::instance()->screens[0].leftProj.makeFrustum(lc_left, lc_right, lc_bottom, lc_top, coVRConfig::instance()->nearClip(), coVRConfig::instance()->farClip());
        coVRConfig::instance()->screens[0].camera->setProjectionMatrixAsFrustum(mc_left, mc_right, mc_bottom, mc_top, coVRConfig::instance()->nearClip(), coVRConfig::instance()->farClip());

        // take the normal to the plane as orientation this is (0,1,0)
        coVRConfig::instance()->screens[0].rightView.makeLookAt(osg::Vec3(rightEye[0], rightEye[1], rightEye[2]), osg::Vec3(rightEye[0], rightEye[1] + 1, rightEye[2]), osg::Vec3(0, 0, 1));
        coVRConfig::instance()->screens[0].rightView.preMult(offsetMat);

        coVRConfig::instance()->screens[0].leftView.makeLookAt(osg::Vec3(leftEye[0], leftEye[1], leftEye[2]), osg::Vec3(leftEye[0], leftEye[1] + 1, leftEye[2]), osg::Vec3(0, 0, 1));
        coVRConfig::instance()->screens[0].leftView.preMult(offsetMat);

        coVRConfig::instance()->screens[0].camera->setViewMatrix(offsetMat * osg::Matrix::lookAt(osg::Vec3(middleEye[0], middleEye[1], middleEye[2]), osg::Vec3(middleEye[0], middleEye[1] + 1, middleEye[2]), osg::Vec3(0, 0, 1)));
    }
    if (oldViewer != cover->getViewerMat().getTrans()) // update ImageDistortion
    {
        // mono only for now special case for FKFS fasi
        if (!coVRConfig::instance()->stereoState())
        {
            oldViewer = cover->getViewerMat().getTrans();
            if (coVRMSController::instance()->isMaster())
            {
                for (int i = 0; i < numSlaves; i++)
                {
                    float h;
                    float p;
                    float r;
                    float top;
                    float left;
                    float right;
                    float bottom;
                    computeFrustum(oldViewer, h, p, r, top, left, right, bottom, screens[i]);

                    /*      
/updateView (int32)projectorId (int32)viewpoint (float)x (float)y (float)z (float)pitch (float)heading (float)roll (float)top (float)bottom (float)left (float)right
/updateOpenWarp [channel]
/saveDatabase [projectorId]*/
                    packet->Clear();
                    std::cerr << "/updateView"
                              << " i " << i << " _ " << (int)0 << " x " << (float)oldViewer[0] << " y " << (float)oldViewer[1] << " z " << (float)oldViewer[2] << " p " << p << " h " << h << " r " << r << " top " << top << " bottom " << bottom << " left " << left << " right " << right << std::endl;
                    *packet << osc::BeginMessage("/updateView")
                            << i << (int)0 << (float)oldViewer[0] << (float)oldViewer[1] << (float)oldViewer[2] << p << h << r << top << bottom << left << right << osc::EndMessage;
                    udpSocket->Send(packet->Data(), packet->Size());
                }
                /*   packet->Clear();
               *packet << osc::BeginMessage( "/updateOpenWarp" )
                  << 0 << osc::EndMessage;
               udpSocket->Send( packet->Data(), packet->Size() );
               packet->Clear();
               *packet << osc::BeginMessage( "/saveDatabase" )
                  << 0 << osc::EndMessage;
               udpSocket->Send( packet->Data(), packet->Size() );*/
                { // master
                    float h;
                    float p;
                    float r;
                    float top;
                    float left;
                    float right;
                    float bottom;
                    computeFrustum(oldViewer, h, p, r, top, left, right, bottom, &coVRConfig::instance()->screens[0]);

                    /*     
               std::cerr <<  "/updateView" <<" i " << 100 <<" _ " <<  (int)0 <<" x " <<  (float)oldViewer[0] << " y " << (float)oldViewer[1] << " z " << (float)oldViewer[2] <<" p " <<  p << " h " << h << " r " << r << " top " << top << " bottom " << bottom << " left " << left << " right " << right << std::endl;
               packet->Clear();
               *packet << osc::BeginMessage( "/updateView" )
                  << i << (int)0 << (float)oldViewer[0] << (float)oldViewer[1] << (float)oldViewer[2] << p << h << r << top << bottom << left << right << osc::EndMessage;
               udpSocket->Send( packet->Data(), packet->Size() );
               packet->Clear();
               *packet << osc::BeginMessage( "/updateOpenWarp" )
                  << i << osc::EndMessage;
               udpSocket->Send( packet->Data(), packet->Size() );
               packet->Clear();
               *packet << osc::BeginMessage( "/saveDatabase" )
                  << i << osc::EndMessage;
               udpSocket->Send( packet->Data(), packet->Size() ); */
                }
            }
        }
    }
}

COVERPLUGIN(MirrorViewsPlugin)
