/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
/****************************************************************************\
 **                                                            (C)2010 HLRS  **
 **                                                                          **
 ** Description: OrthographicSnapShot Plugin                                 **
 **                                                                          **
 **                                                                          **
 ** Author: Frank Naegele                                                    **
 **                                                                          **
 ** History:                                                                 **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

//#define PB_TESTING

#include "OrthographicSnapShot.h"

#include <config/CoviseConfig.h>
#define QT_CLEAN_NAMESPACE

#include <QString>
#include <QDebug>
#include <QCoreApplication>
#include <QFile>
#include <QDir>
#include <QDataStream>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/RenderObject.h>
#include <cover/VRViewer.h>
#include <cover/coVRSceneView.h>
#include <cover/coVRMSController.h>
#include <cover/VRWindow.h>
#include <cover/coVRTui.h>
#include <cover/coVRRenderer.h>
#include <util/coFileUtil.h>

#include <osgViewer/Viewer>
#include <osg/BoundingSphere>
#include <osg/Vec3d>
#include <osg/PositionAttitudeTransform>
#include <cover/coIntersection.h>
#include <osg/Camera>
#include <osg/MatrixTransform>
#include <osgDB/WriteFile>

#include <iostream>

#define OSNAP_RESOLUTION_MAX_X 4096
#define OSNAP_RESOLUTION_MAX_Y 4096

using namespace std;

//!##########################//
//! DrawCallback             //
//!##########################//

class DrawCallback : public osg::Camera::DrawCallback
{
public:
    DrawCallback(OrthographicSnapShot *plugin)
        : plugin_(plugin)
    {
    }

    virtual void operator()(const osg::Camera &cam) const;

private:
    OrthographicSnapShot *plugin_;
};

/*! \brief Save the image in a file after a snapshot.
*
* Saves the image and tells the plugin that the image has been saved.
*/
void
    DrawCallback::
    operator()(const osg::Camera &cam) const
{
    (void)cam;

    if (plugin_->doSnap_)
    {
        //fprintf(stderr,"\n--- OrthographicSnapShot::op write file\n" );
        if (plugin_->createScreenshot_)
        {
            cout << "\nOrthographicSnapShot: Writing file " << plugin_->filename_.c_str() << std::endl;
            osgDB::writeImageFile(*(plugin_->image_.get()), plugin_->filename_); // save file
        }

        //cerr << plugin_->filename_ << std::endl;

        plugin_->cameraCallbackExit(); // tell the plugin
    }
}

//!##########################//
//! OrthographicSnapShot     //
//!##########################//

OrthographicSnapShot::OrthographicSnapShot()
: coVRPlugin(COVER_PLUGIN_NAME)
, removeCamera_(false)
, doSnap_(false)
, hijackCam_(false)
, createScreenshot_(true)
, createHeightmap_(false)
, filename_("")
, heightFilename_("")
, xPos_(0.0)
, yPos_(0.0)
, width_(2000.0)
, height_(2000.0)
, scale_(1000.0)
,
//       scale_(1.0),
image_(new osg::Image())
, drawCallback_(new DrawCallback(this))
, pBufferCamera_(NULL)
{
}

OrthographicSnapShot::~OrthographicSnapShot()
{
    // Clean up //
    //
    if (doSnap_)
    {
        if (pBufferCamera_.get())
        {

            pBufferCamera_->detach(osg::Camera::COLOR_BUFFER);
            pBufferCamera_->setGraphicsContext(NULL);
            VRViewer::instance()->removeCamera(pBufferCamera_.get());
        }
    }

    VRViewer::instance()->overwriteViewAndProjectionMatrix(false);

    deleteUI();
}

bool
OrthographicSnapShot::init()
{
    initUI();

    // Adjust height to aspect ratio //
    //
    height_ = width_ * tuiResolutionY->getValue() / tuiResolutionX->getValue();
    tuiHeight->setValue(width_ * tuiResolutionY->getValue() / tuiResolutionX->getValue());

    return true;
}

void
OrthographicSnapShot::prepareSnapshot()
{
    if (coVRMSController::instance()->isMaster())
    {
        removeCamera_ = false;

        // Create the Camera //
        //
        pBufferCamera_ = new osg::Camera();

        osg::Camera *cam = dynamic_cast<osg::Camera *>(coVRConfig::instance()->channels[0].camera.get());

        image_.get()->allocateImage(tuiResolutionX->getValue(), tuiResolutionY->getValue(), 1, GL_RGB, GL_UNSIGNED_BYTE);

        pBufferCamera_->setViewport(0, 0, tuiResolutionX->getValue(), tuiResolutionY->getValue());

        pBufferCamera_->setRenderOrder(osg::Camera::PRE_RENDER);

        pBufferCamera_->setPostDrawCallback(drawCallback_.get());

        pBufferCamera_->setClearColor(cam->getClearColor());
        pBufferCamera_->setClearMask(cam->getClearMask());

        pBufferCamera_->setProjectionMatrix(cam->getProjectionMatrix());
        pBufferCamera_->setViewMatrix(cam->getViewMatrix());

        pBufferCamera_->setReferenceFrame(osg::Transform::ABSOLUTE_RF);

        pBufferCamera_->setView(cam->getView());
        pBufferCamera_->setRenderTargetImplementation((osg::Camera::RenderTargetImplementation)(osg::Camera::PIXEL_BUFFER));
        pBufferCamera_->setViewport(0, 0, tuiResolutionX->getValue(), tuiResolutionY->getValue());

        // Renderer //
        //
        osgViewer::Renderer *renderer = new coVRRenderer(pBufferCamera_.get(), 0);
        pBufferCamera_->setRenderer(renderer);
        pBufferCamera_->setGraphicsContext(cam->getGraphicsContext());

        VRViewer::instance()->addCamera(pBufferCamera_.get());
        pBufferCamera_->attach(osg::Camera::COLOR_BUFFER, image_.get());
        pBufferCamera_->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
        pBufferCamera_->setPostDrawCallback(drawCallback_.get()); // doppelt

        pBufferCamera_->setLODScale(0.000000001);

        renderer->getSceneView(0)->setSceneData(cover->getScene());
        renderer->getSceneView(1)->setSceneData(cover->getScene());
    }
}

void
OrthographicSnapShot::preFrame()
{
    // Take over camera //
    //
    if (hijackCam_)
    {
        cover->setXformMat(osg::Matrix());

        // ProjectionMatrix //
        //
        osg::Matrix projMat = osg::Matrix::ortho(-width_ / 2.0, width_ / 2.0, -height_ / 2.0, height_ / 2.0, 10000.0, 4000000.0);

        coVRConfig::instance()->channels[0].rightProj = projMat;
        coVRConfig::instance()->channels[0].leftProj = projMat;

        // ViewMatrix //
        //
        osg::Matrix scaleMat = cover->getObjectsScale()->getMatrix();
        osg::Matrix viewMat = cover->getXformMat();

        scaleMat.postMult(viewMat);

        viewMat = scaleMat;

        viewMat.invert(viewMat);
        viewMat.postMult(osg::Matrix::lookAt(osg::Vec3d(xPos_, yPos_, 1800000.0), osg::Vec3d(xPos_, yPos_, -1000000.0), osg::Vec3d(0.0, 1.0, 0.0)));

        coVRConfig::instance()->channels[0].rightView = viewMat;
        coVRConfig::instance()->channels[0].leftView = viewMat;
    }

    // The snapshot has been made, now clean up //
    //
    if (removeCamera_)
    {
        removeCamera_ = false;
        pBufferCamera_->setGraphicsContext(NULL);
        VRViewer::instance()->removeCamera(pBufferCamera_.get());
        pBufferCamera_ = NULL;
    }

    // Heightmap and OpenDrive tag //
    //
    if (doSnap_)
    {
        cout << "\n\n########################################################\nRunning OrthographicSnapShot plugin..." << std::endl;

        // Filenames //
        //
        filename_ = tuiFileName->getText();
        heightFilename_ = tuiHeightmapFileName->getText();

        // Bounding box //
        //
        double bboxRadius = cover->getObjectsXform()->getBound().radius();
        osg::Vec3d bboxCenter = cover->getObjectsXform()->getBound().center();
        cout << "\nBoundingBox radius: " << bboxRadius << std::endl;
        cout << "\nCenter: " << bboxCenter.x() << " " << bboxCenter.y() << " " << bboxCenter.z() << "\n" << std::endl;

        // Dimensions //
        //
        double xMin = xPos_ - width_ / 2.0; // [m]
        double yMin = yPos_ - height_ / 2.0;

        int xRes = tuiResolutionX->getValue(); // [px]
        int yRes = tuiResolutionY->getValue();

        double stepSizeX = width_ / xRes; // [m/px]
        double stepSizeY = height_ / yRes;

        cout << "\nMin/Max: " << xMin << ", " << yMin << std::endl;
        cout << "Size: " << width_ << ", " << height_ << std::endl;
        cout << "Resolution: " << xRes << ", " << yRes << std::endl;

        // Heightmap //
        //
        if (createHeightmap_)
        {
            // Heightmap file //
            //
            QFile heightfile(heightFilename_.c_str());
            heightfile.open(QIODevice::WriteOnly);
            QDataStream out(&heightfile);

            // Create height values //
            //
            double *heightArray;
            heightArray = (double *)malloc(tuiResolutionX->getValue() * tuiResolutionY->getValue() * sizeof(double));

            double minHeightValue = 100000000.0;
            double maxHeightValue = -100000000.0;
            double startTime = cover->currentTime();

            for (int row = 0; row < yRes; ++row)
            {
                cout << ".";
                for (int column = 0; column < xRes; ++column)
                {
                    int pixelPos = (yRes - (row + 1)) * xRes + column;

                    double x = (xMin + stepSizeX / 2.0 + stepSizeX * column) * scale_;
                    double y = (yMin + stepSizeY / 2.0 + stepSizeY * row) * scale_;
                    static double oldTime = 0.0;
                    if (cover->currentTime() > oldTime + 10.0)
                    {
                        oldTime = cover->currentTime();
                        float perc = (((float)row * (float)xRes + (float)column) / ((float)yRes * (float)xRes)) * 100.0;
                        fprintf(stderr, "Percentage: %f Time left: %fs, %d ; %d\n", perc, (1.0 - (perc / 100.0)) * ((oldTime - startTime) / (perc / 100.0)), row, column);
                    }


                    osg::Vec3 rayP = osg::Vec3(x, y, 9999999);
                    osg::Vec3 rayQ = osg::Vec3(x, y, -9999999);


                    coIntersector* isect = coIntersection::instance()->newIntersector(rayP, rayQ);
                    osgUtil::IntersectionVisitor visitor(isect);
                    visitor.setTraversalMask(Isect::Collision);

                    cover->getObjectsXform()->accept(visitor);

                    //std::cerr << "Hits ray num: " << num1 << ", down (" << ray->start()[0] << ", " << ray->start()[1] <<  ", " << ray->start()[2] << "), up (" << ray->end()[0] << ", " << ray->end()[1] <<  ", " << ray->end()[2] << ")" <<  std::endl;
                    if (!isect->containsIntersections())
                    {
                        heightArray[pixelPos] = 0.0;
                        continue;
                    }
                    else
                    {

                        auto results = isect->getFirstIntersection();
                        osg::Vec3d terrainHeight = results.getWorldIntersectPoint();

                        double height = terrainHeight.z() / scale_;
                        if (height < minHeightValue)
                        {
                            minHeightValue = height;
                        }
                        if (height > maxHeightValue)
                        {
                            maxHeightValue = height;
                        }

                        heightArray[pixelPos] = height;
                    }
                }
            }

            cout << "\nmin/max: " << minHeightValue << ", " << maxHeightValue << std::endl;

            // Save //
            //
            cout << "\nOrthographicSnapShot: Writing file " << heightFilename_.c_str() << std::endl;
            out.writeRawData((char *)heightArray, xRes * yRes * sizeof(double));
            heightfile.close();
        }

        // Print out OpenDRIVE (extension) tags //
        //
        cout << "\n<scenery>" << std::endl;
        cout << "<heightmap  x=\"" << xMin << "\" y=\"" << yMin << "\" width=\"" << width_ << "\" height=\"" << height_
             << "\" filename=\"" << filename_ << "\" data=\"" << heightFilename_ << "\" id=\"map0\" />" << std::endl;
        cout << "</scenery>\n" << std::endl;
    }
}

void
OrthographicSnapShot::cameraCallbackExit() const
{
    // Done //
    //
    doSnap_ = false;
    removeCamera_ = true;

    cout << "Done.\n########################################################\n" << std::endl;
}

//!##########################//
//! TABLET UI                //
//!##########################//

void
OrthographicSnapShot::initUI()
{
    // Tab //
    //
    tuiSnapTab = new coTUITab("O-Snapshot", coVRTui::instance()->mainFolder->getID());
    tuiSnapTab->setPos(0, 0);

    // Resolution //
    //
    (new coTUILabel("Resolution: ", tuiSnapTab->getID()))->setPos(0, 0);

    tuiResolutionX = new coTUIEditIntField("Resolution X", tuiSnapTab->getID());
    tuiResolutionX->setValue(1024);
    tuiResolutionX->setEventListener(this);
    tuiResolutionX->setImmediate(true);
    tuiResolutionX->setMin(0);
    tuiResolutionX->setMax(OSNAP_RESOLUTION_MAX_X);
    tuiResolutionX->setPos(1, 0);
    tuiResolutionX->setValue(coVRConfig::instance()->windows[0].sx);

    tuiResolutionY = new coTUIEditIntField("Resolution Y", tuiSnapTab->getID());
    tuiResolutionY->setValue(1024);
    tuiResolutionY->setEventListener(this);
    tuiResolutionY->setImmediate(true);
    tuiResolutionY->setMin(0);
    tuiResolutionY->setMax(OSNAP_RESOLUTION_MAX_Y);
    tuiResolutionY->setPos(2, 0);
    tuiResolutionY->setValue(coVRConfig::instance()->windows[0].sy);

    // Pos and size //
    //
    (new coTUILabel("Center: ", tuiSnapTab->getID()))->setPos(0, 1);

    tuiXPos = new coTUIEditFloatField("Center position x in [m]", tuiSnapTab->getID());
    tuiXPos->setEventListener(this);
    tuiXPos->setImmediate(true);
    tuiXPos->setPos(1, 1);
    tuiXPos->setValue(xPos_);

    tuiYPos = new coTUIEditFloatField("Center position y in [m]", tuiSnapTab->getID());
    tuiYPos->setEventListener(this);
    tuiYPos->setImmediate(true);
    tuiYPos->setPos(2, 1);
    tuiYPos->setValue(yPos_);

    (new coTUILabel("Size: ", tuiSnapTab->getID()))->setPos(0, 2);

    tuiWidth = new coTUIEditFloatField("Width in [m]", tuiSnapTab->getID());
    tuiWidth->setEventListener(this);
    tuiWidth->setImmediate(true);
    tuiWidth->setPos(1, 2);
    tuiWidth->setValue(width_);

    tuiHeight = new coTUIEditFloatField("Height in [m]", tuiSnapTab->getID());
    tuiHeight->setEventListener(this);
    tuiHeight->setImmediate(true);
    tuiHeight->setPos(2, 2);
    tuiHeight->setValue(height_);

    // Filename //
    //
    (new coTUILabel("Snapshot file: ", tuiSnapTab->getID()))->setPos(0, 3);

    tuiFileName = new coTUIEditField("Filename", tuiSnapTab->getID());
    tuiFileName->setEventListener(this);
    tuiFileName->setText("heightmap.png");
    tuiFileName->setPos(1, 3);

    tuiToggleSnapshot = new coTUIToggleButton("On/Off", tuiSnapTab->getID(), createScreenshot_);
    tuiToggleSnapshot->setEventListener(this);
    tuiToggleSnapshot->setPos(2, 3);

    (new coTUILabel("Heightmap file: ", tuiSnapTab->getID()))->setPos(0, 4);

    tuiHeightmapFileName = new coTUIEditField("Filename", tuiSnapTab->getID());
    tuiHeightmapFileName->setEventListener(this);
    tuiHeightmapFileName->setText("heightmap.dat");
    tuiHeightmapFileName->setPos(1, 4);

    tuiToggleHeightmap = new coTUIToggleButton("On/Off", tuiSnapTab->getID(), createHeightmap_);
    tuiToggleHeightmap->setEventListener(this);
    tuiToggleHeightmap->setPos(2, 4);

    (new coTUILabel("", tuiSnapTab->getID()))->setPos(0, 5);

    // Hijack //
    //
    tuiHijackButton = new coTUIToggleButton("Hijack view", tuiSnapTab->getID());
    tuiHijackButton->setEventListener(this);
    tuiHijackButton->setPos(0, 6);

    // Snapshot //
    //
    tuiSnapButton = new coTUIButton("Run", tuiSnapTab->getID());
    tuiSnapButton->setEventListener(this);
    tuiSnapButton->setPos(1, 6);
}

void
OrthographicSnapShot::deleteUI()
{
    // UI //
    //

    delete tuiToggleSnapshot;
    delete tuiToggleHeightmap;

    delete tuiFileName;
    delete tuiHeightmapFileName;

    delete tuiXPos;
    delete tuiYPos;
    delete tuiWidth;
    delete tuiHeight;

    delete tuiResolutionX;
    delete tuiResolutionY;

    delete tuiHijackButton;
    delete tuiSnapButton;

    delete tuiSnapTab;
}

void
OrthographicSnapShot::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == tuiResolutionX)
    {
    }

    if (tUIItem == tuiResolutionY)
    {
    }

    if (tUIItem == tuiXPos)
    {
        xPos_ = tuiXPos->getValue();
    }

    if (tUIItem == tuiYPos)
    {
        yPos_ = tuiYPos->getValue();
    }

    if (tUIItem == tuiWidth)
    {
        width_ = tuiWidth->getValue();
    }

    if (tUIItem == tuiHeight)
    {
        height_ = tuiHeight->getValue();
    }

    if (tUIItem == tuiToggleSnapshot)
    {
        createScreenshot_ = tuiToggleSnapshot->getState();
    }

    if (tUIItem == tuiToggleHeightmap)
    {
        createHeightmap_ = tuiToggleHeightmap->getState();
    }

    if (tUIItem == tuiHijackButton)
    {
        if (coVRMSController::instance()->isMaster())
        {
            hijackCam_ = tuiHijackButton->getState();

            // Tell the viewer that I set the view and projection matrices //
            //
            VRViewer::instance()->overwriteViewAndProjectionMatrix(hijackCam_);
        }
    }
}

void
OrthographicSnapShot::tabletPressEvent(coTUIElement *tUIItem)
{
}

void
OrthographicSnapShot::tabletReleaseEvent(coTUIElement *tUIItem)
{
    // Snapshot //
    //
    if (tUIItem == tuiSnapButton)
    {
        if (createScreenshot_ || createHeightmap_)
        {
            if (coVRMSController::instance()->isMaster())
            {
                doSnap_ = true;
                prepareSnapshot();
            }
        }
    }
}

//!##########################//
//! STUFF                    //
//!##########################//

std::string
OrthographicSnapShot::suggestFileName(std::string directory)
{
    // Check //
    //
    QDir dir(directory.c_str());
    if (!dir.exists())
    {
        if (QDir::temp().exists())
        {
            directory = QDir::temp().absolutePath().toStdString();
        }
        else
        {
            directory = "/var/tmp";
        }
        dir.setCurrent(directory.c_str());
    }

    // Trailing slash //
    //
    if (directory[directory.length() - 1] != '/')
    {
        directory += '/';
    }

    // Filename //
    //
    std::string fileName("heightmap");
    fileName += ".png";
    directory += fileName;

    return directory;
}

COVERPLUGIN(OrthographicSnapShot)
