/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: ALVAR Plugin                                            **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                 **
**                                                                          **
** History:  								                                 **
** Mar-05  v1	    				       		                             **
**                                                                          **
**                                                                          **
\****************************************************************************/

#ifndef GLUT_NO_LIB_PRAGMA
#define GLUT_NO_LIB_PRAGMA
#endif

#undef HAVE_CUDA
#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#endif
#include "ALVARTrackerPlugin.h"
#include "../common/RemoteAR.h"
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/RenderObject.h>
#include <cover/MarkerTracking.h>
#include <config/CoviseConfig.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>
#include <Pose.h>

#include "Alvar.h"
#include <vector>
#include <string>

using std::cout;
using std::endl;
#include <signal.h>
#include <osg/MatrixTransform>

#ifdef __MINGW32__
#include <GL/glext.h>
#endif

#define MODE_1280x960_MONO 130

#ifdef __linux__
#include <asm/ioctls.h>
#define sigset signal
#endif
#ifndef _WIN32
#include <sys/ipc.h>
#include <sys/msg.h>
#endif
struct myMsgbuf
{
    long mtype;
    char mtext[100];
};

int ALVARPlugin::loadPattern(const char *p)
{
    int pattID = atoi(p);
    if (pattID <= 0)
    {
        fprintf(stderr, "pattern load error !!\n");
        pattID = 0;
    }
    if (pattID > 100)
    {
        fprintf(stderr, "Pattern ID out of range !!\n");
        pattID = 0;
    }
    return pattID;
}

bool ALVARPlugin::isVisible(int pattID)
{

    //if(pattID==(*MarkerTracking::instance()->markers.begin())->getPattern())
    //return true;
    /* check for marker visibility */
    for (size_t i = 0; i < marker_detector.markers->size(); i++)
    {
        std::list<MarkerTrackingMarker *>::iterator it;
        /*	for(it=MarkerTracking::instance()->markers.begin();it!=MarkerTracking::instance()->markers.end();it++)
		{
			if((*it)->getPattern()==pattID)
			{
				if((*it)->isObjectMarker())
					return false;
				break;
			}
		}*/
        if ((*(marker_detector.markers))[i].data.id == pattID)
        {
            return true;
        }
    }
    return false;
}

double pattSize;
double pattCenter[2];
double pattTrans[3][4];
osg::Matrix ALVARPlugin::getMat(int pattID, double pattCenter[2], double pattSize, double pattTrans[3][4])
{

    osg::Matrix markerTrans;
    markerTrans.makeIdentity();
    for (size_t i = 0; i < marker_detector.markers->size(); i++)
    {
        /*if(pattID==(*MarkerTracking::instance()->markers.begin())->getPattern())
		{	
			double markerPosed[16];
			CvMat markerPoseMat = cvMat(4, 4, CV_64F, markerPosed);
			bundlePose.GetMatrix(&markerPoseMat);
			
			int u,v;
			for(u=0;u<4;u++)
				for(v=0;v<4;v++)
					markerTrans(v,u)=markerPosed[(u*4)+v];
		}
		else*/
        {
            if ((*(marker_detector.markers))[i].data.id == pattID)
            {
                alvar::Pose p = (*(marker_detector.markers))[i].pose; /* get the transformation between the marker and the real camera */
                /*if(pattID==(*MarkerTracking::instance()->markers.begin())->getPattern())
					p = bundlePose;*/

                double markerPosed[16];
                CvMat markerPoseMat = cvMat(4, 4, CV_64F, markerPosed);
                p.GetMatrix(&markerPoseMat);

                int u, v;
                for (u = 0; u < 4; u++)
                    for (v = 0; v < 4; v++)
                        markerTrans(v, u) = markerPosed[(u * 4) + v];

                /*	if(pattID==(*MarkerTracking::instance()->markers.begin())->getPattern())
				{
					return OpenGLToOSGMatrix*markerTrans*OpenGLToOSGMatrix*(*MarkerTracking::instance()->markers.begin())->getOffset();
				}
				else*/
                {
                    return OpenGLToOSGMatrix * markerTrans * OpenGLToOSGMatrix;
                }
            }
        }
    }

    return OpenGLToOSGMatrix * markerTrans * OpenGLToOSGMatrix;
}

ALVARPlugin::ALVARPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    marker_num = 0;

    OpenGLToOSGMatrix.makeRotate(M_PI / -2.0, 1, 0, 0);
    OSGToOpenGLMatrix.makeRotate(M_PI / 2.0, 1, 0, 0);
    //marker_info = NULL;

    dataPtr = NULL;
    numNames = 0;
}

bool ALVARPlugin::init()
{
    //sleep(6);

    MarkerTracking::instance()->arInterface = this;
    MarkerTracking::instance()->remoteAR = NULL;
	MarkerTracking::instance()->videoData = NULL;

    multiMarkerInitializer = NULL;
    multiMarkerBundle = NULL;
    cap = NULL;
    doCalibrate = false;
    calibrated = false;
    calibCount = 0;

    fprintf(stderr, "ALVARPlugin::ALVARPlugin\n");

    if (coCoviseConfig::isOn("COVER.Plugin.ALVAR.Capture", false))
    {

        if (coCoviseConfig::isOn("COVER.Plugin.ALVAR.MirrorRight", false))
            MarkerTracking::instance()->videoMirrorRight = true;
        if (coCoviseConfig::isOn("COVER.Plugin.ALVAR.MirrorLeft", false))
            MarkerTracking::instance()->videoMirrorLeft = true;
        if (coCoviseConfig::isOn("COVER.Plugin.ALVAR.RemoteAR.Transmit", true))
        {
            bitrateSlider = new coTUISlider("Bitrate", MarkerTracking::instance()->artTab->getID());
            bitrateSlider->setValue(300);
            bitrateSlider->setTicks(4950);
            bitrateSlider->setMin(50);
            bitrateSlider->setMax(5000);
            bitrateSlider->setPos(3, 0);
            bitrateSlider->setEventListener(this);
        }
        MarkerTracking::instance()->flipH = coCoviseConfig::isOn("COVER.Plugin.ALVAR.FlipHorizontal", false);
        flipBufferH = coCoviseConfig::isOn("COVER.Plugin.ALVAR.FlipBufferH", false);
        flipBufferV = coCoviseConfig::isOn("COVER.Plugin.ALVAR.FlipBufferV", true);
        std::string VideoDevice = coCoviseConfig::getEntry("value", "COVER.Plugin.ALVAR.VideoDevice", "0");
        calibrationFilename = coCoviseConfig::getEntry("value", "COVER.Plugin.ALVAR.CameraCalibrationFile", "/data/MarkerTracking/defaultCalib.xml");
        xsize = coCoviseConfig::getInt("width", "COVER.Plugin.ALVAR.VideoDevice", 640);
        ysize = coCoviseConfig::getInt("height", "COVER.Plugin.ALVAR.VideoDevice", 480);
        thresh = coCoviseConfig::getInt("COVER.Plugin.ALVAR.Threshold", 100);
        msgQueue = -1;

        arDebugButton = new coTUIToggleButton("Debug", MarkerTracking::instance()->artTab->getID());
        arDebugButton->setPos(0, 0);
        arDebugButton->setEventListener(this);
        arSettingsButton = new coTUIButton("Settings", MarkerTracking::instance()->artTab->getID());
        arSettingsButton->setPos(1, 1);
        arSettingsButton->setEventListener(this);

        calibrateButton = new coTUIToggleButton("CalibrateCamera", MarkerTracking::instance()->artTab->getID());
        calibrateButton->setPos(2, 1);
        calibrateButton->setEventListener(this);
        calibrateLabel = new coTUILabel("notCalibrated", MarkerTracking::instance()->artTab->getID());
        calibrateLabel->setPos(3, 1);

        visualizeButton = new coTUIToggleButton("VisualizeMarkers", MarkerTracking::instance()->artTab->getID());
        visualizeButton->setPos(4, 1);
        visualizeButton->setEventListener(this);

        detectAdditional = new coTUIToggleButton("DetectAdditional", MarkerTracking::instance()->artTab->getID());
        detectAdditional->setPos(5, 1);
        detectAdditional->setEventListener(this);
        detectAdditional->setState(false);

        useSFM = new coTUIToggleButton("useSFM", MarkerTracking::instance()->artTab->getID());
        useSFM->setPos(6, 1);
        useSFM->setEventListener(this);
        useSFM->setState(false);

        sfm = NULL;

        //if(coCoviseConfig::isOn("COVER.Plugin.ALVAR.Stereo", false))
        //captureRightVideo();

        try
        {

            // Create capture object from camera (argv[1] is a number) or from file (argv[1] is a string)
            std::string uniqueName;
            if ((VideoDevice.length() > 0) && (!isdigit(VideoDevice[0])))
            {
                // Manually create capture device and initialize capture object
                alvar::CaptureDevice device("file", VideoDevice.c_str());
                cap = alvar::CaptureFactory::instance()->createCapture(device);
                uniqueName = "file";
            }
            else
            {
                // Enumerate possible capture plugins
                alvar::CaptureFactory::CapturePluginVector plugins = alvar::CaptureFactory::instance()->enumeratePlugins();
                if (plugins.size() < 1)
                {
                    std::cout << "Could not find any capture plugins." << std::endl;
                    return 0;
                }

                // Display capture plugins
                std::cout << "Available Plugins: ";
                outputEnumeratedPlugins(plugins);
                std::cout << std::endl;

                // Enumerate possible capture devices
                alvar::CaptureFactory::CaptureDeviceVector devices = alvar::CaptureFactory::instance()->enumerateDevices();
                if (devices.size() < 1)
                {
                    std::cout << "Could not find any capture devices." << std::endl;
                    return 0;
                }

                // Check command line argument for which device to use
                int selectedDevice;
                selectedDevice = atoi(VideoDevice.c_str());
                if (selectedDevice >= (int)devices.size())
                {
                    selectedDevice = defaultDevice(devices);
                }

                // Display capture devices
                std::cout << "Enumerated Capture Devices:" << std::endl;
                outputEnumeratedDevices(devices, selectedDevice);
                std::cout << std::endl;

                // Create capture object from camera
                cap = alvar::CaptureFactory::instance()->createCapture(devices[selectedDevice]);
                uniqueName = devices[selectedDevice].uniqueName();
            }

            // Handle capture lifecycle and start video capture
            // Note that loadSettings/saveSettings are not supported by all plugins
            if (cap)
            {
                std::stringstream settingsFilename;
                settingsFilename << "camera_settings_" << uniqueName << ".xml";
                //calibrationFilename << "camera_calibration_" << uniqueName << ".xml";

                if (!cap->start())
                {
                    delete cap;
                    cap=NULL;
                    return false;
                }
                cap->setResolution(xsize, ysize);
                xsize = cap->xResolution();
                ysize = cap->yResolution();

                if (cap->loadSettings(settingsFilename.str()))
                {
                    std::cout << "Loading settings: " << settingsFilename.str() << std::endl;
                }

                if (cap->saveSettings(settingsFilename.str()))
                {
                    std::cout << "Saving settings: " << settingsFilename.str() << std::endl;
                }
                float marker_size = 57;
                marker_detector.SetMarkerSize(marker_size); // for marker ids larger than 255, set the content resolution accordingly

                updateMarkerParams();

                //static MarkerDetector<MarkerArtoolkit> marker_detector;
                //marker_detector.SetMarkerSize(2.8, 3, 1.5);
                std::cout << "Loading calibration: " << calibrationFilename;

                if (cam.SetCalib(calibrationFilename.c_str(), xsize, ysize))
                {
                    std::cout << " [Ok]" << endl;
                }
                else
                {
                    cam.SetRes(xsize, ysize);
                    std::cout << " [Fail]" << endl;
                }
                MarkerTracking::instance()->running = true;

                MarkerTracking::instance()->videoMode = GL_BGR;
                //MarkerTracking::instance()->videoData=new unsigned char[xsize*ysize*3];
                MarkerTracking::instance()->videoDepth = 3;
                MarkerTracking::instance()->videoWidth = xsize;
                MarkerTracking::instance()->videoHeight = ysize;

                adjustScreen();
            }
            else
            {
                std::cout << "Could not initialize the selected capture backend." << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            std::cout << "Exception: " << e.what() << endl;
        }
        catch (...)
        {
            std::cout << "Exception: unknown" << std::endl;
        }
    }

    MarkerTracking::instance()->remoteAR = new RemoteAR();
    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens
ALVARPlugin::~ALVARPlugin()
{
    delete MarkerTracking::instance()->remoteAR;
    MarkerTracking::instance()->remoteAR = 0;
    MarkerTracking::instance()->arInterface = NULL;

    MarkerTracking::instance()->running = false;
    fprintf(stderr, "ALVARPlugin::~ALVARPlugin\n");
    if(cap)
    {
        cap->stop();
        if (msgQueue >= 0)
        {
#ifndef _WIN32
            msgctl(msgQueue, IPC_RMID, NULL);
#endif
        }
    }
}

void ALVARPlugin::updateMarkerParams()
{
    //delete multiMarkerInitializer;
    delete multiMarkerBundle;
    vector<int> ids;
    std::list<MarkerTrackingMarker *>::iterator it;
    for (it = MarkerTracking::instance()->markers.begin(); it != MarkerTracking::instance()->markers.end(); it++)
    {
        marker_detector.SetMarkerSizeForId((*it)->getPattern(), (*it)->getSize());
        ids.push_back((*it)->getPattern());
    }

    multiMarkerInitializer = new alvar::MultiMarkerInitializer(ids);
    multiMarkerBundle = new alvar::MultiMarkerBundle(ids);
    /*	for(it=MarkerTracking::instance()->markers.begin();it!=MarkerTracking::instance()->markers.end();it++)
	{
        alvar::Pose pose;
		pose.Reset();
		osg::Matrix markerTrans= OSGToOpenGLMatrix*(*it)->getOffset()*OpenGLToOSGMatrix;
		CvMat *posMat = cvCreateMat(4, 4, CV_64F);

		int u,v;
		for(u=0;u<4;u++)
			for(v=0;v<4;v++)
				posMat->data.db[(u*4)+v] = markerTrans(v,u);
		pose.SetMatrix(posMat);
		multiMarkerBundle->PointCloudAdd((*it)->getPattern(), (*it)->getSize(), pose);
	}*/
};

void ALVARPlugin::adjustScreen()
{
    if (coCoviseConfig::isOn("COVER.Plugin.ALVAR.AdjustScreenParameters", true))
    {
        osg::Vec3 viewPos;

        float sxsize = xsize;
        float sysize = ysize;

        float d;

        d = cam.calib_K_data[0][0];
        sysize = ((double)ysize / cam.calib_K_data[1][1]) * d;

        coVRConfig::instance()->screens[0].hsize = sxsize;
        coVRConfig::instance()->screens[0].vsize = sysize;

        viewPos.set(cam.calib_K_data[0][2] - ((double)xsize / 2.0), -d, ((double)ysize / 2.0) - cam.calib_K_data[1][2]);

        VRViewer::instance()->setInitialViewerPos(viewPos);
        osg::Matrix viewMat;
        viewMat.makeIdentity();
        viewMat.setTrans(viewPos);
        VRViewer::instance()->setViewerMat(viewMat);
    }
}

void ALVARPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == useSFM)
    {
        if (useSFM->getState())
        {
            sfm = new alvar::SimpleSfM();

            sfm->Clear();
            cout << "Loading calibration: " << calibrationFilename;

            if (sfm->GetCamera()->SetCalib(calibrationFilename.c_str(), xsize, ysize))
            {
                cout << " [Ok]" << endl;
            }
            else
            {
                sfm->GetCamera()->SetRes(xsize, ysize);
                cout << " [Fail]" << endl;
            }

            sfm->SetScale(10);
            std::cout << "Couldn't load mmarker.xml. Using default 'SampleMultiMarker' setup." << std::endl;

            for (size_t i = 0; i < marker_detector.markers->size(); i++)
            {
                if ((*(marker_detector.markers))[i].data.id == 38)
                {
                    alvar::Pose pose;
                    pose.Reset();
                    sfm->AddMarker(38, (*(marker_detector.markers))[i].GetMarkerEdgeLength(), pose);
                }
            }
            sfm->SetResetPoint();
        }
        else
        {
            delete sfm;
            sfm = NULL;
        }
    }
    if (tUIItem == arDebugButton)
    {
        //arDebug = arDebugButton->getState();
    }
    if (tUIItem == arSettingsButton)
    {
#ifdef WIN32
//arVideoShowDialog(1);
#endif
    }
    if (tUIItem == bitrateSlider)
    {
        MarkerTracking::instance()->remoteAR->updateBitrate(bitrateSlider->getValue());
    }
    else if (tUIItem == calibrateButton)
    {
        doCalibrate = calibrateButton->getState();
    }
}

void ALVARPlugin::tabletPressEvent(coTUIElement * /*tUIItem*/)
{
}

bool
ALVARPlugin::update()
{
    return MarkerTracking::instance()->running;
}

void
ALVARPlugin::preFrame()
{
#ifndef _WIN32
    struct myMsgbuf message;
#endif
    if (MarkerTracking::instance()->running)
    {
#ifndef _WIN32
        if (msgQueue > 0)
        {
            // allow right capture process to continue
            message.mtype = 1;
            msgsnd(msgQueue, &message, 1, 0);
        }
#endif
        if (cap)
        {
            IplImage *frame = cap->captureImage();
            if (frame)
            {
                
                MarkerTracking::instance()->videoData = (unsigned char *)frame->imageData;
                if (frame->imageData == NULL)
                {
                    fprintf(stderr, "Video input dropped frame\n");
                }
                else
                {
                    
                    double error = 0.0;
                    marker_detector.Detect(frame, &cam, true, visualizeButton->getState());
                    
                    if (marker_detector.Detect(frame, &cam, true, visualizeButton->getState(), 0.0))
                    {
                        if (detectAdditional->getState())
                        {
                            error = multiMarkerBundle->Update(marker_detector.markers, &cam, bundlePose);
                            multiMarkerBundle->SetTrackMarkers(marker_detector, &cam, bundlePose, visualizeButton->getState() ? frame : NULL);
                            marker_detector.DetectAdditional(frame, &cam, visualizeButton->getState());
                        }
                        if (visualizeButton->getState())
                            error = multiMarkerBundle->Update(marker_detector.markers, &cam, bundlePose, frame);
                        else
                            error = multiMarkerBundle->Update(marker_detector.markers, &cam, bundlePose);
                    }
                    
                    static double oldTime = 0;
                    for (size_t i = 0; i < marker_detector.markers->size(); i++)
                    {
                        if (i >= 32)
                            break;
                        
                        alvar::Pose p = (*(marker_detector.markers))[i].pose;
                        if (cover->frameTime() > oldTime + 2)
                        {
                            fprintf(stderr, "Marker: %d\n", (*(marker_detector.markers))[i].data.id);
                            if (i == marker_detector.markers->size() - 1)
                                oldTime = cover->frameTime();
                        }
                        /*if((*(marker_detector.markers))[i].data.id== 1)
                          {
                          }*/
                        /*
                          p.GetMatrixGL(d[i].gl_mat);
                          
                          int id = (*(marker_detector.markers))[i].GetId();
                          double r = 1.0 - double(id+1)/32.0;
                          double g = 1.0 - double(id*3%32+1)/32.0;
                          double b = 1.0 - double(id*7%32+1)/32.0;
                          d[i].SetColor(r, g, b);
                          
                          GlutViewer::DrawableAdd(&(d[i]));*/
                    }
                    if (sfm)
                    {
                        if (sfm->Update(frame, false, true, 7.f, 15.f))
                        {
                            // Draw the camera (The GlutViewer has little weirdness here...)q
                            alvar::Pose pose = *(sfm->GetPose());
                            double gl[16];
                            pose.GetMatrixGL(gl, true);
                            
                            if (visualizeButton->getState())
                            {
                                // Draw features
                                std::map<int, alvar::SimpleSfM::Feature>::iterator iter;
                                iter = sfm->container.begin();
                                for (; iter != sfm->container.end(); iter++)
                                {
                                    if (sfm->container_triangulated.find(iter->first) != sfm->container_triangulated.end())
                                        continue;
                                    if (iter->second.has_p3d)
                                    {
                                        /*if (own_drawable_count < 1000) {
                                          memset(d_points[own_drawable_count].gl_mat, 0, 16*sizeof(double));
                                          d_points[own_drawable_count].gl_mat[0]  = 1;
                                          d_points[own_drawable_count].gl_mat[5]  = 1;
                                          d_points[own_drawable_count].gl_mat[10] = 1;
                                          d_points[own_drawable_count].gl_mat[15] = 1;
                                          d_points[own_drawable_count].gl_mat[12] = iter->second.p3d.x;
                                          d_points[own_drawable_count].gl_mat[13] = iter->second.p3d.y;
                                          d_points[own_drawable_count].gl_mat[14] = iter->second.p3d.z;
                                          if (iter->second.type_id == 0) d_points[own_drawable_count].SetColor(1,0,0);
                                          else d_points[own_drawable_count].SetColor(0,1,0);
                                          GlutViewer::DrawableAdd(&(d_points[own_drawable_count]));
                                          own_drawable_count++;
                                          }*/
                                    }
                                }
                                
                                // Draw triangulated features
                                iter = sfm->container_triangulated.begin();
                                for (; iter != sfm->container_triangulated.end(); iter++)
                                {
                                    if (iter->second.has_p3d)
                                    {
                                        /*if (own_drawable_count < 1000) {
                                          memset(d_points[own_drawable_count].gl_mat, 0, 16*sizeof(double));
                                          d_points[own_drawable_count].gl_mat[0]  = 1;
                                          d_points[own_drawable_count].gl_mat[5]  = 1;
                                          d_points[own_drawable_count].gl_mat[10] = 1;
                                          d_points[own_drawable_count].gl_mat[15] = 1;
                                          d_points[own_drawable_count].gl_mat[12] = iter->second.p3d.x;
                                          d_points[own_drawable_count].gl_mat[13] = iter->second.p3d.y;
                                          d_points[own_drawable_count].gl_mat[14] = iter->second.p3d.z;
                                          d_points[own_drawable_count].SetColor(0,0,1);
                                          GlutViewer::DrawableAdd(&(d_points[own_drawable_count]));
                                          own_drawable_count++;
                                          }*/
                                    }
                                }
                            }
                        }
                    }
                    if (doCalibrate)
                    {
                        
                        const int calibCountMax = 50;
                        const int calibRows = 6;
                        const int calibColumns = 9;
                        if (!calibrated)
                        {
                            // If we have already collected enough data to make the calibration
                            // - We are ready to end the capture loop
                            // - Calibrate
                            // - Save the calibration file
                            if (calibCount >= calibCountMax)
                            {
                                std::cout << "Calibrating..." << endl;
                                calibCount = 0;
                                cam.Calibrate(projPoints);
                                projPoints.Reset();
                                cam.SaveCalib(calibrationFilename.c_str());
                                std::cout << "Saving calibration: " << calibrationFilename << endl;
                                adjustScreen();
                                calibrated = true;
                            }
                            // If we are still collecting calibration data
                            // - For every 1.5s add calibration data from detected 7*9 chessboard (and visualize it if true)
                            else
                            {
                                static double lastTime = 0;
                                double currentTime = cover->frameTime();
                                if (currentTime > (lastTime + 0.5))
                                {
                                    if (projPoints.AddPointsUsingChessboard(frame, 2.42, calibRows, calibColumns, true))
                                    {
                                        lastTime = currentTime;
                                        calibCount++;
                                        //cout<<calibCount<<"/"<<calibCountMax<<endl;
                                        char tmpText[100];
                                        sprintf(tmpText, "%d%%", (int)(((float)calibCount / (float)calibCountMax) * 100.0));
                                        calibrateLabel->setLabel(tmpText);
                                    }
                                }
                            }
                        }
                        else
                        {
                            if (projPoints.AddPointsUsingChessboard(frame, 2.5, calibRows, calibColumns, true))
                            {
                                alvar::Pose pose;
                                cam.CalcExteriorOrientation(projPoints.object_points, projPoints.image_points, &pose);
                                cam.ProjectPoints(projPoints.object_points, &pose, projPoints.image_points);
                                for (size_t i = 0; i < projPoints.image_points.size(); i++)
                                {
                                    cvCircle(frame, cvPoint((int)projPoints.image_points[i].x, (int)projPoints.image_points[i].y), 6, CV_RGB(0, 0, 255));
                                }
                                projPoints.Reset();
                            }
                        }
                    }
                }
            }
        }
    }
}

void ALVARPlugin::outputEnumeratedPlugins(alvar::CaptureFactory::CapturePluginVector &plugins)
{
    for (int i = 0; i < (int)plugins.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ", ";
        }
        std::cout << plugins.at(i);
    }

    std::cout << std::endl;
}

void ALVARPlugin::outputEnumeratedDevices(alvar::CaptureFactory::CaptureDeviceVector &devices, int selectedDevice)
{
    for (int i = 0; i < (int)devices.size(); ++i)
    {
        if (selectedDevice == i)
        {
            std::cout << "* ";
        }
        else
        {
            std::cout << "  ";
        }

        std::cout << i << ": " << devices.at(i).uniqueName();

        if (devices[i].description().length() > 0)
        {
            std::cout << ", " << devices.at(i).description();
        }

        std::cout << std::endl;
    }
}

int ALVARPlugin::defaultDevice(alvar::CaptureFactory::CaptureDeviceVector &devices)
{
    for (int i = 0; i < (int)devices.size(); ++i)
    {
        if (devices.at(i).captureType() == "highgui")
        {
            return i;
        }
    }

    return 0;
}

COVERPLUGIN(ALVARPlugin)
