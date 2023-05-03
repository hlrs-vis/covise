/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  SystemCover.cpp
//  A class to contain system-dependent/non-standard utilities
//
#ifdef WIN32
#ifdef WIN32_LEAN_AND_MEAN
#undef WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <Windows.h>
#endif

#include <boost/filesystem.hpp>
#include <boost/locale.hpp>
#include <locale>
#include <codecvt>

#include <util/common.h>
#include <util/unixcompat.h>
#include <fcntl.h>
#include <vrml97/vrml/config.h>
#include <vrml97/vrml/System.h>

#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlNodeViewpoint.h>
#include <config/CoviseConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/input/input.h>
#include <cover/coVRConfig.h>
#include <cover/coVRMSController.h>
#include <cover/coVRCommunication.h>
#include <cover/coVRPartner.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRCollaboration.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRFileManager.h>
#include <cover/VRSceneGraph.h>
#include <cover/OpenCOVER.h>
#include <cover/VRAvatar.h>
#include <net/message.h>
#include <net/message_types.h>
#include <vrb/client/VRBClient.h>
#include <vrb/client/VRBMessage.h>

#include <cover/coVRLighting.h>
#include <cover/coVRTui.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Group.h>
#include <cover/ui/Action.h>

#include "SystemCover.h"
#include "ViewerObject.h"
#include "ViewerOsg.h"
#include "Vrml97Plugin.h"

#ifdef _WIN32
#include <fcntl.h>
#include <exdisp.h>
#ifndef MINGW
#include <atlbase.h>
#include "atlconv.h"
#endif
#endif

#include <osg/MatrixTransform>
#include <osgDB/Registry>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileNameUtils>
#include <osgDB/ReaderWriter>
#include <osgUtil/Optimizer>

using namespace covise;

namespace
{

const char cacheExt[] = ".ive";

}

ViewpointEntry::ViewpointEntry(VrmlNodeViewpoint *aViewPoint, VrmlScene *aScene)
{
    scene = aScene;
    viewPoint = aViewPoint;
    entryNumber = ((SystemCover *)System::the)->maxEntryNumber++;

    tuiItem = new coTUIToggleButton(viewPoint->description(), ((SystemCover *)System::the)->vrmlTab->getID());
    tuiItem->setEventListener((SystemCover *)System::the);
    tuiItem->setPos((int)(entryNumber / 10.0) + 1, entryNumber % 10);
}

ViewpointEntry::~ViewpointEntry()
{
    delete tuiItem;
    delete menuItem;
}

void ViewpointEntry::setMenuItem(ui::Button *aButton)
{
    menuItem = aButton;
    menuItem->setCallback([this](bool state){
        if (state)
            activate();
    });
}

void ViewpointEntry::activate()
{
    double timeNow = System::the->time();
    VrmlSFBool flag(true);
    viewPoint->eventIn(timeNow, "set_bind", &flag);
    menuItem->setState(true);
    tuiItem->setState(true);
}

SystemCover::SystemCover()
    : ui::Owner("SystemCover", cover->ui)
{
    mFileManager = coVRFileManager::instance();
    maxEntryNumber = 0;
    record = false;
    fileNumber = 0;
	m_optimize = coCoviseConfig::isOn("COVER.Plugin.Vrml97.DoOptimize", true);
	cerr << "vrml optimizer  = " << m_optimize << endl;
    if (coVRMSController::instance()->isMaster())
    {
        if (const char *cache = getenv("COCACHE"))
        {
            if (strcasecmp(cache, "disable")==0)
                cacheMode = CACHE_DISABLE;
            else if (strcasecmp(cache, "write")==0 || strcasecmp(cache, "rewrite")==0)
                cacheMode = CACHE_REWRITE;
            else if (strcasecmp(cache, "use")==0)
                cacheMode = CACHE_USE;
            else if (strcasecmp(cache, "useold")==0)
                cacheMode = CACHE_USEOLD;

            std::cerr << "Vrml97 Inline cache (disable|rewrite|create|use|useold): ";
            switch(cacheMode)
            {
            case CACHE_DISABLE:
                std::cerr << "disable";
                break;
            case CACHE_REWRITE:
                std::cerr << "forcing rewrite";
                break;
            case CACHE_CREATE:
                std::cerr << "use or create";
                break;
            case CACHE_USE:
                std::cerr << "use only";
                break;
            case CACHE_USEOLD:
                std::cerr << "use only, even if outdated";
                break;
            }
            std::cerr << std::endl;
        }
    }
    coVRMSController::instance()->syncData(&cacheMode, sizeof(cacheMode));
}

bool SystemCover::loadUrl(const char *url, int np, char **parameters)
{
    if (!url)
        return false;
    char *buf = new char[strlen(url) + 200];
    int result = 1;
#if !defined(_WIN32) || defined(MINGW)
    if (np)
        sprintf(buf, "/bin/csh -c \"netscape -remote 'openURL(%s, %s)'\" &",
                url, parameters[0]);
    else
        sprintf(buf, "/bin/csh -c \"netscape -remote 'openURL(%s)'\" &", url);
    result = system(buf);
    fprintf(stderr, "%s\n", buf);
#else
    ::CoInitialize(NULL);
    static IWebBrowser2 *browser = NULL;

    if (browser == NULL)
    {
        HRESULT hRes = CoCreateInstance(CLSID_InternetExplorer, NULL, CLSCTX_LOCAL_SERVER,
                                        IID_IWebBrowser2, (void **)&browser);
        if (FAILED(hRes))
        {
            return false;
        }
    }

    {
        VARIANT vEmpty;
        VariantInit(&vEmpty);

        //UpdateData(TRUE);

        USES_CONVERSION;
        BSTR bstrURL = SysAllocString(A2OLE((const char *)url));

        HRESULT hr = browser->Navigate(bstrURL, &vEmpty, &vEmpty, &vEmpty, &vEmpty);
        if (SUCCEEDED(hr))
        {
            browser->put_Visible(VARIANT_TRUE);
        }
        else
        {
            browser->Quit();
        }

        SysFreeString(bstrURL);
        //browser->Release();
        //browser = NULL;
    }
/*if (np)
   {
  ShellExecute(NULL, "open", url,
                parameters[0], NULL, SW_SHOWNORMAL);
   }
   else
   {
  ShellExecute(NULL, "open", url,
                NULL, NULL, SW_SHOWNORMAL);
   }*/
#endif

    delete[] buf;
    return (result >= 0); // _WIN32
}

void SystemCover::createMenu()
{

    cbg = new ui::ButtonGroup("ViewPointsGroup", this);
    vrmlMenu = new ui::Menu("VrmlMenu", this);
    vrmlMenu->setText("VRML");
    viewpointGroup = new ui::Group(vrmlMenu, "Viewpoints");

    reloadButton = new ui::Action(vrmlMenu, "Reload");
    reloadButton->setCallback([this](){
        coVRFileManager::instance()->reloadFile();
    });
    addVPButton = new ui::Action(vrmlMenu, "SaveViewpoint");
    addVPButton->setText("Save viewpoint");
    addVPButton->setCallback([this](){
        printViewpoint();
    });

    vrmlTab = new coTUITab("Vrml97", coVRTui::instance()->mainFolder->getID());
    vrmlTab->setPos(0, 0);

    reload = new coTUIButton("Reload", vrmlTab->getID());
    reload->setEventListener(this);
    reload->setPos(0, 0);

    saveViewpoint = new coTUIButton("Save Viewpoint", vrmlTab->getID());
    saveViewpoint->setEventListener(this);
    saveViewpoint->setPos(0, 1);

    saveAnimation = new coTUIToggleButton("Save Animation", vrmlTab->getID());
    saveAnimation->setEventListener(this);
    saveAnimation->setPos(0, 2);
}

// reload has been moved to tabletPressEvent because it causes a destruction of
// the vrml menu and thus the TUI Item and this causes a crash when
// tabletPressEvent is called after tabletEvent and the menu item is no longer
// valid.
// nothing else is done after tabletPressEvent thus we should be safe.
void SystemCover::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == reload)
    {
        coVRFileManager::instance()->reloadFile();
    }
}

void SystemCover::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == saveViewpoint)
    {
        printViewpoint();
    }
    else if (tUIItem == saveAnimation)
    {
        if (saveAnimation->getState())
            startCapture();
        else
            stopCapture();
    }
    else
    {
        for (list<ViewpointEntry *>::iterator it = viewpointEntries.begin();
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
#define INTERVAL 0.5
void SystemCover::update()
{
    System::update();
    if (record)
    {
        static double oldTime = 0;
        double time = cover->frameTime();
        if (time - oldTime > INTERVAL)
        {
            oldTime = time;
            osg::Matrix mat = cover->getXformMat();
            osg::Matrix rotMat;

            rotMat.makeRotate(M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));
            mat.preMult(rotMat);
            rotMat.makeRotate(-M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));
            mat.postMult(rotMat);

            osg::Quat q;
            q.set(mat);
            osg::Quat::value_type orient[4];
            q.getRotate(orient[3], orient[0], orient[1], orient[2]);
            rotMat.makeRotate(-orient[3], orient[0], orient[1], orient[2]);
            mat.postMult(rotMat);
            osg::Vec3 Trans = mat.getTrans();

            positions[frameNumber * 3] = Trans[0] / (-cover->getScale());
            positions[frameNumber * 3 + 1] = Trans[1] / (-cover->getScale());
            positions[frameNumber * 3 + 2] = Trans[2] / (-cover->getScale());
            orientations[frameNumber * 4] = orient[0];
            orientations[frameNumber * 4 + 1] = orient[1];
            orientations[frameNumber * 4 + 2] = orient[2];
            orientations[frameNumber * 4 + 3] = -orient[3];
            frameNumber++;
            if (frameNumber >= 1200)
            {
                stopCapture();
            }
        }
    }
}

bool SystemCover::doOptimize()
{
	return m_optimize;
}

void SystemCover::startCapture()
{
    if (record)
    {
        stopCapture();
    }
    char fileName[100];
    sprintf(fileName, "Animation.wrl");
    fileNumber++;
    frameNumber = 0;
    fp = fopen(fileName, "a+");
    positions = new float[3 * 1202];
    orientations = new float[4 * 1202];
    if (fp)
    {
        record = true;
    }
    else
        perror("Animation.wrl");
}
void SystemCover::stopCapture()
{
    if (!record)
    {
        return;
    }
    fprintf(fp, "#VRML V2.0 utf8\n\nDEF Camera%d Viewpoint {  \nposition %f %f %f\n  orientation %f %f %f %f\n  fieldOfView 0.6024 description \"Camera%d\"\n}\n", fileNumber, positions[0], positions[1], positions[2], orientations[0], orientations[1], orientations[2], orientations[3], fileNumber);
    fprintf(fp, "DEF Camera%d-TIMER TimeSensor { loop TRUE cycleInterval %f }\n", fileNumber, frameNumber * INTERVAL);
    fprintf(fp, "DEF Camera%d-POS-INTERP PositionInterpolator {\n key [\n", fileNumber);
    int i = 0;
    int z = 0;
    for (i = 0; i < frameNumber; i++)
    {
        fprintf(fp, " %f,", (float)i / (frameNumber - 1));
        z++;
        if (z > 10)
        {
            z = 0;
            fprintf(fp, "\n");
        }
    }
    fprintf(fp, "]\nkeyValue [\n");

    for (i = 0; i < frameNumber; i++)
    {
        fprintf(fp, " %f %f %f,", positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        z++;
        if (z > 3)
        {
            z = 0;
            fprintf(fp, "\n");
        }
    }

    fprintf(fp, "] }\n");

    fprintf(fp, "DEF Camera%d-ROT-INTERP OrientationInterpolator {\n key [\n", fileNumber);
    for (i = 0; i < frameNumber; i++)
    {
        fprintf(fp, " %f,", (float)i / (frameNumber - 1));
        z++;
        if (z > 10)
        {
            z = 0;
            fprintf(fp, "\n");
        }
    }
    fprintf(fp, "]\nkeyValue [\n");

    for (i = 0; i < frameNumber; i++)
    {
        fprintf(fp, " %f %f %f %f,", orientations[i * 4], orientations[i * 4 + 1], orientations[i * 4 + 2], orientations[i * 4 + 3]);
        z++;
        if (z > 3)
        {
            z = 0;
            fprintf(fp, "\n");
        }
    }

    fprintf(fp, "] }\n");
    fprintf(fp, "ROUTE Camera%d-TIMER.fraction_changed TO Camera%d-POS-INTERP.set_fraction\n", fileNumber, fileNumber);
    fprintf(fp, "ROUTE Camera%d-POS-INTERP.value_changed TO Camera%d.set_position\n", fileNumber, fileNumber);
    fprintf(fp, "ROUTE Camera%d-TIMER.fraction_changed TO Camera%d-ROT-INTERP.set_fraction\n", fileNumber, fileNumber);
    fprintf(fp, "ROUTE Camera%d-ROT-INTERP.value_changed TO Camera%d.set_orientation\n", fileNumber, fileNumber);

    fclose(fp);
    record = false;
    delete[] positions;
    delete[] orientations;
}

void SystemCover::printViewpoint()
{
    //write vrml Viewpoint
    // only write VRML viewpoints if VRML_WRITE_VIEWPOINTS is set

    osg::Matrix mat = cover->getXformMat();
    osg::Matrix rotMat;

    rotMat.makeRotate(M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));
    mat.preMult(rotMat);
    rotMat.makeRotate(-M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));
    mat.postMult(rotMat);

    osg::Quat q;
    q.set(mat);
    osg::Quat::value_type orient[4];
    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
    rotMat.makeRotate(-orient[3], orient[0], orient[1], orient[2]);
    mat.postMult(rotMat);
    osg::Vec3 Trans = mat.getTrans();

    fprintf(stderr, "\nDEF Camera6 Viewpoint\n");
    fprintf(stderr, "{\n");
    fprintf(stderr, "position %f %f %f \n", Trans[0] / (-cover->getScale()), Trans[1] / (-cover->getScale()), Trans[2] / (-cover->getScale()));
    fprintf(stderr, "orientation %f %f %f %f\n", orient[0], orient[1], orient[2], -orient[3]);
    fprintf(stderr, "description \"NoName\"\n");
    fprintf(stderr, "type \"Free\"\n");
    fprintf(stderr, "}\n");
    fprintf(stderr, "\nscale %f\n", cover->getScale());

    rotMat.makeRotate(M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));
    mat.postMult(rotMat);
    Trans = mat.getTrans();
    rotMat.makeRotate(-orient[3], orient[0], orient[1], orient[2]);
    mat.postMult(rotMat);

    q.set(mat);
    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
    cerr << endl << "3DSmax Camera:" << endl;
    fprintf(stderr, "position %f %f %f \n", Trans[0] / (-cover->getScale()), Trans[1] / (-cover->getScale()), Trans[2] / (-cover->getScale()));
    fprintf(stderr, "orientation %f %f %f %f\n", orient[0], orient[1], orient[2], -orient[3] / M_PI * 180);
    fprintf(stderr, "matrix3 [ %f, %f, %f ] [ %f, %f, %f ] [ %f, %f, %f ] [ %f, %f, %f ] \n", mat(0, 0), mat(0, 1), mat(0, 2), mat(1, 0), mat(1, 1), mat(1, 2), mat(2, 0), mat(2, 1), mat(2, 2), Trans[0] / (-cover->getScale()), Trans[1] / (-cover->getScale()), Trans[2] / (-cover->getScale()));
}

void SystemCover::destroyMenu()
{
    delete vrmlMenu;
    delete cbg;

    delete saveAnimation;
    delete saveViewpoint;
    delete reload;
    delete vrmlTab;
}

double SystemCover::time()
{
    return cover->frameTime();
}

std::string SystemCover::remoteFetch(const std::string& filename)
{
	return coVRFileManager::instance()->findOrGetFile(filename);
}

int SystemCover::getFileId(const std::string &url)
{
	return coVRFileManager::instance()->getFileId(url);
}

void SystemCover::setSyncMode(const char *mode)
{
    coVRCollaboration::instance()->setSyncMode(mode);
}

bool SystemCover::isMaster()
{
    return coVRCommunication::instance()->isMaster();
}

void SystemCover::becomeMaster()
{
    coVRCommunication::instance()->becomeMaster();
}

#if 0
void SystemCover::setBuiltInFunctionState(const char *fname, int val)
{
    cover->setBuiltInFunctionState(fname, val);
}

void SystemCover::setBuiltInFunctionValue(const char *fname, float val)
{
    cover->setBuiltInFunctionValue(fname, val);
}

void SystemCover::callBuiltInFunctionCallback(const char *fname)
{
    cover->callBuiltInFunctionCallback(fname);
}
#endif

Player *SystemCover::getPlayer()
{
    return NULL;
}

VrmlMessage *SystemCover::newMessage(size_t len)
{
    VrmlMessage *msg = new VrmlMessage(len + sizeof(int));
    int tag = coVRPluginSupport::VRML_EVENT;
    msg->append(tag);

    return msg;
}

void SystemCover::sendAndDeleteMessage(VrmlMessage *msg)
{
    if (msg->pos > msg->size)
    {
        cerr << "SystemCover::sendAndDeleteMessage: msg->pos > msg->size !!!" << endl;
    }
    Message message{ COVISE_MESSAGE_RENDER_MODULE, DataHandle{msg->buf, msg->size, false} };
    cover->sendVrbMessage(&message);

    delete msg;
}

bool SystemCover::hasRemoteConnection()
{
    return coVRCommunication::instance()->collaborative();
}

long SystemCover::getMaxHeapBytes()
{
    long max_heap = coCoviseConfig::getInt("COVER.Plugin.Vrml97.MaxHeapBytes", -1);
    if (max_heap < 2)
        max_heap = -1;
    else if (max_heap > 10000)
        max_heap = -1;
    else
        max_heap *= 1024L * 1024L;
    return max_heap;
}

bool SystemCover::getHeadlight()
{
    return coCoviseConfig::isOn("COVER.Headlight", true);
}

void SystemCover::setHeadlight(bool enable)
{
    if (enable)
    {
        //setBuiltInFunctionState("Headlight", 1);
        coVRLighting::instance()->switchLight(coVRLighting::instance()->headlight, true);
    }
    else
    {
        //setBuiltInFunctionState("Headlight", 0);
        coVRLighting::instance()->switchLight(coVRLighting::instance()->headlight, false);
    }
}

bool SystemCover::getPreloadSwitch()
{
    static bool firstTime = true;
    static bool preload = true;
    if (firstTime)
    {
        preload = !coCoviseConfig::isOn("COVER.Plugin.Vrml97.NoSwitchChildrenPreload", false);
        firstTime = false;
    }
    return preload;
}

float SystemCover::getSyncInterval()
{
    return coVRCollaboration::instance()->getSyncInterval();
}

void SystemCover::addViewpoint(VrmlScene *scene, VrmlNodeViewpoint *viewpoint)
{
    if (viewpointEntries.empty())
        viewPointCount = 0;
    ++viewPointCount;

    // add viewpoint to menu
    ViewpointEntry *vpe = new ViewpointEntry(viewpoint, scene);
    auto menuEntry = new ui::Button(viewpointGroup, "Viewpoint"+std::to_string(viewPointCount), cbg);
    menuEntry->setText(viewpoint->description());
    menuEntry->setState(viewpoint == scene->bindableViewpointTop(), true);
    vpe->setMenuItem(menuEntry);
    viewpointEntries.push_back(vpe);
}

bool SystemCover::removeViewpoint(VrmlScene *scene, const VrmlNodeViewpoint *viewpoint)
{
    (void)scene;
    maxEntryNumber = 0;
    for (list<ViewpointEntry *>::iterator it = viewpointEntries.begin();
         it != viewpointEntries.end(); it++)
    {
        if ((*it)->getViewpoint() == viewpoint)
        {
            delete *it;
            viewpointEntries.erase(it);

            for (it = viewpointEntries.begin();
                 it != viewpointEntries.end(); it++)
            {
                maxEntryNumber = (*it)->entryNumber;
            }
            return true;
        }
        else
        {
            maxEntryNumber = (*it)->entryNumber;
        }
    }

    return false;
}

bool SystemCover::setViewpoint(VrmlScene *scene, const VrmlNodeViewpoint *viewpoint)
{
    std::cerr << "setting viewpoint to " << viewpoint->name() << std::endl;
    (void)scene;

    list<ViewpointEntry *>::iterator it = viewpointEntries.begin();
    while (it != viewpointEntries.end())
    {
        if ((*it)->getViewpoint() == viewpoint)
        {
            (*it)->getTUIItem()->setState(true);
        }
        else
        {
            (*it)->getTUIItem()->setState(false);
        }
        it++;
    }
    it = viewpointEntries.begin();

    while (it != viewpointEntries.end())
    {
        if ((*it)->getViewpoint() == viewpoint)
        {
            cbg->setActiveButton((*it)->getMenuItem());
            return true;
        }
        else
        {
            it++;
        }
    }

    return false;
}

void SystemCover::setCurrentFile(const char *filename)
{
    Vrml97Plugin::plugin->isNewVRML = true;
}

void SystemCover::setMenuVisibility(bool vis)
{
    vrmlMenu->setVisible(vis);
}

void SystemCover::setTimeStep(int ts) // set the timestep number for COVISE Animations
{
    coVRAnimationManager::instance()->requestAnimationFrame(ts);
}

void SystemCover::setActivePerson(int p) // set the active Person
{
    Input::instance()->setActivePerson(p);
}

void SystemCover::setNavigationType(std::string modeName)
{
    coVRNavigationManager::instance()->setNavMode(modeName);
}

void SystemCover::setNavigationStepSize(double stepsize)
{
    coVRNavigationManager::instance()->setStepSize(stepsize);
}

void SystemCover::setNavigationDriveSpeed(double speed)
{
    coVRNavigationManager::instance()->setDriveSpeed(speed);
}

void SystemCover::setNearFar(float nearC, float farC)
{
    coVRConfig::instance()->setNearFar(nearC, farC);
}

double SystemCover::getAvatarHeight()
{
    return VRSceneGraph::instance()->floorHeight();
}

int SystemCover::getNumAvatars()
{
    return coVRPartnerList::instance()->numberOfPartners(); //maybe return number of partners in current session instead
}

bool SystemCover::getAvatarPositionAndOrientation(int num, float pos[3], float ori[4])
{
    coVRPartner *p = coVRPartnerList::instance()->get(num);
    if (!p || !p->getAvatar() || !p->getAvatar()->initialized)
    {
        return false;
    }
    osg::Matrix feet;
    feet = p->getAvatar()->feetTransform->getMatrix();

    return getPositionAndOrientationFromMatrix(feet, pos, ori);
}

bool SystemCover::getViewerPositionAndOrientation(float pos[3], float ori[4])
{
    osg::Matrix invbase = cover->getInvBaseMat();
    osg::Matrix headmat = cover->getViewerMat();
    headmat *= invbase;

    return getPositionAndOrientationFromMatrix(headmat, pos, ori);
}

bool SystemCover::getLocalViewerPositionAndOrientation(float pos[3], float ori[4])
{
    osg::Matrix headmat = cover->getViewerMat();

    return getPositionAndOrientationFromMatrix(headmat, pos, ori);
}

bool SystemCover::getViewerFeetPositionAndOrientation(float pos[3], float ori[4])
{
    osg::Matrix invbase = cover->getInvBaseMat();
    osg::Matrix headmat = cover->getViewerMat();
    osg::Vec3 toFeet;
    toFeet = headmat.getTrans();
    toFeet[2] = getAvatarHeight();
    osg::Matrix feetmat;
    feetmat.makeTranslate(toFeet[0], toFeet[1], toFeet[2]);
    feetmat *= invbase;

    return getPositionAndOrientationFromMatrix(feetmat, pos, ori);
}

bool SystemCover::getPositionAndOrientationFromMatrix(const double *M, float pos[3], float ori[4])
{
    osg::Matrix mat;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            mat(i, j) = M[i * 4 + j];
    return getPositionAndOrientationFromMatrix(mat, pos, ori);
}

void SystemCover::getInvBaseMat(double *M)
{
    osg::Matrix mat = cover->getInvBaseMat();
    osg::Matrix rotMat;
    rotMat.makeRotate(-M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));
    mat.preMult(rotMat);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            M[i * 4 + j] = mat(i, j);
}

void SystemCover::getPositionAndOrientationOfOrigin(const double *M, float pos[3], float ori[4])
{
    osg::Matrix VRMLTrans;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            VRMLTrans(i, j) = M[i * 4 + j];

    osg::Matrix mat = cover->getInvBaseMat();
    osg::Matrix rotMat;
    rotMat.makeRotate(-M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));
    osg::Matrix VRMLRotMat;
    VRMLRotMat.makeRotate(M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));
    mat.postMult(rotMat);
    mat.postMult(VRMLTrans);
    mat.preMult(VRMLRotMat);

    //mat.getOrthoCoord(&coord);
    coCoord coord(mat);
    //mat.makeCoord(&coord);
    coord.makeMat(mat);
    if (pos)
    {
        for (int i = 0; i < 3; i++)
            pos[i] = mat(3, i);
    }
    if (ori)
    {
        osg::Quat q;
        //m.getOrthoQuat(q);
        q.set(mat);
        osg::Quat::value_type orient[4];
        q.getRotate(orient[3], orient[0], orient[1], orient[2]);
        for (int i = 0; i < 4; i++)
        {
            ori[i] = orient[i];
        }
    }
}

bool SystemCover::getPositionAndOrientationFromMatrix(const osg::Matrix &mat, float pos[3], float ori[4])
{
    if (!ViewerOsg::viewer)
        return false;

    osg::Matrix VRMLRootMat = ViewerOsg::viewer->VRMLRoot->getMatrix();
    osg::Matrix invVRMLRootMat;
    invVRMLRootMat.invert(VRMLRootMat);
    osg::Matrix m = VRMLRootMat * mat * invVRMLRootMat;
    coCoord coord(m);
    coord.makeMat(m);
    if (pos)
    {
        for (int i = 0; i < 3; i++)
            pos[i] = m(3, i);
    }
    if (ori)
    {
        osg::Quat q;
        q.set(m);
        osg::Quat::value_type orient[4];
        q.getRotate(orient[3], orient[0], orient[1], orient[2]);
        for (int i = 0; i < 4; i++)
        {
            ori[i] = orient[i];
        }
    }

    return true;
}

void SystemCover::transformByMatrix(const double *M, float pos[3], float ori[4])
{
    osg::Matrix mat;
    osg::Matrix imat;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            imat(i, j) = M[i * 4 + j];

    mat.invert(imat);
    osg::Matrix mat2;
    osg::Matrix rotMat;
    mat2.makeTranslate(pos[0], pos[1], pos[2]);
    rotMat.makeRotate(ori[3], osg::Vec3(ori[0], ori[1], ori[2]));
    osg::Matrix m = rotMat * mat2 * mat;
    // remove scale and shear from matrix the question is why... coCoord has issues with angles < 0.002 rad
    //coCoord coord(m);
    //coord.makeMat(m);
    if (pos)
    {
        for (int i = 0; i < 3; i++)
            pos[i] = m(3, i);
    }
    if (ori)
    {
        osg::Quat q;
        //m.getOrthoQuat(q);
        q.set(m);
        osg::Quat::value_type orient[4];
        q.getRotate(orient[3], orient[0], orient[1], orient[2]);
        for (int i = 0; i < 4; i++)
        {
            ori[i] = orient[i];
        }
    }
}

bool SystemCover::loadPlugin(const char *name)
{
    return cover->addPlugin(name);
}

std::string SystemCover::getConfigEntry(const char *key)
{
    return coCoviseConfig::getEntry(key);
}

bool SystemCover::getConfigState(const char *key, bool defaultVal)
{
    return coCoviseConfig::isOn(key, defaultVal);
}

System::CacheMode SystemCover::getCacheMode() const
{
    return cacheMode;
}

std::string SystemCover::getCacheName(const char *url, const char *pathname) const
{
    namespace fs = boost::filesystem;

    (void)url;

    if (!pathname)
        return std::string();
    if (!strcmp(pathname, ""))
        return std::string();

    fs::path p(pathname);
    fs::path name = p.filename();
    if (name.empty())
        return std::string();

    auto pathstat = status(p);
    if (!fs::exists(pathstat))
        return std::string();

    fs::path dir = p.remove_filename();

    fs::path cache = dir;
    cache /= ".covercache";
    auto stat = status(cache);
    if (!fs::exists(stat))
    {
        try
        {
            if (fs::create_directory(cache))
                stat = status(cache);
        }
        catch (fs::filesystem_error)
        {
            std::cerr << "Vrml: SystemCover:getCacheName: could not create cache directory " << cache.string() << std::endl;
        }
    }
    if (!fs::is_directory(stat))
    {
        std::cerr << "Vrml: SystemCover::getCacheName(pathname=" << pathname << "): not a directory" << std::endl;
    }
    cache /= name;
    cache += cacheExt;

    return cache.string();
}

void SystemCover::storeInline(const char *name, const Viewer::Object d_viewerObject)
{
    if (d_viewerObject)
    {
        osg::Node *osgNode = ((osgViewerObject *)d_viewerObject)->pNode.get();
        if (osgNode)
        {

            // run optimization over the scene graph
			if (m_optimize)
			{
				osgUtil::Optimizer optimzer;
				optimzer.optimize(osgNode);
			}
            if (coVRMSController::instance()->isMaster() || !coVRFileManager::instance()->isInSharedDir(name))
                osgDB::writeNodeFile(*osgNode, name);
        }
    }
}


int SystemCover::isUTF8(const char* data, size_t size)
{
    const unsigned char* str = (unsigned char*)data;
    const unsigned char* end = str + size;
    unsigned char byte;
    unsigned int code_length, i;
    uint32_t ch;
    while (str != end) {
        byte = *str;
        if (byte <= 0x7F) {
            /* 1 byte sequence: U+0000..U+007F */
            str += 1;
            continue;
        }

        if (0xC2 <= byte && byte <= 0xDF)
            /* 0b110xxxxx: 2 bytes sequence */
            code_length = 2;
        else if (0xE0 <= byte && byte <= 0xEF)
            /* 0b1110xxxx: 3 bytes sequence */
            code_length = 3;
        else if (0xF0 <= byte && byte <= 0xF4)
            /* 0b11110xxx: 4 bytes sequence */
            code_length = 4;
        else {
            /* invalid first byte of a multibyte character */
            return 0;
        }

        if (str + (code_length - 1) >= end) {
            /* truncated string or invalid byte sequence */
            return 0;
        }

        /* Check continuation bytes: bit 7 should be set, bit 6 should be
         * unset (b10xxxxxx). */
        for (i = 1; i < code_length; i++) {
            if ((str[i] & 0xC0) != 0x80)
                return 0;
        }

        if (code_length == 2) {
            /* 2 bytes sequence: U+0080..U+07FF */
            ch = ((str[0] & 0x1f) << 6) + (str[1] & 0x3f);
            /* str[0] >= 0xC2, so ch >= 0x0080.
               str[0] <= 0xDF, (str[1] & 0x3f) <= 0x3f, so ch <= 0x07ff */
        }
        else if (code_length == 3) {
            /* 3 bytes sequence: U+0800..U+FFFF */
            ch = ((str[0] & 0x0f) << 12) + ((str[1] & 0x3f) << 6) +
                (str[2] & 0x3f);
            /* (0xff & 0x0f) << 12 | (0xff & 0x3f) << 6 | (0xff & 0x3f) = 0xffff,
               so ch <= 0xffff */
            if (ch < 0x0800)
                return 0;

            /* surrogates (U+D800-U+DFFF) are invalid in UTF-8:
               test if (0xD800 <= ch && ch <= 0xDFFF) */
            if ((ch >> 11) == 0x1b)
                return 0;
        }
        else if (code_length == 4) {
            /* 4 bytes sequence: U+10000..U+10FFFF */
            ch = ((str[0] & 0x07) << 18) + ((str[1] & 0x3f) << 12) +
                ((str[2] & 0x3f) << 6) + (str[3] & 0x3f);
            if ((ch < 0x10000) || (0x10FFFF < ch))
                return 0;
        }
        str += code_length;
    }
    return 1;
}

Viewer::Object SystemCover::getInline(const char *name)
{
    osg::ref_ptr<osg::Group> g = new osg::Group;

    std::string validFileName(name);
#ifdef WIN32
    if (isUTF8(name, strlen(name)))
    {
        validFileName = boost::locale::conv::from_utf<char>(validFileName,"ISO-8859-1");// we hope  the system locale is Latin1
    }
#endif


    coVRFileManager::instance()->loadFile(validFileName.c_str(), NULL, g);

    if (g->getNumChildren() > 0)
    {
        osg::Node *loadedNode = g->getChild(0);
        loadedNode->ref();
        g->removeChild(loadedNode);
        loadedNode->unref_nodelete(); //refcount now back to 0 but the node is not deleted
        //will be added to something later on and thus deleted when removed from there.
        return ((Viewer::Object)loadedNode);
    }

    return 0L;
}

void SystemCover::insertObject(Viewer::Object d_viewerObject, Viewer::Object sgObject)
{
    ((osgViewerObject *)d_viewerObject)->pNode = (osg::Node *)sgObject;
    ViewerOsg::viewer->addToScene(((osgViewerObject *)d_viewerObject));
}

float SystemCover::getLODScale()
{
    return coVRConfig::instance()->getLODScale();
}

float SystemCover::defaultCreaseAngle()
{
    static float angle = 0.0;
    static bool firstTime = true;
    if (firstTime)
    {
        angle = coCoviseConfig::getFloat("COVER.Plugin.Vrml97.DefaultCreaseAngle", M_PI_4);
        firstTime = false;
    }
    return angle;
}
