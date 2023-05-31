/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)2008 HLRS  **
**                                                                          **
** Description: TouchTable Plugin (does TouchTable )                        **
**                                                                          **
**                                                                          **
** Author: B. Burbaum und                                                  **
**         U. Woessner		                                               **
**                                                                          **
** History:  								                                **
** Nov-08  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "TouchTablePlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/RenderObject.h>
#include <config/CoviseConfig.h>
#include <cover/ARToolKit.h>
#include <cover/coVRConfig.h>
#include <osg/GraphicsContext>

#include <opencv/cv.h>
//#include <highgui.h>

#include "TouchScreenDevice.h"
#include "TouchData.h"
#include "TouchlibFilter.h"

using namespace touchlib;

TouchTablePlugin::TouchTablePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "TouchTablePlugin::TouchTablePlugin\n");
    screen = TouchScreenDevice::getTouchScreen();
    screen->setDebugMode(false);
    std::string configFile = std::string("config/") + coCoviseConfig::getEntry("value", "COVER.Plugin.TouchTable.ConfigFile", "config.touchtable.xml");
    buttonState = 0;
    const char *fullFileName = coVRFileManager::instance()->getName(configFile.c_str());
    if (fullFileName == NULL || !screen->loadConfig(fullFileName)) // in case off missing Configfile use filters below
    {
        std::string label;
        label = screen->pushFilter("dsvlcapture");
        screen->setParameter(label, "source", "cam");
        screen->pushFilter("mono");
        screen->pushFilter("smooth");
        screen->pushFilter("backgroundremove");
        label = screen->pushFilter("brightnesscontrast");
        screen->setParameter(label, "brightness", "0.1");
        screen->setParameter(label, "contrast", "0.4");
        label = screen->pushFilter("rectify");
        screen->setParameter(label, "level", "25");
    }

    std::string bgLabel = screen->findFirstFilter("backgroundremove");
    std::string recLabel = screen->findFirstFilter("rectify");
    screen->registerListener(this);
    // Note: Begin processing should only be called after the screen is set up

    screen->setParameter(bgLabel, "mask", (char *)screen->getCameraPoints());
    screen->setParameter(bgLabel, "capture", "");

    screen->beginProcessing();
    screen->beginTracking();

    TouchTableTab = new coTUITab("TouchTable", coVRTui::instance()->mainFolder->getID());
    TouchTableTab->setPos(0, 0);

    for (int Zeile = 0; Zeile < 20; Zeile++)
    {
        for (int Spalte = 0; Spalte < 7; Spalte++)
        {
            MyFirstLabel = new coTUILabel(" ", TouchTableTab->getID());
            MyFirstLabel->setPos(Spalte, Zeile);
        }
    }

    //	  MyFirstLabel = new coTUILabel("MyFirstLabel  #", TouchTableTab->getID());
    //    MyFirstLabel->setPos(2, 4);

    // ----XXX
    //------
    std::vector<Filter *> myfilters = screen->getAllFilters();
    std::vector<Filter *>::iterator filterIter;
    filterIter = myfilters.begin();
    ParameterMap pMap;
    int NoFilter = 0;
    int NoSlider = 0;
    while (filterIter != myfilters.end())
    {
        std::string MyFilter = (*filterIter)->getType(); // getName() Holt den Filter Name; Stimmt das ??

        coTUIToggleButton *locButton = new coTUIToggleButton(MyFilter.c_str(), TouchTableTab->getID());

        //MyButton[NoFilter]  = new coTUIToggleButton(MyFilter,TouchTableTab->getID());
        locButton->setPos(0, (2 * (NoFilter + 1)));
        locButton->setEventListener(this);
        isNoVisible[NoFilter] = false;
        locButton->setState(isNoVisible[NoFilter]);

        myButtons.push_back(locButton);

        (*filterIter)->getParameters(pMap);

        ParameterMap::iterator pMapItr; // Create an iterator to cycle through each attribute

        NoSlider = 0; //           OK

        for (pMapItr = pMap.begin(); pMapItr != pMap.end(); pMapItr++) // Cycle through each attribute
        {
            std::string mytype = pMapItr->first.c_str(); // Create a <[attribute type]> tag
            std::string myvalue = pMapItr->second.c_str(); // Add value="[attribute value]"
            int myval = atoi(myvalue.c_str());

            MyFirstLabel = new coTUILabel(mytype.c_str(), TouchTableTab->getID());
            MyFirstLabel->setPos((2 * NoSlider + 1), (2 * (NoFilter + 1)) - 1);

            FilterSlider[NoFilter][NoSlider] = new coTUIFloatSlider(mytype.c_str(), TouchTableTab->getID());
            FilterSlider[NoFilter][NoSlider]->setPos((2 * NoSlider + 1), (2 * (NoFilter + 1)));
            FilterSlider[NoFilter][NoSlider]->setEventListener(this);
            FilterSlider[NoFilter][NoSlider]->setMin(10);
            FilterSlider[NoFilter][NoSlider]->setMax(200);

            FilterSlider[NoFilter][NoSlider]->setValue(myval);
            NoSlider++;
        }
        pMap.clear(); // Reinitialize pMap for next filter

        NoFilter++;
        filterIter++;
    }
    //  -----
    // ----XXX

    MyFirstLabel = new coTUILabel("Filter einstellen", TouchTableTab->getID());
    MyFirstLabel->setPos(2, 4);

    changeFingerSize = new coTUIToggleButton("changeFingerSize", TouchTableTab->getID());
    changeFingerSize->setPos(0, 19);
    changeFingerSize->setEventListener(this);
    isVisible = false;
    changeFingerSize->setState(isVisible);

    FingerSizeSlider = new coTUIFloatSlider("FingerSizeSlider", TouchTableTab->getID());
    FingerSizeSlider->setPos(1, 19);
    FingerSizeSlider->setEventListener(this);
    FingerSizeSlider->setMin(-0.1); // Vorzeichen !
    FingerSizeSlider->setMax(-0.001); // Vorzeichen !
    FingerSizeSlider->setValue(FingerSizeValue);

    FingerSizeValue = -0.004;
    FingerSizeSlider->setValue(FingerSizeValue);

    NoShowButton = new coTUIToggleButton("NoShow", TouchTableTab->getID());
    NoShowButton->setPos(2, 17);
    NoShowButton->setEventListener(this);
    isNixShow = false;
    NoShowButton->setState(isNixShow);

    recapture_background = new coTUIToggleButton("recapture_background", TouchTableTab->getID());
    recapture_background->setPos(4, 17);
    recapture_background->setEventListener(this);
    isRecapBGR = false;
    recapture_background->setState(isRecapBGR);
}

// this is called if the plugin is removed at runtime
TouchTablePlugin::~TouchTablePlugin()
{
    fprintf(stderr, "TouchTablePlugin::~TouchTablePlugin\n");
    TouchScreenDevice::destroy();
}

void TouchTablePlugin::tabletEvent(coTUIElement *tUIItem)
{
    std::vector<Filter *> myFilters = screen->getAllFilters();

    for (int NoFilter = 0; NoFilter < MaxFilter; NoFilter++) // --> Check all slider
    { // ---------------------
        for (int NoSlider = 0; NoSlider < MaxSlider; NoSlider++)
        {
            if (tUIItem == FilterSlider[NoFilter][NoSlider]) // Slider changed ??
            {
                Filter *locFilter = myFilters[NoFilter]; // get the Filter
                std::map<string, string>::iterator mapIter;
                ParameterMap pMap;
                locFilter->getParameters(pMap); // get pMap(with ParameterS) for the Filter
                mapIter = pMap.begin();
                for (int i = 0; i < NoSlider; i++) // winding  pMap to correct NoSlider, via mapIter
                {
                    mapIter++;
                }
                std::stringstream strm; // for typechange
                strm << FilterSlider[NoFilter][NoSlider]->getValue(); // change Type I? to stringstream

                (*mapIter).second = strm.str(); // Change stringstream to string

                myFilters[NoFilter]->setParameter((*mapIter).first.c_str(), (*mapIter).second.c_str()); // Write back changes to Filter

                //    hnier gehte weiter  locFilter->getParameters(pMap);
            }
        }
    }

    std::vector<coTUIToggleButton *>::iterator myButtonsItr; // --> Check all Buttons
    int NoFilter = 0; // ---------------------
    for (myButtonsItr = myButtons.begin(), NoFilter = 0; myButtonsItr != myButtons.end(); myButtonsItr++, NoFilter++) // Cycle through each myButtons
    {
        if (tUIItem == *myButtonsItr) // Button  changed ??
        {
            isNoVisible[NoFilter] = (*myButtonsItr)->getState();

            int NoFilterZwo = 0;
            std::vector<coTUIToggleButton *>::iterator myButtonsItrZwo;
            for (myButtonsItrZwo = myButtons.begin(), NoFilterZwo = 0; myButtonsItrZwo != myButtons.end(); myButtonsItrZwo++, NoFilterZwo++) // Cycle through each myButtonsZwo, to switch of
            {
                if (*myButtonsItr != *myButtonsItrZwo)
                {
                    (*myButtonsItrZwo)->setState(false);
                    isNoVisible[NoFilterZwo] = false;
                }
            }
        }
    }
    if (tUIItem == changeFingerSize)
    {
        isVisible = changeFingerSize->getState();
    }
    if (tUIItem == NoShowButton)
    {
        isNixShow = NoShowButton->getState();
    }
    if (tUIItem == recapture_background)
    {
        isRecapBGR = recapture_background->getState();
    }
    if (tUIItem == FingerSizeSlider)
    {
        FingerSizeValue = FingerSizeSlider->getValue();
    }
}

void TouchTablePlugin::tabletPressEvent(coTUIElement *tUIItem)
{
}
// this will be called wen a key is pressed
void TouchTablePlugin::key(int type, int keySym, int mod)
{

    if (keySym == 98) // b = recapture background
    {
        std::string bgLabel = screen->findFirstFilter("backgroundremove");
        screen->setParameter(bgLabel, "capture", "");
    }

    if (keySym == 114) // r = auto rectify..
    {
        std::string recLabel = screen->findFirstFilter("rectify");
        screen->setParameter(recLabel, "level", "auto");
    }
}

// this will be called in PreFrame
void TouchTablePlugin::preFrame()
{

    screen->getEvents();

    //		std::vector<coTUIToggleButton*> myButtons;  aus *.h
    std::vector<coTUIToggleButton *>::iterator myButtonsItr;
    int NoFilter = 0;
    for (myButtonsItr = myButtons.begin(), NoFilter = 0; myButtonsItr != myButtons.end(); myButtonsItr++, NoFilter++) // Cycle through each myButtons
    {
        if (isNoVisible[NoFilter])
        {
            std::string recLabel = screen->findFirstFilter((*myButtonsItr)->getName().c_str());
            IplImage *image = screen->getFilterImage(recLabel);
            if (image)
            {
                if (image->nChannels == 1)
                    ARToolKit::instance()->videoMode = GL_LUMINANCE;
                if (image->nChannels == 3)
                    ARToolKit::instance()->videoMode = GL_RGB;
                if (image->nChannels == 4)
                    ARToolKit::instance()->videoMode = GL_RGBA;
                if (image->depth == IPL_DEPTH_1U)
                    ARToolKit::instance()->videoDepth = 1;
                else if (image->depth == IPL_DEPTH_8U)
                    ARToolKit::instance()->videoDepth = image->nChannels;
                ARToolKit::instance()->videoData = (unsigned char *)image->imageData;
                ARToolKit::instance()->videoWidth = image->width;
                ARToolKit::instance()->videoHeight = image->height;
                ARToolKit::instance()->flipH = false;
            }
            else
            {
                if ((*myButtonsItr)->getState())
                    (*myButtonsItr)->setState(false);
            }
        }
        else if (isVisible)
        {
            std::string recLabel = screen->findFirstFilter("backgroundremove");
            IplImage *image = screen->getFilterImage(recLabel);
            if (image)
            {
                if (image->nChannels == 1)
                    ARToolKit::instance()->videoMode = GL_LUMINANCE;
                if (image->nChannels == 3)
                    ARToolKit::instance()->videoMode = GL_RGB;
                if (image->nChannels == 4)
                    ARToolKit::instance()->videoMode = GL_RGBA;
                if (image->depth == IPL_DEPTH_1U)
                    ARToolKit::instance()->videoDepth = 1;
                else if (image->depth == IPL_DEPTH_8U)
                    ARToolKit::instance()->videoDepth = image->nChannels;
                ARToolKit::instance()->videoData = (unsigned char *)image->imageData;
                ARToolKit::instance()->videoWidth = image->width;
                ARToolKit::instance()->videoHeight = image->height;
                ARToolKit::instance()->flipH = false;
            }
            else
            {
                if (changeFingerSize->getState())
                    changeFingerSize->setState(false);
            }
        }
        else if (isNixShow)
        {
            if (changeFingerSize->getState())
                changeFingerSize->setState(false);

            ARToolKit::instance()->videoWidth = 0;
            ARToolKit::instance()->videoData = NULL;
        }

        if (isRecapBGR)
        {
            std::string bgLabel = screen->findFirstFilter("backgroundremove");
            screen->setParameter(bgLabel, "capture", "");
            isRecapBGR = false;
            recapture_background->setState(isRecapBGR);
        }
    }
}

//! Notify that a finger has just been made active.
void TouchTablePlugin::fingerDown(TouchData data)
{
    if (!(data.X == 0.00 && data.Y == 0.00))
    {
        printf("Blob Detected X:%f Y:%f Area:%f width:%f  \n", data.X, data.Y, data.area, data.width);
    }
}

//! Notify that a finger has moved
void TouchTablePlugin::fingerUpdate(TouchData data)
{
    const osg::GraphicsContext::Traits *traits = coVRConfig::instance()->windows[0].window->getTraits();
    if (!traits)
        return;

    printf("BB1 Finger Update X:%f Y:%f Area:%f width:%f  \n", data.X, data.Y, data.area, data.width);
    //   printf("BB1 Finger Update X:%f Y:%f Area:%f width:%f angle=%f  id:%i idtag: %i \n", data.X, data.Y, data.area, data.width,data.angle , data.ID, data.tagID);

    cover->getMouseButton()->setWheel((int)data.angle);
    cover->handleMouseEvent(osgGA::GUIEventAdapter::DRAG, data.X * traits->width, (1.0 - data.Y) * traits->height);
    // if(data.area < -0.004 && buttonState == 0)
    if (data.area < FingerSizeValue && buttonState == 0)
    {
        buttonState = 1;
        cover->handleMouseEvent(osgGA::GUIEventAdapter::PUSH, buttonState, 0);
    }
}

//! A finger is no longer active..
void TouchTablePlugin::fingerUp(TouchData data)
{
    printf("BB2 Finger Up ID=%d\n", data.ID);

    buttonState = 0;
    cover->handleMouseEvent(osgGA::GUIEventAdapter::RELEASE, buttonState, 0);
}

COVERPLUGIN(TouchTablePlugin)
