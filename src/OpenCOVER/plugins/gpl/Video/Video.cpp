#define __STDC_CONSTANT_MACROS
#ifdef WIN32
#pragma warning (disable: 4005)
#endif
#include <config/CoviseConfig.h>

#include <cover/coVRTui.h>
#include <cover/coVRConfig.h>
#include <cover/coVRAnimationManager.h>
#include <PluginUtil/PluginMessageTypes.h>
#include <grmsg/coGRMsg.h>
#include <grmsg/coGRSnapshotMsg.h>
#ifdef _MSC_VER
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif
#ifdef HAVE_WMFSDK
#include "WINAVIVideo.h"
#endif
#ifdef HAVE_FFMPEG
#include "FFMPEGVideo.h"
#endif

#include <stdio.h>

#include <config/coConfigConstants.h>
#include "Video.h"

using namespace covise;
using namespace grmsg;

SysPlugin::~SysPlugin()
{
}

VideoPlugin::VideoPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    frameCount = 0;
    captureActive = 0;
    captureAnimActive = false;
    waitForFrame = false;
    pixels = NULL;
    time_base = 25;
    stereoEye = "";
    rowcount = 0;

    changeFilename_ = true;
    if (coVRMSController::instance()->isMaster())
        hostIndex = 0;
    else
        hostIndex = coVRMSController::instance()->getID();
    char buffer[3];
    sprintf(buffer, "%d", hostIndex);
    hostName = buffer;

    if (coVRConfig::instance()->channels[0].stereoMode == osg::DisplaySettings::RIGHT_EYE)
        stereoEye = "RIGHT";
    else if (coVRConfig::instance()->channels[0].stereoMode == osg::DisplaySettings::LEFT_EYE)
        stereoEye = "LEFT";

    sysPlug = nullptr;
    sysPlugFfmpeg = nullptr;
    sysPlugWinAvi = nullptr;
#ifdef HAVE_WMFSDK
    sysPlugWinAvi = new WINAVIPlugin;
    sysPlugWinAvi->myPlugin = this;
    if (!sysPlug)
        sysPlug = sysPlugWinAvi;
#endif
#ifdef HAVE_FFMPEG
    sysPlugFfmpeg = new FFMPEGPlugin;
    sysPlugFfmpeg->myPlugin = this;
    if (!sysPlug)
        sysPlug = sysPlugFfmpeg;
#endif

    // start record from gui
    recordFromGui_ = false;
}

bool VideoPlugin::opt_frame_size(int w, int h)
{
    char buf[1000];

    errorLabel->setLabel("");
    if (w <= 0 || h <= 0)
    {
        sprintf(buf, "INCORRECT FRAME SIZE");
        fprintf(stderr, "incorrect frame size h=%d w=%d\n", h, w);
        errorLabel->setLabel(buf);
        return (true);
    }
    if ((w % 2) != 0 || (h % 2) != 0)
    {
        sprintf(buf, "FRAME SIZE MUST BE A MULTIPLE OF 2");
        fprintf(stderr, "frame size not a multiple of 2\n");
        errorLabel->setLabel(buf);
        return (true);
    }
    return (false);
}

void VideoPlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- Plugin Video coVRGuiToRenderMsg msg=%s\n", msg.getString().c_str());

    if (msg.isValid() && msg.getType() == coGRMsg::SNAPSHOT)
    {
        auto &snapshotMsg = msg.as<coGRSnapshotMsg>();
        if (strcmp(snapshotMsg.getIntention(), "startCapturing") == 0)
        {
            if (!recordFromGui_)
            {
                recordFromGui_ = true;
                // set format to WMV
                selectFormat->setSelectedEntry(1);
                tabletEvent(selectFormat);

                // set compression codex to Windows Media Video 9
                /// sysPlug->selectCodec->setSelectedEntry(0);
                sysPlug->selectCodec->setSelectedText("Windows Media Video 9");
                tabletEvent(sysPlug->selectCodec);

                // set bitrate choice to bitrate
                //sysPlug->bitrate->setSelectedEntry(0);
                sysPlug->bitrateField->setValue(12000000);
                tabletEvent(sysPlug->bitrateField);

                // set bitrate from config file again (the profile may have overwritten it)
                sysPlug->bitrateField->setValue(bitrate_);
                tabletEvent(sysPlug->bitrateField);

                // set video size to cover window size
                int x, y, width, height;
                coVRConfig::instance()->windows[0].window->getWindowRectangle(x, y, width, height);
                if ((width % 4) != 0) // 5
                {
                    x++;
                    width--;
                } // 5
                while ((width % 4) != 0)
                    width--;

                if ((height % 2) != 0)
                    height -= 1;

                xPosField->setValue(x);
                tabletEvent(xPosField);
                yPosField->setValue(y);
                tabletEvent(yPosField);
                widthField->setValue(width);
                tabletEvent(widthField);
                heightField->setValue(height);
                tabletEvent(heightField);
                outWidthField->setValue(-1);
                tabletEvent(outWidthField);
                outHeightField->setValue(-1);
                tabletEvent(outHeightField);

                // change filename
                const char *filename = snapshotMsg.getFilename();
                fileNameField->setText(filename);
                changeFilename_ = false;
                fillFilenameField(filename, false, changeFilename_);

                coVRConfig::instance()->windows[0].window->grabFocus();
                coVRConfig::instance()->windows[0].window->raiseWindow();

                // press capture button
                captureButton->setState(true);
                tabletEvent(captureButton);
            }
        }

        else if (strcmp(snapshotMsg.getIntention(), "stopCapturing") == 0)
        {
            if (recordFromGui_)
            {
                // press capture button again
                captureButton->setState(false);
                tabletReleaseEvent(captureButton);
            }
        }
        else
        {
            fprintf(stderr, "VideoPlugin::guiToRenderMsg unknown intention %s\n", snapshotMsg.getIntention());
        }
    }
}

void VideoPlugin::key(int type, int keySym, int mod)
{
    switch (type)
    {
    case (osgGA::GUIEventAdapter::KEYDOWN):
        if (mod & osgGA::GUIEventAdapter::MODKEY_ALT)
        {
            if (keySym == 'g' || keySym == 'G')
            {
                if (captureButton->getState())
                {
                    captureButton->setState(false);
                    tabletReleaseEvent(captureButton);
                }
                else
                {
                    captureButton->setState(true);
                    tabletEvent(captureButton);
                }
            }
        }
        break;
    case (osgGA::GUIEventAdapter::KEYUP):
        break;
    default:
        cerr << "VideoPlugin::keyEvent: unknown key event type " << type << endl;
        return;
    }
}

void checkPosSize(coTUIEditIntField *pos, coTUIEditIntField *size, int refSize)
{
    if (pos->getValue() < 0)
        pos->setValue(0);
    else if (pos->getValue() > refSize)
        pos->setValue(refSize - 2);
    else if (size->getValue() < 0)
        size->setValue(2);
    if ((pos->getValue() + size->getValue()) > refSize)
        size->setValue(refSize - pos->getValue());
}

void VideoPlugin::fillFilenameField(const string &name, bool browser, bool changeName)
{
    string filename;

    if (!changeName)
    {
        fileNameField->setText(name.c_str());
        filename = name;
    }
    else if (browser)
    {
        filename = fileNameBrowser->getSelectedPath();
    }
    else if (name == "")
        filename += "temp";
    else
        filename = name;

    if (filename != "")
    {
        size_t found = filename.rfind("://");
        if (found != string::npos)
            filename = filename.substr(found + 3);

        found = sysPlug->filterList.find(".");
        string singleSuffix;
        if (found != string::npos)
            singleSuffix = sysPlug->filterList.substr(found);

        found = singleSuffix.find(";");
        if (found != string::npos)
            singleSuffix = singleSuffix.substr(0, found);

        found = filename.find(".");
        if (found != string::npos)
            filename = filename.substr(0, found);

        found = filename.rfind("RIGHT");
        if (found != string::npos)
            filename = filename.substr(0, found);
        else if ((found = filename.rfind("LEFT")) != string::npos)
            filename = filename.substr(0, found);
        string newname = filename;

        found = filename.rfind("_");
        int count = 1;

        if (found != string::npos)
        {
            count = atoi(filename.substr(found + 1).c_str());
            if (count != 0)
                filename = filename.substr(0, found);
            else
                count = 1;
        }

        if (stereoEye == "")
        {

            do
            {
                string name = newname + "-" + hostName + singleSuffix;
                ifstream ifs(name.c_str(), ifstream::in);
                if (!ifs.fail())
                {
                    ifs.close();
                    char index[5];
                    sprintf(index, "_%03d", count++);
                    newname = filename + index;
                }
                else
                    break;
            } while (true);
        }
        else
        {
            do
            {
                string name = newname + "-" + hostName + "RIGHT" + singleSuffix;
                ifstream ifs(name.c_str(), ifstream::in);
                bool fileNotExists = ifs.fail();
                if (fileNotExists)
                {
                    name = newname + "-" + hostName + "LEFT" + singleSuffix;
                    ifstream ifs(name.c_str(), ifstream::in);
                    fileNotExists = ifs.fail();
                }

                if (!fileNotExists)
                {
                    ifs.close();
                    char index[5];
                    sprintf(index, "_%03d", count++);
                    newname = filename + index;
                }
                else
                    break;
            } while (true);
        }

        filename = newname + singleSuffix;

        if (filename != "")
        {
            fileNameField->setText(filename.c_str());
            if (fileError)
            {
                fileErrorLabel->setLabel("");
                fileError = false;
            }
        }
    }
}

bool VideoPlugin::init()
{
    if (!sysPlug)
    {
        std::cerr << "VideoPlugin::init: no system plugin (install ffmpeg?)" << std::endl;
        return false;
    }

    resize = false;
    sizeError = false;
    fileError = false;
#ifdef _MSC_VER
    GL_fmt = GL_BGR_EXT;
#else
    GL_fmt = GL_BGRA;
#endif

    captureActive = false;
    waitForFrame = false;

    VideoTab = new coTUITab("Video", coVRTui::instance()->mainFolder->getID());
    VideoTab->setPos(0, rowcount);

    captureButton = new coTUIToggleButton("Capture", VideoTab->getID());
    captureButton->setEventListener(this);
    captureButton->setState(false);
    captureButton->setPos(0, rowcount);

    captureAnimButton = new coTUIToggleButton("Capture Animation", VideoTab->getID());
    captureAnimButton->setEventListener(this);
    captureAnimButton->setState(false);
    captureAnimButton->setPos(1, rowcount++);

    hostLabel = new coTUILabel("Hosts to capture from:", VideoTab->getID());
    hostLabel->setPos(0, rowcount++);
    hostLabel->setColor(Qt::black);

    captureHostButton[0] = new coTUIToggleButton("Master", VideoTab->getID(), false);
    captureHostButton[0]->setPos(0, rowcount++);
    captureHostButton[0]->setState(true);
    captureHostButton[0]->setEventListener(this);

    for (int i = 1; i <= coVRMSController::instance()->clusterSize() - 1; i++)
    {
        std::stringstream s;
        s << i;
        std::string buttonText = s.str();
        captureHostButton[i] = new coTUIToggleButton("Slave " + buttonText, VideoTab->getID(), false);
        captureHostButton[i]->setPos(i - 1, rowcount);
        captureHostButton[i]->setEventListener(this);
    }

    if (coVRMSController::instance()->clusterSize() != 1)
        rowcount++;

    GLformatButton = new coTUIComboBox("GLFormat", VideoTab->getID());
#ifdef _MSC_VER
    GLformatButton->addEntry("BGR");
    GLformatButton->addEntry("RGB");
#else
    GLformatButton->addEntry("BGRA");
    GLformatButton->addEntry("RGBA");
#endif
    GLformatButton->setEventListener(this);
    GLformatButton->setSelectedEntry(0);
    GLformatButton->setPos(0, rowcount++);

    int x, y, width, height;
    coVRConfig::instance()->windows[0].window->getWindowRectangle(x, y, width, height);
    x = y = 0;
    capPos = new coTUILabel("Capture Area Position(left,down):", VideoTab->getID());
    capPos->setPos(0, rowcount);
    capPos->setColor(Qt::black);
    xPosField = new coTUIEditIntField("xPos", VideoTab->getID());
    xPosField->setEventListener(this);
    xPosField->setValue(x);
    xPosField->setPos(1, rowcount);
    yPosField = new coTUIEditIntField("yPos", VideoTab->getID());
    yPosField->setEventListener(this);
    yPosField->setValue(y);
    yPosField->setPos(2, rowcount++);

    capArea = new coTUILabel("Capture Area (width,height):", VideoTab->getID());
    capArea->setPos(0, rowcount);
    capArea->setColor(Qt::black);
    widthField = new coTUIEditIntField("width", VideoTab->getID());
    widthField->setEventListener(this);
    widthField->setValue(width);
    widthField->setPos(1, rowcount);
    heightField = new coTUIEditIntField("height", VideoTab->getID());
    heightField->setEventListener(this);
    heightField->setValue(height);
    heightField->setPos(2, rowcount);

    sizeButton = new coTUIToggleButton("Get Window-Size", VideoTab->getID());
    sizeButton->setEventListener(this);
    sizeButton->setState(false);
    sizeButton->setPos(3, rowcount++);

    videoSize = new coTUILabel("Video Size (width,height):", VideoTab->getID());
    videoSize->setPos(0, rowcount);
    videoSize->setColor(Qt::black);
    outWidthField = new coTUIEditIntField("outWidth", VideoTab->getID());
    outWidthField->setEventListener(this);
    outWidthField->setValue(-1);
    outWidthField->setPos(1, rowcount);
    outHeightField = new coTUIEditIntField("outHeight", VideoTab->getID());
    outHeightField->setEventListener(this);
    outHeightField->setValue(-1);
    outHeightField->setPos(2, rowcount++);
    checkPosSize(xPosField, widthField, width);
    checkPosSize(yPosField, heightField, height);
    errorLabel = new coTUILabel("", VideoTab->getID());
    errorLabel->setPos(0, rowcount++);
    errorLabel->setColor(Qt::red);

    fileNameBrowser = new coTUIFileBrowserButton("File", VideoTab->getID());
    fileNameBrowser->setMode(coTUIFileBrowserButton::SAVE);
    fileNameBrowser->setPos(0, rowcount);
    fileNameBrowser->setEventListener(this);

    fileNameField = new coTUIEditField("filename", VideoTab->getID());
    fileNameField->setEventListener(this);
    fileNameField->setPos(1, rowcount);

    messageBox = new coTUIPopUp("Format and Codecs Help", VideoTab->getID());
    messageBox->setText(helpText.c_str());
    messageBox->setPos(3, rowcount++);
    messageBox->setEventListener(this);

    fileErrorLabel = new coTUILabel("", VideoTab->getID());
    fileErrorLabel->setPos(0, rowcount++);
    fileErrorLabel->setColor(Qt::red);

    frameRate = new coTUILabel("Frames/sec:", VideoTab->getID());
    frameRate->setPos(0, rowcount);
    frameRate->setColor(Qt::black);
    selectFrameRate = new coTUIComboBox("fps:", VideoTab->getID());
    selectFrameRate->setPos(1, rowcount);
    selectFrameRate->setEventListener(this);
    selectFrameRate->addEntry("25");
    selectFrameRate->addEntry("30");
    selectFrameRate->addEntry("Constant");
    selectFrameRate->setSelectedEntry(2);

    cfpsLabel = new coTUILabel("Constant frames/s:", VideoTab->getID());
    cfpsLabel->setPos(2, rowcount);
    cfpsLabel->setColor(Qt::black);
    cfpsEdit = new coTUIEditFloatField("videoPluginConstantFpsEdit", VideoTab->getID());
    //   cfpsEdit->setValue(coVRConfig::instance()->frameRate());
    cfpsEdit->setValue(25);
    cfpsEdit->setPos(3, rowcount++);
    cfpsEdit->setEventListener(this);

    selectFormat = new coTUIComboBox("Format:", VideoTab->getID());
    selectFormat->setPos(0, ++rowcount);
    selectFormat->setEventListener(this);

    sysPlug->Menu(++rowcount);
    fillFilenameField("", false);
    selectFormat->setSelectedEntry(0);

    showFrameLabel = new coTUILabel("Frames captured:", VideoTab->getID());
    showFrameLabel->setPos(0, rowcount + 10);
    showFrameLabel->setColor(Qt::black);
    showFrameCountField = new coTUIEditIntField("framecount", VideoTab->getID());
    showFrameCountField->setEventListener(this);
    showFrameCountField->setPos(1, rowcount + 11);
    showFrameCountField->setValue(frameCount);

    // default off cover->getScene()->addChild( ephemerisModel.get() );

    // use Config to fill default values
    time_base = coCoviseConfig::getInt("COVER.Plugin.Video.Framerate", 25);
    cfpsEdit->setValue(time_base);
    outWidth = coCoviseConfig::getInt("COVER.Plugin.Video.VideoSizeX", -1);
    outWidthField->setValue(outWidth);
    outHeight = coCoviseConfig::getInt("COVER.Plugin.Video.VideoSizeY", -1);
    outHeightField->setValue(outHeight);

    filename = coCoviseConfig::getEntry("COVER.Plugin.Video.Filename");
    fillFilenameField(filename, false, false);

    bitrate_ = coCoviseConfig::getInt("COVER.Plugin.Video.Bitrate", 10000);
    sysPlug->bitrateField->setValue(bitrate_);

    return true;
}

bool VideoPlugin::update()
{
    return captureActive || captureAnimActive;
}

void VideoPlugin::cfpsHide(bool hidden)
{
    cfpsEdit->setHidden(hidden);
    cfpsLabel->setHidden(hidden);
}

void VideoPlugin::tabletEvent(coTUIElement *tUIItem)
{
    int x, y, width, height;

    if (tUIItem == outWidthField || tUIItem == outHeightField)
        sizeError = opt_frame_size(outWidthField->getValue(), outHeightField->getValue());

    coVRConfig::instance()->windows[0].window->getWindowRectangle(x, y, width, height);
    if ((tUIItem == xPosField) || (tUIItem == widthField))
        checkPosSize(xPosField, widthField, width);
    else if ((tUIItem == yPosField) || (tUIItem == heightField))
        checkPosSize(yPosField, heightField, height);
    else if (tUIItem == sizeButton)
    {
        widthField->setValue(width);
        checkPosSize(xPosField, widthField, width);
        heightField->setValue(height);
        checkPosSize(yPosField, heightField, height);

        sizeButton->setState(false);
    }
    else if (tUIItem == selectFrameRate)
    {
        switch (selectFrameRate->getSelectedEntry())
        {
        case 0:
            time_base = 25;
            cfpsEdit->setValue(coVRConfig::instance()->frameRate());
            cfpsHide(true);
            break;
        case 1:
            time_base = 30;
            cfpsEdit->setValue(coVRConfig::instance()->frameRate());
            cfpsHide(true);
            break;
        case 2:
            time_base = cfpsEdit->getValue();
            if (time_base <= 0)
            {
                time_base = 25;
            }
            cfpsEdit->setValue(time_base);
            cfpsHide(false);
            break;
        default:
            assert("framerate list inconsistent" == 0);
        }
    }

    if (tUIItem == selectFormat)
        sysPlug->changeFormat(tUIItem, rowcount);

    if (tUIItem == cfpsEdit)
    {
        if (selectFrameRate->getSelectedEntry() == 2)
        {
            time_base = cfpsEdit->getValue();
            if (time_base <= 0)
            {
                time_base = 25;
            }
            cfpsEdit->setValue(time_base);
        }
        coVRConfig::instance()->setFrameRate(cfpsEdit->getValue());
    }

    if ((tUIItem == fileNameField) || (tUIItem == selectFormat))
        fillFilenameField(fileNameField->getText(), false);
    //  if (tUIItem == selectFormat) fillFilenameField(fileNameField->getText(),false);

    if (tUIItem == fileNameBrowser)
        fillFilenameField("", true);

    if (!captureActive && (tUIItem == captureButton || tUIItem == captureAnimButton)) // only capture the
    // first window and only
    // on the master
    {
        cover->sendMessage(this, coVRPluginSupport::TO_ALL, PluginMessageTypes::VideoStartCapture, fileNameField->getText().size() + 1, fileNameField->getText().c_str());
        if (captureHostButton[hostIndex]->getState())
        {
            frameCount = 0;
            showFrameCountField->setValue(frameCount);

            if (tUIItem == captureAnimButton)
            {
                captureButton->setState(true);
                captureAnimActive = true;
                wasAnimating = coVRAnimationManager::instance()->animationRunning();
                coVRAnimationManager::instance()->enableAnimation(false);
                captureAnimFrame = coVRAnimationManager::instance()->getStartFrame();
                waitForFrame = true;
                coVRAnimationManager::instance()->requestAnimationFrame(captureAnimFrame);
            }

            filename = fileNameField->getText();
            if (changeFilename_)
            {
                size_t found = filename.rfind("_");
                if (found == string::npos)
                    found = filename.rfind(".");
                if (found != string::npos)
                    filename = filename.substr(0, found) + "-" + hostName + stereoEye + filename.substr(found, filename.length());
            }
            outWidth = outWidthField->getValue();
            if (outWidth <= 0)
                outWidth = widthField->getValue();
            outHeight = outHeightField->getValue();
            if (outHeight <= 0)
                outHeight = heightField->getValue();
            sysPlug->checkFileFormat(filename);

            //////////////////////////////////////////////////////////////////////////////
            // Standart procedure
            //////////////////////////////////////////////////////////////////////////////

            if (fileError || sizeError)
            {
                if (captureAnimActive)
                    coVRAnimationManager::instance()->enableAnimation(wasAnimating);
                captureAnimActive = false;
                captureButton->setState(false);
                captureAnimButton->setState(false);
                // fprintf(stderr,"fileError=%d sizeError=%d\n", fileError, sizeError);
            }
            else if (!sizeError && !captureActive && captureHostButton[hostIndex]->getState())
            {

                coVRConfig::instance()->windows[0].window->grabFocus();
                coVRConfig::instance()->windows[0].window->raiseWindow();
                inWidth = widthField->getValue();
                inHeight = heightField->getValue();

                if (inWidth != outWidth || inHeight != outHeight)
                    resize = true;
                else
                    resize = false;

                if (sysPlug->videoCaptureInit(filename, selectFormat->getSelectedEntry(),
                                              GLformatButton->getSelectedEntry()))
                {

                    sysPlug->init_GLbuffers();

                    captureActive = true;
                    starttime = cover->frameTime();
                    recordingTime = 0.0;
                    recordingFrames = 0;
                }
                else
                    fprintf(stderr, "videoCaptureInit failed\n");
            }
        }
        else if (hostIndex == 0)
        {
            int i;
            for (i = 1; i <= coVRMSController::instance()->clusterSize() - 1; i++)
                if (captureHostButton[i]->getState())
                    break;
            if (i > coVRMSController::instance()->clusterSize() - 1)
                captureButton->setState(false);
        }
    }
}

void VideoPlugin::tabletReleaseEvent(coTUIElement *tUIItem)
{
    if (captureActive && captureHostButton[hostIndex]->getState() && (tUIItem == captureButton || tUIItem == captureAnimButton))
    {
        stopCapturing();
        // prepare a new name
        fillFilenameField(fileNameField->getText(), false);
    }
}

void VideoPlugin::stopCapturing()
{
    sysPlug->close_all(true, selectFormat->getSelectedEntry());

    showFrameCountField->setValue(frameCount);
    captureActive = false;
    if (captureAnimActive)
        coVRAnimationManager::instance()->enableAnimation(wasAnimating);
    captureAnimActive = false;
    waitForFrame = false;
    captureButton->setState(false);
    captureAnimButton->setState(false);
    frameCount = 0;
    recordFromGui_ = false;
    cover->sendMessage(this, coVRPluginSupport::TO_ALL, PluginMessageTypes::VideoEndCapture, fileNameField->getText().size() + 1, fileNameField->getText().c_str());
}

// this is called if the plugin is removed at runtime
VideoPlugin::~VideoPlugin()
{
    fprintf(stderr, "VideoPlugin::~VideoPlugin\n");
    if (captureActive)
    {
        sysPlug->close_all(false);
    }
#ifdef HAVE_WMFSDK
    delete sysPlugWinAvi;
#endif
#ifdef HAVE_FFMPEG
    delete sysPlugFfmpeg;
#endif

    delete captureButton;
    delete GLformatButton;
    delete capPos;
    delete capArea;
    delete videoSize;
    delete fileNameField;
    delete xPosField;
    delete yPosField;
    delete widthField;
    delete heightField;
    delete outWidthField;
    delete outHeightField;
    delete VideoTab;
    delete errorLabel;
    delete fileErrorLabel;
    delete showFrameLabel;
    delete showFrameCountField;
    delete sizeButton;
    delete frameRate;
    delete selectFrameRate;
    delete cfpsLabel;
    delete cfpsEdit;
    delete messageBox;
    delete fileNameBrowser;
    delete hostLabel;
    for (int i = 0; i <= coVRMSController::instance()->clusterSize() - 1; i++)
        delete captureHostButton[i];

    if (selectFormat)
        delete selectFormat;
}

void VideoPlugin::postFrame()
{
    if (captureAnimActive)
    {
        if (!waitForFrame)
        {
            ++captureAnimFrame;
            if (captureAnimFrame > coVRAnimationManager::instance()->getStopFrame())
            {
                stopCapturing();
            }
            else
            {
                waitForFrame = true;
                coVRAnimationManager::instance()->requestAnimationFrame(captureAnimFrame);
            }
        }
    }
    else
    {
        waitForFrame = false;
    }


    // if (record_)
    // fprintf(stderr,"frameCount=%d time_base=%d\n", frameCount, time_base);
}

void VideoPlugin::setTimestep(int t)
{
    if (captureAnimActive && t == captureAnimFrame)
    {
        waitForFrame = false;
    }
}

void VideoPlugin::preSwapBuffers(int windowNumber)
{
    if (waitForFrame)
        return;

    auto &coco = *coVRConfig::instance();

    // only capture the first window and only on the master
    if (captureActive && windowNumber == 0 && captureHostButton[hostIndex])
    {
        // fprintf(stderr,"glRead...\n");
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        if (coco.windows[windowNumber].doublebuffer)
            glReadBuffer(GL_BACK);
        // for depth to work, it might be necessary to read from GL_FRONT (chang this, if it does not work like
        // this)

        glReadPixels(xPosField->getValue(), yPosField->getValue(), inWidth, inHeight, GL_fmt, GL_UNSIGNED_BYTE,
                     pixels);

        //      recordingTime += cover->frameDuration(); 
        //      int frameDelta = int(recordingTime * double(time_base)) - recordingFrames;
        //      for (int i=0; i<frameDelta; ++i)
        //      {
        //         sysPlug->videoWrite(selectFormat->getSelectedEntry());
        //         ++recordingFrames;
        //      }

        sysPlug->videoWrite(selectFormat->getSelectedEntry());
#ifdef WRITE_DEPTH_VIDEO
        float *buf = new float[inWidth * inHeight];
        memset(buf, 0, sizeof(float) * inWidth * inHeight);
        glReadPixels(xPosField->getValue(), yPosField->getValue(), inWidth, inHeight, GL_DEPTH_COMPONENT,
                     GL_FLOAT, buf);
        for (int i = 0; i < inWidth * inHeight; i++)
        {
            pixels[i * 3] = buf[i] * 256;
            pixels[i * 3 + 1] = buf[i] * 256;
            pixels[i * 3 + 2] = buf[i] * 256;
        }

        sysPlug->videoWriteDepth(selectFormat->getSelectedEntry(), buf);
        delete[] buf;
#endif
    }
}

void VideoPlugin::message(int toWhom, int type, int len, const void *data)
{
    const char *buf = (const char *)data;
    // fprintf(stderr,"VideoPlugin::message type=%d, len=%d data=%s\n", type, len, data);
    if (strncmp(buf, "stopCapturing", len) == 0)
    {
        if (captureActive)
        {
            captureButton->setState(false);
            tabletReleaseEvent(captureButton);
        }
    }
}

COVERPLUGIN(VideoPlugin)
