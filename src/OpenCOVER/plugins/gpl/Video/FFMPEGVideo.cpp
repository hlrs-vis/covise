#define __STDC_CONSTANT_MACROS
#include "FFMPEGVideo.h"

#include <config/CoviseConfig.h>

#include <cover/coVRTui.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>
#include <config/coConfigConstants.h>
#include <util/threadname.h>
#include <util/string_util.h>
#ifdef WIN32
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif

#if XERCES_VERSION_MAJOR < 3
#include <xercesc/dom/DOMWriter.hpp>
#else
#include <xercesc/dom/DOMLSSerializer.hpp>
#endif
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLUni.hpp>

#include <stdio.h>

using namespace covise;

FFMPEGPlugin::~FFMPEGPlugin()
{
    delete paramNameField;
    delete paramErrorLabel;
    delete paramLabel;
    delete bitrateField;
    delete maxBitrateField;
    delete saveButton;
    delete bitrateLabel;
    delete maxBitrateLabel;

    delete selectCodec;
    delete selectParams;
}

void FFMPEGPlugin::unInitialize()
{
    m_encoder.reset(nullptr);

    ifstream ifs(myPlugin->fileNameField->getText().c_str(), ifstream::in);
    if (!ifs.fail())
    {
        ifs.close();
        unlink(myPlugin->fileNameField->getText().c_str());
    }
}

void FFMPEGPlugin::close_all(bool stream, int format)
{
    m_encoder.reset(nullptr);
}

void FFMPEGPlugin::init_GLbuffers()
{
    if (m_encoder)
        myPlugin->pixels = m_encoder->getPixelBuffer();
}

bool FFMPEGPlugin::videoCaptureInit(const string &filename, int format, int RGBFormat)
{

    myPlugin->GL_fmt = RGBFormat == 1 ? GL_BGRA : GL_RGBA;

    linesize = myPlugin->inWidth * 4;
    FFmpegEncoder::VideoFormat input, output;
    input.colorFormat = RGBFormat == 1 ? AV_PIX_FMT_RGB32 : AV_PIX_FMT_BGR32;
    input.resolution.w = myPlugin->inWidth;
    input.resolution.h = myPlugin->inHeight;
    std::cerr << "input res " << input.resolution.w << " x " << input.resolution.h << std::endl;
    if (auto codec = getSelectedCodec())
    {
        output.codecName = codec->name;
    }
    else
    {
        myPlugin->errorLabel->setLabel("Codec not found");
        output.codecName = "";
        return false;
    }
    output.resolution.w = myPlugin->outWidth;
    output.resolution.h = myPlugin->outHeight;
    output.colorFormat = AV_PIX_FMT_NONE;
    output.fps = myPlugin->time_base;
    output.outputFormat = getSelectedOutputFormat();
    output.bitrate = bitrateField->getValue() > maxBitrateField->getValue() ? maxBitrateField->getValue() *1000 : bitrateField->getValue() * 1000;
    output.max_bitrate = maxBitrateField->getValue() * 1000;
    if(output.bitrate == output.max_bitrate)
        bitrateField->setValue(output.max_bitrate);
    m_encoder.reset(new FFmpegEncoder(input, output, filename));
    if(!m_encoder->isValid())
    {
        myPlugin->errorLabel->setLabel("Memory error, codec not installed or invalid output parameters");
        myPlugin->captureButton->setState(false);
        m_encoder.reset(nullptr);
        return false;
    }
    return true;
}

const AVOutputFormat *FFMPEGPlugin::getSelectedOutputFormat()
{
    auto it = formatList.begin();

    if (myPlugin->selectFormat->getSelectedEntry() < 0)
        myPlugin->selectFormat->setSelectedEntry(0);
    for (; it != formatList.end(); it++)
    {
        if (strcmp((*it).first->long_name, myPlugin->selectFormat->getSelectedText().c_str()) == 0)
        {
            return it->first;
            break;
        }
    }
    if (it == formatList.end())
    {
        std::cerr << "Format not available on " << coVRMSController::instance()->getID() << std::endl;
        return nullptr;
    }
    return nullptr;
}

const AVCodec *FFMPEGPlugin::getSelectedCodec()
{
    if (selectCodec->getSelectedEntry() < 0)
        selectCodec->setSelectedEntry(0);
    std::list<CodecListEntry *>::iterator itlist;
    auto fmt = getSelectedOutputFormat();
    auto codecs = formatList.find(fmt);
    if (codecs == formatList.end())
    {
        std::cerr << "Failed to get selected codec for format " << fmt->long_name << std::endl;
        return nullptr;
    }
    for (const auto codec : codecs->second)
    {
        if (strcmp(codec.codec->long_name, selectCodec->getSelectedText().c_str()) == 0)
        {
            return codec.codec;
        }
    }
    return nullptr;
}

void FFMPEGPlugin::checkFileFormat(const string &filename)
{
    char buf[1000];

    ofstream dataFile(filename.c_str(), ios::ate);
    if (!dataFile)
    {
        sprintf(buf, "Could not open file. Please check file name.");
        myPlugin->fileErrorLabel->setLabel(buf);
        myPlugin->fileError = true;
        return;
    }
    else
    {
        dataFile.close();
        myPlugin->fileError = false;
    }

    myPlugin->sizeError = myPlugin->opt_frame_size(myPlugin->outWidth, myPlugin->outHeight);
    if (!myPlugin->sizeError)
    {

        const char *codecName, *formatName;
        auto it = formatList.begin();
        int formatEntry = myPlugin->selectFormat->getSelectedEntry();
        if (formatEntry < 0)
            formatName = (*formatList.begin()).first->long_name;
        else
        {
            formatName = myPlugin->selectFormat->getSelectedText().c_str();
            for (int i = 0; i < formatEntry; i++)
                it++;
        }

        if (selectCodec->getSelectedEntry() < 0)
            codecName = (*(*it).second.begin()).codec->long_name;
        else
            codecName = selectCodec->getSelectedText().c_str();

        if (strcmp(codecName, "DV (Digital Video)") == 0)
        {
            if (!((myPlugin->outWidth == 720) && (myPlugin->outHeight == 576)) && !((myPlugin->outWidth == 720) && (myPlugin->outHeight == 480)))
            {
                sprintf(buf, "DV only supports 720x576 or 720x480");
                myPlugin->errorLabel->setLabel(buf);
                myPlugin->sizeError = true;
                return;
            }
        }
        else if (strcmp(formatName, "GXF format") == 0)
        {
            if ((myPlugin->outHeight != 480) && (myPlugin->outHeight != 512) && (myPlugin->outHeight != 576) && (myPlugin->outHeight != 608))
            {
                sprintf(buf, "gxf muxer only accepts PAL or NTSC resolutions");
                myPlugin->errorLabel->setLabel(buf);
                myPlugin->sizeError = true;
                return;
            }
        }
    }
}

void FFMPEGPlugin::videoWrite(int format)
{
    m_encoder->writeVideo(myPlugin->frameCount++, myPlugin->pixels, false);

    if (cover->frameTime() - myPlugin->starttime >= 1)
    {
        myPlugin->showFrameCountField->setValue(myPlugin->frameCount);
        myPlugin->starttime = cover->frameTime();
    }
}

void FFMPEGPlugin::ListFormatsAndCodecs(const string &filename)
{
    unInitialize();
    formatList = covise::listFormatsAndCodecs();
}

void FFMPEGPlugin::FillComboBoxSetExtension(int selection, int row)
{
    selectCodec = new coTUIComboBox("Codec:", myPlugin->VideoTab->getID());
    selectCodec->setPos(0, row);
    selectCodec->setEventListener(this);
    std::vector<std::string> codecNames;
    if (coVRMSController::instance()->isMaster())
    {
        auto it = formatList.begin();
        std::advance(it, selection);
        if (it->first->extensions)
        {
            filterList = "";
            auto extensions = split(it->first->extensions, ',');
            for (const auto &ext : extensions)
            {
                if (filterList != "")
                    filterList += "; ";
                filterList += "*." + ext;
            }
        }
        for (const auto &c : it->second)
            codecNames.push_back(c.codec->long_name);
    }
    filterList = coVRMSController::instance()->syncString(filterList);
    myPlugin->fileNameBrowser->setFilterList(filterList);
    int count = codecNames.size();
    coVRMSController::instance()->syncData(&count, sizeof(count));
    if (coVRMSController::instance()->isSlave())
        codecNames.resize(count);

    for (auto &codecName : codecNames)
    {
        codecName = coVRMSController::instance()->syncString(codecName);
        selectCodec->addEntry(codecName);
    }
}

void FFMPEGPlugin::ParamMenu(int row)
{

    bitrateLabel = new coTUILabel("Average Bitrate kbit/s:", myPlugin->VideoTab->getID());
    bitrateLabel->setPos(0, row);
    bitrateLabel->setColor(Qt::black);
    bitrateField = new coTUIEditIntField("average bitrate kbit/s", myPlugin->VideoTab->getID());
    bitrateField->setValue(4000);
    bitrateField->setEventListener(this);
    bitrateField->setPos(1, row);

    maxBitrateLabel = new coTUILabel("Maximum Bitrate kbit/s:", myPlugin->VideoTab->getID());
    maxBitrateLabel->setPos(2, row);
    maxBitrateLabel->setColor(Qt::black);
    maxBitrateField = new coTUIEditIntField("maximum bitrate kbit/s", myPlugin->VideoTab->getID());
    maxBitrateField->setValue(6000);
    maxBitrateField->setEventListener(this);
    maxBitrateField->setPos(3, row);

    saveButton = new coTUIToggleButton("Enter Description and Save Parameter", myPlugin->VideoTab->getID());
    saveButton->setEventListener(this);
    saveButton->setState(false);
    saveButton->setPos(1, row + 2);

    paramNameField = new coTUIEditField("paramsDescription", myPlugin->VideoTab->getID());
    paramNameField->setText("");
    paramNameField->setEventListener(this);
    paramNameField->setPos(0, row + 2);

    paramErrorLabel = new coTUILabel("", myPlugin->VideoTab->getID());
    paramErrorLabel->setPos(0, row + 3);
    paramErrorLabel->setColor(Qt::black);
}

void FFMPEGPlugin::hideParamMenu(bool hide)
{
    bitrateLabel->setHidden(hide);
    bitrateField->setHidden(hide);
    maxBitrateLabel->setHidden(hide);
    maxBitrateField->setHidden(hide);
    saveButton->setHidden(hide);
    paramNameField->setHidden(hide);
    paramLabel->setHidden(hide);
    paramErrorLabel->setHidden(hide);
}

void FFMPEGPlugin::fillParamComboBox(int row)
{
    paramLabel = new coTUILabel("Parameter Dataset", myPlugin->VideoTab->getID());
    paramLabel->setPos(0, row);
    paramLabel->setColor(Qt::black);

    selectParams = new coTUIComboBox("Params:", myPlugin->VideoTab->getID());
    selectParams->setPos(1, row);
    selectParams->setEventListener(this);

    std::list<VideoParameter>::iterator it;
    for (it = VPList.begin(); it != VPList.end(); it++)
    {
        selectParams->addEntry((*it).name);
    }
}

void FFMPEGPlugin::Menu(int row)
{

    ParamMenu(row + 2);
    ListFormatsAndCodecs(myPlugin->fileNameField->getText());
    
    int count = formatList.size();
    coVRMSController::instance()->syncData(&count, sizeof(count));

    auto it = formatList.begin();
    for (int i = 0; i < count; i++)
    {
        std::string format;
        if (coVRMSController::instance()->isMaster())
            format = it->first->long_name;
        myPlugin->selectFormat->addEntry(coVRMSController::instance()->syncString(format));
        ++it;
    }

    if (count != 0)
        FillComboBoxSetExtension(0, row);

    myPlugin->helpText = "The video plugin offers file formats installed. Most suitable codecs for a selected format are "
                         "listed "
                         "first.\n The bitrates depend on the complexity and the size of the image and should be chosen as "
                         "small as possible to get an appropriate quality. The following list shows some suitable values for "
                         "the average and the maximum bitrate:\n\n MPEG-4 and H.264 formats:\n 1920x1080 24 MBits 30MBits\n "
                         "720x576 3 MBits 6 MBits\n 1440x1080 15 MBits 18 MBits\n 320x240 0,4 MBits 0,6 MBits\n\n WindowsMedia "
                         "formats:\n 1920x1080 8000 Kbits 10000 Kbits\n 1280x720  5000 Kbits 8000 Kbits\n 720x576 1500 KBits "
                         "6000 Kbits\n\n The parameter dataset file (videoparams.xml) contains the dimensions, the bitrates "
                         "and "
                         "the frame rate and is located in .covise.\n\n [ALT]g is the shortcut to start capturing in the "
                         "OpenCover window.";
    myPlugin->messageBox->setText(myPlugin->helpText);

    int size = 0;
    if (coVRMSController::instance()->isMaster())
    {
        size = readParams();
        sendParams();
    }
    else
    {
        size = getParams();
    }
    if (size > 0)
    {
        loadParams(VPList.size() - 1);
    }
    fillParamComboBox(row + 1);
    selectParams->setSelectedEntry(VPList.size());
}

void FFMPEGPlugin::changeFormat(coTUIElement *tUIItem, int row)
{
    if (selectCodec)
    {
        delete selectCodec;
        selectCodec = NULL;
    }
    FillComboBoxSetExtension(myPlugin->selectFormat->getSelectedEntry(), row);
}

void FFMPEGPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == selectCodec)
    {
        const char *codecName, *formatName;
        auto it = formatList.begin();
        int formatEntry = myPlugin->selectFormat->getSelectedEntry();
        if (formatEntry < 0)
            formatName = it->first->long_name;
        else
        {
            formatName = myPlugin->selectFormat->getSelectedText().c_str();
            for (int i = 0; i < formatEntry; i++)
                it++;
        }

        if (selectCodec->getSelectedEntry() < 0)
            codecName = (*(*it).second.begin()).codec->long_name;
        else
            codecName = selectCodec->getSelectedText().c_str();

        if (strcmp(codecName, "DV (Digital Video)") == 0)
            hideParamMenu(true);
        else
            hideParamMenu(false);
    }
    else if (tUIItem == saveButton)
    {
        addParams();
        if (coVRMSController::instance()->isMaster())
            saveParams();
        saveButton->setState(false);
    }
    else if (tUIItem == selectParams)
    {
        loadParams(selectParams->getSelectedEntry());
    }
}

void FFMPEGPlugin::loadParams(int select)
{
    std::list<VideoParameter>::iterator it = VPList.begin();
    for (int i = 0; i < select; i++)
        it++;
    VideoParameter VP = (*it);

    paramNameField->setText(VP.name);
    myPlugin->outWidth = VP.width;
    myPlugin->outWidthField->setValue(myPlugin->outWidth);
    myPlugin->outHeight = VP.height;
    myPlugin->outHeightField->setValue(myPlugin->outHeight);
    if (VP.fps == "Constant")
    {
        myPlugin->selectFrameRate->setSelectedEntry(2);
        myPlugin->time_base = VP.constFrames;
        myPlugin->cfpsEdit->setValue(myPlugin->time_base);
        myPlugin->cfpsHide(false);
    }
    else
    {
        myPlugin->selectFrameRate->setSelectedText(VP.fps);
        sscanf(VP.fps.c_str(), "%d", &myPlugin->time_base);
        myPlugin->cfpsHide(true);
    }
    bitrateField->setValue(VP.avgBitrate);
    if (VP.maxBitrate < VP.avgBitrate)
        VP.maxBitrate = VP.avgBitrate;
    maxBitrateField->setValue(VP.maxBitrate);
}

int FFMPEGPlugin::readParams()
{
    std::string pathname = coConfigDefaultPaths::getDefaultLocalConfigFilePath();
    pathname += "videoparams.xml";
#ifndef WIN32
    XMLCh *t1 = NULL;
    xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

    try
    {
        parser->parse(pathname.c_str());
    }
    catch (...)
    {
        cerr << "error parsing param table" << endl;
    }

    xercesc::DOMDocument *xmlDoc = parser->getDocument();
    xercesc::DOMElement *rootElement = NULL;
    if (xmlDoc)
    {
        rootElement = xmlDoc->getDocumentElement();
    }

    if (rootElement)
    {
        xercesc::DOMNodeList *nodeList = rootElement->getChildNodes();
        for (int i = 0; i < nodeList->getLength(); ++i)
        {
            xercesc::DOMElement *node = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
            if (!node)
                continue;
            VideoParameter VP;
            VP.name = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("name")));
            xercesc::XMLString::release(&t1);
            xercesc::DOMNodeList *childList = node->getChildNodes();
            for (int j = 0; j < childList->getLength(); j++)
            {
                xercesc::DOMElement *child = dynamic_cast<xercesc::DOMElement *>(childList->item(j));
                if (!child)
                    continue;

                char *w = xercesc::XMLString::transcode(child->getAttribute(t1 = xercesc::XMLString::transcode("outWidth")));
                xercesc::XMLString::release(&t1);
                char *h = xercesc::XMLString::transcode(child->getAttribute(t1 = xercesc::XMLString::transcode("outHeight")));
                xercesc::XMLString::release(&t1);
                VP.fps = xercesc::XMLString::transcode(child->getAttribute(t1 = xercesc::XMLString::transcode("frameRate")));
                xercesc::XMLString::release(&t1);
                char *constFrameRate = xercesc::XMLString::transcode(
                    child->getAttribute(t1 = xercesc::XMLString::transcode("constantFrameRate")));
                xercesc::XMLString::release(&t1);
                char *avgBitrate = xercesc::XMLString::transcode(
                    child->getAttribute(t1 = xercesc::XMLString::transcode("bitrateAverage")));
                xercesc::XMLString::release(&t1);
                char *maxBitrate = xercesc::XMLString::transcode(
                    child->getAttribute(t1 = xercesc::XMLString::transcode("bitrateMax")));
                xercesc::XMLString::release(&t1);

                sscanf(w, "%d", &VP.width);
                sscanf(h, "%d", &VP.height);
                if (VP.fps == "Constant")
                {
                    sscanf(constFrameRate, "%d", &VP.constFrames);
                }
                else
                    VP.constFrames = 0;

                sscanf(avgBitrate, "%d", &VP.avgBitrate);
                sscanf(maxBitrate, "%d", &VP.maxBitrate);

                VPList.push_back(VP);
                xercesc::XMLString::release(&w);
                xercesc::XMLString::release(&h);
                xercesc::XMLString::release(&constFrameRate);
                xercesc::XMLString::release(&avgBitrate);
                xercesc::XMLString::release(&maxBitrate);
            }
        }
    }
#endif

    return VPList.size();
}

void FFMPEGPlugin::saveParams()
{
    std::string pathname = coConfigDefaultPaths::getDefaultLocalConfigFilePath();
    pathname += "videoparams.xml";

#ifndef WIN32

    XMLCh *t1 = NULL;
    XMLCh *t2 = NULL;
    xercesc::DOMDocument *xmlDoc = impl->createDocument(0, t1 = xercesc::XMLString::transcode("VideoParams"), 0);
    xercesc::XMLString::release(&t1);
    xercesc::DOMElement *rootElement = NULL;
    if (xmlDoc)
    {
        rootElement = xmlDoc->getDocumentElement();
    }

    if (rootElement)
    {
        std::list<VideoParameter>::iterator it;
        for (it = VPList.begin(); it != VPList.end(); it++)
        {
            xercesc::DOMElement *VPElement = xmlDoc->createElement(t1 = xercesc::XMLString::transcode("VPEntry"));
            xercesc::XMLString::release(&t1);
            VPElement->setAttribute(t1 = xercesc::XMLString::transcode("name"),
                                    t2 = xercesc::XMLString::transcode((*it).name.c_str()));
            xercesc::XMLString::release(&t1);
            xercesc::XMLString::release(&t2);
            xercesc::DOMElement *VPChild = xmlDoc->createElement(t1 = xercesc::XMLString::transcode("VPValues"));
            xercesc::XMLString::release(&t1);
            char nr[100];
            sprintf(nr, "%d", (*it).width);
            VPChild->setAttribute(t1 = xercesc::XMLString::transcode("outWidth"), t2 = xercesc::XMLString::transcode(nr));
            xercesc::XMLString::release(&t1);
            xercesc::XMLString::release(&t2);
            sprintf(nr, "%d", (*it).height);
            VPChild->setAttribute(t1 = xercesc::XMLString::transcode("outHeight"), t2 = xercesc::XMLString::transcode(nr));
            xercesc::XMLString::release(&t1);
            xercesc::XMLString::release(&t2);
            VPChild->setAttribute(t1 = xercesc::XMLString::transcode("frameRate"),
                                  t2 = xercesc::XMLString::transcode((*it).fps.c_str()));
            xercesc::XMLString::release(&t1);
            xercesc::XMLString::release(&t2);
            sprintf(nr, "%d", (*it).constFrames);
            VPChild->setAttribute(t1 = xercesc::XMLString::transcode("constantFrameRate"),
                                  t2 = xercesc::XMLString::transcode(nr));
            xercesc::XMLString::release(&t1);
            xercesc::XMLString::release(&t2);
            sprintf(nr, "%d", (*it).maxBitrate);
            VPChild->setAttribute(t1 = xercesc::XMLString::transcode("bitrateMax"),
                                  t2 = xercesc::XMLString::transcode(nr));
            xercesc::XMLString::release(&t1);
            xercesc::XMLString::release(&t2);
            sprintf(nr, "%d", (*it).avgBitrate);
            VPChild->setAttribute(t1 = xercesc::XMLString::transcode("bitrateAverage"),
                                  t2 = xercesc::XMLString::transcode(nr));
            xercesc::XMLString::release(&t1);
            xercesc::XMLString::release(&t2);

            VPElement->appendChild(VPChild);
            rootElement->appendChild(VPElement);
        }

#if XERCES_VERSION_MAJOR < 3
        xercesc::DOMWriter *writer = impl->createDOMWriter();
        // set the format-pretty-print feature
        if (writer->canSetFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
            writer->setFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
        xercesc::XMLFormatTarget *xmlTarget = new xercesc::LocalFileFormatTarget(pathname.c_str());
        bool written = writer->writeNode(xmlTarget, *rootElement);
        if (!written)
            fprintf(stderr, "VideoParams::save info: Could not open file for writing !\n");

        delete writer;
        delete xmlTarget;
#else

        xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();

        xercesc::DOMLSOutput *theOutput = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();
        theOutput->setEncoding(t1 = xercesc::XMLString::transcode("utf8"));
        xercesc::XMLString::release(&t1);

        bool written = writer->writeToURI(rootElement, t1 = xercesc::XMLString::transcode(pathname.c_str()));
        xercesc::XMLString::release(&t1);
        if (!written)
            fprintf(stderr, "save info: Could not open file for writing %s!\n", pathname.c_str());
        delete writer;

#endif
    }
    delete xmlDoc;
#endif
}

void FFMPEGPlugin::addParams()
{
    VideoParameter VP;
    static bool overwrite = false;
    static std::string oldName = "";

    std::list<VideoParameter>::iterator it;
    int pos = 0;
    myPlugin->errorLabel->setLabel("");
    for (it = VPList.begin(); it != VPList.end(); it++)
    {
        if ((*it).name == paramNameField->getText())
        {
            if (overwrite && (oldName == paramNameField->getText()))
            {
                break;
            }
            else
            {
                char buf[1000];
                sprintf(buf, "Parameter name already in List. Press button again to overwrite.");
                myPlugin->errorLabel->setLabel(buf);
                myPlugin->sizeError = true;
                overwrite = true;
                oldName = paramNameField->getText();

                return;
            }
        }
        else
            pos++;
    }
    if (it == VPList.end())
    {
        if (overwrite)
            overwrite = false;
    }

    VP.name = paramNameField->getText();
    VP.width = myPlugin->outWidthField->getValue();
    VP.height = myPlugin->outHeightField->getValue();
    VP.fps = myPlugin->selectFrameRate->getSelectedText();
    VP.constFrames = myPlugin->cfpsEdit->getValue();
    VP.maxBitrate = maxBitrateField->getValue();
    VP.avgBitrate = bitrateField->getValue();
    if (VP.maxBitrate < VP.avgBitrate)
    {
        VP.maxBitrate = VP.avgBitrate;
        maxBitrateField->setValue(VP.maxBitrate);
    }

    if (!overwrite)
    {
        VPList.push_back(VP);
        selectParams->addEntry(VP.name);
    }
    else
    {
        (*it) = VP;
        overwrite = false;
    }

    selectParams->setSelectedEntry(pos);
}

void FFMPEGPlugin::sendParams()
{
    std::list<VideoParameter>::iterator it;
    int listSize = VPList.size();
    coVRMSController::instance()->sendSlaves((char *)&listSize, sizeof(int));
    for (it = VPList.begin(); it != VPList.end(); it++)
    {
        int length = strlen((*it).name.c_str());
        coVRMSController::instance()->sendSlaves((char *)&length, sizeof(int));
        coVRMSController::instance()->sendSlaves((*it).name.c_str(), length + 1);
        coVRMSController::instance()->sendSlaves((char *)&(*it).width, sizeof(int));
        coVRMSController::instance()->sendSlaves((char *)&(*it).height, sizeof(int));
        length = strlen((*it).fps.c_str());
        coVRMSController::instance()->sendSlaves((char *)&length, sizeof(length));
        coVRMSController::instance()->sendSlaves((*it).fps.c_str(), length + 1);
        coVRMSController::instance()->sendSlaves((char *)&(*it).constFrames, sizeof(int));
        coVRMSController::instance()->sendSlaves((char *)&(*it).maxBitrate, sizeof(int));
        coVRMSController::instance()->sendSlaves((char *)&(*it).avgBitrate, sizeof(int));
    }
}

int FFMPEGPlugin::getParams()
{
    int listSize = 0;

    coVRMSController::instance()->readMaster((char *)&listSize, sizeof(int));
    for (int i = 1; i <= listSize; i++)
    {
        VideoParameter VP;
        int length;
        coVRMSController::instance()->readMaster((char *)&length, sizeof(int));
        char *charString = new char[length + 1];
        coVRMSController::instance()->readMaster(charString, length + 1);
        VP.name = charString;
        delete[] charString;

        coVRMSController::instance()->readMaster((char *)&VP.width, sizeof(int));
        coVRMSController::instance()->readMaster((char *)&VP.height, sizeof(int));
        coVRMSController::instance()->readMaster((char *)&length, sizeof(int));
        charString = new char[length + 1];
        coVRMSController::instance()->readMaster(charString, length + 1);
        VP.fps = charString;
        delete[] charString;

        coVRMSController::instance()->readMaster((char *)&VP.constFrames, sizeof(int));
        coVRMSController::instance()->readMaster((char *)&VP.maxBitrate, sizeof(int));
        coVRMSController::instance()->readMaster((char *)&VP.avgBitrate, sizeof(int));

        VPList.push_back(VP);
    }

    return VPList.size();
}
