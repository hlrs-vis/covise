
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
#endif
#ifdef WIN32
#pragma warning (disable: 4005)
#endif
#include <windows.h>
#include "WINAVIVideo.h"
#include "cover/coVRPluginSupport.h"
#include "osg/Matrix"
#include "osg/MatrixTransform"

using namespace covise;

#pragma comment(lib, "vfw32")
#pragma comment(lib, "winmm")

#ifndef M_PI
#define M_PI 3.1415926535897931
#endif

WINAVIPlugin::WINAVIPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    HRESULT hr = S_OK;
    aviFile = NULL;
    aviStream = NULL;
    aviFile2 = NULL;
    aviStream2 = NULL;
    transFP = NULL;
    aviCompressed = NULL;
    aviCompressed2 = NULL;
    AVIButton = NULL;
    aviSample = 0;
    inPicture = outPicture = NULL;
    tmppixels = NULL;
    compressSet = new COMPVARS;
    compressSet->cbSize = sizeof(COMPVARS);
    compressSet->dwFlags = 0;
    compressSet->hic = NULL;
    compressSet->fccType = 0;

    std::string codec = coCoviseConfig::getEntry("COVER.Plugin.Video.Codec");
    if (codec.compare("XVID") != 0)
        compressSet->fccHandler = mmioFOURCC('x', 'v', 'i', 'd');
    else
        compressSet->fccHandler = NULL;
    compressSet->lpbiOut = compressSet->lpbiIn = NULL;
    compressSet->lpBitsOut = compressSet->lpBitsPrev = NULL;
    compressSet->lFrame = 0;
    compressSet->lKey = 0;
    compressSet->lDataRate = 0;
    compressSet->lQ = 0;
    compressSet->lKeyCount = 0;
    compressSet->lpState = NULL;
    compressSet->cbState = 0;
#ifdef HAVE_WMFSDK
    profile = NULL;
    writer = NULL;
    selectCodec = NULL;
#endif
    hr = CoInitialize(NULL);
    if (FAILED(hr))
    {
        fprintf(stderr, "COM Initializing failed\n");
        exit(1);
    }

    swsconvertctx = NULL;
    hres = S_OK;
}

WINAVIPlugin::~WINAVIPlugin()
{
#ifdef HAVE_WMFSDK
    profileMap.clear();
    codecList.clear();
    SAFE_RELEASE(profile);
    if (myPlugin->selectFormat->getSelectedEntry() == 1)
        RemoveWMVMenu();
    else
#endif
        if (AVIButton)
        delete AVIButton;
    CoUninitialize();
}

void WINAVIPlugin::Menu(int row)
{
    filterList = "*.avi";
    myPlugin->selectFormat->addEntry("AVI");
    myPlugin->fileNameBrowser->setFilterList(filterList);

    AVIButton = new coTUIToggleButton("AVI Properties", myPlugin->VideoTab->getID());
    AVIButton->setEventListener(this);
    AVIButton->setState(false);
    AVIButton->setPos(1, row++);

#ifdef HAVE_WMFSDK
    LPWSTR profileString = NULL;

    if (profileMap.empty())
        if (ListCodecs(true) == S_OK)
        {
            myPlugin->selectFormat->addEntry("WMV");
            if (coVRMSController::instance()->isMaster())
            {
                ListCustomProfiles();
                ListSystemProfiles();
                int mapSize = profileMap.size();
                coVRMSController::instance()->sendSlaves((char *)&mapSize, sizeof(int));
                std::multimap<WMT_VERSION, LPWSTR>::iterator it = profileMap.begin();
                for (; it != profileMap.end(); it++)
                {
                    coVRMSController::instance()->sendSlaves((char *)&(*it).first, sizeof(int));
                    int numBytes = WideCharToMultiByte(CP_ACP, 0, (*it).second, -1, NULL, 0, NULL, NULL);
                    char *sendString = new char[numBytes];
                    WideCharToMultiByte(CP_ACP, 0, (*it).second, -1, sendString, numBytes, NULL, NULL);
                    coVRMSController::instance()->sendSlaves((char *)&numBytes, sizeof(int));
                    coVRMSController::instance()->sendSlaves(sendString, numBytes);
                    delete[] sendString;
                }
            }
            else
            {
                int count;
                coVRMSController::instance()->readMaster((char *)&count, sizeof(int));
                for (int i = 0; i < count; i++)
                {
                    coVRMSController::instance()->readMaster((char *)&wmtVersion, sizeof(int));
                    int length;
                    coVRMSController::instance()->readMaster((char *)&length, sizeof(int));
                    char *buffer = new char[length];
                    coVRMSController::instance()->readMaster(buffer, length);
                    length = MultiByteToWideChar(CP_ACP, 0, buffer, -1, NULL, 0);
                    if (length != 0)
                    {
                        profileString = new WCHAR[length];
                        MultiByteToWideChar(CP_ACP, 0, buffer, -1, profileString, length);

                        SlaveAddProfile(profileString, false);
                    }
                    delete[] buffer;
                }
            }
            WMVMenu(row);

            if (profileMap.size() != 0)
                FillCompareComboBox(NULL, NULL);
            else if (coVRMSController::instance()->isMaster())
            {
                int count = 0;
                CreateProfile(NULL, myPlugin->outWidthField->getValue(), myPlugin->outHeightField->getValue(),
                              bitrate->getSelectedEntry(), bitrateField->getValue(),
                              &(*selectCompressionCodec->getSelectedText().c_str()), profileString, true);
                if (profileString != NULL)
                {
                    count = 1;
                    coVRMSController::instance()->sendSlaves((char *)&count, sizeof(int));
                    int numBytes = WideCharToMultiByte(CP_ACP, 0, profileString, -1, NULL, 0, NULL, NULL);
                    char *sendString = new char[numBytes];
                    WideCharToMultiByte(CP_ACP, 0, profileString, -1, sendString, numBytes, NULL, NULL);
                    coVRMSController::instance()->sendSlaves((char *)&numBytes, sizeof(int));
                    coVRMSController::instance()->sendSlaves(sendString, numBytes);
                    delete[] sendString;
                }
                else
                    coVRMSController::instance()->sendSlaves((char *)&count, sizeof(int));
            }
            else
            {
                int count;
                coVRMSController::instance()->readMaster((char *)&count, sizeof(int));
                if (count != 0)
                {
                    coVRMSController::instance()->readMaster((char *)&count, sizeof(int));
                    char *buffer = new char[count];
                    coVRMSController::instance()->readMaster(buffer, count);

                    int length = MultiByteToWideChar(CP_ACP, 0, buffer, -1, NULL, 0);
                    if (length != 0)
                    {
                        LPWSTR profileString = new WCHAR[length];
                        MultiByteToWideChar(CP_ACP, 0, buffer, -1, profileString, length);

                        SlaveAddProfile(profileString, true);
                    }
                    delete[] buffer;
                }
            }
        }
    HideWMVMenu(true);
#endif

    myPlugin->helpText = "Format and Codecs Help:\n\n COVISE lists video formats and codecs installed. The Windows Media Video "
                         "9 Screen codec is optimized for screen recording. Nevertheless for screen encodings with graphics, "
                         "many different colors and a lot of motion it is preferable to choose another codec. \n If "
                         "WindowsMedia is available predefined Profiles are displayed. Profiles, which are configured to "
                         "compress only audio, can not be used for video capturing. For quality based variable bitrate encoding "
                         "you have to enter the quality level instead of the bitrate.\n You can change the parameters and save "
                         "a customized profile in your local COVISE directory. Be careful to submit all necessary arguments.\n "
                         "[ALT]g is the shortcut to start capturing in the OpenCover window.";
}

void WINAVIPlugin::videoWrite(int format)
{

    if (format == 0)
    {
        unsigned int errorCode;

        if (myPlugin->resize)
        {
            inPicture->data[0] = myPlugin->pixels;
            sws_scale(swsconvertctx, inPicture->data, inPicture->linesize, 0, myPlugin->inHeight,
                      outPicture->data, outPicture->linesize);

            errorCode = AVIStreamWrite(aviCompressed, aviSample++, 1, outPicture->data[0],
                                       bmpInfo.bmiHeader.biSizeImage, 0, NULL, NULL);
        }
        else
            errorCode = AVIStreamWrite(aviCompressed, aviSample++, 1, myPlugin->pixels,
                                       bmpInfo.bmiHeader.biSizeImage, 0, NULL, NULL);
        myPlugin->frameCount++;

        if (cover->frameTime() - myPlugin->starttime >= 1)
        {
            myPlugin->showFrameCountField->setValue(myPlugin->frameCount);
            myPlugin->starttime = cover->frameTime();
        }

        if (errorCode != 0)
        {
            std::cout << "Unable to write stream to file" << std::endl;
            close_all(true, 0);
            myPlugin->captureActive = false;
            return;
        }
    }
#ifdef HAVE_WMFSDK
    else
    {
        HRESULT hr = WriteWMVSample();
        if (hr != S_OK)
        {
            myPlugin->captureButton->setState(false);
            close_all(true, 1);
            myPlugin->captureActive = false;
        }
    }
#endif
}
void WINAVIPlugin::videoWriteDepth(int format, float *buf)
{

    if (format == 0)
    {
        if (transFP)
        {
            osg::Matrix m = opencover::cover->getObjectsXform()->getMatrix();
            fprintf(transFP, "%d Model  [%f %f %f %f , %f %f %f %f , %f %f %f %f , %f %f %f %f ]\n",
                    myPlugin->frameCount, m(0, 0), m(0, 1), m(0, 2), m(0, 3), m(1, 0), m(1, 1), m(1, 2), m(1, 3),
                    m(2, 0), m(2, 1), m(2, 2), m(2, 3), m(3, 0), m(3, 1), m(3, 2), m(3, 3));
            m = opencover::cover->getViewerMat();
            fprintf(transFP, "%d Viewer [%f %f %f %f , %f %f %f %f , %f %f %f %f , %f %f %f %f ]\n",
                    myPlugin->frameCount, m(0, 0), m(0, 1), m(0, 2), m(0, 3), m(1, 0), m(1, 1), m(1, 2), m(1, 3),
                    m(2, 0), m(2, 1), m(2, 2), m(2, 3), m(3, 0), m(3, 1), m(3, 2), m(3, 3));
        }
        unsigned int errorCode;
        unsigned char *cbuf = new unsigned char[myPlugin->inWidth * myPlugin->inHeight * 3];

        for (int i = 0; i < myPlugin->inWidth * myPlugin->inHeight; i++)
        {
            int tmp = buf[i] * 16777216;
            cbuf[i * 3] = tmp & 0x00ff;
            cbuf[i * 3 + 1] = (tmp >> 8) & 0x00ff;
            cbuf[i * 3 + 2] = (tmp >> 16) & 0x00ff;
        }

        if (myPlugin->resize)
        {
            inPicture->data[0] = cbuf;
            sws_scale(swsconvertctx, inPicture->data, inPicture->linesize, 0, myPlugin->inHeight,
                      outPicture->data, outPicture->linesize);

            errorCode = AVIStreamWrite(aviCompressed2, aviSample, 1, outPicture->data[0],
                                       bmpInfo.bmiHeader.biSizeImage, 0, NULL, NULL);
        }
        else
            errorCode = AVIStreamWrite(aviCompressed2, aviSample, 1, cbuf, bmpInfo.bmiHeader.biSizeImage, 0, NULL, NULL);
        myPlugin->frameCount++;

        if (cover->frameTime() - myPlugin->starttime >= 1)
        {
            myPlugin->showFrameCountField->setValue(myPlugin->frameCount);
            myPlugin->starttime = cover->frameTime();
        }

        if (errorCode != 0)
        {
            std::cout << "Unable to write stream to file" << std::endl;
            close_all(true, 0);
            myPlugin->captureActive = false;
            return;
        }
        delete[] cbuf;
    }
#ifdef HAVE_WMFSDK
    else
    {
        HRESULT hr = WriteWMVSample();
        if (hr != S_OK)
        {
            myPlugin->captureButton->setState(false);
            close_all(true, 1);
            myPlugin->captureActive = false;
        }
    }
#endif
}

void WINAVIPlugin::close_all(bool stream, int format)
{
    if (stream)
    {
        if (format == 0)
        {
            if (aviCompressed)
            {
                AVIStreamRelease(aviCompressed);
                aviCompressed = NULL;
            }
            if (aviCompressed2)
            {
                AVIStreamRelease(aviCompressed2);
                aviCompressed2 = NULL;
            }

            if (aviStream)
            {
                AVIStreamRelease(aviStream);
                aviStream = NULL;
            }

            if (aviFile)
            {
                AVIFileRelease(aviFile);
                aviFile = NULL;
            }

            if (aviStream2)
            {
                AVIStreamRelease(aviStream2);
                aviStream2 = NULL;
            }
            if (transFP)
            {
                fclose(transFP);
                transFP = NULL;
            }

            if (aviFile2)
            {
                AVIFileRelease(aviFile2);
                aviFile2 = NULL;
            }
            ICCompressorFree(compressSet);
            AVIFileExit();
            aviSample = 0;
        }
#ifdef HAVE_WMFSDK
        else
            CloseVideoWMV();
#endif
    }

    free_all();
}

void WINAVIPlugin::free_all()
{

    if (myPlugin->resize)
    {
        if (swsconvertctx)
            sws_freeContext(swsconvertctx);

        if (inPicture)
        {
            delete (inPicture->data[0]);
            delete (inPicture);
        }
        if (outPicture)
        {
            delete (outPicture->data[0]);
            delete (outPicture);
        }
        if (tmppixels)
            tmppixels = NULL;
    }
    else
    {
        if (myPlugin->pixels)
            delete (myPlugin->pixels);
    }

    myPlugin->pixels = NULL;
}

void WINAVIPlugin::init_GLbuffers()
{

    if (myPlugin->GL_fmt == GL_BGR_EXT)
        myPlugin->pixels = new uint8_t[myPlugin->inWidth * myPlugin->inHeight * 24 / 8];
    else
        myPlugin->pixels = new uint8_t[myPlugin->inWidth * myPlugin->inHeight * 32 / 8];
}

void WINAVIPlugin::checkFileFormat(const string &name)
{

    ofstream dataFile(name.c_str(), ios::app);
    if (!dataFile)
    {
        myPlugin->fileErrorLabel->setLabel("Could not open file. Please check file name.");
        myPlugin->fileError = true;
    }
    else
    {
        myPlugin->fileErrorLabel->setLabel("");
        dataFile.close();
        myPlugin->fileError = false;
    }

    myPlugin->sizeError = myPlugin->opt_frame_size(myPlugin->outWidth, myPlugin->outHeight);
}

bool WINAVIPlugin::ChooseVideoCodec()
{
    bool chosenCodec = false;

    if (coVRMSController::instance()->isMaster())
    {
        HWND hwnd = GetForegroundWindow();

        if (hwnd != NULL)
        {
            fprintf(stderr, "WINAVIPlugin::ChooseVideoCodec hwnd not NULL\n");
            BITMAPINFOHEADER bih;
            bih.biSize = sizeof(BITMAPINFOHEADER);
            bih.biPlanes = 1;
            bih.biWidth = myPlugin->outWidthField->getValue();
            bih.biHeight = myPlugin->outHeightField->getValue();
            bih.biCompression = BI_RGB;
            bih.biBitCount = 24;

            bih.biXPelsPerMeter = 0;
            bih.biYPelsPerMeter = 0;
            bih.biClrUsed = 0;
            bih.biClrImportant = 0;

            bih.biSizeImage = bih.biWidth * bih.biHeight * bih.biBitCount / 8;
            chosenCodec = ICCompressorChoose(hwnd, ICMF_CHOOSE_DATARATE | ICMF_CHOOSE_KEYFRAME, &bih, NULL,
                                             compressSet, NULL);
            //	 hwnd = GetForegroundWindow();

            SetWindowPos(hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
        }
        else
            fprintf(stderr, "WINAVIPlugin::ChooseVideoCodec hwnd=NULL\n");

        coVRMSController::instance()->sendSlaves((char *)&chosenCodec, sizeof(int));
        if (chosenCodec)
        {
            coVRMSController::instance()->sendSlaves((char *)&compressSet->fccHandler, sizeof(int));
            coVRMSController::instance()->sendSlaves((char *)&compressSet->lQ, sizeof(int));
            coVRMSController::instance()->sendSlaves((char *)&compressSet->lKey, sizeof(int));
            coVRMSController::instance()->sendSlaves((char *)&compressSet->lDataRate, sizeof(int));
        }
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&chosenCodec, sizeof(int));
        if (chosenCodec)
        {
            compressSet->fccType = ICTYPE_VIDEO;
            coVRMSController::instance()->readMaster((char *)&compressSet->fccHandler, sizeof(int));
            compressSet->hic = ICOpen(compressSet->fccType, compressSet->fccHandler, ICMODE_FASTCOMPRESS);
            coVRMSController::instance()->readMaster((char *)&compressSet->lQ, sizeof(int));
            coVRMSController::instance()->readMaster((char *)&compressSet->lKey, sizeof(int));
            coVRMSController::instance()->readMaster((char *)&compressSet->lDataRate, sizeof(int));
        }
    }

    return (chosenCodec);
}

HRESULT WINAVIPlugin::aviInit(const string &filename, short frame_rate)
{

    AVIFileInit();

    if (AVIFileOpen(&aviFile, filename.c_str(), OF_CREATE | OF_WRITE, NULL) != 0)
    {
        std::cout << "Unable to open movie file" << std::endl;
        return E_INVALIDARG;
    }
    aviFile2 = NULL;
#ifdef WRITE_DEPTH_VIDEO
    std::string filename2 = "depth_" + filename;
    std::string filenameTrans = "trans_" + filename + ".txt";
    if (AVIFileOpen(&aviFile2, filename2.c_str(), OF_CREATE | OF_WRITE, NULL) != 0)
    {
        std::cout << "Unable to open movie file" << std::endl;
        return E_INVALIDARG;
    }

    transFP = fopen(filenameTrans.c_str(), "w");
#endif

    ZeroMemory(&aviStreamInfo, sizeof(AVISTREAMINFO));
    aviStreamInfo.fccType = streamtypeVIDEO;
    aviStreamInfo.dwQuality = compressSet->lQ;
    aviStreamInfo.fccHandler = compressSet->fccHandler;

    aviStreamInfo.dwScale = 1;
    aviStreamInfo.dwRate = frame_rate;

    aviStreamInfo.dwSuggestedBufferSize = myPlugin->outWidth * myPlugin->outHeight * 4;

    if (AVIFileCreateStream(aviFile, &aviStream, &aviStreamInfo) != 0)
    {
        std::cout << "Unable to create movie stream" << std::endl;
        return E_INVALIDARG;
    }
    aviStream2 = NULL;
    if (aviFile2)
    {
        if (AVIFileCreateStream(aviFile2, &aviStream2, &aviStreamInfo) != 0)
        {
            std::cout << "Unable to create movie stream2" << std::endl;
            return E_INVALIDARG;
        }
    }

    ZeroMemory(&aviCompressOpt, sizeof(AVICOMPRESSOPTIONS));
    aviCompressOpt.fccType = streamtypeVIDEO;
    aviCompressOpt.fccHandler = aviStreamInfo.fccHandler;
    aviCompressOpt.dwQuality = aviStreamInfo.dwQuality;
    aviCompressOpt.dwKeyFrameEvery = compressSet->lKey;
    aviCompressOpt.dwBytesPerSecond = compressSet->lDataRate;
    aviCompressOpt.dwFlags = AVICOMPRESSF_KEYFRAMES | AVICOMPRESSF_VALID | AVICOMPRESSF_DATARATE;

    HRESULT error = AVIMakeCompressedStream(&aviCompressed, aviStream, &aviCompressOpt, NULL);

    if (error != AVIERR_OK)
    {
        if (error == AVIERR_NOCOMPRESSOR)
            std::cout << "Suitable Compressor not found" << std::endl;
        else if (error == AVIERR_MEMORY)
            std::cout << "Not enough memory" << std::endl;
        else if (error == AVIERR_UNSUPPORTED)
            std::cout << "Compression is not supported for this type of data" << std::endl;
        else
            std::cout << "Can't find codec or codec is not properly installed" << std::endl;
        return E_INVALIDARG;
    }
    aviCompressed2 = NULL;
    if (aviStream2)
    {
        error = AVIMakeCompressedStream(&aviCompressed2, aviStream2, &aviCompressOpt, NULL);

        if (error != AVIERR_OK)
        {
            if (error == AVIERR_NOCOMPRESSOR)
                std::cout << "Suitable Compressor not found" << std::endl;
            else if (error == AVIERR_MEMORY)
                std::cout << "Not enough memory" << std::endl;
            else if (error == AVIERR_UNSUPPORTED)
                std::cout << "Compression is not supported for this type of data" << std::endl;
            else
                std::cout << "Can't find codec or codec is not properly installed" << std::endl;
            return E_INVALIDARG;
        }
    }

    ZeroMemory(&bmpInfo, sizeof(BITMAPINFO));
    bmpInfo.bmiHeader.biPlanes = 1;
    bmpInfo.bmiHeader.biWidth = myPlugin->outWidth;
    bmpInfo.bmiHeader.biHeight = myPlugin->outHeight;
    bmpInfo.bmiHeader.biCompression = BI_RGB;
    bmpInfo.bmiHeader.biBitCount = 24;

    bmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmpInfo.bmiHeader.biSizeImage = bmpInfo.bmiHeader.biWidth * bmpInfo.bmiHeader.biHeight * bmpInfo.bmiHeader.biBitCount / 8;

    if (AVIStreamSetFormat(aviCompressed, 0, (LPVOID)&bmpInfo, bmpInfo.bmiHeader.biSize) != 0)
    {
        std::cout << "Codec requirements and bitmap size or other features differ" << std::endl;
        return E_INVALIDARG;
    }
    if (aviCompressed2)
    {
        if (AVIStreamSetFormat(aviCompressed2, 0, (LPVOID)&bmpInfo, bmpInfo.bmiHeader.biSize) != 0)
        {
            std::cout << "Codec requirements and bitmap size or other features differ" << std::endl;
            return E_INVALIDARG;
        }
    }

    if (myPlugin->resize)
    {
#ifdef HAVE_FFMPEG
		capture_fmt = AV_PIX_FMT_RGB24;
#else
		capture_fmt = PIX_FMT_RGB24;
#endif
        swsconvertctx = sws_getContext(myPlugin->widthField->getValue(), myPlugin->heightField->getValue(),
                                       capture_fmt, bmpInfo.bmiHeader.biWidth, bmpInfo.bmiHeader.biHeight,
                                       capture_fmt, SWS_BICUBIC, NULL, NULL, NULL);

        inPicture = new Picture;
        memset(inPicture, 0, sizeof(Picture));
        inPicture->linesize[0] = linesize;

        outPicture = new Picture;
        memset(outPicture, 0, sizeof(Picture));
        outPicture->linesize[0] = bmpInfo.bmiHeader.biWidth * bmpInfo.bmiHeader.biBitCount / 8;
        tmppixels = new uint8_t
            [bmpInfo.bmiHeader.biWidth * bmpInfo.bmiHeader.biHeight * bmpInfo.bmiHeader.biBitCount / 8];
        outPicture->data[0] = tmppixels;
    }

    return S_OK;
}

bool WINAVIPlugin::videoCaptureInit(const string &name, int format, int RGBFormat)
{
    HRESULT hr = S_OK;

    if (RGBFormat == 1)
        myPlugin->GL_fmt = GL_RGB;
    linesize = myPlugin->inWidth * 3;
    sampleSize = myPlugin->outWidth * myPlugin->outHeight * 24 / 8;

    if (format == 0)
    {
        hr = aviInit(name, myPlugin->time_base);
        if (hr != S_OK)
            close_all(true, 0);
    }
#ifdef HAVE_WMFSDK
    else
    {
        string temp;
        temp = selectCompressionCodec->getSelectedText();
        int bits = bitrateField->getValue();
        if (checkBitrateOrQuality(bitrate->getSelectedEntry(), bits))
        {
            SetStreamFeatures(myPlugin->outWidth, myPlugin->outHeight, bits, temp.c_str());
            if (hr == S_OK)
                hr = InitWMVWriter(name);
            if (hr != S_OK)
                CloseVideoWMV();
        }
        else
        {
            myPlugin->captureButton->setState(false);
            return false;
        }
    }
#endif
    if (hr != S_OK)
    {
        myPlugin->errorLabel->setLabel("Codec or profile is not suitable for screen recording.");
        myPlugin->captureButton->setState(false);
        return false;
    }
    return true;
}
