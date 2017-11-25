#ifdef HAVE_WMFSDK
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
#endif
#ifdef WIN32
#pragma warning (disable: 4005)
#endif
#include <windows.h>
#include <nserror.h>
#include "WINAVIVideo.h"

#include <cover/coVRConfig.h>

#pragma comment(lib, "vfw32")
#pragma comment(lib, "winmm")
using namespace covise;

#ifndef M_PI
#define M_PI 3.1415926535897931
#endif

void WINAVIPlugin::WMVMenu(int row)
{

    selectCodec = new coTUIComboBox("Profile:", myPlugin->VideoTab->getID());
    selectCodec->setPos(0, row);
    selectCodec->setEventListener(this);

    selectCompressionCodec = new coTUIComboBox("Compression Codec:", myPlugin->VideoTab->getID());
    selectCompressionCodec->setPos(0, ++row);
    selectCompressionCodec->setEventListener(this);
    ListCodecs(false);
    selectCompressionCodec->setSelectedText("Windows Media Video 9");

    bitrate = new coTUIComboBox("Bitrate/Quality based", myPlugin->VideoTab->getID());
    bitrate->setPos(2, row);
    bitrate->addEntry("Bitrate kbit/s");
    bitrate->addEntry("Quality (1-100)");
    bitrate->setEventListener(this);

    bitrateField = new coTUIEditIntField("bitrate", myPlugin->VideoTab->getID());
    bitrateField->setValue(4000);
    bitrateField->setEventListener(this);
    bitrateField->setPos(1, row);

    saveButton = new coTUIToggleButton("Enter Description and Save Profile", myPlugin->VideoTab->getID());
    saveButton->setEventListener(this);
    saveButton->setState(false);
    saveButton->setPos(1, row + 1);

    profileNameField = new coTUIEditField("profileDescription", myPlugin->VideoTab->getID());
    profileNameField->setText("");
    profileNameField->setEventListener(this);
    profileNameField->setPos(0, row + 1);

    profileErrorLabel = new coTUILabel("", myPlugin->VideoTab->getID());
    profileErrorLabel->setPos(0, row + 2);
}

void WINAVIPlugin::RemoveWMVMenu()
{
    if (selectCompressionCodec)
    {
        delete selectCodec;
        delete selectCompressionCodec;
        delete bitrate;
        delete bitrateField;
        delete saveButton;
        delete profileNameField;
        delete profileErrorLabel;
    }
}

void WINAVIPlugin::HideWMVMenu(bool hide)
{
    selectCodec->setHidden(hide);
    selectCompressionCodec->setHidden(hide);
    bitrate->setHidden(hide);
    bitrateField->setHidden(hide);
    saveButton->setHidden(hide);
    profileNameField->setHidden(hide);
    profileErrorLabel->setHidden(hide);
}

HRESULT WINAVIPlugin::WriteWMVSample()
{
    HRESULT hr = NULL;
    INSSBuffer *sample;
    BYTE *buffer;
    DWORD bufferLength;

    hr = writer->AllocateSample(sampleSize, &sample);

    do
    {
        assert(SUCCEEDED(hr));

        hr = sample->GetBufferAndLength(&buffer, &bufferLength);
        assert(SUCCEEDED(hr));

        if (myPlugin->resize)
        {
            inPicture->data[0] = myPlugin->pixels;
            sws_scale(swsconvertctx, inPicture->data, inPicture->linesize, 0, myPlugin->inHeight,
                      outPicture->data, outPicture->linesize);
            memcpy(buffer, outPicture->data[0], sampleSize);
        }
        else
            memcpy(buffer, myPlugin->pixels, sampleSize);

        hr = sample->SetLength(sampleSize);
        hr = writer->WriteSample(videoInput, (QWORD)videoTime, 0, sample);
        if (FAILED(hr))
        {
            profileErrorLabel->setLabel("IWMWriter wasn't successful, change codec or size");
            hr = NS_E_INVALID_STREAM;
            SAFE_RELEASE(sample);
            return (hr);
        }
        else
            myPlugin->frameCount++;

        if (cover->frameTime() - myPlugin->starttime >= 1)
        {
            myPlugin->showFrameCountField->setValue(myPlugin->frameCount);
            myPlugin->starttime = cover->frameTime();
        }

    } while (false);

    videoTime += 10000000 / myPlugin->time_base;
    SAFE_RELEASE(sample);

    return (hr);
}

HRESULT WINAVIPlugin::ProfileNameTest()
{
    HRESULT hr = S_OK;
    char name[200];
    bool compare = false;
    WMT_VERSION wmtv = wmtVersion;

    strcpy(name, profileNameField->getText().c_str());

    if (strcmp(name, "") != 0)
    {
        hr = FillCompareComboBox(name, &compare);
        if (compare)
        {
            profileErrorLabel->setLabel("Profile already present-please enter another description");
            hr = NS_E_INVALID_DATA;
        }
        else
            profileErrorLabel->setLabel("");
    }
    else
    {
        profileErrorLabel->setLabel("Please enter a description for the profile");
        hr = NS_E_INVALID_DATA;
    }
    wmtVersion = wmtv;

    return (hr);
}

HRESULT WINAVIPlugin::ClearComboBox()
{
    HRESULT hr = S_OK;
    IWMProfileManager *profileMgr = NULL;
    IWMProfileManager2 *profileMgr2 = NULL;

    //
    // Create profile manager
    //
    hr = WMCreateProfileManager(&profileMgr);

    if (hr == S_OK)
        do
        {

            hr = profileMgr->QueryInterface(IID_IWMProfileManager2, (void **)&profileMgr2);
            if (FAILED(hr))
                break;

            //
            // Set system profile version
            //
            std::multimap<WMT_VERSION, LPWSTR>::iterator it = profileMap.begin();
            for (it = profileMap.begin(); it != profileMap.end(); it++)
            {
                hr = profileMgr2->SetSystemProfileVersion((*it).first);
                if (FAILED(hr))
                    break;

                hr = profileMgr->LoadProfileByData((*it).second, &profile);
                if (FAILED(hr))
                    break;

                hr = ProfileBoxName(&coTUIComboBox::delEntry, NULL, NULL);
                if (FAILED(hr))
                    break;
            }

            if (profile)
                profile->Release();
            if (profileMgr2)
                profileMgr2->Release();
        } while (FALSE);

    if (profileMgr)
        profileMgr->Release();

    return (hr);
}

HRESULT WINAVIPlugin::ProfileBoxName(void (coTUIComboBox::*AddDel)(const string &), LPWSTR profileString,
                                     bool *compare)
{
    HRESULT hr = S_OK;
    WCHAR *profileName = NULL;

    do
    {
        DWORD nameLength = 0;
        hr = profile->GetName(NULL, &nameLength);
        if (FAILED(hr))
            break;

        profileName = new WCHAR[nameLength];
        if (NULL == profileName)
        {
            hr = E_OUTOFMEMORY;
            break;
        }

        hr = profile->GetName(profileName, &nameLength);
        if (FAILED(hr))
            break;

        //
        // Display the system profile index and name and description
        //
        int numBytes = WideCharToMultiByte(CP_ACP, 0, profileName, -1, NULL, 0, NULL, NULL);
        char *prof = new char[numBytes];
        WideCharToMultiByte(CP_ACP, 0, profileName, -1, prof, numBytes, NULL, NULL);

        if (AddDel)
            (selectCodec->*AddDel)((const char *)prof);
        else if (compare != NULL)
        {
            if (wcscmp(profileString, profileName) == 0)
                *compare = true;
        }
        else
            profileMap.insert(std::pair<WMT_VERSION, LPWSTR>(wmtVersion, profileString));

    } while (FALSE);

    if (profileName)
        delete[] profileName;
    return (hr);
}

HRESULT WINAVIPlugin::FillCompareComboBox(char *profName, bool *compare)
{
    HRESULT hr = S_OK;
    IWMProfileManager *profileMgr = NULL;
    IWMProfileManager2 *profileMgr2 = NULL;

    //
    // Create profile manager
    //
    hr = WMCreateProfileManager(&profileMgr);

    if (hr == S_OK)
        do
        {

            hr = profileMgr->QueryInterface(IID_IWMProfileManager2, (void **)&profileMgr2);
            if (FAILED(hr))
                break;

            // Set profiles
            std::multimap<WMT_VERSION, LPWSTR>::iterator it;
            for (it = profileMap.begin(); it != profileMap.end(); it++)
            {
                hr = profileMgr2->SetSystemProfileVersion((*it).first);
                if (FAILED(hr))
                    continue;

                hr = profileMgr->LoadProfileByData((*it).second, &profile);
                if (FAILED(hr))
                    continue;

                if (compare != NULL)
                {
                    int numBytes = MultiByteToWideChar(CP_ACP, 0, profName, -1, NULL, 0);
                    LPWSTR buffer = new WCHAR[numBytes];
                    MultiByteToWideChar(CP_ACP, 0, profName, -1, buffer, numBytes);
                    hr = ProfileBoxName(NULL, buffer, compare);
                    if (*compare)
                        goto compare_end;
                }
                else
                    hr = ProfileBoxName(&coTUIComboBox::addEntry, NULL, compare);
                if (FAILED(hr))
                    continue;
            }

        compare_end:
            if (profileMgr2)
                profileMgr2->Release();
        } while (FALSE);

    if (profileMgr)
        profileMgr->Release();

    return (hr);
}

HRESULT WINAVIPlugin::checkCodec(codecStruct *codec)
{
    HRESULT hr = S_OK;
    IWMWriterAdvanced2 *advancedWriter = NULL;
    DWORD inputNumber;
    DWORD inputFormatCount = 0;
    LPWSTR profileString = NULL;

    hr = WMCreateWriter(NULL, &writer);
    if (FAILED(hr))
        return (hr);

    hr = writer->QueryInterface(IID_IWMWriterAdvanced2, (void **)&advancedWriter);
    if (FAILED(hr))
        return (hr);
    //	hr = advancedWriter->SetLiveSource(true);
    //	if (FAILED(hr)) return(hr);

    // Create a standard profile
    hr = CreateProfile(NULL, 640, 480, 0, 1024, &(*codec->version), profileString, false, &codec->guidType);
    if (FAILED(hr))
        return (hr);

    //  Save profile to writer
    hr = writer->SetProfile(profile);
    if (FAILED(hr))
        return (hr);

    hr = writer->GetInputCount(&inputNumber);
    if (FAILED(hr))
        return (hr);

    // look for the properties of all inputs
    for (DWORD i = 0; i < inputNumber; i++)
        do
        {
            // Find the number of formats supported by this input.
            hr = writer->GetInputFormatCount(i, &inputFormatCount);
            if (FAILED(hr))
                break;
        } while (false);

    SAFE_RELEASE(advancedWriter);
    SAFE_RELEASE(writer);
    return (hr);
}

HRESULT WINAVIPlugin::ListCodecs(bool test)
{
    HRESULT hr = S_OK;

    IWMProfileManager *profileMgr = NULL;
    IWMProfileManager2 *profileMgr2 = NULL;
    IWMCodecInfo2 *codecInfo = NULL;
    WCHAR *codecName = NULL;
    DWORD codecs = 0;
    IWMStreamConfig *streamConfig = NULL;
    IWMMediaProps *mediaProps = NULL;
    WM_MEDIA_TYPE *mediaType = NULL;
    codecStruct codec;

    //
    // Create profile manager
    //
    hr = WMCreateProfileManager(&profileMgr);

    if (hr == S_OK)
        do
        {

            hr = profileMgr->QueryInterface(IID_IWMProfileManager2, (void **)&profileMgr2);
            if (FAILED(hr))
                break;

            hr = profileMgr->QueryInterface(IID_IWMCodecInfo2, (void **)&codecInfo);
            if (FAILED(hr))
                break;

            // Retrieve the number of supported video codecs on the system.
            hr = codecInfo->GetCodecInfoCount(WMMEDIATYPE_Video, &codecs);
            if (FAILED(hr))
                break;
            else if (test)
            {
                if (codecs != 0)
                    return (S_OK);
                else
                    return (E_FAIL);
            }

            // Loop through all the video codecs.
            for (DWORD codecIndex = 0; codecIndex < codecs; codecIndex++)
            {
                // Get the codec name:
                // First, get the size of the name.
                DWORD codecNameSize = 0;
                hr = codecInfo->GetCodecName(WMMEDIATYPE_Video, codecIndex, NULL, &codecNameSize);
                if (FAILED(hr))
                    break;

                // Allocate a string of the appropriate size.
                codecName = new WCHAR[codecNameSize];
                if (codecName == NULL)
                {
                    hr = E_OUTOFMEMORY;
                    break;
                }

                // Retrieve the codec name.
                hr = codecInfo->GetCodecName(WMMEDIATYPE_Video, codecIndex, codecName, &codecNameSize);
                if (FAILED(hr))
                    break;

                int numBytes = WideCharToMultiByte(CP_ACP, 0, codecName, -1, NULL, 0, NULL, NULL);
                char *version = new char[numBytes];
                WideCharToMultiByte(CP_ACP, 0, codecName, -1, version, numBytes, NULL, NULL);

                hr = codecInfo->GetCodecFormat(WMMEDIATYPE_Video, codecIndex, 0, &streamConfig);
                if (FAILED(hr))
                    break;

                hr = streamConfig->QueryInterface(IID_IWMVideoMediaProps, (void **)&mediaProps);
                if (FAILED(hr))
                    break;

                DWORD mediaTypeSize;
                hr = mediaProps->GetMediaType(NULL, &mediaTypeSize);
                if (FAILED(hr))
                    break;
                mediaType = (WM_MEDIA_TYPE *)new BYTE[mediaTypeSize];
                if (!mediaType)
                {
                    hr = E_OUTOFMEMORY;
                    break;
                }
                hr = mediaProps->GetMediaType(mediaType, &mediaTypeSize);

                codec.version = version;
                codec.guidType = mediaType->subtype;

                hr = checkCodec(&codec);
                if (!FAILED(hr))
                {
                    selectCompressionCodec->addEntry(version);
                    codecList.push_back(codec);
                }
            }
            if (profileMgr2)
                profileMgr2->Release();
            SAFE_RELEASE(streamConfig);
            SAFE_RELEASE(mediaProps);
            delete[] mediaType;
        } while (FALSE);

    if (profileMgr)
        profileMgr->Release();

    return (hr);
}

void WINAVIPlugin::fillCodecComboBox()
{
    if (coVRMSController::instance()->isMaster())
    {
        int size = codecList.size();
        coVRMSController::instance()->sendSlaves((char *)&size, sizeof(int));
        list<codecStruct>::iterator it = codecList.begin();
        for (; it != codecList.end(); it++)
        {
            selectCompressionCodec->addEntry((*it).version);
            int count = sizeof((*it).version);
            coVRMSController::instance()->sendSlaves((char *)&count, sizeof(int));
            coVRMSController::instance()->sendSlaves((*it).version, count);
        }
    }
    else
    {
        int count;
        coVRMSController::instance()->readMaster((char *)&count, sizeof(int));
        for (int i = 1; i <= count; i++)
        {
            int size;
            coVRMSController::instance()->readMaster((char *)&size, sizeof(int));
            char *buffer = new char[size + 1];
            coVRMSController::instance()->readMaster(buffer, size + 1);
            std::string codecVersion = buffer;
            selectCompressionCodec->addEntry(codecVersion);
            delete[] buffer;
        }
    }
}

HRESULT WINAVIPlugin::checkProfile()
{
    HRESULT hr = S_OK;
    IWMStreamConfig *streamConfig = NULL;
    DWORD numberStreams;
    WORD streamNumber = 1;

    hr = profile->GetStreamCount(&numberStreams);
    if (hr == S_OK)
        for (WORD i = 1; i <= numberStreams; i++)
            do
            {
                do
                {
                    hr = profile->GetStreamByNumber(streamNumber, &streamConfig);
                    streamNumber++;
                } while (FAILED(hr));

                WORD streamNameLength;
                hr = streamConfig->GetStreamName(NULL, &streamNameLength);
                WCHAR *streamName = new WCHAR[streamNameLength];
                if (streamName == NULL)
                {
                    hr = E_OUTOFMEMORY;
                    break;
                }

                hr = streamConfig->GetStreamName(streamName, &streamNameLength);
                if (FAILED(hr))
                    break;

                if (wcsncmp(streamName, L"Video", 5) == 0)
                {
                    SAFE_RELEASE(streamConfig);
                    return (S_OK);
                }
                SAFE_RELEASE(streamConfig);

            } while (false);

    if (FAILED(hr))
        return (hr);
    else
        return (E_FAIL);
}

HRESULT WINAVIPlugin::ListSystemProfiles()
{
    HRESULT hr = S_OK;

    DWORD numProfiles = 0;
    IWMProfileManager *profileMgr = NULL;
    IWMProfileManager2 *profileMgr2 = NULL;
    WCHAR *profileName = NULL, *descrName = NULL;
    //	WMT_VERSION			wmtVersionField[] = {WMT_VER_4_0, WMT_VER_7_0, WMT_VER_8_0, WMT_VER_9_0};
    WMT_VERSION wmtVersionField[] = { WMT_VER_8_0, WMT_VER_9_0 };

    //
    // Create profile manager
    //
    hr = WMCreateProfileManager(&profileMgr);

    if (hr == S_OK)
        do
        {

            hr = profileMgr->QueryInterface(IID_IWMProfileManager2, (void **)&profileMgr2);
            if (FAILED(hr))
                break;

            for (DWORD j = 0; j < sizeof(wmtVersionField); j++)
            {
                wmtVersion = wmtVersionField[j];
                //
                // Set system profile version
                //
                hr = profileMgr2->SetSystemProfileVersion(wmtVersionField[j]);
                if (FAILED(hr))
                    break;

                hr = profileMgr->GetSystemProfileCount(&numProfiles);
                if (FAILED(hr))
                    break;

                //
                // Iterate all system profiles
                //

                //		fprintf(stderr,"Profiles for WMT Version: %s\n", version);
                for (DWORD i = 0; i < numProfiles; i++)
                {
                    hr = profileMgr->LoadSystemProfile(i, &profile);
                    if (FAILED(hr))
                        break;
                    hr = checkProfile();
                    if (FAILED(hr))
                        break;

                    // Save profile to buffer
                    LPWSTR buffer = NULL;
                    DWORD bufferSize = 0;
                    hr = profileMgr->SaveProfile(profile, NULL, &bufferSize);

                    buffer = new WCHAR[bufferSize];
                    if (buffer == NULL)
                    {
                        hr = E_OUTOFMEMORY;
                        break;
                    }
                    hr = profileMgr->SaveProfile(profile, buffer, &bufferSize);
                    if (FAILED(hr))
                        break;

                    hr = ProfileBoxName(NULL, buffer, NULL);
                    if (FAILED(hr))
                        break;
                }
            }

            delete[] profileName;
            delete[] descrName;

            if (profileMgr2)
                profileMgr2->Release();
        } while (FALSE);
    if (profileMgr)
        profileMgr->Release();

    return (hr);
}

HRESULT WINAVIPlugin::FindCustomProfiles(QString *path)
{
    HRESULT hr = S_OK;
    std::string pathname;
    char *filename = NULL;
    WIN32_FIND_DATA fileData;
    HANDLE hFile;
    LPWSTR profileString = NULL;

    pathname = path->toStdString();

    DWORD nameLength = path->size() + strlen("*.prx") + 1;
    filename = new char[nameLength];
    if (filename == NULL)
    {
        hr = E_OUTOFMEMORY;
        return (hr);
    }

    strcpy(filename, pathname.c_str());
    strcat(filename, "*.prx ");

    hFile = FindFirstFile(filename, &fileData);
    if (hFile != INVALID_HANDLE_VALUE)
        do
        {
            nameLength = path->size() + strlen(fileData.cFileName) + 1;
            char *name = new char[nameLength];
            if (name == NULL)
            {
                hr = E_OUTOFMEMORY;
                return (hr);
            }
            strcpy_s(name, nameLength, pathname.c_str());
            strcat_s(name, nameLength, fileData.cFileName);
            hr = LoadCustomProfile(name, profileString);
            if (FAILED(hr))
                return (hr);
            hr = checkProfile();
            if (FAILED(hr))
                return (hr);

            hr = ProfileBoxName(NULL, profileString, NULL);
            if (FAILED(hr))
                return (hr);

        } while (FindNextFile(hFile, &fileData));

    FindClose(hFile);

    return (hr);
}

HRESULT WINAVIPlugin::ListCustomProfiles()
{
    HRESULT hr = S_OK;

    QString path = coConfigDefaultPaths::getDefaultGlobalConfigFilePath();
    hr = FindCustomProfiles(&path);

    path = coConfigDefaultPaths::getDefaultLocalConfigFilePath();
    hr = FindCustomProfiles(&path);

    return (hr);
}

HRESULT WINAVIPlugin::SlaveAddProfile(LPWSTR profileString, bool fillCombo)
{
    HRESULT hr = S_OK;
    IWMProfileManager *profileMgr = NULL;

    //
    // Create profile manager
    //
    hr = WMCreateProfileManager(&profileMgr);

    do
    {
        hr = profileMgr->LoadProfileByData(profileString, &profile);
        if (FAILED(hr))
            break;

        if (!fillCombo)
            ProfileBoxName(NULL, profileString, NULL);
        else
        {
            if (profileMap.size() != 0)
            {
                IWMProfile *profileSelection = profile;
                ClearComboBox();
                profile = profileSelection;
                ProfileBoxName(NULL, profileString, NULL);
                FillCompareComboBox(NULL, NULL);
                profile = profileSelection;
            }
            else
            {
                ProfileBoxName(NULL, profileString, NULL);
                FillCompareComboBox(NULL, NULL);
            }
        }
    } while (false);

    SAFE_RELEASE(profileMgr);

    return (hr);
}

HRESULT WINAVIPlugin::SaveProfile(LPWSTR &profileString)
{
    HRESULT hr = S_OK;
    HANDLE file;
    IWMProfileManager *profileMgr = NULL;
    TCHAR *filename = NULL;
    std::string pathname;

    //
    // Create profile manager
    //
    hr = WMCreateProfileManager(&profileMgr);

    do
    {
        // Save profile to buffer
        DWORD bufferSize = 0;
        hr = profileMgr->SaveProfile(profile, NULL, &bufferSize);

        LPWSTR buffer = new WCHAR[bufferSize];
        if (buffer == NULL)
        {
            hr = E_OUTOFMEMORY;
            break;
        }
        hr = profileMgr->SaveProfile(profile, buffer, &bufferSize);
        if (FAILED(hr))
            break;
        profileString = buffer;

        QString path = coConfigDefaultPaths::getDefaultLocalConfigFilePath();
        pathname = path.toStdString();

        DWORD nameLength = path.size() + 40;
        filename = new TCHAR[nameLength];
        if (filename == NULL)
        {
            hr = E_OUTOFMEMORY;
            break;
        }

        strcpy_s(filename, nameLength, pathname.c_str());
        time_t rawtime;
        time(&rawtime);
        tm *utcTime = localtime(&rawtime);
        char prof[40];
        sprintf(prof, "CustomProfile_%02d%02d%4d_%02d%02d%02d.prx", utcTime->tm_mday, utcTime->tm_mon + 1,
                utcTime->tm_year + 1900, utcTime->tm_hour, utcTime->tm_min, utcTime->tm_sec);
        strcat_s(filename, nameLength, prof);

        // Write Profile to file
        file = CreateFile(filename, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (file == INVALID_HANDLE_VALUE)
        {
            hr = HRESULT_FROM_WIN32(GetLastError());
            break;
        }

        DWORD bytesWritten;
        if (!WriteFile(file, buffer, bufferSize * sizeof(WCHAR), &bytesWritten, NULL))
        {
            hr = HRESULT_FROM_WIN32(GetLastError());
            break;
        }
        if (file != NULL)
            CloseHandle(file);

        if (profileMap.size() != 0)
        {
            IWMProfile *profileSelection = profile;
            ClearComboBox();
            profile = profileSelection;
            ProfileBoxName(NULL, buffer, NULL);
            FillCompareComboBox(NULL, NULL);
            profile = profileSelection;
        }
        else
        {
            ProfileBoxName(NULL, buffer, NULL);
            FillCompareComboBox(NULL, NULL);
        }
    } while (false);

    //	if (buffer) delete[] buffer;

    SAFE_RELEASE(profileMgr);

    return (hr);
}

HRESULT WINAVIPlugin::CreateStream(int width, int height, DWORD compression, DWORD bitrate, const char *codec,
                                   GUID *subtype)
{
    HRESULT hr = S_OK;
    IWMStreamConfig *streamConfig = NULL;
    IWMVideoMediaProps *mediaProps = NULL;
    WM_MEDIA_TYPE *mediaType = NULL;
    IWMPropertyVault *propertyBag = NULL;

    do
    {

        hr = profile->CreateNewStream(WMMEDIATYPE_Video, &streamConfig);
        if (FAILED(hr))
            break;

        WORD streamCount;
        hr = streamConfig->GetStreamNumber(&streamCount);
        if (FAILED(hr))
            break;

        if (myPlugin->selectFrameRate->getSelectedEntry() == 2)
            streamConfig->SetStreamName(L"Video Stream");
        else
            streamConfig->SetStreamName(L"Video Stream OpenCOVERVarfps");
        if (FAILED(hr))
            break;

        LPWSTR connectName = (LPWSTR) new WCHAR[100];

        swprintf_s(connectName, 100, L"%s%d", L"Video", streamCount);
        hr = streamConfig->SetConnectionName(connectName);
        if (FAILED(hr))
            break;

        hr = streamConfig->SetBufferWindow(5000);
        if (FAILED(hr))
            break;

        hr = streamConfig->QueryInterface(IID_IWMVideoMediaProps, (void **)&mediaProps);
        if (FAILED(hr))
            break;

        DWORD mediaTypeSize;
        hr = mediaProps->GetMediaType(NULL, &mediaTypeSize);
        if (FAILED(hr))
            break;
        mediaType = (WM_MEDIA_TYPE *)new BYTE[mediaTypeSize];
        if (!mediaType)
        {
            hr = E_OUTOFMEMORY;
            break;
        }
        hr = mediaProps->GetMediaType(mediaType, &mediaTypeSize);

        if (mediaType->formattype != WMFORMAT_VideoInfo)
        {
            SAFE_RELEASE(streamConfig);
            continue;
        }

        hr = mediaProps->SetMediaType(mediaType);

        WMVIDEOINFOHEADER *videoHeader = (WMVIDEOINFOHEADER *)mediaType->pbFormat;

        hr = streamConfig->QueryInterface(IID_IWMPropertyVault, (void **)&propertyBag);
        if (FAILED(hr))
            break;

        if (compression == 1)
        {
            propertyBag->SetProperty(g_wszVBREnabled, WMT_TYPE_BOOL, (BYTE *)&compression,
                                     sizeof(WMT_TYPE_BOOL));
            propertyBag->SetProperty(g_wszVBRQuality, WMT_TYPE_DWORD, (BYTE *)&bitrate, sizeof(WMT_TYPE_DWORD));
            DWORD tmp = 0;
            propertyBag->SetProperty(g_wszVBRBitrateMax, WMT_TYPE_DWORD, (BYTE *)&tmp, sizeof(WMT_TYPE_DWORD));
            propertyBag->SetProperty(g_wszVBRBufferWindowMax, WMT_TYPE_DWORD, (BYTE *)&tmp,
                                     sizeof(WMT_TYPE_DWORD));
            hr = streamConfig->SetBitrate(0);
            if (FAILED(hr))
                break;

            videoHeader->dwBitRate = 1;
        }
        else
        {
            bitrate = bitrate * 1000;
            hr = streamConfig->SetBitrate(bitrate);
            if (FAILED(hr))
                break;

            videoHeader->dwBitRate = bitrate;
        }

        videoHeader->rcSource.top = videoHeader->rcSource.left = 0;
        videoHeader->rcSource.right = width;
        videoHeader->rcSource.bottom = height;
        videoHeader->rcTarget = videoHeader->rcSource;

        videoHeader->AvgTimePerFrame = (LONGLONG)1000 * 100 * 100 / myPlugin->time_base;
        hr = mediaProps->SetMediaType(mediaType);

        videoHeader->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        videoHeader->bmiHeader.biBitCount = 24;
        videoHeader->bmiHeader.biWidth = width;
        videoHeader->bmiHeader.biHeight = height;
        videoHeader->bmiHeader.biPlanes = 1;
        videoHeader->bmiHeader.biSizeImage = width * height * 24 / 8;
        hr = mediaProps->SetMediaType(mediaType);

        mediaType->majortype = WMMEDIATYPE_Video;
        if (!subtype)
        {
            list<codecStruct>::iterator it = codecList.end();
            it--;
            while (strcmp((*it).version, codec) != 0)
                it--;

            mediaType->subtype = (*it).guidType;
        }
        else
            mediaType->subtype = *subtype;
        videoHeader->bmiHeader.biCompression = mediaType->subtype.Data1;

        hr = mediaProps->SetMediaType(mediaType);

        mediaType->bFixedSizeSamples = false;
        mediaType->bTemporalCompression = true;
        mediaType->lSampleSize = 0;
        mediaType->formattype = WMFORMAT_VideoInfo;
        mediaType->pUnk = NULL;
        mediaType->cbFormat = sizeof(WMVIDEOINFOHEADER);

        hr = mediaProps->SetMediaType(mediaType);
        if (FAILED(hr))
            return (hr);

        hr = mediaProps->SetQuality(100);
        if (FAILED(hr))
            break;

        if (bitrate < 300000)
            hr = mediaProps->SetMaxKeyFrameSpacing(10000000 * (QWORD)8);
        else if (bitrate < 600000)
            hr = mediaProps->SetMaxKeyFrameSpacing(10000000 * (QWORD)6);
        else if (bitrate < 2000000)
            hr = mediaProps->SetMaxKeyFrameSpacing(10000000 * (QWORD)4);
        else
            hr = mediaProps->SetMaxKeyFrameSpacing(10000000 * (QWORD)3);
        if (FAILED(hr))
            break;

        hr = profile->AddStream(streamConfig);
        if (FAILED(hr))
            break;
    } while (false);

    if (mediaType)
        delete[] mediaType;
    SAFE_RELEASE(propertyBag);
    SAFE_RELEASE(mediaProps);
    SAFE_RELEASE(streamConfig);
    return (hr);
}

HRESULT WINAVIPlugin::SetStreamFeatures(int width, int height, DWORD bitrate, const char *codec)
{
    HRESULT hr = S_OK;
    IWMStreamConfig *streamConfig = NULL;
    IWMVideoMediaProps *mediaProps = NULL;
    WM_MEDIA_TYPE *mediaType = NULL;
    IWMPropertyVault *propertyBag = NULL;
    WMT_ATTR_DATATYPE *vbr = NULL;
    DWORD numberStreams;
    WORD streamNumber = 1;

    hr = profile->GetStreamCount(&numberStreams);
    if (hr == S_OK)
        for (WORD i = 1; i <= numberStreams; i++)
            do
            {
                do
                {
                    hr = profile->GetStreamByNumber(streamNumber, &streamConfig);
                    streamNumber++;
                } while (FAILED(hr));

                WORD streamNameLength;
                hr = streamConfig->GetStreamName(NULL, &streamNameLength);
                WCHAR *streamName = new WCHAR[streamNameLength];
                if (streamName == NULL)
                {
                    hr = E_OUTOFMEMORY;
                    break;
                }

                hr = streamConfig->GetStreamName(streamName, &streamNameLength);
                if (FAILED(hr))
                    break;

                if (wcsncmp(streamName, L"Audio", 5) == 0)
                {
                    profile->RemoveStreamByNumber(i);
                    break;
                }

                hr = streamConfig->QueryInterface(IID_IWMVideoMediaProps, (void **)&mediaProps);
                if (FAILED(hr))
                    break;

                DWORD mediaTypeSize;
                hr = mediaProps->GetMediaType(NULL, &mediaTypeSize);
                if (FAILED(hr))
                    break;
                mediaType = (WM_MEDIA_TYPE *)new BYTE[mediaTypeSize];
                if (!mediaType)
                {
                    hr = E_OUTOFMEMORY;
                    break;
                }
                hr = mediaProps->GetMediaType(mediaType, &mediaTypeSize);

                if (mediaType->formattype != WMFORMAT_VideoInfo)
                {
                    SAFE_RELEASE(streamConfig);
                    continue;
                }

                hr = mediaProps->SetMediaType(mediaType);

                WMVIDEOINFOHEADER *videoHeader = (WMVIDEOINFOHEADER *)mediaType->pbFormat;

                hr = streamConfig->QueryInterface(IID_IWMPropertyVault, (void **)&propertyBag);
                if (FAILED(hr))
                    break;

                WMT_ATTR_DATATYPE WMTAttrType = WMT_TYPE_BOOL;
                DWORD size = sizeof(bool);
                hr = propertyBag->GetPropertyByName(g_wszVBREnabled, &WMTAttrType, NULL, &size);
                if (!FAILED(hr))
                {
                    vbr = (WMT_ATTR_DATATYPE *)new BYTE[size];
                    if (!vbr)
                    {
                        hr = E_OUTOFMEMORY;
                        break;
                    }
                    propertyBag->GetPropertyByName(g_wszVBREnabled, &WMTAttrType, (BYTE *)&vbr, &size);
                }
                else
                    hr = S_OK;

                if (vbr)
                {
                    propertyBag->SetProperty(g_wszVBRQuality, WMT_TYPE_DWORD, (BYTE *)&bitrate,
                                             sizeof(WMT_TYPE_DWORD));
                    DWORD tmp = 0;
                    propertyBag->SetProperty(g_wszVBRBitrateMax, WMT_TYPE_DWORD, (BYTE *)&tmp,
                                             sizeof(WMT_TYPE_DWORD));
                    propertyBag->SetProperty(g_wszVBRBufferWindowMax, WMT_TYPE_DWORD, (BYTE *)&tmp,
                                             sizeof(WMT_TYPE_DWORD));
                    hr = streamConfig->SetBitrate(0);
                    if (FAILED(hr))
                        break;

                    videoHeader->dwBitRate = 1;
                }
                else
                {
                    bitrate = bitrate * 1000;
                    hr = streamConfig->SetBitrate(bitrate);
                    if (FAILED(hr))
                        break;

                    videoHeader->dwBitRate = bitrate;
                }

                videoHeader->rcSource.left = videoHeader->rcSource.top = 0;
                videoHeader->rcSource.right = width;
                videoHeader->rcSource.bottom = height;
                videoHeader->rcTarget = videoHeader->rcSource;

                videoHeader->AvgTimePerFrame = (LONGLONG)1000 * 100 * 100 / myPlugin->time_base;

                videoHeader->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                videoHeader->bmiHeader.biBitCount = 24;
                videoHeader->bmiHeader.biWidth = width;
                videoHeader->bmiHeader.biHeight = height;

                videoHeader->bmiHeader.biSizeImage = width * height * 24 / 8;

                list<codecStruct>::iterator it = codecList.begin();
                while (strcmp((*it).version, codec) != 0)
                    it++;
                mediaType->subtype = (*it).guidType;
                videoHeader->bmiHeader.biCompression = mediaType->subtype.Data1;

                hr = mediaProps->SetMediaType(mediaType);
                if (FAILED(hr))
                    return (hr);

                if (bitrate < 300000)
                    hr = mediaProps->SetMaxKeyFrameSpacing(10000000 * (QWORD)8);
                else if (bitrate < 600000)
                    hr = mediaProps->SetMaxKeyFrameSpacing(10000000 * (QWORD)6);
                else if (bitrate < 2000000)
                    hr = mediaProps->SetMaxKeyFrameSpacing(10000000 * (QWORD)4);
                else
                    hr = mediaProps->SetMaxKeyFrameSpacing(10000000 * (QWORD)3);
                if (FAILED(hr))
                    break;

                hr = profile->ReconfigStream(streamConfig);
                if (FAILED(hr))
                    break;

            } while (false);

    if (mediaType)
        delete[] mediaType;
    SAFE_RELEASE(propertyBag);
    SAFE_RELEASE(mediaProps);
    SAFE_RELEASE(streamConfig);
    return (hr);
}

HRESULT WINAVIPlugin::LoadProfileFeatures()
{
    HRESULT hr = S_OK;
    IWMStreamConfig *streamConfig = NULL;
    IWMVideoMediaProps *mediaProps = NULL;
    IWMPropertyVault *propertyBag = NULL;
    WM_MEDIA_TYPE *mediaType = NULL;
    WMT_ATTR_DATATYPE *vbr = NULL;
    DWORD numberStreams;
    WORD streamNumber = 1;

    hr = profile->GetStreamCount(&numberStreams);
    if (hr == S_OK)
        for (WORD i = 1; i <= numberStreams; i++)
            do
            {
                do
                {
                    hr = profile->GetStreamByNumber(streamNumber, &streamConfig);
                    streamNumber++;
                } while (FAILED(hr));

                WORD streamNameLength;
                hr = streamConfig->GetStreamName(NULL, &streamNameLength);
                WCHAR *streamName = new WCHAR[streamNameLength];
                if (streamName == NULL)
                {
                    hr = E_OUTOFMEMORY;
                    break;
                }

                hr = streamConfig->GetStreamName(streamName, &streamNameLength);
                if (FAILED(hr))
                    break;

                if (wcsncmp(streamName, L"Audio", 5) == 0)
                    break;

                hr = streamConfig->QueryInterface(IID_IWMVideoMediaProps, (void **)&mediaProps);
                if (FAILED(hr))
                    break;

                DWORD mediaTypeSize;
                hr = mediaProps->GetMediaType(NULL, &mediaTypeSize);
                if (FAILED(hr))
                    break;
                mediaType = (WM_MEDIA_TYPE *)new BYTE[mediaTypeSize];
                if (!mediaType)
                {
                    hr = E_OUTOFMEMORY;
                    break;
                }
                hr = mediaProps->GetMediaType(mediaType, &mediaTypeSize);

                if (mediaType->formattype != WMFORMAT_VideoInfo)
                {
                    SAFE_RELEASE(streamConfig);
                    continue;
                }

                hr = streamConfig->QueryInterface(IID_IWMPropertyVault, (void **)&propertyBag);
                if (FAILED(hr))
                    break;

                WMT_ATTR_DATATYPE WMTAttrType = WMT_TYPE_BOOL;
                DWORD size = sizeof(bool);
                DWORD bits;
                hr = propertyBag->GetPropertyByName(g_wszVBREnabled, &WMTAttrType, NULL, &size);
                if (!FAILED(hr))
                {
                    vbr = (WMT_ATTR_DATATYPE *)new BYTE[size];
                    if (!vbr)
                    {
                        hr = E_OUTOFMEMORY;
                        break;
                    }
                    propertyBag->GetPropertyByName(g_wszVBREnabled, &WMTAttrType, (BYTE *)&vbr, &size);
                }
                else
                    hr = S_OK;

                if (vbr)
                {
                    bitrate->setSelectedEntry(1);
                    WMTAttrType = WMT_TYPE_DWORD;
                    hr = propertyBag->GetPropertyByName(g_wszVBRQuality, &WMTAttrType, NULL, &size);
                    if (!FAILED(hr))
                    {
                        WMT_ATTR_DATATYPE *quality = (WMT_ATTR_DATATYPE *)new BYTE[size];
                        if (!vbr)
                        {
                            hr = E_OUTOFMEMORY;
                            break;
                        }
                        propertyBag->GetPropertyByName(g_wszVBRQuality, &WMTAttrType, (BYTE *)&bits, &size);
                        bitrateField->setValue(bits);
                    }
                }
                else
                {
                    bitrate->setSelectedEntry(0);
                    hr = streamConfig->GetBitrate(&bits);
                    if (FAILED(hr))
                        break;
                    bitrateField->setValue(bits / 1000);
                }
                checkBitrateOrQuality(bitrate->getSelectedEntry(), bitrateField->getValue());

                WMVIDEOINFOHEADER *videoHeader = (WMVIDEOINFOHEADER *)mediaType->pbFormat;

                if (videoHeader->rcSource.right > videoHeader->rcSource.left)
                {
                    myPlugin->outWidthField->setValue(videoHeader->rcSource.right);
                    myPlugin->outHeightField->setValue(videoHeader->rcSource.bottom);
                }
                else
                {
                    myPlugin->outWidthField->setValue(videoHeader->rcSource.left);
                    myPlugin->outHeightField->setValue(videoHeader->rcSource.top);
                }

                if (!wcsstr(streamName, L"OpenCOVERVarfps"))
                {
                    myPlugin->selectFrameRate->setSelectedEntry(2);
                    myPlugin->cfpsEdit->setValue(10000000 / videoHeader->AvgTimePerFrame);
                }
                else
                {
                    if (fabs((double)(1000 * 100 * 100 / videoHeader->AvgTimePerFrame - 25)) < fabs((double)(1000 * 100 * 100 / videoHeader->AvgTimePerFrame - 30)))
                        myPlugin->selectFrameRate->setSelectedEntry(0);
                    else
                        myPlugin->selectFrameRate->setSelectedEntry(1);
                    myPlugin->cfpsEdit->setValue(coVRConfig::instance()->frameRate());
                }

                DWORD j = 0;
                std::list<codecStruct>::iterator it = codecList.begin();
                while ((it != codecList.end()) && ((*it).guidType != mediaType->subtype))
                {
                    it++;
                    j++;
                }
                selectCompressionCodec->setSelectedEntry(j);

            } while (false);

    if (mediaType)
        delete[] mediaType;
    SAFE_RELEASE(mediaProps);
    SAFE_RELEASE(streamConfig);
    SAFE_RELEASE(propertyBag);

    return (hr);
}

HRESULT WINAVIPlugin::SetProfileFeatures(bool newProfile, const char *name)
{
    HRESULT hr = S_OK;
    IWMProfileManager *profileMgr = NULL;
    IWMProfileManager2 *profileMgr2 = NULL;

    //
    // Create profile manager
    //

    do
    {
        hr = WMCreateProfileManager(&profileMgr);
        if (FAILED(hr))
            continue;

        hr = profileMgr->QueryInterface(IID_IWMProfileManager2, (void **)&profileMgr2);
        if (FAILED(hr))
            break;

        if (newProfile)
        {
            hr = profileMgr->CreateEmptyProfile(wmtVersion, &profile);
            if (FAILED(hr))
                break;
        }

        WCHAR *prof = L"";
        int numBytes = MultiByteToWideChar(CP_ACP, 0, name, -1, prof, 0);
        prof = new WCHAR[numBytes];
        MultiByteToWideChar(CP_ACP, 0, name, -1, prof, numBytes);
        hr = profile->SetName(prof);
        if (FAILED(hr))
            break;

        hr = profile->SetDescription(prof);
        if (FAILED(hr))
            break;

    } while (false);

    SAFE_RELEASE(profileMgr2);
    SAFE_RELEASE(profileMgr);
    return (hr);
}

HRESULT WINAVIPlugin::CreateProfile(const char *name, int width, int height, DWORD compression, DWORD bitrate,
                                    const char *codec, LPWSTR &profileString, bool save, GUID *subtype)
{
    HRESULT hr = S_OK;

    if (!name)
    {
        char newName[200];
        if (strchr(codec, '8') != NULL)
        {
            sprintf(newName, "Windows Media Video 8 %dx%d %d kbit/s", width, height, bitrate);
            wmtVersion = WMT_VER_8_0;
        }
        else if (strchr(codec, '7') != NULL)
        {
            sprintf(newName, "Windows Media Video 7 %dx%d %d kbit/s", width, height, bitrate);
            wmtVersion = WMT_VER_7_0;
        }
        else
        {
            sprintf(newName, "Windows Media Video 9 %dx%d %d kbit/s", width, height, bitrate);
            wmtVersion = WMT_VER_9_0;
        }
        hr = SetProfileFeatures(true, newName);
    }
    else
        hr = SetProfileFeatures(true, name);
    if (FAILED(hr))
        return (hr);

    hr = CreateStream(width, height, compression, bitrate, codec, subtype);
    if (FAILED(hr))
        return (hr);

    if (save)
        hr = SaveProfile(profileString);

    return (hr);
}

HRESULT WINAVIPlugin::LoadProfile(DWORD profileIndex)
{
    HRESULT hr = S_OK;
    IWMProfileManager *profileMgr = NULL;
    IWMProfileManager2 *profileMgr2 = NULL;

    //
    // Create profile manager
    //
    hr = WMCreateProfileManager(&profileMgr);

    if (hr == S_OK)
        do
        {

            hr = profileMgr->QueryInterface(IID_IWMProfileManager2, (void **)&profileMgr2);
            if (FAILED(hr))
                break;

            std::multimap<WMT_VERSION, LPWSTR>::iterator it = profileMap.begin();
            for (DWORD i = 0; i < profileIndex; i++)
                it++;
            hr = profileMgr2->SetSystemProfileVersion((*it).first);
            if (FAILED(hr))
                break;

            hr = profileMgr->LoadProfileByData((*it).second, &profile);
            if (FAILED(hr))
                break;

            hr = profile->GetVersion(&wmtVersion);
            if (FAILED(hr))
                break;

            hr = LoadProfileFeatures();
            if (FAILED(hr))
                break;

            if (profileMgr2)
                profileMgr2->Release();
        } while (FALSE);

    if (profileMgr)
        profileMgr->Release();

    return (hr);
}

HRESULT WINAVIPlugin::LoadCustomProfile(LPCTSTR filename, LPWSTR &profileString)
{
    HRESULT hr = S_OK;
    IWMProfileManager *profileMgr = NULL;
    HANDLE hFile = NULL;
    DWORD profileLength = 0;

    do
    {
        hr = WMCreateProfileManager(&profileMgr);
        if (FAILED(hr))
            break;

        // Open the profile file
        if (filename != NULL)
            hFile = CreateFile(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile != INVALID_HANDLE_VALUE)
        {

            profileLength = GetFileSize(hFile, NULL);

            // Memory for profile buffer, null-terminated
            LPWSTR buffer = (LPWSTR) new BYTE[profileLength + sizeof(WCHAR)];
            if (buffer == NULL)
            {
                hr = E_OUTOFMEMORY;
                break;
            }
            memset(buffer, 0, profileLength + sizeof(WCHAR));

            DWORD bytesRead;
            ReadFile(hFile, buffer, profileLength, &bytesRead, NULL);

            // Load profile from buffer
            hr = profileMgr->LoadProfileByData(buffer, &profile);
            if (FAILED(hr))
                break;

            profileString = buffer;

            hr = profile->GetVersion(&wmtVersion);
            if (FAILED(hr))
                break;
        }

    } while (FALSE);

    SAFE_RELEASE(profileMgr);
    if (hFile != NULL)
        CloseHandle(hFile);

    return (hr);
}

HRESULT WINAVIPlugin::FindWriterInputs(int width, int height)
{
    HRESULT hr = S_OK;
    DWORD inputNumber;
    DWORD inputFormatCount = 0;
    IWMInputMediaProps *inputProps = NULL;
    IWMStreamConfig *streamConfig = NULL;
    GUID inputType;
    WM_MEDIA_TYPE *inputMediaType = NULL;
    DWORD sizeMediaType = 0;

    hr = writer->GetInputCount(&inputNumber);
    if (FAILED(hr))
        return (hr);

    DWORD i = 0;
    // look for the properties of all inputs
    do
    {
        // Find the number of formats supported by this input.
        hr = writer->GetInputFormatCount(i, &inputFormatCount);
        if (FAILED(hr))
            return (hr);

        for (DWORD formatIndex = 0; formatIndex < inputFormatCount; formatIndex++)
        {
            // Get the input media properties for the input format.
            hr = writer->GetInputFormat(i, formatIndex, &inputProps);
            if (FAILED(hr))
                return (hr);

            hr = inputProps->GetType(&inputType);
            if (FAILED(hr))
                return (hr);
            if (inputType == WMMEDIATYPE_Video)
                videoInput = i;
            else
                break;

            hr = inputProps->GetMediaType(NULL, &sizeMediaType);
            if (FAILED(hr))
                return (hr);

            inputMediaType = (WM_MEDIA_TYPE *)new BYTE[sizeMediaType];
            if (NULL == inputMediaType)
            {
                hr = E_OUTOFMEMORY;
                return (hr);
            }
            hr = inputProps->GetMediaType(inputMediaType, &sizeMediaType);
            if (FAILED(hr))
                return (hr);

            if (inputMediaType->subtype == WMMEDIASUBTYPE_RGB24)
            {
                WMVIDEOINFOHEADER *videoInfo = (WMVIDEOINFOHEADER *)inputMediaType->pbFormat;

                videoInfo->bmiHeader.biWidth = width;
                videoInfo->bmiHeader.biHeight = height;

                // Stride = (width * bytes/pixel), rounded to the next DWORD boundary.
                long stride = (width * (videoInfo->bmiHeader.biBitCount / 8) + 3) & ~3;

                // Image size = stride * height.
                videoInfo->bmiHeader.biSizeImage = height * stride;

                hr = inputProps->SetMediaType(inputMediaType);
                if (FAILED(hr))
                    return (hr);

                hr = writer->SetInputProps(i, inputProps);
                if (FAILED(hr))
                    return (hr);

                //		fprintf(stderr,"Majortype: %lx, Subtype: %lx, Formattype: %lx\n", inputMediaType->majortype,
                // inputMediaType->subtype, inputMediaType->formattype);

                hr = writer->SetProfile(profile);
                if (FAILED(hr))
                    return (hr);

                delete[] inputMediaType;
                SAFE_RELEASE(inputProps);
                SAFE_RELEASE(streamConfig);
                break;
            }
            else
                inputMediaType = NULL;
        }

    } while (!inputMediaType && (++i < inputNumber));

    return (hr);
}

HRESULT WINAVIPlugin::InitWMVWriter(const string &filename)
{
    HRESULT hr = S_OK;
    IWMWriterAdvanced2 *advancedWriter = NULL;

    profileErrorLabel->setLabel("");
    videoTime = 0;
    hr = WMCreateWriter(NULL, &writer);
    if (FAILED(hr))
        return (hr);

    hr = writer->QueryInterface(IID_IWMWriterAdvanced2, (void **)&advancedWriter);
    if (FAILED(hr))
        return (hr);
    // Memory leak during live recording
    //	hr = advancedWriter->SetLiveSource(true);
    //		hr = advancedWriter->SetSyncTolerance(6000);
    //	if (FAILED(hr)) return(hr);

    //  Save profile to writer
    hr = writer->SetProfile(profile);
    if (FAILED(hr))
        return (hr);

    WCHAR *wFilename = NULL;

    int nLen = MultiByteToWideChar(CP_ACP, 0, filename.c_str(), -1, wFilename, 0);
    if (nLen == 0)
        return (E_FAIL);

    wFilename = (WCHAR *)malloc(nLen * sizeof(WCHAR) + 1);
    MultiByteToWideChar(CP_ACP, 0, filename.c_str(), -1, wFilename, nLen);
    hr = writer->SetOutputFilename(wFilename);
    free(wFilename);

    if (myPlugin->resize)
    {
        swsconvertctx = sws_getContext(myPlugin->widthField->getValue(), myPlugin->heightField->getValue(),
                                       myPlugin->capture_fmt, myPlugin->outWidthField->getValue(),
                                       myPlugin->outHeightField->getValue(), myPlugin->capture_fmt, SWS_BICUBIC,
                                       NULL, NULL, NULL);

        inPicture = new Picture;
        memset(inPicture, 0, sizeof(Picture));
        inPicture->linesize[0] = linesize;

        outPicture = new Picture;
        memset(outPicture, 0, sizeof(Picture));
        outPicture->linesize[0] = myPlugin->outWidthField->getValue() * 3;
        tmppixels = new uint8_t[sampleSize];
        outPicture->data[0] = tmppixels;
    }

    hr = FindWriterInputs(myPlugin->outWidthField->getValue(), myPlugin->outHeightField->getValue());
    if (FAILED(hr))
        return (hr);

    hr = writer->BeginWriting();
    if (FAILED(hr))
        return (hr);

    SAFE_RELEASE(advancedWriter);
    return (hr);
}

HRESULT WINAVIPlugin::CloseVideoWMV()
{
    HRESULT hr = S_OK;

    if (writer)
    {
        hr = writer->EndWriting();
        SAFE_RELEASE(writer);
    }

    return (hr);
}

bool WINAVIPlugin::checkBitrateOrQuality(DWORD entry, int bits)
{
    if ((bits <= 0) || ((entry == 1) && (bits > 100)))
    {
        myPlugin->errorLabel->setLabel("Please check bitrate or quality.");
        return false;
    }
    else
        myPlugin->errorLabel->setLabel("");

    return true;
}

void WINAVIPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if ((tUIItem == profileNameField) && coVRMSController::instance()->isMaster())
        ProfileNameTest();
    if (tUIItem == selectCodec)
    {
        LoadProfile(selectCodec->getSelectedEntry());
    }
    if (tUIItem == selectCompressionCodec)
    {
        myPlugin->errorLabel->setLabel("");
        if (strchr(selectCompressionCodec->getSelectedText().c_str(), '8') != NULL)
            wmtVersion = WMT_VER_8_0;
        else if (strchr(selectCompressionCodec->getSelectedText().c_str(), '7') != NULL)
            wmtVersion = WMT_VER_7_0;
        else
            wmtVersion = WMT_VER_9_0;
    }
    if ((tUIItem == bitrateField) || (tUIItem == bitrate))
        checkBitrateOrQuality(bitrate->getSelectedEntry(), bitrateField->getValue());

    if (tUIItem == saveButton)
    {
        HRESULT hr = S_OK;
        LPWSTR profileString = NULL;
        int count = 0;
        if (coVRMSController::instance()->isMaster())
        {
            if (ProfileNameTest() != S_OK)
                coVRMSController::instance()->sendSlaves((char *)&count, sizeof(int));
            else
            {
                hr = CreateProfile(&(*profileNameField->getText().c_str()), myPlugin->outWidthField->getValue(),
                                   myPlugin->outHeightField->getValue(), bitrate->getSelectedEntry(),
                                   bitrateField->getValue(),
                                   &(*selectCompressionCodec->getSelectedText().c_str()), profileString, true);
                selectCodec->setSelectedText(profileNameField->getText());

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
        }
        else
        {
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

                    hr = SlaveAddProfile(profileString, true);
                }
                delete[] buffer;
            }
        }
        saveButton->setState(false);
    }

    if (AVIButton && (tUIItem == AVIButton))
    {
        myPlugin->errorLabel->setLabel("");
        ChooseVideoCodec();
        AVIButton->setState(false);
    }
}

void WINAVIPlugin::changeFormat(coTUIElement *tUIItem, int row)
{
    if (tUIItem == myPlugin->selectFormat)
        if (myPlugin->selectFormat->getSelectedEntry() == 0)
        {
            filterList = "*.avi";
            HideWMVMenu(true);
            AVIButton->setHidden(false);
            myPlugin->fileNameBrowser->setFilterList(filterList);
        }
        else
        {
            filterList = "*.wmv";
            AVIButton->setHidden(true);
            HideWMVMenu(false);
            LoadProfile(0);
            selectCodec->setSelectedEntry(0);
            myPlugin->fileNameBrowser->setFilterList(filterList);
        }
    myPlugin->fillFilenameField("", false);
}

#endif
