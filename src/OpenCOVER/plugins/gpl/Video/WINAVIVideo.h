
#include <vfw.h>

#include <config/coConfigConstants.h>

#include <vfw.h>
#include <wmsdk.h>
#include <qstring.h>

extern "C" {
#include "swscale.h"
};

#include "Video.h"

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(x) \
    if (NULL != x)      \
    {                   \
        x->Release();   \
        x = NULL;       \
    }
#endif // SAFE_RELEASE

enum codecType
{
    CODEC_ID_RAWVIDEO,
    CODEC_ID_MPEG4,
    CODEC_ID_INDEO,
    CODEC_ID_CINEPAK,
    CODEC_ID_YUV,
    CODEC_ID_WMV
};

typedef struct
{
    uint8_t *data[4];
    int linesize[4];
} Picture;

typedef struct
{
    char *version;
    GUID guidType;
} codecStruct;

class WINAVIPlugin : public coVRPlugin, public coTUIListener
{
public:
    WINAVIPlugin();
    ~WINAVIPlugin();

    VideoPlugin *myPlugin;
    coTUILabel *frameRate;
    coTUIToggleButton *AVIButton;
    coTUIComboBox *selectCompressionCodec;
    coTUIComboBox *selectCodec;
    coTUIComboBox *bitrate;
    coTUIEditField *profileNameField;
    coTUILabel *profileErrorLabel;
    coTUIEditIntField *bitrateField;
    coTUIToggleButton *saveButton;

    friend class VideoPlugin;

    void tabletEvent(coTUIElement *);

private:
    std::string filterList;
    HRESULT aviInit(const string &filename, short frame_rate);
    bool ChooseVideoCodec();
    void videoWrite(int format);
    void videoWriteDepth(int format, float *buf);
    void free_all();
    void Menu(int row);
    void changeFormat(coTUIElement *, int row);
    void checkFileFormat(string &name);
    bool videoCaptureInit(string &name, int format, int RGBFormat);
    void fillCodecComboBox();

    void init_GLbuffers();
    void close_all(bool stream, int format = 0);

    AVISTREAMINFO aviStreamInfo;
    AVICOMPRESSOPTIONS aviCompressOpt;
    PAVIFILE aviFile;
    PAVISTREAM aviStream;
    PAVIFILE aviFile2;
    PAVISTREAM aviStream2;
    PAVISTREAM aviCompressed;
    PAVISTREAM aviCompressed2;
    FILE *transFP;
    BITMAPINFO bmpInfo;
    SwsContext *swsconvertctx;
    COMPVARS *compressSet;
    LONG aviSample;
    uint8_t *tmppixels;
    int linesize;
    Picture *inPicture, *outPicture;
    DWORD sampleSize;
    HRESULT hres;

#ifdef HAVE_WMFSDK
    IWMProfile *profile;
    IWMWriter *writer;
    double videoTime;
    DWORD videoInput;
    WMT_VERSION wmtVersion;
    std::multimap<WMT_VERSION, LPWSTR> profileMap;
    std::list<codecStruct> codecList;

    HRESULT WriteWMVSample();
    HRESULT ListSystemProfiles();
    HRESULT ListCustomProfiles();
    HRESULT LoadProfileFeatures();
    HRESULT ListCodecs(bool test);

    HRESULT ProfileBoxName(void (coTUIComboBox::*AddDel)(const string &), LPWSTR profileString, bool *compare);
    HRESULT FillCompareComboBox(char *name, bool *compare);
    HRESULT ClearComboBox(void);
    HRESULT SaveProfile(LPWSTR &profileString);
    HRESULT CreateStream(int width, int height, DWORD compression, DWORD bitrate, const char *codec,
                         GUID *subtype = NULL);
    HRESULT SetStreamFeatures(int widht, int height, DWORD bitrate, const char *codec);
    HRESULT SetProfileFeatures(bool newProfile, const char *name);
    HRESULT LoadProfile(DWORD profileIndex);
    HRESULT LoadCustomProfile(LPCTSTR filename, LPWSTR &profileString);
    HRESULT FindWriterInputs(int width, int height);
    HRESULT InitWMVWriter(const string &filename);
    HRESULT CreateProfile(const char *name, int width, int height, DWORD compression, DWORD bitrate,
                          const char *codec, LPWSTR &profileString, bool save = false, GUID *subtype = NULL);
    HRESULT CloseVideoWMV(void);
    HRESULT ProfileNameTest();
    HRESULT FindCustomProfiles(QString *path);
    HRESULT SlaveAddProfile(LPWSTR profileString, bool fillCombo);
    void WMVMenu(int row);
    void RemoveWMVMenu();
    void HideWMVMenu(bool);
    bool checkBitrateOrQuality(DWORD entry, int bits);
    HRESULT checkCodec(codecStruct *codec);
    HRESULT checkProfile();
#endif
};
