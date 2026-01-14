#ifndef CEF_H_INCLUDED
#define CEF_H_INCLUDED

#include <cover/coVRPluginSupport.h>
#include <cover/ui/Owner.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/sginterface/vruiCollabInterface.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
namespace vrui
{
    class coRowMenu;
    class coSubMenuItem;
    class coCheckboxMenuItem;
    class coButtonMenuItem;
    class coPotiMenuItem;
    class coTrackerButtonInteraction;
    class coTexturedBackground;
    class coPopupHandle;
    class coLabel;
    class coCombinedButtonInteraction;
    class coColoredBackground;
    class vruiHit;
    class OSGVruiTransformNode;
}
namespace opencover
{
    namespace ui
    {
        class Menu;
        class Label;
        class Group;
        class Action;
        class Button;
        class EditField;
    }
    class coVRLabel;
    class coCOIM;
}

#include "include/cef_app.h"
#include "include/cef_client.h"
#include "include/cef_render_handler.h"
#include "include/cef_app.h"


using namespace vrui;
using namespace opencover;
class CefAppHandler;
class CEF;

struct ImageBuffer
{
    std::vector<unsigned char> buffer;
    bool bufferChanged = false;
    std::mutex mutex;
    ImageBuffer(const std::vector<unsigned char> &v) : buffer(v) {};
    // ImageBuffer() = default;
    ImageBuffer(const ImageBuffer &) = delete;
    ImageBuffer &operator=(const ImageBuffer &) = delete;
};

struct InputEvent{
enum Type{
    None,
    MouseMove,
    LeftClick,
    RightClick,
    MiddleClick,
    MouseWheel,
    KeyEvent, 
    GoBack,
    GoForward,
    Reload, 
    Text,
    OpenURL, 
    SetFocus,
    CloseBrowser,
    SelectAll,
    Copy,
    Paste,
} type = None;
std::string text; // for Text and OpenURL
int x = 0, y = 0; // for MouseMove and MouseClick
bool on = true; // for MouseClick and SetFocus
int wheelDelta = 0; // for MouseWheel
CefRefPtr<CefBrowser> browser = nullptr; // for CloseBrowser
CefKeyEvent keyEvent;
};


class CEF_client : public CefClient, public CefRenderHandler, public CefLifeSpanHandler, public CefContextMenuHandler
{

public:
    CEF_client(CefAppHandler *c);
    

    CefRefPtr<CefRenderHandler> GetRenderHandler() override;
    CefRefPtr<CefLifeSpanHandler> GetLifeSpanHandler() override;
    CefRefPtr<CefContextMenuHandler> GetContextMenuHandler() override;
    void OnBeforeClose(CefRefPtr<CefBrowser> browser) override;
    bool DoClose(CefRefPtr<CefBrowser> browser) override;

#ifdef _WIN32
    void OnBeforeContextMenu(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, CefRefPtr<CefContextMenuParams> params, CefRefPtr<CefMenuModel> model) override;
    bool OnContextMenuCommand(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, CefRefPtr<CefContextMenuParams> params, int command_id, EventFlags event_flags) override;
#endif
    void GetViewRect(CefRefPtr<CefBrowser> browser, CefRect& rect) override;
    void OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList& dirtyRects, const void* buffer, int width, int height) override;

    void setImageBuffer(ImageBuffer &buffer)
    {
        imageBuffer = &buffer;
    }
    void resize(int resolution, float aspect);

private:
    CefAppHandler *cef = nullptr;
    ImageBuffer *imageBuffer=nullptr;
    int width = 1024;
    int height = 1024;
    bool bufferChangedFlag = false;

    IMPLEMENT_REFCOUNTING(CEF_client);
};

class VRUI_client: public vrui::vruiCollabInterface, public vrui::coAction
{
public:
   
    VRUI_client(CEF *c);
    virtual ~VRUI_client();
    void update();
    void show();
    void hide();
    void resize(int resolution, float aspect);

    // hit is called whenever the button
    // with this action is intersected
    // return ACTION_CALL_ON_MISS if you want miss to be called
    // otherwise return ACTION_DONE
    int hit(vrui::vruiHit* hit) override;

    // miss is called once after a hit, if the button is not intersected
    // anymore
    void miss() override;
    ImageBuffer &getImageBuffer()
    {
        return imageBuffer;
    }

    CefAppHandler *cef = nullptr;
private:
    
    int width = 1024;
    int height = 1024;
    CEF *cefPlugin = nullptr;
    bool unregister = false;
    bool haveFocus = false;

    vrui::coCombinedButtonInteraction* interactionA;     ///< interaction for first button
    vrui::coCombinedButtonInteraction* interactionB;     ///< interaction for second button
    vrui::coCombinedButtonInteraction* interactionC;     ///< interaction for third button
    vrui::coCombinedButtonInteraction* interactionWheel; ///< interaction for wheel

    ImageBuffer imageBuffer;
    coPopupHandle *popupHandle = nullptr;
    coTexturedBackground* videoTexture = nullptr;
};

class CefAppHandler : public CefApp, public CefBrowserProcessHandler
{
public:
    CefAppHandler(ImageBuffer& imageBuffer);
    void init(const std::string & frameworkDir, const std::string &browserSubprocessPath, const std::string &logfile, int loglevel);
    void queueInputEvent(const InputEvent &event);
    void open(const std::string &url);
    void loop();
    void terminate()
    {
        m_terminate = true;
    }   
    private:
    CefRefPtr<CefBrowser> browser = nullptr; //only on master
    CEF* cefPlugin;
    CefRefPtr<CefBrowserProcessHandler> GetBrowserProcessHandler() override
    {
        return this;
    }
    void OnContextInitialized() override;
    CefRefPtr<CefClient> GetDefaultClient() override;
    CefRefPtr<CEF_client> cef_client = nullptr;
    ImageBuffer &m_imageBuffer;
    bool m_initFailed = false;
    std::mutex m_eventMutex;
    std::deque<InputEvent> m_pendingInputs;
    std::atomic<bool> m_terminate = false;
    IMPLEMENT_REFCOUNTING(CefAppHandler);

};

class CEF : public coVRPlugin, public coMenuListener, public ui::Owner
{

    private:

        ui::Menu* menu = nullptr;
        ui::Action* backButton = nullptr;
        ui::Action* forwardButton = nullptr;
        ui::Action* reloadButton = nullptr;
        ui::EditField* commandLine = nullptr;
        ui::EditField* urlLine = nullptr;

        std::string url;
        int resolution = 1024;
        float aspect = 1;
        bool focus = false;
        bool ctrlUsed = false;
        int mX = -1;
        int mY = -1;

        // browser thread
        std::unique_ptr<std::thread> m_browserThread;

        std::mutex m_cefInitMutex;
        std::condition_variable m_cefInitCv;
        
        
        bool update() override;
        void message(int toWhom, int type, int length, const void *data) override;
        
    public:
        CEF();
        bool init() override;
        virtual ~CEF();
        
        std::unique_ptr<VRUI_client> vrui_client;
        
        // void setResolution(float a);
        // void setAspectRatio(float a);
        
        
        void reload();
        const std::string &getURL();
        // void resize();
        virtual void key(int type, int keySym, int mod) override;
        std::unique_ptr<CefAppHandler> m_cefAppHandler;


};


#endif
