#ifndef CEF_H_INCLUDED
#define CEF_H_INCLUDED

#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/sginterface/vruiCollabInterface.h>
#include <cover/ui/Owner.h>

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
class CEF;

class CEF_client : public CefClient, public CefRenderHandler, public CefLifeSpanHandler, public CefContextMenuHandler
{

public:
    CEF_client(CEF *c);
    

    CefRefPtr<CefRenderHandler> GetRenderHandler() override;
    CefRefPtr<CefLifeSpanHandler> GetLifeSpanHandler() override;
    CefRefPtr<CefContextMenuHandler> GetContextMenuHandler() override;

    virtual bool DoClose(CefRefPtr<CefBrowser> browser) override;

#ifdef _WIN32
    void OnBeforeContextMenu(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, CefRefPtr<CefContextMenuParams> params, CefRefPtr<CefMenuModel> model) override;
    bool OnContextMenuCommand(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, CefRefPtr<CefContextMenuParams> params, int command_id, EventFlags event_flags) override;
#endif
    void GetViewRect(CefRefPtr<CefBrowser> browser, CefRect& rect) override;
    void OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList& dirtyRects, const void* buffer, int width, int height) override;

    void setImageBuffer(unsigned char* buffer)
    {
        imageBuffer = buffer;
    }
    void resize(int resolution, float aspect);
    bool bufferChanged() const
    {
        return bufferChangedFlag;
    }
    void setBufferChanged(bool changed)
    {
        bufferChangedFlag = changed;
    }
private:
    CEF *cef = nullptr;
    unsigned char*imageBuffer=nullptr;
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
    unsigned char* getImageBuffer()
    {
        return imageBuffer;
    }
    void setBufferChanged(bool changed)
    {
        bufferChanged = changed;
    }

private:
    
    int width = 1024;
    int height = 1024;
    CEF *cef = nullptr;

    bool unregister = false;
    bool haveFocus = false;

    vrui::coCombinedButtonInteraction* interactionA;     ///< interaction for first button
    vrui::coCombinedButtonInteraction* interactionB;     ///< interaction for second button
    vrui::coCombinedButtonInteraction* interactionC;     ///< interaction for third button
    vrui::coCombinedButtonInteraction* interactionWheel; ///< interaction for wheel

    unsigned char*imageBuffer=nullptr;
    coPopupHandle *popupHandle = nullptr;
    coTexturedBackground* videoTexture = nullptr;
    bool bufferChanged = false;
};

class CEF : public coVRPlugin, public coMenuListener, public CefApp, public CefBrowserProcessHandler, public ui::Owner
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
        bool m_initFailed = false;
        int mX = -1;
        int mY = -1;

        bool update() override;

        // CefApp methods:
        CefRefPtr<CefBrowserProcessHandler> GetBrowserProcessHandler() override {
            return this;
        }

        // CefBrowserProcessHandler methods:
        void OnContextInitialized() override;
        CefRefPtr<CefClient> GetDefaultClient() override;
        void message(int toWhom, int type, int length, const void *data) override;

    public:
        CEF();
        bool init() override;
        virtual ~CEF();
        CefRefPtr<CefBrowser> browser = nullptr; //only on master
        CefRefPtr<CEF_client> cef_client = nullptr;
        std::unique_ptr<VRUI_client> vrui_client;

        void setResolution(float a);
        void setAspectRatio(float a);


        void open(const std::string &url);
        void reload();
        const std::string &getURL();
        void resize();
        virtual void key(int type, int keySym, int mod) override;

        IMPLEMENT_REFCOUNTING(CEF);

};


#endif
