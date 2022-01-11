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

class CEF_client : public CefClient, public CefRenderHandler, public CefContextMenuHandler, public vrui::vruiCollabInterface, public vrui::coAction
{
private:
    int width = 1024;
    int height = 1024;
    CEF* cef;
    vrui::coCombinedButtonInteraction* interactionA; ///< interaction for first button
    vrui::coCombinedButtonInteraction* interactionB; ///< interaction for second button
    vrui::coCombinedButtonInteraction* interactionC; ///< interaction for third button
    bool unregister = false;
    bool haveFocus = false;

public:
    CEF_client(CEF *c);
    ~CEF_client();

    CefRefPtr<CefRenderHandler> GetRenderHandler() override;
    CefRefPtr<CefContextMenuHandler> GetContextMenuHandler() override;

    // hit is called whenever the button
    // with this action is intersected
    // return ACTION_CALL_ON_MISS if you want miss to be called
    // otherwise return ACTION_DONE
    int hit(vrui::vruiHit* hit) override;

    // miss is called once after a hit, if the button is not intersected
    // anymore
    void miss() override;


    void update();
    void show();
    void hide();

#ifdef _WIN32
    void GetViewRect(CefRefPtr<CefBrowser> browser, CefRect& rect) override;
    void OnBeforeContextMenu(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, CefRefPtr<CefContextMenuParams> params, CefRefPtr<CefMenuModel> model) override;
    bool OnContextMenuCommand(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, CefRefPtr<CefContextMenuParams> params, int command_id, EventFlags event_flags) override;
#else
#ifdef __APPLE__
    void GetViewRect(CefRefPtr<CefBrowser> browser, CefRect& rect) override;
#else
    bool GetViewRect(CefRefPtr<CefBrowser> browser, CefRect& rect) override;
#endif
#endif
    void OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList& dirtyRects, const void* buffer, int width, int height) override;
    void resize(int resolution, float aspect);

private:

    unsigned char*imageBuffer=nullptr;
    coPopupHandle* popupHandle;
    coTexturedBackground* videoTexture;
    bool bufferChanged = false;

    IMPLEMENT_REFCOUNTING(CEF_client);
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


    public:
        CEF();
        bool init() override;
        virtual ~CEF();
        CefRefPtr<CefBrowser> browser;
        CefRefPtr<CEF_client> client;

        void setResolution(float a);
        void setAspectRatio(float a);


        void open(const std::string &url);
        void reload();
        const std::string &getURL();
        void resize();
        opencover::coCOIM* coim;
        virtual void key(int type, int keySym, int mod) override;

        IMPLEMENT_REFCOUNTING(CEF);

};


#endif
