#include "CEF.h"
#include "CEFWindowsKey.h"

#include <OpenSG/OSGTextureEnvChunk.h>
#include <OpenSG/OSGTextureObjChunk.h>
#include <OpenSG/OSGGeometry.h>
#include <OpenSG/OSGGeoProperties.h>
#include <OpenSG/OSGImage.h>

#include "core/scene/VRSceneManager.h"
#include "core/scene/VRScene.h"
#include "core/setup/devices/VRDevice.h"
#include "core/setup/devices/VRKeyboard.h"
#include "core/objects/material/VRMaterial.h"
#include "core/objects/material/VRTexture.h"
#include "core/objects/material/VRTextureGenerator.h"
#include "core/utils/VRLogger.h"

using namespace OSG;

template<> string typeName(const CEF& o) { return "CEF"; }

vector< weak_ptr<CEF> > instances;
bool cef_gl_init = false;

CEF_handler::CEF_handler() {
    VRTextureGenerator g;
    g.setSize(Vec3i(2,2,1), false);
    g.drawFill(Color4f(0,0,0,1));
    image = g.compose();
}

CEF_handler::~CEF_handler() {
    cout << "~CEF_handler\n";
}

#ifdef _WIN32
void CEF_handler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect& rect) {
    rect = CefRect(0, 0, max(8,width), max(8,height)); // never give an empty rectangle!!
}

//Disable context menu
//Define below two functions to essentially do nothing, overwriting defaults
void CEF_handler::OnBeforeContextMenu( CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, CefRefPtr<CefContextMenuParams> params, CefRefPtr<CefMenuModel> model) {
    //CEF_REQUIRE_UI_THREAD();
    model->Clear();
}

bool CEF_handler::OnContextMenuCommand( CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, CefRefPtr<CefContextMenuParams> params, int command_id, EventFlags event_flags) {
    //CEF_REQUIRE_UI_THREAD();
    //MessageBox(browser->GetHost()->GetWindowHandle(), L"The requested action is not supported", L"Unsupported Action", MB_OK | MB_ICONINFORMATION);
    return false;
}
#else
bool CEF_handler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect& rect) {
    rect = CefRect(0, 0, max(8, width), max(8, height)); // never give an empty rectangle!!
    return true;
}
#endif

void CEF_handler::OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList& dirtyRects, const void* buffer, int width, int height) {
    if (!image) return;
    auto img = image->getImage();
    if (img) {
        img->set(Image::OSG_BGRA_PF, width, height, 1, 0, 1, 0.0, (const uint8_t*)buffer, Image::OSG_UINT8_IMAGEDATA, true, 1);
    }
}

OSG::VRTexturePtr CEF_handler::getImage() { return image; }

void CEF_handler::resize(int resolution, float aspect) {
    width = resolution;
    height = width/aspect;
}

CEF_client::CEF_client() {
    handler = new CEF_handler();
}

CEF_client::~CEF_client() {
    cout << "~CEF_client\n";
}

CefRefPtr<CefRenderHandler> CEF_client::GetRenderHandler() { return handler; }
CefRefPtr<CEF_handler> CEF_client::getHandler() { return handler; }
CefRefPtr<CefContextMenuHandler> CEF_client::GetContextMenuHandler() { return handler; }

CEF::CEF() {
    global_initiate();
    client = new CEF_client();
    update_callback = VRUpdateCb::create("webkit_update", bind(&CEF::update, this));
    auto scene = VRScene::getCurrent();
    if (scene) scene->addUpdateFkt(update_callback);
}

CEF::~CEF() {
    cout << "CEF destroyed " << client->HasOneRef() << " " << browser->HasOneRef() << endl;
    browser->GetHost()->CloseBrowser(false);
}

void CEF::shutdown() { if (!cef_gl_init) return; cout << "CEF shutdown\n"; CefShutdown(); }

CEFPtr CEF::create() {
    auto cef = CEFPtr(new CEF());
    instances.push_back(cef);
    return cef;
}

void CEF::global_initiate() {
    static bool global_init = false;
    if (global_init) return;
    global_init = true;

    cout << "Global CEF init\n";
    cef_gl_init = true;
    CefSettings settings;

#ifdef _WIN32
    string path = "/ressources/cefWin";
#elif defined(CEF18)
    string path = "/ressources/cef18";
#else
    string path = "/ressources/cef";
#endif

#ifdef _WIN32
    string bsp = VRSceneManager::get()->getOriginalWorkdir() + path + "/CefSubProcessWin.exe";
#else
    string bsp = VRSceneManager::get()->getOriginalWorkdir() + path + "/CefSubProcess";
#endif

    string ldp = VRSceneManager::get()->getOriginalWorkdir() + path + "/locales";
    string rdp = VRSceneManager::get()->getOriginalWorkdir() + path;
    string lfp = VRSceneManager::get()->getOriginalWorkdir() + path + "/wblog.log";
    CefString(&settings.browser_subprocess_path).FromASCII(bsp.c_str());
    CefString(&settings.locales_dir_path).FromASCII(ldp.c_str());
    CefString(&settings.resources_dir_path).FromASCII(rdp.c_str());
    CefString(&settings.log_file).FromASCII(lfp.c_str());
    settings.no_sandbox = true;
#ifdef _WIN32
    settings.windowless_rendering_enabled = true;
    //settings.log_severity = LOGSEVERITY_VERBOSE;
#endif

    CefMainArgs args;
    CefInitialize(args, settings, 0, 0);
}

void CEF::initiate() {
    init = true;
    CefWindowInfo win;
    CefBrowserSettings browser_settings;
#ifdef _WIN32
    win.SetAsWindowless(0);
    win.shared_texture_enabled = false;
#elif defined(CEF18)
    win.SetAsWindowless(0);
#else
    win.SetAsWindowless(0, true);
#endif

#ifdef _WIN32
    //requestContext = CefRequestContext::CreateContext(handler.get());
    browser = CefBrowserHost::CreateBrowserSync(win, client, "www.google.de", browser_settings, 0, 0);
    browser->GetHost()->WasResized();
#else
    browser = CefBrowserHost::CreateBrowserSync(win, client, "www.google.de", browser_settings, 0);
#endif
}

void CEF::setMaterial(VRMaterialPtr mat) {
    if (!mat) return;
    if (!client->getHandler()) return;
    this->mat = mat;
    mat->setTexture(client->getHandler()->getImage());
}

string CEF::getSite() { return site; }
void CEF::reload() { if (browser) browser->Reload(); if (auto m = mat.lock()) m->updateDeferredShader(); }

void CEF::update() {
    if (!init || !client->getHandler()) return;
    auto img = client->getHandler()->getImage();
    int dim1= img->getImage()->getDimension();
    CefDoMessageLoopWork();
    int dim2= img->getImage()->getDimension();
    if (dim1 != dim2)
        if (auto m = mat.lock()) m->updateDeferredShader();
}

void CEF::open(string site) {
    if (!init) initiate();
    this->site = site;
    if (browser) {
        browser->GetMainFrame()->LoadURL(site);
#ifdef _WIN32
        browser->GetHost()->WasResized();
#endif
    }
}

void CEF::resize() {
    if (!client->getHandler()) return;
    client->getHandler()->resize(resolution, aspect);
    if (init && browser) browser->GetHost()->WasResized();
    if (init) reload();
}

vector<CEFPtr> CEF::getInstances() {
    vector<CEFPtr> res;
    for (auto i : instances) {
        auto cef = i.lock();
        if (!cef) continue;
        res.push_back(cef);
    }
    return res;
}

void CEF::reloadScripts(string path) {
    for (auto i : instances) {
        auto cef = i.lock();
        if (!cef) continue;
        string s = cef->getSite();
        stringstream ss(s); vector<string> res; while (getline(ss, s, '/')) res.push_back(s); // split by ':'
        if (res.size() == 0) continue;
        if (res[res.size()-1] == path) {
            cef->resize();
            cef->reload();
        }
    }
}

void CEF::setResolution(float a) { resolution = a; resize(); }
void CEF::setAspectRatio(float a) { aspect = a; resize(); }

// dev callbacks:

void CEF::addMouse(VRDevicePtr dev, VRObjectPtr obj, int lb, int rb, int wu, int wd) {
    if (!obj) obj = this->obj.lock();
    if (dev == 0 || obj == 0) return;
    this->obj = obj;

    auto k = dev.get();
    if (!mouse_dev_callback.count(k)) mouse_dev_callback[k] = VRFunction<VRDeviceWeakPtr>::create( "CEF::MOUSE", bind(&CEF::mouse, this, lb,rb,wu,wd,_1 ) );
    dev->newSignal(-1,0)->add(mouse_dev_callback[k]);
    dev->newSignal(-1,1)->add(mouse_dev_callback[k]);

    if (!mouse_move_callback.count(k)) mouse_move_callback[k] = VRUpdateCb::create( "CEF::MM", bind(&CEF::mouse_move, this, dev) );
    auto scene = VRScene::getCurrent();
    if (scene) scene->addUpdateFkt(mouse_move_callback[k]);
}

void CEF::addKeyboard(VRDevicePtr dev) {
    if (dev == 0) return;
    if (!keyboard_dev_callback) keyboard_dev_callback = VRFunction<VRDeviceWeakPtr>::create( "CEF::KR", bind(&CEF::keyboard, this, _1 ) );
    dev->newSignal(-1, 0)->add( keyboard_dev_callback );
    dev->newSignal(-1, 1)->add( keyboard_dev_callback );
}

void CEF::mouse_move(VRDeviceWeakPtr d) {
    if (!focus) return;
    auto dev = d.lock();
    if (!dev) return;
    auto geo = obj.lock();
    if (!geo) return;
    VRIntersection ins = dev->intersect(geo);

    if (!ins.hit) return;
    if (ins.object.lock() != geo) return;

    CefMouseEvent me;
    me.x = ins.texel[0]*resolution;
    me.y = ins.texel[1]*(resolution/aspect);
    if (!browser) return;
    auto host = browser->GetHost();
    if (!host) return;
    if (me.x != mX || me.y != mY) {
        host->SendMouseMoveEvent(me, false);
        mX = me.x;
        mY = me.y;
    }
}

void CEF::mouse(int lb, int rb, int wu, int wd, VRDeviceWeakPtr d) {
    auto dev = d.lock();
    if (!dev) return;
    int b = dev->key();
    bool down = dev->getState();

    if (b == lb) b = 0;
    else if (b == rb) b = 2;
    else if (b == wu) b = 3;
    else if (b == wd) b = 4;
    else return;

    auto geo = obj.lock();
    if (!geo) return;

    VRIntersection ins = dev->intersect(geo);
    auto iobj = ins.object.lock();

    if (VRLog::tag("net")) {
        string o = "NONE";
        if (iobj) o = iobj->getName();
        stringstream ss;
        ss << "CEF::mouse " << this << " dev " << dev->getName();
        ss << " hit " << ins.hit << " " << o << ", trg " << geo->getName();
        ss << " b: " << b << " s: " << down;
        ss << " texel: " << ins.texel;
        ss << endl;
        VRLog::log("net", ss.str());
    }

    if (!browser) return;
    auto host = browser->GetHost();
    if (!host) return;
    if (!ins.hit) { host->SendFocusEvent(false); focus = false; return; }
    if (iobj != geo) { host->SendFocusEvent(false); focus = false; return; }
    host->SendFocusEvent(true); focus = true;

    int width = resolution;
    int height = resolution/aspect;

    CefMouseEvent me;
    me.x = ins.texel[0]*width;
    me.y = ins.texel[1]*height;

    if (b < 3) {
        cef_mouse_button_type_t mbt;
        if (b == 0) mbt = MBT_LEFT;
        if (b == 1) mbt = MBT_MIDDLE;
        if (b == 2) mbt = MBT_RIGHT;
        //cout << "CEF::mouse " << me.x << " " << me.y << " " << !down << endl;
        host->SendMouseClickEvent(me, mbt, !down, 1);
    }

    if (b == 3 || b == 4) {
        int d = b==3 ? -1 : 1;
        host->SendMouseWheelEvent(me, d*width*0.05, d*height*0.05);
    }
}

void CEF::keyboard(VRDeviceWeakPtr d) {
    auto dev = d.lock();
    if (!dev) return;
    if (!focus) return;
    if (dev->getType() != "keyboard") return;
    //bool down = dev->getState();
    VRKeyboardPtr keyboard = dynamic_pointer_cast<VRKeyboard>(dev);
    if (!keyboard) return;
    auto event = keyboard->getGtkEvent();
    if (!browser) return;
    auto host = browser->GetHost();
    if (!host) return;

    //cout << "CEF::keyboard " << event->keyval << " " << ctrlUsed << " " << keyboard->ctrlDown() << endl;

    if (keyboard->ctrlDown() && event->type == GDK_KEY_PRESS) {
        if (event->keyval == 'a') { browser->GetFocusedFrame()->SelectAll(); ctrlUsed = true; }
        if (event->keyval == 'c') { browser->GetFocusedFrame()->Copy(); ctrlUsed = true; }
        if (event->keyval == 'v') { browser->GetFocusedFrame()->Paste(); ctrlUsed = true; }
        return;
    }

    if (!keyboard->ctrlDown() && event->type == GDK_KEY_PRESS) ctrlUsed = false;

    if (ctrlUsed && !keyboard->ctrlDown() && event->type == GDK_KEY_RELEASE && event->keyval != GDK_KEY_Control_L && event->keyval != GDK_KEY_Control_R ) {
        ctrlUsed = false;
        return; // ignore next key up event when ctrl was used for a shortcut above!
    }

    CefKeyEvent kev;
    kev.modifiers = GetCefStateModifiers(event->state);
    if (event->keyval >= GDK_KEY_KP_Space && event->keyval <= GDK_KEY_KP_9) kev.modifiers |= EVENTFLAG_IS_KEY_PAD;
    if (kev.modifiers & EVENTFLAG_ALT_DOWN) kev.is_system_key = true;

    KeyboardCode windows_key_code = GdkEventToWindowsKeyCode(event);
    kev.windows_key_code = GetWindowsKeyCodeWithoutLocation(windows_key_code);

    kev.native_key_code = event->keyval;

    if (windows_key_code == VKEY_RETURN) kev.unmodified_character = '\r'; else
    kev.unmodified_character = static_cast<int>(gdk_keyval_to_unicode(event->keyval));

    if (kev.modifiers & EVENTFLAG_CONTROL_DOWN) kev.character = GetControlCharacter(windows_key_code, kev.modifiers & EVENTFLAG_SHIFT_DOWN); 
    else kev.character = kev.unmodified_character;

    if (event->type == GDK_KEY_PRESS) {
        //cout << " CEF::keyboard press " << event->keyval << " " << kev.native_key_code << " " << kev.character << endl;
        kev.type = KEYEVENT_RAWKEYDOWN; host->SendKeyEvent(kev);
    } else {
        //cout << " CEF::keyboard release " << event->keyval << " " << kev.native_key_code << " " << kev.character << endl;
        kev.type = KEYEVENT_KEYUP; host->SendKeyEvent(kev);
        kev.type = KEYEVENT_CHAR; host->SendKeyEvent(kev);
    }
}
