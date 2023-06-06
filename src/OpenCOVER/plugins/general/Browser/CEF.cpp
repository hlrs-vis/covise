#include "CEF.h"
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/coVRCollaboration.h>
#include <cover/coCollabInterface.h>
#include <cover/coIntersection.h>
#include <util/unixcompat.h>
#include "include/cef_browser.h"
#include "include/cef_request_context.h"
#include "include/cef_command_line.h"
#include "include/views/cef_browser_view.h"
#include "include/views/cef_window.h"
#include "include/wrapper/cef_helpers.h"
#include "tests/cefsimple/simple_handler.h"
#include "CEFWindowsKey.h"
#include <config/CoviseConfig.h>
#include <boost/smart_ptr/scoped_ptr.hpp>

#include <OpenVRUI/coToolboxMenu.h>
#include <OpenVRUI/coRowMenu.h>

#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coTexturedBackground.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/osg/OSGVruiHit.h>

#include <cover/ui/Manager.h>
#include <cover/ui/Menu.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Label.h>
#include <cover/ui/Action.h>
#include <PluginUtil/PluginMessageTypes.h>
#include <algorithm>


boost::scoped_ptr<coCOIM> CEFCoim; // keep before other items (to be destroyed last)


void CEF::OnContextInitialized()
{
    //CEF_REQUIRE_UI_THREAD();


    client = new CEF_client(this);

    CefWindowInfo win;
    CefBrowserSettings browser_settings;
    win.SetAsWindowless(0);
#ifdef _WIN32
    win.shared_texture_enabled = false;
#endif


    browser = CefBrowserHost::CreateBrowserSync(win, client, "www.google.de", browser_settings, nullptr, nullptr);
    browser->GetHost()->WasResized();
    browser->GetHost()->WasHidden(false);

    // Force a call to OnPaint.
    browser->GetHost()->Invalidate(PET_VIEW);
}

CefRefPtr<CefClient> CEF::GetDefaultClient()
{
    // Called when a new browser window is created via the Chrome runtime UI.
    return client;
}

void CEF::message(int toWhom, int type, int length, const void *data)
{
    if (type == opencover::PluginMessageTypes::Browser)
    {
        std::string url(static_cast<const char *>(data), length);
        open(url.c_str());
    }
}


bool CEF_client::DoClose(CefRefPtr<CefBrowser> browser)
{
    if (cef->browser == nullptr)
    {
        // done already
        return false;
    }
    if (browser->IsSame(cef->browser))
    {
        LOG(INFO) << "CEF::DoClose: Closing the browser";
        cef->browser = nullptr;
        //HWND hwnd = getHwnd();
        //::DestroyWindow(hwnd);
        // we have to return false, otherwise this browser will not be removed before destruction
        return false;
        // true=we've handled the event ourselves; do not send WM_CLOSE
    }
    else
    {
        LOG(INFO) << "CEF::DoClose: Closing a sub-browser (may be dev tools)";
        return false; // false=close the window; WM_CLOSE will bubble up to the parent window
    }
}

void CEF_client::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect)
{
    rect = CefRect(0, 0, std::max(8, width), std::max(8, height)); // never give an empty rectangle!!
}

#ifdef _WIN32
//Disable context menu
//Define below two functions to essentially do nothing, overwriting defaults
void CEF_client::OnBeforeContextMenu(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
                                     CefRefPtr<CefContextMenuParams> params, CefRefPtr<CefMenuModel> model)
{
    //CEF_REQUIRE_UI_THREAD();
    model->Clear();
}

bool CEF_client::OnContextMenuCommand(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
                                      CefRefPtr<CefContextMenuParams> params, int command_id, EventFlags event_flags)
{
    //CEF_REQUIRE_UI_THREAD();
    //MessageBox(browser->GetHost()->GetWindowHandle(), L"The requested action is not supported", L"Unsupported Action", MB_OK | MB_ICONINFORMATION);
    return false;
}
#endif

void CEF_client::OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList &dirtyRects,
                         const void *buffer, int width, int height)
{
    /* if (!image) return;
    auto img = image->getImage();
    if (img) {
        img->set(Image::OSG_BGRA_PF, width, height, 1, 0, 1, 0.0, (const uint8_t*)buffer, Image::OSG_UINT8_IMAGEDATA, true, 1);
    }*/
    memcpy(imageBuffer, buffer, (size_t)width * height * 4);
    bufferChanged = true;
    //std::cerr << "Render" << std::endl;
}

void CEF_client::resize(int resolution, float aspect)
{
    width = resolution;
    height = width / aspect;
}


CEF_client::CEF_client(CEF *c): vruiCollabInterface(CEFCoim.get(), "CEFBrowser", vruiCollabInterface::PinEditor)
{
    cef = c;


    interactionA = new coCombinedButtonInteraction(coInteraction::ButtonA, "CEFBrowser", coInteraction::Menu);
    interactionB = new coCombinedButtonInteraction(coInteraction::ButtonB, "CEFBrowser", coInteraction::Menu);
    interactionC = new coCombinedButtonInteraction(coInteraction::ButtonC, "CEFBrowser", coInteraction::Menu);

    imageBuffer = new unsigned char[(size_t)width * height * 4];
    videoTexture = new vrui::coTexturedBackground((uint *)imageBuffer, NULL, NULL, 4, width, height, 0);

    coIntersection::getIntersectorForAction("coAction")->add(videoTexture->getDCS(), this);

    fprintf(stderr, "Set size!\n");
    videoTexture->setSize(width, height, 0);

    fprintf(stderr, "Set Textsize!\n");
    videoTexture->setTexSize(width, -height);

    fprintf(stderr, "Min width!\n");
    videoTexture->setMinWidth(width);

    fprintf(stderr, "Minheight!\n");
    videoTexture->setMinHeight(height);
    fprintf(stderr, "Texture creation  finished!\n");

    popupHandle = new coPopupHandle("BrowserHeadline");
    popupHandle->setScale(2 * cover->getSceneSize() / 2500);
    popupHandle->setPos(-width * cover->getSceneSize() / 2500, 0, -height * cover->getSceneSize() / 2500);
    popupHandle->addElement(videoTexture);
    show();
}

CEF_client::~CEF_client()
{
    delete interactionA;
    delete interactionB;
    delete interactionC;
    delete popupHandle;
}

void CEF_client::update()
{
    if (unregister)
    {
        if (interactionA->isRegistered() && (interactionA->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionA);
        }
        if (interactionB->isRegistered() && (interactionB->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionB);
        }
        if (interactionC->isRegistered() && (interactionC->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionC);
        }
        if ((!interactionA->isRegistered()) && (!interactionB->isRegistered()) && (!interactionC->isRegistered()))
        {
            unregister = false;
        }
    }
    if (videoTexture && popupHandle)
    {
        char c = 1;
        popupHandle->update();
        if (coVRMSController::instance()->isCluster())
        {
            if (coVRMSController::instance()->isMaster())
            {
                if (bufferChanged)
                {
                    coVRMSController::instance()->sendSlaves(&c, 1);
                    coVRMSController::instance()->sendSlaves((char *)imageBuffer, width * height * 4);
                }
                else
                {
                    c = 0;
                    coVRMSController::instance()->sendSlaves(&c, 1);
                }
            }
            else
            {
                coVRMSController::instance()->readMaster(&c, 1);
                if (c)
                {
                    bufferChanged = true;
                    videoTexture->setUpdated(true);
                    coVRMSController::instance()->readMaster((char *)imageBuffer, width * height * 4);
                    videoTexture->setImage((uint *)imageBuffer, NULL, NULL, 4, width, height, 0,
                                           coTexturedBackground::TextureSet::PF_BGRA);
                }
            }
        }
        if (bufferChanged)
        {
            bufferChanged = false;

            videoTexture->setUpdated(true);

            videoTexture->setImage((uint *)imageBuffer, NULL, NULL, 4, width, height, 0,
                                   coTexturedBackground::TextureSet::PF_BGRA);
        }
    }
}

void CEF_client::show()
{
    if (popupHandle)
        popupHandle->setVisible(true);
}

void CEF_client::hide()
{
    if (popupHandle)
        popupHandle->setVisible(false);
}

CefRefPtr<CefRenderHandler> CEF_client::GetRenderHandler()
{
    return this;
}
CefRefPtr<CefContextMenuHandler> CEF_client::GetContextMenuHandler()
{
    return this;
}

CEF::CEF()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("BrowserPlugin", cover->ui)
{
    
}

CEF::~CEF()
{
    delete backButton;
    delete forwardButton;
    delete reloadButton;
    delete urlLine;
    delete menu;
    if(!m_initFailed)
    {
        std::cout << "CEF destroyed " << client->HasOneRef() << " " << browser->HasOneRef() << std::endl;
        browser->GetHost()->CloseBrowser(true);

        for (int attempts = 0; browser != nullptr && attempts < 1000; ++attempts) // waiting for the Browser to close
        {
            usleep(100000);
            CefDoMessageLoopWork();
        }

        CefShutdown();
    }

}


int CEF_client::hit(vruiHit *hit)
{
    if (coVRCollaboration::instance()->getCouplingMode() == coVRCollaboration::MasterSlaveCoupling &&
        !coVRCollaboration::instance()->isMaster())
        return ACTION_DONE;

    if (!interactionA->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionA);
        interactionA->setHitByMouse(hit->isMouseHit());
    }
    if (!interactionB->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionB);
        interactionB->setHitByMouse(hit->isMouseHit());
    }
    if (!interactionC->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionC);
        interactionC->setHitByMouse(hit->isMouseHit());
    }

    osgUtil::LineSegmentIntersector::Intersection osgHit = dynamic_cast<OSGVruiHit *>(hit)->getHit();

    static char message[100];

    float x = 0.f, y = 0.f;
    if (osgHit.drawable.valid())
    {
        osg::Vec3 point = osgHit.getLocalIntersectPoint();
        x = (point[0]) / width;
        y = 1.0 - ((point[1]) / height);
        if (x > 1)
            x = 1;
        if (x < 0)
            x = 0;
        if (y > 1)
            y = 1;
        if (y < 0)
            y = 0;

        if (!haveFocus)
        {
            cover->grabKeyboard(cef);
            haveFocus = true;
        }
        CefMouseEvent me;
        me.x = x * width;
        me.y = y * height;
        if ((interactionA->getState() == coInteraction::Idle) && (interactionB->getState() == coInteraction::Idle) &&
            (interactionC->getState() == coInteraction::Idle))
        {
            cef->browser->GetHost()->SetFocus(true);
        }
        if (interactionA->wasStarted())
        {
            cef->browser->GetHost()->SendMouseClickEvent(me, CefBrowserHost::MouseButtonType::MBT_LEFT, false, 1);
            cerr << "ADown" << endl;
        }
        else if (interactionA->wasStopped())
        {
            cef->browser->GetHost()->SendMouseClickEvent(me, CefBrowserHost::MouseButtonType::MBT_LEFT, true, 1);
            cerr << "AUp" << endl;
        }
        else if (interactionB->wasStarted())
        {
            cef->browser->GetHost()->SendMouseClickEvent(me, CefBrowserHost::MouseButtonType::MBT_MIDDLE, false, 1);
        }
        else if (interactionB->wasStopped())
        {
            cef->browser->GetHost()->SendMouseClickEvent(me, CefBrowserHost::MouseButtonType::MBT_MIDDLE, true, 1);
        }
        else if (interactionC->wasStarted())
        {
            cef->browser->GetHost()->SendMouseClickEvent(me, CefBrowserHost::MouseButtonType::MBT_RIGHT, false, 1);
        }
        else if (interactionC->wasStopped())
        {
            cef->browser->GetHost()->SendMouseClickEvent(me, CefBrowserHost::MouseButtonType::MBT_RIGHT, true, 1);
        }
        else
        {
            cef->browser->GetHost()->SendMouseMoveEvent(me, false);
        }
    }


    if (interactionA->wasStarted() || interactionB->wasStarted() || interactionC->wasStarted())
    {
    }
    return ACTION_CALL_ON_MISS;
}

void CEF_client::miss()
{
    cef->browser->GetHost()->SetFocus(false);
    unregister = true;

    if (haveFocus)
    {
        cover->releaseKeyboard(cef);
        haveFocus = false;
    }
}
const std::string &CEF::getURL()
{
    return url;
}
void CEF::reload()
{
    if (browser)
        browser->Reload();
}

bool CEF::init()
{
    AddRef(); // count our own reference as well.

    CEFCoim.reset(new coCOIM(this));

    browser = nullptr;
    menu = new ui::Menu("Browser", this);
    menu->setVisible(false);
    menu->setVisible(true, ui::View::Tablet);


    backButton = new ui::Action(menu, "back");
    backButton->setText("back");
    backButton->setCallback([this]() { browser->GoBack(); });
    forwardButton = new ui::Action(menu, "forward");
    forwardButton->setText("forward");
    forwardButton->setCallback([this]() { browser->GoForward(); });
    reloadButton = new ui::Action(menu, "reload");
    reloadButton->setText("reload");
    reloadButton->setCallback([this]() { browser->Reload(); });
    urlLine = new ui::EditField(menu, "urlLine");
    urlLine->setText("URL");
    urlLine->setCallback(
        [this](const std::string &cmd)
        {
            if (cmd.length() > 0)
            {
                open(cmd);
            }
        });

    commandLine = new ui::EditField(menu, "CommandLine");
    commandLine->setText("Command line");
    commandLine->setCallback(
        [this](const std::string &cmd)
        {
            if (cmd.length() > 0)
            {
                for (int i = 0; i < cmd.length(); i++)
                {
                    CefKeyEvent keyEvent;
                    keyEvent.character = cmd[i];
                    keyEvent.unmodified_character = keyEvent.character;
                    keyEvent.native_key_code = cmd[i];
                    keyEvent.windows_key_code = cmd[i];
                    keyEvent.focus_on_editable_field = true;
                    keyEvent.is_system_key = false;
                    keyEvent.modifiers = 0;

                    keyEvent.type = KEYEVENT_RAWKEYDOWN;
                    browser->GetHost()->SendKeyEvent(keyEvent);
                    keyEvent.type = KEYEVENT_KEYUP;
                    browser->GetHost()->SendKeyEvent(keyEvent);
                    keyEvent.type = KEYEVENT_CHAR;
                    browser->GetHost()->SendKeyEvent(keyEvent);
                }

                CefKeyEvent keyEvent;
                keyEvent.character = '\r';
                keyEvent.unmodified_character = '\r';
                keyEvent.native_key_code = osgGA::GUIEventAdapter::KEY_Return;
                keyEvent.windows_key_code = osgGA::GUIEventAdapter::KEY_Return;
                keyEvent.focus_on_editable_field = true;
                keyEvent.is_system_key = false;
                keyEvent.modifiers = 0;
                keyEvent.type = KEYEVENT_RAWKEYDOWN;
                browser->GetHost()->SendKeyEvent(keyEvent);
                keyEvent.type = KEYEVENT_KEYUP;
                browser->GetHost()->SendKeyEvent(keyEvent);
                keyEvent.type = KEYEVENT_CHAR;
                browser->GetHost()->SendKeyEvent(keyEvent);
                commandLine->setText("");
            }
        });

    CefSettings settings;
    CefSettingsTraits::init(&settings);

    char *cd;
    char *as;
    std::string coviseDir;
    std::string archSuffix;
    if ((cd = getenv("COVISEDIR")) == NULL)
    {
        cerr << "COVISEDIR variable not set !!" << endl;
        coviseDir = "c:\\Program Files\\Covise";
    }
    else
        coviseDir = cd;
    if ((as = getenv("ARCHSUFFIX")) == NULL)
    {
        cerr << "ARCHSUFFIX variable not set !!" << endl;
        archSuffix = "zebuopt";
    }
    else
        archSuffix = as;

    std::string bsp = coviseDir + "/" + archSuffix + "/bin/CEFBrowserHelper";
#ifdef _WIN32
    bsp += ".exe";
#endif
    CefString(&settings.browser_subprocess_path).FromASCII(bsp.c_str());

    std::string lfp = "/tmp/cef.log";
    CefString(&settings.log_file)
        .FromASCII(covise::coCoviseConfig::getEntry("logFile", "COVER.Plugin.Browser", lfp).c_str());
#ifdef __APPLE__
    std::string extlib;
    if (auto el = getenv("EXTERNLIBS"))
    {
        extlib = el;
    }
    else
    {
        cerr << "EXTERNLIBS variable not set !!" << endl;
        extlib = coviseDir + "/extern_libs/" + archSuffix;
    }
    std::string fwpath = extlib + "/cef/Release/Chromium Embedded Framework.framework";
    CefString(&settings.framework_dir_path) =
        covise::coCoviseConfig::getEntry("frameworkDirPath", "COVER.Plugin.Browser", fwpath);
#endif
    settings.log_severity = (cef_log_severity_t)covise::coCoviseConfig::getInt("logLevel", "COVER.Plugin.Browser", 99);
    settings.no_sandbox = true;
    settings.windowless_rendering_enabled = true;
    settings.external_message_pump = true;
#ifndef __APPLE__
    settings.multi_threaded_message_loop = false;
#endif
#ifdef _WIN32
    CefMainArgs args;
#else
    std::vector<const char *> cmdArgs;
    cmdArgs.push_back("--enable-media-stream=1");
    cmdArgs.push_back("--use-fake-ui-for-media-stream=1");
    CefMainArgs args(cmdArgs.size(), (char**)cmdArgs.data());
#endif
    if (!CefInitialize(args, settings, this, nullptr))
    {
        std::cerr << "CefInitialize failed" << std::endl;
        m_initFailed = true;
        return false;
    }
    auto path = getenv("COVISE_BROWSER_INIT_URL");
    if(path)
        open(path);
    return true;
}

bool CEF::update()
{
    CefDoMessageLoopWork();
    if (client)
    {
        client->update();
    }


    return true;
}

void CEF::open(const std::string &url)
{
    this->url = url;
    if (browser)
    {
        browser->GetMainFrame()->LoadURL(url);
#ifdef _WIN32
        browser->GetHost()->WasResized();
#endif
    }
}

void CEF::resize()
{
    client->resize(resolution, aspect);
    if (browser)
        browser->GetHost()->WasResized();
    reload();
}

void CEF::key(int type, int keySym, int mod)
{
    CefKeyEvent keyEvent;
    keyEvent.character = keySym;
    keyEvent.unmodified_character = keySym;
    keyEvent.native_key_code = keySym;
    keyEvent.windows_key_code = keySym;
    keyEvent.focus_on_editable_field = true;
    keyEvent.is_system_key = false;
    keyEvent.modifiers = 0;

    if (mod & osgGA::GUIEventAdapter::MODKEY_CTRL && type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        if (keySym == 'a')
        {
            browser->GetFocusedFrame()->SelectAll();
        }
        if (keySym == 'c')
        {
            browser->GetFocusedFrame()->Copy();
        }
        if (keySym == 'v')
        {
            browser->GetFocusedFrame()->Paste();
        }
        return;
    }

    if (mod & osgGA::GUIEventAdapter::MODKEY_SHIFT)
        keyEvent.modifiers |= EVENTFLAG_SHIFT_DOWN;
    if (mod & osgGA::GUIEventAdapter::MODKEY_CAPS_LOCK)
        keyEvent.modifiers |= EVENTFLAG_CAPS_LOCK_ON;
    if (mod & osgGA::GUIEventAdapter::MODKEY_CTRL)
        keyEvent.modifiers |= EVENTFLAG_CONTROL_DOWN;
    if (mod & osgGA::GUIEventAdapter::MODKEY_ALT)
        keyEvent.modifiers |= EVENTFLAG_ALT_DOWN;
    if (mod & osgGA::GUIEventAdapter::MODKEY_META)
        keyEvent.modifiers |= EVENTFLAG_LEFT_MOUSE_BUTTON;
    if (mod & osgGA::GUIEventAdapter::MODKEY_HYPER)
        keyEvent.modifiers |= EVENTFLAG_MIDDLE_MOUSE_BUTTON;
    if (mod & osgGA::GUIEventAdapter::MODKEY_SUPER)
        keyEvent.modifiers |= EVENTFLAG_RIGHT_MOUSE_BUTTON;

    if (keySym >= osgGA::GUIEventAdapter::KEY_KP_Space && keySym <= osgGA::GUIEventAdapter::KEY_KP_9)
        keyEvent.modifiers |= EVENTFLAG_IS_KEY_PAD;
    if (keyEvent.modifiers & EVENTFLAG_ALT_DOWN)
        keyEvent.is_system_key = true;

    keyEvent.unmodified_character = keySym;
    KeyboardCode WindowsKeyCode = KeyboardCodeFromXKeysym(keySym);
    int ctrlChar = GetControlCharacter(WindowsKeyCode, mod & osgGA::GUIEventAdapter::MODKEY_SHIFT);
    if (WindowsKeyCode != VKEY_UNKNOWN)
    {
        if (ctrlChar != 0)
        {
            keyEvent.unmodified_character = ctrlChar;
        }
        keyEvent.windows_key_code = WindowsKeyCode;
        keyEvent.native_key_code = keySym;
    }


    if (keyEvent.modifiers & EVENTFLAG_SHIFT_DOWN)
    {
        if (keySym >= 'a' && keySym <= 'z')
            keyEvent.character = keySym + 'A' - 'a';
    }
    else
        keyEvent.character = keyEvent.unmodified_character;


    if (type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        keyEvent.type = KEYEVENT_RAWKEYDOWN;
        browser->GetHost()->SendKeyEvent(keyEvent);
    }
    else
    {
        keyEvent.type = KEYEVENT_KEYUP;
        browser->GetHost()->SendKeyEvent(keyEvent);
        keyEvent.type = KEYEVENT_CHAR;
        browser->GetHost()->SendKeyEvent(keyEvent);
    }
}


void CEF::setResolution(float a)
{
    resolution = a;
    resize();
}
void CEF::setAspectRatio(float a)
{
    aspect = a;
    resize();
}


bool init()
{
    return true;
}


COVERPLUGIN(CEF)
