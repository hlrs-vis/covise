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
#include <chrono>

boost::scoped_ptr<coCOIM> CEFCoim; // keep before other items (to be destroyed last)

CefAppHandler::CefAppHandler(ImageBuffer& imageBuffer)
    : m_imageBuffer(imageBuffer)
{
    AddRef(); // count our own reference as well.
}

void CefAppHandler::init(const std::string & frameworkDir, const std::string &browserSubprocessPath, const std::string &logfile, int loglevel)
{
    CefSettings settings;
    CefSettingsTraits::init(&settings);

    
    CefString(&settings.browser_subprocess_path).FromASCII(browserSubprocessPath.c_str());

    CefString(&settings.log_file)
        //.FromASCII(covise::coCoviseConfig::getEntry("logFile", "COVER.Plugin.Browser", lfp).c_str());
        .FromASCII(logfile.c_str());
#ifdef __APPLE__

    
    CefString(&settings.framework_dir_path) = CefString::FromASCII(frameworkDir.c_str());
#endif
    settings.log_severity = (cef_log_severity_t)loglevel;
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
        return;
    }
    auto path = getenv("COVISE_BROWSER_INIT_URL");
    if(path)
        open(path);
}

void CefAppHandler::queueInputEvent(const InputEvent &event)
{
    std::lock_guard<std::mutex> lock(m_eventMutex);
    m_pendingInputs.push_back(event);
}

void handleEvents(const InputEvent &event, CefRefPtr<CefBrowser> browser)
{
    CefMouseEvent me;
    me.x = event.x;
    me.y = event.y;
    auto host = browser->GetHost();
    switch(event.type)
    {
    case InputEvent::GoBack:
            browser->GoBack();
        break;
    case InputEvent::GoForward:
            browser->GoForward();
        break;
    case InputEvent::Reload:
            browser->Reload();
        break;
    case InputEvent::Text:
    {
        auto & cmd = event.text;
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
                host->SendKeyEvent(keyEvent);
                keyEvent.type = KEYEVENT_KEYUP;
                host->SendKeyEvent(keyEvent);
                keyEvent.type = KEYEVENT_CHAR;
                host->SendKeyEvent(keyEvent);
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
            host->SendKeyEvent(keyEvent);
            keyEvent.type = KEYEVENT_KEYUP;
            host->SendKeyEvent(keyEvent);
            keyEvent.type = KEYEVENT_CHAR;
            host->SendKeyEvent(keyEvent);
        }
    }
    break;
    case InputEvent::OpenURL:
    {
        auto url = event.text;
        if (url.length() > 0)
        {
            browser->GetMainFrame()->LoadURL(url);
#ifdef _WIN32
            host->WasResized();
#endif                    
        }
    }
    break;
    case InputEvent::SetFocus:
        host->SetFocus(event.on);
    break;
    case InputEvent::LeftClick:
        host->SendMouseClickEvent(me, CefBrowserHost::MouseButtonType::MBT_LEFT, !event.on, 1);
    break;
    case InputEvent::RightClick:
        host->SendMouseClickEvent(me, CefBrowserHost::MouseButtonType::MBT_RIGHT, !event.on, 1);
    break;
    case InputEvent::MiddleClick:
        host->SendMouseClickEvent(me, CefBrowserHost::MouseButtonType::MBT_MIDDLE, !event.on, 1);
    break;
    case InputEvent::MouseWheel:
        host->SendMouseWheelEvent(me, 0, event.wheelDelta);
    break;
    case InputEvent::MouseMove:
        host->SendMouseMoveEvent(me, false);
    break;
    case InputEvent::CloseBrowser:
    {
        if(event.browser->IsSame(browser))
            browser = nullptr;
    }
    break;
    case InputEvent::SelectAll:
        browser->GetFocusedFrame()->SelectAll();
        break;
    case InputEvent::Copy:
        browser->GetFocusedFrame()->Copy();
        break;
    case InputEvent::Paste:
        browser->GetFocusedFrame()->Paste();
        break;
    case InputEvent::KeyEvent:
        host->SendKeyEvent(event.keyEvent);
        break;
    default:
        break;
    }
}

void CefAppHandler::loop()
{
    while(!m_terminate)
    {
        bool workToDo = false;
        {
            std::lock_guard<std::mutex> lock(m_eventMutex);
            workToDo = m_pendingInputs.size() > 0;
            for(const auto & event : m_pendingInputs)
            {
                handleEvents(event, browser);
            }
            m_pendingInputs.clear();
        }
        
        CefDoMessageLoopWork();
        // if(!workToDo)
        //     usleep(1000); // sleep for 1ms if no events to process
    }
    std::cout << "CEF destroyed " << cef_client->HasOneRef() << " " << browser->HasOneRef() << std::endl;
    browser->GetHost()->CloseBrowser(false);

    for (int attempts = 0; browser->IsValid() && attempts < 1000; ++attempts)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CefDoMessageLoopWork();
    }
    browser = nullptr;
    cef_client = nullptr;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for(int i=0; i<100; ++i)
        CefDoMessageLoopWork();
     std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CefShutdown();
}

void CefAppHandler::OnContextInitialized()
{
    //CEF_REQUIRE_UI_THREAD();

    if(!coVRMSController::instance()->isMaster())
        return;
    cef_client = new CEF_client(this);
    cef_client->setImageBuffer(m_imageBuffer);
    CefWindowInfo win;
    CefBrowserSettings browser_settings;
    win.SetAsWindowless(0);
#ifdef _WIN32
    win.shared_texture_enabled = false;
#endif


    browser = CefBrowserHost::CreateBrowserSync(win, cef_client, "www.google.de", browser_settings, nullptr, nullptr);
    browser->GetHost()->WasResized();
    browser->GetHost()->WasHidden(false);

    // Force a call to OnPaint.
    browser->GetHost()->Invalidate(PET_VIEW);
}

CefRefPtr<CefClient> CefAppHandler::GetDefaultClient()
{
    // Called when a new browser window is created via the Chrome runtime UI.
    return cef_client;
}

void CEF::message(int toWhom, int type, int length, const void *data)
{
    if (type == opencover::PluginMessageTypes::Browser)
    {
        url = (static_cast<const char *>(data), length);
        if(m_cefAppHandler)
            m_cefAppHandler->open(url);
    }
}

void CEF_client::OnBeforeClose(CefRefPtr<CefBrowser> browser)
{
    std::cout << "OnBeforeClose called - browser is fully closed" << std::endl;
}

bool CEF_client::DoClose(CefRefPtr<CefBrowser> browser)
{
    InputEvent e{InputEvent::CloseBrowser};
    e.browser = browser;
    cef->queueInputEvent(e);
    return false;
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
    std::lock_guard<std::mutex> lock(imageBuffer->mutex);
    memcpy(imageBuffer->buffer.data(), buffer, (size_t)width * height * 4);
    imageBuffer->bufferChanged = true;
    //std::cerr << "Render" << std::endl;
}

void VRUI_client::resize(int resolution, float aspect)
{
    width = resolution;
    height = width / aspect;
    imageBuffer.buffer.resize((size_t)width * height * 4);
}

void CEF_client::resize(int resolution, float aspect)
{
    width = resolution;
    height = width / aspect;
}


CEF_client::CEF_client(CefAppHandler *c)
{
    cef = c;
}

VRUI_client::VRUI_client(CEF *c)
: vruiCollabInterface(CEFCoim.get(), "CEFBrowser", vruiCollabInterface::PinEditor)
, cefPlugin(c)
, popupHandle(new coPopupHandle("BrowserHeadline"))
, interactionA(new coCombinedButtonInteraction(coInteraction::ButtonA, "CEFBrowser", coInteraction::Menu))
, interactionB(new coCombinedButtonInteraction(coInteraction::ButtonB, "CEFBrowser", coInteraction::Menu))
, interactionC(new coCombinedButtonInteraction(coInteraction::ButtonC, "CEFBrowser", coInteraction::Menu))
, imageBuffer(std::vector<unsigned char>((size_t)width * height * 4))
, videoTexture(new vrui::coTexturedBackground((uint *)imageBuffer.buffer.data(), NULL, NULL, 4, width, height, 0))
, interactionWheel(new coCombinedButtonInteraction(coInteraction::WheelVertical, "CEFBrowser", coInteraction::Menu))
{
    coIntersection::getIntersectorForAction("coAction")->add(videoTexture->getDCS(), this);
    videoTexture->setSize(width, height, 0);
    videoTexture->setTexSize(width, -height);
    videoTexture->setMinWidth(width);
    videoTexture->setMinHeight(height);

    popupHandle->setScale(2 * cover->getSceneSize() / 2500);
    popupHandle->setPos(-width * cover->getSceneSize() / 2500, 0, -height * cover->getSceneSize() / 2500);
    popupHandle->addElement(videoTexture);
    show();
}

VRUI_client::~VRUI_client()
{
    delete interactionA;
    delete interactionB;
    delete interactionC;
    delete interactionWheel;
    delete popupHandle;
}

void VRUI_client::update()
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
        if (interactionWheel->isRegistered() && (interactionWheel->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionWheel);
        }
        if (!interactionA->isRegistered() && !interactionB->isRegistered() && !interactionC->isRegistered() && !interactionWheel->isRegistered())
        {
            unregister = false;
        }
    }
    popupHandle->update();
    std::lock_guard<std::mutex> lock(imageBuffer.mutex);
    imageBuffer.bufferChanged = coVRMSController::instance()->syncBool(imageBuffer.bufferChanged);
    if(imageBuffer.bufferChanged)
    {
        // without mpi syncDate is quite slow and can cause fps drops
        coVRMSController::instance()->syncData(imageBuffer.buffer.data(), width * height * 4);
        imageBuffer.bufferChanged = false;
        videoTexture->setUpdated(true);
        videoTexture->setImage((uint *)imageBuffer.buffer.data(), NULL, NULL, 4, width, height, 0,
                                coTexturedBackground::TextureSet::PF_BGRA);
    }
}

void VRUI_client::show()
{
    if (popupHandle)
        popupHandle->setVisible(true);
}

void VRUI_client::hide()
{
    if (popupHandle)
        popupHandle->setVisible(false);
}

CefRefPtr<CefRenderHandler> CEF_client::GetRenderHandler()
{
    return this;
}

CefRefPtr<CefLifeSpanHandler> CEF_client::GetLifeSpanHandler()
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
, vrui_client(std::make_unique<VRUI_client>(this))
{
}

CEF::~CEF()
{
    delete backButton;
    delete forwardButton;
    delete reloadButton;
    delete urlLine;
    delete menu;
    if(m_cefAppHandler)
        m_cefAppHandler->terminate();
    if(m_browserThread && m_browserThread->joinable())
        m_browserThread->join();


}


int VRUI_client::hit(vruiHit *hit)
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
    if (!interactionWheel->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionWheel);
        interactionWheel->setHitByMouse(hit->isMouseHit());
    }

    osgUtil::LineSegmentIntersector::Intersection osgHit = dynamic_cast<OSGVruiHit *>(hit)->getHit();

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
            cover->grabKeyboard(cefPlugin);
            haveFocus = true;
        }
        if(!cef)
            return ACTION_CALL_ON_MISS;
        InputEvent event;
        event.x = x*width;
        event.y = y*height;
        if ((interactionA->getState() == coInteraction::Idle) && (interactionB->getState() == coInteraction::Idle) &&
            (interactionC->getState() == coInteraction::Idle) && (interactionWheel->getState() == coInteraction::Idle))
        {
            event.type = InputEvent::SetFocus;
            event.on = true;
        }
        if (interactionA->wasStarted())
        {
            event.type = InputEvent::LeftClick;
            cef->queueInputEvent(event);
        }
        else if (interactionA->wasStopped())
        {
            event.type = InputEvent::LeftClick;
            event.on = false;
        }
        else if (interactionB->wasStarted())
        {
            event.type = InputEvent::MiddleClick;
        }
        else if (interactionB->wasStopped())
        {
            event.type = InputEvent::MiddleClick;
            event.on = false;
        }
        else if (interactionC->wasStarted())
        {
            event.type = InputEvent::RightClick;
        }
        else if (interactionC->wasStopped())
        {
            event.type = InputEvent::RightClick;
            event.on = false;
        }
        else if (interactionWheel->wasStarted() || interactionWheel->isRunning())
        {
            event.type = InputEvent::MouseWheel;
            event.wheelDelta = (int)(interactionWheel->getWheelCount() * 120.0f);
        }
        else
        {
            event.type = InputEvent::MouseMove;
        }
        if(event.type != InputEvent::None)
            cef->queueInputEvent(event);

        return ACTION_CALL_ON_MISS;
        
    }

    return ACTION_CALL_ON_MISS;
}

void VRUI_client::miss()
{
    if(cef)
    {
        InputEvent e{InputEvent::SetFocus};
        e.on = false;
        cef->queueInputEvent(e);
    }
    
    unregister = true;

    if (haveFocus)
    {
        cover->releaseKeyboard(cefPlugin);
        haveFocus = false;
    }
}
const std::string &CEF::getURL()
{
    return url;
}

void CEF::reload()
{
    if (m_cefAppHandler)
        m_cefAppHandler->queueInputEvent(InputEvent{InputEvent::Reload});
}

bool CEF::init()
{
    

    CEFCoim.reset(new coCOIM(this));

    menu = new ui::Menu("Browser", this);
    menu->setVisible(false);
    menu->setVisible(true, ui::View::Tablet);


    backButton = new ui::Action(menu, "back");
    backButton->setText("back");
    forwardButton = new ui::Action(menu, "forward");
    forwardButton->setText("forward");
    reloadButton = new ui::Action(menu, "reload");
    reloadButton->setText("reload");
    urlLine = new ui::EditField(menu, "urlLine");
    urlLine->setText("URL");
    
    commandLine = new ui::EditField(menu, "CommandLine");
    commandLine->setText("Command line");
    if(coVRMSController::instance()->isMaster())
    {
        backButton->setCallback([this]() {m_cefAppHandler->queueInputEvent(InputEvent{InputEvent::GoBack}); });
        forwardButton->setCallback([this]() {m_cefAppHandler->queueInputEvent(InputEvent{InputEvent::GoForward}); });
        reloadButton->setCallback([this]() { m_cefAppHandler->queueInputEvent(InputEvent{InputEvent::Reload}); });
        urlLine->setCallback([this](const std::string &cmd)
            {
                if (cmd.length() > 0)
                {
                    m_cefAppHandler->queueInputEvent(InputEvent{InputEvent::OpenURL, cmd});
                }
            });
        commandLine->setCallback([this](const std::string &cmd)
            {
                if (cmd.length() > 0)
                {
                    m_cefAppHandler->queueInputEvent(InputEvent{InputEvent::Text, cmd});
                    commandLine->setText("");
                }
            });
            
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
        std::string lfp = "/tmp/cef.log";
        std::string fwpath = extlib + "/cef/Release/Chromium Embedded Framework.framework";
        std::string logfile = *configString("log", "file", lfp);
        int loglevel = *configInt("log", "level", LOGSEVERITY_VERBOSE);

        std::unique_lock<std::mutex> lk(m_cefInitMutex);

        m_browserThread = std::make_unique<std::thread>([this, frameworkDir=fwpath, browserSubprocessPath=bsp, logfile, loglevel]() {
            auto handler = std::make_unique<CefAppHandler>(vrui_client->getImageBuffer());
            {
                std::lock_guard<std::mutex> g(m_cefInitMutex);
                m_cefAppHandler = std::move(handler);
            }
            m_cefInitCv.notify_one();

            m_cefAppHandler->init(frameworkDir, browserSubprocessPath, logfile, loglevel);
            m_cefAppHandler->loop();
        });

        // wait until handler is created
        m_cefInitCv.wait(lk, [this]{ return m_cefAppHandler != nullptr; });
        vrui_client->cef = m_cefAppHandler.get();
        
    }
    return true;
}

bool CEF::update()
{
    if(vrui_client)
    {
        vrui_client->update();
    }
    return true;
}

void CefAppHandler::open(const std::string &url)
{
    queueInputEvent(InputEvent{InputEvent::OpenURL, url});
}

// void CEF::resize()
// {
//     vrui_client->resize(resolution, aspect);
//     if (m_cefAppHandler)
//     {
//         m_cefAppHandler->resize();
//         cef_client->resize(resolution, aspect);
//         browser->GetHost()->WasResized();
//     }
//     reload();
// }

void CEF::key(int type, int keySym, int mod)
{
    if(!coVRMSController::instance()->isMaster())
        return;
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
            m_cefAppHandler->queueInputEvent(InputEvent{InputEvent::SelectAll});
        if (keySym == 'c')
            m_cefAppHandler->queueInputEvent(InputEvent{InputEvent::Copy}); 
        if (keySym == 'v')
            m_cefAppHandler->queueInputEvent(InputEvent{InputEvent::Paste});
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
        InputEvent e{InputEvent::KeyEvent};
        e.keyEvent = keyEvent;
        m_cefAppHandler->queueInputEvent(e);
    }
    else
    {
        InputEvent e{InputEvent::KeyEvent};
        keyEvent.type = KEYEVENT_KEYUP;
        e.keyEvent = keyEvent;
        m_cefAppHandler->queueInputEvent(e);
        keyEvent.type = KEYEVENT_CHAR;
        e.keyEvent = keyEvent;
        m_cefAppHandler->queueInputEvent(e);
    }
}


// void CEF::setResolution(float a)
// {
//     resolution = a;
//     resize();
// }
// void CEF::setAspectRatio(float a)
// {
//     aspect = a;
//     resize();
// }


bool init()
{
    return true;
}


COVERPLUGIN(CEF)
