#include <config/CoviseConfig.h>

#include <cover/coVRConfig.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>

#include <PluginUtil/PluginMessageTypes.h>

#include <osgGA/GUIEventAdapter>
#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <OpenThreads/Condition>

#include "RRServer.h"
#include "rrxevent.h"

#include "rrdisplayclient.h"
#include "rrsocket.h"
#include "fakerconfig.h"
#include "rrframe.h"

#include "tjplanar.h"

#include <RHR/ReadBackCuda.h>

#ifdef __linux__
#include <sys/prctl.h>
#endif

#define checkgl(m) \
    if (glerror()) \
        _throw("Could not " m);

#define _isright(drawbuf) (drawbuf == GL_RIGHT || drawbuf == GL_FRONT_RIGHT \
                           || drawbuf == GL_BACK_RIGHT)
#define leye(buf) (buf == GL_BACK ? GL_BACK_LEFT : (buf == GL_FRONT ? GL_FRONT_LEFT : buf))
#define reye(buf) (buf == GL_BACK ? GL_BACK_RIGHT : (buf == GL_FRONT ? GL_FRONT_RIGHT : buf))

class SendThread : public OpenThreads::Thread
{
public:
    SendThread(RRPlugin *plugin);
    ~SendThread();
    virtual void run();
    void send(rrdisplayclient *rrdpy, rrframe *b);
    void complete();
    void stop();

private:
    RRPlugin *m_plugin;
    OpenThreads::Mutex m_mutexReady;
    OpenThreads::Mutex m_mutexData;
    OpenThreads::Condition m_condData;
    rrdisplayclient *m_rrdpy;
    rrframe *m_b;
};

RRPlugin::RRPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    //fprintf(stderr, "new RRServer plugin\n");
}

bool RRPlugin::init()
{
    m_rrdpy = NULL;
    m_cudaReadBack = NULL;
    m_sendThread = NULL;
    m_cudapinnedmemory = false;
    std::string config("COVER.Plugin.RRServer");

    if (covise::coCoviseConfig::isOn("sendthread", config, false))
        m_sendThread = new SendThread(this);

    fconfig.verbose = cover->debugLevel(2);
    fconfig.spoil = 0;
    fconfig.np = covise::coCoviseConfig::getInt("threads", config, 4);
    ;
    fconfig.tilesize = covise::coCoviseConfig::getInt("tilesize", config, 256);
    ;
    fconfig.interframe = covise::coCoviseConfig::isOn("interframe", config, true);
    ;
    fconfig.qual = covise::coCoviseConfig::getInt("quality", config, 95);
    ;
    fconfig.subsamp = covise::coCoviseConfig::getInt("subsampling", config, 4);
    ;
    benchmark = covise::coCoviseConfig::isOn("benchmark", config, false);

    fconfig.compress = RRCOMP_JPEG;
#ifdef HAVE_CUDA
    if (covise::coCoviseConfig::isOn("cudaread", config, false))
    {
        m_cudaReadBack = new ReadBackCuda();
    }

    if (m_cudaReadBack)
    {
        if (covise::coCoviseConfig::isOn("cudayuv2rgb", config, true))
            fconfig.compress = RRCOMP_YUV2JPEG;

        m_cudapinnedmemory = covise::coCoviseConfig::isOn("cudapinned", config, true);
    }
#endif
    //fconfig.qual = 50;
    fconfig.transvalid[RRTRANS_VGL] = 1;
    if (cover->debugLevel(2))
        fconfig_print(fconfig);

    std::string host = covise::coCoviseConfig::getEntry("clientHost", config);
    if (!host.empty())
    {
        Client c;
        c.host = host;
        c.port = covise::coCoviseConfig::getInt("clientPort", config, 31042);
        m_clientList.push_back(c);
    }
    host = covise::coCoviseConfig::getEntry("alternateClientHost", config);
    if (!host.empty())
    {
        Client c;
        c.host = host;
        c.port = covise::coCoviseConfig::getInt("alternateClientPort", config, 31042);
        m_clientList.push_back(c);
    }

#if 0
   if(m_clientList.empty())
   {
      Client c;
      c.host = "localhost";
      c.port = 31042;
      m_clientList.push_back(c);
   }
#endif

    m_touchEventReceiver = covise::coCoviseConfig::getEntry("touchEventReceiver", "COVER.Plugin.RRServer", "Utouch3D");

    if (m_sendThread)
        m_sendThread->start();

    return true;
}

// this is called if the plugin is removed at runtime
RRPlugin::~RRPlugin()
{
    if (m_sendThread)
    {
        m_sendThread->stop();
        m_sendThread->complete();
        m_sendThread->cancel();
        delete m_sendThread;
    }

#ifdef HAVE_CUDA
    delete m_cudaReadBack;
#endif
    delete m_rrdpy;
    //fprintf(stderr,"RRPlugin::~RRPlugin\n");
}

void
RRPlugin::preFrame()
{
    //fprintf(stderr, "RRServer preFrame\n");

    if (m_sendThread)
        m_sendThread->complete();

    while (!m_eventQueue.empty())
    {
        rrxevent rev = m_eventQueue.front();
        m_eventQueue.pop_front();
        switch (rev.type)
        {
        case RREV_MOTION:
        {
            osgViewer::GraphicsWindow *win = coVRConfig::instance()->windows[0].window;
            int x, y, w, h;
            win->getWindowRectangle(x, y, w, h);
            cover->handleMouseEvent(osgGA::GUIEventAdapter::DRAG, (int)(w * rev.x + 0.5f), (int)(h * rev.y + 0.5f));
        }
        break;
        case RREV_WHEEL:
            while (rev.d2 > 0)
            {
                cover->handleMouseEvent(osgGA::GUIEventAdapter::SCROLL, osgGA::GUIEventAdapter::SCROLL_UP, 0);
                --rev.d2;
            }
            while (rev.d2 < 0)
            {
                cover->handleMouseEvent(osgGA::GUIEventAdapter::SCROLL, osgGA::GUIEventAdapter::SCROLL_DOWN, 0);
                ++rev.d2;
            }
            break;
        case RREV_BTNPRESS:
            cover->handleMouseEvent(osgGA::GUIEventAdapter::PUSH, rev.d2, 0);
            break;
        case RREV_BTNRELEASE:
            cover->handleMouseEvent(osgGA::GUIEventAdapter::RELEASE, rev.d2, 0);
            break;
        case RREV_KEYPRESS:
            OpenCOVER::instance()->handleEvents(osgGA::GUIEventAdapter::KEYDOWN, rev.d1, rev.d2);
            break;
        case RREV_KEYRELEASE:
            OpenCOVER::instance()->handleEvents(osgGA::GUIEventAdapter::KEYUP, rev.d1, rev.d2);
            break;
        case RREV_RESIZE:
        {
            osgViewer::GraphicsWindow *win = coVRConfig::instance()->windows[0].window;
            int x, y, w, h;
            win->getWindowRectangle(x, y, w, h);
            win->setWindowRectangle(x, y, (int)rev.x, (int)rev.y);
        }
        break;
        case RREV_TOUCHPRESS:
        case RREV_TOUCHRELEASE:
        case RREV_TOUCHMOVE:
            cover->sendMessage(this, m_touchEventReceiver.c_str(), PluginMessageTypes::RRZK_rrxevent, sizeof(rrxevent), &rev, true);
            break;
        }
    }
}

void RRPlugin::tryConnect(bool block)
{
    do
    {
        for (int i = 0; i < m_clientList.size(); ++i)
        {
            int port = m_clientList[i].port;
            std::string host = m_clientList[i].host;
            host += ":0";

            try
            {
                fprintf(stderr, "RRServer: trying connection to %s:%d\n", host.c_str(), port);
                m_rrdpy = NULL;
                m_rrdpy = new rrdisplayclient();
                m_rrdpy->connect(const_cast<char *>(host.c_str()), port);
            }
            catch (...)
            {
                delete m_rrdpy;
                m_rrdpy = NULL;
            }

            if (m_rrdpy)
            {
                fprintf(stderr, "RRServer: connected to %s:%d\n", m_clientList[i].host.c_str(), port);
                break;
            }
        }

        sleep(1);
    } while (block && !m_rrdpy);
}

void RRPlugin::waitForConnection()
{
    if (m_rrdpy)
        return;

    unsigned short port = covise::coCoviseConfig::getInt("serverPort", "COVER.Plugin.RRServer", 31043);
    rrsocket *sock = NULL;

    fprintf(stderr, "RRServer: waiting for connections...\n");

    rrsocket *temp = NULL;
    try
    {
        temp = new rrsocket(false);

        temp->listen(port);

        sock = temp->accept();

        m_rrdpy = new rrdisplayclient();
        m_rrdpy->connect(sock);

        delete temp;
        temp = NULL;
    }
    catch (rrerror & /*e*/)
    {
        delete temp;
        delete sock;
        delete m_rrdpy;
        m_rrdpy = NULL;
    }

    if (m_rrdpy)
    {
        fprintf(stderr, "RRServer: connected to %s:%d\n", sock->remotename(), port);
    }
}

void
RRPlugin::postSwapBuffers(int windowNumber)
{
    if (windowNumber != 0)
        return;
    if (!coVRMSController::instance()->isMaster())
        return;

    if (!m_rrdpy)
    {
        if (m_clientList.empty())
            waitForConnection();
        else
            tryConnect(true);
    }

    if (!m_rrdpy)
        return;

    try
    {
        sendvgl(m_rrdpy, GL_FRONT, false,
                fconfig.compress, fconfig.qual, fconfig.subsamp,
                true /*block*/);

        while (rrxevent *rev = m_rrdpy->getevent())
        {
            m_eventQueue.push_back(*rev);
            delete rev;
        }
    }
    catch (...)
    {
        delete m_rrdpy;
        m_rrdpy = NULL;
        fprintf(stderr, "sendvgl failed\n");
    }
}

void RRPlugin::sendvgl(rrdisplayclient *rrdpy, GLint drawbuf, bool spoillast,
                       int compress, int qual, int subsamp, bool block)
{

#if 0
   while(block && !rrdpy->frameready())
   {
      fprintf(stderr, "pause "); fflush(stderr);
      usleep(100000);
   }
#else
    (void)block;
#endif

    osgViewer::GraphicsWindow *win = coVRConfig::instance()->windows[0].window;
    int x, y, w, h;
    win->getWindowRectangle(x, y, w, h);

    if (spoillast && fconfig.spoil && !rrdpy->frameready())
        return;
    rrframe *b;
    int flags = RRBMP_BOTTOMUP, format = GL_RGB;
#ifdef GL_BGR_EXT
    if (littleendian() && compress != RRCOMP_RGB)
    {
        format = GL_BGR_EXT;
        flags |= RRBMP_BGR;
    }
#endif
    if (m_cudaReadBack && m_cudapinnedmemory)
        flags |= RRBMP_CUDAALLOC;
    errifnot(b = rrdpy->getbitmap(w, h, 3, flags,
                                  false /*stereo*/, fconfig.spoil));

    GLint buf = drawbuf;

    //b->_h.winid=_win;
    b->_h.dpynum = 0;
    b->_h.winid = 0;
    b->_h.framew = b->_h.width;
    b->_h.frameh = b->_h.height;
    b->_h.x = 0;
    b->_h.y = 0;
    b->_h.qual = qual;
    b->_h.subsamp = subsamp;
    b->_h.compress = (unsigned char)compress;

    double start = 0.;
    if (benchmark)
        start = cover->currentTime();
    double bpp = 4.;
#ifdef HAVE_CUDA
    if (m_cudaReadBack)
    {
        int hs = tjMCUWidth[jpegsub(subsamp)] / 8;
        int vs = tjMCUHeight[jpegsub(subsamp)] / 8;
        if (fconfig.compress == RRCOMP_YUV2JPEG)
        {
            m_cudaReadBack->readpixelsyuv(0, 0, b->_h.framew, b->_pitch, b->_h.frameh, format,
                                          b->_pixelsize, b->_bits, buf, hs, vs);
            bpp = 1. + 2. / hs / vs;
        }
        else
            m_cudaReadBack->readpixels(0, 0, b->_h.framew, b->_pitch, b->_h.frameh, format,
                                       b->_pixelsize, b->_bits, buf);
    }
    else
#endif
        readpixels(0, 0, b->_h.framew, b->_pitch, b->_h.frameh, format,
                   b->_pixelsize, b->_bits, buf);
    double dur = 0.;
    if (benchmark)
        dur = cover->currentTime() - start;
    double pix = b->_h.framew * b->_h.frameh;
    double bytes = pix * bpp;
    if (benchmark)
        fprintf(stderr, "%fs: %f mpix/s, %f gb/s (cuda=%d, yuv=%d)\n",
                dur,
                pix / dur / 1e6, bytes / dur / (1024 * 1024 * 1024),
                m_cudaReadBack != NULL, fconfig.compress == RRCOMP_YUV2JPEG);
    //b->_h.winid=_win;
    b->_h.dpynum = 0;
    b->_h.winid = 0;
    b->_h.framew = b->_h.width;
    b->_h.frameh = b->_h.height;
    b->_h.x = 0;
    b->_h.y = 0;
    b->_h.qual = qual;
    b->_h.subsamp = subsamp;
    b->_h.compress = (unsigned char)compress;

    if (m_sendThread)
    {
        m_sendThread->send(rrdpy, b);
    }
    else
        rrdpy->sendframe(b);
}

void RRPlugin::readpixels(GLint x, GLint y, GLint w, GLint pitch, GLint h,
                          GLenum format, int ps, GLubyte *bits, GLint buf)
{

    GLint readbuf = GL_BACK;
    glGetIntegerv(GL_READ_BUFFER, &readbuf);

    //tempctx tc(_localdpy, EXISTING_DRAWABLE, GetCurrentDrawable());

    glReadBuffer(buf);
    glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);

    if (pitch % 8 == 0)
        glPixelStorei(GL_PACK_ALIGNMENT, 8);
    else if (pitch % 4 == 0)
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
    else if (pitch % 2 == 0)
        glPixelStorei(GL_PACK_ALIGNMENT, 2);
    else if (pitch % 1 == 0)
        glPixelStorei(GL_PACK_ALIGNMENT, 1);

    int e = glGetError();
    while (e != GL_NO_ERROR)
        e = glGetError(); // Clear previous error
    //_prof_rb.startframe();
    glReadPixels(x, y, w, h, format, GL_UNSIGNED_BYTE, bits);
    //_prof_rb.endframe(w*h, 0, stereo? 0.5 : 1);
    //checkgl("Read Pixels");

    // If automatic faker testing is enabled, store the FB color in an
    // environment variable so the test program can verify it
    if (fconfig.autotest)
    {
        unsigned char *rowptr, *pixel;
        int match = 1;
        int color = -1, i, j, k;
        color = -1;
        //if(buf!=GL_FRONT_RIGHT && buf!=GL_BACK_RIGHT) _autotestframecount++;
        for (j = 0, rowptr = bits; j < h && match; j++, rowptr += pitch)
            for (i = 1, pixel = &rowptr[ps]; i < w && match; i++, pixel += ps)
                for (k = 0; k < ps; k++)
                {
                    if (pixel[k] != rowptr[k])
                    {
                        match = 0;
                        break;
                    }
                }
        if (match)
        {
            if (format == GL_COLOR_INDEX)
            {
                unsigned char index;
                glReadPixels(0, 0, 1, 1, GL_COLOR_INDEX, GL_UNSIGNED_BYTE, &index);
                color = index;
            }
            else
            {
                unsigned char rgb[3];
                glReadPixels(0, 0, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, rgb);
                color = rgb[0] + (rgb[1] << 8) + (rgb[2] << 16);
            }
        }
    }

    glPopClientAttrib();
    //tc.restore();
    glReadBuffer(readbuf);
}

SendThread::SendThread(RRPlugin *plugin)
    : m_plugin(plugin)
    , m_rrdpy(NULL)
    , m_b(NULL)
{
}

SendThread::~SendThread()
{
}

void SendThread::run()
{
#ifdef __linux__
    prctl(PR_SET_NAME, "RRServer send", 0, 0, 0);
#endif
    for (;;)
    {
        m_mutexData.lock();
        m_condData.wait(&m_mutexData);
        if (m_rrdpy && m_b)
        {
            try
            {
                m_rrdpy->sendframe(m_b);
                m_rrdpy = NULL;
                m_b = NULL;
            }
            catch (...)
            {
                fprintf(stderr, "send failed, client disconnectd?\n");
            }
        }
        else
        {
            // finish thread
            m_mutexData.unlock();
            return;
        }
        m_mutexData.unlock();
    }
}

void SendThread::send(rrdisplayclient *rrdpy, rrframe *b)
{
    m_mutexData.lock();
    m_rrdpy = rrdpy;
    m_b = b;
    m_condData.signal();
    m_mutexData.unlock();
}

void SendThread::complete()
{
    m_mutexData.lock();
    m_mutexData.unlock();
}

void SendThread::stop()
{
    m_mutexData.lock();
    m_rrdpy = NULL;
    m_b = NULL;
    m_condData.signal();
    m_mutexData.unlock();
}

COVERPLUGIN(RRPlugin)
