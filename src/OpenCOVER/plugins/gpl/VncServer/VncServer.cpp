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

#include "VncServer.h"

#include "ReadBackCuda.h"

#ifdef __linux__
#include <sys/prctl.h>
#endif

#include <rfb/rfb.h>


bool VncPlugin::s_changed = false;


class SendThread : public OpenThreads::Thread
{
   public:
      SendThread(VncPlugin *plugin);
      ~SendThread();
      virtual void run();
      void complete();
      void stop();
   private:
      VncPlugin *m_plugin;
      OpenThreads::Mutex m_mutexReady;
      OpenThreads::Mutex m_mutexData;
      OpenThreads::Condition m_condData;
};

VncPlugin::VncPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
   //fprintf(stderr, "new VncServer plugin\n");
}





bool VncPlugin::init()
{
   s_changed = false;
   m_cudaReadBack = NULL;
   m_sendThread = NULL;
   m_cudapinnedmemory = false;
   std::string config("COVER.Plugin.VncServer");

   if(covise::coCoviseConfig::isOn("sendthread", config, false))
       m_sendThread = new SendThread(this);

   osgViewer::GraphicsWindow *win = coVRConfig::instance()->windows[0].window;
   int x,y,w,h;
   win->getWindowRectangle(x,y,w,h);
   m_width = w;
   m_height = h;

   int argc = 1;
   char *argv[] = { (char *)"OpenCOVER", NULL };
   m_screen = rfbGetScreen(&argc, argv, w, h, 8, 3, 4);
   m_screen->desktopName = "OpenCOVER";

   m_screen->kbdAddEvent = &keyEvent;
   m_screen->ptrAddEvent = &pointerEvent;

   m_screen->frameBuffer = new char[w*h*4];
   rfbInitServer(m_screen);

   benchmark = covise::coCoviseConfig::isOn("benchmark", config, false);

#ifdef HAVE_CUDA
   if(covise::coCoviseConfig::isOn("cudaread", config, false))
   {
      try
      {
         m_cudaReadBack = new ReadBackCuda();
      }
      catch(ReadBackError err)
      {
         std::cerr << "failed to initialize ReadBackCuda: " << err.getMessage() << std::endl;
      }
   }

   if(m_cudaReadBack)
   {
#if 0
      if(covise::coCoviseConfig::isOn("cudayuv2rgb", config, true))
          fconfig.compress = RRCOMP_YUV2JPEG;
#endif

      m_cudapinnedmemory = covise::coCoviseConfig::isOn("cudapinned", config, true);
   }
#endif

   std::string host = covise::coCoviseConfig::getEntry("clientHost", config);
   if(!host.empty())
   {
      Client c;
      c.host = host;
      c.port = covise::coCoviseConfig::getInt("clientPort", config, 31042);
      m_clientList.push_back(c);
   }
   host = covise::coCoviseConfig::getEntry("alternateClientHost", config);
   if(!host.empty())
   {
      Client c;
      c.host = host;
      c.port = covise::coCoviseConfig::getInt("alternateClientPort", config, 31042);
      m_clientList.push_back(c);
   }

   m_touchEventReceiver = covise::coCoviseConfig::getEntry("touchEventReceiver", "COVER.Plugin.VncServer", "Utouch3D");

   if(m_sendThread)
      m_sendThread->start();

   return true;
}


// this is called if the plugin is removed at runtime
VncPlugin::~VncPlugin()
{
   if(m_sendThread)
   {
      m_sendThread->stop();
      m_sendThread->complete();
      m_sendThread->cancel();
      delete m_sendThread;
   }

   delete[] m_screen->frameBuffer;

#ifdef HAVE_CUDA
   delete m_cudaReadBack;
#endif

   //fprintf(stderr,"VncPlugin::~VncPlugin\n");
}


void VncPlugin::keyEvent(rfbBool down, rfbKeySym sym, rfbClientPtr cl)
{
   s_changed = true;

   static int modifiermask = 0;
   int modifierbit = 0;
   switch(sym) {
      case 0xffe1: // shift
         modifierbit = 0x01;
         break;
      case 0xffe3: // control
         modifierbit = 0x04;
         break;
      case 0xffe7: // meta
         modifierbit = 0x40;
         break;
      case 0xffe9: // alt
         modifierbit = 0x10;
         break;
   }
   if (modifierbit) {
      if (down)
         modifiermask |= modifierbit;
      else
         modifiermask &= ~modifierbit;
   }
   fprintf(stderr, "key %d %s, mod=%02x\n", sym, down?"down":"up", modifiermask);
   OpenCOVER::instance()->handleEvents(down ? osgGA::GUIEventAdapter::KEYDOWN : osgGA::GUIEventAdapter::KEYUP, modifiermask, sym);
}

namespace {
void handleMouseEvent(int type, int x, int y)
{
    OpenCOVER::instance()->handleEvents(type, x, y);
}
}


void VncPlugin::pointerEvent(int buttonmask, int ex, int ey, rfbClientPtr cl)
{
   s_changed = true;

   static int lastmask = 0;
   int x,y,w,h;
   osgViewer::GraphicsWindow *win = coVRConfig::instance()->windows[0].window;
   win->getWindowRectangle(x,y,w,h);
   handleMouseEvent(osgGA::GUIEventAdapter::DRAG, ex, h-1-ey);
   int changed = lastmask ^ buttonmask;
   lastmask = buttonmask;

   for(int i=0; i<3; ++i) {
      if (changed&1) {
         handleMouseEvent((buttonmask&1) ? osgGA::GUIEventAdapter::PUSH : osgGA::GUIEventAdapter::RELEASE, lastmask, 0);
         fprintf(stderr, "button %d %s\n", i+1, buttonmask&1 ? "push" : "release");
      }
      buttonmask >>= 1;
      changed >>= 1;
   }
}


bool
VncPlugin::update()
{
   rfbProcessEvents(m_screen, 10000);
   //fprintf(stderr, "VncServer preFrame\n");

   if(m_sendThread)
      m_sendThread->complete();

   bool ret = s_changed;
   s_changed = false;
   return ret;

#if 0
   while(!m_eventQueue.empty())
   {
      rrxevent rev = m_eventQueue.front();
      m_eventQueue.pop_front();
      switch(rev.type) {
         case RREV_MOTION:
            {
               osgViewer::GraphicsWindow *win = coVRConfig::instance()->windows[0].window;
               int x,y,w,h;
               win->getWindowRectangle(x,y,w,h);
               handleMouseEvent(osgGA::GUIEventAdapter::DRAG, (int)(w * rev.x + 0.5f), (int)(h * rev.y + 0.5f));
            }
            break;
         case RREV_WHEEL:
            while(rev.d2 > 0) {
               handleMouseEvent(osgGA::GUIEventAdapter::SCROLL, osgGA::GUIEventAdapter::SCROLL_UP, 0);
               --rev.d2;
            }
            while(rev.d2 < 0) {
               handleMouseEvent(osgGA::GUIEventAdapter::SCROLL, osgGA::GUIEventAdapter::SCROLL_DOWN, 0);
               ++rev.d2;
            }
            break;
         case RREV_BTNPRESS:
            handleMouseEvent(osgGA::GUIEventAdapter::PUSH, rev.d2, 0);
            break;
         case RREV_BTNRELEASE:
            handleMouseEvent(osgGA::GUIEventAdapter::RELEASE, rev.d2, 0);
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
               int x,y,w,h;
               win->getWindowRectangle(x,y,w,h);
               win->setWindowRectangle(x,y,(int)rev.x,(int)rev.y);
            }
            break;
         case RREV_TOUCHPRESS:
         case RREV_TOUCHRELEASE:
         case RREV_TOUCHMOVE:
            cover->sendMessage(this, m_touchEventReceiver.c_str(), PluginMessageTypes::RRZK_rrxevent, sizeof(rrxevent), &rev, true);
            break;
      }
   }
#endif
}

void VncPlugin::waitForConnection()
{
   unsigned short port = covise::coCoviseConfig::getInt("serverPort", "COVER.Plugin.VncServer", 31043);

   fprintf(stderr, "VncServer: waiting for connections...\n");
}

static void flip_upside_down(char *buf, unsigned w, unsigned h, unsigned bpp)
{
   char *front = buf;
   char *back = &buf[w * (h-1) * bpp];
   std::vector<char> temp(w*bpp);

   while (front < back) {
      memcpy(&temp[0], front, w*bpp);
      memcpy(front, back, w*bpp);
      memcpy(back, &temp[0], w*bpp);

      front += w*bpp;
      back -= w*bpp;
   }
}

void
VncPlugin::postSwapBuffers(int windowNumber)
{
   if(windowNumber != 0)
      return;
   if(!coVRMSController::instance()->isMaster())
      return;

   double start = 0.;
   if(benchmark)
      start = cover->currentTime();
   double bpp = 4.;
   GLint buf = GL_FRONT;
   GLenum format = GL_RGBA;
#ifdef HAVE_CUDA
   if(m_cudaReadBack && 0)
   {
#if 0
      int hs = tjmcuw[jpegsub(subsamp)]/8;
      int vs = tjmcuh[jpegsub(subsamp)]/8;
      if(fconfig.compress == RRCOMP_YUV2JPEG)
      {
         m_cudaReadBack->readpixelsyuv(0, 0, m_width, b->_pitch, m_height, format,
               4, m_screen->frameBuffer, buf, hs, vs);
         bpp = 1. + 2./hs/vs;
      }
      else
#endif
         m_cudaReadBack->readpixels(0, 0, m_width, m_width, m_height, format,
               4, (GLubyte *)m_screen->frameBuffer, buf);
   }
   else
#endif
      readpixels(0, 0, m_width, m_width, m_height, format,
            4, (GLubyte*)m_screen->frameBuffer, buf);

   flip_upside_down(m_screen->frameBuffer, m_width, m_height, 4);

   rfbMarkRectAsModified(m_screen,0,0,m_width,m_height);

   double pix = m_width*m_height;
   double bytes = pix * bpp;
   if(benchmark) {
      double dur = cover->currentTime() - start;
      fprintf(stderr, "%fs: %f mpix/s, %f gb/s (cuda=%d, yuv=%d)\n",
            dur,
            pix/dur/1e6, bytes/dur/(1024*1024*1024),
            m_cudaReadBack!=NULL, 0 /*fconfig.compress==RRCOMP_YUV2JPEG*/);
   }
}


void VncPlugin::readpixels(GLint x, GLint y, GLint w, GLint pitch, GLint h,
      GLenum format, int ps, GLubyte *bits, GLint buf)
{

   GLint readbuf=GL_BACK;
   glGetIntegerv(GL_READ_BUFFER, &readbuf);

   glReadBuffer(buf);
   glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);

   if(pitch%8==0) glPixelStorei(GL_PACK_ALIGNMENT, 8);
   else if(pitch%4==0) glPixelStorei(GL_PACK_ALIGNMENT, 4);
   else if(pitch%2==0) glPixelStorei(GL_PACK_ALIGNMENT, 2);
   else if(pitch%1==0) glPixelStorei(GL_PACK_ALIGNMENT, 1);

   // Clear previous error
   while (glGetError() != GL_NO_ERROR)
      ;

   glReadPixels(x, y, w, h, format, GL_UNSIGNED_BYTE, bits);

   glPopClientAttrib();
   glReadBuffer(readbuf);
}

SendThread::SendThread(VncPlugin *plugin)
: m_plugin(plugin)
{
}

SendThread::~SendThread()
{
}

void SendThread::run()
{
#ifdef __linux__
   prctl(PR_SET_NAME, "VncServer send", 0, 0, 0);
#endif
   for(;;)
   {
      m_mutexData.lock();
      m_condData.wait(&m_mutexData);

      m_mutexData.unlock();
   }
}

void SendThread::complete()
{
   m_mutexData.lock();
   m_mutexData.unlock();
}

void SendThread::stop()
{
   m_mutexData.lock();

   m_condData.signal();
   m_mutexData.unlock();
}

COVERPLUGIN(VncPlugin)
