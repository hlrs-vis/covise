#ifndef RR_PLUGIN_H
#define RR_PLUGIN_H

#include <cover/coVRPluginSupport.h>

#include <deque>
#include <string>

#include <rfb/rfb.h>

using namespace covise;
using namespace opencover;

class ReadBackCuda;
class SendThread;

class VncPlugin: public coVRPlugin
{
public:
   VncPlugin();
   ~VncPlugin();
   bool init() override;

   bool update() override;
   void postSwapBuffers(int windowNumber) override;

private:
   struct Client
   {
      std::string host;
      int port;
   };

   std::vector<Client> m_clientList;
   ReadBackCuda *m_cudaReadBack;
   SendThread *m_sendThread;
   std::string m_touchEventReceiver;
   bool benchmark;

   void waitForConnection();

   void readpixels(GLint x, GLint y, GLint w, GLint pitch, GLint h,
	GLenum format, int ps, GLubyte *bits, GLint buf);
   bool m_cudapinnedmemory;

   rfbScreenInfoPtr m_screen;
   int m_width, m_height;

   static bool s_changed;
   static void keyEvent(rfbBool down, rfbKeySym sym, rfbClientPtr cl);
   static void pointerEvent(int buttonmask, int x, int y, rfbClientPtr cl);
};
#endif
