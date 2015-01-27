#ifndef RR_PLUGIN_H
#define RR_PLUGIN_H

#include <cover/coVRPluginSupport.h>

#include "rrxevent.h"
#include <deque>
#include <string>

using namespace covise;
using namespace opencover;

class rrdisplayclient;
class rrsocket;

class ReadBackCuda;
class SendThread;

class RRPlugin : public coVRPlugin
{
public:
    RRPlugin();
    ~RRPlugin();
    bool init();

    void preFrame();
    void postSwapBuffers(int windowNumber);

private:
    struct Client
    {
        std::string host;
        int port;
    };

    std::deque<rrxevent> m_eventQueue;
    std::vector<Client> m_clientList;
    rrdisplayclient *m_rrdpy;
    ReadBackCuda *m_cudaReadBack;
    SendThread *m_sendThread;
    std::string m_touchEventReceiver;
    bool benchmark;

    void waitForConnection();

    void tryConnect(bool block = false);

    void sendvgl(rrdisplayclient *rrdpy, GLint drawbuf, bool spoillast,
                 int compress, int qual, int subsamp, bool block = false);

    void readpixels(GLint x, GLint y, GLint w, GLint pitch, GLint h,
                    GLenum format, int ps, GLubyte *bits, GLint buf);
    bool m_cudapinnedmemory;
};
#endif
