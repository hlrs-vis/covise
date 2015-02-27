#ifndef MSEVENTHANDLER_H
#define MSEVENTHANDLER_H

#include <osgGA/GUIEventHandler>

namespace opencover
{
class MSEventHandler : public osgGA::GUIEventHandler
{
public:
    MSEventHandler();
    ~MSEventHandler();

    void update();
    virtual bool handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &);

protected:
    static const int NumEvents = 1000;
    int eventBuffer[NumEvents];
    int keyBuffer[NumEvents];
    int modBuffer[NumEvents];
    int numEventsToSync;

    bool handleTerminal; //< read keyboard input from controlling terminal
    // evdev keyboard
    int keyboardFd, notifyFd, watchFd;
    std::string devicePathname;
    std::string devicePath, deviceName;
    int modifierState;

    bool openEvdev();
};
}

#endif
