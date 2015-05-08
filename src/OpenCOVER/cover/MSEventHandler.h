#ifndef MSEVENTHANDLER_H
#define MSEVENTHANDLER_H

#include <osgGA/GUIEventHandler>
#include <vector>

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
    struct Event {
	Event(): event(0), mod(0), key(0) {}
	Event(int event, int mod, int key): event(event), mod(mod), key(key) {}

	int event, mod, key;
    };
    std::vector<Event> eventQueue;
#if 0
    int eventBuffer[NumEvents];
    int keyBuffer[NumEvents];
    int modBuffer[NumEvents];
    int numEventsToSync;
#endif

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
