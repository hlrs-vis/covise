#ifndef MSEVENTHANDLER_H
#define MSEVENTHANDLER_H

#include <osgGA/GUIEventHandler>

namespace opencover
{
class MSEventHandler : public osgGA::GUIEventHandler
{
public:
    MSEventHandler()
    {
        numEventsToSync = 0;
    }

    void update();
    virtual bool handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &);

    /* virtual void accept(osgGA::GUIEventHandlerVisitor& v)
      {
         v.visit(*this);
      }*/

protected:
    int eventBuffer[1000];
    int keyBuffer[1000];
    int modBuffer[1000];
    int numEventsToSync;
};
}

#endif
