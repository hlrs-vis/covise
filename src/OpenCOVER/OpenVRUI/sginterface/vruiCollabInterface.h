/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_COLLAB_INTERFACE_H
#define VRUI_COLLAB_INTERFACE_H

#include <util/coTypes.h>

#include <list>
#include <string>

//class OPENVRUIEXPORT vruiCollabInterface;
namespace vrui
{
class vruiCollabInterface;
}

/// collaborative interface manager
EXPORT_TEMPLATE(template class OPENVRUIEXPORT std::list<vrui::vruiCollabInterface *>)

namespace vrui
{

class OPENVRUIEXPORT vruiCOIM
{

protected:
    vruiCOIM();
    virtual ~vruiCOIM();

public:
    void receiveMessage(int type, int len, const void *buf);

    void addInterface(vruiCollabInterface *myinterface);
    void removeInterface(vruiCollabInterface *myinterface);

private:
    std::list<vruiCollabInterface *> interfaces; ///< list of all collaborative UIs
};

/// base class for collaborative Userinterface elements
class OPENVRUIEXPORT vruiCollabInterface
{
public:
    enum
    {
        NONE = 10,
        VALUEPOTI = 11,
        HSVWHEEL = 12,
        PUSHBUTTON = 13,
        TOGGLEBUTTON = 14,
        FunctionEditor = 15,
        PinEditor = 16
    };

    vruiCollabInterface(vruiCOIM *manager, const std::string &interfaceName, int iType = NONE);
    virtual ~vruiCollabInterface();

    int getType() const;
    vruiCOIM *getManager()
    {
        return coim;
    }

    virtual void parseMessage(int type, unsigned int len, const char *message);

protected:
    int remoteContext; ///< a remote context (if this UI element is used in several contexts)
    void setType(int interfaceType);
    void sendLockMessage(const char *message);
    void sendOngoingMessage(const char *message);
    void sendReleaseMessage(const char *message);
    virtual void remoteLock(const char *message);
    virtual void remoteOngoing(const char *message);
    virtual void releaseRemoteLock(const char *message);

private:
    inline int composeMessage(const char messageType, const char *message);

    int interfaceType; ///< type of this interface
    std::string name; ///< symbolic name of this interface
	size_t bufLen; ///< message buffer length
    char *sendBuf; ///< message buffer
    bool locked; ///< true if this interface is locked
    vruiCOIM *coim; ///< the collaborative interaction manager
};
}
//typedef vruiCollabInterface coCollabInterface;
#endif
