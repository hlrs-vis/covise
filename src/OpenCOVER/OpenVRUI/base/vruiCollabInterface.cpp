/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiCollabInterface.h>
#include <OpenVRUI/util/vruiLog.h>

using namespace std;

namespace vrui
{

/** Constructor.
 */
vruiCOIM::vruiCOIM()
{
}

/// Destructor.
vruiCOIM::~vruiCOIM()
{
}

/** this method is called when a message from a remote UI arrives
   @param type message type
   @param len message length
   @param message the message
  */
void vruiCOIM::receiveMessage(int type, int len, const void *buf)
{
    for (list<vruiCollabInterface *>::iterator i = interfaces.begin(); i != interfaces.end(); ++i)
    {
        (*i)->parseMessage(type, len, static_cast<const char *>(buf));
    }
}

void vruiCOIM::addInterface(vruiCollabInterface *myinterface)
{
    interfaces.push_back(myinterface);
}

void vruiCOIM::removeInterface(vruiCollabInterface *myinterface)
{
    interfaces.remove(myinterface);
}

/** Constructor.
  @param c                       the current coim
  @param interfaceName           symbolic name of this UI
  @param iType                   interface type
*/
vruiCollabInterface::vruiCollabInterface(vruiCOIM *c, const string &interfaceName, int iType)
{

    bufLen = 0;
    sendBuf = 0;

    coim = c;
    interfaceType = iType;

    if (interfaceName != "")
        name = interfaceName;
    else
        name = "noname";

    remoteContext = -1;

    if (coim)
        coim->addInterface(this);
}

/// Destructor, removes references to this interface from its coim.
vruiCollabInterface::~vruiCollabInterface()
{
    if (coim)
    {
        coim->removeInterface(this);
    }

    delete[] sendBuf;
}

/** set the type of the this interface
 @param iType interface type
  */
void vruiCollabInterface::setType(int iType)
{
    interfaceType = iType;
}

/** get the interface type
 @return the type of this interface
  */
int vruiCollabInterface::getType() const
{
    return interfaceType;
}

size_t vruiCollabInterface::composeMessage(const char messageType, const char *message)
{

    size_t len;
    if (message)
        len = name.length() + 2 + strlen(message);
    else
        len = name.length() + 2;

    if (bufLen < len)
    {
        delete[] sendBuf;
        sendBuf = new char[len];
        bufLen = len;
    }

    strcpy(sendBuf, name.c_str());
    *(sendBuf + name.length()) = messageType;

    if (message)
        strcpy(sendBuf + name.length() + 1, message);
    else
        *(sendBuf + name.length() + 1) = '\0';

    return len;
}

/** lock the interaction with remote interfaces
 @param message custom message sent to remote uis
  */
void vruiCollabInterface::sendLockMessage(const char *message)
{

    size_t len = composeMessage('L', message);

    if (coim)
        vruiRendererInterface::the()->sendCollabMessage(this, sendBuf, (int)len);
}

/** send interactions to remote interfaces
 @param message custom message (e.g. current value) sent to remote uis
  */
void vruiCollabInterface::sendOngoingMessage(const char *message)
{

    int len = (int)composeMessage('O', message);

    if (coim)
        vruiRendererInterface::the()->sendCollabMessage(this, sendBuf, len);
}

/** unlock the interaction with remote interfaces
 @param message custom message sent to remote uis
  */
void vruiCollabInterface::sendReleaseMessage(const char *message)
{

    int len = (int)composeMessage('R', message);

    if (coim)
        vruiRendererInterface::the()->sendCollabMessage(this, sendBuf, len);
}

/** this method is called whenever a remote interaction is started.
    local ineractions should be locked now.
 @param message custom message from remote ui
  */
void vruiCollabInterface::remoteLock(const char *message)
{
    size_t retval;
    retval = sscanf(message, "%d", &remoteContext);
    if (retval != 1)
    {
        std::cerr << "vruiCollabInterface::remoteLock: sscanf failed" << std::endl;
        return;
    }
}

/** this method is called whenever a remote interaction is going on.
    local events should be generated.
 @param message custom message from remote ui (e.g. current value)
  */
void vruiCollabInterface::remoteOngoing(const char * /*message*/)
{
}

/** this method is called whenever a remote interaction is finished.
    local ineractions should be allowed again.
 @param message custom message from remote ui
  */
void vruiCollabInterface::releaseRemoteLock(const char * /*message*/)
{
}

/** this method is called when a message from a remote UI arrives
   @param type message type
   @param len message length
   @param message the message
  */
void vruiCollabInterface::parseMessage(int type, unsigned int len, const char *message)
{
    if (type == interfaceType)
    {

        if ((len > name.length()) && (name == message))
        {

            const char *myMessage = message + name.length();
            char messageType = *myMessage;
            myMessage++;
            switch (messageType)
            {
            case 'L':
                remoteLock(myMessage);
                break;
            case 'O':
                remoteOngoing(myMessage);
                break;
            case 'R':
                releaseRemoteLock(myMessage);
                break;
            default:
                VRUILOG("vruiCollabInterface::parseMessage: err: unknown messageType " << messageType)
                break;
            }
        }
    }
}
}
