/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TABLET_UI_LISTENER_H
#define CO_TABLET_UI_LISTENER_H

/*! \file
 \brief Tablet user interface proxy classes

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2004
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#include <util/coExport.h>
#include <queue>
#include <map>

#include <tui/coAbstractTabletUI.h>

namespace covise
{
class TokenBuffer;
class Host;
class Message;
class Connection;
class ClientConnection;
class ServerConnection;
}
namespace vrb{
class VRBClient;
}
namespace osg
{
class Node;
};

namespace vive
{
class vvTabletUI;
class vvTUIElement;
class SGTextureThread;
class LocalData;
class IData;
class IRemoteData;
#ifdef WIN32
#pragma warning(push)
#pragma warning(disable: 4275)
#endif
/// Action listener for events triggered by any vvTUIElement.
class VVCORE_EXPORT vvTUIListener : public covise::coAbstractTUIListener
{

public:
    /** Action listener for events triggered by vvTUIElement.
      @param tUIItem pointer to element item which triggered this event
      */
    virtual ~vvTUIListener()
    {
    }
    virtual void tabletEvent(vvTUIElement *tUIItem);
    virtual void tabletPressEvent(vvTUIElement *tUIItem);
    virtual void tabletSelectEvent(vvTUIElement *tUIItem);
    virtual void tabletChangeModeEvent(vvTUIElement *tUIItem);
    virtual void tabletFindEvent(vvTUIElement *tUIItem);
    virtual void tabletLoadFilesEvent(char *nodeName);
    virtual void tabletReleaseEvent(vvTUIElement *tUIItem);
    virtual void tabletCurrentEvent(vvTUIElement *tUIItem);
    virtual void tabletDataEvent(vvTUIElement *tUIItem, covise::TokenBuffer &tb);
};

#ifdef WIN32
#pragma warning(pop)
#endif

}
#endif
