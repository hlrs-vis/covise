/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_ABSTRACT_TABLET_UI_H
#define CO_ABSTRACT_TABLET_UI_H

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

namespace covise
{
class coAbstractTabletUI;
class coAbstractTUIElement;
class ClientConnection;
class TokenBuffer;
class Message;

/// Action listener for events triggered by any coAbstractTUIElement.
class coAbstractTUIListener
{
public:
    /** Action listener for events triggered by coAbstractTUIElement.
	@param tUIItem pointer to element item which triggered this event
	*/
    virtual ~coAbstractTUIListener()
    {
    }
#if 0
	virtual void tabletEvent(coAbstractTUIElement* tUIItem) = 0;
	virtual void tabletPressEvent(coAbstractTUIElement* tUIItem) = 0;
   virtual void tabletSelectEvent(coAbstractTUIElement* tUIItem) = 0;
   virtual void tabletFindEvent(coAbstractTUIElement* tUIItem) = 0;
	virtual void tabletReleaseEvent(coAbstractTUIElement* tUIItem) = 0;
   virtual void tabletCurrentEvent(coAbstractTUIElement* tUIItem) = 0;
#endif
};

/**
* Tablet PC Userinterface Mamager.
* This class provides a connection to a Tablet PC and handles all coAbstractTUIElement.
*/
class coAbstractTabletUI
{
public:
    virtual ~coAbstractTabletUI()
    {
    }
    virtual void update() = 0;
};

/**
* Base class for Tablet PC UI Elements.
*/
class coAbstractTUIElement
{
public:
    virtual ~coAbstractTUIElement()
    {
    }
    virtual void parseMessage(TokenBuffer &tb) = 0;
    virtual void resend() = 0;
    virtual void setPos(int, int) = 0;
    virtual void setSize(int, int) = 0;
    virtual void setLabel(const char *l) = 0;
    virtual coAbstractTUIListener *getMenuListener() = 0;
};

/**
* the filebrowser push button.
*/
class coAbstractTUIFileBrowserButton : public coAbstractTUIElement
{
public:
    enum DialogMode
    {
        OPEN = 1,
        SAVE = 2
    };
    virtual ~coAbstractTUIFileBrowserButton()
    {
    }
    virtual void setDirList(Message &ms) = 0;
    virtual void setFileList(Message &ms) = 0;
    virtual void setCurDir(Message &msg) = 0;
    virtual void setCurDir(const char *dir) = 0;
    virtual void resend() = 0;
    virtual void parseMessage(TokenBuffer &tb) = 0;
    virtual void setDrives(Message &ms) = 0;
    virtual void setClientList(Message &msg) = 0;
};
}
#endif
