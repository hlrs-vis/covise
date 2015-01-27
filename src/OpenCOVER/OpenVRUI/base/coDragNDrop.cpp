/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coDragNDrop.h>

#include <OpenVRUI/coButtonMenuItem.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiButtons.h>

namespace vrui
{

coDragNDropManager coDragNDropManager::ddManager;

// ------------------------------------------------------------------
// Drag'n'Drop Manager
coDragNDropManager::coDragNDropManager()
{
}

coDragNDropManager::~coDragNDropManager()
{
}

void coDragNDropManager::signOn(coDragNDrop *item)
{
    updateClasses.push_back(item);
}

void coDragNDropManager::signOff(coDragNDrop *item)
{
    // check out from manager
    updateClasses.remove(item);
}

void coDragNDropManager::drag(coDragNDrop *item)
{
    // append item to selection
    selection.push_back(item);
}

void coDragNDropManager::drop(coDragNDrop *item)
{
    // delete item from selection
    selection.remove(item);
}

coDragNDrop *coDragNDropManager::first(int /*mediaType*/)
{
    // search for the first item with this mediatype
    // and return it

    return 0;
}

bool coDragNDropManager::update()
{
    return true;
}

// ------------------------------------------------------------------
// Drag'n'Drop Item
coDragNDrop::coDragNDrop()
{
    // set media type to unknown
    mediaType = MEDIA_UNKNOWN;

    // register at manager
    coDragNDropManager::ddManager.signOn(this);
}

coDragNDrop::coDragNDrop(int myMediaType)
{
    // copy media type
    mediaType = myMediaType;

    // register at manager
    coDragNDropManager::ddManager.signOn(this);
}

coDragNDrop::~coDragNDrop()
{
    coDragNDropManager::ddManager.signOff(this);
}

/// register Item into selection

bool coDragNDrop::processHit()
{

    // check if drag or drop is requested
    if (vruiRendererInterface::the()->getButtons()->wasPressed() & vruiButtons::DRAG_BUTTON)
    {
        // drag button was pressed
        dragOperation();
    }
    else if (vruiRendererInterface::the()->getButtons()->wasReleased() & vruiButtons::DRAG_BUTTON)
    {
        // drag button was released
        // CHECK MEDIA TYPE FIRST???
        // or let target do this?
        if (dropOperation(0))
        {
            // operation sucessful
            // remove item from list
        }
    }

    return true;
}

void coDragNDrop::dragOperation()
{
    // ad this object to the selection
    coDragNDropManager::ddManager.drag(this);
}
}
