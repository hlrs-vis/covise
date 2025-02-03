/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cassert>

#include "vvPluginSupport.h"
#include "vvShutDownHandler.h"

//vvShutDownHandlerList///////////////////////////////////////////////

using namespace covise;
using namespace vive;

// set protected static pointer for singleton to NULL
vvShutDownHandlerList *vvShutDownHandlerList::p_sdhl = NULL;

// constructor, destructor, instance ---------------------------------
vvShutDownHandlerList::vvShutDownHandlerList()
{
    assert(!p_sdhl);

    if (vv->debugLevel(2))
        std::cerr << "vvShutDownHandlerList::vvShutDownHandlerList" << std::endl;
    // create instance of list
    p_handlerList = new std::list<vvShutDownHandler *>;
}

vvShutDownHandlerList::~vvShutDownHandlerList()
{
    if (vv->debugLevel(2))
        std::cerr << "vvShutDownHandlerList::vvShutDownHandlerList" << std::endl;

    std::list<vvShutDownHandler *>::iterator i;

    // delete vvShutDownHandlers in list / call public destructors
    for (i = p_handlerList->begin(); i != p_handlerList->end(); ++i)
    {
        // dereference and call destructor
        delete (*i);
    }

    // free list
    delete p_handlerList;

    p_sdhl = NULL;
}

// singleton
vvShutDownHandlerList *vvShutDownHandlerList::instance()
{
    if (p_sdhl == NULL)
    {
        p_sdhl = new vvShutDownHandlerList();
    }
    return p_sdhl;
}
//--------------------------------------------------------------------

// public methods ----------------------------------------------------
void vvShutDownHandlerList::addHandler(vvShutDownHandler *handler)
{
    if (vv->debugLevel(2))
        std::cerr << "vvShutDownHandlerList::addHandler" << std::endl;

    // check if this vvShutDownHandler is not already in list
    std::list<vvShutDownHandler *>::iterator i = p_handlerList->begin();

    while (i != p_handlerList->end())
    {
        if ((*i) == handler)
        {
            return;
        }
        i++;
    }

    // add to list
    p_handlerList->push_back(handler);

    // find algorithm?
    // set?
}

void vvShutDownHandlerList::shutAllDown()
{
    if (vv->debugLevel(2))
        std::cerr << "vvShutDownHandlerList::shutAllDown" << std::endl;

    std::list<vvShutDownHandler *>::iterator i;

    // call shutDown() of all vvShutDownHandlers in list
    for (i = p_handlerList->begin(); i != p_handlerList->end(); ++i)
    {
        (*i)->shutDown();
    }
}
//--------------------------------------------------------------------
