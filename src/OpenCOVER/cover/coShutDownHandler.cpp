/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cassert>

#include "coVRPluginSupport.h"
#include "coShutDownHandler.h"

//coShutDownHandlerList///////////////////////////////////////////////

using namespace covise;
using namespace opencover;

// set protected static pointer for singleton to NULL
coShutDownHandlerList *coShutDownHandlerList::p_sdhl = NULL;

// constructor, destructor, instance ---------------------------------
coShutDownHandlerList::coShutDownHandlerList()
{
    assert(!p_sdhl);

    if (cover->debugLevel(2))
        std::cerr << "coShutDownHandlerList::coShutDownHandlerList" << std::endl;
    // create instance of list
    p_handlerList = new std::list<coShutDownHandler *>;
}

coShutDownHandlerList::~coShutDownHandlerList()
{
    if (cover->debugLevel(2))
        std::cerr << "coShutDownHandlerList::coShutDownHandlerList" << std::endl;

    std::list<coShutDownHandler *>::iterator i;

    // delete coShutDownHandlers in list / call public destructors
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
coShutDownHandlerList *coShutDownHandlerList::instance()
{
    if (p_sdhl == NULL)
    {
        p_sdhl = new coShutDownHandlerList();
    }
    return p_sdhl;
}
//--------------------------------------------------------------------

// public methods ----------------------------------------------------
void coShutDownHandlerList::addHandler(coShutDownHandler *handler)
{
    if (cover->debugLevel(2))
        std::cerr << "coShutDownHandlerList::addHandler" << std::endl;

    // check if this coShutDownHandler is not already in list
    std::list<coShutDownHandler *>::iterator i = p_handlerList->begin();

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

void coShutDownHandlerList::shutAllDown()
{
    if (cover->debugLevel(2))
        std::cerr << "coShutDownHandlerList::shutAllDown" << std::endl;

    std::list<coShutDownHandler *>::iterator i;

    // call shutDown() of all coShutDownHandlers in list
    for (i = p_handlerList->begin(); i != p_handlerList->end(); ++i)
    {
        (*i)->shutDown();
    }
}
//--------------------------------------------------------------------
