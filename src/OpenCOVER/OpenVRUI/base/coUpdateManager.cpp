/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coUpdateManager.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/util/vruiLog.h>

// undef this to get no START/END messaged
#undef VERBOSE

#ifdef VERBOSE
#define START(msg) fprintf(stderr, "START: %s\n", (msg))
#define END(msg) fprintf(stderr, "END:   %s\n", (msg))
#else
#define START(msg)
#define END(msg)
#endif

#include <algorithm>

using namespace std;

namespace vrui
{

coUpdateable::~coUpdateable()
{
    START("coUpdateable::~coUpdateable");

    if (vruiRendererInterface::the() && vruiRendererInterface::the()->getUpdateManager())
        vruiRendererInterface::the()->getUpdateManager()->remove(this);
}

coUpdateable::coUpdateable()
{
    START("coUpdateable::coUpdateable");
}

coUpdateManager::coUpdateManager()
{
    START("coUpdateManager::coUpdateManager");
}

coUpdateManager::~coUpdateManager()
{
    START("coUpdateManager::~coUpdateManager");
}

void coUpdateManager::add(coUpdateable *element, bool first)
{
    START("coUpdateManager::add");
    if (element && find(updateList.begin(), updateList.end(), element) == updateList.end())
    {
        if (first)
            updateList.push_front(element);
        else
            updateList.push_back(element);
    }
}

void coUpdateManager::remove(coUpdateable *element)
{
    START("coUpdateManager::remove");
    updateList.remove(element);
}

void coUpdateManager::removeAll()
{
    START("coUpdateManager::removeAll");
    updateList.clear();
}

void coUpdateManager::update()
{

    //MARK0("COVER update manager");

    START("coUpdateManager::update");
    //VRUILOG("coUpdateManager::update info: called for " << updateList.size() << " items")

    for (list<coUpdateable *>::iterator item = updateList.begin();
         item != updateList.end();)
    {
        if (!(*item)->update())
            item = updateList.erase(item);
        else
            ++item;
    }

    //MARK0("done");
}
}
