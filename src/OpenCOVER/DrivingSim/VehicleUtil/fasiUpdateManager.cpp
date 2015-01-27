/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <fasiUpdateManager.h>

#include <algorithm>

fasiUpdateManager *fasiUpdateManager::fum = NULL;
fasiUpdateable::~fasiUpdateable()
{
    fasiUpdateManager::instance()->remove(this);
}

fasiUpdateable::fasiUpdateable()
{
    fasiUpdateManager::instance()->add(this);
}

fasiUpdateManager::fasiUpdateManager()
{
    fum = this;
}

fasiUpdateManager *fasiUpdateManager::instance()
{
    if (fum == NULL)
        fum = new fasiUpdateManager();
    return fum;
}

fasiUpdateManager::~fasiUpdateManager()
{
}

void fasiUpdateManager::add(fasiUpdateable *element, bool first)
{
    if (element && find(updateList.begin(), updateList.end(), element) == updateList.end())
    {
        if (first)
            updateList.push_front(element);
        else
            updateList.push_back(element);
    }
}

void fasiUpdateManager::remove(fasiUpdateable *element)
{
    updateList.remove(element);
}

void fasiUpdateManager::removeAll()
{
    updateList.clear();
}

void fasiUpdateManager::update()
{

    for (std::list<fasiUpdateable *>::iterator item = updateList.begin();
         item != updateList.end();)

    {
        if (!(*item)->update())
            item = updateList.erase(item);
        else
            ++item;
    }
}
