/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include <util/coExport.h>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <utility>

#ifndef SHARED_STATE_CONTAINER_FUNCTIONS_H
#define SHARED_STATE_CONTAINER_FUNCTIONS_H

namespace vrb {

template <class T>
std::vector<T> getNewObjects(const std::vector<T> oldObjs, const std::vector<T> newObjs)
{
	std::vector<T> onlyNew;
	std::vector<T> oldCopie = oldObjs;
	for (T newObj : newObjs)
	{
		bool found = false;
		for (auto it = oldCopie.begin(); it != oldCopie.end(); it++)
		{
			if (newObj == *it)
			{
				found = true;
				oldCopie.erase(it);
				break;
			}
		}
		if (!found)
		{
			onlyNew.push_back(newObj);
		}
	}
	return onlyNew;
};

template <class T>
std::vector<T> getRemovedObjects(const std::vector<T> oldObjs, const std::vector<T> newObjs)
{
	std::vector<T> removed;
	std::vector<T> newCopie = newObjs;
	for (T oldObj : oldObjs)
	{
		bool found = false;
		for (auto it = newCopie.begin(); it != newCopie.end(); it++)
		{
			if (oldObj == *it)
			{
				found = true;
				newCopie.erase(it);
				break;
			}
		}
		if (!found)
		{
			removed.push_back(oldObj);
		}
	}
	return removed;
};

template <class T>
std::set<T> getNewObjects(const std::set<T> oldObjs, const std::set<T> newObjs)
{
	std::set<T> onlyNew;
	for (T newObj : newObjs)
	{
		if (oldObjs.find(newObj) == oldObjs.end())
		{
			onlyNew.insert(newObj);
		}
	}
	return onlyNew;
};

template <class T>
std::set<T> getRemovedObjects(const std::set<T> oldObjs, const std::set<T> newObjs)
{
	std::set<T> removed;
	for (T oldObj : oldObjs)
	{
		if (newObjs.find(oldObj) == newObjs.end())
		{
			removed.insert(oldObj);
		}
	}
	return removed;
};

}
#endif
