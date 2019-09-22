/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "attributeContainer.h"
#include <do/coDoSet.h>

attributeContainer::attributeContainer(coDistributedObject *obj)
    : _dummyName("")
    , _isASet(false)
    , _timeSteps(0)
{
    const coDistributedObject *const *setChildren = NULL;
    int noChildren;

    if (!obj)
        return;

    _dummyName = obj->getName();
    const coDoSet *set = dynamic_cast<const coDoSet *>(obj);
    if (set)
    {
        _isASet = true;
        setChildren = set->getAllElements(&noChildren);
        if (obj->getAttribute("TIMESTEP"))
        {
            _timeSteps = noChildren;
        }
    }
    // set primary attributes
    const char **attributes;
    const char **contents;
    int noPrimaryAttributes = obj->getAllAttributes(&attributes, &contents);
    int attribute;
    for (attribute = 0; attribute < noPrimaryAttributes; ++attribute)
    {
        _primaryAttributes.push_back(
            pair<string, string>(attributes[attribute], contents[attribute]));
    }
    // set secondary attributes
    if (set)
    {
        int child;
        for (child = 0; child < noChildren; ++child)
            addSecondary(setChildren[child]);
    }
    // set object list using visiting the objects using preorder
    preOrderList(obj);
}

void
attributeContainer::preOrderList(coDistributedObject *obj)
{
    if (!obj)
        return;

    if (coDoSet *set = dynamic_cast<coDoSet *>(obj))
    {
        coDistributedObject **setChildren = NULL;
        int noChildren;
        setChildren = const_cast<coDistributedObject **>(set->getAllElements(&noChildren));
        int child;
        for (child = 0; child < noChildren; ++child)
        {
            preOrderList(setChildren[child]);
        }
    }
    _objects.push_back(obj);
}

void
attributeContainer::addSecondary(const coDistributedObject *obj)
{
    if (!obj)
        return;

    const char **attributes;
    const char **contents;
    int noAttributes = obj->getAllAttributes(&attributes, &contents);
    int attribute;
    for (attribute = 0; attribute < noAttributes; ++attribute)
    {
        list<pair<string, string> >::iterator it;
        for (it = _secondaryAttributes.begin();
             it != _secondaryAttributes.end(); ++it)
        {
            if (attributes[attribute] == it->first)
            {
                if (contents[attribute] != it->second)
                {
                    _secondaryAttributes.erase(it);
                    it = _secondaryAttributes.end();
                }
                break;
            }
        }
        if (it == _secondaryAttributes.end()) // add new attribute
        {
            _secondaryAttributes.push_back(
                pair<string, string>(attributes[attribute], contents[attribute]));
        }
    }

    if (const coDoSet *set = dynamic_cast<const coDoSet *>(obj))
    {
        const coDistributedObject *const *setChildren = NULL;
        int noChildren;
        setChildren = set->getAllElements(&noChildren);
        int child;
        for (child = 0; child < noChildren; ++child)
            addSecondary(setChildren[child]);
    }
}

attributeContainer::~attributeContainer()
{
}

int
attributeContainer::timeSteps()
{
    return _timeSteps;
}

string
attributeContainer::dummyName()
{
    return _dummyName;
}

void
attributeContainer::clean() // destroy objects
{
    vector<coDistributedObject *>::iterator it;
    for (it = _objects.begin(); it != _objects.end(); ++it)
    {
        (*it)->destroy();
        delete *it; //FIXME
    }
}

#include <algorithm>
#include <functional>

bool
firstPart(pair<string, string> left,
          pair<string, string> right)
{
    return (left.first == right.first);
}

void
attributeContainer::addAttributes(coDistributedObject *obj,
                                  vector<pair<string, string> > theseAttributes)
{
    if (!obj)
        return;
    // primary attributes
    vector<pair<string, string> >::iterator it;
    for (it = _primaryAttributes.begin(); it != _primaryAttributes.end(); ++it)
    {
        vector<pair<string, string> >::iterator wo = find_if(theseAttributes.begin(), theseAttributes.end(),
                [it](const pair<string,string> &right) {
                    return (it->first == right.first);
                });
        if (wo == theseAttributes.end())
        {
            obj->addAttribute(it->first.c_str(), it->second.c_str());
        }
        else
        {
            obj->addAttribute(it->first.c_str(), wo->second.c_str());
        }
    }
    // secondary attributes
    list<pair<string, string> >::iterator lit;
    if (coDoSet *set = dynamic_cast<coDoSet *>(obj))
    {
        coDistributedObject *const *setChildren = NULL;
        int noChildren;
        setChildren = const_cast<coDistributedObject **>(set->getAllElements(&noChildren));
        // use only the first child
        if (setChildren[0])
        {
            for (lit = _secondaryAttributes.begin();
                 lit != _secondaryAttributes.end(); ++lit)
            {
                setChildren[0]->addAttribute(lit->first.c_str(), lit->second.c_str());
            }
            delete setChildren[0];
            delete[] setChildren;
        }
    }
}
