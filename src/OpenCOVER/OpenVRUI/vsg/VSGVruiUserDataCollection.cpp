/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiUserDataCollection.h>

#include <OpenVRUI/sginterface/vruiUserData.h>

#include <OpenVRUI/util/vruiLog.h>
#include <vsg/nodes/Node.h>

using namespace std;
using namespace vsg;

namespace vrui
{

VSGVruiUserDataCollection::VSGVruiUserDataCollection()
{
}

VSGVruiUserDataCollection::~VSGVruiUserDataCollection()
{

    for (map<string, vruiUserData *>::iterator i = data.begin(); i != data.end(); ++i)
        delete i->second;
    data.clear();
}

void VSGVruiUserDataCollection::setUserData(const std::string &name, vruiUserData *userData)
{

    map<string, vruiUserData *>::iterator oldData = data.find(name);
    if (oldData != data.end())
    {
        //VRUILOG("VSGVruiUserDataCollection::addUserData info: replacing old data " << name)
        //delete oldData->second; vruiUserData wird von vruiIntersection geloescht, ausserdem sollte es vom vruiRenderInterface gemacht werden... richtig ?
        oldData->second = 0;
    }

    data[name] = userData;
}

//static vruiUserData * getUserData(vsg::Node *,const std::string & name);

void VSGVruiUserDataCollection::removeUserData(const std::string &name)
{
    data.erase(name);
}

vruiUserData *VSGVruiUserDataCollection::getUserData(const std::string &name)
{
    map<string, vruiUserData *>::iterator rData = data.find(name);
    if (rData != data.end())
        return rData->second;
    else
        return 0;
}

vruiUserData *VSGVruiUserDataCollection::getUserData(vsg::Node *node, const std::string &name)
{
    if(node)
    {
        VSGVruiUserDataCollection* collection;
        if(node->getValue("UserData", collection))
        {
            if (collection)
                return collection->getUserData(name);
        }
    }

    return 0;
}

void VSGVruiUserDataCollection::setUserData(vsg::Node *node, const std::string &name, vruiUserData *data)
{
    VSGVruiUserDataCollection* collection;
    if (node->getValue("UserData", collection))
    {
        if (collection)
            collection->setUserData(name, data);
    }
    else
    {
        VSGVruiUserDataCollection *collection = new VSGVruiUserDataCollection();
        collection->setUserData(name, data);
        node->setValue("UserData",collection);
    }
}
}
