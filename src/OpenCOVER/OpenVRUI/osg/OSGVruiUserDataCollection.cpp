/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiUserDataCollection.h>

#include <OpenVRUI/sginterface/vruiUserData.h>

#include <OpenVRUI/util/vruiLog.h>

using namespace std;
using namespace osg;

namespace vrui
{

OSGVruiUserDataCollection::OSGVruiUserDataCollection()
{
}

OSGVruiUserDataCollection::~OSGVruiUserDataCollection()
{

    for (map<string, vruiUserData *>::iterator i = data.begin(); i != data.end(); ++i)
        delete i->second;
    data.clear();
}

void OSGVruiUserDataCollection::setUserData(const std::string &name, vruiUserData *userData)
{

    map<string, vruiUserData *>::iterator oldData = data.find(name);
    if (oldData != data.end())
    {
        //VRUILOG("OSGVruiUserDataCollection::addUserData info: replacing old data " << name)
        //delete oldData->second; vruiUserData wird von vruiIntersection geloescht, ausserdem sollte es vom vruiRenderInterface gemacht werden... richtig ?
        oldData->second = 0;
    }

    data[name] = userData;
}

//static vruiUserData * getUserData(osg::Node *,const std::string & name);

void OSGVruiUserDataCollection::removeUserData(const std::string &name)
{
    data.erase(name);
}

vruiUserData *OSGVruiUserDataCollection::getUserData(const std::string &name)
{
    map<string, vruiUserData *>::iterator rData = data.find(name);
    if (rData != data.end())
        return rData->second;
    else
        return 0;
}

vruiUserData *OSGVruiUserDataCollection::getUserData(osg::Node *node, const std::string &name)
{
    if(node)
    {
        Referenced *nodeData = node->getUserData();
        if (nodeData)
        {
            OSGVruiUserDataCollection *collection = dynamic_cast<OSGVruiUserDataCollection *>(nodeData);
            if (collection)
                return collection->getUserData(name);
        }
    }

    return 0;
}

void OSGVruiUserDataCollection::setUserData(osg::Node *node, const std::string &name, vruiUserData *data)
{
    Referenced *nodeData = node->getUserData();
    if (nodeData)
    {
        OSGVruiUserDataCollection *collection = dynamic_cast<OSGVruiUserDataCollection *>(nodeData);
        if (collection)
            collection->setUserData(name, data);
    }
    else
    {
        OSGVruiUserDataCollection *collection = new OSGVruiUserDataCollection();
        collection->setUserData(name, data);
        node->setUserData(collection);
    }
}
}
