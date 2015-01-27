/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_USER_DATA_COLLECTION_H
#define OSG_VRUI_USER_DATA_COLLECTION_H

#include <util/coTypes.h>

#include <osg/Referenced>
#include <osg/Node>

#include <map>
#include <string>

namespace vrui
{

class vruiUserData;

class OSGVRUIEXPORT OSGVruiUserDataCollection : public osg::Referenced
{

public:
    OSGVruiUserDataCollection();

    void setUserData(const std::string &name, vruiUserData *data);
    void removeUserData(const std::string &name);

    vruiUserData *getUserData(const std::string &name);
    static vruiUserData *getUserData(osg::Node *, const std::string &name);
    static void setUserData(osg::Node *, const std::string &name, vruiUserData *data);

protected:
    virtual ~OSGVruiUserDataCollection();

private:
    std::map<std::string, vruiUserData *> data;
};
}
#endif
