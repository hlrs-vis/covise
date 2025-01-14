/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <util/coTypes.h>

#include <vsg/core/Inherit.h>
#include <vsg/core/Object.h>

#include <map>
#include <string>

namespace vrui
{

class vruiUserData;

class VSGVRUIEXPORT VSGVruiUserDataCollection : public vsg::Inherit<vsg::Object, VSGVruiUserDataCollection>
{

public:
    VSGVruiUserDataCollection();

    void setUserData(const std::string &name, vruiUserData *data);
    void removeUserData(const std::string &name);

    vruiUserData *getUserData(const std::string &name);
    static vruiUserData *getUserData(vsg::Node *, const std::string &name);
    static void setUserData(vsg::Node *, const std::string &name, vruiUserData *data);

protected:
    virtual ~VSGVruiUserDataCollection();

private:
    std::map<std::string, vruiUserData *> data;
};
}

EVSG_type_name(vrui::VSGVruiUserDataCollection);
