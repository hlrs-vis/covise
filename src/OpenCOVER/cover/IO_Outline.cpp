/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// -*-c++-*-

#include "Outline.h"

#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>

bool Outline_readLocalData(osg::Object &obj, osgDB::Input &fr);
bool Outline_writeLocalData(const osg::Object &obj, osgDB::Output &fw);

osgDB::RegisterDotOsgWrapperProxy Outline_Proxy(
    new osgFX::Outline,
    "osgFX::Outline",
    "Object Node Group osgFX::Effect osgFX::Outline",
    Outline_readLocalData,
    Outline_writeLocalData);

bool Outline_readLocalData(osg::Object &obj, osgDB::Input &fr)
{
    osgFX::Outline &myobj = static_cast<osgFX::Outline &>(obj);
    bool itAdvanced = false;

    if (fr[0].matchWord("outlineWidth"))
    {
        float w;
        if (fr[1].getFloat(w))
        {
            myobj.setWidth(w);
            fr += 2;
            itAdvanced = true;
        }
    }

    if (fr[0].matchWord("outlineColor"))
    {
        osg::Vec4 col;
        if (fr[1].getFloat(col.x()) && fr[2].getFloat(col.y()) && fr[3].getFloat(col.z()) && fr[4].getFloat(col.w()))
        {
            myobj.setColor(col);
            fr += 5;
            itAdvanced = true;
        }
    }

    return itAdvanced;
}

bool Outline_writeLocalData(const osg::Object & /*obj*/, osgDB::Output & /*fw*/)
{
    //const osgFX::Outline& myobj = static_cast<const osgFX::Outline&>(obj);

    //fw.indent() << "outlineWidth " << myobj.getWidth() << std::endl;
    //fw.indent() << "outlineColor " << myobj.getColor() << std::endl;

    return true;
}
