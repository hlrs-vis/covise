/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Design.h"

Design::Design(std::string _symbol, std::string _name)
    : symbol(_symbol)
    , name(_name)
{
}

Design::~Design()
{
}

void Design::addAtomConfig(int element, int charge, osg::Vec3 position)
{
    config.push_back(AtomConfig(element, charge, position));
}

//-------------------------------------------------------------------

AtomConfig::AtomConfig(int _element, int _charge, osg::Vec3 _position)
    : element(_element)
    , charge(_charge)
    , position(_position)
{
}

AtomConfig::AtomConfig()
    : element(0)
    , position(osg::Vec3(0.0f, 0.0f, 0.0f))
{
}
