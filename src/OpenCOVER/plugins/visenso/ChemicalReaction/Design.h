/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DESIGN_H
#define _DESIGN_H

#include <osg/Geode>
#include <osg/Vec3>

#include <iostream>
#include <vector>

struct AtomConfig
{
    AtomConfig(int _element, int _charge, osg::Vec3 _position);
    AtomConfig();

    int element;
    int charge;
    osg::Vec3 position;
};

class Design
{
public:
    Design(std::string _symbol, std::string _name);
    ~Design();

    void addAtomConfig(int element, int charge, osg::Vec3 position);

    std::string symbol;
    std::string name;
    std::vector<AtomConfig> config;
};

#endif
