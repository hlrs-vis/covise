/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef LOADFILEVISITOR
#define LOADFILEVISITOR

#include <osg/Group>
#include <osg/NodeVisitor>

class LoadFileVisitor : public osg::NodeVisitor
{

public:
    LoadFileVisitor();
    virtual ~LoadFileVisitor();

    virtual void apply(osg::Group &node);
};

class LoadFile
{
public:
    LoadFile();
    virtual ~LoadFile();
    virtual void load(osg::Group *);
};

#endif
