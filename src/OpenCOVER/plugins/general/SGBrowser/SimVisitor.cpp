/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRFileManager.h>
#include <iostream>
#include <cstdio>
#include <PluginUtil/SimReference.h>
#include "SimVisitor.h"

using namespace opencover;

SimVisitor::SimVisitor()
    : osg::NodeVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN)
{
}
SimVisitor::~SimVisitor()
{
}

void SimVisitor::apply(osg::Group &node)
{
    osg::Referenced *data = node.getUserData();
    SimReference *simRef;
    if ((simRef = dynamic_cast<SimReference *>(data)))
    {
        std::cout << "?????????????????Der Knoten " << node.getName() << " hat eine Simulation" << std::endl;
        printf("-------apply %p %p\n", &node, data);
    }
    traverse(node);
}
//-----------------------------------------------------------------------------------------------
Simul::Simul()
{
}
Simul::~Simul()
{
}

/*std::string Simul::hasSim ( osg::Group *node )
{

    osg::Referenced *data = node->getUserData();

    SimReference *simRef;
    if ( ( simRef = dynamic_cast<SimReference *> ( data ) ) )
    {
        std::cout<<"***Der Knoten "<<node->getName() <<" hat eine Simulation ***"<<node<<std::endl;
        std::cout<<"Userdata: "<<simRef->getSimName() <<std::endl;
        return simRef->getSimName();
    }
    else
    {
        return "";
    }

}*/
