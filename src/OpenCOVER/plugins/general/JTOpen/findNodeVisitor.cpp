/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "findNodeVisitor.h"

// Default constructor - initialize searchForName to "" and
// set the traversal mode to TRAVERSE_ALL_CHILDREN
findNodeVisitor::findNodeVisitor()
    : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
    , searchForName()
{
}

// Constructor that accepts string argument
// Initializes searchForName to user string
// set the traversal mode to TRAVERSE_ALL_CHILDREN
findNodeVisitor::findNodeVisitor(const std::string &searchName)
    : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
    , searchForName(searchName)
{
}

// The 'apply' method for 'node' type instances.
// Compare the 'searchForName' data member against the node's name.
// If the strings match, add this node to our list
void findNodeVisitor::apply(osg::Node &searchNode)
{
    if (searchNode.getName() == searchForName)
    {
        foundNodeList.push_back(&searchNode);
    }
    traverse(searchNode);
}

// Set the searchForName to user-defined string
void findNodeVisitor::setNameToFind(const std::string &searchName)
{
    searchForName = searchName;
    foundNodeList.clear();
}
// Throw-away code:
//
//if ( strstr(searchNode.getName().c_str(), searchForName.c_str()) ) {
//   foundNodeList.push_back(&searchNode);
//}
