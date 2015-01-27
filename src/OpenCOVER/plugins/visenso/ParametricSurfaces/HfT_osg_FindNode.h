/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef HFT_OSG_FINDNODE_H
#define HFT_OSG_FINDNODE_H

#include <osg/NodeVisitor>
#include <osg/Node>
#include <list>

class HfT_osg_FindNode : public osg::NodeVisitor
{
public:
    // Default constructor - initialize searchForName to ""
    HfT_osg_FindNode();
    HfT_osg_FindNode(const std::string &searchName, bool select);

    ~HfT_osg_FindNode();

    // The 'apply' method for 'node' type instances.
    // Compare the 'searchForName' data member against the node's name.
    // If the strings match, add this node to  list

    virtual void apply(osg::Node &searchNode);

    // Setter
    void setNameToFind(const std::string &searchName);
    void setSelect(bool select);

    // Return a pointer to the first node in the list
    osg::Node *getFirstNode();

    // return a reference to the list of nodes we found
    std::list<osg::Node *> getNodeList();

private:
    // the name we are looking for
    std::string m_searchForName;
    bool m_searchSelect;
    // List of nodes with names that match the searchForName string
    std::list<osg::Node *> m_NodeList;
};
#endif
