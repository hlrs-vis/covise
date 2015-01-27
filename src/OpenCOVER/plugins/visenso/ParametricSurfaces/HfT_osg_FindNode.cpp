/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "HfT_osg_FindNode.h"

using namespace osg;

HfT_osg_FindNode::HfT_osg_FindNode()
    : NodeVisitor(TRAVERSE_ALL_CHILDREN)
{
    setSelect(false);
    setNameToFind("");
    std::list<osg::Node *> m_NodeList;
}
HfT_osg_FindNode::HfT_osg_FindNode(const std::string &searchName, bool select)
    : NodeVisitor(TRAVERSE_ALL_CHILDREN)
{
    setSelect(select);
    setNameToFind(searchName);
    std::list<osg::Node *> m_NodeList;
}
HfT_osg_FindNode::~HfT_osg_FindNode()
{
    // delete m_NodeList;
}
void HfT_osg_FindNode::apply(osg::Node &searchNode)
{
    if ((m_searchForName != "") && (m_searchForName == searchNode.getName()))
    {
        m_NodeList.push_back(&searchNode);
    }
    traverse(searchNode);
}

void HfT_osg_FindNode::setNameToFind(const std::string &searchName)
{
    m_searchForName = searchName;
}
void HfT_osg_FindNode::setSelect(bool select)
{
    m_searchSelect = select;
}
std::list<osg::Node *> HfT_osg_FindNode::getNodeList()
{
    return m_NodeList;
}
osg::Node *HfT_osg_FindNode::getFirstNode()
{
    std::list<osg::Node *>::iterator it;

    if (!m_NodeList.empty())
    {
        it = m_NodeList.begin();
        return *it;
    }
    else
        return 0L;
}
