/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PLMXMLSIMVISITOR
#define PLMXMLSIMVISITOR

#include <osg/Group>
#include <osg/NodeVisitor>

class PLMXMLSimVisitor : public osg::NodeVisitor
{

public:
    PLMXMLSimVisitor(opencover::coVRPlugin *plugin, osg::Node *node, const char *targetNode);
    virtual ~PLMXMLSimVisitor();

    virtual void apply(osg::Group &node);

private:
    opencover::coVRPlugin *plugin;
    osg::Node *simNode;
    const char *target;
    map<osg::Node *, osg::Node *> CAD_SIM_Node;
};
class PLMXMLCadVisitor : public osg::NodeVisitor
{

public:
    PLMXMLCadVisitor(opencover::coVRPlugin *plugin);
    virtual ~PLMXMLCadVisitor(){};

    virtual void apply(osg::Group &node);

private:
    opencover::coVRPlugin *plugin;
};
#endif
