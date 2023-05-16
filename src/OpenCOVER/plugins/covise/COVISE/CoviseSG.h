/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! \file
 \brief  handle adding COVISE objects to scene graph

 */

#ifndef COVISE_SG_H
#define COVISE_SG_H

#include <string>
#include <map>

#include <osg/Node>
#include <osg/Group>

#include <cover/coVRLabel.h>
#include <cover/RenderObject.h>

namespace opencover
{
class coVRPlugin;
}

class CoviseSG
{
 public:
     CoviseSG(opencover::coVRPlugin *plugin);
     ~CoviseSG();

     typedef std::map<std::string, osg::Node *> NodeList;
     typedef std::map<std::string, opencover::coVRLabel *> LabelList;
     typedef std::map<std::string, std::string> FileList;

     void addNode(osg::Node *node, osg::Group *parent, opencover::RenderObject *ro);
     void addNode(osg::Node *node, const char *parentName, opencover::RenderObject *ro);
     void deleteNode(const char *nodeName, bool isGroup);
     osg::Node *findNode(const std::string &name);

     // attach a node to another (the attached node will be deleted with the other node)
     void attachNode(const char *attacheeName, osg::Node *attached, const char *filename = nullptr);
     void attachLabel(const char *attacheeName, const char *label);

 private:
    bool sgDebug_; /// scenegraph debug prints
    const char *hostName_;

    NodeList m_addedNodeList, m_attachedNodeList;
    LabelList m_attachedLabelList;
    FileList m_attachedFileList;
    opencover::coVRPlugin *m_plugin = nullptr;
};
#endif
