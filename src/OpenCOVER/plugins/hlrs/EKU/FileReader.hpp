/* This file is part of COVISE.
 *

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#pragma once

#include <stdio.h>
#include <string.h>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include<osg/Node>

// Derive a class from NodeVisitor to find a node with a
// specific name.
class FindNamedNode : public osg::NodeVisitor
{
public:
    FindNamedNode( const std::string& name,std::vector<osg::Vec3> *pos )
      : osg::NodeVisitor( // Traverse all children.
                osg::NodeVisitor::TRAVERSE_ALL_CHILDREN ),
        _name( name ),_pos(pos)
    {
       // _pos = new osg::Vec3Array; //NOTE: right way to make new here?
    }

    // This method gets called for every node in the scene
    //   graph. Check each node to see if its name matches
    //   out target. If so, return position
    virtual void apply( osg::Node& node )
    {
        if(node.getName().find(_name) != std::string::npos)
        {
            _node = &node;
            const osg::Vec3 posInWorld = _node->getBound().center() * osg::computeLocalToWorld(_node->getParentalNodePaths()[0]);
            _pos->push_back(posInWorld);
            std::cout<<node.getName()<<": "<<posInWorld.x()<<"|"<<posInWorld.y()<<"|"<<posInWorld.z()<<std::endl;

        }// Keep traversing the rest of the scene graph.
        traverse( node );
    }

    osg::Node* getNode() { return _node.get(); }
  //  osg::Vec3Array* getPos(){return _pos;}

protected:
    std::string _name;
    osg::ref_ptr<osg::Node> _node;
    std::vector<osg::Vec3> *_pos=nullptr;
};



class FileReaderWriter
{
private:
     FindNamedNode *fnn;

public:
    osg::ref_ptr<osg::Node> scene;
    FileReaderWriter();
    osg::Vec3Array getPossibleCamPos();


};


