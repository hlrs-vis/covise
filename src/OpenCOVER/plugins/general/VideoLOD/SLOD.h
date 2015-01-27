/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- OpenSceneGraph - Copyright (C) 1998-2006 Robert Osfield 
 *
 * This library is open source and may be redistributed and/or modified under  
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or 
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * OpenSceneGraph Public License for more details.
*/

#ifndef OSG_SLOD
#define OSG_SLOD 1

#include <osg/PagedLOD>
#include <osg/Texture2D>
#include <osg/Geometry>
#include <osgDB/ReaderWriter>

namespace osg
{

/** SLOD.
*/
class SLOD : public PagedLOD
{
public:
    SLOD();

    /** Copy constructor using CopyOp to manage deep vs shallow copy.*/
    SLOD(const PagedLOD &, const CopyOp &copyop = CopyOp::SHALLOW_COPY);

    META_Node(osg, SLOD);

    virtual void traverse(NodeVisitor &nv);

    void setDatabasePath(const std::string &path);
    void init(Node *master, int mmlevel, char *filename, int parentpath, vector<Node *> *parentlist, StateSet *state, Group *p);
    bool setfile(char *name, int mmlevel, int sync);
    bool setTile(char *name, int sync);
    void pause(int sync);
    float getarea();

    ref_ptr<Node> loaded;
    int loadedsync;
    Group *parent;

    friend class VideoLOD;

protected:
    float farea(float *points);

    osgDB::ReaderWriter::Options *ops;
    Node *_master;
    int _mmlevel;
    int _parentpath;
    int tilemode;
    StateSet *_state;
    StateSet *_state2;
    string end;
    string end2;
    int isEndTile;
    int endlevel;
    vector<Node *> _parentlist;
    int _sync;
    int _sync2;

    virtual ~SLOD()
    {
    }
};
}

#endif
