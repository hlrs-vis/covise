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

#include <cover/coVRPluginSupport.h>
#include "SLOD.h"
#include <osg/CullStack>
#include <osg/Notify>
#include <iostream>
#include <algorithm>
#include <string>
#include <osgDB/ReadFile>
#include <CUI.h>

#include "VideoLOD.h"

using namespace std;
using namespace osg;
using namespace osgDB;
using namespace cui;

#define LIMIT -50.0

float SLOD::farea(float *points)
{
    int maxxindex = 0, minxindex = 0, maxyindex = 0, minyindex = 0;
    float minx, maxx, miny, maxy, s1, s2, offset = 0.0f;

    int index;

    if ((points[0] >= points[2]) && (points[0] >= points[6]))
    {
        if (points[7] > points[3])
        {
            float temp;
            temp = points[2];
            points[2] = points[6];
            points[6] = temp;
            temp = points[3];
            points[3] = points[7];
            points[7] = temp;
        }
    }
    else if ((points[2] >= points[4]) && (points[2] >= points[0]))
    {
        if (points[1] > points[5])
        {
            float temp;
            temp = points[4];
            points[4] = points[0];
            points[0] = temp;
            temp = points[5];
            points[5] = points[1];
            points[1] = temp;
        }
    }
    else if ((points[4] >= points[6]) && (points[4] >= points[2]))
    {
        if (points[3] > points[7])
        {
            float temp;
            temp = points[6];
            points[6] = points[2];
            points[2] = temp;
            temp = points[7];
            points[7] = points[3];
            points[3] = temp;
        }
    }
    else if ((points[6] >= points[0]) && (points[6] >= points[4]))
    {
        if (points[5] > points[1])
        {
            float temp;
            temp = points[0];
            points[0] = points[4];
            points[4] = temp;
            temp = points[1];
            points[1] = points[5];
            points[5] = temp;
        }
    }

    //cerr << points[0] << " " << points[1] << " " << points[2] << " " << points[3] << " " << points[4] << " " << points[5] << " " << points[6] << " " << points[7] << "\n";

    maxx = minx = points[0];
    maxy = miny = points[1];

    for (int i = 1; i < 4; i++)
    {
        if (points[2 * i] > maxx)
        {
            maxx = points[2 * i];
            maxxindex = i;
        }

        if (points[2 * i] < minx)
        {
            minx = points[(2 * i)];
            minxindex = i;
        }
        if (points[(2 * i) + 1] > maxy)
        {
            maxy = points[(2 * i) + 1];
            maxyindex = i;
        }

        if (points[(2 * i) + 1] < miny)
        {
            miny = points[(2 * i) + 1];
            minyindex = i;
        }
    }

    //if((miny > (float)localhalfh) || maxy < -((float)localhalfh) || minx > (float)localhalfw || maxx < -((float)localhalfw)  )
    //return -1.0f;

    if ((maxy == miny) || (maxx == minx))
        return 0.0f;

    //cerr << maxy << " " << miny << " " << maxx << " " << minx << "\n";
    for (int i = 0; i < 4; i++)
    {
        /*if((maxxindex != i) && (minxindex != i) && (maxyindex != i) && (minyindex != i))
    {
      //cerr << "i: " << i << "\n";
      index = i - 1;
      if(index < 0)
        index = 3;
      if(((maxy == points[(2 * index)+1]) && minx == points[2 * index]) || ((miny == points[(2 * index)+1]) && maxx == points[2 * index]))
      //if(((maxyindex == index) && minxindex == index) || ((minyindex == index) && maxxindex == index))
      {
          //s1 = fabs(points[(2*index)+1] - points[(2*i)+1]); 
          s1 = fabs(points[(2*index)] - points[(2*i)]);
      }
      else
      {
        //s1 = fabs(points[(2*index)] - points[(2*i)]); 
        s1 = fabs(points[(2*index)+1] - points[(2*i)+1]);
      }

      index = i + 1;
      if(index == 4)
        index = 0;
      if(((maxy == points[(2 * index)+1]) && minx == points[2 * index]) || ((miny == points[(2 * index)+1]) && maxx == points[2 * index]))
      //if(((maxyindex == index) && minxindex == index) || ((minyindex == index) && maxxindex == index))
        s2 = fabs(points[(2*index)+1] - points[(2*i)+1]); 
        //s2 = fabs(points[(2*index)] - points[(2*i)]);
      else
        s2 = fabs(points[(2*index)] - points[(2*i)]); 
        //s2 = fabs(points[(2*index)+1] - points[(2*i)+1]);

      offset += (s1 * s2);
    }*/
        //if((maxxindex != i) && (minxindex != i) && (maxyindex != i) && (minyindex != i))

        if ((maxy != points[(2 * i) + 1]) && (miny != points[(2 * i) + 1]) && (maxx != points[2 * i]) && (minx != points[2 * i]))
        {
            //cerr << "i: " << i << "\n";
            index = i - 1;
            if (index < 0)
                index = 3;
            if (((maxy == points[(2 * index) + 1]) && !(minx == points[2 * index])) || ((miny == points[(2 * index) + 1]) && !(maxx == points[2 * index])))
            //if(((maxyindex == index) && minxindex == index) || ((minyindex == index) && maxxindex == index))
            {
                s1 = fabs(points[(2 * index) + 1] - points[(2 * i) + 1]);
                //s1 = fabs(points[(2*index)] - points[(2*i)]);
                index = i + 1;
                if (index == 4)
                    index = 0;
                s2 = fabs(points[(2 * index)] - points[(2 * i)]);
            }
            else
            {
                s1 = fabs(points[(2 * index)] - points[(2 * i)]);
                //s1 = fabs(points[(2*index)+1] - points[(2*i)+1]);
                index = i + 1;
                if (index == 4)
                    index = 0;
                s2 = fabs(points[(2 * index) + 1] - points[(2 * i) + 1]);
            }

            /*index = i + 1;
      if(index == 4)
        index = 0;
      if(((maxy == points[(2 * index)+1]) && !(minx == points[2 * index])) || ((miny == points[(2 * index)+1]) && !(maxx == points[2 * index])))
      //if(((maxyindex == index) && minxindex == index) || ((minyindex == index) && maxxindex == index))
        //s2 = fabs(points[(2*index)+1] - points[(2*i)+1]); 
        s2 = fabs(points[(2*index)] - points[(2*i)]);
      else
        //s2 = fabs(points[(2*index)] - points[(2*i)]); 
        s2 = fabs(points[(2*index)+1] - points[(2*i)+1]);*/

            offset += (s1 * s2);
        }
    }

    //cerr << "offset: " << offset << "\n";

    s1 = fabs(points[6] - points[0]);
    s2 = fabs(points[7] - points[1]);
    //cerr << "s1: " << s1 << " s2: " << s2 << "\n";

    offset += ((s1 * s2) / 2.0);

    //cerr << "offset: " << offset << "\n";

    s1 = fabs(points[0] - points[2]);
    s2 = fabs(points[1] - points[3]);
    //cerr << "s1: " << s1 << " s2: " << s2 << "\n";
    offset += ((s1 * s2) / 2.0);

    //cerr << "offset: " << offset << "\n";

    s1 = fabs(points[2] - points[4]);
    s2 = fabs(points[3] - points[5]);
    //cerr << "s1: " << s1 << " s2: " << s2 << "\n";
    offset += ((s1 * s2) / 2.0);

    //cerr << "offset: " << offset << "\n";

    s1 = fabs(points[4] - points[6]);
    s2 = fabs(points[5] - points[7]);
    //cerr << "s1: " << s1 << " s2: " << s2 << "\n";
    offset += ((s1 * s2) / 2.0);

    //cerr << "offset: " << offset << "\n\n";
    return fabs(((maxx - minx) * (maxy - miny)) - offset);
}

SLOD::SLOD()
{
    _frameNumberOfLastTraversal = 0;
    _centerMode = USER_DEFINED_CENTER;
    _radius = -1;
    _numChildrenThatCannotBeExpired = 0;
}

SLOD::SLOD(const PagedLOD &plod, const CopyOp &copyop)
    : PagedLOD(plod, copyop)
{
    //cerr << "range: "   << _rangeList.size() << "\n";
}

void SLOD::pause(int sync)
{
    if (_mmlevel == 0)
    {
        _sync = sync;
    }
    else
    {
        ((SLOD *)(_parentlist.at(0)))->pause(sync);
    }
}

void SLOD::init(Node *master, int mmlevel, char *filename, int parentpath, vector<Node *> *parentlist, StateSet *state, Group *p)
{
    //cerr << "filename: " << filename << "\n";
    _master = master;
    _mmlevel = mmlevel;
    _parentpath = parentpath;
    parentlist->push_back(this);
    _state = state;
    isEndTile = 0;
    _sync = -1;
    _sync2 = -1;
    isEndTile = 0;
    loadedsync = -1;
    parent = p;
    int listindex;

    if (mmlevel == 0)
        _sync = 1;

    //cerr << "range: "   << _rangeList.size() << "\n";

    char *temp = (char *)filename + (string(filename).size() - 8);
    int count = 0;
    for (;; temp--)
    {
        if ((*temp) == '_')
            count++;
        if (count == 4)
            break;
    }

    end = string(temp);

    //cerr << "Checkpoint1: " << end << "\n";

    for (listindex = 0; listindex < _rangeList.size(); listindex++)
    {
        if (_perRangeDataList[listindex]._filename != string(""))
            break;
    }

    cerr << "Loading file: " << (_databasePath + _perRangeDataList[listindex]._filename) << "\n";

    //  (this->getChild(0))->setStateSet(_state);

    StateSet *stateset = new StateSet();
    stateset->setRenderBinDetails((mmlevel + 4), "RenderBin");
    stateset->setMode(GL_DEPTH_TEST, 0);
    stateset->setMode(GL_LIGHTING, StateAttribute::OFF);

    //cerr <<_databasePath << _perRangeDataList[listindex]._filename << "\n";

    ops = new ReaderWriter::Options();

    ops->setObjectCacheHint(ReaderWriter::Options::CACHE_NONE);

    ref_ptr<osg::Node> nodeRefPtr1 = osgDB::readNodeFile((_databasePath + _perRangeDataList[listindex]._filename).c_str(), ops);
    Node *root = nodeRefPtr1.get();
    if (strcmp((root)->className(), string("Group").c_str()))
    {
        cerr << "Root Node not of type Group, of type " << (nodeRefPtr1.get())->className() << "\n";
    }

    this->addChild(root);

    //cerr << "Checkpoint2\n";

    for (int j = 0; j < ((Group *)root)->getNumChildren(); j++)
    {
        Node *tempnode = ((Group *)root)->getChild(j);
        if (strcmp((tempnode)->className(), string("PagedLOD").c_str()))
        {
            //cerr << "Child Node not of type PagedLOD, of type " << (tempnode)->className() << "\n";

            isEndTile = 1;
            endlevel = mmlevel + 1;
            //_sync2 = 1;
            _parentlist = *parentlist;
            (((VideoLOD *)master)->treenodes).push_back(this);
            (((VideoLOD *)master)->numoftiles)++;
            ((VideoLOD *)master)->maxlevel = endlevel;
            Geometry *geo = (Geometry *)((Geode *)(((Group *)tempnode)->getChild(0)))->getDrawable(0);
            BoundingBox b = geo->getBound();
            double d1, d2;
            d1 = fabs((b.corner(0) - b.corner(1)).length());
            d2 = fabs((b.corner(0) - b.corner(2)).length());

            if (((Group *)root)->getNumChildren() == 2)
            {
                Geometry *gp = (Geometry *)((Geode *)((Group *)this->getChild(0))->getChild(0))->getDrawable(0);
                BoundingBox bp = gp->getBound();
                double dp1 = fabs((bp.corner(0) - bp.corner(1)).length());
                //double dp2 = fabs((bp.corner(0) - bp.corner(2)).length());
                if (dp1 == d1)
                {
                    tilemode = 0;
                }
                else
                {
                    tilemode = 1;
                }
            }

            //cerr << "d1: " << d1 << " d2: " << d2 << "\n";
            //Texture2D* text = (Texture2D*)(geo->getStateSet())->getTextureAttribute(0, StateAttribute::TEXTURE);
            //Image* cimg = (text)->getImage();
            //cerr << "s: " << cimg->s() << " t: " << cimg->t() << " children: " << ((Group*)root)->getNumChildren() << "\n";
            //((VideoLOD*)master)->tilearea = cimg->s() * cimg->t() * ((Group*)root)->getNumChildren();
            ((VideoLOD *)master)->tilearea = (((int)(d1 * d2)) * ((Group *)root)->getNumChildren());
            _state2 = stateset;
            //      (this->getChild(1))->setStateSet(_state2);
            temp = (char *)(_perRangeDataList[listindex]._filename).c_str() + (((_perRangeDataList[listindex]._filename).size()) - 8);
            count = 0;
            for (;; temp--)
            {
                if ((*temp) == '_')
                    count++;
                if (count == 4)
                    break;
            }
            end2 = string(temp);
            break;
        }
        else
        {
            SLOD *tempSLOD = new SLOD(*((PagedLOD *)tempnode), *(new CopyOp()));
            ((Group *)tempnode)->removeChildren(0, ((Group *)tempnode)->getNumChildren());
            ((Group *)root)->replaceChild(tempnode, tempSLOD);
            ((PagedLOD *)tempSLOD)->setDatabasePath(_databasePath);
            tempSLOD->init(master, mmlevel + 1, (char *)(_perRangeDataList[listindex]._filename).c_str(), j, new vector<Node *>(*parentlist), stateset, this);
        }
    }

    cerr << "End of SLOD init\n";
}

bool SLOD::setfile(char *name, int mmlevel, int sync)
{
    if (mmlevel == endlevel)
    {
        //cerr << "SLOD setfile: " << (string(name) + end2).c_str() << "\n";
        Node *tempnode = osgDB::readNodeFile((string(name) + end2).c_str(), ops);
        if (tempnode == NULL)
        {
            cerr << "Unable to open file " << (string(name) + end2) << "\n";
            return false;
        }
        //cerr << "after load.\n";
        //    tempnode->setStateSet(_state2);
        this->replaceChild(this->getChild(1), tempnode);
        //cerr << "after child replace.\n";
        if (_state2 == NULL)
            cerr << "Null stateset.\n";

        //cerr << "after stateset.\n";
        _sync2 = sync;
    }
    else
    {
        if (!((SLOD *)(_parentlist.at(mmlevel)))->setTile(name, sync))
        {
            return false;
        }
    }
    return true;
}

bool SLOD::setTile(char *name, int sync)
{
    if (_sync == sync)
        return true;

    Node *root;
    if (_mmlevel > 0)
    {
        if (((SLOD *)parent)->loadedsync != sync)
        {
            //if((((SLOD*)parent)->loaded).valid())
            //(((SLOD*)parent)->loaded).release();
            ((SLOD *)parent)->loaded = osgDB::readNodeFile((string(name) + end).c_str(), ops);
            if (((SLOD *)parent)->loaded == NULL)
            {
                cerr << "Unable to open file " << (string(name) + end) << "\n";
                return false;
            }
        }
        ((SLOD *)parent)->loadedsync = sync;
        root = (((SLOD *)parent)->loaded).get();
    }
    else
    {
        if (((VideoLOD *)parent)->loadedsync != sync)
        {
            //if((((VideoLOD*)parent)->loaded).valid())
            //(((VideoLOD*)parent)->loaded).release();
            ((VideoLOD *)parent)->loaded = osgDB::readNodeFile((string(name) + end).c_str(), ops);
            if (((VideoLOD *)parent)->loaded == NULL)
            {
                cerr << "Unable to open file " << (string(name) + end) << "\n";
                return false;
            }
        }
        ((VideoLOD *)parent)->loadedsync = sync;
        root = (((VideoLOD *)parent)->loaded).get();
    }

    //ref_ptr<osg::Node> nodeRefPtr1;

    //nodeRefPtr1 = osgDB::readNodeFile((string(name) + end).c_str());
    //Node * root = nodeRefPtr1.get();
    Node *tempnode = ((Group *)((Group *)root)->getChild(_parentpath))->getChild(0);
    //  tempnode->setStateSet(_state);
    this->replaceChild(this->getChild(0), tempnode);
    ((Group *)((Group *)root)->getChild(_parentpath))->removeChild(tempnode);

    _sync = sync;
    return true;
}

void SLOD::traverse(NodeVisitor &nv)
{
    //if(firstrun)
    //{
    //init();
    //firstrun = 0;
    //}

    switch (nv.getTraversalMode())
    {
    case (NodeVisitor::TRAVERSE_ALL_CHILDREN):
        std::for_each(_children.begin(), _children.end(), NodeAcceptOp(nv));
        break;
    case (NodeVisitor::TRAVERSE_ACTIVE_CHILDREN):
    {
        if (isEndTile)
        {
            if (_sync == ((VideoLOD *)_master)->sync)
            {
                _children[0]->accept(nv);
                //((Group*)(_children[1]).get())->releaseGLObjects();
            }
            else if (_sync2 == ((VideoLOD *)_master)->sync)
            {
                _children[1]->accept(nv);
                //((Group*)(_children[0]).get())->releaseGLObjects();
            }
            //else
            //{
            //((Group*)(_children[0]).get())->releaseGLObjects();
            //((Group*)(_children[1]).get())->releaseGLObjects();
            //}
        }
        else
        {
            //if(loaded.valid())
            //loaded.release();

            if (_sync == ((VideoLOD *)_master)->sync)
            {
                _children[0]->accept(nv);
            }
            //else
            //{
            //((Group*)(_children[0]).get())->releaseGLObjects();
            //}

            _children[1]->accept(nv);
        }
        break;
    }
    default:
        break;
    }
}

float SLOD::getarea()
{
    float input[8];
    float zarray[4];
    bool valid;
    int numvalid = 0;
    float retarea = 0;
    int up, down, left, right, out;
    int isout = 0;

    //cerr << "In getarea\n";
    int localhalfw = ((VideoLOD *)_master)->halfw;
    int localhalfh = ((VideoLOD *)_master)->halfh;
    //cerr << "children: " << ((Group*)(this->getChild(1)))->getNumChildren() << " " << ((VideoLOD*)_master)->numOfCameras << "\n";
    for (int i = 0; i < ((VideoLOD *)_master)->numOfCameras; i++)
    {
        Vec3 testvec;
        out = up = down = left = right = 0;
        valid = false;
        Matrix localview = ((VideoLOD *)_master)->mview[i];
        Matrix localpro = ((VideoLOD *)_master)->pro[i];
        Matrix locall2r = ((VideoLOD *)_master)->l2r;
        Matrix bigmat = ((VideoLOD *)_master)->l2r * ((VideoLOD *)_master)->mview[i] * ((VideoLOD *)_master)->pro[i];
        int oldisout = isout;
        //cerr << "children: " << ((Group*)(this->getChild(0)))->getNumChildren() << "\n";
        if (((Group *)(this->getChild(1)))->getNumChildren() == 1)
        {

            //cerr << "in right part\n";
            Vec3 tempvec;
            Geometry *geo1 = (Geometry *)(((Geode *)((Group *)((Group *)(this->getChild(1)))->getChild(0))->getChild(0)))->getDrawable(0);

            BoundingBox b = geo1->getBound();
            tempvec = b.corner(0);
            //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(0)))->getMatrix() * bigmat;
            tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(0)))->getMatrix() * locall2r * localview;
            if (tempvec.z() > LIMIT)
            {
                if (tempvec.z() > 0)
                {
                    zarray[0] = 1.0;
                }
                else
                {
                    zarray[0] = 0.0;
                }
                isout = 1;
                tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
            }
            else
            {
                zarray[0] = 0.0;
            }

            tempvec = tempvec * localpro;
            if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[0] == 0.0)
            {

                valid = true;
            }

            //tempvec = tempvec * ((VideoLOD*)_master)->win[i];
            //input[0] = tempvec.x();
            //input[1] = tempvec.y();
            input[0] = (tempvec.x()) * (float)localhalfw;
            input[1] = (tempvec.y()) * (float)localhalfh;
            //zarray[0] = tempvec.z();

            tempvec = b.corner(1);
            //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(0)))->getMatrix() * bigmat;
            tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(0)))->getMatrix() * locall2r * localview;
            if (tempvec.z() > LIMIT)
            {
                if (tempvec.z() > 0)
                {
                    zarray[1] = 1.0;
                }
                else
                {
                    zarray[1] = 0.0;
                }
                isout = 1;
                tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
            }
            else
            {
                zarray[1] = 0.0;
            }

            tempvec = tempvec * localpro;
            if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[1] == 0.0)
            {

                valid = true;
            }

            //tempvec = tempvec * ((VideoLOD*)_master)->win[i];
            //input[2] = tempvec.x();
            //input[3] = tempvec.y();
            input[2] = (tempvec.x()) * (float)localhalfw;
            input[3] = (tempvec.y()) * (float)localhalfh;
            //zarray[1] = tempvec.z();

            tempvec = b.corner(3);
            //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(0)))->getMatrix() * bigmat;
            tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(0)))->getMatrix() * locall2r * localview;
            if (tempvec.z() > LIMIT)
            {
                if (tempvec.z() > 0)
                {
                    zarray[2] = 1.0;
                }
                else
                {
                    zarray[2] = 0.0;
                }
                isout = 1;
                tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
            }
            else
            {
                zarray[2] = 0.0;
            }

            tempvec = tempvec * localpro;
            if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[2] == 0.0)
            {

                valid = true;
            }

            //tempvec = tempvec * ((VideoLOD*)_master)->win[i];
            //input[4] = tempvec.x();
            //input[5] = tempvec.y();
            input[4] = (tempvec.x()) * (float)localhalfw;
            input[5] = (tempvec.y()) * (float)localhalfh;
            //zarray[2] = tempvec.z();

            tempvec = b.corner(2);
            //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(0)))->getMatrix() * bigmat;
            tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(0)))->getMatrix() * locall2r * localview;
            if (tempvec.z() > LIMIT)
            {
                if (tempvec.z() > 0)
                {
                    zarray[3] = 1.0;
                }
                else
                {
                    zarray[3] = 0.0;
                }
                isout = 1;
                tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
            }
            else
            {
                zarray[3] = 0.0;
            }

            tempvec = tempvec * localpro;
            if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[3] == 0.0)
            {

                valid = true;
            }

            //tempvec = tempvec * ((VideoLOD*)_master)->win[i];
            //input[6] = tempvec.x();
            //input[7] = tempvec.y();
            input[6] = (tempvec.x()) * (float)localhalfw;
            input[7] = (tempvec.y()) * (float)localhalfh;
            //zarray[3] = tempvec.z();

            //cerr << input[0] << " " << input[1] << " " << input[2] << " " << input[3] << " " << input[4] << " " << input[5] << " " << input[6] << " " << input[7] << "\n";
            if (valid)
            {
                retarea += farea(input);
                numvalid++;
            }
            else
            {
                if ((zarray[0] + zarray[1] + zarray[2] + zarray[3]) == 4.0)
                {
                    isout = oldisout;
                    continue;
                }
                for (int k = 0; k < 4; k++)
                {
                    if (zarray[k] == 1.0 || zarray[(k + 1) % 4] == 1.0)
                        continue;
                    int i1 = (2 * k) % 8;
                    int i2 = (2 * (k + 1)) % 8;
                    if ((input[i1] < -(localhalfw) && input[i2] > -(localhalfw)) || (input[i2] < -(localhalfw) && input[i1] > -(localhalfw)))
                    {
                        float slope, b;
                        if (input[i1] == input[i2])
                            continue;

                        slope = (((float)input[i1 + 1]) - ((float)input[i2 + 1])) / (((float)input[i1]) - ((float)input[i2]));
                        b = ((float)input[i1 + 1]) - (slope * input[i1]);
                        int y = (int)((slope * (-(localhalfw))) + b);
                        //cerr << "1: " << input[i1] << "," << input[i1+1] << " 2: " << input[i2] << "," << input[i2+1] <<  " y: " << y << "\n";
                        if (y >= -(localhalfh) && y <= localhalfh)
                        {
                            //cerr << "turning valid\n";
                            valid = true;
                            break;
                        }
                        else if (y < -(localhalfh))
                        {
                            down = 1;
                        }
                        else
                        {
                            up = 1;
                        }
                    }

                    if ((input[i1] < localhalfw && input[i2] > localhalfw) || (input[i2] < localhalfw && input[i1] > localhalfw))
                    {
                        float slope, b;
                        if (input[i1] == input[i2])
                            continue;

                        slope = (((float)input[i1 + 1]) - ((float)input[i2 + 1])) / (((float)input[i1]) - ((float)input[i2]));
                        b = ((float)input[i1 + 1]) - (slope * input[i1]);
                        int y = (int)((slope * (localhalfw)) + b);
                        //cerr << "1: " << input[i1] << "," << input[i1+1] << " 2: " << input[i2] << "," << input[i2+1] <<  " y: " << y << "\n";
                        if (y >= -(localhalfh) && y <= localhalfh)
                        {
                            //cerr << "turning valid\n";
                            valid = true;
                            break;
                        }
                        else if (y < -(localhalfh))
                        {
                            down = 1;
                        }
                        else
                        {
                            up = 1;
                        }
                    }

                    if ((input[i1 + 1] < -(localhalfh) && input[i2 + 1] > -(localhalfh)) || (input[i2 + 1] < -(localhalfh) && input[i1 + 1] > -(localhalfh)))
                    {
                        float slope, b;
                        if (input[i1] == input[i2])
                        {
                            if (input[i1] >= -localhalfw && input[i1] <= localhalfw)
                            {
                                valid = true;
                                break;
                            }
                            continue;
                        }

                        slope = (((float)input[i1 + 1]) - ((float)input[i2 + 1])) / (((float)input[i1]) - ((float)input[i2]));
                        b = ((float)input[i1 + 1]) - (slope * input[i1]);
                        //int y =(int) ((slope * (-(localhalfw))) + b);
                        int x = (int)((-(localhalfh)-b) / slope);
                        //cerr << "1: " << input[i1] << "," << input[i1+1] << " 2: " << input[i2] << "," << input[i2+1] <<  " y: " << y << "\n";
                        if (x >= -(localhalfw) && x <= localhalfw)
                        {
                            //cerr << "turning valid\n";
                            valid = true;
                            break;
                        }
                        else if (x < -(localhalfw))
                        {
                            left = 1;
                        }
                        else
                        {
                            right = 1;
                        }
                    }

                    if ((input[i1 + 1] < (localhalfh) && input[i2 + 1] > (localhalfh)) || (input[i2 + 1] < (localhalfh) && input[i1 + 1] > (localhalfh)))
                    {
                        float slope, b;
                        if (input[i1] == input[i2])
                        {
                            if (input[i1] >= -localhalfw && input[i1] <= localhalfw)
                            {
                                valid = true;
                                break;
                            }
                            continue;
                        }

                        slope = (((float)input[i1 + 1]) - ((float)input[i2 + 1])) / (((float)input[i1]) - ((float)input[i2]));
                        b = ((float)input[i1 + 1]) - (slope * input[i1]);
                        //int y =(int) ((slope * (-(localhalfw))) + b);
                        int x = (int)(((localhalfh)-b) / slope);
                        //cerr << "1: " << input[i1] << "," << input[i1+1] << " 2: " << input[i2] << "," << input[i2+1] <<  " y: " << y << "\n";
                        if (x >= -(localhalfw) && x <= localhalfw)
                        {
                            //cerr << "turning valid\n";
                            valid = true;
                            break;
                        }
                        else if (x < -(localhalfw))
                        {
                            left = 1;
                        }
                        else
                        {
                            right = 1;
                        }
                    }
                }
                if (valid || (up && down) || (left && right))
                {
                    retarea += farea(input);
                    numvalid++;
                }
            }

            int count = ((int)zarray[0]) + ((int)zarray[1]) + ((int)zarray[2]) + ((int)zarray[3]);
            //cerr << "count: " << count << "\n";
            if (count > 0 && count < 4)
            {
                isout = 1;
            }
            //cerr << "area0: " << area << "\n";
        }
        else if (((Group *)(this->getChild(1)))->getNumChildren() == 4)
        {
            Vec3 tempvec;

            Geometry *geo1;
            BoundingBox b;

            geo1 = (Geometry *)((Geode *)((Group *)((Group *)(this->getChild(1)))->getChild(0))->getChild(0))->getDrawable(0);
            b = geo1->getBound();
            tempvec = b.corner(0);

            //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(0)))->getMatrix() * bigmat;
            tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(0)))->getMatrix() * locall2r * localview;
            if (tempvec.z() > LIMIT)
            {
                if (tempvec.z() > 0)
                {
                    zarray[0] = 1.0;
                }
                else
                {
                    zarray[0] = 0.0;
                }
                isout = 1;
                tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
            }
            else
            {
                zarray[0] = 0.0;
            }

            tempvec = tempvec * localpro;

            if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[0] == 0.0)
            {

                valid = true;
            }

            //tempvec = tempvec * ((VideoLOD*)_master)->win[i];
            //input[0] = tempvec.x();
            //input[1] = tempvec.y();
            //cerr << "\nx: " << testvec.x() << " y: " << testvec.y() << " z: " << testvec.z() << " " << testvec.valid() << " " << testvec.isNaN() <<  "\n";
            input[0] = (tempvec.x()) * (float)localhalfw;
            input[1] = (tempvec.y()) * (float)localhalfh;
            //zarray[0] = tempvec.z();
            //cerr << "\nx: " << input[0] << " y: " << input[1] << " z: " << tempvec.z() << " " << tempvec.valid() << " " << tempvec.isNaN() <<  "\n";

            geo1 = (Geometry *)((Geode *)((Group *)((Group *)(this->getChild(1)))->getChild(1))->getChild(0))->getDrawable(0);
            b = geo1->getBound();
            tempvec = b.corner(1);

            //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(1)))->getMatrix() * bigmat;
            tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(1)))->getMatrix() * locall2r * localview;
            if (tempvec.z() > LIMIT)
            {
                if (tempvec.z() > 0)
                {
                    zarray[1] = 1.0;
                }
                else
                {
                    zarray[1] = 0.0;
                }
                isout = 1;
                tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
            }
            else
            {
                zarray[1] = 0.0;
            }

            tempvec = tempvec * localpro;
            if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[1] == 0.0)
            {

                valid = true;
            }

            //tempvec = tempvec * ((VideoLOD*)_master)->win[i];
            //input[2] = tempvec.x();
            //input[3] = tempvec.y();
            //cerr << "x: " << testvec.x() << " y: " << testvec.y() << " z: " << testvec.z() << " " << testvec.valid() << " " << testvec.isNaN() <<  "\n";
            input[2] = (tempvec.x()) * (float)localhalfw;
            input[3] = (tempvec.y()) * (float)localhalfh;
            //zarray[1] = tempvec.z();
            //cerr << "x: " << input[2] << " y: " << input[3] << " z: " << tempvec.z() << " " << tempvec.valid() << " " << tempvec.isNaN() << "\n";

            geo1 = (Geometry *)((Geode *)((Group *)((Group *)(this->getChild(1)))->getChild(3))->getChild(0))->getDrawable(0);
            b = geo1->getBound();
            tempvec = b.corner(3);

            //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(3)))->getMatrix() * bigmat;
            tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(3)))->getMatrix() * locall2r * localview;
            if (tempvec.z() > LIMIT)
            {
                if (tempvec.z() > 0)
                {
                    zarray[2] = 1.0;
                }
                else
                {
                    zarray[2] = 0.0;
                }
                isout = 1;
                tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
            }
            else
            {
                zarray[2] = 0.0;
            }

            tempvec = tempvec * localpro;
            if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[2] == 0.0)
            {

                valid = true;
            }

            //tempvec = tempvec * ((VideoLOD*)_master)->win[i];
            //input[4] = tempvec.x();
            //input[5] = tempvec.y();
            //cerr << "x: " << testvec.x() << " y: " << testvec.y() << " z: " << testvec.z() << " " << testvec.valid() << " " << testvec.isNaN() <<  "\n";
            input[4] = (tempvec.x()) * (float)localhalfw;
            input[5] = (tempvec.y()) * (float)localhalfh;
            //zarray[2] = tempvec.z();
            //cerr << "x: " << input[4] << " y: " << input[5] << " z: " << tempvec.z() << " " << tempvec.valid() << " " << tempvec.isNaN() << "\n";

            geo1 = (Geometry *)((Geode *)((Group *)((Group *)(this->getChild(1)))->getChild(2))->getChild(0))->getDrawable(0);
            b = geo1->getBound();
            tempvec = b.corner(2);

            //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(2)))->getMatrix() * bigmat;
            tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(2)))->getMatrix() * locall2r * localview;
            if (tempvec.z() > LIMIT)
            {
                if (tempvec.z() > 0)
                {
                    zarray[3] = 1.0;
                }
                else
                {
                    zarray[3] = 0.0;
                }
                isout = 1;
                tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
            }
            else
            {
                zarray[3] = 0.0;
            }

            tempvec = tempvec * localpro;
            if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[3] == 0.0)
            {

                valid = true;
            }

            //tempvec = tempvec * ((VideoLOD*)_master)->win[i];
            //input[6] = tempvec.x();
            //input[7] = tempvec.y();
            //cerr << "x: " << testvec.x() << " y: " << testvec.y() << " z: " << testvec.z() << " " << testvec.valid() << " " << testvec.isNaN() <<  "\n";
            input[6] = (tempvec.x()) * (float)localhalfw;
            input[7] = (tempvec.y()) * (float)localhalfh;
            //zarray[3] = tempvec.z();
            //cerr << "x: " << input[6] << " y: " << input[7] << " z: " << tempvec.z() << " " << tempvec.valid() << " " << tempvec.isNaN() << "\n";

            //cerr << input[0] << " " << input[1] << " " << input[2] << " " << input[3] << " " << input[4] << " " << input[5] << " " << input[6] << " " << input[7] << "\n";
            if (valid)
            {
                retarea += farea(input);
                numvalid++;
            }
            else
            {
                if ((zarray[0] + zarray[1] + zarray[2] + zarray[3]) == 4.0)
                {
                    isout = oldisout;
                    continue;
                }
                for (int k = 0; k < 4; k++)
                {
                    if (zarray[k] == 1.0 || zarray[(k + 1) % 4] == 1.0)
                        continue;
                    int i1 = (2 * k) % 8;
                    int i2 = (2 * (k + 1)) % 8;
                    if ((input[i1] < -(localhalfw) && input[i2] > -(localhalfw)) || (input[i2] < -(localhalfw) && input[i1] > -(localhalfw)))
                    {
                        float slope, b;
                        if (input[i1] == input[i2])
                            continue;

                        slope = (((float)input[i1 + 1]) - ((float)input[i2 + 1])) / (((float)input[i1]) - ((float)input[i2]));
                        b = ((float)input[i1 + 1]) - (slope * input[i1]);
                        int y = (int)((slope * (-(localhalfw))) + b);
                        //cerr << "1: " << input[i1] << "," << input[i1+1] << " 2: " << input[i2] << "," << input[i2+1] <<  " y: " << y << "\n";
                        if (y >= -(localhalfh) && y <= localhalfh)
                        {
                            //cerr << "turning valid\n";
                            valid = true;
                            break;
                        }
                        else if (y < -(localhalfh))
                        {
                            down = 1;
                        }
                        else
                        {
                            up = 1;
                        }
                    }

                    if ((input[i1] < localhalfw && input[i2] > localhalfw) || (input[i2] < localhalfw && input[i1] > localhalfw))
                    {
                        float slope, b;
                        if (input[i1] == input[i2])
                            continue;

                        slope = (((float)input[i1 + 1]) - ((float)input[i2 + 1])) / (((float)input[i1]) - ((float)input[i2]));
                        b = ((float)input[i1 + 1]) - (slope * input[i1]);
                        int y = (int)((slope * (localhalfw)) + b);
                        //cerr << "1: " << input[i1] << "," << input[i1+1] << " 2: " << input[i2] << "," << input[i2+1] <<  " y: " << y << "\n";
                        if (y >= -(localhalfh) && y <= localhalfh)
                        {
                            //cerr << "turning valid\n";
                            valid = true;
                            break;
                        }
                        else if (y < -(localhalfh))
                        {
                            down = 1;
                        }
                        else
                        {
                            up = 1;
                        }
                    }

                    if ((input[i1 + 1] < -(localhalfh) && input[i2 + 1] > -(localhalfh)) || (input[i2 + 1] < -(localhalfh) && input[i1 + 1] > -(localhalfh)))
                    {
                        float slope, b;
                        if (input[i1] == input[i2])
                        {
                            if (input[i1] >= -localhalfw && input[i1] <= localhalfw)
                            {
                                valid = true;
                                break;
                            }
                            continue;
                        }

                        slope = (((float)input[i1 + 1]) - ((float)input[i2 + 1])) / (((float)input[i1]) - ((float)input[i2]));
                        b = ((float)input[i1 + 1]) - (slope * input[i1]);
                        //int y =(int) ((slope * (-(localhalfw))) + b);
                        int x = (int)((-(localhalfh)-b) / slope);
                        //cerr << "1: " << input[i1] << "," << input[i1+1] << " 2: " << input[i2] << "," << input[i2+1] <<  " y: " << y << "\n";
                        if (x >= -(localhalfw) && x <= localhalfw)
                        {
                            //cerr << "turning valid\n";
                            valid = true;
                            break;
                        }
                        else if (x < -(localhalfw))
                        {
                            left = 1;
                        }
                        else
                        {
                            right = 1;
                        }
                    }

                    if ((input[i1 + 1] < (localhalfh) && input[i2 + 1] > (localhalfh)) || (input[i2 + 1] < (localhalfh) && input[i1 + 1] > (localhalfh)))
                    {
                        float slope, b;
                        if (input[i1] == input[i2])
                        {
                            if (input[i1] >= -localhalfw && input[i1] <= localhalfw)
                            {
                                valid = true;
                                break;
                            }
                            continue;
                        }

                        slope = (((float)input[i1 + 1]) - ((float)input[i2 + 1])) / (((float)input[i1]) - ((float)input[i2]));
                        b = ((float)input[i1 + 1]) - (slope * input[i1]);
                        //int y =(int) ((slope * (-(localhalfw))) + b);
                        int x = (int)(((localhalfh)-b) / slope);
                        //cerr << "1: " << input[i1] << "," << input[i1+1] << " 2: " << input[i2] << "," << input[i2+1] <<  " y: " << y << "\n";
                        if (x >= -(localhalfw) && x <= localhalfw)
                        {
                            //cerr << "turning valid\n";
                            valid = true;
                            break;
                        }
                        else if (x < -(localhalfw))
                        {
                            left = 1;
                        }
                        else
                        {
                            right = 1;
                        }
                    }
                }
                if (valid || (up && down) || (left && right))
                {
                    retarea += farea(input);
                    numvalid++;
                }
            }
            //cerr << "area1: " << area << "\n";
            int count = ((int)zarray[0]) + ((int)zarray[1]) + ((int)zarray[2]) + ((int)zarray[3]);
            //cerr << "count: " << count << "\n";
            if (count > 0 && count < 4)
            {
                isout = 1;
            }
        }
        else if (((Group *)(this->getChild(1)))->getNumChildren() == 2)
        {
            Vec3 tempvec;

            Geometry *geo1;
            BoundingBox b;

            geo1 = (Geometry *)((Geode *)((Group *)((Group *)(this->getChild(1)))->getChild(0))->getChild(0))->getDrawable(0);
            b = geo1->getBound();
            if (tilemode)
            {
                tempvec = b.corner(0);
                //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(0)))->getMatrix() * bigmat;
                tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(0)))->getMatrix() * locall2r * localview;
                if (tempvec.z() > LIMIT)
                {
                    if (tempvec.z() > 0)
                    {
                        zarray[0] = 1.0;
                    }
                    else
                    {
                        zarray[0] = 0.0;
                    }
                    isout = 1;
                    tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
                }
                else
                {
                    zarray[0] = 0.0;
                }

                tempvec = tempvec * localpro;
                input[0] = (tempvec.x()) * (float)localhalfw;
                input[1] = (tempvec.y()) * (float)localhalfh;
                //zarray[0] = tempvec.z();
                if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[0] == 0.0)
                {

                    valid = true;
                }
            }
            else
            {
                tempvec = b.corner(2);
                //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(0)))->getMatrix() * bigmat;
                tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(0)))->getMatrix() * locall2r * localview;
                if (tempvec.z() > LIMIT)
                {
                    if (tempvec.z() > 0)
                    {
                        zarray[3] = 1.0;
                    }
                    else
                    {
                        zarray[3] = 0.0;
                    }
                    isout = 1;
                    tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
                }
                else
                {
                    zarray[3] = 0.0;
                }
                tempvec = tempvec * localpro;
                input[6] = (tempvec.x()) * (float)localhalfw;
                input[7] = (tempvec.y()) * (float)localhalfh;
                //zarray[3] = tempvec.z();
                if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[3] == 0.0)
                {

                    valid = true;
                }
            }
            //cerr << "\nx: " << tempvec.x() << " y: " << tempvec.y() << "\n";

            //tempvec = tempvec * ((VideoLOD*)_master)->win[i];
            //input[0] = tempvec.x();
            //input[1] = tempvec.y();

            //cerr << "\nx: " << input[0] << " y: " << input[1] << "\n";

            //geo1 = (Geometry*)((Geode*)((Group*)((Group*)(this->getChild(1)))->getChild(1))->getChild(0))->getDrawable(0);
            //b = geo1->getBound();
            if (tilemode)
            {
                tempvec = b.corner(2);
                //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(0)))->getMatrix() * bigmat;
                tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(0)))->getMatrix() * locall2r * localview;
                if (tempvec.z() > LIMIT)
                {
                    if (tempvec.z() > 0)
                    {
                        zarray[3] = 1.0;
                    }
                    else
                    {
                        zarray[3] = 0.0;
                    }
                    isout = 1;
                    tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
                }
                else
                {
                    zarray[3] = 0.0;
                }
                tempvec = tempvec * localpro;
                input[6] = (tempvec.x()) * (float)localhalfw;
                input[7] = (tempvec.y()) * (float)localhalfh;
                //zarray[3] = tempvec.z();
                if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[3] == 0.0)
                {

                    valid = true;
                }
            }
            else
            {
                tempvec = b.corner(3);
                //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(0)))->getMatrix() * bigmat;
                tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(0)))->getMatrix() * locall2r * localview;
                if (tempvec.z() > LIMIT)
                {
                    if (tempvec.z() > 0)
                    {
                        zarray[2] = 1.0;
                    }
                    else
                    {
                        zarray[2] = 0.0;
                    }
                    isout = 1;
                    tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
                }
                else
                {
                    zarray[2] = 0.0;
                }
                tempvec = tempvec * localpro;
                input[4] = (tempvec.x()) * (float)localhalfw;
                input[5] = (tempvec.y()) * (float)localhalfh;
                //zarray[2] = tempvec.z();
                if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[2] == 0.0)
                {

                    valid = true;
                }
            }
            //cerr << "x: " << tempvec.x() << " y: " << tempvec.y() << "\n";

            //tempvec = tempvec * ((VideoLOD*)_master)->win[i];
            //input[6] = tempvec.x();
            //input[7] = tempvec.y();

            //cerr << "x: " << input[6] << " y: " << input[7] << "\n";

            //Node* nptr = ((Group*)((Group*)this->getChild(1))->getChild(1))->getChild(0);
            //cerr << "node: " << nptr->className() << "\n";
            geo1 = (Geometry *)((Geode *)((Group *)((Group *)(this->getChild(1)))->getChild(1))->getChild(0))->getDrawable(0);
            b = geo1->getBound();
            if (tilemode)
            {
                tempvec = b.corner(1);
                //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(1)))->getMatrix() * bigmat;
                tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(1)))->getMatrix() * locall2r * localview;
                if (tempvec.z() > LIMIT)
                {
                    if (tempvec.z() > 0)
                    {
                        zarray[1] = 1.0;
                    }
                    else
                    {
                        zarray[1] = 0.0;
                    }
                    isout = 1;
                    tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
                }
                else
                {
                    zarray[1] = 0.0;
                }
                tempvec = tempvec * localpro;
                input[2] = (tempvec.x()) * (float)localhalfw;
                input[3] = (tempvec.y()) * (float)localhalfh;
                //zarray[1] = tempvec.z();
                if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[1] == 0.0)
                {

                    valid = true;
                }
            }
            else
            {
                tempvec = b.corner(0);
                //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(1)))->getMatrix() * bigmat;
                tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(1)))->getMatrix() * locall2r * localview;
                if (tempvec.z() > LIMIT)
                {
                    if (tempvec.z() > 0)
                    {
                        zarray[0] = 1.0;
                    }
                    else
                    {
                        zarray[0] = 0.0;
                    }
                    isout = 1;
                    tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
                }
                else
                {
                    zarray[0] = 0.0;
                }
                tempvec = tempvec * localpro;
                input[0] = (tempvec.x()) * (float)localhalfw;
                input[1] = (tempvec.y()) * (float)localhalfh;
                //zarray[0] = tempvec.z();
                if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[0] == 0.0)
                {

                    valid = true;
                }
            }
            //cerr << "x: " << tempvec.x() << " y: " << tempvec.y() << "\n";

            //tempvec = tempvec * ((VideoLOD*)_master)->win[i];
            //input[2] = tempvec.x();
            //input[3] = tempvec.y();

            //cerr << "x: " << input[2] << " y: " << input[3] << "\n";

            //(Geometry*)((Geode*)((Group*)((Group*)(this->getChild(1)))->getChild(2))->getChild(0))->getDrawable(0);
            //b = geo1->getBound();
            if (tilemode)
            {
                tempvec = b.corner(3);
                //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(1)))->getMatrix() * bigmat;
                tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(1)))->getMatrix() * locall2r * localview;
                if (tempvec.z() > LIMIT)
                {
                    if (tempvec.z() > 0)
                    {
                        zarray[2] = 1.0;
                    }
                    else
                    {
                        zarray[2] = 0.0;
                    }
                    isout = 1;
                    tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
                }
                else
                {
                    zarray[2] = 0.0;
                }
                tempvec = tempvec * localpro;
                input[4] = (tempvec.x()) * (float)localhalfw;
                input[5] = (tempvec.y()) * (float)localhalfh;
                //zarray[2] = tempvec.z();
                if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[2] == 0.0)
                {

                    valid = true;
                }
            }
            else
            {
                tempvec = b.corner(1);
                //tempvec = tempvec * ((MatrixTransform *)(((Group*)(this->getChild(1)))->getChild(1)))->getMatrix() * bigmat;
                tempvec = tempvec * ((MatrixTransform *)(((Group *)(this->getChild(1)))->getChild(1)))->getMatrix() * locall2r * localview;
                if (tempvec.z() > LIMIT)
                {
                    if (tempvec.z() > 0)
                    {
                        zarray[1] = 1.0;
                    }
                    else
                    {
                        zarray[1] = 0.0;
                    }
                    isout = 1;
                    tempvec = Vec3(tempvec.x(), tempvec.y(), LIMIT);
                }
                else
                {
                    zarray[1] = 0.0;
                }
                tempvec = tempvec * localpro;
                input[2] = (tempvec.x()) * (float)localhalfw;
                input[3] = (tempvec.y()) * (float)localhalfh;
                //zarray[1] = tempvec.z();
                if ((tempvec.x() >= -1.0 && tempvec.x() <= 1.0) && (tempvec.y() >= -1.0 && tempvec.y() <= 1.0) && zarray[1] == 0.0)
                {

                    valid = true;
                }
            }
            //cerr << "x: " << tempvec.x() << " y: " << tempvec.y() << "\n";

            //tempvec = tempvec * ((VideoLOD*)_master)->win[i];
            //input[4] = tempvec.x();
            //input[5] = tempvec.y();

            //cerr << "x: " << input[4] << " y: " << input[5] << "\n";

            //cerr << input[0] << " " << input[1] << " " << input[2] << " " << input[3] << " " << input[4] << " " << input[5] << " " << input[6] << " " << input[7] << "\n";
            if (valid)
            {
                retarea += farea(input);
                numvalid++;
            }
            else
            {
                if ((zarray[0] + zarray[1] + zarray[2] + zarray[3]) == 4.0)
                {
                    isout = oldisout;
                    continue;
                }
                for (int k = 0; k < 4; k++)
                {
                    if (zarray[k] == 1.0 || zarray[(k + 1) % 4] == 1.0)
                        continue;
                    int i1 = (2 * k) % 8;
                    int i2 = (2 * (k + 1)) % 8;
                    if ((input[i1] < -(localhalfw) && input[i2] > -(localhalfw)) || (input[i2] < -(localhalfw) && input[i1] > -(localhalfw)))
                    {
                        float slope, b;
                        if (input[i1] == input[i2])
                            continue;

                        slope = (((float)input[i1 + 1]) - ((float)input[i2 + 1])) / (((float)input[i1]) - ((float)input[i2]));
                        b = ((float)input[i1 + 1]) - (slope * input[i1]);
                        int y = (int)((slope * (-(localhalfw))) + b);
                        //cerr << "1: " << input[i1] << "," << input[i1+1] << " 2: " << input[i2] << "," << input[i2+1] <<  " y: " << y << "\n";
                        if (y >= -(localhalfh) && y <= localhalfh)
                        {
                            //cerr << "turning valid\n";
                            valid = true;
                            break;
                        }
                        else if (y < -(localhalfh))
                        {
                            down = 1;
                        }
                        else
                        {
                            up = 1;
                        }
                    }

                    if ((input[i1] < localhalfw && input[i2] > localhalfw) || (input[i2] < localhalfw && input[i1] > localhalfw))
                    {
                        float slope, b;
                        if (input[i1] == input[i2])
                            continue;

                        slope = (((float)input[i1 + 1]) - ((float)input[i2 + 1])) / (((float)input[i1]) - ((float)input[i2]));
                        b = ((float)input[i1 + 1]) - (slope * input[i1]);
                        int y = (int)((slope * (localhalfw)) + b);
                        //cerr << "1: " << input[i1] << "," << input[i1+1] << " 2: " << input[i2] << "," << input[i2+1] <<  " y: " << y << "\n";
                        if (y >= -(localhalfh) && y <= localhalfh)
                        {
                            //cerr << "turning valid\n";
                            valid = true;
                            break;
                        }
                        else if (y < -(localhalfh))
                        {
                            down = 1;
                        }
                        else
                        {
                            up = 1;
                        }
                    }

                    if ((input[i1 + 1] < -(localhalfh) && input[i2 + 1] > -(localhalfh)) || (input[i2 + 1] < -(localhalfh) && input[i1 + 1] > -(localhalfh)))
                    {
                        float slope, b;
                        if (input[i1] == input[i2])
                        {
                            if (input[i1] >= -localhalfw && input[i1] <= localhalfw)
                            {
                                valid = true;
                                break;
                            }
                            continue;
                        }

                        slope = (((float)input[i1 + 1]) - ((float)input[i2 + 1])) / (((float)input[i1]) - ((float)input[i2]));
                        b = ((float)input[i1 + 1]) - (slope * input[i1]);
                        //int y =(int) ((slope * (-(localhalfw))) + b);
                        int x = (int)((-(localhalfh)-b) / slope);
                        //cerr << "1: " << input[i1] << "," << input[i1+1] << " 2: " << input[i2] << "," << input[i2+1] <<  " y: " << y << "\n";
                        if (x >= -(localhalfw) && x <= localhalfw)
                        {
                            //cerr << "turning valid\n";
                            valid = true;
                            break;
                        }
                        else if (x < -(localhalfw))
                        {
                            left = 1;
                        }
                        else
                        {
                            right = 1;
                        }
                    }

                    if ((input[i1 + 1] < (localhalfh) && input[i2 + 1] > (localhalfh)) || (input[i2 + 1] < (localhalfh) && input[i1 + 1] > (localhalfh)))
                    {
                        float slope, b;
                        if (input[i1] == input[i2])
                        {
                            if (input[i1] >= -localhalfw && input[i1] <= localhalfw)
                            {
                                valid = true;
                                break;
                            }
                            continue;
                        }

                        slope = (((float)input[i1 + 1]) - ((float)input[i2 + 1])) / (((float)input[i1]) - ((float)input[i2]));
                        b = ((float)input[i1 + 1]) - (slope * input[i1]);
                        //int y =(int) ((slope * (-(localhalfw))) + b);
                        int x = (int)(((localhalfh)-b) / slope);
                        //cerr << "1: " << input[i1] << "," << input[i1+1] << " 2: " << input[i2] << "," << input[i2+1] <<  " y: " << y << "\n";
                        if (x >= -(localhalfw) && x <= localhalfw)
                        {
                            //cerr << "turning valid\n";
                            valid = true;
                            break;
                        }
                        else if (x < -(localhalfw))
                        {
                            left = 1;
                        }
                        else
                        {
                            right = 1;
                        }
                    }
                }
                if (valid || (up && down) || (left && right))
                {
                    retarea += farea(input);
                    numvalid++;
                }
            }

            int count = ((int)zarray[0]) + ((int)zarray[1]) + ((int)zarray[2]) + ((int)zarray[3]);
            //cerr << "count: " << count << "\n";
            if (count > 0 && count < 4)
            {
                isout = 1;
            }
        }
        else
        {
            cerr << "Unknown tile format.\n";
            return -1.0;
        }
    }
    // cerr << "numvalid: " << numvalid << " retarea: " << retarea <<  "\n";
    if (numvalid > 0)
    {
        return (retarea / ((float)numvalid));
    }
    else if (isout)
    {
        return -2.0;
    }
    else
    {
        return -1.0;
    }
}
