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
#include <stdlib.h>
#include "VideoLOD.h"
#include <osg/CullStack>
#include <osg/Notify>
#include <iostream>
#include <algorithm>
#include <string>
#include <osgDB/ReadFile>
#include <CUI.h>
#include <osgUtil/Optimizer>

#ifndef WIN32
#include <sys/time.h>
#endif

#include "SLOD.h"

using namespace std;
using namespace osg;
using namespace osgDB;
using namespace cui;

VideoLOD::VideoLOD()
{
    _frameNumberOfLastTraversal = 0;
    _centerMode = USER_DEFINED_CENTER;
    _radius = -1;
    _numChildrenThatCannotBeExpired = 0;
    firstrun = 1;
    _dataSetSet = 0;
    pm = PAUSE;
    _scale = 1.0f;
    _frameRate = 0;
    playTime = 0;
    pt = LOW_DETAIL;
    loop = true;
}

VideoLOD::VideoLOD(const PagedLOD &plod, const CopyOp &copyop)
    : PagedLOD(plod, copyop)
{
    firstrun = 1;
    _dataSetSet = 0;
    pm = PAUSE;
    _scale = 1.0f;
    _frameRate = 0;
    playTime = 0;
    pt = LOW_DETAIL;
    loop = true;
}

bool VideoLOD::getLoop()
{
    return loop;
}

void VideoLOD::setLoop(bool l)
{
    loop = l;
}

void VideoLOD::seek(double s)
{
    if (s >= 0.0 && s <= 1.0)
    {
        frameNum = (int)(s * (file.frameend - file.framestart));
        playTime = frameNum * timePerFrame;
    }
}

void VideoLOD::seek(int i)
{
    if (i >= 0 && i < (file.frameend - file.framestart))
    {
        frameNum = file.framestart + i;
        playTime = frameNum * timePerFrame;
    }
}

void VideoLOD::setPlayMode(playMode mode)
{
    if (pm == mode || !_dataSetSet)
        return;

    pm = mode;

    if (mode == PAUSE)
    {
        pause();
    }
}

int VideoLOD::getPauseType()
{
    return pt;
}

void VideoLOD::setPauseType(pauseType p)
{
    pt = p;
}

int VideoLOD::getPlayMode()
{
    return pm;
}

void VideoLOD::printout()
{
    if (!_dataSetSet || singletile)
        return;

    cerr << "\nArea ratio:\n";
    for (int i = 0; i < numoftiles; i++)
    {
        cerr << dist[i] << " ";
    }
    cerr << "\n scale: " << _scale << "\n";

    cerr << "\nSeen:\n";
    for (int i = 0; i < numoftiles; i++)
    {
        cerr << seen[i] << " ";
    }
    cerr << "\n";

    cerr << "\nMMLevel:\n";
    for (int i = 0; i < numoftiles; i++)
    {
        cerr << level[i] << " ";
    }
    cerr << "\n";
}

void VideoLOD::pause()
{
    if (pt == WHAT_YOU_SEE || singletile)
        return;

    sync++;
    for (int i = 0; i < numoftiles; i++)
    {
        if (pt == LOW_DETAIL)
            ((SLOD *)(treenodes.at(i)))->setfile((char *)lastLoaded.c_str(), 0, sync);
        else
            ((SLOD *)(treenodes.at(i)))->setfile((char *)lastLoaded.c_str(), maxlevel, sync);
        ((SLOD *)(treenodes.at(i)))->pause(sync);
    }
}

void VideoLOD::setMaster(bool b)
{
    if (singletile)
        return;

    if (b)
    {
        cerr << "Set as Master Node.\n";
        onMaster = 1;
        for (int i = 0; i < numoftiles; i++)
        {
            seen[i] = 1;
            level[i] = 0;
            newlevel[i] = 0;
        }
    }
    else
    {
        onMaster = 0;
    }
}

void VideoLOD::setDataSet(string path, string prefix, string suffix, int start, int end, int vnums)
{
    if (_dataSetSet)
    {
        this->unsetDataSet();
    }

    file.path = path;
    file.prefix = prefix;
    file.suffix = suffix;
    file.vnums = vnums;
    file.framestart = start;
    file.frameend = end;

    frameNum = start;

    if (cover->getScene()->getNumParents() == 0 || cover->getScene()->getParent(0)->className() != string("Camera"))
    {
        cerr << "Unable to locate Camera Node.\n";
        return;
    }

    numOfCameras = cover->getScene()->getNumParents();

    cerr << "Scene has " << numOfCameras << " Camera nodes.\n";

    pro = new Matrix[numOfCameras];
    mview = new Matrix[numOfCameras];
    win = new Matrix[numOfCameras];

    for (int i = 0; i < numOfCameras; i++)
    {
        win[i] = ((Camera *)cover->getScene()->getParent(i))->getViewport()->computeWindowMatrix();
    }

    for (int i = 0; i < numOfCameras; i++)
    {
        pro[i] = ((Camera *)(cover->getScene()->getParent(i)))->getProjectionMatrix();
    }

    number = new char[file.vnums + 1];

    char fmt[15];

    sprintf(fmt, "%d", file.vnums);

    file.numFormat = string("%0") + string(fmt) + string("d");
    sprintf(number, (const char *)(file.numFormat).c_str(), frameNum);

    ops = new ReaderWriter::Options();

    ops->setObjectCacheHint(ReaderWriter::Options::CACHE_NONE);

    osg::Node *imageNode = osgDB::readNodeFile((char *)(file.path + file.prefix + string(number, file.vnums) + file.suffix + string(".osga")).c_str(), ops);

    if (imageNode == NULL)
    {
        cerr << "Init failure: Error opening file " << (file.path + file.prefix + string(number, file.vnums) + file.suffix + string(".osga")) << "\n";
        return;
    }

    if (strcmp((imageNode)->className(), string("PagedLOD").c_str()))
    {
        //this->addChild(imageNode);
        singletile = 1;
        _dataSetSet = 1;
        seen = new int[1];
        seen[0] = 1;
        usecull = new int[1];
        usecull[0] = 0;

        MatrixTransform *mtnode = new MatrixTransform();

        Matrix mat;
        mat.makeScale(1.0f, 1.0f, 0.00f);
        mtnode->setMatrix(mat);
        Node *stslod = new SLOD();
        ((SLOD *)stslod)->_master = this;
        this->addChild(mtnode);
        this->addChild(stslod);

        //mtnode->addChild(stslod);
        Node *gnode = new Group();
        slodroot = gnode;
        ((SLOD *)stslod)->addChild(new Node());
        ((SLOD *)stslod)->addChild(gnode);
        mtnode->addChild(gnode);
        //((SLOD*)stslod)->addChild(mtnode);
        ((Group *)gnode)->addChild(imageNode);

        StateSet *ss = new StateSet();
        ss->setMode(GL_DEPTH_TEST, 0);
        ss->setMode(GL_LIGHTING, StateAttribute::OFF);
        ss->setRenderingHint(StateSet::OPAQUE_BIN);
        this->setStateSet(ss);

        halfw = (int)((Viewport *)((Camera *)(cover->getScene()->getParent(0)))->getViewport())->width() / 2;
        halfh = (int)((Viewport *)((Camera *)(cover->getScene()->getParent(0)))->getViewport())->height() / 2;

        return;
    }
    else
    {
        singletile = 0;
    }

    cerr << "imagenode has " << ((PagedLOD *)imageNode)->getNumFileNames() << " filenames\n";

    for (int i = 0; i < ((PagedLOD *)imageNode)->getNumFileNames(); i++)
    {
        cerr << ((PagedLOD *)imageNode)->getFileName(i) << "\n";
        ((PagedLOD *)this)->setFileName(i, ((PagedLOD *)imageNode)->getFileName(i));
    }

    cerr << "i have " << this->getNumFileNames() << " filenames\n";

    init((char *)(file.path + file.prefix + string(number, file.vnums) + file.suffix + string(".osga")).c_str());
}

void VideoLOD::unsetDataSet()
{
    if (_dataSetSet)
        this->removeChildren(0, this->getNumChildren());

    firstrun = 1;
    _dataSetSet = 0;
    pm = PAUSE;
    _scale = 1.0f;
    _frameRate = 0;
    playTime = 0;
}

void VideoLOD::stop()
{
    if (!_dataSetSet)
        return;

    pm = PLAY;
    //if(_frameRate <= 0)
    //frameNum = file.framestart;
    //else
    frameNum = file.frameend;
    playTime = 0.0;
    bool temp = loop;
    loop = true;
    advanceFrame();
    loop = temp;
    pm = PAUSE;
    pause();
}

void VideoLOD::setScale(float scale)
{
    _scale = scale;
}

float VideoLOD::getScale()
{
    if (!_dataSetSet)
        return 1.0f;
    else
        return _scale;
}

int VideoLOD::getFrameRate()
{
    return _frameRate;
}

void VideoLOD::setFrameRate(int fr)
{
    timePerFrame = 1.0 / ((double)fr);
    if (_dataSetSet)
        playTime = ((double)(frameNum - file.framestart)) * timePerFrame;
    else
        playTime = 0.0;
    _frameRate = fr;
}

void VideoLOD::advanceFrame()
{
    if (!_dataSetSet)
    {
        //cerr << "advanceFrame called without loading a data set.\n";
        return;
    }
    //cerr << "advanceframe\n";
    //struct timeval start1, end1, start2, end2, start3, end3;
    //gettimeofday(&start1, NULL);
    //cerr << "advanceframe start\n";
    string basename;

    if (pm == PAUSE)
    {
        //cerr << "leaving advanceframe via pause\n";
        return;
    }

    //cerr << "past pause\n";
    if (_frameRate <= 0)
    {
        if (pm == PLAY)
            frameNum++;
        else if (pm == FF)
            frameNum += 2;
        else if (pm == RW)
            frameNum -= 2;

        if (frameNum > file.frameend)
        {
            if (!loop)
            {
                stop();
                return;
            }
            frameNum = file.framestart + ((frameNum - file.frameend) - 1);
        }

        if (frameNum < file.framestart)
        {
            if (!loop)
            {
                stop();
                return;
            }
            frameNum = file.frameend + ((frameNum - file.framestart) + 1);
        }
    }
    else
    {
        int newFrameNum;

        //int cycle = (int) playTime / ((file.frameend - file.framestart) * timePerFrame);((int)(playTime / ((file.frameend - file.framestart) * timePerFrame)) != cycle)

        if (pm == PLAY)
            playTime += cover->frameDuration();
        else if (pm == FF)
            playTime += (cover->frameDuration()) * 2.0;
        else if (pm == RW)
            playTime -= (cover->frameDuration()) * 2.0;

        if (playTime >= 0)
            newFrameNum = (((int)(playTime / timePerFrame)) % (file.frameend - file.framestart)) + file.framestart;
        else
            newFrameNum = (file.frameend - ((int)((-playTime) / timePerFrame)) % (file.frameend - file.framestart));

        if (!loop && ((pm != RW && (newFrameNum < frameNum)) || ((pm == RW) && (newFrameNum > frameNum))))
        {
            stop();
            return;
        }

        if (frameNum == newFrameNum)
            return;

        frameNum = newFrameNum;
    }

    for (int i = 0; i < numOfCameras; i++)
    {
        pro[i] = ((Camera *)(cover->getScene()->getParent(i)))->getProjectionMatrix();
    }
    //mview = ((Camera*)(cover->getObjectsRoot()->getParent(0)->getParent(0)->getParent(0)->getParent(0)))->getViewMatrix();
    for (int i = 0; i < numOfCameras; i++)
    {
        mview[i] = ((Camera *)(cover->getScene()->getParent(i)))->getViewMatrix();
    }
    l2r = CUI::computeLocal2Root(slodroot);

    float area;

    if (singletile)
    {
        //cerr << "getarea\n";
        usecull[0] = 0;
        float tempf = ((SLOD *)((Group *)slodroot)->getParent(0))->getarea();
        if (tempf == -1.0)
            seen[0] = 0;
        else if (tempf >= 0)
            seen[0] = 1;
        else
            usecull[0] = 1;
        //cerr << "gotarea\n";
    }

    //gettimeofday(&start2, NULL);
    if (!onMaster && !singletile)
    {
        for (int i = 0; i < numoftiles; i++)
        {
            usecull[i] = 0;
            area = ((SLOD *)(treenodes.at(i)))->getarea();
            if (area == -1.0)
            {
                seen[i] = 0;
                continue;
            }
            else if (area >= 0)
            {
                seen[i] = 1;
            }
            else
            {
                usecull[i] = 1;
                newlevel[i] = maxlevel;
                continue;
            }

            dist[i] = (area / (float)tilearea) / _scale;

            float ref = 1.0;

            for (int k = 0; k <= maxlevel; k++)
            {
                ref = ref / 4.0f;
                if (dist[i] > ref)
                {
                    newlevel[i] = (maxlevel - k);
                    break;
                }
            }
            if (dist[i] <= ref)
                newlevel[i] = 0;
        }
    }
    //gettimeofday(&end2, NULL);

    sprintf(number, (const char *)(file.numFormat).c_str(), frameNum);

    basename = file.prefix + string(number, file.vnums) + file.suffix;

    //gettimeofday(&start3, NULL);
    setfile((char *)((file.path + basename + string(".osga/") + basename).c_str()));
    //gettimeofday(&end3, NULL);

    //osgUtil::Optimizer::TextureAtlasVisitor tv = osgUtil::Optimizer::TextureAtlasVisitor();
    //tv.apply(*this);
    //tv.optimize();

    // gettimeofday(&end1, NULL);
    //cerr << ((end1.tv_sec - start1.tv_sec)*1000000 + (end1.tv_usec - start1.tv_usec)) << " " << ((end2.tv_sec - start2.tv_sec)*1000000 + (end2.tv_usec - start2.tv_usec)) << " " << ((end3.tv_sec - start3.tv_sec)*1000000 + (end3.tv_usec - start3.tv_usec)) <<  "\n";
    //cerr << "advanceframe end.\n";
}

void VideoLOD::init(char *name)
{

    cerr << "In VideoLOD init.\n";

    sync = 1;
    numoftiles = 0;
    treenodes = vector<Node *>();
    firstframe = 1;
    tilearea = 0;
    loadedsync = -1;
    onMaster = 0;

    StateSet *ss = new StateSet();
    ss->setMode(GL_DEPTH_TEST, 0);
    ss->setMode(GL_LIGHTING, StateAttribute::OFF);
    ss->setRenderingHint(StateSet::OPAQUE_BIN);
    this->setStateSet(ss);

    this->setNodeMask(this->getNodeMask() & ~2);

    _databasePath.assign(name);
    _databasePath = _databasePath + string("/");

    StateSet *stateset = new StateSet();
    stateset->setRenderBinDetails((0 + 3), "RenderBin");
    stateset->setMode(GL_DEPTH_TEST, 0);
    stateset->setMode(GL_LIGHTING, StateAttribute::OFF);

    //halfw = (int)((Viewport*)((Camera*)(cover->getObjectsRoot()->getParent(0)->getParent(0)->getParent(0)->getParent(0)))->getViewport())->width() / 2;
    halfw = (int)((Viewport *)((Camera *)(cover->getScene()->getParent(0)))->getViewport())->width() / 2;
    //halfh = (int)((Viewport*)((Camera*)(cover->getObjectsRoot()->getParent(0)->getParent(0)->getParent(0)->getParent(0)))->getViewport())->height() / 2;
    halfh = (int)((Viewport *)((Camera *)(cover->getScene()->getParent(0)))->getViewport())->height() / 2;

    cerr << "halfw: " << halfw << " halfh: " << halfh << "\n";

    MatrixTransform *mtnode = new MatrixTransform();
    Matrix mat;
    mat.makeScale(1.0f, 1.0f, 0.00f);

    mtnode->setMatrix(mat);
    this->addChild(mtnode);

    cerr << "size: " << _rangeList.size() << "\n";

    for (unsigned int i = 0; i < this->getNumFileNames(); ++i)
    {
        if (this->getFileName(i) == string(""))
            continue;
        //cerr <<_databasePath << _perRangeDataList[i]._filename << "\n";

        ref_ptr<osg::Node> nodeRefPtr1 = osgDB::readNodeFile((_databasePath + this->getFileName(i)).c_str(), ops);
        Node *root = nodeRefPtr1.get();
        if (root == NULL)
        {
            cerr << "Init failure: Error opening file " << (_databasePath + this->getFileName(i)) << "\n";
            return;
        }
        if (strcmp((root)->className(), string("Group").c_str()))
        {
            cerr << "Root Node not of type Group, of type " << (nodeRefPtr1.get())->className() << "\n";
        }

        slodroot = root;

        mtnode->addChild(root);

        for (int j = 0; j < ((Group *)root)->getNumChildren(); j++)
        {
            Node *tempnode = ((Group *)root)->getChild(j);
            if (strcmp((tempnode)->className(), string("PagedLOD").c_str()))
            {
                cerr << "Child Node not of type PagedLOD, of type " << (tempnode)->className() << "\n";
            }
            //SLOD * tempSLOD = new SLOD(*((PagedLOD*)tempnode), *(new CopyOp()));CopyOp::SHALLOW_COPY
            SLOD *tempSLOD = new SLOD(*((PagedLOD *)tempnode), CopyOp::DEEP_COPY_ALL);
            ((Group *)tempnode)->removeChildren(0, ((Group *)tempnode)->getNumChildren());
            ((Group *)root)->replaceChild(tempnode, tempSLOD);
            cerr << "Before SLOD init.\n";
            ((PagedLOD *)tempSLOD)->setDatabasePath(_databasePath);
            tempSLOD->init((Node *)this, 0, (char *)(this->getFileName(i)).c_str(), (int)j, new vector<Node *>(), stateset, this);
        }
    }

    level = new int[numoftiles];
    dist = new float[numoftiles];
    newlevel = new int[numoftiles];
    seen = new int[numoftiles];
    usecull = new int[numoftiles];

    for (int i = 0; i < numoftiles; i++)
        level[i] = maxlevel;
    for (int i = 0; i < numoftiles; i++)
        newlevel[i] = maxlevel;
    for (int i = 0; i < numoftiles; i++)
        seen[i] = 1;
    for (int i = 0; i < numoftiles; i++)
        usecull[i] = 0;

    cerr << "numoftiles: " << this->numoftiles << " maxlevel: " << this->maxlevel << " tilearea: " << this->tilearea << " sync: " << this->sync << " treenodes: " << this->treenodes.size() << "\n";

    _dataSetSet = 1;
    cerr << "Out of VideoLOD init\n";
}

void VideoLOD::setfile(char *name)
{
    lastLoaded = string(name);

    if (singletile)
    {
        if (seen[0])
        {
            Node *stnode = osgDB::readNodeFile((string(name) + string(".ive")).c_str(), ops);
            if (stnode == NULL)
            {
                cerr << "Unable to open file " << string(name) + string(".ive") << "\n";
                return;
            }
            ((Group *)slodroot)->replaceChild(((Group *)slodroot)->getChild(0), stnode);
        }
        return;
    }

    sync++;
    for (int i = 0; i < numoftiles; i++)
    {
        if (seen[i] || firstframe)
        {
            //cerr << "setfile: " << name << " " << newlevel[i] << " " << sync << "\n";
            if (!((SLOD *)(treenodes.at(i)))->setfile(name, newlevel[i], sync))
            {
                sync--;
                return;
            }

            level[i] = newlevel[i];
        }
    }

    /*int * ttemp;
  ttemp = level;
  level = newlevel;
  newlevel = ttemp;*/
    firstframe = 0;
}

void VideoLOD::traverse(NodeVisitor &nv)
{
    //cerr << "In traverse\n";
    /*if(firstrun)
    {
      //init();
      firstrun = 0;
    }*/

    switch (nv.getTraversalMode())
    {
    case (NodeVisitor::TRAVERSE_ALL_CHILDREN):
        std::for_each(_children.begin(), _children.end(), NodeAcceptOp(nv));
        break;
    case (NodeVisitor::TRAVERSE_ACTIVE_CHILDREN):
    {
        if (!_dataSetSet)
            return;

        osg::CullStack *cullStack = dynamic_cast<osg::CullStack *>(&nv);
        if (cullStack && singletile)
        {
            if (usecull[0])
            {
                //cerr << "Using isculled ";
                if (!cullStack->isCulled(((Group *)slodroot)->computeBound()))
                    seen[0] = 1;
                //cerr << seen[i] << "\n";
            }
        }
        else if (cullStack && pm != PAUSE)
        {
            for (int i = 0; i < numoftiles; i++)
            {
                if (usecull[i])
                {
                    //cerr << "Using isculled ";
                    if (!cullStack->isCulled(((Group *)((Group *)treenodes.at(i))->getChild(1))->computeBound()))
                        seen[i] = 1;
                    //cerr << seen[i] << "\n";
                }
            }
            //cerr << "cullstack\n";
        }

        //struct timeval start1, end1;
        //gettimeofday(&start1, NULL);
        //cerr << "In traverse active\n";
        //cerr << "traverse start\n";
        //if(loaded.valid())
        //loaded.release();
        //struct timeval start1, end1, start2, end2, start3, end3;
        //gettimeofday(&start1, NULL);
        //float area;

        //gettimeofday(&start2, NULL);
        //pro = ((Camera*)(cover->getObjectsRoot()->getParent(0)->getParent(0)->getParent(0)->getParent(0)))->getProjectionMatrix();
        /*for(int i = 0; i < numOfCameras; i++)
          {
            pro[i] = ((Camera*)(cover->getScene()->getParent(i)))->getProjectionMatrix();
          }
          //mview = ((Camera*)(cover->getObjectsRoot()->getParent(0)->getParent(0)->getParent(0)->getParent(0)))->getViewMatrix();
          for(int i = 0; i < numOfCameras; i++)
          {
            mview[i] = ((Camera*)(cover->getScene()->getParent(i)))->getViewMatrix();
          }
          l2r = CUI::computeLocal2Root(slodroot);*/

        //cerr << "checkpoint 1\n";

        //cerr << "is vis\n";
        //osg::CullStack* cullStack = dynamic_cast<osg::CullStack*>(&nv);

        if (singletile)
        {
            _children[0]->accept(nv);
            return;
        }

        /*if (cullStack && !onMaster)
          {
            //cerr << "is stack\n";
            for(int i = 0; i < numoftiles; i++)
            {
              if(cullStack->isCulled((((Group*)((Group*)((Group*)(treenodes.at(i)))->getChild(1)))->computeBound())))
                seen[i] = 0;
              else
                seen[i] = 1;
            }*/
        //}
        //else
        //{
        //gettimeofday(&end2, NULL);
        //gettimeofday(&start3, NULL);

        //cerr << "checkpoint 2\n";
        //printout();
        //cerr << "start of tile loop\n";
        /*for(int i = 0; i < numoftiles; i++)
            {
              //cerr << i << ": seen: " << seen[i] << "\n"; 
              
              //cerr << "before area\n";
              area = ((SLOD*)(treenodes.at(i)))->getarea();
              //cerr << "after area: " << area << "\n";
              if(area == -1.0)
              {
                seen[i] = 0;
                continue;
              }
              else
              {
                seen[i] = 1;
              }

              dist[i] = (area / (float)tilearea) / _scale;

              float ref = 1.0;
              //cerr << "dist: " << dist[i] << "\n";
                //printout();
              //if(dist[i] > (1.0/(4.0 * (float)maxlevel)))
              //{
              for(int k = 0; k <= maxlevel; k++)
              {
                ref = ref / 4.0f;
                if(dist[i] > ref)
                {
                  //cerr << "newlevel: " << (maxlevel - k) << "\n";
                  newlevel[i] = (maxlevel - k);
                  break;
                }
              }
              //}
              //else
              //{
              if(dist[i] <= ref)
                newlevel[i] = 0;
              //}
            
            }
          }*/

        //gettimeofday(&end3, NULL);
        /*for(int i = 0; i < numoftiles; i++)
          {
            float ref = 1.0;
            for(int k = 0; k >= maxlevel; k++)
            {
              ref = ref / 4.0f;
              if(dist[i] > ref)
              {
                newlevel[i] = (maxlevel - k);
                break;
              }
            }
          }*/
        //if(!cullStack)

        for (int i = 0; i < this->getNumChildren(); i++)
        {
            _children[i]->accept(nv);
        }
        //gettimeofday(&end1, NULL);

        //gettimeofday(&end1, NULL);
        //cerr << ((end1.tv_sec - start1.tv_sec)*1000000 + (end1.tv_usec - start1.tv_usec))  << "\n";
        //cerr << "traverse end\n";

        break;
    }
    default:
        break;
    }
}

double VideoLOD::getPos()
{
    if (!_dataSetSet)
        return 0.0;

    return (((double)(frameNum - file.framestart)) / ((double)(file.frameend - file.framestart)));
}

VideoLOD::~VideoLOD()
{
    if (!_dataSetSet)
    {
    }
}
