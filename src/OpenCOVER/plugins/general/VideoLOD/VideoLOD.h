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

#ifndef OSG_VideoLOD
#define OSG_VideoLOD 1

#include <osg/PagedLOD>
#include <osg/Texture2D>
#include <osg/Geometry>

#include <osgDB/ReaderWriter>

namespace osg
{

/** VideoLOD.
*/
class VideoLOD : public PagedLOD
{
public:
    enum playMode
    {
        PAUSE = 0,
        PLAY,
        FF,
        RW
    };
    enum pauseType
    {
        LOW_DETAIL = 0,
        HIGH_DETAIL,
        WHAT_YOU_SEE
    };

    VideoLOD();

    VideoLOD(const PagedLOD &, const CopyOp &copyop = CopyOp::SHALLOW_COPY);

    META_Node(osg, VideoLOD);

    virtual void traverse(NodeVisitor &nv);

    void setDatabasePath(const std::string &path);
    void printout();
    void advanceFrame();
    void stop();

    void setScale(float scale);
    float getScale();

    void setFrameRate(int fr);
    int getFrameRate();

    void setPauseType(pauseType p);
    int getPauseType();

    void setDataSet(string path, string prefix, string suffix, int start, int end, int vnum);
    void unsetDataSet();

    void setPlayMode(playMode mode);
    int getPlayMode();

    bool getLoop();
    void setLoop(bool l);

    void seek(double s);
    void seek(int i);

    void setMaster(bool b);

    double getPos();

    friend class SLOD;

protected:
    void setfile(char *name);
    void init(char *name);
    void pause();
    virtual ~VideoLOD();

    osgDB::ReaderWriter::Options *ops;

    struct fileinfo
    {
        string path;
        string prefix;
        string suffix;
        int vnums;
        int framestart;
        int frameend;
        string numFormat;
    };

    char *number;
    playMode pm;
    pauseType pt;
    bool loop;

    Matrix *pro;
    Matrix *mview;
    Matrix *win;
    Matrix l2r;

    int numOfCameras;
    int _frameRate;
    double playTime;
    double timePerFrame;
    int sync;
    float _scale;

    int loadedsync;
    ref_ptr<Node> loaded;

    int _dataSetSet;
    int frameNum;
    int numoftiles;
    int maxlevel;
    int tilearea;
    int halfw, halfh;
    int singletile;
    vector<Node *> treenodes;
    Node *slodroot;

    struct fileinfo file;
    string lastLoaded;

    float *dist;

    int *level;
    int *newlevel;
    int *seen;
    int *usecull;

    int onMaster;
    int firstrun;
    int firstframe;
};
}

#endif
