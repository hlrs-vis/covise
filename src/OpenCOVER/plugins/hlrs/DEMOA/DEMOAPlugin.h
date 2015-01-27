/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DEMOA_PLUGIN_H
#define _DEMOA_PLUGIN_H

/****************************************************************************\
**                                                            (C)2014 HLRS  **
**                                                                          **
** Description: load DEMOA Simulation files (INSPO)                         **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** 2014  v1	    				       		                                **
**                                                                          **
**                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
using namespace covise;
using namespace opencover;

#include "cover/coTabletUI.h"
#include <osg/Geode>
#include <osg/ref_ptr>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/LineWidth>
#include <PluginUtil/coSphere.h>
#include <QStringList>
#include <QMap>
#include "parmblock.h"

class D_Primitive;

class DEMOAPlugin : public coVRPlugin, public coTUIListener
{
public:
    DEMOAPlugin();
    virtual ~DEMOAPlugin();
    static DEMOAPlugin *instance();
    bool init();

    void setTimestep(int t);

    int loadANI(const char *filename, osg::Group *loadParent);
    static int sloadANI(const char *filename, osg::Group *loadParent, const char *covise_key);
    static int unloadANI(const char *filename, const char *covise_key);

    // this will be called in PreFrame
    void preFrame();
    coTUITab *PathTab;
    coTUIToggleButton *record;
    coTUIButton *stop;
    coTUIComboBox *mapChoice;
    std::string aniFileName;
    std::string path;

    int currentMap;
    QStringList mapNames;
    QMap<QString, int> mapSize;
    QMap<QString, float *> mapValues;

    osg::ref_ptr<osg::StateSet> geoState;
    osg::ref_ptr<osg::Material> linemtl;
    osg::ref_ptr<osg::LineWidth> lineWidth;
    osg::Vec4 getColor(float pos);
    void deleteColorMap(const QString &name);

    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);
    virtual void tabletEvent(coTUIElement *tUIItem);

private:
    char *filename;
    osg::Group *parentNode;
    osg::ref_ptr<osg::Group> DemoaRoot;

    static DEMOAPlugin *thePlugin;
    osg::MatrixTransform *newPrimitive(parmblock &block, int idx);

    void save();

    osg::ref_ptr<osg::Geometry> geom;
    osg::ref_ptr<osg::Geode> geode;

    void getpngbase();
    void getfollow_idx();
    void getgriddim();
    double **ReadData(std::FILE *stream, int *rows, int *cols);
    void getdata(const char file[], int initframe);

    void updatefollow();
    void clear_stdin();

    int nForces;
    int nSegs;

    // data variables
    double **data;
    int nLines, nCols;
    double tt, dtt;

    // PNG image variables
    char pngbase[128];
    int pngno;
    int png_compress;

    // linkage variables
    int **segs;
    int nSegslinks, n_points_per_seg;

    // window handles
    static int win_ani, win_txt;

    // window size
    static int W, H;

    // switches
    GLboolean db;
    GLboolean sm;
    GLboolean lp;
    GLboolean wf;
    GLboolean fw;
    GLboolean gnd;
    GLboolean rc;
    GLboolean notimefact;

    // view
    GLfloat ax;
    GLint ngrid_xlo, ngrid_xhi, ngrid_ylo, ngrid_yhi;
    GLfloat scale;
    GLfloat follow_x, follow_y;
    int follow_idx;

    // perspective transformation
    GLfloat **A;

    // frame-no., animation speed
    GLint frame, inc;

    void Coord2Matrix(GLfloat **AA, double *t_frame, double **dataptr, int frame);

    // segment names
    std::vector<D_Primitive *> primitives;
    static GLint lab_id;

    void toggle_loop();
    GLboolean isloop();
    void toggle_smooth();
    GLboolean issmooth();
    void toggle_wire();
    GLboolean iswire();
    void toggle_follow();
    GLboolean isfollow();
    void toggle_ground();
    GLboolean isground();
    void toggle_record();
    GLboolean isrecord();
};
#endif
