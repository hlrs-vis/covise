/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef INTERACTIONMANAGER_H
#define INTERACTIONMANAGER_H

/****************************************************************************\
 **                                              (C)2010 Anton Baumesberger  **
 **                                                                          **
 ** Description: Utouch3D Plugin                                             **
 **                                                                          **
 **                                                                          **
 ** Author: Anton Baumesberger	                                             **
 **                                                                          **
\****************************************************************************/

#include <vector>
#include <osg/Camera>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include "TuioCursor.h"
#include "TuioPoint.h"
#include "BlobVisualiser.h"
#include <cover/coVRPluginSupport.h>

typedef std::map<TUIO::TuioCursor *, osg::MatrixTransform *> Cursor2MatrixTransformMap;
typedef std::map<TUIO::TuioCursor *, BlobVisualiser *> Cursor2BlobVisualiserMap;

class Utouch3DPlugin;

class InteractionManager
{
public:
    InteractionManager();
    InteractionManager(Utouch3DPlugin *thePlugin, opencover::coVRPluginSupport *cover);
    ~InteractionManager();

    void addBlob(TUIO::TuioCursor *tcur);
    void removeBlob(TUIO::TuioCursor *tcur);
    void updateBlob(TUIO::TuioCursor *tcur);

    enum FilterType
    {
        NONE = 0,
        WEIGHTED_MOVING_AVERAGE = 1,
        ADAPTIVE_LOWPASS = 2
    };

    void setTwoTouchFilter(FilterType t);
    void setThreeTouchFilter(FilterType t);

private:
    Utouch3DPlugin *thePlugin;
    opencover::coVRPluginSupport *theCover;

    int windowWidth, windowHeight;

    /**
          * camera frustum parameter
          */
    double camLeft, camRight, camBottom, camTop, camNear, camFar;

    double camSurfaceDistance;

    FilterType twoTouchFilterType;
    FilterType threeTouchFilterType;

    /** mapping TuioCursor* to osg::MatrixTransform
          *  the graph has the following form:
          *  under the scene a seperate blob-camera is attached
          *  the camera holds a MatrixTransform-node(and its children) per TuioCursor
          *  graph: scene - blobCam - MT1 - Geode1 - BlobVisualiser1
          *                         - MT2 - Geode2 - BlobVisualiser2
          *                         ...
          */
    Cursor2MatrixTransformMap cursors2Subgraphs;

    Cursor2BlobVisualiserMap cursors2Blobs;

    bool isFirstBlob;

    TUIO::TuioPoint *oneTouchStartPoint;

    TUIO::TuioCursor *twoTouchReferenceCursorOne;
    TUIO::TuioCursor *twoTouchReferenceCursorTwo;
    TUIO::TuioCursor *twoTouchFilterCursorOne;
    TUIO::TuioCursor *twoTouchFilterCursorTwo;

    TUIO::TuioCursor *threeTouchReferenceCursorLeft;
    TUIO::TuioCursor *threeTouchReferenceCursorMiddle;
    TUIO::TuioCursor *threeTouchReferenceCursorRight;
    TUIO::TuioCursor *threeTouchFilterCursorLeft;
    TUIO::TuioCursor *threeTouchFilterCursorMiddle;
    TUIO::TuioCursor *threeTouchFilterCursorRight;

    osg::ref_ptr<osg::Camera> blobCam;

    void initBlobCamera(int width, int height);
    void showScreenPlane(bool b);
    void removeBlobs();

    bool blockInteraction;

    void handleOneBlob(TUIO::TuioCursor *tcur);
    void handleTwoBlobs(TUIO::TuioCursor *one, TUIO::TuioCursor *two);
    void handleThreeBlobs(TUIO::TuioCursor *left, TUIO::TuioCursor *middle, TUIO::TuioCursor *right);
    void handleThreeBlobs2(TUIO::TuioCursor *left, TUIO::TuioCursor *middle, TUIO::TuioCursor *right);
    void handleThreeBlobs3(TUIO::TuioCursor *left, TUIO::TuioCursor *middle, TUIO::TuioCursor *right);

    void sortCursorsByXVal(std::vector<TUIO::TuioCursor *> &cursors);

    float weightedMovingAverageAlpha;
    /**
          * applies a weigthed moving average filter to the vector
          * @param current based on the vector @param old
          */
    void filterByWeightedMovingAverage(osg::Vec2d &current, osg::Vec2d const old, float alpha);

    float adaptiveLowpassAlpha;
    /**
          * applies an adaptive lowpass filter to the vector
          * @param current based on the vector @param old,
          * their difference in seconds @param dt
          * and a time-based @param alpha
          */
    void filterByAdaptiveLowpass(osg::Vec2d &current, osg::Vec2d const old, float timeAlpha, float dt);

    // the current root transformation matrix
    osg::Matrix startRootXForm;

    // scale factor for depth translation
    float depthTranslationScale;
};

#endif // INTERACTIONMANAGER_H
