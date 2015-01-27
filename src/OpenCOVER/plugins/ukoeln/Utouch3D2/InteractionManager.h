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

#include "BlobVisualiser.h"
#include "TouchCursor.h"

#include <osg/Camera>
#include <osg/Geode>
#include <osg/MatrixTransform>

#include "coVRPluginSupport.h"

class Utouch3DPlugin;

class InteractionManager
{
public:
    InteractionManager(Utouch3DPlugin *thePlugin, opencover::coVRPluginSupport *cover);
    ~InteractionManager();

    void addBlob(TouchCursor *tcur);
    void removeBlob(TouchCursor *tcur);
    void updateBlob(TouchCursor *tcur);

    //    enum FilterType
    //    {
    //        NONE = 0,
    //        WEIGHTED_MOVING_AVERAGE = 1,
    //        ADAPTIVE_LOWPASS = 2,
    //    };

private:
    typedef std::map<TouchCursor *, osg::MatrixTransform *> Cursor2MatrixTransformMap;
    typedef std::map<TouchCursor *, BlobVisualiser *> Cursor2BlobVisualiserMap;

    Utouch3DPlugin *thePlugin;
    opencover::coVRPluginSupport *theCover;

    int windowWidth, windowHeight;

    /**
     * camera frustum parameter
     */
    double camLeft, camRight, camBottom, camTop, camNear, camFar;
    double camSurfaceDistance;

    //    FilterType twoTouchFilterType;
    //    FilterType threeTouchFilterType;

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

    //bool isFirstBlob;

    TouchCursor *oneTouchStartPoint;

    TouchCursor *twoTouchReferenceCursorOne;
    TouchCursor *twoTouchReferenceCursorTwo;
    TouchCursor *twoTouchFilterCursorOne;
    TouchCursor *twoTouchFilterCursorTwo;

    TouchCursor *threeTouchReferenceCursorLeft;
    TouchCursor *threeTouchReferenceCursorMiddle;
    TouchCursor *threeTouchReferenceCursorRight;
    TouchCursor *threeTouchFilterCursorLeft;
    TouchCursor *threeTouchFilterCursorMiddle;
    TouchCursor *threeTouchFilterCursorRight;

    void handleOneBlob(TouchCursor *tcur);
    void handleTwoBlobs(TouchCursor *one, TouchCursor *two);
    void handleThreeBlobs(TouchCursor *left, TouchCursor *middle, TouchCursor *right);
    void handleThreeBlobs2(TouchCursor *left, TouchCursor *middle, TouchCursor *right);
    void handleThreeBlobs3(TouchCursor *left, TouchCursor *middle, TouchCursor *right);

    void sortCursorsByXVal(vector<TouchCursor *> &cursors);

    osg::ref_ptr<osg::Camera> blobCam;

    void initBlobCamera(int width, int height);
    void showScreenPlane(bool b);
    void removeBlobs();

    bool blockInteraction;

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
    //void filterByAdaptiveLowpass(osg::Vec2d& current, osg::Vec2d const old, float timeAlpha, float dt);

    // the current root transformation matrix
    osg::Matrix startRootXForm;

    // scale factor for depth translation
    float depthTranslationScale;
};

#endif // INTERACTIONMANAGER_H
