/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                              (C)2010 Anton Baumesberger  **
 **                                                                          **
 ** Description: Utouch3D Plugin                                             **
 **                                                                          **
 **                                                                          **
 ** Author: Anton Baumesberger	                                             **
 **                                                                          **
\****************************************************************************/

#include "Utouch3DPlugin.h"
#include "InteractionManager.h"

#include <algorithm>

#include <osg/Matrix>

#include "config/CoviseConfig.h"

#include "coVRNavigationManager.h"
#include "coVRConfig.h"
#include "coRowMenuItem.h"

#include "VRSceneGraph.h"

using namespace osg;
using namespace opencover;

/*
    thomas comment 2010 July 08:
    kick out unneccessary buttons from toolbar
    orbiting gesture? e.g. four finger: three fingers from ND hand for orbit mode and DH finger for rotation;
    idea for future work: render additional info (textual, graphic) at the side of a selected star
*/
InteractionManager::InteractionManager(Utouch3DPlugin *plugin, opencover::coVRPluginSupport *cover)
    : thePlugin(plugin)
    , theCover(cover)
    , windowWidth(0)
    , windowHeight(0)
    , camLeft(0.0)
    , camRight(0.0)
    , camBottom(0.0)
    , camTop(0.0)
    , camNear(0.0)
    , camFar(1.0)
    , camSurfaceDistance(1.0)
    //    , twoTouchFilterType(NONE)
    //    , threeTouchFilterType(NONE)
    , oneTouchStartPoint(NULL)
    , twoTouchReferenceCursorOne(NULL)
    , twoTouchReferenceCursorTwo(NULL)
    , twoTouchFilterCursorOne(NULL)
    , twoTouchFilterCursorTwo(NULL)
    , threeTouchReferenceCursorLeft(NULL)
    , threeTouchReferenceCursorMiddle(NULL)
    , threeTouchReferenceCursorRight(NULL)
    , threeTouchFilterCursorLeft(NULL)
    , threeTouchFilterCursorMiddle(NULL)
    , threeTouchFilterCursorRight(NULL)
    , blobCam(NULL)
    , blockInteraction(false)
    , weightedMovingAverageAlpha(0.4f)
    , adaptiveLowpassAlpha(0.5f)
    , startRootXForm(osg::Matrixd::identity())
    , depthTranslationScale(4000.0f)
{
    //std::cout << "InteractionManager::InteractionManager()" << std::endl;

    //
    // get the window dimensions
    //
    // TODO:    check if it works for stereo mode using two screens
    //          it works for stereo quite well, so stereo/mono is handled
    //          in the same manner using only one camera
    //
    const GraphicsContext::Traits *traits = coVRConfig::instance()->windows[0].window->getTraits();

    windowWidth = traits->width;
    windowHeight = traits->height;

    //std::cout << "InteractionManager: windowWidth = " << windowWidth << std::endl;
    //std::cout << "InteractionManager: windowHeight = " << windowHeight << std::endl;

    //cout << "InteractionManager::initBlobCamera()" << endl;

    initBlobCamera(windowWidth, windowHeight);

    // get camera frustum parameters
    osg::Matrixd projMat = coVRConfig::instance()->screens[0].camera->getProjectionMatrix();

    projMat.getFrustum(camLeft, camRight, camBottom, camTop, camNear, camFar);

    //std::cout << "left: " << camLeft << " right: " << camRight << " top: " << camTop << " bottom: " << camBottom << std::endl;

    //std::cout << "... END InteractionManager::InteractionManager()" << std::endl;
}

InteractionManager::~InteractionManager()
{
    // clean up all blobs if any remain
    if (!cursors2Subgraphs.empty())
    {
        removeBlobs();
    }

    if (int numChildr = blobCam->getNumChildren() > 0)
    {
        blobCam->removeChildren(0, numChildr);
    }

    blobCam = NULL;
}

/**
 * Blob-Visualisation and mapping:
 *
 * creates the nodes bottom-up:
 * 1. BlobVisualiser
 * 2. Geode
 * 3. MatrixTransform
 * then, creates the subgraph from these nodes and adds it to the blobCamera
 * finally, adds the TuioCursor and its MatrixTransform to the cursors2Subgraph map
 *
 * manages interaction mode depending on the number of active cursors
 */
void InteractionManager::addBlob(TouchCursor *tcur)
{
    Vec3d eye, center, up;

    coVRConfig::instance()->screens[0].camera->getViewMatrixAsLookAt(eye, center, up);

    //cout << "viewerScreenDist: " << theCover->getViewerScreenDistance() << endl;
    //cout << "lookAt.Eye.x: " << eye.x() << endl;
    //cout << "lookAt.Eye.y: " << eye.y() << endl;
    //cout << "lookAt.Eye.z: " << eye.z() << endl;

    if (cursors2Subgraphs.empty())
    {
        camSurfaceDistance = -eye.y(); //(double)eye.y() * (double)-1;
    }

    //
    // 1. create a blob
    //

    BlobVisualiser *blob = new BlobVisualiser(
        osg::Vec3f(0.0f, 0.0f, 0.001f),
        10.0f,
        100,
        osg::Vec4f(1.0f, 1.0f, 1.0f, 0.5f));

    StateSet *blobStateSet = blob->getGeometry()->getOrCreateStateSet();

    blobStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    blobStateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    //
    // 2. create the geometry node for the blob
    //

    osg::Geode *blobGeode = new osg::Geode();

    blobGeode->addDrawable(blob->getGeometry());

    // turn lighting off and disable depth test to ensure its always ontop
    osg::StateSet *geodeStateset = blobGeode->getOrCreateStateSet();

    geodeStateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    //
    // 3. create a transformation matrix
    //

    osg::MatrixTransform *m = new osg::MatrixTransform;

    // translate to the position of the TuioCursor
    m->setMatrix(osg::Matrix::translate(tcur->x * windowWidth, (1 - tcur->y) * windowHeight, 0));

    // create the subgraph and add it to the blobCamera
    m->addChild(blobGeode);

    blobCam->addChild(m);

    // add the cursor and its MatrixTransform-node to the map
    cursors2Subgraphs[tcur] = m;

    // add the cursor and its BlobVisualiser-node to the map
    cursors2Blobs[tcur] = blob;

    // save the scene's current transformation matrix
    startRootXForm = theCover->getObjectsXform()->getMatrix();

    //
    // if it is the first blob, save the coordinates (start reference point)
    //

    if (/*isFirstBlob &&*/ cursors2Subgraphs.size() == 1)
    {
        oneTouchStartPoint = new TouchCursor(tcur);
        /*isFirstBlob = false;*/
    }

    //
    // if the second blob was added, save their coordinates (start reference points)
    // save the graph's current root transformation matrix
    //

    else if (cursors2Subgraphs.size() == 2)
    {
        Cursor2MatrixTransformMap::iterator it = cursors2Subgraphs.begin();

        TouchCursor *cur1 = it->first;
        ++it;
        TouchCursor *cur2 = it->first;

        // order reference points according to their ids
        // so that the referencePointOne is assigned the minor id's cursor
        if (cur1->id < cur2->id)
        {
            twoTouchReferenceCursorOne = new TouchCursor(cur1);
            twoTouchFilterCursorOne = new TouchCursor(cur1);
            twoTouchReferenceCursorTwo = new TouchCursor(cur2);
            twoTouchFilterCursorTwo = new TouchCursor(cur2);
        }
        else
        {
            twoTouchReferenceCursorOne = new TouchCursor(cur2);
            twoTouchFilterCursorOne = new TouchCursor(cur2);
            twoTouchReferenceCursorTwo = new TouchCursor(cur1);
            twoTouchFilterCursorTwo = new TouchCursor(cur1);
        }

        blockInteraction = false;
    }

    //
    // if the third blob was added, save their coordinates (start reference points)
    // thereby, the blobs are ordered by their x-value: leftmost, middle and rightmost blob
    //

    else if (cursors2Subgraphs.size() == 3)
    {
        std::vector<TouchCursor *> cursors(3);

        Cursor2MatrixTransformMap::iterator it = cursors2Subgraphs.begin();

        cursors.push_back(it->first);
        ++it;
        cursors.push_back(it->first);
        ++it;
        cursors.push_back(it->first);

        this->sortCursorsByXVal(cursors);

        TouchCursor *left = cursors[0];
        TouchCursor *middle = cursors[1];
        TouchCursor *right = cursors[2];

        threeTouchReferenceCursorLeft = new TouchCursor(left);
        threeTouchReferenceCursorMiddle = new TouchCursor(middle);
        threeTouchReferenceCursorRight = new TouchCursor(right);

        threeTouchFilterCursorLeft = new TouchCursor(left);
        threeTouchFilterCursorMiddle = new TouchCursor(middle);
        threeTouchFilterCursorRight = new TouchCursor(right);
    }
}

void InteractionManager::removeBlob(TouchCursor *tcur)
{
    // iterate over the cursors2Subgraphs map and remove blob(s) if the cursor(s) have been removed
    Cursor2MatrixTransformMap::iterator iter = cursors2Subgraphs.find(tcur);

    if (iter != cursors2Subgraphs.end())
    {
        osg::MatrixTransform *m = dynamic_cast<osg::MatrixTransform *>(iter->second);
        osg::Geode *blobGeode = dynamic_cast<osg::Geode *>(m->getChild(0));

        blobGeode->removeDrawables(0);

        // remove blobGeode
        m->removeChild(blobGeode);

        // remove MatrixTransform
        blobCam->removeChild(m);

        cursors2Subgraphs.erase(iter);
    }

    // cleanup BlobVisualiser objects
    Cursor2BlobVisualiserMap::iterator it = cursors2Blobs.find(tcur);
    if (it != cursors2Blobs.end())
    {
        BlobVisualiser *blob = it->second;
        cursors2Blobs.erase(it);
        delete blob;
    }

    // if just one cursor remains, set it to be the oneTouchReferencePoint
    // delete the twoTouchReferenceCursors if they're not NULL
    if (cursors2Subgraphs.size() == 1)
    {
        Cursor2BlobVisualiserMap::iterator it = cursors2Blobs.begin();

        TouchCursor *tc = dynamic_cast<TouchCursor *>(it->first);

        oneTouchStartPoint->x = tc->x;
        oneTouchStartPoint->y = tc->y;

        startRootXForm = theCover->getObjectsXform()->getMatrix();

        delete twoTouchReferenceCursorOne;
        twoTouchReferenceCursorOne = 0;

        delete twoTouchReferenceCursorTwo;
        twoTouchReferenceCursorTwo = 0;

        delete twoTouchFilterCursorOne;
        twoTouchFilterCursorOne = 0;

        delete twoTouchFilterCursorTwo;
        twoTouchFilterCursorTwo = 0;
    }

    // if two cursors remain, save their coords as the twoTouchReferenceCursors
    // and delete the threeTouchReferenceCursors and filtered cursors if they're not NULL
    else if (cursors2Subgraphs.size() == 2)
    {
        Cursor2BlobVisualiserMap::iterator it = cursors2Blobs.begin();

        TouchCursor *tc1 = dynamic_cast<TouchCursor *>(it->first);
        it++;
        TouchCursor *tc2 = dynamic_cast<TouchCursor *>(it->first);

        delete twoTouchReferenceCursorOne;
        twoTouchReferenceCursorOne = 0;
        delete twoTouchReferenceCursorTwo;
        twoTouchReferenceCursorTwo = 0;

        if (tc1->id < tc2->id)
        {
            twoTouchReferenceCursorOne = new TouchCursor(tc1);
            twoTouchReferenceCursorTwo = new TouchCursor(tc2);
        }
        else if (tc1->id > tc2->id)
        {
            twoTouchReferenceCursorOne = new TouchCursor(tc2);
            twoTouchReferenceCursorTwo = new TouchCursor(tc1);
        }

        delete threeTouchReferenceCursorLeft;
        threeTouchReferenceCursorLeft = 0;

        delete threeTouchReferenceCursorMiddle;
        threeTouchReferenceCursorMiddle = 0;

        delete threeTouchReferenceCursorRight;
        threeTouchReferenceCursorRight = 0;

        delete threeTouchFilterCursorLeft;
        threeTouchFilterCursorLeft = 0;

        delete threeTouchFilterCursorMiddle;
        threeTouchFilterCursorMiddle = 0;

        delete threeTouchFilterCursorRight;
        threeTouchFilterCursorRight = 0;

        blockInteraction = true;
    }

    // if three cursors remain sort them and set the respective referece-cursors and filter-cursors
    else if (cursors2Subgraphs.size() == 3)
    {
        std::vector<TouchCursor *> cursors(3);

        Cursor2MatrixTransformMap::iterator it = cursors2Subgraphs.begin();

        cursors.push_back(it->first);
        ++it;
        cursors.push_back(it->first);
        ++it;
        cursors.push_back(it->first);

        this->sortCursorsByXVal(cursors);

        TouchCursor *left = dynamic_cast<TouchCursor *>(cursors[0]);
        TouchCursor *middle = dynamic_cast<TouchCursor *>(cursors[1]);
        TouchCursor *right = dynamic_cast<TouchCursor *>(cursors[2]);

        delete threeTouchReferenceCursorLeft;
        threeTouchReferenceCursorLeft = 0;

        delete threeTouchReferenceCursorMiddle;
        threeTouchReferenceCursorMiddle = 0;

        delete threeTouchReferenceCursorRight;
        threeTouchReferenceCursorRight = 0;

        threeTouchReferenceCursorLeft = new TouchCursor(left);
        threeTouchReferenceCursorMiddle = new TouchCursor(middle);
        threeTouchReferenceCursorRight = new TouchCursor(right);

        delete threeTouchFilterCursorLeft;
        threeTouchFilterCursorLeft = 0;

        delete threeTouchFilterCursorMiddle;
        threeTouchFilterCursorMiddle = 0;

        delete threeTouchFilterCursorRight;
        threeTouchFilterCursorRight = 0;

        threeTouchFilterCursorLeft = new TouchCursor(left);
        threeTouchFilterCursorMiddle = new TouchCursor(middle);
        threeTouchFilterCursorRight = new TouchCursor(right);
    }

    // if no more cursors exist (i.e. the last touch point was removed),
    // remove oneTouchReferencePoint and send a faked mouse released event
    if (cursors2Subgraphs.empty())
    {
        delete oneTouchStartPoint;
        oneTouchStartPoint = 0;

        /*isFirstBlob = true;*/

        thePlugin->stopTouchInteraction();
    }
}

void InteractionManager::updateBlob(TouchCursor *tcur)
{
    // blob visualisation: iterate over the cursors2Subgraphs map and update blob(s)
    Cursor2MatrixTransformMap::iterator iter = cursors2Subgraphs.find(tcur);

    if (iter != cursors2Subgraphs.end())
    {
        osg::MatrixTransform *m = dynamic_cast<osg::MatrixTransform *>(iter->second);

        m->setMatrix(osg::Matrix::translate(tcur->x * this->windowWidth, (1 - tcur->y) * this->windowHeight, 0));
    }

    // handle different numbers of cursors
    switch (cursors2Subgraphs.size())
    {
    // if the sphrere-selection buttons "single select" or "multiple select" are pressed,
    // do not trigger one-touch navigation; otherwise do one-touch navigation
    case 1:
    {
        bool isSingleSelectionActive = false;
        bool isMultiSelectionActive = false;

        covise::coMenu *menu = theCover->getMenu();

        covise::coMenuItem *singleSelect = menu->getItemByName("single select");
        covise::coMenuItem *multiSelect = menu->getItemByName("multiple select");

        if (singleSelect != NULL)
        {
            covise::coCheckboxMenuItem *cbmi = dynamic_cast<covise::coCheckboxMenuItem *>(singleSelect);
            isSingleSelectionActive = cbmi->getState();
        }

        if (multiSelect != NULL)
        {
            covise::coCheckboxMenuItem *cbmi = dynamic_cast<covise::coCheckboxMenuItem *>(multiSelect);
            isMultiSelectionActive = cbmi->getState();
        }

        if (!isSingleSelectionActive && !isMultiSelectionActive)
        {
            this->handleOneBlob(tcur);
        }
    }
    break;

    case 2:
    {
        if (blockInteraction)
            return;

        Cursor2MatrixTransformMap::iterator itr = cursors2Subgraphs.begin();
        TouchCursor *c1 = dynamic_cast<TouchCursor *>(itr->first);
        itr++;
        TouchCursor *c2 = dynamic_cast<TouchCursor *>(itr->first);

        TouchCursor *one;
        TouchCursor *two;

        // sort one and two by cursorID
        if (c1->id < c2->id)
        {
            one = c1;
            two = c2;
        }
        else
        {
            one = c2;
            two = c1;
        }

        // if cursor one is updated
        if (tcur->id == twoTouchFilterCursorOne->id)
        {
            Vec2d tMinus1 = Vec2d(twoTouchFilterCursorOne->x, twoTouchFilterCursorOne->y);
            Vec2d t = Vec2d(tcur->x, tcur->y);

            this->handleTwoBlobs(one, two);
        }

        // if cursor two is updated
        else if (tcur->id == two->id)
        {
            Vec2d tMinus1 = Vec2d(twoTouchFilterCursorTwo->x, twoTouchFilterCursorTwo->y);
            Vec2d t = Vec2d(tcur->x, tcur->y);

            this->handleTwoBlobs(one, two);
        }

    } // end case 2
    break;

    case 3:
    {
        Cursor2MatrixTransformMap::iterator itr = cursors2Subgraphs.begin();

        TouchCursor *one = dynamic_cast<TouchCursor *>(itr->first);
        itr++;
        TouchCursor *two = dynamic_cast<TouchCursor *>(itr->first);
        itr++;
        TouchCursor *three = dynamic_cast<TouchCursor *>(itr->first);

        TouchCursor *left = 0;
        TouchCursor *middle = 0;
        TouchCursor *right = 0;

        int idLeft = threeTouchReferenceCursorLeft->id;
        int idMiddle = threeTouchReferenceCursorMiddle->id;
        int idRight = threeTouchReferenceCursorRight->id;

        // if one is left, two or three can be middle or right
        if (one->id == idLeft)
        {
            left = one;

            if (two->id == idMiddle && three->id == idRight)
            {
                middle = two;
                right = three;
            }
            else // if ( two->id == idRight && three->id == idMiddle )
            {
                middle = three;
                right = two;
            }
        }

        // if one is middle, two or three can be left or right
        else if (one->id == idMiddle)
        {
            middle = one;

            if (two->id == idLeft && three->id == idRight)
            {
                left = two;
                right = three;
            }
            else // if ( two->id == idRight && three->id == idLeft )
            {
                left = three;
                right = two;
            }
        }

        // if one is right, two or three can be left or middle
        else if (one->id == idRight)
        {
            right = one;

            if (two->id == idLeft && three->id == idMiddle)
            {
                left = two;
                middle = three;
            }
            else // if (two->id == idMiddle && three->id == idLeft)
            {
                left = three;
                middle = two;
            }
        }

        // check which of the three cursors is the currently updated and handle it
        if (tcur->id == idLeft)
        {
            Vec2d tMinus1 = Vec2d(threeTouchFilterCursorLeft->x, threeTouchFilterCursorLeft->y);
            Vec2d t = Vec2d(tcur->x, tcur->y);

            this->handleThreeBlobs2(left, middle, right);
        }

        else if (tcur->id == idMiddle)
        {
            Vec2d tMinus1 = Vec2d(threeTouchFilterCursorMiddle->x, threeTouchFilterCursorMiddle->y);
            Vec2d t = Vec2d(tcur->x, tcur->y);

            this->handleThreeBlobs2(left, middle, right);
        }

        else if (tcur->id == idRight)
        {
            Vec2d tMinus1 = Vec2d(threeTouchFilterCursorRight->x, threeTouchFilterCursorRight->y);
            Vec2d t = Vec2d(tcur->x, tcur->y);

            this->handleThreeBlobs2(left, middle, right);
        }

    } // end case 3
    break;

    default:
        break;
    } // end switch(cursors2Subgraphs.size())
}

/**
 *
 * in-plane rotation (see /doc/oneTouchRotation_sketch.pdf)
 *
 * first, calculates the rotation angle
 * using the theorem on intersecting lines
 *
 * next, the rotational transformation has three steps:
 *
 * 1. translate into rotationCenter (viewerPosition)
 * 2. rotate
 * 3. translate back
 *
 * finally, accumulates these transformations into one matrix (dcsMat)
 * and sets the root's transform-node matrix to that accumulated matrix
 */
void InteractionManager::handleOneBlob(TouchCursor *tcur)
{
    // center is onlz then (0.5, 0.5) if the viewer position is exactly in the middle of the screen
    // on the table, we are way more nearer to the bottom;
    // thus, center is calculated dynamically; in this way, head-tracking is possible
    double vertFrustum = (camTop + fabs(camBottom));
    double horizFrustum = (camRight + fabs(camLeft));

    double centerX = camRight / horizFrustum;
    double centerY = camTop / vertFrustum;

    Vec2f center = Vec2f(centerX, centerY);

    double viewerScreenDist = theCover->getViewerScreenDistance();

    double startDiffX = (double)oneTouchStartPoint->x - (double)center.x();
    double startDiffY = (double)oneTouchStartPoint->y - (double)center.y();

    // for stereo, we are interacting on the screen surface and not on the near plane
    // thus, calculate the motion on the surface
    // alos important: if the camera frustum is assymetric (in stereo depending on the user position)
    // as is the case on the table (vertically assymetric), (top + fabs(bottom))/2 has to be
    // calculated and if horizontally assymetric, (right + fabs(left))/2 is the horizontal component
    double surfaceRight = ((camRight + fabs(camLeft)) / 2) * viewerScreenDist / camNear;
    double surfaceTop = ((camTop + fabs(camBottom)) / 2) * viewerScreenDist / camNear;

    // computing startDiffx and startDiffY in world coords (using the theorem on intersecting lines)
    startDiffX = (startDiffX * viewerScreenDist * surfaceRight * 2) / viewerScreenDist;
    startDiffY = (startDiffY * viewerScreenDist * surfaceTop * 2) / viewerScreenDist;

    double currDiffX = (double)tcur->x - (double)center.x();
    double currDiffY = (double)tcur->y - (double)center.y();
    ;

    // for mono use the following formula for all vectors, calculating touch motion on the enar plane:
    //currDiffY = (currDiffY * theCover->getViewerScreenDistance() * camTop * 2) / camNear;

    // computing currDiffX and currDiffY in world coords (using the theorem on intersecting lines)
    currDiffX = (currDiffX * viewerScreenDist * surfaceRight * 2) / viewerScreenDist;
    currDiffY = (currDiffY * viewerScreenDist * surfaceTop * 2) / viewerScreenDist;

    double startDiffAngleX = atan2(startDiffX, viewerScreenDist);
    double startDiffAngleY = atan2(startDiffY, viewerScreenDist);

    double currDiffAngleX = atan2(currDiffX, viewerScreenDist);
    double currDiffAngleY = atan2(currDiffY, viewerScreenDist);

    double rotAngleX = startDiffAngleX - currDiffAngleX;
    double rotAngleZ = startDiffAngleY - currDiffAngleY;

    // matrices for translation, reverse translation, rotation and an accumulative matrix
    Matrix transM, revTrans, relRotMat, dcsMat;

    Vec3 viewerPosition = cover->getViewerMat().getTrans();

    // 1. translate into origin using the transformation matrix
    //    that has been saved when the blob was added
    transM.makeTranslate(-viewerPosition);
    revTrans.mult(startRootXForm, transM);

    // 2. setup rotation
    relRotMat.setTrans(Vec3(0.0, 0.0, 0.0));
    relRotMat.makeRotate(rotAngleZ, Vec3d(1.0, 0.0, 0.0), 0.0, Vec3d(0.0, 1.0, 0.0), rotAngleX, Vec3d(0.0, 0.0, 1.0));
    dcsMat.mult(revTrans, relRotMat);

    // 3. translate back
    transM.setTrans(viewerPosition);
    dcsMat.mult(dcsMat, transM);

    if (dcsMat.valid())
        VRSceneGraph::instance()->getTransform()->setMatrix(dcsMat);
}

/**
 * This function implements the common two-touch Rotate-Scale-Translate (RST) technique.
 * Rotation is calculated from the angle between the vector formed by the starting touch points
 * and the vector specified by the current touch points. The rotation center of the algorithm
 * is the first touch point, from the user's point of view, there is no difference which touch point
 * is first or second.
 * The ratio of the two aforementioned vectors defines the scale factor.
 * Scaling is performed uniformly and the scaling center is, again from the algorithm's
 * perspective, the first touch point.
 * Translation is defined by the movement vector of the initial first touch point
 * and the current first touch point.
 * Once these values are calculated, rotation, scaling and translation are
 * accumulated into one transformation matrix. This matrix is then assigned to the scene's
 * root transformation matrix (@code theCover->getObjectsXform()->getMatrix()).
 */
void InteractionManager::handleTwoBlobs(TouchCursor *one, TouchCursor *two)
{
    Vec2d a, a1, b, b1, b2, trans2CenterRel;
    Vec3d trans2CenterInWorld;

    Vec2d center = Vec2d(0.5, 0.5);
    double viewerScreenDist = theCover->getViewerScreenDistance();

    // assume that cursor one is the transformation center

    // calculate scalefactor
    a = Vec2d(twoTouchReferenceCursorOne->x, twoTouchReferenceCursorOne->y);
    b = Vec2d(twoTouchReferenceCursorTwo->x, twoTouchReferenceCursorTwo->y);
    a1 = Vec2d(one->x, one->y);
    b1 = Vec2d(two->x, two->y);

    b2 = b + (a1 - a);

    double scaleFactor = (b1 - a1).length() / (b - a).length();

    // calculate relative difference of cursor one to center (translation to transf. center)
    trans2CenterRel = a - center;
    double transInWorldX, transInWorldZ;

    // for stereo, we are interacting on the screen surface and not on the near plane
    // thus, calculate the motion on the surface
    // alos important: if the camera frustum is assymetric (in stereo depending on the user position)
    // as is the case on the table (vertically assymetric), (top + fabs(bottom))/2 has to be
    // calculated and if horizontally assymetric, (right + fabs(left))/2 is the horizontal component
    double surfaceRight = ((camRight + fabs(camLeft)) / 2) * viewerScreenDist / camNear;
    double surfaceTop = ((camTop + fabs(camBottom)) / 2) * viewerScreenDist / camNear;

    // MONO: calculate translation offset on near plane to transformation center in world coords
    //    transInWorldX = trans2CenterRel.x() * theCover->getViewerScreenDistance() * 2 * camRight / camNear;
    //    transInWorldZ = trans2CenterRel.y() * theCover->getViewerScreenDistance() * 2 * camTop / camNear;

    transInWorldX = trans2CenterRel.x() * viewerScreenDist * 2 * surfaceRight / viewerScreenDist;
    transInWorldZ = trans2CenterRel.y() * viewerScreenDist * 2 * surfaceTop / viewerScreenDist;

    trans2CenterInWorld = Vec3d(-transInWorldX, 0, transInWorldZ);

    // calculate rotation angle using dot product: a * b = |a| |b| cos(alpha)
    double alpha;
    Vec2d bMinusA, b1MinusA1;

    bMinusA = b - a;
    b1MinusA1 = b1 - a1;
    bMinusA.normalize();
    b1MinusA1.normalize();

    alpha = acos((bMinusA * b1MinusA1)); // / bMinusA.length() * b1MinusA1.length() );

    // handle quadrant switch, calculating the appropriate rotation direction:
    // the dot product is positive in the top and negative in the bottom quadrants
    // we need distinction between the left and the right quadrants, so we use
    // the dot product of the vector and the normal; this provides the desired
    // left negative and right positive quadrants
    Vec2d normal = Vec2d(-b1MinusA1.y(), b1MinusA1.x());
    double skalProdNormal = bMinusA * normal;

    if (skalProdNormal > 0)
        alpha = -alpha;

    // calculate translation on surface in world coords
    Vec2d t = a1 - a;
    double newX, newZ;
    newX = t.x() * viewerScreenDist * 2 * surfaceRight / viewerScreenDist;
    newZ = t.y() * viewerScreenDist * 2 * surfaceTop / viewerScreenDist;

    // accumulate transformations: translate into transf. center, scale, rotate and then translate back
    // finally translate to the new position
    Matrixd trans2Center, scale, rotation, trans, dcs;

    // scale
    scale.setTrans(0.0, 0.0, 0.0);
    scale.makeScale(scaleFactor, scaleFactor, scaleFactor);

    // rotate about depth-axis
    rotation.setTrans(0.0, 0.0, 0.0);
    rotation.makeRotate(alpha, Vec3d(0.0, 1.0, 0.0));

    // translate into transformation center
    trans2Center.makeTranslate(trans2CenterInWorld);

    // accumulate translation into transf. center, scaling and rotation
    dcs.mult(startRootXForm, trans2Center);
    dcs.mult(dcs, scale);
    dcs.mult(dcs, rotation);

    // reverse translation from the transformation center
    trans2Center.makeTranslate(-trans2CenterInWorld);
    dcs.mult(dcs, trans2Center);

    // accumulate translation
    trans.makeTranslate(newX, 0.0, -newZ);
    dcs.mult(dcs, trans);

    // set the scene's transformation matrix if the calculated matrix is valid
    if (dcs.valid())
        VRSceneGraph::instance()->getTransform()->setMatrix(dcs);
}

/**
 * iTAO as planned (rotating about the camera)
 */
void InteractionManager::handleThreeBlobs(TouchCursor *left, TouchCursor *middle, TouchCursor *right)
{
    // calculate translation
    Vec2d start = Vec2d(threeTouchReferenceCursorMiddle->x, threeTouchReferenceCursorMiddle->y);
    Vec2d curr = Vec2d(middle->x, middle->y);
    Vec2d t = curr - start;

    //    cout << "x-trans: " << t.x() << endl;
    //    if (absolute(t.x()) < 0.02) t.set(0.0, t.y());

    // x-translation is sticky, if *-1 then 'counter-movement'
    double newX = -1 * (t.x() * theCover->getViewerScreenDistance() * 2 * camRight / camNear);

    // y-translation has to be scaled somehow, here with a constant scale factor
    double depthTransScale = theCover->getViewerScreenDistance(); //1000.0;
    double newY = t.y() * depthTransScale;

    // rotation is calculated from the vector right - left
    // the angle is defined by the angle between (right-left) and (threeTouchRefCurRight - threeTouchRefCurLeft)
    // the rotation center is the world origin
    Vec2d aRight = Vec2d(threeTouchReferenceCursorRight->x, threeTouchReferenceCursorRight->y);
    Vec2d aLeft = Vec2d(threeTouchReferenceCursorLeft->x, threeTouchReferenceCursorLeft->y);
    Vec2d a = aRight - aLeft;
    Vec2d bRight = Vec2d(right->x, right->y);
    Vec2d bLeft = Vec2d(left->x, left->y);
    Vec2d b = bRight - bLeft;
    a.normalize();
    b.normalize();

    double rotScale = 1.0;

    // calculate rotation angle using the dot product a * b = |a| |b| cos(alpha)
    double alpha = rotScale * acos((a * b) / a.length() * b.length());

    // determine rotation direction from the scalar product
    // of the start vector a and the normal on the current vector b
    Vec2d bNormal = Vec2d(b.y(), -b.x());
    if (a * bNormal < 0)
        alpha = -alpha;

    // accumulate rotation and translation
    Matrixd trans2Center, rot, trans, dcs;

    trans2Center.makeTranslate(-(theCover->getViewerMat().getTrans()));
    rot.makeTranslate(0.0, 0.0, 0.0);
    rot.makeRotate(alpha, Vec3d(0.0, 0.0, 1.0));

    dcs.mult(startRootXForm, trans2Center);
    dcs.mult(dcs, rot);

    trans2Center.makeTranslate((theCover->getViewerMat().getTrans()));
    dcs.mult(dcs, trans2Center);

    trans.makeTranslate(newX, newY, 0.0);
    dcs.mult(dcs, trans);

    if (dcs.valid())
        VRSceneGraph::instance()->getTransform()->setMatrix(dcs);
}

/**
 * right finger controls depth-translation
 * angle between left and middle finger control rotation about the world origin
 */
void InteractionManager::handleThreeBlobs2(TouchCursor *left, TouchCursor *middle, TouchCursor *right)
{
    float rotAngleScale = 0.27;

    // calculate translation
    Vec2d start = Vec2d(threeTouchReferenceCursorRight->x, threeTouchReferenceCursorRight->y);
    Vec2d curr = Vec2d(right->x, right->y);
    Vec2d t = curr - start;

    // x-translation is sticky
    double newX = (t.x() * theCover->getViewerScreenDistance() * 2 * camRight / camNear) * -1;

    // y-translation has to be scaled somehow, here with a constant scale factor
    double newY = t.y() * depthTranslationScale;

    // rotation is calculated from the vector middle - left
    // the angle is defined by the angle between (middle-left) and (threeTouchRefCurMiddle - threeTouchRefCurLeft)
    // the rotation center is the world origin
    Vec2d aMiddle = Vec2d(threeTouchReferenceCursorMiddle->x, threeTouchReferenceCursorMiddle->y);
    Vec2d aLeft = Vec2d(threeTouchReferenceCursorLeft->x, threeTouchReferenceCursorLeft->y);
    Vec2d a = aMiddle - aLeft;
    Vec2d bMiddle = Vec2d(middle->x, middle->y);
    Vec2d bLeft = Vec2d(left->x, left->y);
    Vec2d b = bMiddle - bLeft;
    a.normalize();
    b.normalize();

    // calculate rotation angle using the dot product a * b = |a| |b| cos(alpha)
    double alpha = acos((a * b)); // / (a.length() * b.length()) );
    // scale rotation angle
    alpha *= rotAngleScale;

    // determine rotation direction from the scalar product
    // of the start vector a and the normal on the current vector b
    Vec2d bNormal = Vec2d(b.y(), -b.x());
    if (a * bNormal > 0)
        alpha = -alpha;

    // accumulate rotation and translation
    Matrixd trans2Center, rot, trans, dcs;

    trans2Center.makeTranslate(-(theCover->getViewerMat().getTrans()));
    rot.makeTranslate(0.0, 0.0, 0.0);
    rot.makeRotate(alpha, Vec3d(0.0, 0.0, 1.0));

    dcs.mult(startRootXForm, trans2Center);
    dcs.mult(dcs, rot);

    trans2Center.makeTranslate((theCover->getViewerMat().getTrans()));
    dcs.mult(dcs, trans2Center);

    trans.makeTranslate(-newX, -newY, 0.0);
    dcs.mult(dcs, trans);

    if (dcs.valid())
        VRSceneGraph::instance()->getTransform()->setMatrix(dcs);
}

/**
 * middle finger controls depth-translation
 * angle between left and right finger control rotation about the world origin
 */
void InteractionManager::handleThreeBlobs3(TouchCursor *left, TouchCursor *middle, TouchCursor *right)
{
    // calculate translation
    Vec2d start = Vec2d(threeTouchReferenceCursorMiddle->x, threeTouchReferenceCursorMiddle->y);
    Vec2d curr = Vec2d(middle->x, middle->y);
    Vec2d t = curr - start;

    // x-translation is sticky, if *-1 then 'counter-movement'
    double newX = /*-1 * */ (t.x() * theCover->getViewerScreenDistance() * 2 * camRight / camNear);

    // y-translation has to be scaled somehow, here with a constant scale factor
    double newY = t.y() * depthTranslationScale;

    // rotation is calculated from the vector right - left
    // the angle is defined by the angle between (right-left) and (threeTouchRefCurRight - threeTouchRefCurLeft)
    // the rotation center is the world origin
    Vec2d aRight = Vec2d(threeTouchReferenceCursorRight->x, threeTouchReferenceCursorRight->y);
    Vec2d aLeft = Vec2d(threeTouchReferenceCursorLeft->x, threeTouchReferenceCursorLeft->y);
    Vec2d a = aRight - aLeft;
    Vec2d bRight = Vec2d(right->x, right->y);
    Vec2d bLeft = Vec2d(left->x, left->y);
    Vec2d b = bRight - bLeft;
    a.normalize();
    b.normalize();

    double rotScale = 1.0;

    // calculate rotation angle using the dot product a * b = |a| |b| cos(alpha)
    double alpha = rotScale * acos((a * b)); // / (a.length() * b.length()) );

    // determine rotation direction from the scalar product
    // of the start vector a and the normal on the current vector b
    Vec2d bNormal = Vec2d(b.y(), -b.x());
    if (a * bNormal < 0)
        alpha = -alpha;

    // accumulate rotation and translation
    Matrixd trans2Center, rot, trans, dcs;

    //    trans2Center.makeTranslate(-(theCover->getViewerMat().getTrans()));
    rot.makeTranslate(0.0, 0.0, 0.0);
    rot.makeRotate(-alpha, Vec3d(0.0, 0.0, 1.0));

    dcs.mult(startRootXForm, trans2Center);
    dcs.mult(dcs, rot);

    //    trans2Center.makeTranslate((theCover->getViewerMat().getTrans()));
    dcs.mult(dcs, trans2Center);

    trans.makeTranslate(newX, -newY, 0.0);
    //    dcs.mult(startRootXForm, dcs);
    dcs.mult(dcs, trans);

    if (dcs.valid())
        VRSceneGraph::instance()->getTransform()->setMatrix(dcs);
}

namespace
{
struct LessThan : public std::binary_function<TouchCursor *, TouchCursor *, bool>
{
    inline bool operator()(TouchCursor *a, TouchCursor *b)
    {
        return a->x < b->x;
    }
};
} // namespace

void InteractionManager::sortCursorsByXVal(vector<TouchCursor *> &cursors)
{
    std::sort(cursors.begin(), cursors.end(), LessThan());
}

void InteractionManager::filterByWeightedMovingAverage(osg::Vec2d &current, osg::Vec2d const old, float alpha)
{
    float newX = (current.x() * alpha) + (old.x() * (1.0f - alpha));
    float newY = (current.y() * alpha) + (old.y() * (1.0f - alpha));

    current.set(newX, newY);
}

void InteractionManager::removeBlobs()
{
    Cursor2MatrixTransformMap::iterator iter;

    for (iter = cursors2Subgraphs.begin(); iter != cursors2Subgraphs.end(); iter++)
    {
        osg::MatrixTransform *m = dynamic_cast<osg::MatrixTransform *>(iter->second);
        osg::Geode *blobGeode = dynamic_cast<osg::Geode *>(m->getChild(0));

        blobGeode->removeDrawables(0);

        // remove blobGeode
        m->removeChild(blobGeode);

        // remove MatrixTransform
        blobCam->removeChild(m);
    }

    cursors2Subgraphs.clear();

    Cursor2BlobVisualiserMap::iterator it;

    for (it = cursors2Blobs.begin(); it != cursors2Blobs.end(); it++)
    {
        BlobVisualiser *blob = it->second;
        delete blob;
    }

    cursors2Blobs.clear();
}

void InteractionManager::initBlobCamera(int width, int height)
{
    // create the camera for the blobs
    blobCam = new osg::Camera;

    blobCam->setProjectionMatrix(osg::Matrix::ortho2D(0, width, 0, height));
    blobCam->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);

    // set the view matrix
    blobCam->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    blobCam->setViewMatrix(osg::Matrix::identity());

    // only clear the depth buffer
    blobCam->setClearMask(GL_DEPTH_BUFFER_BIT);

    // draw subgraph after main camera view.
    blobCam->setRenderOrder(osg::Camera::POST_RENDER);

    // add the camera to the scene
    theCover->getScene()->addChild(blobCam.get());
}

/**
 * renders a transparent plane on the world origin that is parallel to the near/far plane
 * debugging purpose
 */
void InteractionManager::showScreenPlane(bool /*b*/)
{
#if 0
    float width = 400.0;
    float height = 300.0;

    ref_ptr<Geode> screenGeode = new Geode();

    if(b)
    {
        ref_ptr<Geometry> screenGeom = new Geometry();

        ref_ptr<Vec3Array> vertices = new Vec3Array();
        vertices->push_back(Vec3(-width, 0.0, -height));
        vertices->push_back(Vec3(width, 0.0, -height));
        vertices->push_back(Vec3(width, 0.0, height));
        vertices->push_back(Vec3(-width, 0.0, height));
        screenGeom->setVertexArray(vertices);

        ref_ptr<Vec4Array> colors = new Vec4Array();
        colors->push_back(Vec4(0.0, 1.0, 1.0, 0.2));
        screenGeom->setColorArray(colors);
        screenGeom->setColorBinding(Geometry::BIND_OVERALL);

        ref_ptr<Vec3Array> normals = new Vec3Array();
        normals->push_back(Vec3(0.0,1.0, 0.0));
        screenGeom->setNormalArray(normals);
        screenGeom->setNormalBinding(Geometry::BIND_OVERALL);

        /* enable transparency */
        // Enable blending, select transparent bin.
        screenGeom->getOrCreateStateSet()->setMode( GL_BLEND, osg::StateAttribute::ON );
        screenGeom->getOrCreateStateSet()->setRenderingHint( osg::StateSet::TRANSPARENT_BIN );
        // Enable depth test so that an opaque polygon will occlude a transparent one behind it.
        screenGeom->getOrCreateStateSet()->setMode( GL_DEPTH_TEST, osg::StateAttribute::ON );
        // Disable conflicting modes.
        screenGeom->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );


        screenGeom->addPrimitiveSet(new DrawArrays(PrimitiveSet::TRIANGLE_FAN, 0, vertices->size()));
        screenGeode->addDrawable(screenGeom);
        theCover->getScene()->addChild(screenGeode.get());
    }
    else
    {

    }
#endif
}
