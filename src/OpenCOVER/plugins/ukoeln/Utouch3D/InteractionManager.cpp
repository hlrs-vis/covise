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
#include "FakedMouseEvent.h"

#include <cover/coVRNavigationManager.h>
#include <config/CoviseConfig.h>
#include <cover/coVRConfig.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coRowMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <cover/VRSceneGraph.h>

#include <algorithm>

using namespace osg;
using namespace vrui;
using namespace TUIO;
using namespace opencover;

/*
    thomas comment 2010 July 08:
    kick out unneccessary buttons from toolbar
    orbiting gesture? e.g. four finger: three fingers from ND hand for orbit mode and DH finger for rotation;
    idea for future work: render additional info (textual, graphic) at the side of a selected star
*/
InteractionManager::InteractionManager()
{
}

InteractionManager::InteractionManager(Utouch3DPlugin *plugin, opencover::coVRPluginSupport *cover)
{
    //    cout << "InteractionManager::InteractionManager()" << endl;

    thePlugin = plugin;
    theCover = cover;

    // get the window dimensions
    // TODO: check if it works for stereo mode using two screens
    // it works for stereo quite well, so stereo/mono is handled
    // in the same manner using only one camera
    const GraphicsContext::Traits *traits = coVRConfig::instance()->windows[0].window->getTraits();
    windowWidth = traits->width;
    windowHeight = traits->height;

    twoTouchFilterType = NONE;
    threeTouchFilterType = NONE;

    this->initBlobCamera(windowWidth, windowHeight);

    this->setTwoTouchFilter(WEIGHTED_MOVING_AVERAGE);
    this->setThreeTouchFilter(WEIGHTED_MOVING_AVERAGE);

    isFirstBlob = true;
    blockInteraction = false;

    weightedMovingAverageAlpha = 0.4f;
    adaptiveLowpassAlpha = 0.5f;

    depthTranslationScale = 4000;

    // transparent debug plane on the zero parallax plane
    //    this->showScreenPlane(true);

    // get camera frustum parameters
    Matrixd projMat = coVRConfig::instance()->channels[0].camera->getProjectionMatrix();
    projMat.getFrustum(this->camLeft, this->camRight, this->camBottom, this->camTop, this->camNear, this->camFar);

    cout << "left: " << camLeft << " right: " << camRight << endl;
    cout << "top: " << camTop << " bottom: " << camBottom << endl;
}

InteractionManager::~InteractionManager()
{
    // clean up all blobs if any remain
    if (!cursors2Subgraphs.empty())
        removeBlobs();

    if (int numChildr = blobCam->getNumChildren() > 0)
        blobCam->removeChildren(0, numChildr);
    blobCam = NULL;
}

void InteractionManager::setTwoTouchFilter(FilterType t)
{
    twoTouchFilterType = t;
}

void InteractionManager::setThreeTouchFilter(FilterType t)
{
    threeTouchFilterType = t;
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
void InteractionManager::addBlob(TuioCursor *tcur)
{
    Vec3d eye, center, up;
    coVRConfig::instance()->channels[0].camera->getViewMatrixAsLookAt(eye, center, up);

    //    cout << "viewerScreenDist: " << theCover->getViewerScreenDistance() << endl;

    //    cout << "lookAt.Eye.x: " << eye.x() << endl;
    //    cout << "lookAt.Eye.y: " << eye.y() << endl;
    //    cout << "lookAt.Eye.z: " << eye.z() << endl;

    camSurfaceDistance = (double)eye.y() * (double)-1;

    /* 1. create a blob */
    osg::Vec3f blobCenter = osg::Vec3f(0.0f, 0.0f, 0.001f);
    float blobRadius = 10;
    int numOfSegments = 100;
    osg::Vec4f blobColor = osg::Vec4f(1.0f, 1.0f, 1.0f, 0.5f);
    BlobVisualiser *blob = new BlobVisualiser(blobCenter, blobRadius, numOfSegments, blobColor);
    StateSet *blobStateSet = blob->getGeometry().get()->getOrCreateStateSet();
    blobStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    blobStateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    /* 2. create the geometry node for the blob */
    osg::Geode *blobGeode = new osg::Geode();
    // turn lighting off and disable depth test to ensure its always ontop
    osg::StateSet *geodeStateset = blobGeode->getOrCreateStateSet();
    geodeStateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    /* 3.create a transformation matrix */
    osg::MatrixTransform *m = new osg::MatrixTransform;
    // translate to the position of the TuioCursor
    m->setMatrix(osg::Matrix::translate(tcur->getX() * this->windowWidth, (1 - tcur->getY()) * this->windowHeight, 0));

    // create the subgraph and add it to the blobCamera
    blobGeode->addDrawable(blob->getGeometry().get());
    m->addChild(blobGeode);
    blobCam->addChild(m);

    // add the cursor and its MatrixTransform-node to the map
    cursors2Subgraphs.insert(pair<TuioCursor *, MatrixTransform *>(tcur, m));

    // add the cursor and its BlobVisualiser-node to the map
    cursors2Blobs.insert(pair<TuioCursor *, BlobVisualiser *>(tcur, blob));

    // save the scene's current transformation matrix
    startRootXForm = theCover->getObjectsXform()->getMatrix();

    // if it is the first blob, save the coordinates (start reference point)
    if (isFirstBlob && cursors2Subgraphs.size() == 1)
    {
        oneTouchStartPoint = new TuioPoint(tcur->getX(), tcur->getY());
        isFirstBlob = false;

        const float currX = tcur->getX();
        const float currY = tcur->getY();

        thePlugin->startTouchInteraction();

        // slider works if push is sent before move; otherwise slider doesn't react
        // mouse over visualisation on the toolbar works only if move is sent before push

        //        FakedMouseEvent* move = new FakedMouseEvent(osgGA::GUIEventAdapter::MOVE, currX * windowWidth, (1 - currY) * windowHeight);
        //        thePlugin->insertFakedMouseEvent(move);
        FakedMouseEvent *push = new FakedMouseEvent(osgGA::GUIEventAdapter::PUSH, 1 /*1 means push state*/, (int)(1 - currY) * windowHeight);
        thePlugin->insertFakedMouseEvent(push);
        FakedMouseEvent *move = new FakedMouseEvent(osgGA::GUIEventAdapter::MOVE, (int)currX * windowWidth, (int)(1 - currY) * windowHeight);
        thePlugin->insertFakedMouseEvent(move);
    }

    // if the second blob was added, save their coordinates (start reference points)
    // save the graph's current root transformation matrix
    if (cursors2Subgraphs.size() == 2)
    {
        Cursor2MatrixTransformMap::iterator itr = cursors2Subgraphs.begin();
        TuioCursor *cur1 = dynamic_cast<TuioCursor *>(itr->first);
        itr++;
        TuioCursor *cur2 = dynamic_cast<TuioCursor *>(itr->first);

        // order reference points according to their ids
        // so that the referencePointOne is assigned the minor id's cursor
        if (cur1->getCursorID() < cur2->getCursorID())
        {
            twoTouchReferenceCursorOne = new TuioCursor(cur1);
            twoTouchFilterCursorOne = new TuioCursor(cur1);
            twoTouchReferenceCursorTwo = new TuioCursor(cur2);
            twoTouchFilterCursorTwo = new TuioCursor(cur2);
        }
        else
        {
            twoTouchReferenceCursorOne = new TuioCursor(cur2);
            twoTouchFilterCursorOne = new TuioCursor(cur2);
            twoTouchReferenceCursorTwo = new TuioCursor(cur1);
            twoTouchFilterCursorTwo = new TuioCursor(cur1);
        }

        blockInteraction = false;
    }

    // if the third blob was added, save their coordinates (start reference points)
    // thereby, the blobs are ordered by their x-value: leftmost, middle and rightmost blob
    else if (cursors2Subgraphs.size() == 3)
    {
        Cursor2MatrixTransformMap::iterator itr = cursors2Subgraphs.begin();
        TuioCursor *cur1 = dynamic_cast<TuioCursor *>(itr->first);
        itr++;
        TuioCursor *cur2 = dynamic_cast<TuioCursor *>(itr->first);
        itr++;
        TuioCursor *cur3 = dynamic_cast<TuioCursor *>(itr->first);

        vector<TuioCursor *> cursors(3);
        cursors[0] = cur1;
        cursors[1] = cur2;
        cursors[2] = cur3;

        this->sortCursorsByXVal(cursors);

        TuioCursor *left = dynamic_cast<TuioCursor *>(cursors[0]);
        TuioCursor *middle = dynamic_cast<TuioCursor *>(cursors[1]);
        TuioCursor *right = dynamic_cast<TuioCursor *>(cursors[2]);

        threeTouchReferenceCursorLeft = new TuioCursor(left);
        threeTouchReferenceCursorMiddle = new TuioCursor(middle);
        threeTouchReferenceCursorRight = new TuioCursor(right);

        threeTouchFilterCursorLeft = new TuioCursor(left);
        threeTouchFilterCursorMiddle = new TuioCursor(middle);
        threeTouchFilterCursorRight = new TuioCursor(right);

        //        cout << "left.id: " << left->getCursorID() << endl;
        //        cout << "ref.left.id: " << threeTouchReferenceCursorLeft->getCursorID() << endl;
        //        cout << "middle.id: " << middle->getCursorID() << endl;
        //        cout << "ref.middle.id: " << threeTouchReferenceCursorMiddle->getCursorID() << endl;
        //        cout << "right.id: " << right->getCursorID() << endl;
        //        cout <<"ref.right.id: " << threeTouchReferenceCursorRight->getCursorID() << endl;
    }
}

void InteractionManager::removeBlob(TuioCursor *tcur)
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
        TuioCursor *tc = dynamic_cast<TuioCursor *>(it->first);

        oneTouchStartPoint->update(tc->getX(), tc->getY());
        startRootXForm = theCover->getObjectsXform()->getMatrix();

        if (twoTouchReferenceCursorOne != NULL)
            delete twoTouchReferenceCursorOne;
        if (twoTouchReferenceCursorTwo != NULL)
            delete twoTouchReferenceCursorTwo;

        if (twoTouchFilterCursorOne != NULL)
            delete twoTouchFilterCursorOne;
        if (twoTouchFilterCursorTwo != NULL)
            delete twoTouchFilterCursorTwo;

        FakedMouseEvent *release = new FakedMouseEvent(osgGA::GUIEventAdapter::RELEASE, 0, (int)(1 - tcur->getY()) * windowHeight);
        thePlugin->insertFakedMouseEvent(release);

        // if two cursors remain, save their coords as the twoTouchReferenceCursors
        // and delete the threeTouchReferenceCursors and filtered cursors if they're not NULL
    }
    else if (cursors2Subgraphs.size() == 2)
    {
        Cursor2BlobVisualiserMap::iterator it = cursors2Blobs.begin();
        TuioCursor *tc1 = dynamic_cast<TuioCursor *>(it->first);
        it++;
        TuioCursor *tc2 = dynamic_cast<TuioCursor *>(it->first);

        delete twoTouchReferenceCursorOne;
        delete twoTouchReferenceCursorTwo;

        if (tc1->getCursorID() < tc2->getCursorID())
        {
            twoTouchReferenceCursorOne = new TuioCursor(tc1);
            twoTouchReferenceCursorTwo = new TuioCursor(tc2);
        }
        else if (tc1->getCursorID() > tc2->getCursorID())
        {
            twoTouchReferenceCursorOne = new TuioCursor(tc2);
            twoTouchReferenceCursorTwo = new TuioCursor(tc1);
        }

        if (threeTouchReferenceCursorLeft != NULL)
            delete threeTouchReferenceCursorLeft;
        if (threeTouchReferenceCursorMiddle != NULL)
            delete threeTouchReferenceCursorMiddle;
        if (threeTouchReferenceCursorRight != NULL)
            delete threeTouchReferenceCursorRight;

        if (threeTouchFilterCursorLeft != NULL)
            delete threeTouchFilterCursorLeft;
        if (threeTouchFilterCursorMiddle != NULL)
            delete threeTouchFilterCursorMiddle;
        if (threeTouchFilterCursorRight != NULL)
            delete threeTouchFilterCursorRight;

        blockInteraction = true;
    }

    // if three cursors remain sort them and set the respective referece-cursors and filter-cursors
    else if (cursors2Subgraphs.size() == 3)
    {
        Cursor2BlobVisualiserMap::iterator it = cursors2Blobs.begin();
        TuioCursor *tc1 = dynamic_cast<TuioCursor *>(it->first);
        it++;
        TuioCursor *tc2 = dynamic_cast<TuioCursor *>(it->first);
        it++;
        TuioCursor *tc3 = dynamic_cast<TuioCursor *>(it->first);

        vector<TuioCursor *> cursors(3);
        cursors[0] = tc1;
        cursors[1] = tc2;
        cursors[2] = tc3;

        this->sortCursorsByXVal(cursors);

        TuioCursor *left = dynamic_cast<TuioCursor *>(cursors[0]);
        TuioCursor *middle = dynamic_cast<TuioCursor *>(cursors[1]);
        TuioCursor *right = dynamic_cast<TuioCursor *>(cursors[2]);

        if (threeTouchReferenceCursorLeft != NULL)
            delete threeTouchReferenceCursorLeft;
        if (threeTouchReferenceCursorMiddle != NULL)
            delete threeTouchReferenceCursorMiddle;
        if (threeTouchReferenceCursorRight != NULL)
            delete threeTouchReferenceCursorRight;

        threeTouchReferenceCursorLeft = new TuioCursor(left);
        threeTouchReferenceCursorMiddle = new TuioCursor(middle);
        threeTouchReferenceCursorRight = new TuioCursor(right);

        if (threeTouchFilterCursorLeft != NULL)
            delete threeTouchFilterCursorLeft;
        if (threeTouchFilterCursorMiddle != NULL)
            delete threeTouchFilterCursorMiddle;
        if (threeTouchFilterCursorRight != NULL)
            delete threeTouchFilterCursorRight;

        threeTouchFilterCursorLeft = new TuioCursor(left);
        threeTouchFilterCursorMiddle = new TuioCursor(middle);
        threeTouchFilterCursorRight = new TuioCursor(right);
    }

    // if no more cursors exist (i.e. the last touch point was removed),
    // remove oneTouchReferencePoint and send a faked mouse released event
    if (cursors2Subgraphs.empty())
    {
        delete oneTouchStartPoint;
        isFirstBlob = true;

        //        FakedMouseEvent* release = new FakedMouseEvent(osgGA::GUIEventAdapter::RELEASE, 0 , (1 - tcur->getY()) * windowHeight);
        //        thePlugin->insertFakedMouseEvent(release);

        //        FakedMouseEvent* move = new FakedMouseEvent(osgGA::GUIEventAdapter::MOVE, 0, 0);
        //        thePlugin->insertFakedMouseEvent(move);

        thePlugin->stopTouchInteraction();
    }
}

void InteractionManager::updateBlob(TuioCursor *tcur)
{
    // blob visualisation: iterate over the cursors2Subgraphs map and update blob(s)
    Cursor2MatrixTransformMap::iterator iter = cursors2Subgraphs.find(tcur);
    if (iter != cursors2Subgraphs.end())
    {
        osg::MatrixTransform *m = dynamic_cast<osg::MatrixTransform *>(iter->second);
        m->setMatrix(osg::Matrix::translate(tcur->getX() * this->windowWidth, (1 - tcur->getY()) * this->windowHeight, 0));
    }

    // handle different numbers of cursors
    switch (cursors2Subgraphs.size())
    {
    // if the sphrere-selection buttons "single select" or "multiple select" are pressed,
    // do not trigger one-touch navigation; otherwise do one-touch navigation
    case 1:
    {

        //            FakedMouseEvent* move = new FakedMouseEvent(osgGA::GUIEventAdapter::MOVE, tcur->getX() * windowWidth, (1 - tcur->getY()) * windowHeight);
        //            thePlugin->insertFakedMouseEvent(move);

        bool isSingleSelectionActive = false;
        bool isMultiSelectionActive = false;

        vrui::coMenu *menu = theCover->getMenu();

        vrui::coMenuItem *singleSelect = menu->getItemByName("single select");
        vrui::coMenuItem *multiSelect = menu->getItemByName("multiple select");

        if (singleSelect != NULL)
        {
            vrui::coCheckboxMenuItem *cbmi = dynamic_cast<vrui::coCheckboxMenuItem *>(singleSelect);
            isSingleSelectionActive = cbmi->getState();
        }

        if (multiSelect != NULL)
        {
            vrui::coCheckboxMenuItem *cbmi = dynamic_cast<vrui::coCheckboxMenuItem *>(multiSelect);
            isMultiSelectionActive = cbmi->getState();
        }

        if (!isSingleSelectionActive && !isMultiSelectionActive)
            this->handleOneBlob(tcur);

        FakedMouseEvent *move = new FakedMouseEvent(osgGA::GUIEventAdapter::MOVE, (int)tcur->getX() * windowWidth, (int)(1 - tcur->getY()) * windowHeight);
        thePlugin->insertFakedMouseEvent(move);
        FakedMouseEvent *release = new FakedMouseEvent(osgGA::GUIEventAdapter::RELEASE, 0, (int)(1 - tcur->getY()) * windowHeight);
        thePlugin->insertFakedMouseEvent(release);

        //        if(thePlugin->isTouchInteractionRunning())
        //        std::cerr << "touch running" << std::endl;

        //            if ((thePlugin->isTouchInteractionRunning()))
        //                this->handleOneBlob(tcur);
    }
    break;

    case 2:
    {
        if (blockInteraction)
            return;

        // send the faked release event so that two-touch interaction doesn't interfere with selection
        FakedMouseEvent *release = new FakedMouseEvent(osgGA::GUIEventAdapter::RELEASE, 0, (int)(1 - tcur->getY()) * windowHeight);
        thePlugin->insertFakedMouseEvent(release);

        Cursor2MatrixTransformMap::iterator itr = cursors2Subgraphs.begin();
        TuioCursor *c1 = dynamic_cast<TuioCursor *>(itr->first);
        itr++;
        TuioCursor *c2 = dynamic_cast<TuioCursor *>(itr->first);

        TuioCursor *one;
        TuioCursor *two;

        // sort one and two by cursorID
        if (c1->getCursorID() < c2->getCursorID())
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
        if (tcur->getCursorID() == twoTouchFilterCursorOne->getCursorID())
        {
            Vec2d tMinus1 = Vec2d(twoTouchFilterCursorOne->getX(), twoTouchFilterCursorOne->getY());
            Vec2d t = Vec2d(tcur->getX(), tcur->getY());

            switch (twoTouchFilterType)
            {
            case NONE:
                this->handleTwoBlobs(one, two);
                break;

            case WEIGHTED_MOVING_AVERAGE:
            {
                this->filterByWeightedMovingAverage(t, tMinus1, weightedMovingAverageAlpha);

                // create filteredCursor from filtered vector t
                delete twoTouchFilterCursorOne;
                twoTouchFilterCursorOne = new TuioCursor(tcur->getSessionID(), tcur->getCursorID(), t.x(), t.y());

                this->handleTwoBlobs(twoTouchFilterCursorOne, two);
            }
            break;

            case ADAPTIVE_LOWPASS:
            {
                float oldTime = twoTouchFilterCursorOne->getTuioTime().getTotalMilliseconds() * 1000;
                float currentTime = tcur->getTuioTime().getTotalMilliseconds() * 1000;
                float dt = currentTime - oldTime;
                this->filterByAdaptiveLowpass(t, tMinus1, adaptiveLowpassAlpha, dt);

                // create filteredCursor from filtered vector t
                delete twoTouchFilterCursorOne;
                twoTouchFilterCursorOne = new TuioCursor(tcur->getSessionID(), tcur->getCursorID(), t.x(), t.y());

                this->handleTwoBlobs(twoTouchFilterCursorOne, two);
            }

            } // end switch(twoTouchFilterType)

        } // end if

        // if cursor two is updated
        else if (tcur->getCursorID() == two->getCursorID())
        {
            Vec2d tMinus1 = Vec2d(twoTouchFilterCursorTwo->getX(), twoTouchFilterCursorTwo->getY());
            Vec2d t = Vec2d(tcur->getX(), tcur->getY());

            switch (twoTouchFilterType)
            {
            case NONE:
                this->handleTwoBlobs(one, two);
                break;

            case WEIGHTED_MOVING_AVERAGE:
            {
                this->filterByWeightedMovingAverage(t, tMinus1, weightedMovingAverageAlpha);

                // create filteredCursor from filtered vector t
                delete twoTouchFilterCursorTwo;
                twoTouchFilterCursorTwo = new TuioCursor(tcur->getSessionID(), tcur->getCursorID(), t.x(), t.y());

                this->handleTwoBlobs(one, twoTouchFilterCursorTwo);
            }
            break;

            case ADAPTIVE_LOWPASS:
            {
                float oldTime = twoTouchFilterCursorTwo->getTuioTime().getTotalMilliseconds() * 1000;
                float currentTime = tcur->getTuioTime().getTotalMilliseconds() * 1000;
                float dt = currentTime - oldTime;
                this->filterByAdaptiveLowpass(t, tMinus1, adaptiveLowpassAlpha, dt);

                // create filteredCursor from filtered vector t
                delete twoTouchFilterCursorTwo;
                twoTouchFilterCursorTwo = new TuioCursor(tcur->getSessionID(), tcur->getCursorID(), t.x(), t.y());

                this->handleTwoBlobs(one, twoTouchFilterCursorTwo);
            }

            } // end switch(twoTouchFilterType)

        } // end if (tcur->getCursorID() == twoTouchFilterCursorOne->getCursorID())
        ;
    } // end case 2
    break;

    case 3:
    {
        // send the faked release event so that three-touch interaction doesn't interfere with selection
        FakedMouseEvent *release = new FakedMouseEvent(osgGA::GUIEventAdapter::RELEASE, 0, (int)(1 - tcur->getY()) * windowHeight);
        thePlugin->insertFakedMouseEvent(release);

        Cursor2MatrixTransformMap::iterator itr = cursors2Subgraphs.begin();
        TuioCursor *one = dynamic_cast<TuioCursor *>(itr->first);
        itr++;
        TuioCursor *two = dynamic_cast<TuioCursor *>(itr->first);
        itr++;
        TuioCursor *three = dynamic_cast<TuioCursor *>(itr->first);

        TuioCursor *left;
        TuioCursor *middle;
        TuioCursor *right;

        int idLeft = threeTouchReferenceCursorLeft->getCursorID();
        int idMiddle = threeTouchReferenceCursorMiddle->getCursorID();
        int idRight = threeTouchReferenceCursorRight->getCursorID();

        // if one is left, two or three can be middle or right
        if (one->getCursorID() == idLeft)
        {
            left = one;

            if (two->getCursorID() == idMiddle && three->getCursorID() == idRight)
            {
                middle = two;
                right = three;
            }
            else if (two->getCursorID() == idRight && three->getCursorID() == idMiddle)
            {
                middle = three;
                right = two;
            }
        }

        // if one is middle, two or three can be left or right
        else if (one->getCursorID() == idMiddle)
        {
            middle = one;

            if (two->getCursorID() == idLeft && three->getCursorID() == idRight)
            {
                left = two;
                right = three;
            }
            else if (two->getCursorID() == idRight && three->getCursorID() == idLeft)
            {
                left = three;
                right = two;
            }
        }

        // if one is right, two or three can be left or middle
        else if (one->getCursorID() == idRight)
        {
            right = one;

            if (two->getCursorID() == idLeft && three->getCursorID() == idMiddle)
            {
                left = two;
                middle = three;
            }
            else if (two->getCursorID() == idMiddle && three->getCursorID() == idLeft)
            {
                left = three;
                middle = two;
            }
        }

        // check which of the three cursors is the currently updated and handle it
        if (tcur->getCursorID() == idLeft)
        {
            Vec2d tMinus1 = Vec2d(threeTouchFilterCursorLeft->getX(), threeTouchFilterCursorLeft->getY());
            Vec2d t = Vec2d(tcur->getX(), tcur->getY());

            switch (threeTouchFilterType)
            {
            case NONE:
                this->handleThreeBlobs2(left, middle, right);
                break;

            case WEIGHTED_MOVING_AVERAGE:
            {
                this->filterByWeightedMovingAverage(t, tMinus1, weightedMovingAverageAlpha);

                // create filteredCursor from filtered vector t
                delete threeTouchFilterCursorLeft;
                threeTouchFilterCursorLeft = new TuioCursor(tcur->getSessionID(), tcur->getCursorID(), t.x(), t.y());

                this->handleThreeBlobs2(threeTouchFilterCursorLeft, middle, right);
            }
            break;

            case ADAPTIVE_LOWPASS:
            {
                float oldTime = threeTouchFilterCursorLeft->getTuioTime().getTotalMilliseconds() * 1000;
                float currentTime = tcur->getTuioTime().getTotalMilliseconds() * 1000;
                float dt = currentTime - oldTime;
                this->filterByAdaptiveLowpass(t, tMinus1, adaptiveLowpassAlpha, dt);

                delete threeTouchFilterCursorLeft;
                threeTouchFilterCursorLeft = new TuioCursor(tcur->getSessionID(), tcur->getCursorID(), t.x(), t.y());

                this->handleThreeBlobs2(threeTouchFilterCursorLeft, middle, right);
            }
            break;

            } // end switch(threeTouchFilterType)

        } // end if (tcur->getCursorID() == idLeft)

        else if (tcur->getCursorID() == idMiddle)
        {
            Vec2d tMinus1 = Vec2d(threeTouchFilterCursorMiddle->getX(), threeTouchFilterCursorMiddle->getY());
            Vec2d t = Vec2d(tcur->getX(), tcur->getY());

            switch (threeTouchFilterType)
            {
            case NONE:
                this->handleThreeBlobs2(left, middle, right);
                break;

            case WEIGHTED_MOVING_AVERAGE:
            {
                this->filterByWeightedMovingAverage(t, tMinus1, weightedMovingAverageAlpha);

                // create filteredCursor from filtered vector t
                delete threeTouchFilterCursorMiddle;
                threeTouchFilterCursorMiddle = new TuioCursor(tcur->getSessionID(), tcur->getCursorID(), t.x(), t.y());

                this->handleThreeBlobs2(left, threeTouchFilterCursorMiddle, right);
            }
            break;

            case ADAPTIVE_LOWPASS:
            {
                float oldTime = threeTouchFilterCursorMiddle->getTuioTime().getTotalMilliseconds() * 1000;
                float currentTime = tcur->getTuioTime().getTotalMilliseconds() * 1000;
                float dt = currentTime - oldTime;
                this->filterByAdaptiveLowpass(t, tMinus1, adaptiveLowpassAlpha, dt);

                delete threeTouchFilterCursorMiddle;
                threeTouchFilterCursorMiddle = new TuioCursor(tcur->getSessionID(), tcur->getCursorID(), t.x(), t.y());

                this->handleThreeBlobs2(left, threeTouchFilterCursorMiddle, right);
            }
            break;

            } // end switch(threeTouchFilterType)

        } // end else if(tcur->getCursorID() == idMiddle)

        else if (tcur->getCursorID() == idRight)
        {
            Vec2d tMinus1 = Vec2d(threeTouchFilterCursorRight->getX(), threeTouchFilterCursorRight->getY());
            Vec2d t = Vec2d(tcur->getX(), tcur->getY());

            switch (threeTouchFilterType)
            {
            case NONE:
                this->handleThreeBlobs2(left, middle, right);
                break;

            case WEIGHTED_MOVING_AVERAGE:
            {
                this->filterByWeightedMovingAverage(t, tMinus1, weightedMovingAverageAlpha);

                // create filteredCursor from filtered vector t
                delete threeTouchFilterCursorRight;
                threeTouchFilterCursorRight = new TuioCursor(tcur->getSessionID(), tcur->getCursorID(), t.x(), t.y());

                this->handleThreeBlobs2(left, middle, threeTouchFilterCursorRight);
            }
            break;

            case ADAPTIVE_LOWPASS:
            {
                float oldTime = threeTouchFilterCursorRight->getTuioTime().getTotalMilliseconds() * 1000;
                float currentTime = tcur->getTuioTime().getTotalMilliseconds() * 1000;
                float dt = currentTime - oldTime;
                this->filterByAdaptiveLowpass(t, tMinus1, adaptiveLowpassAlpha, dt);

                delete threeTouchFilterCursorRight;
                threeTouchFilterCursorRight = new TuioCursor(tcur->getSessionID(), tcur->getCursorID(), t.x(), t.y());

                this->handleThreeBlobs2(left, middle, threeTouchFilterCursorRight);
            }
            break;

            } // end switch(threeTouchFilterType)

        } // end else if(tcur->getCursorID() == idRight)

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
void InteractionManager::handleOneBlob(TuioCursor *tcur)
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

    double startDiffX = (double)oneTouchStartPoint->getX() - (double)center.x();
    double startDiffY = (double)oneTouchStartPoint->getY() - (double)center.y();

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

    double currDiffX = (double)tcur->getX() - (double)center.x();
    double currDiffY = (double)tcur->getY() - (double)center.y();
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
void InteractionManager::handleTwoBlobs(TuioCursor *one, TuioCursor *two)
{
    Vec2d a, a1, b, b1, b2, trans2CenterRel;
    Vec3d trans2CenterInWorld;

    Vec2d center = Vec2d(0.5, 0.5);
    double viewerScreenDist = theCover->getViewerScreenDistance();

    // assume that cursor one is the transformation center

    // calculate scalefactor
    a = Vec2d(twoTouchReferenceCursorOne->getX(), twoTouchReferenceCursorOne->getY());
    b = Vec2d(twoTouchReferenceCursorTwo->getX(), twoTouchReferenceCursorTwo->getY());
    a1 = Vec2d(one->getX(), one->getY());
    b1 = Vec2d(two->getX(), two->getY());

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

    alpha = acos(bMinusA * b1MinusA1 / bMinusA.length() * b1MinusA1.length());

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
void InteractionManager::handleThreeBlobs(TuioCursor *left, TuioCursor *middle, TuioCursor *right)
{
    // calculate translation
    Vec2d start = Vec2d(threeTouchReferenceCursorMiddle->getX(), threeTouchReferenceCursorMiddle->getY());
    Vec2d curr = Vec2d(middle->getX(), middle->getY());
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
    Vec2d aRight = Vec2d(threeTouchReferenceCursorRight->getX(), threeTouchReferenceCursorRight->getY());
    Vec2d aLeft = Vec2d(threeTouchReferenceCursorLeft->getX(), threeTouchReferenceCursorLeft->getY());
    Vec2d a = aRight - aLeft;
    Vec2d bRight = Vec2d(right->getX(), right->getY());
    Vec2d bLeft = Vec2d(left->getX(), left->getY());
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
void InteractionManager::handleThreeBlobs2(TuioCursor *left, TuioCursor *middle, TuioCursor *right)
{
    float rotAngleScale = 0.27;

    // calculate translation
    Vec2d start = Vec2d(threeTouchReferenceCursorRight->getX(), threeTouchReferenceCursorRight->getY());
    Vec2d curr = Vec2d(right->getX(), right->getY());
    Vec2d t = curr - start;

    // x-translation is sticky
    double newX = (t.x() * theCover->getViewerScreenDistance() * 2 * camRight / camNear) * -1;

    // y-translation has to be scaled somehow, here with a constant scale factor
    double newY = t.y() * depthTranslationScale;

    // rotation is calculated from the vector middle - left
    // the angle is defined by the angle between (middle-left) and (threeTouchRefCurMiddle - threeTouchRefCurLeft)
    // the rotation center is the world origin
    Vec2d aMiddle = Vec2d(threeTouchReferenceCursorMiddle->getX(), threeTouchReferenceCursorMiddle->getY());
    Vec2d aLeft = Vec2d(threeTouchReferenceCursorLeft->getX(), threeTouchReferenceCursorLeft->getY());
    Vec2d a = aMiddle - aLeft;
    Vec2d bMiddle = Vec2d(middle->getX(), middle->getY());
    Vec2d bLeft = Vec2d(left->getX(), left->getY());
    Vec2d b = bMiddle - bLeft;
    a.normalize();
    b.normalize();

    // calculate rotation angle using the dot product a * b = |a| |b| cos(alpha)
    double alpha = acos((a * b) / a.length() * b.length());
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
void InteractionManager::handleThreeBlobs3(TuioCursor *left, TuioCursor *middle, TuioCursor *right)
{
    // calculate translation
    Vec2d start = Vec2d(threeTouchReferenceCursorMiddle->getX(), threeTouchReferenceCursorMiddle->getY());
    Vec2d curr = Vec2d(middle->getX(), middle->getY());
    Vec2d t = curr - start;

    // x-translation is sticky, if *-1 then 'counter-movement'
    double newX = /*-1 * */ (t.x() * theCover->getViewerScreenDistance() * 2 * camRight / camNear);

    // y-translation has to be scaled somehow, here with a constant scale factor
    double newY = t.y() * depthTranslationScale;

    // rotation is calculated from the vector right - left
    // the angle is defined by the angle between (right-left) and (threeTouchRefCurRight - threeTouchRefCurLeft)
    // the rotation center is the world origin
    Vec2d aRight = Vec2d(threeTouchReferenceCursorRight->getX(), threeTouchReferenceCursorRight->getY());
    Vec2d aLeft = Vec2d(threeTouchReferenceCursorLeft->getX(), threeTouchReferenceCursorLeft->getY());
    Vec2d a = aRight - aLeft;
    Vec2d bRight = Vec2d(right->getX(), right->getY());
    Vec2d bLeft = Vec2d(left->getX(), left->getY());
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

/**
 * sorts a vector<TuioCursor*> of cursors by their x-value
 */
void InteractionManager::sortCursorsByXVal(vector<TuioCursor *> &cursors)
{
    // save all cursors mapped to their x-values
    map<float, TuioCursor *> cursorsX2Cur;

    // array for sorting the x-values
    float *xVals = new float[cursors.size()];

    // fill the map and the array
    for (size_t i = 0; i < cursors.size(); i++)
    {
        TuioCursor *c = dynamic_cast<TuioCursor *>(cursors[i]);
        cursorsX2Cur.insert(pair<float, TuioCursor *>(c->getX(), c));
        xVals[i] = c->getX();
    }

    // sort the array
    std::sort(xVals, xVals + cursors.size());

    // fill the result vector with the cursors of the resprective sorted x-values
    for (size_t j = 0; j < cursors.size(); j++)
    {
        map<float, TuioCursor *>::iterator it = cursorsX2Cur.find(xVals[j]);
        if (it != cursorsX2Cur.end())
        {
            cursors[j] = it->second;
        }
    }
    delete[] xVals;
}

void InteractionManager::filterByWeightedMovingAverage(osg::Vec2d &current, osg::Vec2d const old, float alpha)
{
    float newX = (current.x() * alpha) + (old.x() * (1.0f - alpha));
    float newY = (current.y() * alpha) + (old.y() * (1.0f - alpha));

    current.set(newX, newY);
}

void InteractionManager::filterByAdaptiveLowpass(osg::Vec2d &current, osg::Vec2d const old, float timeAlpha, float dt)
{
    float const filterConstant = dt / (dt + timeAlpha);
    float alpha = filterConstant;

    // adaptive filter
    static float const kMinStep = 0.02f;
    static float const kNoiseAttenuation = 3.0f;

    float d = std::abs(current.length() - old.length()) / kMinStep - 1.0f;
    d = std::max<>(0.0f, std::min<>(d, 1.0f));

    alpha = (1.0f - d) * filterConstant / kNoiseAttenuation + d * filterConstant;

    // interpolate
    current = old * alpha + current * (1.0f - alpha);
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
void InteractionManager::showScreenPlane(bool b)
{
    float width = 400.0;
    float height = 300.0;

    ref_ptr<Geode> screenGeode = new Geode();

    if (b)
    {
        ref_ptr<Geometry> screenGeom = new Geometry();

        ref_ptr<Vec3Array> vertices = new Vec3Array();
        vertices->push_back(Vec3(-width, 0.0, -height));
        vertices->push_back(Vec3(width, 0.0, -height));
        vertices->push_back(Vec3(width, 0.0, height));
        vertices->push_back(Vec3(-width, 0.0, height));
        screenGeom->setVertexArray(vertices.get());

        ref_ptr<Vec4Array> colors = new Vec4Array();
        colors->push_back(Vec4(0.0, 1.0, 1.0, 0.2));
        screenGeom->setColorArray(colors.get());
        screenGeom->setColorBinding(Geometry::BIND_OVERALL);

        ref_ptr<Vec3Array> normals = new Vec3Array();
        normals->push_back(Vec3(0.0, 1.0, 0.0));
        screenGeom->setNormalArray(normals.get());
        screenGeom->setNormalBinding(Geometry::BIND_OVERALL);

        /* enable transparency */
        // Enable blending, select transparent bin.
        screenGeom->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
        screenGeom->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        // Enable depth test so that an opaque polygon will occlude a transparent one behind it.
        screenGeom->getOrCreateStateSet()->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
        // Disable conflicting modes.
        screenGeom->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

        screenGeom->addPrimitiveSet(new DrawArrays(PrimitiveSet::TRIANGLE_FAN, 0, vertices->size()));
        screenGeode->addDrawable(screenGeom.get());
        theCover->getScene()->addChild(screenGeode.get());
    }
    else
    {
    }
}
