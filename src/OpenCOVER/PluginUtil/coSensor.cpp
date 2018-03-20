/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <coSensor.h>
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRCollaboration.h>
#include <cover/coVRMSController.h>
#include <cover/coVRCommunication.h>
#include <cover/OpenCOVER.h>
#include <OpenVRUI/coUpdateManager.h>
#include <OpenVRUI/sginterface/vruiButtons.h>
#include <config/CoviseConfig.h>
#include <cover/coHud.h>
#include <assert.h>
#include <util/coStringMultiHash.h>
#include <cover/VRViewer.h>
#include <cover/coTabletUI.h>
#include <cover/coIntersection.h>
#ifdef DOTIMING
#include <util/coTimer.h>
#endif

#include <osg/MatrixTransform>
#include <osg/Texture2D>
#include <osg/DrawPixels>
#include <osg/BoundingSphere>
#include <osg/Geode>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/TexGen>
#include <osgDB/ReadFile>
#include <osg/LineSegment>
#include <osgUtil/IntersectionVisitor>
#include <osgText/Font>
#include <util/unixcompat.h>
#include <util/coHashIter.h>

#ifdef __DARWIN_OSX__
#include <Carbon/Carbon.h>
#endif

// undef this to get no START/END messaged
#undef VERBOSE

#ifdef VERBOSE
#define START(msg) fprintf(stderr, "START: %s\n", (msg))
#define END(msg) fprintf(stderr, "END:   %s\n", (msg))
#else
#define START(msg)
#define END(msg)
#endif

using namespace vrui;
using namespace opencover;

static const unsigned ButtonMask = vruiButtons::ACTION_BUTTON | vruiButtons::DRIVE_BUTTON | vruiButtons::XFORM_BUTTON;

coSensor::coSensor(osg::Node *n)
{
    START("coSensor::coSensor");
    pathLength = 0;
    path = NULL;
    node = n;
    active = 0;
    osg::BoundingSphere bsphere;
    bsphere = node->getBound();
    threshold = bsphere.radius();
    threshold *= threshold;
    sqrDistance = 0;
    buttonSensitive = 0;
    enabled = 1;
}

void coSensor::setButtonSensitive(int s)
{
    START("coSensor::setButtonSensitive");
    buttonSensitive = s;
}

coIsectSensor::coIsectSensor(osg::Node *n)
    : coSensor(n)
{
    START("coIsectSensor::coIsectSensor");
    /// make intersection line segment
    p0.set(0.0f, 0.0f, 0.0f);
    p1.set(0.0f, 2.0f * cover->getSceneSize(), 0.0f);
    pointer = cover->getPointer();
    bsphere = n->getBound();
}

void coIsectSensor::calcDistance()
{
    START("coIsectSensor::calcDistance");
    osg::Vec3 q0, q1;
    osg::Matrix m;
    m = cover->getPointerMat();
    q0 = m.getTrans();
    q1 = m.preMult(p1);

    osg::Matrix mat;
    osg::Vec3 v;
    mat.makeIdentity();
    int i;
    float Scale2 = 1.0; // Scale ^2

    osg::Vec3 sVec;
    if (pathLength > 0)
    {
        for (i = 0; i < pathLength; i++)
        {
            osg::MatrixTransform *mt = path[i]->asMatrixTransform();
            if (mt)
            {
                mat.postMult(mt->getMatrix());
            }
        }
    }
    else
    {
        osg::Node *tmpnode = node;
        osg::Transform *trans = NULL;
        while (tmpnode)
        {
            if ((trans = tmpnode->asTransform()))
                break;
            tmpnode = tmpnode->getParent(0);
        }
        if (trans)
        {
            trans->asTransform()->computeLocalToWorldMatrix(mat, NULL);
        }
    }
    osg::Vec3 svec;
    svec[0] = mat(0, 0);
    svec[1] = mat(0, 1);
    svec[2] = mat(0, 2);
    Scale2 = sVec * sVec; // Scale ^2
    if (node->asTransform())
    {
        v = mat.getTrans();
    }
    else
        v = mat.preMult(bsphere.center());

    osg::Vec3 H, N;
    H = v - q0;
    N = q1 - q0;
    H.normalize();
    N.normalize();
    osg::Vec3 dist;
    dist = v - q0;
	float h=1.0;
	if(fabs(Scale2) > 0.0000001f)
	{
#ifdef WIN32
#pragma warning ( suppress : 4723)
#endif
	    h = dist.length2() / Scale2;
	}
    float sprod = H * (N);
    sprod *= sprod;

    sqrDistance = fabs(h * sprod - h);
}

void coTouchSensor::calcDistance()
{
    START("coTouchSensor::calcDistance")
    osg::Vec3 p;
    p = cover->getPointerMat().getTrans();
    sqrDistance = cover->getSqrDistance(node, p, path, pathLength);
}

void coProximitySensor::calcDistance()
{
    START("coProximitySensor::calcDistance")
    osg::Vec3 p;
    osg::Matrix m = cover->getViewerMat();
    p = m.getTrans();
    sqrDistance = cover->getSqrDistance(node, p, path, pathLength);
}

int coSensor::getType()
{
    START("coSensor::getType");
    return (NONE);
}

int coTouchSensor::getType()
{
    START("coTouchSensor::getType");
    return (coSensor::TOUCH);
}

coPickSensor::coPickSensor(osg::Node *n)
    : coSensor(n)
{
    START("coPickSensor::coPickSensor");
    vNode = new OSGVruiNode(n);
    vruiIntersection::getIntersectorForAction("coAction")->add(vNode, this);
    hitPoint.set(0, 0, 0);
    hitActive = false;
}

coPickSensor::~coPickSensor()
{
    vruiIntersection::getIntersectorForAction("coAction")->remove(vNode);
    delete vNode;
}
/**
@param hitPoint,hit  Performer intersection information
@return ACTION_CALL_ON_MISS if you want miss to be called,
otherwise ACTION_DONE is returned
*/
int coPickSensor::hit(vruiHit *hit)
{

    hitActive = true;
    coVector v = hit->getWorldIntersectionPoint();
    hitPoint.set(v[0], v[1], v[2]);
    return ACTION_CALL_ON_MISS;
}

/// Miss is called once after a hit, if the button is not intersected anymore.
void coPickSensor::miss()
{
    hitActive = false;
}

coHandSensor::coHandSensor(osg::Node *n)
    : coSensor(n)
{
    START("coHandSensor::coHandSensor");
    hitPoint.set(0, 0, 0);
    threshold = 10.0;
}

int coPickSensor::getType()
{
    START("coPickSensor::getType");
    return (coSensor::PICK);
}

int coHandSensor::getType()
{
    START("coHandSensor::getType");
    return (coSensor::HAND);
}

int coProximitySensor::getType()
{
    START("coProximitySensor::getType");
    return (coSensor::PROXIMITY);
}

int coIsectSensor::getType()
{
    START("coIsectSensor::getType");
    return (coSensor::ISECT);
}

void coPickSensor::update()
{
    //START("coPickSensor::update");
    int newActiveFlag;

    if (!enabled)
    {
        if (active)
        {
            disactivate();
            active = 0;
        }
    }

    if (hitActive)
        newActiveFlag = 1;
    else
        newActiveFlag = 0;

    if (newActiveFlag != active)
    {
        if (buttonSensitive && (cover->getPointerButton()->getState() & ButtonMask))
        {
            return; // do not change state, if any button is currently pressed
        }
        active = newActiveFlag;
        if (active)
            activate();
        else
            disactivate();
    }
}

void coHandSensor::update()
{
    //START("coHandSensor::update");
    int newActiveFlag;
    osg::Vec3 start[3], end[3];
    osg::Matrix handMat;
    handMat = cover->getPointerMat();
    start[0].set(0.0f, -threshold, 0.0f);
    end[0].set(0.0f, threshold, 0.0f);

    start[1].set(-threshold, 0.0f, 0.0f);
    end[1].set(threshold, 0.0f, 0.0f);

    start[2].set(0.0f, 0.0f, -threshold);
    end[2].set(0.0f, 0.0f, threshold);

    osg::Node *parent;
    osg::Matrix tr;
    osg::Matrix transformMat;
    tr.makeIdentity();
    parent = node->getParent(0);
    while (parent != NULL)
    {
        if (dynamic_cast<osg::MatrixTransform *>(parent))
        {
            transformMat = (dynamic_cast<osg::MatrixTransform *>(parent))->getMatrix();
            tr.postMult(transformMat);
        }
        if (parent->getNumParents())
            parent = parent->getParent(0);
        else
            parent = NULL;
    }
    osg::Matrix trinv;
    if (!trinv.invert(tr))
        fprintf(stderr, "coHandSensor::update tr is singular\n");
    ;
    trinv.preMult(handMat);

    // xform the intersection line segment
    for (int i=0; i<3; ++i)
    {
        start[i] = trinv.preMult(start[i]);
        end[i] = trinv.preMult(end[i]);
    }

    osg::ref_ptr<coIntersector> intersector[3];
    osg::ref_ptr<osgUtil::IntersectorGroup> igroup = new osgUtil::IntersectorGroup;
    for (int i=0; i<3; ++i)
    {
        intersector[i] = coIntersection::instance()->newIntersector(start[i], end[i]);
        igroup->addIntersector(intersector[i]);
    }

    bool haveIsect = false;
    if (enabled)
    {
        osgUtil::IntersectionVisitor visitor(igroup);
        node->accept(visitor);

        for (int i=0; i<3; ++i)
        {
            if (intersector[i]->containsIntersections())
            {
                haveIsect = true;
                auto isect = intersector[i]->getFirstIntersection();
                hitPoint = isect.getWorldIntersectPoint();
            }

        }
    }
    else
    {
        if (active)
        {
            disactivate();
            active = 0;
        }
    }

    newActiveFlag = haveIsect;
    if (newActiveFlag != active)
    {
        if (buttonSensitive && (cover->getPointerButton()->getState() & ButtonMask))
        {
            return; // do not change state, if any button is currently pressed
        }
        active = newActiveFlag;
        if (active)
            activate();
        else
            disactivate();
    }
}

void coSensor::update()
{
    //START("coSensor::update");
    int newActiveFlag;
    calcDistance();
    newActiveFlag = sqrDistance <= threshold;
    if (newActiveFlag != active)
    {
        if (buttonSensitive && (cover->getPointerButton()->getState() & ButtonMask))
        {
            return; // do not change state, if any button is currently pressed
        }
        active = newActiveFlag;
        if (active)
            activate();
        else
            disactivate();
    }
}

void coSensor::addToPath(osg::Node *n)
{
    START("coSensor::addToPath");
    if ((n->asTransform()) && (n->asTransform()->asMatrixTransform()))
    {
        if (pathLength % 10 == 0)
        {
            osg::MatrixTransform **newPath;
            newPath = new osg::MatrixTransform *[pathLength + 10];
            for (int i = 0; i < pathLength; i++)
            {
                newPath[i] = path[i];
            }
            delete[] path;
            path = newPath;
        }
        path[pathLength] = n->asTransform()->asMatrixTransform();
        pathLength++;
    }
}

void coSensor::activate()
{
    START("coSensor::activate");
    active = 1;
}

void coSensor::disactivate()
{
    START("coSensor::disactivate")
    active = 0;
}

void coSensor::enable()
{
    START("coSensor::enable");
    enabled = 1;
}

void coSensor::disable()
{
    START("coSensor::disable");
    enabled = 0;
}

coSensor::~coSensor()
{
    START("coSensor::~coSensor");
    if (active)
        disactivate();
    delete[] path;
}

coSensorList::coSensorList()
{
    noDelete = 1;
}

void coSensorList::update()
{
    //START("");
    reset();
    while (current())
    {
        if (current()->getType() == coSensor::NONE)
        {
            coSensor *s = current();
            delete s;
            reset();
            while ((current()) && (current()->getType() == coSensor::NONE))
            {
                coSensor *s = current();
                delete s;
                reset();
            }
        }
        if (current())
        {
            current()->update();
            // hack for VRML sensors or others who want to be removed:
            // if they want to be removed, then they set their type to NONE
            // (you can't remove yourself)
            if (current()->getType() == coSensor::NONE)
            {
                coSensor *s = current();
                delete s;
                reset();
                while ((current()) && (current()->getType() == coSensor::NONE))
                {
                    coSensor *s = current();
                    delete s;
                    reset();
                }
            }
            else
                next();
        }
    }
}
