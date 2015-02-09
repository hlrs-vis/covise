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
#include <cover/coCoverConfig.h>
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
#include <osg/PolygonOffset>
#include <osg/BoundingSphere>
#include <osg/Geode>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/TexGen>
#include <osgDB/ReadFile>
#include <osg/LineSegment>
#include <osgUtil/IntersectVisitor>
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
    float h = dist.length2() / Scale2;
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
    osg::Vec3 q0, q1, q2, q3, q4, q5;
    osg::Matrix handMat;
    handMat = cover->getPointerMat();
    q0.set(0.0f, -threshold, 0.0f);
    q1.set(0.0f, threshold, 0.0f);

    q2.set(-threshold, 0.0f, 0.0f);
    q3.set(threshold, 0.0f, 0.0f);

    q4.set(0.0f, 0.0f, -threshold);
    q5.set(0.0f, 0.0f, threshold);

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
    q0 = trinv.preMult(q0);
    q1 = trinv.preMult(q1);

    q2 = trinv.preMult(q2);
    q3 = trinv.preMult(q3);

    q4 = trinv.preMult(q4);
    q5 = trinv.preMult(q5);

    osg::ref_ptr<osg::LineSegment> ray = new osg::LineSegment();
    osg::ref_ptr<osg::LineSegment> ray2 = new osg::LineSegment();
    osg::ref_ptr<osg::LineSegment> ray3 = new osg::LineSegment();
    ray->set(q0, q1);
    ray2->set(q2, q3);
    ray3->set(q4, q5);

    osgUtil::IntersectVisitor visitor;
    visitor.addLineSegment(ray.get());
    visitor.addLineSegment(ray2.get());
    visitor.addLineSegment(ray3.get());

    node->accept(visitor);
    int num1 = visitor.getNumHits(ray.get());
    int num2 = visitor.getNumHits(ray2.get());
    int num3 = visitor.getNumHits(ray3.get());

    osgUtil::Hit hitInformation1;
    osgUtil::Hit hitInformation2;
    osgUtil::Hit hitInformation3;
    if (enabled)
    {
        if (num1)
        {
            hitInformation1 = visitor.getHitList(ray.get()).front();
            hitPoint = hitInformation1.getWorldIntersectPoint();
        }
        if (num2)
        {
            hitInformation2 = visitor.getHitList(ray2.get()).front();
            hitPoint = hitInformation2.getWorldIntersectPoint();
        }
        if (num3)
        {
            hitInformation3 = visitor.getHitList(ray3.get()).front();
            hitPoint = hitInformation3.getWorldIntersectPoint();
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

    newActiveFlag = num1 || num2 || num3;
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
