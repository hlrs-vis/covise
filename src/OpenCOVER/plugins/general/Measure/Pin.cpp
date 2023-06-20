#include "Pin.h"

#include <osg/Switch>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <cover/coVRCollaboration.h>
#include <OpenVRUI/osg/mathUtils.h>


using namespace vrui;
using namespace opencover;


Pin::Pin(double coneSize, int id, int dimensionID, ui::Group *parent)
: coneSize(coneSize)
, id(id)
, dimensionId(dimensionID)
, pos(new osg::MatrixTransform)
, vNode(std::make_unique<OSGVruiNode>(pos))
, interactionA(std::make_unique<coTrackerButtonInteraction>(coInteraction::ButtonA, "MarkPlacement", coInteraction::Medium))
, sc(new osg::MatrixTransform)
, icons(new osg::Switch())
, positionInput(new ui::VectorEditField(parent, "pin" + std::to_string(id)))
{
    pos->addChild(sc);
    sc->addChild(icons);
    geo = coVRFileManager::instance()->loadIcon("marker");
    icons->addChild(geo);
    geo = coVRFileManager::instance()->loadIcon("marker2");
    icons->addChild(geo);
    icons->setSingleChildOn(0);
    cover->getObjectsRoot()->addChild(pos);
    vruiIntersection::getIntersectorForAction("coAction")->add(vNode.get(), this);

    positionInput->setPriority(ui::Element::Priority::Low);
    positionInput->setCallback([this](const osg::Vec3 &vec){
        auto m = pos->getMatrix();
        m.setTrans(vec);
        pos->setMatrix(m);
            timeOfLastChange = std::chrono::system_clock::now(); 

    });
}

Pin::~Pin()
{
    if(vNode)
    {
        vruiIntersection::getIntersectorForAction("coAction")->remove(vNode.get());
        pos->getParent(0)->removeChild(pos);
    }
}

/**
@param hitPoint,hit  Performer intersection information
@return ACTION_CALL_ON_MISS if you want miss to be called,
otherwise ACTION_DONE is returned
*/
int Pin::hit(vruiHit *)
{
    if ((coVRCollaboration::instance()->getCouplingMode() == coVRCollaboration::MasterSlaveCoupling
         && !coVRCollaboration::instance()->isMaster())
        || placing)
        return ACTION_CALL_ON_MISS;

    moveMarker = true;
    setIcon(1);

    return ACTION_CALL_ON_MISS;
}

/// Miss is called once after a hit, if the button is not intersected anymore.
void Pin::miss()
{

    if (!interactionA->isRunning())
    {
        moveMarker = false;
        moveStarted = false;
        setIcon(0);
    }
}

void Pin::setIcon(int i)
{
    icons->setSingleChildOn(i);
}

void Pin::setConeSize(float size)
{
    coneSize = size;
    update();
}

int Pin::getDimensionID() const
{
    return dimensionId;
}

std::chrono::system_clock::time_point Pin::getTimeOfLastChange() const
{
    return timeOfLastChange;
}

void Pin::resize()
{
    osg::Vec3 wpoint1 = osg::Vec3(0, 0, 0);
    osg::Vec3 wpoint2 = osg::Vec3(0, 0, 300);
    osg::Vec3 opoint1 = wpoint1 * cover->getInvBaseMat();
    osg::Vec3 opoint2 = wpoint2 * cover->getInvBaseMat();

    //distance formula
    osg::Vec3 wDiff = wpoint2 - wpoint1;
    osg::Vec3 oDiff = opoint2 - opoint1;
    double distWld = wDiff.length();
    double distObj = oDiff.length();

    //controls the cone size
    double scaleFactor = (coneSize / 150) * distObj / distWld;

    //sc->setMatrix controls the size of the markers
    sc->setMatrix(osg::Matrix::scale(scaleFactor, scaleFactor, scaleFactor));
}

void Pin::setPos(osg::Matrix &mat)
{
    coCoord c;
    c = mat;
    c.makeMat(mat);
    pos->setMatrix(mat);
    //mat.print(1,1,"coorded mat:",stderr);
    positionInput->setValue(mat.getTrans());
    timeOfLastChange = std::chrono::system_clock::now(); 
    resize();
}

osg::Matrix Pin::getMat() const
{
   return pos->getMatrix();
}

void Pin::update()
{

    resize();
    if ((placing) || (moveMarker))
    {
        if (!interactionA->isRegistered())
        {
            coInteractionManager::the()->registerInteraction(interactionA.get());
        }
    }
    else
    {
        if (interactionA->isRegistered())
        {
            coInteractionManager::the()->unregisterInteraction(interactionA.get());
        }
    }
    if (placing)
    {
        if (interactionA->isRegistered())
        {

            osg::Matrix trans;
            osg::Matrix mat;
            trans.makeTranslate(0, 500, 0);
            mat = trans * cover->getPointerMat() * cover->getInvBaseMat();
            setPos(mat);
        }
        if (interactionA->wasStarted()) // button pressed
        {
            // checkboxTog = !checkboxTog;
            // if (checkboxTog)
            // {
            //     linearItem->setState(false);
            // }
            placing = false;
            if (interactionA->isRegistered())
            {
                coInteractionManager::the()->unregisterInteraction(interactionA.get());
            }
        }
    }
    else if (moveMarker)
    {
        if (interactionA->isRegistered())
        {
            if (interactionA->wasStarted()) // button pressed
            {
                invStartHand.invert(cover->getPointerMat());
                startPos = pos->getMatrix() * cover->getBaseMat();
                moveStarted = true;
            }
            if (interactionA->isRunning())
            {
                if (!moveStarted)
                {
                    invStartHand.invert(cover->getPointerMat());
                    startPos = pos->getMatrix() * cover->getBaseMat();
                }
                moveStarted = true;
                osg::Matrix dMat = invStartHand * cover->getPointerMat();
                osg::Matrix current;
                osg::Matrix tmp;
                tmp = startPos * dMat;
                current = tmp * cover->getInvBaseMat();
                setPos(current);
            }
            if (interactionA->wasStopped())
            {
                if (moveStarted)
                {
                    moveMarker = false;
                    moveStarted = false;
                    if (interactionA->isRegistered())
                    {
                        coInteractionManager::the()->unregisterInteraction(interactionA.get());
                    }

                    setIcon(0);
                }
            }
        }
    }
}