/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Atom.h"
#include "AtomBallInteractor.h"
#include "AtomStickInteractor.h"
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <osg/Geometry>
#include <osg/LineWidth>
Atom::Atom(string symbol, const char *interactorName, osg::Matrix m, float size, std::vector<osg::Vec3> connections, osg::Vec4 color)
{
    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "Atom::Atom %d connections\n", (int)connections.size());

    // connection line endpoints
    connectionLineStartPoint_ = m.getTrans();
    connectionLineEndPoint_ = m.getTrans();
    size_ = size;
    symbol_ = symbol;
    color_ = color;
    initialMat_ = m;
    // ball interactor
    osg::Vec3 normal(0, -1, 1);
    normal.normalize();
    atomBall_ = new AtomBallInteractor(symbol, interactorName, m.getTrans(), normal, size, color);
    float w = opencover::coVRConfig::instance()->screens[0].hsize;
    float h = opencover::coVRConfig::instance()->screens[0].vsize;
    osg::BoundingBox box;
    box.xMin() = -0.28 * w;
    box.xMax() = 0.28 * w;
    box.yMin() = -0.3 * h * sin(0.25 * M_PI);
    box.yMax() = 0.3 * h * sin(0.25 * M_PI);
    box.zMin() = -0.3 * h * sin(0.25 * M_PI);
    box.zMax() = 0.3 * h * sin(0.25 * M_PI);
    atomBall_->setBoundingBox(box);
    atomBall_->enableIntersection();

    // stick interactors
    for (size_t i = 0; i < connections.size(); i++)
    {
        atomSticks_.push_back(new AtomStickInteractor(symbol, interactorName, this, m, m.getTrans(), connections[i], size, color));
        atomSticks_[atomSticks_.size() - 1]->enableIntersection();
    }

    // each atoms has a line, which is inside when not in connection position
    // and comes out of the connection when running and in connection position to another atom
    lineCoord_ = new osg::Vec3Array(2);
    (*lineCoord_)[0].set(connectionLineStartPoint_[0], connectionLineStartPoint_[1], connectionLineStartPoint_[2]);
    (*lineCoord_)[1].set(connectionLineEndPoint_[0], connectionLineEndPoint_[1], connectionLineEndPoint_[2]);
    lineColor_ = new osg::Vec4Array(1);
    (*lineColor_)[0].set(1, 0, 0, 1);
    osg::Geometry *geometry = new osg::Geometry();
    geometry->setVertexArray(lineCoord_.get());
    geometry->setColorArray(lineColor_.get());
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 2));
    geometry->setUseDisplayList(false);
    lineGeode_ = new osg::Geode();
    lineGeode_->ref();
    osg::StateSet *state = opencover::VRSceneGraph::instance()->loadUnlightedGeostate();
    osg::LineWidth *lw = new osg::LineWidth(4.0f);
    state->setAttributeAndModes(lw, osg::StateAttribute::ON);
    lineGeode_->setStateSet(state);
    lineGeode_->addDrawable(geometry);
    lineGeode_->setNodeMask(lineGeode_->getNodeMask() & (~opencover::Isect::Intersection) & (~opencover::Isect::Pick));
    lineGeode_->setCullingActive(false);
    opencover::cover->getObjectsScale()->addChild(lineGeode_);

    mySnapAtomStick_ = NULL;
    otherSnapAtomStick_ = NULL;

    lastTime_ = opencover::cover->frameTime();
    lastPos_ = atomBall_->getPosition();
}

Atom::~Atom()
{
    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "Atom::~Atom\n");
}

void Atom::preFrame()
{
    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "Atom::preFrame----\n");

    // update ball
    atomBall_->preFrame();

    currentTime_ = opencover::cover->frameTime();
    currentPos_ = atomBall_->getPosition();
    lastPos_ = atomBall_->getLastPosition();
    // if ball is moving move also the sticks
    if (!atomBall_->isIdle())
    {
        for (size_t i = 0; i < atomSticks_.size(); i++)
        {
            atomSticks_[i]->updateTransform(atomSticks_[i]->getMatrix(), atomBall_->getPosition());
        }

        // abbreissen?
        double speed = (currentPos_ - lastPos_).length() / (currentTime_ - lastTime_);
        //fprintf(stderr,"speed=%f\n", speed);
        if (speed > 2500)
        {
            //fprintf(stderr,"Abreissen von atom %s\n", atomBall_->getInteractorName());

            for (size_t i = 0; i < atomSticks_.size(); i++)
            {
                if (atomSticks_[i]->getConnectedStick())
                {
                    fprintf(stderr, "setze connected stick %s auf NULL\n", atomSticks_[i]->getConnectedStick()->getAtom()->atomBall_->getInteractorName());
                    atomSticks_[i]->getConnectedStick()->setConnectedStick(NULL);
                    atomSticks_[i]->setConnectedStick(NULL);
                }
            }
            mySnapAtomStick_ = NULL;
            otherSnapAtomStick_ = NULL;
        }
        else
            updateConnectedAtoms(atomSticks_[0]->getDiffMat());
    }

    // update sticks
    for (size_t i = 0; i < atomSticks_.size(); i++)
    {
        atomSticks_[i]->preFrame();

        // if stick is rotating, rotate also the ball and the other sticks and connected atoms
        if (!atomSticks_[i]->isIdle())
        {
            atomBall_->updateTransform(atomBall_->getMatrix() * atomSticks_[i]->getFrameDiffMatrix());
            for (size_t j = 0; j < atomSticks_.size(); j++)
            {
                if (atomSticks_[i] != atomSticks_[j])
                    atomSticks_[j]->updateTransform(atomSticks_[j]->getMatrix() * atomSticks_[i]->getFrameDiffMatrix(), atomSticks_[j]->getPosition());
            }
            updateConnectedAtoms(atomSticks_[0]->getDiffMat());
        }
    }

    // in case that checkNear decides not near these are the inside coordinates
    // connection line endpoints
    connectionLineStartPoint_ = atomBall_->getMatrix().getTrans();
    connectionLineEndPoint_ = atomBall_->getMatrix().getTrans();
    (*lineCoord_)[0].set(connectionLineStartPoint_[0], connectionLineStartPoint_[1], connectionLineStartPoint_[2]);
    (*lineCoord_)[1].set(connectionLineEndPoint_[0], connectionLineEndPoint_[1], connectionLineEndPoint_[2]);

    lastTime_ = currentTime_;
    //lastPos_ = currentPos_;
}
void
Atom::updateTransform(osg::Matrix m)
{
    atomBall_->updateTransform(m);
    for (size_t i = 0; i < atomSticks_.size(); i++)
    {
        atomSticks_[i]->updateTransform(m, m.getTrans());
    }
}

bool
Atom::checkNear(Atom *a)
{
    bool r = false;

    if (a == this)
        return false;

    if (isIdle())
        return false;

    for (size_t i = 0; i < atomSticks_.size(); i++)
    {
        // nur unverbundene Verbindungen pruefen
        if (!atomSticks_[i]->getConnectedStick())
        {
            for (size_t j = 0; j < a->atomSticks_.size(); j++)
            {
                if (!a->atomSticks_[j]->getConnectedStick())
                {
                    // transform connection endpoint
                    osg::Vec3 cep1 = atomSticks_[i]->getDir() * size_ * atomSticks_[i]->getMatrix();

                    // transform connection vector
                    osg::Vec3 c1 = osg::Matrix::transform3x3(atomSticks_[i]->getDir(), atomSticks_[i]->getMatrix());
                    c1.normalize();

                    osg::Vec3 cep2 = a->atomSticks_[j]->getDir() * size_ * a->atomSticks_[j]->getMatrix();
                    osg::Vec3 c2 = osg::Matrix::transform3x3(a->atomSticks_[j]->getDir(), a->atomSticks_[j]->getMatrix());
                    c2.normalize();

                    float dce = (cep1 - cep2).length();
                    float dot = c1 * c2;
                    angle_ = acos(dot); // *180/M_PI;
                    //if ( (angle_*180/M_PI > 160) && (angle_*180/M_PI < 180) )
                    //   fprintf(stderr,"dot = %f angle=%f\n", dot, angle_*180/M_PI);
                    if ((dot < -0.93) && (dce < 0.9 * size_)) // 0.98 - 170 Grad 0.94 160 Grad
                    {
                        connectionLineStartPoint_ = cep1;
                        connectionLineEndPoint_ = cep2;
                        //fprintf(stderr,"AtomInteractor::isNear connections cep1=[%f %f %f] cep2=[%f %f %f] distance=%f size=%f\n", cep1_[0], cep1_[1], cep1_[2], cep2_[0], cep2_[1], cep2_[2], dce, size_);
                        //fprintf(stderr," d=%f dot=%f angle=%f\n", d, dot, angle);
                        otherSnapAtomStick_ = a->atomSticks_[j];
                        mySnapAtomStick_ = atomSticks_[i];
                        r = true;
                        break;
                    }
                }
            }
        }
    }

    // update line coordinates
    (*lineCoord_)[0].set(connectionLineStartPoint_[0], connectionLineStartPoint_[1], connectionLineStartPoint_[2]);
    (*lineCoord_)[1].set(connectionLineEndPoint_[0], connectionLineEndPoint_[1], connectionLineEndPoint_[2]);

    return (r);
}

void
Atom::snap(AtomStickInteractor *mySnapAtomStick, Atom *otherAtom, AtomStickInteractor *otherSnapAtomStick)
{
    //fprintf(stderr,"Atom(%s)::snap\n", atomBall_->getInteractorName());
    osg::Matrix m;
    osg::Vec3 pos;

    //rotate me (running atom) to -other connection
    osg::Vec3 otherDir = osg::Matrix::transform3x3(otherSnapAtomStick->getDir(), otherSnapAtomStick->getMatrix());
    m.makeRotate(mySnapAtomStick->getDir(), -otherDir);
    pos = otherAtom->atomBall_->getMatrix().getTrans() + otherDir * 2 * size_;
    m.setTrans(pos);
    updateTransform(m);

    // update all atoms which are already connected to me
    updateConnectedAtoms(mySnapAtomStick->getDiffMat());

    // set the connections
    mySnapAtomStick->setConnectedStick(otherSnapAtomStick);
    otherSnapAtomStick->setConnectedStick(mySnapAtomStick);
}
void
Atom::moveToPlane(opencover::coPlane *plane)
{
    osg::Vec3 pos = atomBall_->getMatrix().getTrans();
    osg::Vec3 ppos = plane->getProjectedPoint(pos);
    osg::Matrix m = atomSticks_[0]->getMatrix();
    m.setTrans(ppos);
    updateTransform(m);
    updateConnectedAtoms(atomSticks_[0]->getDiffMat());
}

void
Atom::reset()
{

    mySnapAtomStick_ = NULL;
    otherSnapAtomStick_ = NULL;
    for (size_t i = 0; i < atomSticks_.size(); i++)
    {
        atomSticks_[i]->setConnectedStick(NULL);
    }
    updateTransform(initialMat_);
}

bool
Atom::isIdle()
{

    if (!atomBall_->isIdle())
        return false;
    for (size_t i = 0; i < atomSticks_.size(); i++)
    {
        if (!atomSticks_[i]->isIdle())
            return false;
    }
    return true;
}
bool
Atom::wasStopped()
{

    if (atomBall_->wasStopped())
        return true;
    for (size_t i = 0; i < atomSticks_.size(); i++)
    {
        if (atomSticks_[i]->wasStopped())
            return true;
    }
    return false;
}

void
Atom::updateConnectedAtoms(osg::Matrix diffMat)
{
    //fprintf(stderr,"AtomInteractor(%s)::updateConnectedAtoms\n", atomBall_->getInteractorName());
    for (size_t i = 0; i < atomSticks_.size(); i++)
    {

        if (atomSticks_[i]->getConnectedStick()) // connection ist connected
        {
            //fprintf(stderr,"also move connected connection %d atom %s diff=[%f %f %f]\n", i, atomSticks_[i]->getConnectedStick()->getAtom()->atomBall_->getInteractorName(), diffMat(3,0), diffMat(3,1), diffMat(3,2));
            osg::Matrix m;
            m = atomSticks_[i]->getConnectedStick()->getMatrix() * diffMat;
            atomSticks_[i]->getConnectedStick()->getAtom()->updateTransform(m);
            atomSticks_[i]->getConnectedStick()->getAtom()->updateConnectedAtoms(atomSticks_[i], diffMat);
        }
    }
}
void
Atom::updateConnectedAtoms(AtomStickInteractor *alreadyUpdated, osg::Matrix diffMat)
{
    //fprintf(stderr,"AtomInteractor(%s)::updateConnectedAtoms2\n", atomBall_->getInteractorName());
    for (size_t i = 0; i < atomSticks_.size(); i++)
    {

        if (atomSticks_[i]->getConnectedStick() && (atomSticks_[i]->getConnectedStick() != alreadyUpdated)) // connection ist connected
        {
            //fprintf(stderr,"also move connected connection %d atom %s\n", i, atomConnections_[i]->getConnectedConnection()->getAtom()->getInteractorName());
            osg::Matrix m;

            m = atomSticks_[i]->getConnectedStick()->getMatrix() * diffMat;
            atomSticks_[i]->getConnectedStick()->getAtom()->updateTransform(m);
            atomSticks_[i]->getConnectedStick()->getAtom()->updateConnectedAtoms(atomSticks_[i], diffMat);
        }
    }
}
bool Atom::allConnectionsConnected(AtomStickInteractor *ommit)
{
    //if (!ommit)
    //fprintf(stderr,"Carbon(%s)::allConnectionsConnected ommit=NULL\n", atomBall_->getInteractorName());
    //else
    //   fprintf(stderr,"Carbon(%s)::allConnectionsConnected ommit=%s\n", atomBall_->getInteractorName(), ommit->getInteractorName());
    bool allConnected = true;
    for (size_t i = 0; i < atomSticks_.size(); i++)
    {
        AtomStickInteractor *myStick, *otherStick;
        myStick = atomSticks_[i];
        otherStick = atomSticks_[i]->getConnectedStick();
        if (myStick != ommit)
        {
            if (!otherStick)
            {
                //fprintf(stderr,"stick %d not connected\n", i);
                allConnected = false;
                break;
            }
            else
            {
                //fprintf(stderr,"stick %d is connected\n", i);
                if (otherStick->getAtom()->getSymbol() == "C")
                {
                    //fprintf(stderr,"stick %d is connected to a C\n", i);
                    Atom *c = otherStick->getAtom();
                    if (!c->allConnectionsConnected(otherStick))
                    {
                        allConnected = false;
                        break;
                    }
                }
            }
        }
        //else
        // fprintf(stderr,"omit stick %d\n", i);
    }

    if (allConnected)
    {
        //fprintf(stderr,"Carbon(%s)::allConnectionsConnected=true\n", atomBall_->getInteractorName());
        return (true);
    }
    else
    {
        //fprintf(stderr,"Carbon(%s)::allConnectionsConnected=false\n", atomBall_->getInteractorName());
        return (false);
    }
}

bool Atom::isUnconnected()
{

    bool unconnected = true;
    for (size_t i = 0; i < atomSticks_.size(); i++)
    {
        AtomStickInteractor *otherStick;
        otherStick = atomSticks_[i]->getConnectedStick();
        if (otherStick)
        {
            //fprintf(stderr,"stick %d not connected\n", i);
            unconnected = false;
            break;
        }
    }

    if (unconnected)
    {
        //fprintf(stderr,"Carbon(%s)::unconnected=true\n", atomBall_->getInteractorName());
        return (true);
    }
    else
    {
        //fprintf(stderr,"Carbon(%s)::unconnected=false\n", atomBall_->getInteractorName());
        return (false);
    }
}

void
Atom::enableIntersection(bool enable)
{
    if (enable)
    {
        atomBall_->enableIntersection();
        for (size_t i = 0; i < atomSticks_.size(); i++)
        {
            atomSticks_[i]->enableIntersection();
        }
    }
    else
    {
        atomBall_->disableIntersection();
        for (size_t i = 0; i < atomSticks_.size(); i++)
        {
            atomSticks_[i]->disableIntersection();
        }
    }
}

void
Atom::show(bool show)
{
    if (show)
    {
        atomBall_->show();
        for (size_t i = 0; i < atomSticks_.size(); i++)
        {
            atomSticks_[i]->show();
        }
    }
    else
    {
        atomBall_->hide();
        for (size_t i = 0; i < atomSticks_.size(); i++)
        {
            atomSticks_[i]->hide();
        }
    }
}
