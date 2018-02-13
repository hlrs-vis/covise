/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRIntersectionInteractor.h"
#include "coVRIntersectionInteractorManager.h"

#include "coVRLabel.h"
#include "VRSceneGraph.h"
#include "coVRPluginSupport.h"
#include "coIntersection.h"
#include <config/CoviseConfig.h>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Material>
#include <osg/PolygonMode>
#include <osg/ComputeBoundsVisitor>
#include <osg/ShapeDrawable>

#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/osg/OSGVruiHit.h>
#include <OpenVRUI/osg/OSGVruiNode.h>

#define max(a, b) (((a) > (b)) ? (a) : (b))

using namespace vrui;
using namespace opencover;

coVRIntersectionInteractor::coVRIntersectionInteractor(float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, enum coInteraction::InteractionPriority priority = Medium)
    : coCombinedButtonInteraction(type, iconName, priority)
{
    //fprintf(stderr,"coVRIntersectionInteractor::coVRIntersectionInteractor interactionName=%s InteractionType=%d\n", interactorName, type);

    osg::Matrix m;

    if (cover->debugLevel(2))
        fprintf(stderr, "new VRIntersectionInteractor(%s) size=%f\n", interactorName, s);

    _interactorName = new char[strlen(interactorName) + 1];
    strcpy(_interactorName, interactorName);

    labelStr_ = new char[strlen(interactorName) + 200];
    strcpy(labelStr_, interactorName);

    if (s < 0.f)
        s *= -1.f * cover->getSceneSize() / 70.f;
    _interSize = s;
    float interScale = _interSize / cover->getScale();

    // initialize flags
    _hit = false;
    _intersectionEnabled = false;
    _justHit = true;
    _standardHL = covise::coCoviseConfig::isOn("COVER.StandardInteractorHighlight", true);

    moveTransform = new osg::MatrixTransform();
    char nodeName[256];
    sprintf(nodeName, "coVRIntersectionInteractor-moveTransform-%s)", interactorName);
    moveTransform->setName(nodeName);

    scaleTransform = new osg::MatrixTransform();
    sprintf(nodeName, "coVRIntersectionInteractor-scaleTransform-%s)", interactorName);
    m.makeScale(interScale, interScale, interScale);
    _scale = interScale;
    scaleTransform->setMatrix(m);

    parent = cover->getObjectsScale();
    //fprintf(stderr,"...parent=%s\n", parent->getName().c_str());
    parent->addChild(moveTransform.get());
    moveTransform->addChild(scaleTransform.get());

    vNode = new OSGVruiNode(moveTransform.get());

    osg::PolygonMode *polymode = new osg::PolygonMode();
    polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL);
    _selectedHl = new osg::StateSet();
    _intersectedHl = new osg::StateSet();

    if (_standardHL)
    {
        // set default materials
        osg::Material *selMaterial = new osg::Material();
        selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.0, 0.6, 0.0, 1.0f));
        selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.0, 0.6, 0.0, 1.0f));
        selMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
        selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
        selMaterial->setColorMode(osg::Material::OFF);
        osg::Material *isectMaterial = new osg::Material();
        isectMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.6, 0.6, 0.0, 1.0f));
        isectMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.6, 0.6, 0.0, 1.0f));
        isectMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
        isectMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
        isectMaterial->setColorMode(osg::Material::OFF);
        _selectedHl->setAttribute(selMaterial, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
        _intersectedHl->setAttribute(isectMaterial, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
    }

    _selectedHl->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);
    _selectedHl->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);

    _oldHl = NULL;

    label_ = NULL;
    if (cover->debugLevel(5))
    {

        // create label
        osg::Vec4 fgColor(0.8, 0.5, 0.0, 1.0);
        osg::Vec4 bgColor(0.2, 0.2, 0.2, 0.0);
        label_ = new coVRLabel(_interactorName, _interSize, 2 * _interSize, fgColor, bgColor);
        label_->hide();
    }

    // constantInteractorSize_= On: Uwe Mode: scale icon to keep size indepened of interactor position and world scale
    // constantInteractorSize_= Off: Daniela Mode: scale icon to keep _interactorSize independed of world scale
    constantInteractorSize_ = covise::coCoviseConfig::isOn("COVER.ConstantInteractorSize", true);

    iconSize_ = 1;
    firstTime = true;

    _wasHit = false;
}

coVRIntersectionInteractor::~coVRIntersectionInteractor()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "delete ~coVRIntersectionInteractor(%s)\n", _interactorName);

    disableIntersection();

    if (moveTransform->getNumParents())
    {
        osg::Group *p = moveTransform->getParent(0);
        p->removeChild(moveTransform.get());
    }
    delete[] _interactorName;

    if (cover->debugLevel(5))
        fprintf(stderr, "delete ~coVRIntersectionInteractor done\n");
}

void coVRIntersectionInteractor::show()
{

    //fprintf(stderr,"coVRIntersectionInteractor::show(%s)\n", _interactorName);

    if (moveTransform->getNumParents() == 0)
    {
        parent->addChild(moveTransform.get());
        //fprintf(stderr,"parent name=%s\n", parent->getName().c_str());
    }
    //pfPrint(cover->getObjectsRoot(),  PFTRAV_SELF|PFTRAV_DESCEND, PFPRINT_VB_NOTICE, NULL);
    //if (label_)
    //   label_->show();
}

void coVRIntersectionInteractor::hide()
{
    //fprintf(stderr,"coVRIntersectionInteractor::hide(%s)\n", _interactorName);

    if (moveTransform->getNumParents())
    {
        osg::Group *p = moveTransform->getParent(0);
        p->removeChild(moveTransform.get());
    }

    if (label_)
        label_->hide();
}

void coVRIntersectionInteractor::enableIntersection()
{
    if (cover->debugLevel(4))
        fprintf(stderr, "coVRIntersectionInteractor(%s)::enableIntersection\n", _interactorName);

    if (!_intersectionEnabled)
    {
        _intersectionEnabled = true;
        vruiIntersection::getIntersectorForAction("coAction")->add(vNode, this);

        coVRIntersectionInteractorManager::the()->add(this);

        if (cover->debugLevel(2))
            fprintf(stderr, "adding to intersectorlist\n");
    }
}

void coVRIntersectionInteractor::disableIntersection()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "coVRIntersectionInteractor(%s)::disableIntersection\n", _interactorName);

    // interactor is normally unregistered in miss
    // if we disable intersection miss is not called anymore
    // therefore we make sure that the interactor is unregistered
    if (registered)
    {
        coInteractionManager::the()->unregisterInteraction(this);
    }
    if (_intersectionEnabled)
    {
        if (cover->debugLevel(2))
            fprintf(stderr, "removing from intersectorlist\n");
        _intersectionEnabled = false;
        vruiIntersection::getIntersectorForAction("coAction")->remove(vNode);
        coVRIntersectionInteractorManager::the()->remove(this);
    }

    resetState();
}

const osg::Matrix &coVRIntersectionInteractor::getPointerMat() const
{
    if (isMouse())
        return cover->getMouseMat();
    else
        return cover->getPointerMat();
}

int coVRIntersectionInteractor::hit(vruiHit *hit)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "coVRIntersectionInteractor(%s)::hit\n", _interactorName);

    _hit = true;
    _wasHit = false;
    if (_justHit)
    {
        if (cover->debugLevel(4))
            fprintf(stderr, "VRIntersectionInteractor(%s)::hit justHit\n", _interactorName);

        _wasHit = true;

        if (label_ && cover->debugLevel(4))
        {
            //fprintf(stderr,"show label\n");
            label_->show();
        }
    }

    if (!registered)
    {
        coInteractionManager::the()->registerInteraction(this);
        if (hit)
            setHitByMouse(hit->isMouseHit());
    }
    // store the current hit position because we don't get it in
    // startInteraction
    if (hit)
    {
        coVector v = hit->getWorldIntersectionPoint();
        osg::Vec3 wp(v[0], v[1], v[2]);
        _hitPos = wp * cover->getInvBaseMat();
        auto osgvruinode = dynamic_cast<OSGVruiNode *>(hit->getNode());
        if (osgvruinode)
        {
            _hitNode = osgvruinode->getNodePtr();
        }
        else
        {
            _hitNode = nullptr;
        }
    }
    else
    {
        _hitPos = getMatrix().getTrans();
        _hitNode = nullptr;
    }

    _justHit = false;

    return ACTION_CALL_ON_MISS;
}

void coVRIntersectionInteractor::miss()
{
    _hit = false;
    _justHit = true;
    _wasHit = false;
    if (cover->debugLevel(4))
        fprintf(stderr, "coVRIntersectionInteractor(%s)::miss\n", _interactorName);

    if (registered)
    {
        coInteractionManager::the()->unregisterInteraction(this);

        if (label_ && cover->debugLevel(5))
            label_->hide();
    }
}

osg::Geode *coVRIntersectionInteractor::findGeode(osg::Node *n)
{
    osg::Geode *geode;
    osg::Group *group;
    geode = dynamic_cast<osg::Geode *>(n);
    if (geode != NULL)
        return geode;
    group = dynamic_cast<osg::Group *>(n);
    for (int i = group->getNumChildren() - 1; i >= 0; i--)
    {
        geode = findGeode(group->getChild(i));
        if (geode != NULL)
            return geode;
    }
    return NULL;
}

void coVRIntersectionInteractor::addIcon()
{
    // add icon to zeigestrahl and hilight the interactor

    //fprintf(stderr,"coVRIntersectionInteractor(%s)::addIcon and show intersected hl\n", _interactorName);
    coInteraction::addIcon();

    if (state == coInteraction::Idle)
    {
        _oldHl = moveTransform->getStateSet();

        if (!_standardHL)
        {
            osg::Material *selMaterial = new osg::Material();
            osg::Vec4 colDiff = osg::Vec4(0.5, 0.5, 0.5, 1.0);
            osg::Vec4 colAmb = osg::Vec4(0.5, 0.5, 0.5, 1.0);
            osg::ref_ptr<osg::Drawable> drawable;
            osg::Geode *geode = findGeode(geometryNode.get());

            for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
            {
                drawable = geode->getDrawable(i);
                drawable->ref();
                bool mtlOn = false;
                if (drawable->getStateSet())
                    mtlOn = (drawable->getStateSet()->getMode(osg::StateAttribute::MATERIAL) == osg::StateAttribute::ON) || (drawable->getStateSet()->getMode(osg::StateAttribute::MATERIAL) == osg::StateAttribute::INHERIT);
                if (mtlOn)
                {
                    osg::Material *mtl = (osg::Material *)drawable->getOrCreateStateSet()->getAttribute(osg::StateAttribute::MATERIAL);
                    colDiff = mtl->getDiffuse(osg::Material::FRONT_AND_BACK);
                    colAmb = mtl->getAmbient(osg::Material::FRONT_AND_BACK);
                }
                else
                {
                    osg::ShapeDrawable *shapeDraw = dynamic_cast<osg::ShapeDrawable *>(drawable.get());
                    if (shapeDraw)
                    {
                        colDiff = shapeDraw->getColor();
                        colAmb = colDiff;
                    }
                }
                drawable->unref();
            }

            selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(colDiff.r() + 0.2, colDiff.g() + 0.2, colDiff.b() + 0.2, colDiff.a()));
            selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(colAmb.r() + 0.2, colAmb.g() + 0.2, colAmb.b() + 0.2, colAmb.a()));
            selMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
            selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
            selMaterial->setColorMode(osg::Material::OFF);
            _intersectedHl->setAttribute(selMaterial, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
        }

        moveTransform->setStateSet(_intersectedHl.get());
    }
    // else it is still active and we want to keep the selected hlight
}

void coVRIntersectionInteractor::removeIcon()
{
    //fprintf(stderr,"coVRIntersectionInteractor(%s)::removeIcon and set hl off\n", _interactorName);
    coInteraction::removeIcon();
}

void coVRIntersectionInteractor::resetState()
{
    _oldHl = NULL;
    moveTransform->setStateSet(NULL);
}

void coVRIntersectionInteractor::startInteraction()
{

    if (cover->debugLevel(4))
        fprintf(stderr, "\ncoVRIntersectionInteractor::startInteraction and set selected hl\n");
    _oldHl = moveTransform->getStateSet();

    if (!_standardHL)
    {
        osg::Material *selMaterial = new osg::Material();
        osg::Vec4 colDiff = osg::Vec4(0.5, 0.5, 0.5, 1.0);
        osg::Vec4 colAmb = osg::Vec4(0.5, 0.5, 0.5, 1.0);
        osg::ref_ptr<osg::Drawable> drawable;
        osg::Geode *geode = findGeode(geometryNode.get());

        for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
        {
            drawable = geode->getDrawable(i);
            drawable->ref();
            bool mtlOn = false;
            if (drawable->getStateSet())
                mtlOn = (drawable->getStateSet()->getMode(osg::StateAttribute::MATERIAL) == osg::StateAttribute::ON) || (drawable->getStateSet()->getMode(osg::StateAttribute::MATERIAL) == osg::StateAttribute::INHERIT);
            if (mtlOn)
            {
                osg::Material *mtl = (osg::Material *)drawable->getOrCreateStateSet()->getAttribute(osg::StateAttribute::MATERIAL);
                colDiff = mtl->getDiffuse(osg::Material::FRONT_AND_BACK);
                colAmb = mtl->getAmbient(osg::Material::FRONT_AND_BACK);
            }
            else
            {
                osg::ShapeDrawable *shapeDraw = dynamic_cast<osg::ShapeDrawable *>(drawable.get());
                if (shapeDraw)
                {
                    colDiff = shapeDraw->getColor();
                    colAmb = colDiff;
                }
            }
            drawable->unref();
        }

        selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(colDiff.r() - 0.2, colDiff.g() - 0.2, colDiff.b() - 0.2, colDiff.a()));
        selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(colAmb.r() - 0.2, colAmb.g() - 0.2, colAmb.b() - 0.2, colAmb.a()));
        selMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
        selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
        selMaterial->setColorMode(osg::Material::OFF);
        _selectedHl->setAttribute(selMaterial, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
    }
    moveTransform->setStateSet(_selectedHl.get());

    //Interactor 0-5 are of clipplane
}

void coVRIntersectionInteractor::stopInteraction()
{
    if (cover->debugLevel(4))
        fprintf(stderr, "\ncoVRIntersectionInteractor::stopInteraction\n");

    // in case that miss was called before but interaction was ongoing
    // we have to unregister

    moveTransform->setStateSet(_oldHl.get());

    resetState();
}

void coVRIntersectionInteractor::doInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVRIntersectionInteractor::doInteraction\n");
}

void coVRIntersectionInteractor::keepSize()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVRIntersectionInteractor::keepSize scale=%f\n", cover->getScale());

    float interScale;
    if (constantInteractorSize_) // Uwe Mode: scale icon to keep size indepened of interactor position and world scale
    {
        osg::Vec3 WorldPos = moveTransform->getMatrix().getTrans() * cover->getBaseMat();
        interScale = cover->getInteractorScale(WorldPos) * _interSize;
    }
    else // Daniela Mode: scale icon to keep COVER.IconSize independed of world scale
    {
        if (firstTime)
        {
            geometryNode->dirtyBound();
            osg::ComputeBoundsVisitor cbv;
            osg::BoundingBox &bb(cbv.getBoundingBox());
            geometryNode->accept(cbv);
            if (bb.valid())
            {
                float sx, sy, sz;
                sx = bb._max.x() - bb._min.x();
                sy = bb._max.y() - bb._min.y();
                sz = bb._max.z() - bb._min.z();
                iconSize_ = max(sx, sy);
                iconSize_ = max(iconSize_, sz);
                firstTime = false;
            }
        }
        interScale = _interSize / (cover->getScale() * iconSize_);
    }
    _scale = interScale;
    scaleTransform->setMatrix(osg::Matrix::scale(interScale, interScale, interScale));
}

float coVRIntersectionInteractor::getScale() const
{
    return _scale;
}

void coVRIntersectionInteractor::preFrame()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVRIntersectionInteractor::preFrame\n");
    keepSize();

    // update label pos
    if (label_)
    {
        osg::Matrix m;
        m = moveTransform->getMatrix();
        // update label position
        osg::Matrix o_to_w = cover->getBaseMat();
        osg::Vec3 pos_w, pos_o;
        pos_o = m.getTrans();
        pos_w = pos_o * o_to_w;
        label_->setPosition(pos_w);

        char regStr[10];
        char stateStr[20];
        char runStateStr[20];

        if (registered)
            strcpy(regStr, "reg");
        else
            strcpy(regStr, "unreg");

        if (getState() == 0)
            strcpy(stateStr, "idle");
        else if (getState() == 1)
            strcpy(stateStr, "pendingActive");
        else if (getState() == 2)
            strcpy(stateStr, "active");
        else if (getState() == 3)
            strcpy(stateStr, "remoteActive");
        else
            strcpy(stateStr, "unknown");

        if (runningState == 0)
            strcpy(runStateStr, "started");
        else if (runningState == 1)
            strcpy(runStateStr, "running");
        else if (runningState == 2)
            strcpy(runStateStr, "stopped");
        else if (runningState == 3)
            strcpy(runStateStr, "notRunning");
        else
            strcpy(runStateStr, "unknown");

        if (label_)
        {
            sprintf(labelStr_, "%s-%s-%s-%s", _interactorName, regStr, stateStr, runStateStr);
            label_->setString(labelStr_);
        }
    }
}

/*
void coVRIntersectionInteractor::startTraverseInteractors()
{
   if (cover->debugLevel(4))
      fprintf(stderr,"\ncoVRIntersectionInteractor::startTraverseInteractors\n");

   traverseIndex = 0;

   //for(int i = 0; i<interactors->size(); i++)
   //{
   //   fprintf(stderr,"\n index %d name %s enabled %d\n",i, interactors->at(i)->getInteractorName(), interactors->at(i)->isEnabled());
   //}

   // set the currentInterPos to enable interactor
   while(traverseIndex<interactors->size())
   {
      if(!interactors->at(traverseIndex)->isEnabled())
      {
         //fprintf(stderr," index %d name %s\n",traverseIndex, interactors->at(traverseIndex)->getInteractorName());
         traverseIndex++;
      }
      else
      {
         //fprintf(stderr," enabled: index %d name %s\n",traverseIndex, interactors->at(traverseIndex)->getInteractorName());
         break;
      }
   }

   if(traverseIndex < interactors->size())
   {
      //fprintf(stderr,"++++ %d pos %f %f %f\n", traverseIndex, interactors->at(traverseIndex)->_interPos.x(), interactors->at(traverseIndex)->_interPos.y(), interactors->at(traverseIndex)->_interPos.z());

      currentInterPos = interactors->at(traverseIndex)->_interPos;
      isTraverseInteractors = true;
      //VRSceneGraph::instance()->getHandTransform()->setMatrix(m);
   }
}
*/

/*
void coVRIntersectionInteractor::traverseInteractors()
{

   if (cover->debugLevel(4))
      fprintf(stderr,"\ncoVRIntersectionInteractor::traverseInteractors size %d index %d\n", (int)interactors->size(), traverseIndex);

//    while(traverseIndex<interactors->size())
//    {
//       if(!interactors->at(traverseIndex)->isEnabled())
//       {
//          fprintf(stderr," index %d name %s\n",traverseIndex, interactors->at(traverseIndex)->getInteractorName());
//          traverseIndex++;
//       }
//       else
//       {
//          fprintf(stderr," enabled: index %d name %s\n",traverseIndex, interactors->at(traverseIndex)->getInteractorName());
//          break;
//       }
//    }

//    if(traverseIndex < interactors->size())
//    {
//       fprintf(stderr,"blubb %d pos %f %f %f\n", traverseIndex, interactors->at(traverseIndex)->_interPos.x(), interactors->at(traverseIndex)->_interPos.y(), interactors->at(traverseIndex)->_interPos.z());
// 
//       currentInterPos = interactors->at(traverseIndex)->_interPos;
//       isTraverseInteractors = true;
//       //VRSceneGraph::instance()->getHandTransform()->setMatrix(m);
//    }
//    else
//    {
//       traverseIndex=0;
//    }
// 
//    if ( !(interactors->at(traverseIndex)->isRegistered()) )
//    {
//       coInteractionManager::the()->registerInteraction(interactors->at(traverseIndex)->isRegistered());
//    }
// 
//    if( traverseIndex < interactors->size() )
//    {
//       if( interactors->at(traverseIndex)->isEnabled() )
//       {
//          fprintf(stderr,"\ncoVRIntersectionInteractor::startInteraction\n");
//          interactors->at(traverseIndex)->startInteraction(true);
//       }
//    }

   //traverseIndex++;
}
*/

/*

void coVRIntersectionInteractor::stopTraverseInteractors()
{

   if (cover->debugLevel(4))
      fprintf(stderr,"\ncoVRIntersectionInteractor::stopTraverseInteractors\n");

   traverseIndex = 0;

   isTraverseInteractors = false;
}
*/
bool coVRIntersectionInteractor::isEnabled()
{
    return _intersectionEnabled;
}

osg::Vec3 coVRIntersectionInteractor::restrictToVisibleScene(osg::Vec3 pos)
{
    osg::Vec3 rpos(0.0, 0.0, 0.0); //restricted position
    rpos = pos;

    osg::BoundingBox box = cover->getBBox(cover->getObjectsRoot());
    if (box.valid())
    {
        if (pos[0] < box.xMin())
        {
            rpos[0] = box.xMin();
            //fprintf(stderr, "restricting posx=[%f] to box minx=[%f]\n", pos[0], box.min[0]);
        }
        if (pos[1] < box.yMin())
        {
            rpos[1] = box.yMin();
            //fprintf(stderr, "restricting posy=[%f] to box miny=[%f]\n", pos[1], box.min[1]);
        }
        if (pos[2] < box.zMin())
        {
            rpos[2] = box.zMin();
            //fprintf(stderr, "restricting posz=[%f] to box minz=[%f]\n", pos[2], box.min[2]);
        }
        if (pos[0] > box.xMax())
        {
            rpos[0] = box.xMax();
            //fprintf(stderr, "restricting posx=[%f] to box maxx=[%f]\n", pos[0], box.max[0]);
        }
        if (pos[1] > box.yMax())
        {
            rpos[1] = box.yMax();
            //fprintf(stderr, "restricting posy=[%f] to box maxy=[%f]\n", pos[1], box.max[1]);
        }
        if (pos[2] > box.zMax())
        {
            rpos[2] = box.zMax();
            //fprintf(stderr, "restricting posz=[%f] to box maxz=[%f]\n", pos[2], box.max[2]);
        }
    }
    return rpos;
}
void coVRIntersectionInteractor::setCaseTransform(osg::MatrixTransform *m)
{
    //fprintf(stderr,"coVRIntersectionInteractor(%s)::setCaseTransform\n", _interactorName);
    if (m)
    {
        interactorCaseTransform = m;
        parent = interactorCaseTransform;
        //fprintf(stderr,"...setting parent to interactorCaseTransform\n");
        // remove moveDCS from scaleDCS
        if (moveTransform->getNumParents() > 0)
        {
            if (moveTransform->getParent(0)->removeChild(moveTransform.get()))
            {
                // add moveDCS to case

                parent->addChild(moveTransform.get());
                //pfPrint(cover->getObjectsRoot(),  PFTRAV_SELF|PFTRAV_DESCEND, PFPRINT_VB_NOTICE, NULL);
            }
        }
        else if (cover->debugLevel(5))
            fprintf(stderr, "coVRIntersectionInteractor(%s)::setCaseTransform moveDCS is actually not under scaleDCS\n", _interactorName);
    }
    else
    {
        interactorCaseTransform = NULL;
        if (cover->debugLevel(5))
            fprintf(stderr, "---coVRIntersectionInteractor::setCaseDCS dcs=NULL\n");
    }
}
