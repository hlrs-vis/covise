/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvIntersectionInteractor.h"
#include "vvIntersectionInteractorManager.h"

#include "vvLabel.h"
#include "vvSceneGraph.h"
#include "vvPluginSupport.h"
#include "vvIntersection.h"
#include <config/CoviseConfig.h>
#include <vsg/nodes/MatrixTransform.h>
#include <vrb/client/SharedState.h>
#include <OpenVRUI/sginterface/vruiHit.h>

#define max(a, b) (((a) > (b)) ? (a) : (b))

using namespace vrui;
using namespace vive;

vvIntersectionInteractor::vvIntersectionInteractor(float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, enum coInteraction::InteractionPriority priority = Medium, bool highliteHitNodeOnly)
    : coCombinedButtonInteraction(type, iconName, priority), _highliteHitNodeOnly(highliteHitNodeOnly)
{
    //fprintf(stderr,"vvIntersectionInteractor::vvIntersectionInteractor interactionName=%s InteractionType=%d\n", interactorName, type);

    vsg::dmat4 m;

    if (vv->debugLevel(2))
        fprintf(stderr, "new VRIntersectionInteractor(%s) size=%f\n", interactorName, s);

    _interactorName = new char[strlen(interactorName) + 1];
    strcpy(_interactorName, interactorName);

    labelStr_ = new char[strlen(interactorName) + 200];
    strcpy(labelStr_, interactorName);

    if (s < 0.f)
        s *= -1.f * vv->getSceneSize() / 70.f;
    _interSize = s;
    float interScale = _interSize / vv->getScale();

    // initialize flags
    _hit = false;
    _intersectionEnabled = false;
    _justHit = true;
    _standardHL = covise::coCoviseConfig::isOn("COVER.StandardInteractorHighlight", true);

    moveTransform = vsg::MatrixTransform::create();

    std::string nodeName = std::string("vvIntersectionInteractor-moveTransform-") + std::string(interactorName);
    
    moveTransform->setValue("name",nodeName);

    scaleTransform = vsg::MatrixTransform::create();
    nodeName = std::string("vvIntersectionInteractor-scaleTransform-") + std::string(interactorName);
    m= vsg::scale(interScale, interScale, interScale);
    _scale = interScale;
    scaleTransform->matrix = (m);

    parent = vv->getObjectsScale();
    //fprintf(stderr,"...parent=%s\n", parent->getName().c_str());
    parent->addChild(moveTransform);
    moveTransform->addChild(scaleTransform);

    vNode = new VSGVruiNode(moveTransform);

    /*
    _oldHl = NULL;

    label_ = NULL;
    if (vv->debugLevel(5))
    {

        // create label
        vsg::vec4 fgColor(0.8, 0.5, 0.0, 1.0);
        vsg::vec4 bgColor(0.2, 0.2, 0.2, 0.0);
        label_ = new vvLabel(_interactorName, _interSize, 2 * _interSize, fgColor, bgColor);
        label_->hide();
    }

    // constantInteractorSize_= On: Uwe Mode: scale icon to keep size indepened of interactor position and world scale
    // constantInteractorSize_= Off: Daniela Mode: scale icon to keep _interactorSize independed of world scale
    constantInteractorSize_ = covise::coCoviseConfig::isOn("COVER.ConstantInteractorSize", true);
    */
    iconSize_ = 1;
    firstTime = true;

    _wasHit = false;
}

vvIntersectionInteractor::~vvIntersectionInteractor()
{
    if (vv->debugLevel(5))
        fprintf(stderr, "delete ~vvIntersectionInteractor(%s)\n", _interactorName);

    disableIntersection();

    /*if (moveTransform->getNumParents())
    {
        vsg::Group *p = moveTransform->getParent(0);
        p->removeChild(moveTransform);
    }*/
    delete[] _interactorName;

    delete vNode;

    if (vv->debugLevel(5))
        fprintf(stderr, "delete ~vvIntersectionInteractor done\n");
}

void vvIntersectionInteractor::show()
{

    //fprintf(stderr,"vvIntersectionInteractor::show(%s)\n", _interactorName);

   /* if (moveTransform->getNumParents() == 0)
    {
        parent->addChild(moveTransform);
        //fprintf(stderr,"parent name=%s\n", parent->getName().c_str());
    }*/
    //pfPrint(vv->getObjectsRoot(),  PFTRAV_SELF|PFTRAV_DESCEND, PFPRINT_VB_NOTICE, NULL);
    //if (label_)
    //   label_->show();
}

void vvIntersectionInteractor::hide()
{
    //fprintf(stderr,"vvIntersectionInteractor::hide(%s)\n", _interactorName);

   /* if (moveTransform->getNumParents())
    {
        vsg::Group *p = moveTransform->getParent(0);
        p->removeChild(moveTransform);
    }*/

    if (label_)
        label_->hide();
}
bool vvIntersectionInteractor::isInitializedThroughSharedState()
{
    return m_isInitializedThroughSharedState;
}
void vvIntersectionInteractor::setShared(bool state)
{
    assert(!state && "sharing of vvIntersectionInteractor state requested, but sharing not implemented for vvIntersectionInteractor type");
}

bool vvIntersectionInteractor::isShared() const
{
    return m_sharedState != nullptr;
}
void vvIntersectionInteractor::enableIntersection()
{
    if (vv->debugLevel(4))
        fprintf(stderr, "vvIntersectionInteractor(%s)::enableIntersection\n", _interactorName);

    if (!_intersectionEnabled)
    {
        _intersectionEnabled = true;
        vruiIntersection::getIntersectorForAction("coAction")->add(vNode, this);

        vvIntersectionInteractorManager::the()->add(this);

        if (vv->debugLevel(2))
            fprintf(stderr, "adding to intersectorlist\n");
    }
}

void vvIntersectionInteractor::disableIntersection()
{
    if (vv->debugLevel(2))
        fprintf(stderr, "vvIntersectionInteractor(%s)::disableIntersection\n", _interactorName);

    // interactor is normally unregistered in miss
    // if we disable intersection miss is not called anymore
    // therefore we make sure that the interactor is unregistered
    if (registered)
    {
        coInteractionManager::the()->unregisterInteraction(this);
    }
    if (_intersectionEnabled)
    {
        if (vv->debugLevel(2))
            fprintf(stderr, "removing from intersectorlist\n");
        _intersectionEnabled = false;
        vruiIntersection::getIntersectorForAction("coAction")->remove(vNode);
        vvIntersectionInteractorManager::the()->remove(this);
    }

    resetState();
}

const vsg::dmat4 &vvIntersectionInteractor::getPointerMat() const
{
    if (isMouse())
        return vv->getMouseMat();
    else
        return vv->getPointerMat();
}
void vvIntersectionInteractor::updateSharedState()
{
    assert(!m_sharedState && "updating shared state of vvIntersectionInteractor requested, but sharing not implemented for interactor type");
}
int vvIntersectionInteractor::hit(vruiHit *hit)
{
    if (vv->debugLevel(4))
        fprintf(stderr, "vvIntersectionInteractor(%s)::hit\n", _interactorName);

    _hit = true;
    _wasHit = false;
    if (_justHit)
    {
        if (vv->debugLevel(4))
            fprintf(stderr, "VRIntersectionInteractor(%s)::hit justHit\n", _interactorName);

        _wasHit = true;

        if (label_ && vv->debugLevel(4))
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
        vsg::dvec3 wp(v[0], v[1], v[2]);
        _hitPos = wp * vv->getInvBaseMat();
        auto VSGVruinode = dynamic_cast<VSGVruiNode *>(hit->getNode());
        if (VSGVruinode)
        {
            if(_highliteHitNodeOnly)
            {
                vsg::Node* oldHitNode = _hitNode.get(); 
                vsg::Node* newHitNode =  VSGVruinode->getNodePtr();

               /* if (oldHitNode != nullptr && oldHitNode != newHitNode && _interactionHitNode == nullptr) // reset color, if you move from one hit node directly to another hit node                                                                                                                                                                                                                                                                                                                           
                    _hitNode->setStateSet(NULL);     */           
                
                _hitNode = VSGVruinode->getNodePtr();

               /* if (_interactionHitNode == nullptr && _hitNode->getStateSet() != _intersectedHl)
                    _hitNode->setStateSet(_intersectedHl);*/
            }
            else
                _hitNode = VSGVruinode->getNodePtr();
        }
        else
        {
            _hitNode = nullptr;
        }
    }
    else
    {
        _hitPos = getTrans(getMatrix());
        _hitNode = nullptr;
    }

    _justHit = false;

    return ACTION_CALL_ON_MISS;
}

void vvIntersectionInteractor::miss()
{
    _hit = false;
    _justHit = true;
    _wasHit = false;
    if (vv->debugLevel(4))
        fprintf(stderr, "vvIntersectionInteractor(%s)::miss\n", _interactorName);

    if (registered)
    {
        coInteractionManager::the()->unregisterInteraction(this);

        if (label_ && vv->debugLevel(5))
            label_->hide();
    }
}

vsg::Node *vvIntersectionInteractor::findGeode(vsg::Node *n)
{
    vsg::Node *geode;
    vsg::Group *group;
    geode = dynamic_cast<vsg::Node *>(n);
    if (geode != NULL)
        return geode;
    group = dynamic_cast<vsg::Group *>(n);
    for (size_t i = group->children.size() - 1; i >= 0; i--)
    {
        geode = findGeode(group->children[i]);
        if (geode != NULL)
            return geode;
    }
    return NULL;
}

void vvIntersectionInteractor::addIcon()
{
    // add icon to zeigestrahl and hilight the interactor (if _highliteHitNodeOnly = true the interactor gets highlited in the hit() function)

    //fprintf(stderr,"vvIntersectionInteractor(%s)::addIcon and show intersected hl\n", _interactorName);
    coInteraction::addIcon();

    if (getState() == coInteraction::Idle)
    {
        /*_oldHl = moveTransform->getStateSet();

        if (!_standardHL)
        {
            osg::Material *selMaterial = new osg::Material();
            vsg::vec4 colDiff = vsg::vec4(0.5, 0.5, 0.5, 1.0);
            vsg::vec4 colAmb = vsg::vec4(0.5, 0.5, 0.5, 1.0);
            vsg::Node *geode = findGeode(geometryNode.get());

            for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
            {
                vsg::ref_ptr<vsg::Node> drawable = geode->getDrawable(i);
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
            }

            selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, vsg::vec4f(colDiff.r() + 0.2, colDiff.g() + 0.2, colDiff.b() + 0.2, colDiff.a()));
            selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, vsg::vec4f(colAmb.r() + 0.2, colAmb.g() + 0.2, colAmb.b() + 0.2, colAmb.a()));
            selMaterial->setEmission(osg::Material::FRONT_AND_BACK, vsg::vec4f(0.1f, 0.1f, 0.1f, 1.0f));
            selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
            selMaterial->setColorMode(osg::Material::OFF);
            _intersectedHl->setAttribute(selMaterial, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
        }
        if(!_highliteHitNodeOnly)
            moveTransform->setStateSet(_intersectedHl.get()); */
    }
    // else it is still active and we want to keep the selected hlight
}

void vvIntersectionInteractor::removeIcon()
{
    //fprintf(stderr,"vvIntersectionInteractor(%s)::removeIcon and set hl off\n", _interactorName);
    coInteraction::removeIcon();
}

void vvIntersectionInteractor::resetState()
{
    /*_oldHl = NULL;
    moveTransform->setStateSet(NULL);
    
    if(_hitNode)    
        _hitNode->setStateSet(NULL);
    
    if(_interactionHitNode != nullptr)         
    {
        _interactionHitNode->setStateSet(NULL);
        _interactionHitNode = nullptr;
    }*/
    
}

void vvIntersectionInteractor::startInteraction()
{

    if (vv->debugLevel(4))
        fprintf(stderr, "\nvvIntersectionInteractor::startInteraction and set selected hl\n");
   /* _oldHl = moveTransform->getStateSet();

    if (!_standardHL)
    {
        osg::Material *selMaterial = new osg::Material();
        vsg::vec4 colDiff = vsg::vec4(0.5, 0.5, 0.5, 1.0);
        vsg::vec4 colAmb = vsg::vec4(0.5, 0.5, 0.5, 1.0);
        vsg::Node *geode = findGeode(geometryNode.get());

        for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
        {
            vsg::ref_ptr<vsg::Node> drawable = geode->getDrawable(i);
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
        }

        selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, vsg::vec4f(colDiff.r() - 0.2, colDiff.g() - 0.2, colDiff.b() - 0.2, colDiff.a()));
        selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, vsg::vec4f(colAmb.r() - 0.2, colAmb.g() - 0.2, colAmb.b() - 0.2, colAmb.a()));
        selMaterial->setEmission(osg::Material::FRONT_AND_BACK, vsg::vec4f(0.1f, 0.1f, 0.1f, 1.0f));
        selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
        selMaterial->setColorMode(osg::Material::OFF);
        _selectedHl->setAttribute(selMaterial, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
    }
    if(_highliteHitNodeOnly)
    {
        _interactionHitNode = _hitNode.get();
        if(_interactionHitNode != nullptr)
            _interactionHitNode->setStateSet(_selectedHl.get());
    }
    else
        moveTransform->setStateSet(_selectedHl.get());

    //Interactor 0-5 are of clipplane*/
}

void vvIntersectionInteractor::stopInteraction()
{
    if (vv->debugLevel(4))
        fprintf(stderr, "\nvvIntersectionInteractor::stopInteraction\n");

    // in case that miss was called before but interaction was ongoing
    // we have to unregister
    /*if (!_highliteHitNodeOnly)
        moveTransform->setStateSet(_oldHl.get());*/
   
    resetState();
}

void vvIntersectionInteractor::doInteraction()
{
    if (vv->debugLevel(5))
        fprintf(stderr, "\nvvIntersectionInteractor::doInteraction\n");
}

void vvIntersectionInteractor::keepSize()
{
    if (vv->debugLevel(5))
        fprintf(stderr, "\nvvIntersectionInteractor::keepSize scale=%f\n", vv->getScale());

    float interScale;
    if (constantInteractorSize_) // Uwe Mode: scale icon to keep size indepened of interactor position and world scale
    {
        vsg::dvec3 WorldPos = getTrans(moveTransform->matrix) * vv->getBaseMat();
        interScale = vv->getInteractorScale(WorldPos) * _interSize;
    }
    else // Daniela Mode: scale icon to keep COVER.IconSize independed of world scale
    {
        if (firstTime)
        {
           /* geometryNode->dirtyBound();
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
            }*/
        }
        interScale = _interSize / (vv->getScale() * iconSize_);
    }
    _scale = interScale;
    scaleTransform->matrix = (vsg::scale(interScale, interScale, interScale));
}

float vvIntersectionInteractor::getScale() const
{
    return _scale;
}

void vvIntersectionInteractor::preFrame()
{
    if (vv->debugLevel(5))
        fprintf(stderr, "\nvvIntersectionInteractor::preFrame\n");
    keepSize();

    // update label pos
    if (label_)
    {
        vsg::dmat4 m;
        m = moveTransform->matrix;
        // update label position
        vsg::dmat4 o_to_w = vv->getBaseMat();
        vsg::dvec3 pos_w, pos_o;
        pos_o = getTrans(m);
        pos_w = o_to_w * pos_o;
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
void vvIntersectionInteractor::startTraverseInteractors()
{
   if (vv->debugLevel(4))
      fprintf(stderr,"\nvvIntersectionInteractor::startTraverseInteractors\n");

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
      //vvSceneGraph::instance()->getHandTransform()->matrix = (m);
   }
}
*/

/*
void vvIntersectionInteractor::traverseInteractors()
{

   if (vv->debugLevel(4))
      fprintf(stderr,"\nvvIntersectionInteractor::traverseInteractors size %d index %d\n", (int)interactors->size(), traverseIndex);

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
//       //vvSceneGraph::instance()->getHandTransform()->matrix = (m);
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
//          fprintf(stderr,"\nvvIntersectionInteractor::startInteraction\n");
//          interactors->at(traverseIndex)->startInteraction(true);
//       }
//    }

   //traverseIndex++;
}
*/

/*

void vvIntersectionInteractor::stopTraverseInteractors()
{

   if (vv->debugLevel(4))
      fprintf(stderr,"\nvvIntersectionInteractor::stopTraverseInteractors\n");

   traverseIndex = 0;

   isTraverseInteractors = false;
}
*/
bool vvIntersectionInteractor::isEnabled()
{
    return _intersectionEnabled;
}

vsg::vec3 vvIntersectionInteractor::restrictToVisibleScene(vsg::vec3 pos)
{
    vsg::vec3 rpos(0.0, 0.0, 0.0); //restricted position
    rpos = pos;

    /*osg::BoundingBox box = vv->getBBox(vv->getObjectsRoot());
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
    }*/
    return rpos;
}
void vvIntersectionInteractor::setCaseTransform(vsg::MatrixTransform *m)
{
    //fprintf(stderr,"vvIntersectionInteractor(%s)::setCaseTransform\n", _interactorName);
    if (m)
    {
        interactorCaseTransform = m;
        parent = interactorCaseTransform;
        //fprintf(stderr,"...setting parent to interactorCaseTransform\n");
        // remove moveDCS from scaleDCS
        /*if (moveTransform->getNumParents() > 0)
        {
            if (moveTransform->getParent(0)->removeChild(moveTransform.get()))
            {
                // add moveDCS to case

                parent->addChild(moveTransform.get());
                //pfPrint(vv->getObjectsRoot(),  PFTRAV_SELF|PFTRAV_DESCEND, PFPRINT_VB_NOTICE, NULL);
            }
        }
        else if (vv->debugLevel(5))
            fprintf(stderr, "vvIntersectionInteractor(%s)::setCaseTransform moveDCS is actually not under scaleDCS\n", _interactorName);*/
    }
    else
    {
        interactorCaseTransform = NULL;
        if (vv->debugLevel(5))
            fprintf(stderr, "---vvIntersectionInteractor::setCaseDCS dcs=NULL\n");
    }
}
