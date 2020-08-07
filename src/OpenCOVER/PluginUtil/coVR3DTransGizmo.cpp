#include "coVR3DTransGizmo.h"
#include <OpenVRUI/osg/mathUtils.h>
#include <osg/MatrixTransform>
#include <cover/coVRNavigationManager.h>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>
#include <osg/io_utils>
#include <vrbclient/SharedState.h>

const float ArrowLength = 5.0f;

using namespace opencover;

coVR3DTransGizmo::coVR3DTransGizmo(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority = Medium)
    :coVRIntersectionInteractor(s, type, iconName, interactorName, priority, true)
{
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "new coVR3DTransGizmo(%s)\n", interactorName);
    }

    createGeometry();

    coVR3DTransGizmo::updateTransform(m);
}

coVR3DTransGizmo::~coVR3DTransGizmo()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete ~coVR3DTransGizmo\n");
}

void
coVR3DTransGizmo::createGeometry()
{
   if (cover->debugLevel(4))
        fprintf(stderr, "\ncoVR3DTransGizmo::createGeometry\n");

    bool restrictedInteraction = true; 

    osg::ShapeDrawable *sphereDrawable, *xCylDrawable, *yCylDrawable, *zCylDrawable, *xyPlaneDrawable,*xzPlaneDrawable,*yzPlaneDrawable;

    osg::Vec3 origin(0, 0, 0), px(1, 0, 0), py(0, 1, 0), pz(0, 0, 1);
    osg::Vec3 yaxis(0, 1, 0);
    osg::Vec4 red(1, 0, 0, 1), green(0, 1, 0, 1), blue(0, 0, 1, 1), color(0.5, 0.5, 0.5, 1);
    //warum initialisierung und dann das ? 
    red.set(0.5, 0.2, 0.2, 1.0);
    green.set(0.2, 0.5, 0.2, 1.0);
    blue.set(0.2, 0.2, 0.5, 1.0);

    axisTransform = new osg::MatrixTransform();
    axisTransform->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));
    geometryNode = axisTransform.get();
    scaleTransform->addChild(geometryNode.get());

    osg::Sphere *mySphere = new osg::Sphere(origin, 0.5);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    sphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    sphereDrawable->setColor(color);
    sphereGeode = new osg::Geode();
    sphereGeode->addDrawable(sphereDrawable);
    axisTransform->addChild(sphereGeode.get());

    //Merke die Geodes die gebraucht werden, werden im originial in if(restricted erzeugt) alle anderen sind temporär
    //create axis
    auto cyl = new osg::Cylinder(origin, 0.15, ArrowLength);
    xCylDrawable = new osg::ShapeDrawable(cyl, hint);
    yCylDrawable = new osg::ShapeDrawable(cyl, hint);
    zCylDrawable = new osg::ShapeDrawable(cyl, hint);

    xCylDrawable->setColor(red);
    yCylDrawable->setColor(green);
    zCylDrawable->setColor(blue);

    xAxisTransform = new osg::MatrixTransform();
    xAxisTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(90.), 0, 1, 0)*osg::Matrix::translate(osg::Vec3(ArrowLength, 0, 0)*0.5));
    yAxisTransform = new osg::MatrixTransform();
    yAxisTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(-90.), 1, 0, 0)*osg::Matrix::translate(osg::Vec3(0, ArrowLength, 0)*0.5));
    zAxisTransform = new osg::MatrixTransform();
    zAxisTransform->setMatrix(osg::Matrix::translate(osg::Vec3(0, 0, ArrowLength)*0.5));

    translateXaxisGeode = new osg::Geode;
    translateXaxisGeode->setName("xAxis");
    translateXaxisGeode->addDrawable(xCylDrawable);
    xAxisTransform->addChild(translateXaxisGeode);
    translateYaxisGeode = new osg::Geode;
    translateYaxisGeode->setName("yAxis");
    translateYaxisGeode->addDrawable(yCylDrawable);
    yAxisTransform->addChild(translateYaxisGeode);
    translateZaxisGeode = new osg::Geode;
    translateZaxisGeode->setName("zAxis");
    translateZaxisGeode->addDrawable(zCylDrawable);
    zAxisTransform->addChild(translateZaxisGeode);

    axisTransform->addChild(xAxisTransform);
    axisTransform->addChild(yAxisTransform);
    axisTransform->addChild(zAxisTransform);
    
    //create planes
    auto plane = new osg::Box(origin,ArrowLength/3, 0.1, ArrowLength/3);
    xzPlaneDrawable = new osg::ShapeDrawable(plane, hint);
    xyPlaneDrawable = new osg::ShapeDrawable(plane, hint);
    yzPlaneDrawable = new osg::ShapeDrawable(plane, hint);

    xyPlaneDrawable->setColor(blue);
    xzPlaneDrawable->setColor(green);
    yzPlaneDrawable->setColor(red);

    xzPlaneTransform = new osg::MatrixTransform();
    xzPlaneTransform->setMatrix(osg::Matrix::translate(osg::Vec3(ArrowLength/2, 0, ArrowLength/2)));
    xyPlaneTransform = new osg::MatrixTransform();
    xyPlaneTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(-90.), 1, 0, 0)*osg::Matrix::translate(osg::Vec3(ArrowLength/2, ArrowLength/2, 0)));
    yzPlaneTransform = new osg::MatrixTransform();

    yzPlaneTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(-90.), 0, 0, 1)*osg::Matrix::translate(osg::Vec3(0, ArrowLength/2, ArrowLength/2)));

    translateXZplaneGeode = new osg::Geode;
    translateXZplaneGeode->setName("xz");
    translateXZplaneGeode->addDrawable(xzPlaneDrawable);
    xzPlaneTransform->addChild(translateXZplaneGeode);
    translateXYplaneGeode = new osg::Geode;
    translateXYplaneGeode->setName("xy");
    translateXYplaneGeode->addDrawable(xyPlaneDrawable);
    xyPlaneTransform->addChild(translateXYplaneGeode);
    translateYZplaneGeode = new osg::Geode;
    translateYZplaneGeode->setName("yz");
    translateYZplaneGeode->addDrawable(yzPlaneDrawable);
    yzPlaneTransform->addChild(translateYZplaneGeode);

    axisTransform->addChild(xzPlaneTransform);
    axisTransform->addChild(xyPlaneTransform);
    axisTransform->addChild(yzPlaneTransform);




}

//wozu brauche ich das ?
void coVR3DTransGizmo::updateSharedState()
{
    if (auto st = static_cast<SharedMatrix *>(m_sharedState.get()))
    {
        *st = _oldInteractorXformMat_o;//myPosition
    }
}
void coVR3DTransGizmo::startInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DTransGizmo::startInteraction\n");

    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix o_to_w = cover->getBaseMat();

    osg::Matrix hm = getPointerMat(); // hand matrix weltcoord
    osg::Matrix hm_o = hm * w_to_o; // hand matrix objekt coord
    _oldHandMat = hm;
    _invOldHandMat_o.invert(hm_o); // store the inv hand matrix

    osg::Matrix interMat = _interMat_o * o_to_w;

    _oldInteractorXformMat_o = _interMat_o;

    osg::Vec3 interPos = getMatrix().getTrans();
    // get diff between intersection point and sphere center
    _diff = interPos - _hitPos;
    _distance = (_hitPos - hm_o.getTrans()).length();

    _translateXonly = _hitNode == translateXaxisGeode;
    _translateYonly = _hitNode == translateYaxisGeode;
    _translateZonly = _hitNode == translateZaxisGeode;
    _translateXYonly = _hitNode == translateXYplaneGeode;
    _translateXZonly = _hitNode == translateXZplaneGeode;
    _translateYZonly = _hitNode == translateYZplaneGeode;

    /* wie setze ich das hier um, brauch man das ? ###################################
    if (!_rotateOnly && !_translateOnly)
    {
        _translateOnly = is2D();
    }
    */

    coVRIntersectionInteractor::startInteraction();
    /* add sphere to show old position of gizmo
    osg::ShapeDrawable *sphereDrawable;
    osg::Sphere *mySphere = new osg::Sphere(interPos, 0.2);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    sphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    sphereDrawable->setColor(osg::Vec4(0,1,1,1));
    sphereOldPosGeode = new osg::Geode();
    sphereOldPosGeode->addDrawable(sphereDrawable);
    geometryNode->addChild(sphereOldPosGeode);
    */
}
void coVR3DTransGizmo::doInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DTransGizmo::move\n");
    
    osg::Vec3 origin(0, 0, 0);
    osg::Vec3 yaxis(0, 1, 0);

    osg::Matrix currHandMat = getPointerMat();
    // forbid translation in y-direction if traverseInteractors is on ############## wozu brauche ich das ? 
    if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors && coVRConfig::instance()->useWiiNavigationVisenso())
    {
        osg::Vec3 trans = currHandMat.getTrans();
        trans[1] = _oldHandMat.getTrans()[1];
        currHandMat.setTrans(trans);
    }

    osg::Matrix o_to_w = cover->getBaseMat();
    // get hand mat in object coords
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix currHandMat_o = currHandMat * w_to_o;

    // translate from interactor to hand and back
    osg::Matrix transToHand_o, revTransToHand_o;

    transToHand_o.makeTranslate(currHandMat_o.getTrans() - _oldInteractorXformMat_o.getTrans());
    revTransToHand_o.makeTranslate(_oldInteractorXformMat_o.getTrans() - currHandMat_o.getTrans());

    osg::Matrix relHandMoveMat_o = _invOldHandMat_o * currHandMat_o;
    osg::Matrix interactorXformMat_o = _oldInteractorXformMat_o;

    auto lp1_o = origin * currHandMat_o;
    auto lp2_o = yaxis * currHandMat_o; 
    auto pointerDir_o = lp2_o - lp1_o;
    pointerDir_o.normalize();   
    // get hand pos in object coords
    auto currHandPos_o = currHandMat_o.getTrans();  
    auto interPos = currHandPos_o + pointerDir_o * _distance + _diff;

    if(_translateXonly)
    {
        interPos.y() = _oldInteractorXformMat_o.getTrans().y();
        interPos.z() = _oldInteractorXformMat_o.getTrans().z();
        interactorXformMat_o.setTrans(interPos); 
    }
    else if(_translateYonly)
    {
        interPos.x() = _oldInteractorXformMat_o.getTrans().x();
        interPos.z() = _oldInteractorXformMat_o.getTrans().z();
        interactorXformMat_o.setTrans(interPos); 
    }
    else if(_translateZonly)
    {
        interPos.y() = _oldInteractorXformMat_o.getTrans().y();
        interPos.x() = _oldInteractorXformMat_o.getTrans().x();
        interactorXformMat_o.setTrans(interPos); 
    }
    else if(_translateXYonly)
    {
        interPos.z() = _oldInteractorXformMat_o.getTrans().z();
        interactorXformMat_o.setTrans(interPos);    
    }
    else if(_translateXZonly)
    {  
        interPos.y() = _oldInteractorXformMat_o.getTrans().y();
        interactorXformMat_o.setTrans(interPos); 
    }
    else if(_translateYZonly)
    {
        interPos.x() = _oldInteractorXformMat_o.getTrans().x();
        interactorXformMat_o.setTrans(interPos); 
    }
    else // allow translatation in all directions
        interactorXformMat_o.setTrans(interPos); 

    // save old transformation
    //_oldInteractorXformMat_o = interactorXformMat_o;

    _oldHandMat = currHandMat; // save current hand for rotation start
    _invOldHandMat_o.invert(currHandMat_o);

    if (cover->restrictOn())
    {
        // restrict to visible scene
        osg::Vec3 pos_o, restrictedPos_o;
        pos_o = interactorXformMat_o.getTrans();
        restrictedPos_o = restrictToVisibleScene(pos_o);
        interactorXformMat_o.setTrans(restrictedPos_o);
    }

    if (coVRNavigationManager::instance()->isSnapping())
    {
        if (coVRNavigationManager::instance()->isDegreeSnapping())
        {
            // snap orientation
            snapToDegrees(coVRNavigationManager::instance()->snappingDegrees(), &interactorXformMat_o);
        }
        else
        {
            // snap orientation to 45 degree
            snapTo45Degrees(&interactorXformMat_o);
        }
    }


    // and now we apply it
    updateTransform(interactorXformMat_o);
}

void coVR3DTransGizmo::updateTransform(osg::Matrix m)
{
    if (cover->debugLevel(5))
        fprintf(stderr, "coVR3DTransGizmo:setMatrix\n");
    _interMat_o = m;
    moveTransform->setMatrix(m);
    if (m_sharedState)
    {
        if (auto st = static_cast<SharedMatrix *>(m_sharedState.get()))
        {
            *st = m;
        }
    }
}
void coVR3DTransGizmo::setShared(bool shared)
{
    if (shared)
    {
        if (!m_sharedState)
        {
            m_sharedState.reset(new SharedMatrix("interactor." + std::string(_interactorName), _oldInteractorXformMat_o));//myPosition
            m_sharedState->setUpdateFunction([this]() {
                m_isInitializedThroughSharedState = true;
                osg::Matrix interactorXformMat_o = *static_cast<SharedMatrix *>(m_sharedState.get());
                if (cover->restrictOn())
                {
                    // restrict to visible scene
                    osg::Vec3 pos_o, restrictedPos_o;
                    pos_o = interactorXformMat_o.getTrans();
                    restrictedPos_o = restrictToVisibleScene(pos_o);
                    interactorXformMat_o.setTrans(restrictedPos_o);
                }

                if (coVRNavigationManager::instance()->isSnapping())
                {
                    if (coVRNavigationManager::instance()->isDegreeSnapping())
                    {
                        // snap orientation
                        snapToDegrees(coVRNavigationManager::instance()->snappingDegrees(), &interactorXformMat_o);
                    }
                    else
                    {
                        // snap orientation to 45 degree
                        snapTo45Degrees(&interactorXformMat_o);
                    }
                }
                updateTransform(interactorXformMat_o);
            });
        }
    }
    else
    {
        m_sharedState.reset(nullptr);
    }
}

