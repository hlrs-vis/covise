/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: Class to generate an animation                            **
 **              for Cyberclassroom mathematics                            **
 **                                                                        **
 ** cpp file                                                               **
 ** Author: A.Cyran                                                        **
 **                                                                        **
 ** History:                                                               **
 **     12.2010 initial version                                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/
#include "HfT_osg_Animation.h"
#include "HfT_osg_Parametric_Surface.h"
#include "HfT_osg_Sphere.h"
#include "HfT_osg_MobiusStrip.h"
#include "HfT_osg_Plane.h"
#include <cover/coVRPluginSupport.h>
#include <osg/NodeCallback>
#include <osgGA/TrackballManipulator>

using namespace osg;
using namespace covise;
using namespace opencover;

//---------------------------------------------------
//Implements HfT_osg_Animation::HfT_osg_Animation()
//---------------------------------------------------
HfT_osg_Animation::HfT_osg_Animation()
{
    //unit sphere
    m_rpAnimatedSphere = new Sphere(Vec3d(0.0, 0.0, 0.0), 1.0);
    m_animationTime = 0.0;
    m_animationTimeStep = 0.0;
    createAnimationSubtree();
    m_rpAnimationPath = new AnimationPath();
    // Loop mode for the flow of the animation
    m_rpAnimationPath->setLoopMode(AnimationPath::SWING);
    setAnimationPath();
    //Set the animation path to the transform node
    m_rpAnimTrafoNode->setUpdateCallback(new AnimationPathCallback(m_rpAnimationPath));
    m_pColor = new Vec4(1.0, 1.0, 1.0, 1.0);
    createAnimationColor();
}

//---------------------------------------------------
//Implements HfT_osg_Animation(const Sphere& iAnimatedSphere, Vec4 iColor)
//---------------------------------------------------
HfT_osg_Animation::HfT_osg_Animation(const Sphere &iAnimatedSphere, Vec4 iColor)
{
    m_rpAnimatedSphere = new Sphere(iAnimatedSphere);
    m_animationTime = 0.0;
    m_animationTimeStep = 0.0;
    createAnimationSubtree();
    m_rpAnimationPath = new AnimationPath();
    // Loop mode for the flow of the animation
    m_rpAnimationPath->setLoopMode(AnimationPath::SWING);
    setAnimationPath();
    //Set the animation path to the transform node
    m_rpAnimTrafoNode->setUpdateCallback(new AnimationPathCallback(m_rpAnimationPath));
    m_pColor = new Vec4(iColor);
    createAnimationColor();
}

//---------------------------------------------------
//Implements HfT_osg_Animation(const Sphere& iAnimatedSphere,
//                                                         Vec3Array iPathCoordinates, Vec4 iColor);
//---------------------------------------------------
HfT_osg_Animation::HfT_osg_Animation(const Sphere &iAnimatedSphere,
                                     Vec3Array *iPathCoordinates, Vec4 iColor)
{
    m_rpAnimatedSphere = new Sphere(iAnimatedSphere);
    m_animationTime = 0.0;
    m_animationTimeStep = 1.0;
    createAnimationSubtree();
    m_rpAnimationPath = new AnimationPath();
    // Loop mode for the flow of the animation
    m_rpAnimationPath->setLoopMode(AnimationPath::SWING);

    for (int i = 0; i < iPathCoordinates->size(); i++)
    {
        m_rpAnimationPath->insert(m_animationTimeStep * i,
                                  AnimationPath::ControlPoint(iPathCoordinates->at(i)));
        m_animationTime = m_animationTime + m_animationTimeStep * i;
    }
    //Set the animation path to the transform node
    m_rpAnimTrafoNode->setUpdateCallback(new AnimationPathCallback(m_rpAnimationPath));
    m_pColor = new Vec4(iColor);
    createAnimationColor();
}

//---------------------------------------------------
//Implements HfT_osg_Animation(const Sphere& iAnimatedSphere,
//                                                        HfT_osg_Parametric_Surface& iPathObject,
//                                                        Vec4 iColor, char iLine)
//---------------------------------------------------
HfT_osg_Animation::HfT_osg_Animation(const Sphere &iAnimatedSphere,
                                     HfT_osg_Parametric_Surface &iPathObject, Vec4 iColor, char iLine)
{
    m_rpAnimatedSphere = new Sphere(iAnimatedSphere);
    m_animationTime = 0.0;
    m_animationTimeStep = 1.0 / 30.0;
    createAnimationSubtree();
    m_rpAnimationPath = new AnimationPath();
    // Loop mode for the flow of the animation
    m_rpAnimationPath->setLoopMode(AnimationPath::SWING);

    int edgeNumber;

    switch (iLine)
    {
    case 'E':
        for (int i = 0; i < iPathObject.m_rpEquatorEdges->size(); i++)
        {
            edgeNumber = (iPathObject.m_rpEquatorEdges->at(i));
            m_rpAnimationPath->insert((i * m_animationTimeStep),
                                      AnimationPath::ControlPoint((iPathObject.SupportingPoints()->at(edgeNumber)) + iPathObject.Normals()->at(edgeNumber) * (m_rpAnimatedSphere->getRadius())));
            m_animationTime = m_animationTime + m_animationTimeStep * i;
        }
        break;

    case 'U':
        for (int i = 0; i < iPathObject.m_rpDirectrixUEdges->size(); i++)
        {
            edgeNumber = (iPathObject.m_rpDirectrixUEdges->at(i));
            m_rpAnimationPath->insert((i * m_animationTimeStep),
                                      AnimationPath::ControlPoint((iPathObject.SupportingPoints()->at(edgeNumber)) + iPathObject.Normals()->at(edgeNumber) * (m_rpAnimatedSphere->getRadius())));
            m_animationTime = m_animationTime + m_animationTimeStep * i;
        }
        break;

    case 'V':
        for (int i = 0; i < iPathObject.m_rpDirectrixVEdges->size(); i++)
        {
            edgeNumber = (iPathObject.m_rpDirectrixVEdges->at(i));
            m_rpAnimationPath->insert((i * m_animationTimeStep),
                                      AnimationPath::ControlPoint((iPathObject.SupportingPoints()->at(edgeNumber)) + iPathObject.Normals()->at(edgeNumber) * (m_rpAnimatedSphere->getRadius())));
            m_animationTime = m_animationTime + m_animationTimeStep * i;
        }
        break;
    }
    //Set the animation path to the transform node
    m_rpAnimTrafoNode->setUpdateCallback(new AnimationPathCallback(m_rpAnimationPath));
    m_pColor = new Vec4(iColor);
    createAnimationColor();
}

//---------------------------------------------------
//Implements HfT_osg_Animation::~HfT_osg_Animation()
//---------------------------------------------------
HfT_osg_Animation::~HfT_osg_Animation()
{
    m_animationTime = 0.0;
    m_animationTimeStep = 0.0;

    if (m_pColor != NULL)
    {
        //Cleanup the pointer to the surface color
        //Calls destructor of the vec4
        delete m_pColor;
    }
}

//---------------------------------------------------
//Implements HfT_osg_Animation::setAnimationPath(Vec3Array* iPath)
//---------------------------------------------------
void HfT_osg_Animation::setAnimationPath(Vec3Array *iPath)
{
    //If animation path is already set, clear it
    if (!(m_rpAnimationPath->empty()))
    {
        m_rpAnimationPath = new AnimationPath();
        m_rpAnimationPath->setLoopMode(AnimationPath::SWING);
        m_animationTime = 0.0;
        m_animationTimeStep = 0.0;
    }
    //If there is no optional parameter set, set a default animation path
    if (iPath == NULL)
    {
        m_animationTimeStep = 2.0;
        m_rpAnimationPath->insert(0.0, AnimationPath::ControlPoint(Vec3(1, 1, 0)));
        m_rpAnimationPath->insert(2.0, AnimationPath::ControlPoint(Vec3(-1, 1, 0)));
        m_rpAnimationPath->insert(4.0, AnimationPath::ControlPoint(Vec3(-1, -1, 0)));
        m_rpAnimationPath->insert(6.0, AnimationPath::ControlPoint(Vec3(1, -1, 0)));
        m_rpAnimationPath->insert(8.0, AnimationPath::ControlPoint(Vec3(1, 1, 0)));
        m_animationTime = 8.0;
    }
    else
    {
        m_animationTimeStep = 1.0;
        for (int i = 0; i < iPath->size(); i++)
        {
            m_rpAnimationPath->insert(m_animationTimeStep * i,
                                      AnimationPath::ControlPoint(iPath->at(i)));
            m_animationTime = m_animationTime + m_animationTimeStep * i;
        }
    }
    m_rpAnimTrafoNode->setUpdateCallback(new AnimationPathCallback(m_rpAnimationPath));
}

//---------------------------------------------------
//Implements HfT_osg_Animation::setReverseAnimationPath()
//---------------------------------------------------
void HfT_osg_Animation::setReverseAnimationPath()
{
    AnimationPath::TimeControlPointMap timeMap(m_rpAnimationPath->getTimeControlPointMap());

    m_rpAnimationPath = new AnimationPath();
    m_rpAnimationPath->setLoopMode(AnimationPath::SWING);

    for (int i = 0; i < timeMap.size(); i++)
    {
        m_rpAnimationPath->insert(i * m_animationTimeStep, timeMap[((timeMap.size() - 1) - i) * m_animationTimeStep]);
    }
    m_rpAnimTrafoNode->setUpdateCallback(new AnimationPathCallback(m_rpAnimationPath));
}

//---------------------------------------------------
//Implements HfT_osg_Animation::createAnimationSubtree()
//---------------------------------------------------
void HfT_osg_Animation::createAnimationSubtree()
{
    m_rpAnimSphereDrawable = new ShapeDrawable(m_rpAnimatedSphere);
    m_rpAnimSphereDrawable->setDataVariance(Object::DYNAMIC);
    m_rpAnimGeode = new Geode();
    m_rpAnimGeode->addDrawable(m_rpAnimSphereDrawable);
    m_rpAnimTrafoNode = new MatrixTransform();
    m_rpAnimTrafoNode->setDataVariance(Object::DYNAMIC);
    m_rpAnimGeode->setNodeMask(m_rpAnimGeode->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    m_rpSphereStateSet = m_rpAnimSphereDrawable->getOrCreateStateSet();
    m_rpAnimTrafoNode->addChild(m_rpAnimGeode);
}

//---------------------------------------------------
//Implements HfT_osg_Animation::createAnimationColor()
//---------------------------------------------------
void HfT_osg_Animation::createAnimationColor()
{
    m_rpSphereMaterial = new Material();
    m_rpSphereStateSet->setAttributeAndModes(m_rpSphereMaterial,
                                             osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    m_rpSphereMaterial->setColorMode(Material::OFF);
    m_rpSphereMaterial->setDiffuse(Material::FRONT_AND_BACK, *m_pColor);
    m_rpSphereMaterial->setSpecular(Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    m_rpSphereMaterial->setAmbient(Material::FRONT_AND_BACK, Vec4((m_pColor->x()) * 0.3,
                                                                  (m_pColor->y()) * 0.3, (m_pColor->z()) * 0.3,
                                                                  m_pColor->w()));
    m_rpSphereMaterial->setShininess(Material::FRONT_AND_BACK, 100.0);
}
