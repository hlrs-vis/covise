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
 ** header file                                                            **
 ** Author: A.Cyran                                                        **
 **                                                                        **
 ** History:                                                               **
 **     12.2010 initial version                                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef HFT_OSG_ANIMATION_H_
#define HFT_OSG_ANIMATION_H_

#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/Shape>
#include <osg/AnimationPath>
#include <osg/Array>
#include <osg/Vec3>
#include <osg/PrimitiveSet>
#include <osg/Material>
#include <osg/Vec4>
#include <osg/StateSet>

class HfT_osg_Parametric_Surface;
class HfT_osg_Sphere;
class HfT_osg_MobiusStrip;

using namespace osg;

class HfT_osg_Animation
{

public:
    /****** variables ******/
    /*
   * Ref pointer to the matrix transform node, which is parent of the geode
   * of the sphere.
   */
    ref_ptr<MatrixTransform> m_rpAnimTrafoNode;

    /****** constructors and destructors ******/
    /*
   * Standard constructor form a white sphere
   */
    HfT_osg_Animation();

    /*
   * Constructor
   * Creates a osg sphere with an animation path 
   * and the belonging tree hierarchy
   *
   * Parameters:      const Sphere& iAnimatedSphere
   *                  existing osg sphere to copy it
   *
   *                             Vec4 iColor
   *                  color for the new sphere
   */
    HfT_osg_Animation(const Sphere &iAnimatedSphere, Vec4 iColor);

    /*
   * Constructor
   * Creates a osg sphere with an animation path 
   * and the belonging tree hierarchy
   *
   * Parameters:      const Sphere& iAnimatedSphere
   *                  existing osg sphere to copy it
   *
   *                  Vec3Array iPathCoordinates
   *                  array with the path coordinates
   *
   *                             Vec4 iColor
   *                  color for the new sphere
   */
    HfT_osg_Animation(const Sphere &iAnimatedSphere,
                      Vec3Array *iPathCoordinates, Vec4 iColor);

    /*
   * Constructor
   * Creates a osg sphere with an animation path 
   * and the belonging tree hierarchy
   *
   * Parameters:      const Sphere& iAnimatedSphere
   *                  existing osg sphere to copy it
   *
   *                  HfT_osg_Parametric_Surface& iPathObject
   *                  parametric surface to get the equator as path
   *
   *                             Vec4 iColor
   *                  color for the new sphere
   *
   *                             char iLine
   *                  char to differ between the different lines of a surface
   *                  'E':   for the equator
   *                  'U':   for the u directrix
   *                  'V':   for the v directrix
   */
    HfT_osg_Animation(const Sphere &iAnimatedSphere,
                      HfT_osg_Parametric_Surface &iPathObject,
                      Vec4 iColor, char iLine);

    /*
   * Destructor
   * Destructs all allocated objects,
   * except osg objects which are destructed by themselves.
   */
    ~HfT_osg_Animation();

    /****** getter and setter methods ******/
    /*
   * Sets the animation path of the osg sphere.
   * If a path is already set it is emptied and either
   * set with a standard path or the passed Vec3Array.
   *
   * Parameters:      Vec3Array* iPath
   *                  optional parameter for the animation path
   *
   * return:      void
   */
    void setAnimationPath(Vec3Array *iPath = NULL);

    /*
    * Sets the actual animation path in the reverse direction.
    * Reverses the control points.
    *
    * return:      void
    */
    void setReverseAnimationPath();

protected:
    /****** methods ******/
    /*
   * Creates the animation subtree. With a matrix transform, geode 
   * and a shapeDrawable.
   * Is called in each constructor.
   *
   * return:      void
   */
    void createAnimationSubtree();

private:
    /****** variables ******/
    /*
    * Variable which stores the whole time for an animation
    */
    double m_animationTime;

    /*
    * Variable which stores the time for an animation step
    */
    double m_animationTimeStep;

    /*
   * Ref pointer to the geode node, which is parent of the shape drawable
   * of the sphere.
   */
    ref_ptr<Geode> m_rpAnimGeode;

    /*
   * Ref pointer to the osg sphere, which is the animated object.
   */
    ref_ptr<Sphere> m_rpAnimatedSphere;

    /*
   * Ref pointer to the geometry, which stores the geometry of the
   * animation sphere.
   */
    ref_ptr<ShapeDrawable> m_rpAnimSphereDrawable;

    /*
   * Ref pointer to the animation path, which describes the path of the
   * animated sphere.
   */
    ref_ptr<AnimationPath> m_rpAnimationPath;

    /*
   * Ref pointer to the material of the sphere.
   */
    ref_ptr<Material> m_rpSphereMaterial;

    /*
   * Pointer to the vec4 which stores the color of the animation sphere
   */
    Vec4 *m_pColor;

    /*
   * Ref pointer to the state set of the surface drawable which sets
   * the visualization attributes.
   */
    ref_ptr<StateSet> m_rpSphereStateSet;

    /****** methods ******/
    /*
   * Creates the color and material of the sphere.
   *
   * return:      void
   */
    void createAnimationColor();
};

#endif /* HFT_OSG_ANIMATION_H_ */
