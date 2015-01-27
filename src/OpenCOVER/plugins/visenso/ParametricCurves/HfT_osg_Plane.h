/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: Class to generate a plane                                 **
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

#ifndef HFT_OSG_PLANE_H_
#define HFT_OSG_PLANE_H_

#include "HfT_osg_Parametric_Surface.h"
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Array>
#include <osg/ShapeDrawable>
#include <osg/Shape>
#include <PluginUtil/coArrow.h>
#include <osg/MatrixTransform>
#include "cover/coVRLabel.h"
#include <string>

using namespace osg;
using namespace opencover;

class HfT_osg_Plane : public HfT_osg_Parametric_Surface
{

public:
    /****** constructors and destructors ******/
    /*
   * Standard constructor
   */
    HfT_osg_Plane();

    /*
   * Copy constructor
   */
    HfT_osg_Plane(const HfT_osg_Plane &iPlane);

    /*
   * Constructor
   * Creates a transparent plane. Calls a constructor of the superclass.
   *
   * Parameters:      Vec3d iOrigin
   *                  origin point of the plane
   *
   *                  Vec3d iFirst
   *                  first direction vector of the plane
   *
   *                  Vec3d iSecond
   *                  second direction vector of the plane
   */
    HfT_osg_Plane(Vec3d iOrigin, Vec3d iFirst, Vec3d iSecond);

    /*
   * Constructor
   * Creates a transparent plane. Calls a constructor of the superclass.
   *
   * Parameters:      int iMode
   *                  Sets the mode in which the plane will be created
   *
   *                  double iLowU
   *                  Sets the lower bound for the u parameter
   *
   *                  double iUpU
   *                  Sets the upper bound for the u parameter
   *
   *                  double iLowV
   *                  Sets the lower bound for the v parameter
   *
   *                  double iUpV
   *                  Sets the upper bound for the v parameter
   */
    HfT_osg_Plane(int iMode, double iLowU,
                  double iUpU, double iLowV,
                  double iUpV);

    /*
   * Constructor
   * Creates a transparent plane. Calls a constructor of the superclass.
   *
   * Parameters:      Vec3d iOrigin
   *                  Sets the origin point of the plane
   *
   *                  Vec3d iFirst
   *                  Sets the first direction vector of the plane
   *
   *                  Vec3d iSecond
   *                  Sets the second direction vector of the plane
   *
   *                  int iPatchesU
   *                  Sets the number of patches in u direction
   *
   *                  int iPatchesV
   *                  Sets the number of patches in v direction
   *
   *                  double iLowU
   *                  Sets the lower bound for the u parameter
   *
   *                  double iUpU
   *                  Sets the upper bound for the u parameter
   *
   *                  double iLowV
   *                  Sets the lower bound for the v parameter
   *
   *                  double iUpV
   *                  Sets the upper bound for the v parameter
   *
   *                  int iMode
   *                  Sets the mode in which the plane will be created
   */
    HfT_osg_Plane(Vec3d iOrigin, Vec3d iFirst, Vec3d iSecond,
                  int iPatchesU, int iPatchesV,
                  double iLowU, double iUpU,
                  double iLowV, double iUpV,
                  int iMode);

    /*
   * Destructor
   * Destructs all allocated objects,
   * except osg objects which are destructed by themselves.
   * Like the auxiliar geometry.
   */
    virtual ~HfT_osg_Plane();

    /****** getter and setter methods ******/
    /*
   * Gets a pointer to the origin point of the plane.
   *
   * return:      Vec3d*
   *              constant pointer to the origin
   */
    Vec3d *OriginPoint() const;

    /*
   * Gets a pointer to the first direction vector of the plane.
   *
   * return:      Vec3d*
   *              constant pointer to the first direction
   */
    Vec3d *FirstDirectionVector() const;

    /*
   * Gets a pointer to the second direction vector of the plane.
   *
   * return:      Vec3d*
   *              constant pointer to the second direction
   */
    Vec3d *SecondDirectionVector() const;

    /*
   * Sets the origin point of a plane and arranges a recomputation.
   *
   * Parameters:      Vec3d& iOrigin
   *                  reference to a origin vector
   *
   * return:      void
   */
    void setOriginPoint(Vec3d &iOrigin);

    /*
   * Sets the first direction vector of a plane
   * and arranges a recomputation.
   *
   * Parameters:      Vec3d& iFirst
   *                  reference to a direction vector
   *
   * return:      void
   */
    void setFirstDirectionVector(Vec3d &iFirst);

    /*
   * Sets the second direction vector of a plane
   * and arranges a recomputation.
   *
   * Parameters:      Vec3d& iSecond
   *                  reference to a direction vector
   *
   * return:      void
   */
    void setSecondDirectionVector(Vec3d &iSecond);

    /*
   * Sets both direction vectors and the origin point of a plane
   * and arranges a recomputation.
   *
   * Parameters:      Vec3d& iOrigin
   *                  reference to a origin vector
   *
   *                  Vec3d& iFirst
   *                  reference to a direction vector
   *
   *                  Vec3d& iSecond
   *                  reference to a direction vector
   *
   * return:      void
   */
    void setPlaneVectors(Vec3d &iOrigin, Vec3d &iFirst, Vec3d &iSecond);

    /*
   * Sets both direction vectors, the origin point of a plane,
   * another patch values and arranges a recomputation.
   *
   * Parameters:      Vec3d& iOrigin
   *                  reference to a origin vector
   *
   *                  Vec3d& iFirst
   *                  reference to a direction vector
   *
   *                  Vec3d& iSecond
   *                  reference to a direction vector
   *
   *                  int iPatchesU
   *                  patch value in u direction
   *
   *                  int iPatchesV
   *                  patch value in v direction
   *
   * return:      void
   */
    void setPlaneVectorsAndPatches(Vec3d &iOrigin, Vec3d &iFirst, Vec3d &iSecond,
                                   int iPatchesU, int iPatchesV);

    /*
   * Sets both direction vectors, the origin point of a plane,
   * another patch values, a new visualisation mode
   * and arranges a recomputation.
   *
   * Parameters:      Vec3d& iOrigin
   *                  reference to a origin vector
   *
   *                  Vec3d& iFirst
   *                  reference to a direction vector
   *
   *                  Vec3d& iSecond
   *                  reference to a direction vector
   *
   *                  int iPatchesU
   *                  patch value in u direction
   *
   *                  int iPatchesV
   *                  patch value in v direction
   *
   *                  int iNewMode
   *                  number for the new mode
   *
   * return:      void
   */
    void setPlaneVectorsPatchesAndMode(Vec3d &iOrigin, Vec3d &iFirst, Vec3d &iSecond,
                                       int iPatchesU, int iPatchesV, int iNewMode);

    /*
   * Sets the image path for the image of the plane and attaches
   * the image to the texture variable.
   * The image path is read from the config file.
   *
   * return:      void
   */
    void setImageAndTexture();

    /*
   * Sets auxiliar geometrys as drawable to the geode.
   *
   * return:      void
   */
    void setAuxiliarGeometrys();

    /*
   * Sets auxiliar geometrys visible or invisible.
   *
   * Parameters:      bool iIsVisible
   *                  is true if the geometrys shall be visible
   *
   * return:      void
   */
    void setCallbacks(bool iIsVisible);

    /*
   * Sets labels vvisible or invisible.
   *
   * Parameters:      bool iIsVisible
   *                  is true if the labels shall be visible
   *
   * return:      void
   */
    void setLabels(bool iIsVisible);

    /****** methods ******/
    /*
   * Overwritten method from the superclass which computes the directrix in
   * u direction at a certain position.
   *
   * Parameters:      int iSlideNumberU
   *                  position of the directrix
   *
   * return:      void
   */
    void computeDirectrixU(int iSlideNumberU);

    /*
   * Overwritten method from the superclass which computes the directrix in
   * v direction at a certain position.
   *
   * Parameters:      int iSlideNumberV
   *                  position of the directrix
   *
   * return:      void
   */
    void computeDirectrixV(int iSlideNumberV);

    /*
   * Overwritten method from the superclass which computes the
   * equator of a plane.
   *
   * return:      bool
   *              is true if equator is computed correctly
   */
    bool computeEquator();

    /*
   * Compute angle between z axis and direction vector
   *
   * return:      double
   *              value of the computed angle
   */
    double computeAngleFromAxisZToVec(Vec3d *iVector);

    /*
    * Internal preframe method for a plane
    * Overwritten from the base class. Positions the labels.
    *
    * return:      void
    */
    virtual void surfacePreframe();

protected:
    /****** methods ******/
    /*
   * Overwritten method from the superclass which computes the
   * supporting points, normals and texture coordinates of the surface.
   *
   * return:      void
   */
    void digitalize();

    /*
   * Creates the visualisation objects for
   * origin point and direction vectors.
   * Origin point is visualized as sphere and 
   * each direction vector is visualized as smaller sphere.
   *
   * return:      void
   */
    void createAuxiliarObjects();

private:
    /****** variables ******/

    /*
   * Value for the sphere radian
   */
    double m_sphereRadian;

    /*
   * Value for the height of the arrow
   */
    double m_arrowHeight;

    /*
   * Pointer to a Vec3d which stores the origin point
   */
    Vec3d *m_pOriginPoint;

    /*
   * Pointer to a Vec3d which stores the first direction vector
   */
    Vec3d *m_pFirstDirectionVector;

    /*
   * Pointer to a Vec3d which stores the second direction vector
   */
    Vec3d *m_pSecondDirectionVector;

    /*
   * Ref pointer to a sphere object which visualizes the origin point
   */
    ref_ptr<Sphere> m_rpSphereOriginPoint;

    /*
   * Ref pointer to a ShapeDrawable object which
   * stores the origin point sphere
   */
    ref_ptr<ShapeDrawable> m_rpSphereOriginDrawable;

    /*
   * Ref pointer to a coArrow object which visualizes the
   * arrow of the first direction vector
   */
    ref_ptr<coArrow> m_rpArrowFirstDir;

    /*
   * Ref pointer to a coArrow object which visualizes the
   * arrow of the second direction vector
   */
    ref_ptr<coArrow> m_rpArrowSecondDir;

    /*
   * Ref pointer to a stateSet for the sphere
   */
    ref_ptr<StateSet> m_rpSphereStateSet;

    /*
   * Ref pointer to a stateSet for the first direction arrow
   */
    ref_ptr<StateSet> m_rpStateSetFirstArrow;

    /*
   * Ref pointer to a stateSet for the second direction arrow
   */
    ref_ptr<StateSet> m_rpStateSetSecondArrow;

    /*
   * Pointer to the material for the arrows
   */
    Vec4 *m_pColorSphere;

    /*
   * Pointer to the material for the arrows
   */
    Vec4 *m_pColorArrow;

    /*
   * Ref pointer to a material for the sphere
   */
    ref_ptr<Material> m_rpMatSphere;

    /*
   * Ref pointer to a material for the first direction vector
   */
    ref_ptr<Material> m_rpMatFristArrow;

    /*
   * Ref pointer to a material for the second direction vector
   */
    ref_ptr<Material> m_rpMatSecondArrow;

    /*
   * Ref pointer to a matrix transform object which positions
   * the first arrow.
   */
    ref_ptr<MatrixTransform> m_rpTrafoFirst;

    /*
   * Ref pointer to a matrix transform object which positions
   * the second arrow.
   */
    ref_ptr<MatrixTransform> m_rpTrafoSecond;

    /*
   * Pointer to a label for the first direction arrow;
   */
    coVRLabel *m_pLabelFirstArrow;

    /*
   * Pointer to a label for the second direction arrow;
   */
    coVRLabel *m_pLabelSecondArrow;

    /*
   * Pointer to the text for the label for the first direction arrow;
   */
    std::string *m_pFirstLabelText;

    /*
   * Pointer to the text for the label for the second direction arrow;
   */
    std::string *m_pSecondLabelText;
};

#endif /* HFT_OSG_PLANE_H_ */
