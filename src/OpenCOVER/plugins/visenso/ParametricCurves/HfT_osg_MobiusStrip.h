/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: Class to generate a mobius strip                          **
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

#ifndef HFT_OSG_MOBIUSSTRIP_H_
#define HFT_OSG_MOBIUSSTRIP_H_

#include "HfT_osg_Parametric_Surface.h"
#include <osg/CullFace>
#include <osg/ShapeDrawable>
#include <osg/Shape>

/*
 * The special feature of this class is the second geode, which enables
 * with the CullFace attribute, to set a different image at the frontside
 * and at the backside of the object.
 */
class HfT_osg_MobiusStrip : public HfT_osg_Parametric_Surface
{

public:
    /****** variables ******/
    /*
   * Ref pointer to the state set of the main geode, to
   * enable the setting of the cullface on or off.
   */
    ref_ptr<StateSet> m_rpStateSetGeode;

    /*
   * Ref pointer to the state set of the texture geode, to
   * enable the setting of the cullface on or off.
   */
    ref_ptr<StateSet> m_rpStateSetFrontFaceGeode;

    /****** constructors and destructors ******/
    /*
   * Standard constructor
   */
    HfT_osg_MobiusStrip();

    /*
   * Copy constructor
   */
    HfT_osg_MobiusStrip(const HfT_osg_MobiusStrip &iMobiusStrip);

    /*
   * Constructor
   * Creates a mobius strip. Calls a constructor of the superclass.
   *
   * Parameters:      double iRad
   *                  radian of the mobius strip
    */
    HfT_osg_MobiusStrip(double iRad);

    /*
   * Constructor
   * Creates a mobius strip. Calls a constructor of the superclass.
   *
   * Parameters:     int iMode
   *                 Sets the mode in which the mobius strip will be created
   *
   *                 double iLowU
   *                 Sets the lower bound for the u parameter
   *
   *                 double iUpU
   *                 Sets the upper bound for the u parameter
   *
   *                 double iLowV
   *                 Sets the lower bound for the v parameter
   *
   *                 double iUpV
   *                 Sets the upper bound for the v parameter
   */
    HfT_osg_MobiusStrip(int iMode, double iLowU,
                        double iUpU, double iLowV,
                        double iUpV);

    /*
   * Constructor
   * Creates a mobius strip. Calls a constructor of the superclass.
   *
   * Parameters:      double iRad
   *                  Sets the radian of the mobius strip
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
   *                  Sets the mode in which the mobius strip will be created
   */
    HfT_osg_MobiusStrip(double iRad, int iPatchesU,
                        int iPatchesV, double iLowU,
                        double iUpU, double iLowV,
                        double iUpV, int iMode);

    /*
   * Destructor
   * Destructs all allocated objects,
   * except osg objects which are destructed by themselves.
   * Like the auxiliar geometry.
   */
    virtual ~HfT_osg_MobiusStrip();

    /****** getter and setter methods ******/
    /*
   * Gets the value of the radian of the mobius strip.
   *
   * return:      double
   *              value of the radian
   */
    double Radian() const;

    /*
   * Sets the radian of a mobius strip and arranges a recomputation.
   *
   * Parameters:      const double& iRadian
   *                  reference to a constant radian value
   *
   * return:      void
   */
    void setRadian(const double &iRadian);

    /*
   * Sets the radian and boundries of a mobius strip
   * and arranges a recomputation.
   *
   * Parameters:      const double& iRadian
   *                  reference to a constant radian value
   *
   *                  const double& iLowU
   *                  reference to a value for the lower u parameter
   *
   *                  const double& iUpU
   *                  reference to a value for the upper u parameter
   *
   *                  const double& iLowV
   *                  reference to a value for the lower v parameter
   *
   *                  const double& iUpV
   *                  reference to a value for the upper v parameter
   *
   * return:      void
   */
    void setRadianAndBoundries(const double &iRadian, const double &iLowU,
                               const double &iUpU, const double &iLowV,
                               const double &iUpV);

    /*
   * Sets the radian, boundries and patches of a mobius strip
   * and arranges a recomputation.
   *
   * Parameters:      const double& iRadian
   *                  reference to a constant radian value
   *
   *                  const double& iLowU
   *                  reference to a value for the lower u parameter
   *
   *                  const double& iUpU
   *                  reference to a value for the upper u parameter
   *
   *                  const double& iLowV
   *                  reference to a value for the lower v parameter
   *
   *                  const double& iUpV
   *                  reference to a value for the upper v parameter
   *
   *                  int iPatchesU
   *                  patch value in u direction
   *
   *                  int iPatchesV
   *                  patch value in v direction
   *
   * return:      void
   */
    void setRadianBoundriesAndPatches(const double &iRadian, const double &iLowU,
                                      const double &iUpU, const double &iLowV,
                                      const double &iUpV, int iPatchesU,
                                      int iPatchesV);

    /*
   * Sets the radian and new mode of a mobius strip
   * and arranges a recomputation.
   *
   * Parameters:      const double& iRadian
   *                  reference to a constant radian value
   *
   *                  int iNewMode
   *                  number for the new mode
   *
   * return:      void
   */
    void setRadianAndMode(const double &iRadian, int iNewMode);

    /*
   * Sets the radian, boundries, patches and a new mode of a mobius strip
   * and arranges a recomputation.
   *
   * Parameters:      const double& iRadian
   *                  reference to a constant radian value
   *
   *                  const double& iLowU
   *                  reference to a value for the lower u parameter
   *
   *                  const double& iUpU
   *                  reference to a value for the upper u parameter
   *
   *                  const double& iLowV
   *                  reference to a value for the lower v parameter
   *
   *                  const double& iUpV
   *                  reference to a value for the upper v parameter
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
    void setRadianBoundriesPatchesAndMode(const double &iRadian, const double &iLowU,
                                          const double &iUpU, const double &iLowV,
                                          const double &iUpV, int iPatchesU,
                                          int iPatchesV, int iNewMode);

    /*
   * Sets the image path for the image of the mobius strip and attaches
   * the image to the texture variable.
   * The image path is read from the config file.
   *
   *
   * return:      void
   */
    void setImageAndTexture();

    /*
    * Sets the image, texture, cullFace and attributes for the
    * frontface geode.
    * The image path is read from the config file.
    *
    * return:      void
    */
    void setFrontFaceGeode();

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
    * equator of a mobius strip.
    *
    * return:      bool
    *              is true if equator is computed correctly
    */
    bool computeEquator();

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
   * origin point and radian.
   * Origin point is visualized as sphere and radian
   * is visualized as cylinder.
   *
   * return:      void
   */
    void createAuxiliarObjects();

private:
    /****** variables ******/
    /*
   * Variable which stores the value of the radian of the mobius strip
   */
    double m_radian;

    /*
   * Ref pointer to the frontface geode
   * which sets her cullface to "front"
   */
    ref_ptr<Geode> m_rpFrontFaceGeode;

    /*
   * Pointer to the CullFace of the main geode.
   * CullFace is set to "front".
   */
    ref_ptr<CullFace> m_rpCullFaceGeode;

    /*
   * Pointer to the CullFace of the frontFace geode.
   * CullFace is set to "front".
   */
    ref_ptr<CullFace> m_rpCullFaceFrontFaceGeode;

    /*
   * Variable which stores the image path of the second picture
   * which is shown at the frontside of the surface.
   */
    std::string m_frontFaceImagePath;

    /*
   * Pointer to the Texture2D which stores the second
   * image as her attribute
   */
    ref_ptr<Texture2D> m_rpFrontFaceTexture;

    /*
   * Ref pointer to the second image at the frontside
   */
    ref_ptr<Image> m_rpFrontFaceImage;

    /*
   * Pointer to a sphere object which visualizes the origin point
   */
    ref_ptr<Sphere> m_rpSphereOriginPoint;

    /*
   * Pointer to a cylinder object which visualizes the radian
   */
    ref_ptr<Cylinder> m_rpLineRadian;

    /*
   * Ref pointer to a drawable object which visualizes
   * the auxiliar geometry.
   */
    ref_ptr<ShapeDrawable> m_rpSphereOriginDrawable;

    /*
   * Ref pointer to a drawable object which visualizes
   * the auxiliar geometry.
   */
    ref_ptr<ShapeDrawable> m_rpLineRadianDrawable;
};

#endif /* HFT_OSG_MOBIUSSTRIP_H_ */
