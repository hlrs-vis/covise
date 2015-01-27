/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: Abstract base class, parametric surface                    **
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

#ifndef HFT_OSG_PARAMETRIC_SURFACE_H_
#define HFT_OSG_PARAMETRIC_SURFACE_H_

#include <osg/MatrixTransform>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/PolygonMode>
#include <osg/Drawable>
#include <osgDB/ReadFile>
#include <osg/Image>
#include <osg/Texture2D>
#include <osg/Material>
#include <osg/LineWidth>
#include <string>

using namespace osg;

/*
 * Defines the skeletal structure for parametric surface objects.
 * Sets the tree structure of an object.
 */
class HfT_osg_Parametric_Surface
{

public:
    /****** variables ******/
    /*
   * Ref pointer to the matrix transform node, which is parent of the geode
   * of this surface.
   */
    ref_ptr<MatrixTransform> m_rpTrafoNode;

    /*
   * Ref pointer to the DrawElement which stores the edge-bonding for
   * the equator in reference to the *m_pSupportingPoints indices.
   * Needs to be public for the animation.
   */
    ref_ptr<DrawElementsUInt> m_rpEquatorEdges;

    /*
   * Ref pointer to the DrawElement which stores the edge-bonding for
   * the directrix in reference to the *m_pSupportingPoints indices.
   * Needs to be public for the slider menu.
   */
    ref_ptr<DrawElementsUInt> m_rpDirectrixUEdges;

    /*
   * Ref pointer to the DrawElement which stores the edge-bonding for
   * the directrix in reference to the *m_pSupportingPoints indices.
   * Needs to be public for the slider menu.
   */
    ref_ptr<DrawElementsUInt> m_rpDirectrixVEdges;

    /*
   * Ref pointer to the geometry which stores supporting points, edges and a
   * primitive set for the directrix in u direction.
   */
    ref_ptr<Geometry> m_rpDirectrixUGeom;

    /*
   * Ref pointer to the geometry which stores supporting points, edges and a
   * primitive set for the directrix in v direction.
   */
    ref_ptr<Geometry> m_rpDirectrixVGeom;

    /*
   * Variable which stores an image path.
   */
    std::string m_imagePath;

    /****** constructors and destructors ******/
    /*
   * Standard constructor
   */
    HfT_osg_Parametric_Surface();

    /*
   * Constructor
   * Creates a surface.
   *
   * Parameters:      int iMode
   *                  Sets the mode in which the surface will be created
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
    HfT_osg_Parametric_Surface(int iMode, double iLowU,
                               double iUpU, double iLowV,
                               double iUpV);

    /*
   * Constructor
   * Creates a surface.
   *
   * Parameters:      double iLowU
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
    HfT_osg_Parametric_Surface(double iLowU, double iUpU,
                               double iLowV, double iUpV);

    /*
   * Constructor
   * Creates a surface.
   *
   * Parameters:      int iPatU
   *                  Sets the number of patches in u direction
   *
   *                  int iPatV
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
   *                  Sets the mode in which the surface will be created
   */
    HfT_osg_Parametric_Surface(int iPatU, int iPatV,
                               double iLowU, double iUpU,
                               double iLowV, double iUpV,
                               int iMode);

    /*
   * Destructor
   * Destructs all allocated objects,
   * except osg objects which are destructed by themselves.
   */
    virtual ~HfT_osg_Parametric_Surface();

    /****** getter and setter methods ******/
    /*
   * Gets the number of patches in u direction.
   *
   * return:      int
   *              returns the value
   */
    int PatchesU() const;

    /*
   * Gets the number of patches in v direction.
   *
   * return:      int
   *              returns the value
   */
    int PatchesV() const;

    /*
   * Gets the lower bound of the u parameter.
   *
   * return:      double
   *              returns the value
   */
    double LowerBoundU() const;

    /*
   * Gets the upper bound of the u parameter.
   *
   * return:      double
   *              returns the value
   */
    double UpperBoundU() const;

    /*
   * Gets the lower bound of the v parameter.
   *
   * return:      double
   *              returns the value
   */
    double LowerBoundV() const;

    /*
   * Gets the upper bound of the v parameter.
   *
   * return:      double
   *              returns the value
   */
    double UpperBoundV() const;

    /*
   * Gets the mode in which the surface is displayed.
   *
   * return:      int
   *              returns the mode value
   */
    int Mode() const;

    /*
    * Gets the color of the surface.
    *
    * return:      Vec4*
    *              returns the pointer to the surface
    */
    Vec4 *SurfaceColor() const;

    /*
   * Gets the supporting points of the surface.
   *
   * return:      Vec3Array*
   *              returns the pointer to the Vec3Array
   *              of supporting points
   */
    Vec3Array *SupportingPoints() const;

    /*
    * Gets the normals of the surface for each supporting point.
    *
    * return:      Vec3Array*
    *              returns the pointer to the Vec3Array
    *              of normals
    */
    Vec3Array *Normals() const;

    /*
   * Sets the boundries of a surface and arranges a recomputation.
   *
   * Parameters:      const double& iLowU
   *                  value for the lower u bound
   *
   *                  const double& iUpU
   *                  value for the upper u bound
   *
   *                  const double& iLowV
   *                  value for the lower v bound
   *
   *                  const double& iUpV
   *                  value for the upper v bound
   *
   *                  return:      void
   */
    void setBoundries(const double &iLowU, const double &iUpU,
                      const double &iLowV, const double &iUpV);

    /*
   * Sets the patches of a surface and arranges a recomputation.
   *
   * Parameters:      int iPatchesU
   *                  number of patches in u direction
   *
   *                  int iPatchesV
   *                  number of patches in v direction
   *
   *                  return:      void
   */
    void setPatches(int iPatchesU, int iPatchesV);

    /*
   * Sets the boundries and the patches of a surface.
   * and arranges a recomputation.
   *
   * Parameters:      const double& iLowU
   *                  value for the lower u bound
   *
   *                  const double& iUpU
   *                  value for the upper u bound
   *
   *                  const double& iLowV
   *                  value for the lower v bound
   *
   *                  const double& iUpV
   *                  value for the upper v bound
   *
   *                  int iPatchesU
   *                  number of patches in u direction
   *
   *                  int iPatchesV
   *                  number of patches in v direction
   *
   *                  return:      void
   */
    void setBoundriesAndPatches(const double &iLowU, const double &iUpU,
                                const double &iLowV, const double &iUpVconst,
                                int iPatchesU, int iPatchesV);

    /*
   * Sets the patches of a surface
   * and a new visualisation mode and arranges a recomputation.
   *
   * Parameters:      int iNumPatchesU
   *                  number of patches in u direction
   *
   *                  int iNumPatchesV
   *                  number of patches in v direction
   *
   *                  int iNewMode
   *                  visualisation mode
   *
   *                  return:      void
   */
    void setPatchesAndMode(int iNumPatchesU, int iNumPatchesV,
                           int iNewMode);

    /*
   * Sets the boundries of a surface and arranges a recomputation.
   * Visualizes the surface in the new mode.
   *
   * Parameters:      const double& iLowU
   *                  value for the lower u bound
   *
   *                  const double& iUpU
   *                  value for the upper u bound
   *
   *                  const double& iLowV
   *                  value for the lower v bound
   *
   *                  const double& iUpV
   *                  value for the upper v bound
   *
   *                  int iNewMode
   *                  visualisation mode
   *
   *                  return:      void
   */
    void setBoundriesAndMode(const double &iLowU, const double &iUpU,
                             const double &iLowV, const double &iUpV,
                             int iNewMode);

    /*
   * Sets the boundries and patches of a surface and arranges a recomputation.
   * Visualizes the surface in the new mode.
   *
   * Parameters:      const double& iLowU
   *                  value for the lower u bound
   *
   *                  const double& iUpU
   *                  value for the upper u bound
   *
   *                  const double& iLowV
   *                  value for the lower v bound
   *
   *                  const double& iUpV
   *                  value for the upper v bound
   *
   *                  int iNumPatchesU
   *                  number of patches in u direction
   *
   *                  int iNumPatchesV
   *                  number of patches in v direction
   *
   *                  int iNewMode
   *                  visualisation mode
   *
   *                  return:      void
   */
    void setBoundriesPatchesAndMode(const double &iLowU, const double &iUpU,
                                    const double &iLowV, const double &iUpV,
                                    int iNumPatchesU, int iNumPatchesV,
                                    int iNewMode);

    /*
   * Sets the mode in which the surface will be displayed.
   *
   * Parameters:      int iNewMode
   *                  visualisation mode
   *
   *                  return:      void
   */
    void setMode(int iNewMode);

    /*
   * Sets the new image for the surface
   *
   * Parameters:      string iImagePath
   *                  string which stores the new image path
   *
   *                  return:      void
   */
    void setImage(std::string iImagePath);

    /****** methods ******/
    /*
   * Initializes the geometries with supporting points
   * and as necessary with normals and texture coordinates.
   * Calls the createObjectInMode method for visualisation.
   *
   * return:      void
   */
    void createSurface();

    /*
   * Computes the directrix in u or v direction
   * for a surface at a certain position.
   * The aim of this function is that the directrix snaps
   * to the shown directrices which are created by the wireframe
   * modus.
   * Important for the usage of the slider manu.
   *
   * parameters:      char iUV
   *                  u or v direction
   *
   *                  double iSliderValue
   *                  value of the correspondant slider
   *
   * return:      int
   *              returns the directrix number
   */
    int computeDirectrixNumber(char iUV, double iSliderValue);

    /*
   * Internal preframe method for each surface.
   * Can be overwritten. Is called in the plugin class of the renderer.
   *
   * return:      void
   */
    virtual void surfacePreframe();

    /*
   * Virtual abstract method to set the image and
   * the texture of the surface.
   * Implementation in the derived class.
   */
    virtual void setImageAndTexture() = 0;

    /*
   * Virtual abstract method to set certain geometrys
   * visible or invisible.
   * Implementation in the derived class.
   */
    virtual void setCallbacks(bool iIsVisible) = 0;

    /*
   * Virtual abstract method to set auxiliar geometrys
   * as drawable to the geode.
   * Implementation in the derived class.
   */
    virtual void setAuxiliarGeometrys() = 0;

    /*
   * Virtual abstract method to compute the supporting points,
   * normals and texture coordinates of the surface.
   * Implementation in the derived class.
   */
    virtual void digitalize() = 0;

    /*
   * Virtual abstract method to compute the directrix in
   * u direction at a certain position.
   * Implementation in the derived class.
   */
    virtual void computeDirectrixU(int iSlideNumberU) = 0;

    /*
   * Virtual abstract method to compute the directrix in
   * v direction at a certain position.
   * Implementation in the derived class.
   */
    virtual void computeDirectrixV(int iSlideNumberV) = 0;

    /*
   * Virtual abstract method to compute the equator.
    * Implementation in the derived class.
    */
    virtual bool computeEquator() = 0;

    //access only for the derived classes
protected:
    /****** variables ******/
    /*
   * Epsilon value for the comparison of two double values
   */
    double m_epsilon;

    /*
   * Stores the number of patches in u direction
   */
    int m_patchesU;

    /*
   * Stores the number of patches in v direction
   */
    int m_patchesV;

    /*
   * Stores the visualisation mode of the object
   */
    int m_creatingMode;

    /*
   * Stores the lower u bound
   */
    double m_lowerBoundU;

    /*
   * Stores the upper u bound
   */
    double m_upperBoundU;

    /*
   * Stores the lower v bound
   */
    double m_lowerBoundV;

    /*
   * Stores the upper v bound
   */
    double m_upperBoundV;

    /*
   * Is true if only the boundries are changed and the mode is still the same.
   * The settings of the primitive set and the geometry can be continued to use
   */
    bool m_isSet;

    /*
   * Is true if the edges for the quads are already computed and
   * can be continued to use for the changed mode.
   */
    bool m_areQuadsComputed;

    /*
   * Is true if the surface has a transparent surface.
   */
    bool m_isTransparent;

    /*
   * Is true if the directrix and equator geometries are already set for
   * the first time.
   */
    bool m_isFirstSet;

    /*
   * Ref pointer to the geode which stores in her drawables the geometry
   * of this surface.
   */
    ref_ptr<Geode> m_rpGeode;

    /*
   * Ref pointer to the geometry which stores supporting points, normals,
   * texture coordinates, edges, primitive sets for the main surface
   */
    ref_ptr<Geometry> m_rpGeom;

    /*
   * Ref pointer to the geometry which stores supporting points, edges and a
   * primitive set for the equator.
   */
    ref_ptr<Geometry> m_rpEquatorGeom;

    /*
   * Ref pointer to the state set which defines the visualisation of the
   * primitive sets.
   */
    ref_ptr<StateSet> m_rpStateSet;

    /*
   * Ref pointer to the state set which defines the visualisation of the
   * primitive set which stores the directrix edges.
   */
    ref_ptr<StateSet> m_rpStateSetDirectrixU;

    /*
   * Ref pointer to the state set which defines the visualisation of the
   * primitive set which stores the directrix edges.
   */
    ref_ptr<StateSet> m_rpStateSetDirectrixV;

    /*
   * Ref pointer to the state set which defines the visualisation of the
   * primitive set which stores the equator edges.
   */
    ref_ptr<StateSet> m_rpStateSetEquator;

    /*
   * Ref pointer to the Vec3Array which stores the supporting points of
   * the surface.
   */
    ref_ptr<Vec3Array> m_rpSupportingPoints;

    /*
   * Ref pointer to the Vec3Array which stores the normals of the surface.
   */
    ref_ptr<Vec3Array> m_rpNormals;

    /*
   * Ref pointer to the DrawElement which stores the edge-bonding for a
   * triangle mesh in reference to the *m_pSupportingPoints indices.
   */
    ref_ptr<DrawElementsUInt> m_rpTriangEdges;

    /*
   * Ref pointer to the DrawElement which stores the edge-bonding for a
   * quad mesh in reference to the *m_pSupportingPoints indices.
   */
    ref_ptr<DrawElementsUInt> m_rpQuadEdges;

    /*
   * Ref pointer to the DrawArrays which store the visualisation of the
   * VertexArray of the geometry. In this case it concerns the
   * *m_pSupportingPoints
   */
    ref_ptr<DrawArrays> m_rpPointCloud;

    /*
   * Ref pointer to the PolygonMode which stores the visualisation mode of the
   * primitive set.
   * It is an attribute of a state set.
   */
    ref_ptr<PolygonMode> m_rpPolyMode;

    /*
   * Ref pointer to the Texture2D which stores an image as her attribute
   */
    ref_ptr<Texture2D> m_rpTexture;

    /*
   * Ref pointer to an image
   */
    ref_ptr<Image> m_rpImage;

    /*
   * Ref pointer to the Vec2Aray which stores the texture coordinates
   * of the surface.
   */
    ref_ptr<Vec2Array> m_rpTexCoords;

    /*
   * Ref pointer to the Material for the whole surface
   */
    ref_ptr<Material> m_rpMaterialSurface;

    /*
   * Ref pointer to the Material for the directrix in u direction
   */
    ref_ptr<Material> m_rpMaterialDirectU;

    /*
   * Ref pointer to the Material for the directrix in v direction
   */
    ref_ptr<Material> m_rpMaterialDirectV;

    /*
   * Ref pointer to the Material for the equator
   */
    ref_ptr<Material> m_rpMaterialEquat;

    /*
   * Pointer to the vector which stores the color for the
   * directrix in u direction
   */
    Vec4 *m_pColorSurface;

    /*
   * Pointer to the vector which stores the color for the
   * directrix in u direction
   */
    Vec4 *m_pColorDirectU;

    /*
   * Pointer to the vector which stores the color for the
   * directrix in v direction
   */
    Vec4 *m_pColorDirectV;

    /*
   * Pointer to the vector which stores the color for the equator
   */
    Vec4 *m_pColorEquat;

    /*
   * Ref pointer to a variable which stores the line width
   */
    ref_ptr<LineWidth> m_rpLineWidth;

    /****** setter methods ******/
    /*
   * Sets the color and the material for a geometry or a geode
   *
   * Parameters:      Vec4 &iColor
   *                  vector which stores the color of the material
   *
   *                  Material &iMat
   *                  variable for the material
   *
   * return:      void
   */
    void setColorAndMaterial(Vec4 &iColor, Material &iMat);

    /*
   * Method to set the surface visualisation with texture
   * and material.
   * Calls the setColorAnd Material and the setImageAndTexture
   * functions.
   *
   * Parameters:      Vec4 *iColorArray
   *                  pointer to the color of the material
   *
   * return:      void
   */
    void setSurface(Vec4 iColorArray);

    /****** methods ******/
    /*
   * Recomputes the surface with new bounding parameters or/and
   * new patch parameters.
   * Recomputation of supporting points, normals and texture coordinates
   * for the new parameters.
   * Mode is the same and has not to be changed.
   *
   * parameters:      char iChar
   *                  'B': only boundries have changed
   *                       no recalcluation of the edge list
   *                  'P': patches (and boundries) have changed
   *                       recalculation of the edge list
   *
   * return:      void
   */
    void recomputeSurface(char iChar);

    /*
   * Recomputes the surface with new bounding parameters or/and
   * new patch parameters and a new visualisation mode.
   * Recomputation of supporting points, normals and texture coordinates
   * for the new parameters.
   * Mode is the same and has not to be changed.
   * Calls the recomputeSurface(char iChar) method with its parameter.
   *
   * parameters:      int iOldMode
   *                  parameter for the old visualisation mode
   *                  needed for comparison with the new mode
   *
   *                  char iChar
   *                  'B': only boundries have changed
   *                       no recalcluation of the edges
   *                  'P': patches (and boundries) have changed
   *                       recalculation of the edges
   *                  'M': only visualisation mode has changed
   *
   * return:      void
   */
    void recomputeSurface(int iOldMode, char iChar);

    /*
   * Computation of the edges which are created by a triangule mesh.
   * They are stored in a DrawElementsUInt object,
   * that only stores the order of points which create the edges.
   *
   * return:      void
   */
    void computeTriangleEdges();

    /*
   * Computation of the edges which are created by a quad mesh.
   * They are stored in a DrawElementsUInt object,
   * that only stores the order of points which create the edges.
   *
   * return:      void
   */
    void computeQuadEdges();

    /*
   * Creates an object in the mode,
   * which is given by the m_creatingMode variable.
   *
   * return:      void
   */
    void createObjectInMode();

    /*
   * Creates the visualisation objects for e.g.
   * origin point, direction vectors, radian etc.
   * Implementation in the base class.
   */
    virtual void createAuxiliarObjects() = 0;

    //No access, also for derived classes
private:
    /****** setter methods ******/
    /*
   * Sets attributes like PolygonMode, Material and LineWidth for
   * the state sets.
   *
   * Parameters:      StateSet *ioStateSet
   *                  pointer to the state set which will be set
   *
   *                  Material *ioMaterial
   *                  pointer to the material which will be
   *                  attached to this state set
   *
   * return:      void
   */
    void setStateSetAttributes(StateSet *ioStateSet,
                               Material *ioMaterial);

    /****** methods ******/
    /*
   * Initializes all member variables.
   * Is called in each constructor.
   *
   * return:      void
   */
    void initializeMembers();

    /*
   * Creates the tree hierarchy.
   * Adds a geode as child of an transform node.
   * Adds a geometry as child of an geode.
   *
   * return:      void
   */
    void createSubtree();
};

#endif /* HFT_OSG_PARAMETRIC_SURFACE_H_ */
