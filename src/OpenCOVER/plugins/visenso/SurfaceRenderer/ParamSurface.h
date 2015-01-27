/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2011 Visenso  **
 **                                                                        **
 ** Description: Surface class                                             **
 **                                                                        **
 ** header file                                                            **
 ** Author: A.Cyran                                                        **
 **                                                                        **
 ** History:                                                               **
 **     01.2011 initial version                                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef PARAMSURFACE_H_
#define PARAMSURFACE_H_

#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/PolygonMode>
#include <osg/Drawable>
#include <osg/Image>
#include <osg/Texture2D>
#include <osg/Material>
#include <osg/MatrixTransform>

#include <string>
#include <iostream>

#include <Python.h>
#include <PluginUtil/GenericGuiObject.h>
#include <cover/coVRPlugin.h>
#include <cover/OpenCOVER.h>

using namespace osg;
using namespace std;

using namespace covise;
using namespace opencover;

class ParamSurface
{
    //-------------------------------------------------------------------
    //public:
    //-------------------------------------------------------------------

public:
    //Constructor
    ParamSurface(int iPatU, int iPatV,
                 float iLowU, float iUpU,
                 float iLowV, float iUpV,
                 int iMode, string iX,
                 string iY, string iZ,
                 string iDerX, string iDerY,
                 string iDerZ);
    //Destructor
    virtual ~ParamSurface();

    //----------------------------------------------------------------
    //Variables
    //----------------------------------------------------------------

    ref_ptr<Geode> m_rpGeode;
    ref_ptr<MatrixTransform> m_rpTransformNode;
    ref_ptr<Texture2D> m_rpTexture;
    ref_ptr<Image> m_rpImage;
    ref_ptr<Vec2Array> m_rpTexCoords;
    std::string m_imagePath;
    PyObject *math_mod;
    PyObject *cosine;
    PyObject *sinus;
    PyObject *tangent;
    PyObject *logarithm;
    PyObject *loc;
    PyObject *glb;
    PyObject *code;
    PyObject *ret;
    double r;
    double py_paramU;
    double py_paramV;

    //-----------------------------------------------------------
    //Getter methods
    //-----------------------------------------------------------
    int PatchesU() const;
    int PatchesV() const;
    double LowerBoundU() const;
    double UpperBoundU() const;
    double LowerBoundV() const;
    double UpperBoundV() const;
    int Mode() const;
    Vec3Array *SupportingPoints() const;
    Vec3Array *Normals() const;
    string X() const;
    string Y() const;
    string Z() const;
    string DerX() const;
    string DerY() const;
    string DerZ() const;

    //--------------------------------------------------------------
    //Setter methods
    //--------------------------------------------------------------
    void setBoundries(const double &iLowU, const double &iUpU,
                      const double &iLowV, const double &iUpV);
    void setPatches(int iPatchesU, int iPatchesV);
    void setBoundriesAndPatches(const double &iLowU, const double &iUpU,
                                const double &iLowV, const double &iUpVconst,
                                int iPatchesU, int iPatchesV);
    void setPatchesAndMode(int iNumPatchesU, int iNumPatchesV,
                           int iNewMode);
    void setBoundriesAndMode(const double &iLowU, const double &iUpU,
                             const double &iLowV, const double &iUpV,
                             int iNewMode);
    void setBoundriesPatchesAndMode(const double &iLowU, const double &iUpU,
                                    const double &iLowV, const double &iUpV,
                                    int iNumPatchesU, int iNumPatchesV,
                                    int iNewMode);
    void setMode(int iNewMode);
    void setImage(string iImagePath);
    void setImage(int image_);

    //---------------------------------------------------------------------
    //Methods
    //--------------------------------------------------------------------
    void createSurface();

    //--------------------------------------------------------------------
    //protected:
    //--------------------------------------------------------------------

protected:
    //----------------------------------------------------------------------
    //Variables
    //---------------------------------------------------------------------
    double m_epsilon;
    int m_patchesU;
    int m_patchesV;
    int m_creatingMode;
    double m_lowerBoundU;
    double m_upperBoundU;
    double m_lowerBoundV;
    double m_upperBoundV;
    int _image;

    //-----------------------------------------------------------------------
    //Parametric representation
    //-------------------------------------------------------------------------
    string m_x; //x coordinate of the parametric representation
    string m_y; //y coordinate of the parametric representation
    string m_z; //z coordinate of the parametric representation
    string m_xDerived; //derived x coordinate
    string m_yDerived; //derived y coordinate
    string m_zDerived; //derived z coordinate

    bool m_isSet;
    bool m_areQuadsComputed;
    ref_ptr<Geometry> m_rpGeom;
    ref_ptr<StateSet> m_rpStateSet;
    ref_ptr<Vec3Array> m_rpSupportingPoints;
    ref_ptr<Vec3Array> m_rpNormals;
    ref_ptr<DrawElementsUInt> m_rpTriangEdges;
    ref_ptr<DrawElementsUInt> m_rpQuadEdges;
    ref_ptr<DrawArrays> m_rpPointCloud;
    ref_ptr<PolygonMode> m_rpPolyMode;

    ref_ptr<Material> m_rpMaterialSurface;
    Vec4 *m_pColorSurface;

    //Methods
    void setImageAndTexture();
    void digitalize();
    void setColorAndMaterial();
    void setSurface();

    /*
   * Recomputes the surface with new bounding parameters or/and
   * new patch parameters.
   * Recomputation of supporting points, normals and texture coordinates
   * for the new parameters.
   * Mode is the same and has not to be changed.
   *
   * parameters:      char iChar
   *                  'B': only boundries have changed
   *                       no recalculation of the edge list
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
	*                  parameter for the old visualization mode
	*                  needed for comparison with the new mode
	*
	*                  char iChar
	*                  'B': only boundries have changed
	*                       no recalculation of the edges
	*                  'P': patches (and boundries) have changed
	*                       recalculation of the edges
	*                  'M': only visualization mode has changed
	*
	* return:      void
	*/
    void recomputeSurface(int iOldMode, char iChar);

    void computeTriangleEdges();
    void computeQuadEdges();
    void createObjectInMode();

private:
    //Methods
    void initializeMembers();
    void createSubtree();

    //Python Berechnungsmethode
    double computeInPython(string iParametricRep);
};

#endif /* PARAMSURFACE_H_ */
