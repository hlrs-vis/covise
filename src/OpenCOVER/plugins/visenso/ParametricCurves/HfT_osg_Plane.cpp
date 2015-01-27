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
 ** cpp file                                                               **
 ** Author: A.Cyran                                                        **
 **                                                                        **
 ** History:                                                               **
 **     12.2010 initial version                                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

/* Parametric representation of a plane
 *
 * x = a1 + s1*u + r1*v
 * x = a2 + s2*u + r2*v
 * x = a3 + s3*u + r3*v
 * u = [0.0, 1.0] ; v = [0.0, 1.0]
 */

#include "HfT_osg_Plane.h"
#include <stdexcept>
#include <config/CoviseConfig.h>
#include <osg/Matrix>
#include <osg/Vec3d>
#include <cover/coVRPluginSupport.h>
#include "cover/VRSceneGraph.h"

#include "cover/coTranslator.h"

using namespace osg;
using namespace std;
using namespace covise;
using namespace opencover;

//---------------------------------------------------
//Implements HfT_osg_Plane::HfT_osg_Plane()
//---------------------------------------------------
HfT_osg_Plane::HfT_osg_Plane()
    : HfT_osg_Parametric_Surface(0.0, 1.0, 0.0, 1.0)
{
    m_pOriginPoint = new Vec3d(0.0, 0.0, 0.0);
    m_pFirstDirectionVector = new Vec3d(0.0, 1.0, 0.0);
    m_pSecondDirectionVector = new Vec3d(0.0, 0.0, 1.0);
    //white and transparent material
    setSurface(Vec4(1.0, 1.0, 1.0, 1.0));
    m_rpStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    m_rpMaterialSurface->setTransparency(osg::Material::FRONT_AND_BACK, 0.8);
    m_isTransparent = true;
    m_sphereRadian = 0.05;
    m_arrowHeight = 0.3;
    this->createAuxiliarObjects();
}

//---------------------------------------------------
//Implements HfT_osg_Plane::HfT_osg_Plane(const HfT_osg_Plane& iPlane)
//---------------------------------------------------
HfT_osg_Plane::HfT_osg_Plane(const HfT_osg_Plane &iPlane)
    : HfT_osg_Parametric_Surface(iPlane.PatchesU(),
                                 iPlane.PatchesV(),
                                 iPlane.LowerBoundU(),
                                 iPlane.UpperBoundU(),
                                 iPlane.LowerBoundV(),
                                 iPlane.UpperBoundV(),
                                 iPlane.Mode())
{
    m_pOriginPoint = new Vec3d((*iPlane.OriginPoint()));
    m_pFirstDirectionVector = new Vec3d((*iPlane.FirstDirectionVector()));
    m_pSecondDirectionVector = new Vec3d((*iPlane.SecondDirectionVector()));
    setSurface(*iPlane.SurfaceColor());
    m_rpStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    m_rpMaterialSurface->setTransparency(osg::Material::FRONT, 0.8);
    m_isTransparent = true;
    m_sphereRadian = 0.05;
    m_arrowHeight = 0.3;
    this->createAuxiliarObjects();
}

//---------------------------------------------------
//Implements HfT_osg_Plane::HfT_osg_Plane( Vec3d iOrigin, Vec3d iFirst, Vec3d iSecond)
//---------------------------------------------------
HfT_osg_Plane::HfT_osg_Plane(Vec3d iOrigin, Vec3d iFirst, Vec3d iSecond)
    : HfT_osg_Parametric_Surface(0.0, 1.0, 0.0, 1.0)
{
    m_pOriginPoint = new Vec3d(iOrigin);

    if (iFirst.length() != 0.0)
    {
        m_pFirstDirectionVector = new Vec3d(iFirst);
    }
    else
    {
        throw out_of_range("First direction vector is zero vector. \n");
    }

    if ((iSecond.length() != 0.0))
    {
        m_pSecondDirectionVector = new Vec3d(iSecond);
    }
    else
    {
        throw out_of_range("Second direction vector is zero vectro. \n");
    }

    if (iFirst * iSecond != 0)
    {
        throw out_of_range("Both vectors are not rectangular. \n");
    }
    //white and transparent material
    setSurface(Vec4(1.0, 1.0, 1.0, 1.0));
    m_rpStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    m_rpMaterialSurface->setTransparency(osg::Material::FRONT_AND_BACK, 0.8);
    m_isTransparent = true;
    m_sphereRadian = 0.05;
    m_arrowHeight = 0.3;
    this->createAuxiliarObjects();
}

//---------------------------------------------------
//Implements HfT_osg_Plane::HfT_osg_Plane(int iMode, double iLowU,
//                                        double iUpU, double iLowV,
//                                        double iUpV)
//---------------------------------------------------
HfT_osg_Plane::HfT_osg_Plane(int iMode, double iLowU,
                             double iUpU, double iLowV,
                             double iUpV)
    : HfT_osg_Parametric_Surface(iMode, iLowU, iUpU, iLowV, iUpV)
{
    m_pOriginPoint = new Vec3d(0.0, 0.0, 0.0);
    m_pFirstDirectionVector = new Vec3d(0.0, 1.0, 0.0);
    m_pSecondDirectionVector = new Vec3d(0.0, 0.0, 1.0);
    //white and transparent material
    setSurface(Vec4(1.0, 1.0, 1.0, 1.0));
    m_rpStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    m_rpMaterialSurface->setTransparency(osg::Material::FRONT_AND_BACK, 0.8);
    m_isTransparent = true;
    m_sphereRadian = 0.05;
    m_arrowHeight = 0.3;
    this->createAuxiliarObjects();
}

//---------------------------------------------------
//Implements HfT_osg_Plane::HfT_osg_Plane(Vec3d iOrigin, Vec3d iFirst, Vec3d iSecond,
//                                        int iPatchesU, int iPatchesV,
//                                        double iLowU, double iUpU,
//                                        double iLowV, double iUpV, int iMode)
//---------------------------------------------------
HfT_osg_Plane::HfT_osg_Plane(Vec3d iOrigin, Vec3d iFirst, Vec3d iSecond,
                             int iPatchesU, int iPatchesV,
                             double iLowU, double iUpU,
                             double iLowV, double iUpV, int iMode)
    : HfT_osg_Parametric_Surface(iPatchesU, iPatchesV, iLowU,
                                 iUpU, iLowV, iUpV, iMode)
{
    m_pOriginPoint = new Vec3d(iOrigin);

    if (iFirst.length() != 0.0)
    {
        m_pFirstDirectionVector = new Vec3d(iFirst);
    }
    else
    {
        throw out_of_range("First direction vector is zero vector. \n");
    }

    if (iSecond.length() != 0.0)
    {
        m_pSecondDirectionVector = new Vec3d(iSecond);
    }
    else
    {
        throw out_of_range("Second direction vector is zero vectro. \n");
    }

    if (iFirst * iSecond != 0)
    {
        throw out_of_range("Both vectors are not rectangular. \n");
    }
    //white and transparent material
    setSurface(Vec4(1.0, 1.0, 1.0, 1.0));
    m_rpStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    m_rpMaterialSurface->setTransparency(osg::Material::FRONT_AND_BACK, 0.8);
    m_isTransparent = true;
    m_sphereRadian = 0.05;
    m_arrowHeight = 0.3;
    this->createAuxiliarObjects();
}

//---------------------------------------------------
//Implements HfT_osg_Plane::~HfT_osg_Plane()
//---------------------------------------------------
HfT_osg_Plane::~HfT_osg_Plane()
{
    m_sphereRadian = 0.0;
    m_arrowHeight = 0.0;

    if (m_pOriginPoint != NULL)
    {
        delete m_pOriginPoint;
    }

    if (m_pFirstDirectionVector != NULL)
    {
        delete m_pFirstDirectionVector;
    }

    if (m_pSecondDirectionVector != NULL)
    {
        delete m_pSecondDirectionVector;
    }
}

//---------------------------------------------------
//Implements HfT_osg_Plane::OriginPoint()
//---------------------------------------------------
Vec3d *HfT_osg_Plane::OriginPoint() const
{
    return m_pOriginPoint;
}

//---------------------------------------------------
//Implements HfT_osg_Plane::FirstDirectionVector()
//---------------------------------------------------
Vec3d *HfT_osg_Plane::FirstDirectionVector() const
{
    return m_pFirstDirectionVector;
}

//---------------------------------------------------
//Implements HfT_osg_Plane::SecondDirectionVector()
//---------------------------------------------------
Vec3d *HfT_osg_Plane::SecondDirectionVector() const
{
    return m_pSecondDirectionVector;
}

//---------------------------------------------------
//Implements HfT_osg_Plane::setOriginPoint(Vec3d& iOrigin)
//---------------------------------------------------
void HfT_osg_Plane::setOriginPoint(Vec3d &iOrigin)
{
    m_pOriginPoint->set(iOrigin);
    m_rpSphereOriginPoint->setCenter(*m_pOriginPoint);
    m_rpSphereOriginDrawable->dirtyDisplayList();
    recomputeSurface('B');
}

//---------------------------------------------------
//Implements HfT_osg_Plane::setFirstDirectionVector(Vec3d& iFirst)
//---------------------------------------------------
void HfT_osg_Plane::setFirstDirectionVector(Vec3d &iFirst)
{
    Matrix m;
    m.makeRotate(*m_pFirstDirectionVector, iFirst);
    m_rpTrafoFirst->setMatrix(m);
    if (iFirst.length() != 0.0)
    {
        m_pFirstDirectionVector->set(iFirst);
    }
    else
    {
        throw out_of_range("First direction vector is zero vector. \n");
    }

    if (iFirst * (*m_pSecondDirectionVector) != 0)
    {
        throw out_of_range("Both vectors are not rectangular. \n");
    }
    recomputeSurface('B');
}

//---------------------------------------------------
//Implements HfT_osg_Plane::setSecondDirectionVector(Vec3d& iSecond)
//---------------------------------------------------
void HfT_osg_Plane::setSecondDirectionVector(Vec3d &iSecond)
{
    Matrix m;
    m.makeRotate(*m_pSecondDirectionVector, iSecond);
    m_rpTrafoSecond->setMatrix(m);
    if (iSecond.length() != 0.0)
    {
        m_pSecondDirectionVector->set(iSecond);
    }
    else
    {
        throw out_of_range("Second direction vector is zero vector. \n");
    }

    if ((*m_pFirstDirectionVector) * iSecond != 0)
    {
        throw out_of_range("Both vectors are not rectangular. \n");
    }
    recomputeSurface('B');
}

//---------------------------------------------------
//Implements HfT_osg_Plane::setPlaneVectors(Vec3d& iOrigin, Vec3d& iFirst, Vec3d& iSecond)
//---------------------------------------------------
void HfT_osg_Plane::setPlaneVectors(Vec3d &iOrigin, Vec3d &iFirst, Vec3d &iSecond)
{
    m_pOriginPoint->set(iOrigin);
    m_rpSphereOriginPoint->setCenter(*m_pOriginPoint);
    m_rpSphereOriginDrawable->dirtyDisplayList();

    Matrix rotation;
    Matrix transl;
    rotation.makeRotate(*m_pFirstDirectionVector, iFirst);
    transl.makeTranslate(*m_pOriginPoint);
    m_rpTrafoFirst->setMatrix(rotation);
    m_rpTrafoFirst->setMatrix(transl);

    rotation.makeRotate(*m_pSecondDirectionVector, iSecond);
    m_rpTrafoSecond->setMatrix(rotation);
    //m_rpTrafoSecond -> setMatrix(transl);

    if (iFirst.length() != 0.0)
    {
        m_pFirstDirectionVector->set(iFirst);
    }
    else
    {
        throw out_of_range("First direction vector is zero vector. \n");
    }

    if (iSecond.length() != 0.0)
    {
        m_pSecondDirectionVector->set(iSecond);
    }
    else
    {
        throw out_of_range("Second direction vector is zero vector. \n");
    }

    if (iFirst * iSecond != 0)
    {
        throw out_of_range("Both vectors are not rectangular. \n");
    }
    recomputeSurface('B');
}

//---------------------------------------------------
//Implements HfT_osg_Plane::setPlaneVectorsAndPatches(Vec3d& iOrigin, Vec3d& iFirst,
//                                                    Vec3d& iSecond, int iPatchesU,
//                                                    int iPatchesV)
//---------------------------------------------------
void HfT_osg_Plane::setPlaneVectorsAndPatches(Vec3d &iOrigin, Vec3d &iFirst,
                                              Vec3d &iSecond, int iPatchesU,
                                              int iPatchesV)
{
    m_pOriginPoint->set(iOrigin);
    m_rpSphereOriginPoint->setCenter(*m_pOriginPoint);
    m_rpSphereOriginDrawable->dirtyDisplayList();

    Matrix m;
    m.makeRotate(*m_pFirstDirectionVector, iFirst);
    m_rpTrafoFirst->setMatrix(m);

    m.makeRotate(*m_pSecondDirectionVector, iSecond);
    m_rpTrafoSecond->setMatrix(m);

    if (iFirst.length() != 0.0)
    {
        m_pFirstDirectionVector->set(iFirst);
    }
    else
    {
        throw out_of_range("First direction vector is zero vector. \n");
    }

    if (iSecond.length() != 0.0)
    {
        m_pSecondDirectionVector->set(iSecond);
    }
    else
    {
        throw out_of_range("Second direction vector is zero vector. \n");
    }

    if (iFirst * iSecond != 0)
    {
        throw out_of_range("Both vectors are not rectangular. \n");
    }
    setPatches(iPatchesU, iPatchesV);
}

//---------------------------------------------------
//Implements HfT_osg_Plane::setPlaneVectorsAndPatches(Vec3d& iOrigin, Vec3d& iFirst,
//                                                    Vec3d& iSecond, int iPatchesU,
//                                                    int iPatchesV, int iNewMode)
//---------------------------------------------------
void HfT_osg_Plane::setPlaneVectorsPatchesAndMode(Vec3d &iOrigin, Vec3d &iFirst,
                                                  Vec3d &iSecond, int iPatchesU,
                                                  int iPatchesV, int iNewMode)
{
    m_pOriginPoint->set(iOrigin);
    m_rpSphereOriginPoint->setCenter(*m_pOriginPoint);
    m_rpSphereOriginDrawable->dirtyDisplayList();

    Matrix m;
    m.makeRotate(*m_pFirstDirectionVector, iFirst);
    m_rpTrafoFirst->setMatrix(m);

    m.makeRotate(*m_pSecondDirectionVector, iSecond);
    m_rpTrafoSecond->setMatrix(m);

    if (iFirst.length() != 0.0)
    {
        m_pFirstDirectionVector->set(iFirst);
    }
    else
    {
        throw out_of_range("First direction vector is zero vector. \n");
    }

    if (iSecond.length() != 0.0)
    {
        m_pSecondDirectionVector->set(iSecond);
    }
    else
    {
        throw out_of_range("Second direction vector is zero vector. \n");
    }

    if (iFirst * iSecond != 0)
    {
        throw out_of_range("Both vectors are not rectangular. \n");
    }
    setPatchesAndMode(iPatchesU, iPatchesV, iNewMode);
}

//---------------------------------------------------
//Implements HfT_osg_Plane::setImageAndTexture()
//---------------------------------------------------
void HfT_osg_Plane::setImageAndTexture()
{
    const std::string m_imagePath = (std::string)coCoviseConfig::getEntry("plane",
                                                                          "COVER.Plugin.ParametricCurves.Image", "/work/ac_te/rot.jpg");
    m_rpImage = osgDB::readImageFile(m_imagePath);
    m_rpTexture->setImage(m_rpImage.get());
}

//---------------------------------------------------
//Implements HfT_osg_Plane::setAuxiliarGeometrys()
//---------------------------------------------------
void HfT_osg_Plane::setAuxiliarGeometrys()
{
    m_rpGeode->addDrawable(m_rpSphereOriginDrawable);
    m_rpTrafoFirst->addChild(m_rpArrowFirstDir.get());
    m_rpTrafoNode->addChild(m_rpTrafoFirst.get());
    m_rpTrafoSecond->addChild(m_rpArrowSecondDir.get());
    m_rpTrafoNode->addChild(m_rpTrafoSecond.get());
}

//---------------------------------------------------
//Implements HfT_osg_Plane::setCallbacks(bool iIsVisible)
//---------------------------------------------------
void HfT_osg_Plane::setCallbacks(bool iIsVisible)
{
    if (iIsVisible)
    {
        m_rpSphereOriginDrawable->setDrawCallback(NULL);
        m_rpTrafoFirst->setNodeMask(0xffffffff);
        m_rpTrafoSecond->setNodeMask(0xffffffff);
        m_pLabelFirstArrow->show();
        m_pLabelSecondArrow->show();
    }
    else
    {
        m_rpSphereOriginDrawable->setDrawCallback(new Drawable::DrawCallback());
        m_rpTrafoFirst->setNodeMask(0x00000000);
        m_rpTrafoSecond->setNodeMask(0x00000000);
        m_pLabelFirstArrow->hide();
        m_pLabelSecondArrow->hide();
        m_rpSphereOriginDrawable->dirtyDisplayList();
    }
}

//---------------------------------------------------
//Implements HfT_osg_Plane::setLabels(bool iIsVisible)
//---------------------------------------------------
void HfT_osg_Plane::setLabels(bool iIsVisible)
{
    if (iIsVisible)
    {
        m_pLabelFirstArrow->show();
        m_pLabelSecondArrow->show();
    }
    else
    {
        m_pLabelFirstArrow->hide();
        m_pLabelSecondArrow->hide();
    }
}

//---------------------------------------------------
//Implements HfT_osg_Plane::digitalize()
//---------------------------------------------------
void HfT_osg_Plane::digitalize()
{
    double paramU = 0.0;
    double paramV = 0.0;
    int counterU = 0;
    int counterV = 0;
    //Length of the vector
    double length;

    // External loop with v, sweep direction from bottom to top
    paramV = m_lowerBoundV;

    while (paramV <= (m_upperBoundV + m_epsilon))
    {
        counterU = 0;
        paramU = m_lowerBoundU;

        // Internal loop with u, sweep direction to the right
        while (paramU <= (m_upperBoundU + m_epsilon))
        {
            // parametric representation of a plane
            // compute supporting points
            double x = m_pOriginPoint->x()
                       + paramU * m_pFirstDirectionVector->x()
                       + paramV * m_pSecondDirectionVector->x();
            double y = m_pOriginPoint->y()
                       + paramU * m_pFirstDirectionVector->y()
                       + paramV * m_pSecondDirectionVector->y();
            double z = m_pOriginPoint->z()
                       + paramU * m_pFirstDirectionVector->z()
                       + paramV * m_pSecondDirectionVector->z();
            m_rpSupportingPoints->push_back(Vec3(x, y, z));

            // Derivation with respect to u and derivation with respect to v
            // The cross product of both results in the normal vector
            // compute normal for each supporting point
            double xn = m_pFirstDirectionVector->y() * m_pSecondDirectionVector->z()
                        - m_pFirstDirectionVector->z() * m_pSecondDirectionVector->y();
            double yn = m_pFirstDirectionVector->z() * m_pSecondDirectionVector->x()
                        - m_pFirstDirectionVector->x() * m_pSecondDirectionVector->z();
            double zn = m_pFirstDirectionVector->x() * m_pSecondDirectionVector->y()
                        - m_pFirstDirectionVector->y() * m_pSecondDirectionVector->x();
            length = xn * xn + yn * yn + zn * zn;
            //Normalized normal vector
            m_rpNormals->push_back(Vec3(-xn / sqrt(length), -yn / sqrt(length), -zn / sqrt(length)));

            // Texture coordinates are in a interval from 0 to 1
            // compute texture coordinate for each supporting point
            double ut = double(counterU) / double(m_patchesU);
            double vt = double(counterV) / double(m_patchesV);
            m_rpTexCoords->push_back(Vec2(ut, vt));

            //fprintf(stderr,"vecArray:%f %f %i %i \t %f %f %f \n",
            //paramU,paramV,counterU,counterV,x,y,z);

            counterU++;
            paramU = m_lowerBoundU + ((m_upperBoundU - m_lowerBoundU) / m_patchesU) * counterU;
        }
        if ((counterV + 1) > m_patchesV)
        {
            break;
        }
        else
        {
            counterV++;
            paramV = m_lowerBoundV + ((m_upperBoundV - m_lowerBoundV) / m_patchesV) * counterV;
        }
    }
}

//---------------------------------------------------
//Implements HfT_osg_Plane::computeDirectrixU(int iSlideNumberU)
//---------------------------------------------------
void HfT_osg_Plane::computeDirectrixU(int iSlideNumberU)
{
    int numSuppPointsU = m_patchesU + 1;
    int position = numSuppPointsU * iSlideNumberU;

    for (int i = 0; i < numSuppPointsU; i++)
    {
        //Line is created between the following points
        m_rpDirectrixUEdges->push_back(i + position);
    }
}

//---------------------------------------------------
//Implements HfT_osg_Plane::computeDirectrixV(int iSlideNumberV)
//---------------------------------------------------
void HfT_osg_Plane::computeDirectrixV(int iSlideNumberV)
{
    int numSuppPointsU = m_patchesU + 1;
    int numSuppPointsV = m_patchesV + 1;

    for (int i = 0; i < numSuppPointsV; i++)
    {
        //Line is created between the following points
        m_rpDirectrixVEdges->push_back(i * numSuppPointsU + iSlideNumberV);
    }
}

//---------------------------------------------------
//Implements HfT_osg_Plane::computeEquator()
//---------------------------------------------------
bool HfT_osg_Plane::computeEquator()
{
    int numSuppPointsU = m_patchesU + 1;
    int numSuppPointsV = m_patchesV + 1;
    //computation of the position of the equator
    int position = (int)(floor((float)(numSuppPointsV / 2)) * numSuppPointsU);

    for (int i = 0; i < numSuppPointsU; i++)
    {
        m_rpEquatorEdges->push_back(position + i);
    }
    return true;
}

//---------------------------------------------------
//Implements HfT_osg_Plane::computeAngleFromAxisZToVec(Vec3* iVector)
//---------------------------------------------------
double HfT_osg_Plane::computeAngleFromAxisZToVec(Vec3d *iVector)
{
    double angle;
    angle = acos((iVector->z()) / iVector->length());

    //choose the smaller angle of both
    if (angle > PI)
    {
        angle = 2 * PI - angle;
    }
    return angle;
}

//---------------------------------------------------
//Implements HfT_osg_Plane::createAuxiliarObjects()
//---------------------------------------------------
void HfT_osg_Plane::createAuxiliarObjects()
{
    //Sphere for the origin point
    m_rpSphereOriginPoint = new Sphere(*m_pOriginPoint, m_sphereRadian);
    m_rpSphereOriginDrawable = new ShapeDrawable(m_rpSphereOriginPoint);
    //Color and material for the sphere
    m_rpMatSphere = new Material();
    m_pColorSphere = new Vec4(1.0, 1.0, 1.0, 1.0);
    m_rpSphereStateSet = m_rpSphereOriginDrawable->getOrCreateStateSet();
    m_rpSphereStateSet->setAttributeAndModes(m_rpMatSphere,
                                             osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    m_rpMatSphere->setColorMode(Material::OFF);
    m_rpMatSphere->setDiffuse(Material::FRONT_AND_BACK, *m_pColorSphere);
    m_rpMatSphere->setSpecular(Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    m_rpMatSphere->setAmbient(Material::FRONT_AND_BACK, Vec4((m_pColorSphere->x()) * 0.3,
                                                             (m_pColorSphere->y()) * 0.3, (m_pColorSphere->z()) * 0.3,
                                                             m_pColorSphere->w()));
    m_rpMatSphere->setShininess(Material::FRONT_AND_BACK, 100.0);

    m_pColorArrow = new Vec4(1.0, 1.0, 1.0, 1.0);
    //Arrow for the first direction
    m_rpArrowFirstDir = new coArrow(m_sphereRadian / 4.0, m_arrowHeight, false, false);
    m_rpArrowFirstDir->drawArrow(*m_pOriginPoint, m_sphereRadian / 4.0, m_arrowHeight);
    //Color and material for the first direction arrow
    m_rpMatFristArrow = new Material();
    setColorAndMaterial(*m_pColorArrow, *m_rpMatFristArrow);
    m_rpStateSetFirstArrow = m_rpArrowFirstDir->getOrCreateStateSet();
    m_rpStateSetFirstArrow->setAttributeAndModes(m_rpMatFristArrow, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

    //Arrow for the second direction
    m_rpArrowSecondDir = new coArrow(m_sphereRadian / 4.0, m_arrowHeight, false, false);
    m_rpArrowSecondDir->drawArrow(*m_pOriginPoint, m_sphereRadian / 4.0, m_arrowHeight);
    //Color and material for the second direction arrow
    m_rpMatSecondArrow = new Material();
    setColorAndMaterial(*m_pColorArrow, *m_rpMatSecondArrow);
    m_rpStateSetSecondArrow = m_rpArrowSecondDir->getOrCreateStateSet();
    m_rpStateSetSecondArrow->setAttributeAndModes(m_rpMatSecondArrow, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

    //matrix to rotate the arrows in the right direction
    Matrix m;
    m_rpTrafoFirst = new MatrixTransform();
    m.makeRotate(Vec3d(0.0, 0.0, 1.0), *m_pSecondDirectionVector);
    m_rpTrafoFirst->setMatrix(m);

    m_rpTrafoSecond = new MatrixTransform();
    m.makeRotate(Vec3d(0.0, 0.0, 1.0), *m_pFirstDirectionVector);
    m_rpTrafoSecond->setMatrix(m);

    m_pFirstLabelText = new std::string(coTranslator::coTranslate("Vektor r"));
    m_pSecondLabelText = new std::string(coTranslator::coTranslate("Vektor s"));
    //Label for the first arrow with "Vector r"
    m_pLabelFirstArrow = new coVRLabel(m_pFirstLabelText->c_str(), 0.06 * cover->getSceneSize(),
                                       0.02 * cover->getSceneSize(), Vec4(0.5451, 0.7020, 0.2431, 1.0), Vec4(0.0, 0.0, 0.0, 0.8));
    m_pLabelFirstArrow->setPosition(*m_pFirstDirectionVector * 0.1 + Vec3d(0.0, 0.0, 0.2));
    //m_pLabelFirstArrow -> setRotMode(coBillboard::POINT_ROT_EYE);
    //m_pLabelFirstArrow -> hide();

    //Label for the first arrow with "Vector s"
    m_pLabelSecondArrow = new coVRLabel(m_pSecondLabelText->c_str(), 0.06 * cover->getSceneSize(),
                                        0.02 * cover->getSceneSize(), Vec4(0.5451, 0.7020, 0.2431, 1.0), Vec4(0.0, 0.0, 0.0, 0.8));
    m_pLabelSecondArrow->setPosition(*m_pSecondDirectionVector * 0.1 + Vec3d(0.0, 0.0, 0.2));
    //m_pLabelSecondArrow -> setRotMode(coBillboard::POINT_ROT_EYE);
}

void HfT_osg_Plane::surfacePreframe()
{
    //position
    Matrixd o_to_w = cover->getBaseMat();
    Vec3 position_world;
    position_world = ((*m_pFirstDirectionVector * 0.2) + Vec3d(0.0, 0.0, 0.05)) * o_to_w;
    m_pLabelFirstArrow->setPosition(position_world);

    position_world = ((*m_pSecondDirectionVector * 0.2) + Vec3d(0.0, 0.0, 0.05)) * o_to_w;
    m_pLabelSecondArrow->setPosition(position_world);
}
