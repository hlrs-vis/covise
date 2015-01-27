/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2011 Visenso  **
 **                                                                        **
 ** Description: Surface class                                             **
 **                                                                        **
 ** cpp file                                                               **
 ** Author: A.Cyran                                                        **
 **                                                                        **
 ** History:                                                               **
 **     01.2011 initial version                                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "ParamSurface.h"
#include <cover/RenderObject.h>
#include <cover/coVRPluginSupport.h>
#include <osgDB/ReadFile>
#include <config/CoviseConfig.h>
#include <stdexcept>

using namespace std;
using namespace osg;

//---------------------------------------------------
//Implements ParamSurface::ParamSurface(int iPatU, double iPatV,
//                                                                float iLowU, float iUpU,
//                                                                float iLowV, float iUpV,
//                                                                int iMode, string iX,
//                                                                string iY, string iZ,
//                                                                string iDerX, string iDerY,
//                                                                string iDerZ)
//---------------------------------------------------
ParamSurface::ParamSurface(int iPatU, int iPatV,
                           float iLowU, float iUpU,
                           float iLowV, float iUpV,
                           int iMode, string iX,
                           string iY, string iZ,
                           string iDerX, string iDerY,
                           string iDerZ)
{
    if ((iPatU > 1) && (!isNaN((float)iPatU)))
    {
        m_patchesU = iPatU;
    }
    else
    {
        throw out_of_range("Too many patches. Choose a number greater than 1. \n");
    }

    if ((iPatV > 1) && (!isNaN((float)iPatV)))
    {
        m_patchesV = iPatV;
    }
    else
    {
        throw out_of_range("Too many patches. Choose a number greater than 1. \n");
    }

    if ((iMode >= 0) && (iMode <= 5) && (!isNaN((float)iMode)))
    {
        m_creatingMode = iMode;
    }
    else
    {
        throw out_of_range("Undefined mode. Choose a mode between 0 and 5. \n");
    }

    if ((iLowU < iUpU) && (!isNaN(iLowU)) && (!isNaN(iUpU)))
    {
        m_lowerBoundU = iLowU;
        m_upperBoundU = iUpU;
    }
    else
    {
        throw out_of_range("Lower bound U greater than or equal to upper bound U or bound is not a number.\n");
    }

    if ((iLowV < iUpV) && (!isNaN(iLowV)) && (!isNaN(iUpV)))
    {
        m_lowerBoundV = iLowV;
        m_upperBoundV = iUpV;
    }
    else
    {
        throw out_of_range("Lower bound V greater than or equal to upper bound V or bound is not a number.\n");
    }
    m_x = iX;
    m_y = iY;
    m_z = iZ;
    m_xDerived = iDerX;
    m_yDerived = iDerY;
    m_zDerived = iDerZ;

    initializeMembers();
    setSurface();
    math_mod = PyImport_ImportModule("math");
    cosine = PyObject_GetAttrString(math_mod, "cos");
    sinus = PyObject_GetAttrString(math_mod, "sin");
    tangent = PyObject_GetAttrString(math_mod, "tan");
    logarithm = PyObject_GetAttrString(math_mod, "log");
}

//---------------------------------------------------
//Implements ParamSurface::~ParamSurface()
//---------------------------------------------------
ParamSurface::~ParamSurface()
{
    m_patchesU = 0;
    m_patchesV = 0;
    m_creatingMode = 0;
    m_lowerBoundU = 0.0;
    m_upperBoundU = 0.0;
    m_lowerBoundV = 0.0;
    m_upperBoundV = 0.0;
    m_isSet = false;
    m_areQuadsComputed = false;
    m_x.clear();
    m_y.clear();
    m_z.clear();
    m_xDerived.clear();
    m_yDerived.clear();
    m_zDerived.clear();

    if (m_pColorSurface != NULL)
    {
        //Cleanup the pointer to the surface color
        //Calls destructor of the vec4
        delete m_pColorSurface;
    }

    //Delete from subtree
    m_rpGeode->getParent(0)->removeChild(m_rpGeode.get());
    Py_DECREF(code);
    Py_DECREF(ret);
    Py_Finalize(); //Python Finalization
}

//---------------------------------------------------
//Implements ParamSurface::initializeMembers()
//---------------------------------------------------
void ParamSurface::initializeMembers()
{
    m_epsilon = 0.000001;
    m_isSet = false;
    m_areQuadsComputed = false;
    m_rpGeode = new Geode();
    m_rpTransformNode = new MatrixTransform();

    m_rpGeom = new Geometry();
    m_rpStateSet = m_rpGeom->getOrCreateStateSet();
    m_rpSupportingPoints = new Vec3Array;
    m_rpNormals = new Vec3Array;
    m_rpTriangEdges = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
    m_rpQuadEdges = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
    m_rpPointCloud = new DrawArrays(PrimitiveSet::POINTS);
    m_rpPolyMode = new PolygonMode(PolygonMode::FRONT_AND_BACK, PolygonMode::LINE);
    m_rpTexture = new Texture2D();
    m_rpTexCoords = new Vec2Array;
    //Matrials for the colors of the objects
    m_rpMaterialSurface = new Material();
    //white color
    m_pColorSurface = new Vec4(1.0, 1.0, 1.0, 1.0);
}

//---------------------------------------------------
//Implements ParamSurface::PatchesU()
//---------------------------------------------------
int ParamSurface::PatchesU() const
{
    return m_patchesU;
}

//---------------------------------------------------
//Implements ParamSurface::PatchesV()
//---------------------------------------------------
int ParamSurface::PatchesV() const
{
    return m_patchesV;
}

//---------------------------------------------------
//Implements ParamSurface::LowerBoundU()
//---------------------------------------------------
double ParamSurface::LowerBoundU() const
{
    return m_lowerBoundU;
}

//---------------------------------------------------
//Implements ParamSurface::UpperBoundU()
//---------------------------------------------------
double ParamSurface::UpperBoundU() const
{
    return m_upperBoundU;
}

//---------------------------------------------------
//Implements ParamSurface::LowerBoundV()
//---------------------------------------------------
double ParamSurface::LowerBoundV() const
{
    return m_lowerBoundV;
}

//---------------------------------------------------
//Implements ParamSurface::UpperBoundV()
//---------------------------------------------------
double ParamSurface::UpperBoundV() const
{
    return m_upperBoundV;
}

//---------------------------------------------------
//Implements ParamSurface::Mode()
//---------------------------------------------------
int ParamSurface::Mode() const
{
    return m_creatingMode;
}

//---------------------------------------------------
//Implements ParamSurface::SupportingPoints()
//---------------------------------------------------
Vec3Array *ParamSurface::SupportingPoints() const
{
    return m_rpSupportingPoints;
}

//---------------------------------------------------
//Implements ParamSurface::Normals()
//---------------------------------------------------
Vec3Array *ParamSurface::Normals() const
{
    return m_rpNormals;
}

//---------------------------------------------------
//Implements ParamSurface::X()
//---------------------------------------------------
string ParamSurface::X() const
{
    return m_x;
}

//---------------------------------------------------
//Implements ParamSurface::Y()
//---------------------------------------------------
string ParamSurface::Y() const
{
    return m_y;
}

//---------------------------------------------------
//Implements ParamSurface::Z()
//---------------------------------------------------
string ParamSurface::Z() const
{
    return m_z;
}

//---------------------------------------------------
//Implements ParamSurface::DerX()
//---------------------------------------------------
string ParamSurface::DerX() const
{
    return m_xDerived;
}

//---------------------------------------------------
//Implements ParamSurface::DerY()
//---------------------------------------------------
string ParamSurface::DerY() const
{
    return m_yDerived;
}

//---------------------------------------------------
//Implements ParamSurface::DerZ()
//---------------------------------------------------
string ParamSurface::DerZ() const
{
    return m_zDerived;
}

//---------------------------------------------------
//Implements ParamSurface::setBoundries(const double& iLowU, const double& iUpU,
//                                                   const double& iLowV, const double& iUpV)
//---------------------------------------------------
void ParamSurface::setBoundries(const double &iLowU, const double &iUpU,
                                const double &iLowV, const double &iUpV)
{
    if ((iLowU < iUpU) && (!isNaN(iLowU)) && (!isNaN(iUpU)))
    {
        m_lowerBoundU = iLowU;
        m_upperBoundU = iUpU;
    }
    else
    {
        throw out_of_range("Lower bound U greater than or equal to upper bound U or bound is not a number.\n");
    }
    if ((iLowV < iUpV) && (!isNaN(iLowV)) && (!isNaN(iUpV)))
    {
        m_lowerBoundV = iLowV;
        m_upperBoundV = iUpV;
    }
    else
    {
        throw out_of_range("Lower bound V greater than or equal to upper bound V or bound is not a number.\n");
    }
    //recalculation of the surface with its new boundries
    recomputeSurface('B');
}

//---------------------------------------------------
//Implements ParamSurface::setPatches(int iPatchesU, int iPatchesV)
//---------------------------------------------------
void ParamSurface::setPatches(int iPatchesU, int iPatchesV)
{
    if ((iPatchesU > 1) && (!isNaN((float)iPatchesU)))
    {
        m_patchesU = iPatchesU;
    }
    else
    {
        throw out_of_range("Too many patches. Choose a number greater than 1. \n");
    }

    if ((iPatchesV > 1) && (!isNaN((float)iPatchesV)))
    {
        m_patchesV = iPatchesV;
    }
    else
    {
        throw out_of_range("Too many patches. Choose a number greater than 1. \n");
    }
    //recalculation of the surface with its new patch values
    recomputeSurface('P');
}

//---------------------------------------------------
//Implements ParamSurface::setBoundriesAndPatches(const double& iLowU, const double& iUpU,
//                                                             const double& iLowV, const double& iUpV,
//                                                             int iPatchesU, int iPatchesV)
//---------------------------------------------------
void ParamSurface::setBoundriesAndPatches(const double &iLowU, const double &iUpU,
                                          const double &iLowV, const double &iUpV,
                                          int iPatchesU, int iPatchesV)
{
    if ((iLowU < iUpU) && (!isNaN(iLowU)) && (!isNaN(iUpU)))
    {
        m_lowerBoundU = iLowU;
        m_upperBoundU = iUpU;
    }
    else
    {
        throw out_of_range("Lower bound U greater than or equal to upper bound U or bound is not a number.\n");
    }
    if ((iLowV < iUpV) && (!isNaN(iLowV)) && (!isNaN(iUpV)))
    {
        m_lowerBoundV = iLowV;
        m_upperBoundV = iUpV;
    }
    else
    {
        throw out_of_range("Lower bound V greater than or equal to upper bound V or bound is not a number.\n");
    }
    if ((iPatchesU > 1) && (!isNaN((float)iPatchesU)))
    {
        m_patchesU = iPatchesU;
    }
    else
    {
        throw out_of_range("Too many patches. Choose a number greater than 1. \n");
    }
    if ((iPatchesV > 1) && (!isNaN((float)iPatchesV)))
    {
        m_patchesV = iPatchesV;
    }
    else
    {
        throw out_of_range("Too many patches. Choose a number greater than 1. \n");
    }
    //recalculation of the surface with its new boundries and patch values
    recomputeSurface('P');
}
//---------------------------------------------------
//Implements ParamSurface::setPatchesAndMode(int iPatchesU, int iPatchesV,
//                                                        int iNewMode)
//---------------------------------------------------
void ParamSurface::setPatchesAndMode(int iPatchesU, int iPatchesV,
                                     int iNewMode)
{
    if ((iPatchesU > 1) && (!isNaN((float)iPatchesU)))
    {
        this->m_patchesU = iPatchesU;
    }
    else
    {
        throw out_of_range("Too many U patches. Choose a number greater than 1. \n");
    }
    if ((iPatchesV > 1) && (!isNaN((float)iPatchesV)))
    {
        this->m_patchesV = iPatchesV;
    }
    else
    {
        throw out_of_range("Too many V patches. Choose a number greater than 1. \n");
    }
    //Storage of the old mode for the later comparison
    int oldMode = m_creatingMode;
    if ((iNewMode >= 0) && (iNewMode <= 3) && (!isNaN((float)iNewMode)))
    {
        m_creatingMode = iNewMode;
    }
    else
    {
        throw out_of_range("Undefined mode. Choose a mode between 0 and 3. ");
    }
    //recalculation of the surface with its new patch values and its new mode
    recomputeSurface(oldMode, 'P');
}

//---------------------------------------------------
//Implements ParamSurface::setBoundriesAndMode(const double& iLowU, const double& iUpU,
//                                                          const double& iLowV, const double& iUpV,
//                                                          int iNewMode)
//---------------------------------------------------
void ParamSurface::setBoundriesAndMode(const double &iLowU, const double &iUpU,
                                       const double &iLowV, const double &iUpV,
                                       int iNewMode)
{
    if ((iLowU < iUpU) && (!isNaN(iLowU)) && (!isNaN(iUpU)))
    {
        m_lowerBoundU = iLowU;
        m_upperBoundU = iUpU;
    }
    else
    {
        throw out_of_range("Lower bound U greater than or equal to upper bound U or bound is not a number.\n");
    }
    if ((iLowV < iUpV) && (!isNaN(iLowV)) && (!isNaN(iUpV)))
    {
        m_lowerBoundV = iLowV;
        m_upperBoundV = iUpV;
    }
    else
    {
        throw out_of_range("Lower bound V greater than or equal to upper bound V or bound is not a number.\n");
    }
    int oldMode = m_creatingMode;
    if ((iNewMode >= 0) && (iNewMode <= 3) && (!isNaN((float)iNewMode)))
    {
        m_creatingMode = iNewMode;
    }
    else
    {
        throw out_of_range("Undefined mode. Choose a mode between 0 and 3. ");
    }
    //Patches are the same so the quads have not to be recomputed.
    //Setting the m_areQuadsComputed variable true if the new mode also
    //needs the quad mesh.
    if (oldMode == 2 | oldMode == 3)
    {
        if (m_creatingMode == 2 | m_creatingMode == 3)
        {
            m_areQuadsComputed = true;
        }
    }
    recomputeSurface(oldMode, 'B');
    m_areQuadsComputed = false;
}

//---------------------------------------------------
//Implements ParamSurface::setBoundriesPatchesAndMode(const double& iLowU, const double& iUpU,
//                                                                const double& iLowV, const double& iUpV,
//                                                                int iPatchesU, int iPatchesV,
//                                                                int iNewMode)
//---------------------------------------------------
void ParamSurface::setBoundriesPatchesAndMode(const double &iLowU, const double &iUpU,
                                              const double &iLowV, const double &iUpV,
                                              int iPatchesU, int iPatchesV,
                                              int iNewMode)
{
    if ((iLowU < iUpU) && (!isNaN(iLowU)) && (!isNaN(iUpU)))
    {
        m_lowerBoundU = iLowU;
        m_upperBoundU = iUpU;
    }
    else
    {
        throw out_of_range("Lower bound U greater than or equal to upper bound U or bound is not a number.\n");
    }
    if ((iLowV < iUpV) && (!isNaN(iLowV)) && (!isNaN(iUpV)))
    {
        m_lowerBoundV = iLowV;
        m_upperBoundV = iUpV;
    }
    else
    {
        throw out_of_range("Lower bound V greater than or equal to upper bound V or bound is not a number.\n");
    }
    if ((iPatchesU > 1) && (!isNaN((float)iPatchesU)))
    {
        m_patchesU = iPatchesU;
    }
    else
    {
        throw out_of_range("Too many U patches. Choose a number greater than 1. \n");
    }
    if ((iPatchesV > 1) && (!isNaN((float)iPatchesV)))
    {
        m_patchesV = iPatchesV;
    }
    else
    {
        throw out_of_range("Too many V patches. Choose a number greater than 1. \n");
    }
    int oldMode = m_creatingMode;
    if ((iNewMode >= 0) && (iNewMode <= 3) && (!isNaN((float)iNewMode)))
    {
        m_creatingMode = iNewMode;
    }
    else
    {
        throw out_of_range("Undefined mode. Choose a mode between 0 and 3. ");
    }

    //Recalculation of the surface because all parameters have been changed
    recomputeSurface(oldMode, 'P');
}

//---------------------------------------------------
//Implements ParamSurface::setMode( int iNewMode)
//---------------------------------------------------
void ParamSurface::setMode(int iNewMode)
{
    int oldMode = m_creatingMode;

    if ((iNewMode >= 0) && (iNewMode <= 3) && (!isNaN((float)iNewMode)))
    {
        m_creatingMode = iNewMode;
    }
    else
    {
        throw out_of_range("Undefined mode. Choose a mode between 0 and 3. \n");
    }

    //Patches are the same so the quads have not to be recomputed.
    //Setting the m_areQuadsComputed variable true if the new mode also
    //needs the quad mesh.
    if (oldMode == 2 | oldMode == 3)
    {
        if (m_creatingMode == 2 | m_creatingMode == 3)
        {
            m_areQuadsComputed = true;
        }
    }
    recomputeSurface(oldMode, 'M');
    m_areQuadsComputed = false;
}

//---------------------------------------------------
//Implements ParamSurface::setImage(std::string iImagePath)
//---------------------------------------------------
void ParamSurface::setImage(std::string iImagePath)
{
    m_imagePath = iImagePath;
    m_rpImage = osgDB::readImageFile(m_imagePath);
    m_rpTexture->setImage(m_rpImage.get());
}

//---------------------------------------------------
//Implements ParamSurface::recomputeSurface(char iChar)
//---------------------------------------------------
void ParamSurface::recomputeSurface(char iChar)
{
    switch (iChar)
    {
    //Only boundries have been changed
    case 'B':
        m_rpSupportingPoints->clear();
        m_rpNormals->clear();
        m_rpTexCoords->clear();
        this->digitalize();
        break;
    //Boundries and patches or only patches have been changed
    case 'P':
        m_rpSupportingPoints->clear();
        m_rpNormals->clear();
        m_rpTexCoords->clear();
        this->digitalize();

        //mode is the same but edges have to be recalculated in a new DrawElementsUInt
        //and the primitive set has to be reseted
        switch (m_creatingMode)
        {
        //Point cloud
        case 0:
            m_rpGeom->removePrimitiveSet(0);
            break;

        //Triangle mesh
        case 1:
            m_rpGeom->removePrimitiveSet(0);
            m_rpTriangEdges = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
            break;

        //Quad mesh
        case 2:

        //Quad mesh with a texture on it
        case 3:
            m_rpGeom->removePrimitiveSet(0);
            m_rpQuadEdges = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
            break;

            m_isSet = true;
            createObjectInMode();
            m_isSet = false;
        }
        break;
    }
}

//---------------------------------------------------
//Implements ParamSurface::recomputeSurface(int iOldMode, char iChar)
//---------------------------------------------------
void ParamSurface::recomputeSurface(int iOldMode, char iChar)
{
    //Recalculation of the supporting points, normals and texture coordinates
    if (iChar == 'B' | iChar == 'P')
    {
        recomputeSurface('B');
    }
    //Cleanup from the old presentation mode
    if (m_creatingMode != iOldMode)
    {
        switch (iOldMode)
        {
        //Point cloud
        case 0:
            m_rpGeom->removePrimitiveSet(0);
            m_rpPointCloud = new DrawArrays(PrimitiveSet::POINTS);
            break;

        //Triangle mesh
        case 1:
            m_rpGeom->removePrimitiveSet(0);
            m_rpTriangEdges = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
            break;

        //Quad mesh
        case 2:
            break;

        //Quad mesh with a texture on it
        case 3:
            m_rpPolyMode->setMode(PolygonMode::FRONT_AND_BACK, PolygonMode::LINE);
            m_rpStateSet->setAttributeAndModes(m_rpPolyMode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            //Set texture off
            m_rpStateSet->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::OFF);
            m_rpMaterialSurface->setColorMode(Material::OFF);
            break;
        }
        //if during the old presentation step the quad mesh was needed
        // and for next presenation step the quad mesh is not needed
        if ((iOldMode == 2 | iOldMode == 3) && !m_areQuadsComputed)
        {
            m_rpGeom->removePrimitiveSet(0);
            m_rpQuadEdges = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
        }
        createObjectInMode();
    }
    else
    {
        switch (m_creatingMode)
        {
        //Point cloud
        case 0:
            m_rpGeom->removePrimitiveSet(0);

        //Triangle mesh
        case 1:
            m_rpGeom->removePrimitiveSet(0);
            m_rpTriangEdges = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
            break;

        //Quad mesh
        case 2:
        //Quad mesh with a texture on it
        case 3:
            m_rpGeom->removePrimitiveSet(0);
            m_rpQuadEdges = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
            break;
        }
        m_isSet = true;
        createObjectInMode();
        m_isSet = false;
    }
}

//---------------------------------------------------
//Implements ParamSurface::createObjectInMode()
//---------------------------------------------------
void ParamSurface::createObjectInMode()
{

    switch (m_creatingMode)
    {
    //pointcloud
    case 0:
        m_rpPointCloud->setFirst(0);
        m_rpPointCloud->setCount(m_rpSupportingPoints->size());
        m_rpGeom->addPrimitiveSet(m_rpPointCloud);
        m_rpGeom->setNormalBinding(Geometry::BIND_OFF);
        break;

    //triangles
    case 1:
        computeTriangleEdges();
        m_rpGeom->addPrimitiveSet(m_rpTriangEdges);

        if (!m_isSet)
        {
            m_rpStateSet->setAttributeAndModes(m_rpPolyMode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
        }
        m_rpGeom->setNormalBinding(Geometry::BIND_OFF);
        break;

    //quads
    case 2:

    //filled polygon mode
    case 3:
        if (m_areQuadsComputed == false)
        {
            computeQuadEdges();
            m_rpGeom->addPrimitiveSet(m_rpQuadEdges);
        }
        if (!m_isSet)
        {
            m_rpStateSet->setAttributeAndModes(m_rpPolyMode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
        }
        break;
    }
    if (m_creatingMode == 3)
    {
        if (!m_isSet)
        {
            m_rpPolyMode->setMode(PolygonMode::FRONT_AND_BACK, PolygonMode::FILL);
            m_rpStateSet->setAttributeAndModes(m_rpPolyMode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            if (m_rpImage.get())
            {
                m_rpTexture->setUnRefImageDataAfterApply(true);
                m_rpStateSet->setAttributeAndModes(m_rpMaterialSurface, osg::StateAttribute::ON);
                m_rpStateSet->setTextureAttributeAndModes(0, m_rpTexture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
                m_rpStateSet->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            }
            m_rpGeom->setNormalBinding(Geometry::BIND_PER_VERTEX);
        }
    }
}

//---------------------------------------------------
//Implements ParamSurface::createSubtree()
//---------------------------------------------------
void ParamSurface::createSubtree()
{
    m_rpGeode->addDrawable(m_rpGeom);
    m_rpGeode->setNodeMask(m_rpGeode->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    m_rpTransformNode->addChild(m_rpGeode);
}

//---------------------------------------------------
//Implements ParamSurface::createSurface()
//---------------------------------------------------
void ParamSurface::createSurface()
{
    createSubtree();
    this->digitalize();
    m_rpGeom->setVertexArray(m_rpSupportingPoints);
    m_rpGeom->setNormalArray(m_rpNormals);
    m_rpGeom->setNormalBinding(Geometry::BIND_PER_VERTEX);
    m_rpGeom->setTexCoordArray(0, m_rpTexCoords);
    createObjectInMode();
}

//---------------------------------------------------
//Implements ParamSurface::computeTriangleEdges()
//---------------------------------------------------
void ParamSurface::computeTriangleEdges()
{
    //Number of supporting points
    int numSupportingPointsU = m_patchesU + 1;
    int i = 0;

    //Don´t triagulate the last row because there exists no upper row
    //The beginning is at bottom left side, clockwise
    while ((i + numSupportingPointsU) < (m_rpSupportingPoints->size()))
    {
        if (((i + 1) % numSupportingPointsU) != 0)
        {
            //counterclockwise
            m_rpTriangEdges->push_back(i);
            m_rpTriangEdges->push_back(i + numSupportingPointsU);
            m_rpTriangEdges->push_back(i + numSupportingPointsU + 1);

            m_rpTriangEdges->push_back(i + numSupportingPointsU + 1);
            m_rpTriangEdges->push_back(i + 1);
            m_rpTriangEdges->push_back(i);
        }
        i++;
    }
    //Test loop
    //for(int i = 0; i < m_rpTriangEdges->size();i++){
    //   fprintf(stderr,"Edge: %i %i \n", i, m_rpTriangEdges->at(i));
    //}
}

//---------------------------------------------------
//Implements ParamSurface::computeQuadEdges()
//---------------------------------------------------
void ParamSurface::computeQuadEdges()
{
    //Number of supporting points
    int numSupportingPointsU = m_patchesU + 1;
    int i = 0;

    //Don´t triagulate the last row because there exists no upper row
    //The beginning is at bottom left side, clockwise
    while ((i + numSupportingPointsU) < (m_rpSupportingPoints->size()))
    {
        if (((i + 1) % numSupportingPointsU) != 0)
        {
            //counterclockwise
            m_rpQuadEdges->push_back(i);
            m_rpQuadEdges->push_back(i + numSupportingPointsU);
            m_rpQuadEdges->push_back(i + numSupportingPointsU + 1);
            m_rpQuadEdges->push_back(i + 1);
        }
        i++;
    }
    //   for(int i = 0; i < m_rpQuadEdges -> size(); i++){
    //      fprintf(stderr, "Verbindungspunkt %d \n",m_rpQuadEdges ->at(i));
    //   }
}

//---------------------------------------------------
//Implements ParamSurface::setSurface()
//---------------------------------------------------
void ParamSurface::setSurface()
{
    this->setColorAndMaterial();
}

//---------------------------------------------------
//Implements ParamSurface::setImage(int image_)
//---------------------------------------------------
void ParamSurface::setImage(int image_)
{
    _image = image_;
    this->setImageAndTexture();
}
//---------------------------------------------------
//Implements ParamSurface::setImageAndTexture()
//---------------------------------------------------
void ParamSurface::setImageAndTexture()
{

    if (_image == 1)
    {
        const std::string m_imagePath = (std::string)coCoviseConfig::getEntry("surface",
                                                                              "COVER.Plugin.SurfaceRenderer.Image", "/work/ac_te/Erdecrystalll.jpg");
        m_rpImage = osgDB::readImageFile(m_imagePath);
        m_rpTexture->setImage(m_rpImage.get());
    }
    else if (_image == 2)
    {
        const std::string m_imagePath = (std::string)coCoviseConfig::getEntry("surface",
                                                                              "COVER.plugin.SurfaceRenderer.Image", "/work/ac_te/VisensoLogo.jpg");
        m_rpImage = osgDB::readImageFile(m_imagePath);
        m_rpTexture->setImage(m_rpImage.get());
    }
    else if (_image == 0)
    {
        const std::string m_imagePath = (std::string)coCoviseConfig::getEntry("surface",
                                                                              "COVER.plugin.SurfaceRenderer.Image", "/work/ac_te/Red.jpg");
        m_rpImage = osgDB::readImageFile(m_imagePath);
        m_rpTexture->setImage(m_rpImage.get());
    }
}

//---------------------------------------------------
//Implements ParamSurface::setColorAndMaterial()
//---------------------------------------------------
void ParamSurface::setColorAndMaterial()
{
    m_rpMaterialSurface->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    m_rpMaterialSurface->setDiffuse(Material::FRONT_AND_BACK, *m_pColorSurface);
    m_rpMaterialSurface->setSpecular(Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    m_rpMaterialSurface->setAmbient(Material::FRONT_AND_BACK, Vec4(m_pColorSurface->x() * 0.3,
                                                                   m_pColorSurface->y() * 0.3,
                                                                   m_pColorSurface->z() * 0.3,
                                                                   m_pColorSurface->w()));
    m_rpMaterialSurface->setShininess(Material::FRONT_AND_BACK, 100.0);
}

//---------------------------------------------------
//Implements ParamSurface::digitalize()
//---------------------------------------------------
void ParamSurface::digitalize()
{
    double paramU = 0.0;
    double paramV = 0.0;
    int counterU = 0;
    int counterV = 0;
    //Length of the vector
    double length;

    // Loop the latitude, then along the meridian
    // External loop with v, sweep direction from bottom to top
    paramV = m_lowerBoundV;

    while (paramV <= (m_upperBoundV + m_epsilon))
    {
        counterU = 0;
        paramU = m_lowerBoundU;

        // Internal loop with u, sweep direction to the right
        while (paramU <= (m_upperBoundU + m_epsilon))
        {
            // parametric representation of a sphere
            // compute supporting points
            py_paramU = paramU;
            py_paramV = paramV;
            double x = computeInPython(m_x); // m_x = "cos(paramU) * cos(paramV)";
            double y = computeInPython(m_y);
            double z = computeInPython(m_z);
            m_rpSupportingPoints->push_back(Vec3(x, y, z));
            // Derivation with respect to u and derivation with respect to v
            // The cross product of both results in the normal vector
            // compute normal for each supporting point

            double xn = computeInPython(m_xDerived);
            double yn = computeInPython(m_yDerived);
            double zn = computeInPython(m_zDerived);
            length = xn * xn + yn * yn + zn * zn;

            //Normalized normal vector
            m_rpNormals->push_back(Vec3(-xn / sqrt(length), -yn / sqrt(length), -zn / sqrt(length)));

            // Texture coordinates are in a interval from 0 to 1
            // compute texture coordinate for each supporting point
            double ut = double(counterU) / double(m_patchesU);
            double vt = double(counterV) / double(m_patchesV);
            m_rpTexCoords->push_back(Vec2(ut, vt));

            //fprintf(stderr,"ArrayList %f %f %i %i \t %f %f %f \n",
            //paramU,paramV,counterU,counterV,x,y,z);
            //fprintf(stderr,"TexCoords %i %i %i %i \t %f %f  \n",
            //counterU,counterV,m_patchesU, m_patchesV, ut,vt);

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
//Implements ParamSurface::computeInPython(string iParametricRep)
//string iParametricRep becomes m_x in computeInPython(m_x)
//Python src evaluates string into double
//r returns output in double
//---------------------------------------------------
double ParamSurface::computeInPython(string iParametricRep)
{

    //build new local dictionary
    //cos/sin into local dic
    loc = PyDict_New();
    PyDict_SetItemString(loc, "cos", cosine);
    PyDict_SetItemString(loc, "sin", sinus);
    PyDict_SetItemString(loc, "tan", tangent);
    PyDict_SetItemString(loc, "log", logarithm);
    //build new global dictionary
    //items into global
    glb = PyDict_New();
    PyDict_SetItemString(glb, "paramU", PyFloat_FromDouble(py_paramU));
    PyDict_SetItemString(glb, "paramV", PyFloat_FromDouble(py_paramV));
    //rPython = cos(paramU) * cos(paramV)
    //string WithReturnVar = "rPython = " + iParametricRep;
    //Umwandlung von String in ConstChar
    const char *mParametricRep = iParametricRep.c_str();
    //teach how to math.h & parameters!!!!
    //pre-compiles iParametricRep
    code = Py_CompileString(mParametricRep, "<math>", Py_eval_input);
    //evaluates code object in environment of glb, loc
    ret = PyEval_EvalCode(code, glb, loc);

    //python ret --> double r
    r = PyFloat_AsDouble(ret);
    return r;
}
