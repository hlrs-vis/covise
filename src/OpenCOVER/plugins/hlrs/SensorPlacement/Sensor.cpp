#include "Sensor.h"
#include "Helper.h"

#include <osg/Geometry>
#include <osg/Material>
#include <osg/LineWidth>

#include <cover/coVRPluginSupport.h>

using namespace opencover;

void checkForObstacles(osg::Vec3& sensorPos)
{
};
void checkVisibility(const osg::Matrix& sensorMatrix, VisbilityMatrix& visMat)
{

};
void calculateVisibilityMatrix(std::vector<float>& visMat)
{

};

SensorPosition::SensorPosition(osg::Matrix matrix):m_Orientation(matrix)
{

};

Camera::Camera(osg::Matrix matrix)
    :SensorWithMultipleOrientations(matrix)
{
    float _interSize = cover->getSceneSize() ;
    m_Interactor = myHelpers::make_unique<coVR3DTransRotInteractor>(matrix, _interSize, vrui::coInteraction::ButtonA, "hand", "CamInteractor", vrui::coInteraction::Medium);
    m_Interactor->show();
    m_Interactor->enableIntersection();

    m_Geode = drawCam();
    m_Geode->setNodeMask(m_Geode->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

    m_CameraMatrix = new osg::MatrixTransform();
    m_CameraMatrix->setMatrix(matrix);
    m_CameraMatrix->addChild(m_Geode.get());
};


osg::Geode* Camera::drawCam()
{
    osg::Geode* geode = new osg::Geode();
    geode->setName("Camera");
    m_Geometry = new osg::Geometry;
    osg::StateSet *stateset = geode->getOrCreateStateSet();
    //necessary for dynamic redraw (command:dirty)
    m_Geometry->setDataVariance(osg::Object::DataVariance::DYNAMIC) ;
    m_Geometry->setUseDisplayList(false);
    m_Geometry->setUseVertexBufferObjects(true);
    //stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    //stateset->setAttributeAndModes(new osg::BlendFunc(GL_SRC_ALPHA ,GL_ONE_MINUS_SRC_ALPHA), osg::StateAttribute::ON);
    // Associate the Geometry with the Geode.
    geode->addDrawable(m_Geometry);
    geode->getStateSet()->setMode( GL_BLEND, osg::StateAttribute::ON );
    geode->getStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    geode->getStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );

    // Declare an array of vertices to create a simple pyramid.
    m_Verts = new osg::Vec3Array;
    m_Verts->push_back( osg::Vec3( -m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView, m_CameraProps.m_ImgHeight/2 )/m_Scale ); // 0 upper  front base
    m_Verts->push_back( osg::Vec3( -m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView,-m_CameraProps.m_ImgHeight/2 )/m_Scale ); // 1 lower front base
    m_Verts->push_back( osg::Vec3(  m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView,-m_CameraProps.m_ImgHeight/2 )/m_Scale ); // 3 lower  back  base
    m_Verts->push_back( osg::Vec3(  m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView, m_CameraProps.m_ImgHeight/2 )/m_Scale ); // 2 upper back  base
    m_Verts->push_back( osg::Vec3( 0,  0,  0) ); // 4 peak

    // Associate this set of vertices with the Geometry.
    m_Geometry->setVertexArray(m_Verts);

    // Next, create primitive sets and add them to the Geometry.
    // Each primitive set represents a face or a Line of the Pyramid
    // 0 base
    osg::DrawElementsUInt* face =
       new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
    face->push_back(3);
    face->push_back(2);
    face->push_back(1);
    face->push_back(0);
    m_Geometry->addPrimitiveSet(face);
    
    osg::DrawElementsUInt* line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(4);
    line->push_back(3);
    m_Geometry->addPrimitiveSet(line);
    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(4);
    line->push_back(2);
    m_Geometry->addPrimitiveSet(line); 
    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(4);
    line->push_back(1);
    m_Geometry->addPrimitiveSet(line); 
    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(4);
    line->push_back(0);
    m_Geometry->addPrimitiveSet(line);    
  
    // Create a separate color for each face.
    m_Colors = new osg::Vec4Array; //magenta 1 1 0; cyan 0 1 1; black 0 0 0
    osg::Vec4 makeTransparent = m_Color;
    makeTransparent.set(m_Color.x(),m_Color.y(),m_Color.z(),0.5);
    m_Colors->push_back( makeTransparent ); // magenta - back
    m_Colors->push_back( m_Color ); 
    m_Colors->push_back( m_Color ); 
    m_Colors->push_back( m_Color ); 
    m_Colors->push_back( m_Color ); 

    // Assign the color indices created above to the geometry and set the
    // binding mode to _PER_PRIMITIVE_SET.
    m_Geometry->setColorArray(m_Colors);
    m_Geometry->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);

    osg::Vec3Array* normals = new osg::Vec3Array();
    normals->push_back(osg::Vec3(-1.f ,-1.f, 0.f)); //left front
    normals->push_back(osg::Vec3(1.f ,-1.f, 0.f)); //right front
    normals->push_back(osg::Vec3(1.f ,1.f, 0.f));//right back
    normals->push_back(osg::Vec3(-1.f ,1.f, 0.f));//left back
    normals->push_back(osg::Vec3(0.f ,0.f, 1.f));//peak

    m_Geometry->setNormalArray(normals);
    // geom->setNormalBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);
    m_Geometry->setNormalBinding(osg::Geometry::BIND_OVERALL);

    //create Material
    //osg::Material *material = new osg::Material;
    //material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    //material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.2f, 0.2f, 1.0f));
    //material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.1f, 0.1f, 0.1f, 1.0f));
    //material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
    //material->setShininess(osg::Material::FRONT_AND_BACK, 25.0);
    //material->setTransparency(osg::Material::FRONT_AND_BACK,0.2);
    //material->setAlpha(osg::Material::FRONT_AND_BACK,0.2);
    //stateset->setAttributeAndModes(material);
    //stateset->setNestRenderBins(false);

    //LineWidth
    osg::LineWidth *lw = new osg::LineWidth(3.0);
    stateset->setAttribute(lw);
    // return the geode as the root of this geometry.
    return geode;

};

bool Camera::preFrame()
{
    if(m_Interactor->isRunning())
    {
        m_CameraMatrix->setMatrix(m_Interactor->getMatrix());
        //update Orientation of Sensor Base class! 
    }
    else if(m_Interactor->wasStopped())
    {
        return false;
        //calculate intersections !
    }
    return true;
}