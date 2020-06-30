#include "Camera.h"
#include "Helper.h"
#include "DataManager.h"
#include "UI.h"
#include "Profiling.h"

#include <osg/Material>
#include <osg/LineWidth>
#include <cover/coVRPluginSupport.h>

Camera::Camera(osg::Matrix matrix, bool visible):
    SensorWithMultipleOrientations(matrix)
{
    float _interSize = cover->getSceneSize() / 50 ;
    m_Interactor = myHelpers::make_unique<coVR3DTransRotInteractor>(matrix, _interSize, vrui::coInteraction::ButtonA, "hand", "CamInteractor", vrui::coInteraction::Medium);
    m_Interactor->enableIntersection();

    m_Geode = draw();
    m_SensorMatrix->addChild(m_Geode.get());

    m_GeodeOrientation = drawOrientation();
    VisualizationVisible(visible);
}

VisibilityMatrix<float> Camera::calcVisibilityMatrix(coCoord& euler)
{
    SP_PROFILE_FUNCTION();

    VisibilityMatrix<float> result = m_VisMatSensorPos;
    osg::Matrix matrix;
    euler.makeMat(matrix);

    osg::Matrix T = osg::Matrix::translate(-matrix.getTrans());
    osg::Matrix zRot = osg::Matrix::rotate(-osg::DegreesToRadians(euler.hpr[0]), osg::Z_AXIS);
    osg::Matrix yRot = osg::Matrix::rotate(-osg::DegreesToRadians(euler.hpr[1]), osg::X_AXIS);
    osg::Matrix xRot = osg::Matrix::rotate(-osg::DegreesToRadians(euler.hpr[2]), osg::Y_AXIS);

    auto ItPoint = DataManager::GetWorldPosOfObervationPoints().begin();
    auto ItVisMat = result.begin();

    if(result.size() == DataManager::GetWorldPosOfObervationPoints().size())
    {
        auto worldPosEnd = DataManager::GetWorldPosOfObervationPoints().end();
        while(ItPoint !=  worldPosEnd || ItVisMat != result.end())
        {
            if(*ItVisMat != 0) // no Obstacles in line of sight -> could be visible
            {
                osg::Vec3 point_CameraCoo = *ItPoint * T * zRot * yRot * xRot;
                point_CameraCoo.set(point_CameraCoo.x(),point_CameraCoo.y()*-1,point_CameraCoo.z());

                if((point_CameraCoo.y() <= m_CameraProps.m_DepthView ) &&
                   (point_CameraCoo.y() >= 0 ) && //TODO: too close is also not visible !
                   (std::abs(point_CameraCoo.x()) <= m_CameraProps.m_ImgWidth/2 * point_CameraCoo.y()/m_CameraProps.m_DepthView) &&
                   (std::abs(point_CameraCoo.z()) <= m_CameraProps.m_ImgHeight/2 * point_CameraCoo.y()/m_CameraProps.m_DepthView))
                {
                    // Point is in FoV
                    *ItVisMat = calcRangeDistortionFactor(point_CameraCoo)*calcWidthDistortionFactor(point_CameraCoo)*calcHeightDistortionFactor(point_CameraCoo);
                }
                else
                    *ItVisMat = 0;
            }
            // increment iterators
            if(ItVisMat != result.end())
            {
                ++ItVisMat;
                ++ItPoint;
            }
        }
    }
    else
    {
        //TODO: Throw error ! 
    }

    return result;
}

double Camera::calcRangeDistortionFactor(const osg::Vec3& point)const
{
    double y = point.y(); //distance between point and sensor in depth direction

    //SRC = Sensor Range Coefficient
    double calibratedValue = 70; // Parameter rangeDisortionDepth was calibrated for DephtView of 70;
    double SRCcoefficient = 23; // Rayleigh distribution with coefficient value of 23 looks reasonable for a camera with depthView of 70m
    double omega = SRCcoefficient * m_CameraProps.m_DepthView  / calibratedValue; // adapt calibratedValue to new depthView
    //normalized Rayleigh distribution function
    double SRC = omega*exp(0.5) * (y / pow(omega,2)) * exp(-(pow(y,2)) / (2*pow(omega,2)));
    return SRC;
}

double Camera::calcWidthDistortionFactor(const osg::Vec3& point)const
{
    double widthFOVatPoint = point.y()*std::tan(m_CameraProps.m_FoV/2*osg::PI/180);
    double x = std::abs(point.x()); //distance between point and sensor in width direction

    double x_scaled = calcValueInRange(0,widthFOVatPoint,0,1,x);
    //SWC = Sensor Width Coefficient SWC = -x² +1
    double SWC = - std::pow(x_scaled,2) + 1;
    return SWC;
}

double Camera::calcHeightDistortionFactor(const osg::Vec3& point)const
{
    double widthFOVatPoint = point.y()*std::tan(m_CameraProps.m_FoV/2 * osg::PI/180);
    double heightFOVatPoint = widthFOVatPoint/(m_CameraProps.getImageWidthPixel()/m_CameraProps.getImageHeightPixel());
    double z = std::abs(point.z()); //distance between point and sensor in width direction
    double z_scaled = calcValueInRange(0,heightFOVatPoint,0,1,z);
    //SWC = Sensor Width Coefficient SWC = -x² +1
    double SHC = - std::pow(z_scaled,2) + 1;
    return SHC;
}

// CameraVisualization::CameraVisualization(Camera* camera):SensorVisualization(camera), m_Camera(camera)
// {
//     float _interSize = cover->getSceneSize() / 50 ;
//     m_Interactor = myHelpers::make_unique<coVR3DTransRotInteractor>(camera->getMatrix(), _interSize, vrui::coInteraction::ButtonA, "hand", "CamInteractor", vrui::coInteraction::Medium);
//     m_Interactor->show();
//     m_Interactor->enableIntersection();

//     m_Geode = draw();
//     m_Geode->setNodeMask(m_Geode->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
//     m_Matrix->addChild(m_Geode.get());

//     m_Orientations = new osg::Group();
//     m_Group->addChild(m_Orientations.get());
//     m_GeodeOrientation = drawOrientation();
//     m_GeodeOrientation->setNodeMask(m_GeodeOrientation->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
// }

osg::Geode* Camera::draw()
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

    // Declare an array of vertices to create a simple pyramid
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
    bool status = SensorWithMultipleOrientations::preFrame();

    if(m_Interactor->wasStarted())
        showOriginalSensorSize();
    else if(m_Interactor->wasStopped())
    {   
        showIconSensorSize();
       if(UI::m_showOrientations)
       {
            for(const auto& orient : m_Orientations)
            {
                osg::ref_ptr<osg::MatrixTransform> matrixTransform = new osg::MatrixTransform(orient.getMatrix());
                matrixTransform->setName("Rotation matrix");
                matrixTransform->addChild(m_GeodeOrientation.get());
                m_OrientationsGroup->addChild(matrixTransform.get());
            }
       }
    }
    return status;
}

osg::Geode* Camera::drawOrientation()
{
    osg::Geode* geode = new osg::Geode;
    geode->setName("Camera orientation");
    m_GeometryOrientation = new osg::Geometry;
    osg::StateSet *stateset = geode->getOrCreateStateSet();
    geode->getStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    geode->addDrawable(m_GeometryOrientation);
    
    m_VertsOrientation = new osg::Vec3Array;
    m_VertsOrientation->push_back( osg::Vec3( -m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView, m_CameraProps.m_ImgHeight/2 ) / m_Scale); // 0 upper  front base
    m_VertsOrientation->push_back( osg::Vec3( -m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView,-m_CameraProps.m_ImgHeight/2 ) / m_Scale); // 1 lower front base
    m_VertsOrientation->push_back( osg::Vec3(  m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView,-m_CameraProps.m_ImgHeight/2 ) / m_Scale); // 3 lower  back  base
    m_VertsOrientation->push_back( osg::Vec3(  m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView, m_CameraProps.m_ImgHeight/2 ) / m_Scale); // 2 upper back  base

    m_GeometryOrientation->setVertexArray(m_VertsOrientation);

    osg::DrawElementsUInt* face = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0); // müsste ref_ptr sein ? 
    face->push_back(3);
    face->push_back(2);
    face->push_back(1);
    face->push_back(0);
    m_GeometryOrientation->addPrimitiveSet(face);

    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
    colors->push_back(osg::Vec4(1,0,1,0.3));
    m_GeometryOrientation->setColorArray(colors);
    m_GeometryOrientation->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);

    return geode;
}

void Camera::showOriginalSensorSize()
{
    m_Verts->at(0) = osg::Vec3(-m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView, m_CameraProps.m_ImgHeight/2); // 0 upper  front base
    m_Verts->at(1) = osg::Vec3(-m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView,-m_CameraProps.m_ImgHeight/2); // 1 lower front base
    m_Verts->at(2) = osg::Vec3( m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView,-m_CameraProps.m_ImgHeight/2); // 3 lower  back  base
    m_Verts->at(3) = osg::Vec3( m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView, m_CameraProps.m_ImgHeight/2) ;// 2 upper back  base
    m_Verts->at(4) = osg::Vec3( 0,  0,  0); // 4 peak
    m_Verts->dirty();
    m_Geometry->dirtyBound();
}

void Camera::showIconSensorSize()
{
    m_Verts->at(0) = osg::Vec3(-m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView, m_CameraProps.m_ImgHeight/2)/m_Scale; // 0 upper  front base
    m_Verts->at(1) = osg::Vec3(-m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView,-m_CameraProps.m_ImgHeight/2)/m_Scale; // 1 lower front base
    m_Verts->at(2) = osg::Vec3( m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView,-m_CameraProps.m_ImgHeight/2)/m_Scale; // 3 lower  back  base
    m_Verts->at(3) = osg::Vec3( m_CameraProps.m_ImgWidth/2,-m_CameraProps.m_DepthView, m_CameraProps.m_ImgHeight/2)/m_Scale;// 2 upper back  base
    m_Verts->at(4) = osg::Vec3( 0,  0,  0); // 4 peak
    m_Verts->dirty();
    m_Geometry->dirtyBound();
}

void Camera::VisualizationVisible(bool status)const
{
    SensorPosition::VisualizationVisible(status);

    if(!status)  
        m_Interactor->hide();
    else
        m_Interactor->show();    
}

