#include "Sensor.h"
#include "Helper.h"
#include "UI.h"
#include "DataManager.h"

#include <osg/Geometry>
#include <osg/Material>
#include <osg/LineWidth>

#include <cover/coVRPluginSupport.h>

#include <bits/stdc++.h>

using namespace opencover;

void printCoCoord(coCoord m)
{
    std::cout<<"translation: "<<"x:"<<m.xyz[0]<< " y:"<<m.xyz[1]<<" z:"<<m.xyz[2]<<std::endl;
    std::cout<<"rotation: "<<"z:"<<m.hpr[0]<< " x:"<<m.hpr[1]<<" y:"<<m.hpr[2]<<std::endl;
}

double calcValueInRange(double oldMin, double oldMax, double newMin, double newMax, double oldValue)
{
    double oldRange = oldMax - oldMin;
    double newRange = newMax - newMin;
    double newValue = (((oldValue - oldMin) * newRange) / oldRange )+ newMin;
    return newValue;
}

std::vector<osg::Vec3> Vec2DimToVec(std::vector<std::vector<osg::Vec3>> input)
{    
  /*  std::vector<osg::Vec3> result;
    
    size_t reserve_size = 0;
    
    for(const auto& i : input)
        reserve_size += i.size();

    result.reserve(reserve_size);
        
    for (int i=0; i<bits.size(); ++i)
    {
        const vector<int> & v = bits[i];  // just to make code more readable (note ..  a reference)
        
        bits1d.insert( bits1d.end() , v.begin() , v.end() );
    }
    for(const auto& i : input.size())
    {
        const std::vector<osg::Vec3> tmp = i;
        result.insert(result.end(),tmp.begin(),tmp.end());
    }
    */
}

/*void checkVisibility(const osg::Matrix& sensorMatrix, VisbilityMatrix& visMat)
{

};
*/
Orientation::Orientation(osg::Matrix matrix):m_Matrix(matrix)
{
}
Orientation::Orientation(osg::Matrix matrix,VisibilityMatrix<float>&& visMat):m_Matrix(matrix),m_Euler(matrix),m_VisibilityMatrix(visMat)
{
}

Orientation::Orientation(coCoord euler,VisibilityMatrix<float>&& visMat):m_Euler(euler),m_VisibilityMatrix(visMat)
{
    euler.makeMat(m_Matrix);
}

void Orientation::setMatrix(osg::Matrix matrix)
{
   m_Matrix = matrix;
   m_Euler = matrix; 
}

void Orientation::setMatrix(coCoord euler)
{
   m_Euler = euler; 
   euler.makeMat(m_Matrix);
}

void Orientation::setVisibilityMatrix(VisibilityMatrix<float>&& visMat)
{
    m_VisibilityMatrix.clear();
    m_VisibilityMatrix = std::move(visMat);
}

// bool Orientation::operator >> (const Orientation& other) const
// {
//     size_t onlyVisibleForThisSensor{0}, onlyVisibleForOtherSensor{0};

//     auto ItThisSensor = m_VisibilityMatrix.begin(); 
//     auto ItOtherSensor = other.m_VisibilityMatrix.begin();

//     if(m_VisibilityMatrix.size() == other.m_VisibilityMatrix.size())
//     {
//         while(ItThisSensor != m_VisibilityMatrix.end() || ItOtherSensor != other.m_VisibilityMatrix.end() )
//         {
//             if(*ItThisSensor > 0 && *ItOtherSensor == 0)
//                 ++onlyVisibleForThisSensor;
//             if(*ItThisSensor == 0 && *ItOtherSensor > 0)
//                 ++onlyVisibleForOtherSensor;

//             if((onlyVisibleForThisSensor && onlyVisibleForOtherSensor) !=0) 
//                 return false; //each sensor can see points, which the other can't see --> keep both 
            
//             // Loop further
//             if(ItThisSensor != m_VisibilityMatrix.end())
//             {
//                 ++ItOtherSensor;
//                 ++ItThisSensor;
//             }
//         }
//     }

//     if(onlyVisibleForThisSensor != 0 && onlyVisibleForOtherSensor == 0)
//         return true;// this sensor can see points, which other can't see
//     else if(onlyVisibleForThisSensor == 0 && onlyVisibleForOtherSensor != 0)
//         return false; // other sensor can see points, which this can't see
//     else if(onlyVisibleForThisSensor == 0 && onlyVisibleForOtherSensor == 0)
//     {
//         //TODO: prefere no Rotation around own axis !
//         //TODO: This is specific to sensor Type ????? 

//         //both sensors see exactly the same points -> check other Sensor characteristics
//         double coefficientsThis = std::accumulate(m_VisibilityMatrix.begin(),m_VisibilityMatrix.end(),0.0);
//         double coefficientOther = std::accumulate(other.m_VisibilityMatrix.begin(),other.m_VisibilityMatrix.end(),0.0);
//         if(coefficientOther > coefficientsThis)
//             return false;
//         else
//             return true;
        
//     }


// }


SensorPosition::SensorPosition(osg::Matrix matrix):m_Orientation(matrix)
{

};

void SensorPosition::checkForObstacles()
{
    auto allObservationPoints = DataManager::GetInstance().GetWorldPosOfObervationPoints();
    
    m_VisMatSensorPos.clear();
    m_VisMatSensorPos.reserve(allObservationPoints.size());
    
    osg::ref_ptr<osgUtil::IntersectorGroup> intersectorGroup = new osgUtil::IntersectorGroup();

    osg::Vec3 sensorPos = m_Orientation.getMatrix().getTrans();
    for(const auto& point : allObservationPoints)
    {
        osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector = new osgUtil::LineSegmentIntersector(sensorPos, point);
        intersectorGroup->addIntersector( intersector.get() );
    }
    osgUtil::IntersectionVisitor visitor(intersectorGroup.get());
    cover->getObjectsRoot()->accept(visitor);

    if(intersectorGroup->containsIntersections())
    {
        osgUtil::IntersectorGroup::Intersectors& intersectors = intersectorGroup->getIntersectors();
        for(osgUtil::IntersectorGroup::Intersectors::iterator intersector_itr = intersectors.begin();
            intersector_itr != intersectors.end(); 
            ++intersector_itr)
        {
            osgUtil::LineSegmentIntersector* lsi = dynamic_cast<osgUtil::LineSegmentIntersector*>(intersector_itr->get());
            if(lsi)
            {
                osgUtil::LineSegmentIntersector::Intersections& hits = lsi->getIntersections();
                //std::cout<<hits.size()<<std::endl;
                if(hits.size() > 1 )//TODO Geometrien des SensorPlacement müssen vom Intersection Test ausgeschlossen werden! 
                    m_VisMatSensorPos.push_back(0);
                else
                    m_VisMatSensorPos.push_back(1);
            }
        }
    }

};

void SensorWithMultipleOrientations::createSensorOrientations()
{
    m_Orientations.clear();
    //m_Orientations.reserve() // sinnvoll weil kann auch nur eine übrig bleiben? 
    osg::Vec3 pos = m_Orientation.getMatrix().getTrans();
    osg::Matrix matrix = osg::Matrix::translate(pos);
    coCoord euler = matrix;

    size_t rotz{0},rotx{0},roty{0};
    while(rotz < 360/m_SensorProps.m_StepSizeZ ) // z rotation
    {
        while(rotx < 180/m_SensorProps.m_StepSizeX) // x rotation
        {
            while(roty < 360/m_SensorProps.m_StepSizeY) // y rotation
            {
                decideWhichOrientationsAreRequired(Orientation{euler,calcVisibilityMatrix(euler)});
                euler.hpr[2] += m_SensorProps.m_StepSizeY;
                roty++;
            }
            euler.hpr[1] += m_SensorProps.m_StepSizeX;
            roty = 0;
            rotx++;
        }
        euler.hpr[0] += m_SensorProps.m_StepSizeZ;
        rotx = 0;
        rotz++;
    }
}

void SensorWithMultipleOrientations::decideWhichOrientationsAreRequired(const Orientation&& orientation)
{
    if(!m_Orientations.empty())
    {
        auto it = m_Orientations.begin();
        while(it != m_Orientations.end())
        {
            if(compareOrientations(*it, orientation))
                return; // this orientation isn't required
            else if(compareOrientations(orientation,*it))
            {
                if(m_Orientations.size() > 1)
                    replaceOrientationWithLastElement(std::distance(m_Orientations.begin(),it));
                else
                    m_Orientations.pop_back();
            
                continue; // do not incremenct iterator !
            }
            else
                ++it;  
        }
    }

    m_Orientations.push_back(orientation);
}

void SensorWithMultipleOrientations::replaceOrientationWithLastElement(int index)
{
    if(!m_Orientations.empty())
    {
        m_Orientations.at(index) = m_Orientations.back();
        m_Orientations.pop_back();
    }   
}

bool SensorWithMultipleOrientations::compareOrientations(const Orientation& lhs, const Orientation& rhs)
{
    size_t onlyVisibleForLhsSensor{0}, onlyVisibleForRhsSensor{0};

    auto ItLhsSensor = lhs.getVisibilityMatrix().begin(); 
    auto ItRhsSensor = rhs.getVisibilityMatrix().begin();

    if(lhs.getVisibilityMatrix().size() == rhs.getVisibilityMatrix().size())
    {
        while(ItLhsSensor != lhs.getVisibilityMatrix().end() || ItRhsSensor != rhs.getVisibilityMatrix().end() )
        {
            if(*ItLhsSensor > 0 && *ItRhsSensor == 0)
                ++onlyVisibleForLhsSensor;
            if(*ItLhsSensor == 0 && *ItRhsSensor > 0)
                ++onlyVisibleForRhsSensor;

            if((onlyVisibleForLhsSensor && onlyVisibleForRhsSensor) !=0) 
                return false; //each sensor can see points, which the other can't see --> keep both 
            
            // Loop further
            if(ItLhsSensor != lhs.getVisibilityMatrix().end())
            {
                ++ItRhsSensor;
                ++ItLhsSensor;
            }
        }
    }

    if(onlyVisibleForLhsSensor != 0 && onlyVisibleForRhsSensor == 0)
        return true;// Lhs sensor can see points, which Rhs can't see
    else if(onlyVisibleForLhsSensor == 0 && onlyVisibleForRhsSensor != 0)
        return false; // Rhs sensor can see points, which Lhs can't see
    else if(onlyVisibleForLhsSensor == 0 && onlyVisibleForRhsSensor == 0)
    {
        //TODO: prefere no Rotation around own axis !
        //TODO: Lhs is specific to sensor Type ????? 

        //both sensors see exactly the same points -> check Rhs Sensor characteristics
        double coefficientsLhs = std::accumulate(lhs.getVisibilityMatrix().begin(),lhs.getVisibilityMatrix().end(),0.0);
        double coefficientRhs = std::accumulate(rhs.getVisibilityMatrix().begin(),rhs.getVisibilityMatrix().end(),0.0);
        if(coefficientRhs > coefficientsLhs)
            return false;
        else
            return true;  
    }

}


Camera::Camera(osg::Matrix matrix)
    :SensorWithMultipleOrientations(matrix)
{
    float _interSize = cover->getSceneSize() / 50 ;
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
    
    if(m_Interactor->wasStarted())
    {   
        if(UI::m_DeleteStatus)
            return false;
    }
    else if(m_Interactor->isRunning())
    {
        osg::Matrix matrix= m_Interactor->getMatrix();
        m_CameraMatrix->setMatrix(matrix);
        m_Orientation.setMatrix(matrix);
    }
    else if(m_Interactor->wasStopped())
    {
        osg::Matrix matrix= m_Interactor->getMatrix();
        coCoord euler = matrix;

        checkForObstacles();

        m_Orientation.setMatrix(matrix);
        m_Orientation.setVisibilityMatrix(calcVisibilityMatrix(euler));

        createSensorOrientations();

        m_OrientationsDrawables.clear();
        for(const auto& mat: m_Orientations)
        {
            m_OrientationsDrawables.push_back(new osg::MatrixTransform(mat.getMatrix()));
            m_OrientationsDrawables.back()->addChild(m_Geode.get());
            m_CameraMatrix->addChild(m_OrientationsDrawables.back().get());
        }
    }

    return true;
}

VisibilityMatrix<float> Camera::calcVisibilityMatrix(coCoord& euler)//hier muss Matrix als input Rein! damit für alle Orientierungen nutzbar!
{
    VisibilityMatrix<float> result = m_VisMatSensorPos;
    //  std::cout<<"Before Visibility Calculation"<<std::endl;
    // for(const auto& x : result)
    //        std::cout<<x<<std::endl;
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
        while(ItPoint != DataManager::GetWorldPosOfObervationPoints().end() || ItVisMat != result.end())
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

    // std::cout<<"After Visibility Calculation"<<std::endl;
    // for(const auto& x : result)
    //        std::cout<<x<<std::endl;

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





















