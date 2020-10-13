#include <iterator>
#include <future>

#include<osg/Geode>
#include<osg/Geometry>
#include<osg/LineWidth>
#include<osg/Material>
#include<osg/LightModel>

#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>

#include "Helper.h"
#include "Zone.h"
#include "UI.h"
#include "Sensor.h"
#include "Factory.h"
#include "Profiling.h"

using namespace opencover;

float findLargestVectorComponent(const osg::Vec3& vec) 
{ 
    float result;
    
    if (vec.x() >= vec.y() && vec.x() >= vec.z())
        result = vec.x();

    if (vec.y() >= vec.x() && vec.y() >= vec.z())
        result = vec.y();

    if (vec.z() >= vec.x() && vec.z() >= vec.y())
        result = vec.z();    

    return result;
}

ZoneShape::ZoneShape(osg::Matrix matrix,osg::Vec4 color, Zone* zone)
:m_ZonePointer(zone), m_Color(color)
{
    m_LocalDCS = new osg::MatrixTransform(matrix);
    
}

void ZoneShape::addPointToVec(osg::Vec3 point) // use rvalue here ? 
{
    float radius = calcGridPointDiameter();
    if(radius>0.5)
        radius = 0.5;
    m_GridPoints.push_back(GridPoint(point,m_Color,radius));
    m_LocalDCS->addChild(m_GridPoints.back().getPoint());
}


ZoneRectangle::ZoneRectangle(osg::Matrix matrix, float length, float width, float height,osg::Vec4 color, Zone* zone)
    :ZoneShape(matrix, color, zone), m_Length(length), m_Width(width), m_Height(height)
{
    
    m_Geode = draw();
    m_LocalDCS->addChild(m_Geode);
    
    float _interSize = cover->getSceneSize() / 50;

    osg::Matrix startPosInterator= osg::Matrix::translate(m_Verts->at(2) *matrix);
    m_Interactor= myHelpers::make_unique<coVR3DTransRotInteractor>(startPosInterator, _interSize, vrui::coInteraction::ButtonA, "hand", "Interactor", vrui::coInteraction::Medium);
    m_Interactor->show();
    m_Interactor->enableIntersection();

    osg::Vec3 startPosSizeInteractor= m_Verts->at(4)*matrix;
    m_SizeInteractor = myHelpers::make_unique<coVR3DTransInteractor>(startPosSizeInteractor, _interSize, vrui::coInteraction::ButtonA, "hand", "SizeInteractor", vrui::coInteraction::Medium);
    m_SizeInteractor->show();
    m_SizeInteractor->enableIntersection();

    osg::Vec3 longestSide = findLongestSide();
    float dist = findLargestVectorComponent(findLongestSide()) / 5;
    osg::Vec3 pos;
    if(longestSide.x() != 0)
       pos = osg::Vec3(-dist,0,0);
    else if(longestSide.y() != 0)
       pos = osg::Vec3(0,dist,0);
    else if(longestSide.z() != 0)
       pos = osg::Vec3(0,0,dist);

    // m_Distance = findLargestVectorComponent(findLongestSide()) / 5;
    // std::cout << "Constructor:" <<m_Distance <<std::endl;
    startPosSizeInteractor= (m_Verts->at(2)+pos)*matrix;
    m_DistanceInteractor = myHelpers::make_unique<coVR3DTransInteractor>(startPosSizeInteractor, _interSize, vrui::coInteraction::ButtonA, "hand", "DistanceInteractor", vrui::coInteraction::Medium);
    m_DistanceInteractor->show();
    m_DistanceInteractor->enableIntersection();

    //m_Distance = findLargestVectorComponent(findLongestSide()) / 5; // dont need this
    //std::cout <<"distance calculated : " <<findLargestVectorComponent(findLongestSide()) / 5<<std::endl;
    drawGrid();
};

osg::Geode* ZoneRectangle::draw()
{
    osg::Geode* geode = new osg::Geode();
    geode->setName("Wireframee");
    m_Geom = new osg::Geometry();
    osg::StateSet *stateset = VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE);
    geode->setStateSet(stateset);
   // setStateSet(stateset);
    //necessary for dynamic redraw (command:dirty)
    m_Geom->setDataVariance(osg::Object::DataVariance::DYNAMIC) ;
    m_Geom->setUseDisplayList(false);
    m_Geom->setNormalBinding(osg::Geometry::BIND_OVERALL);
    m_Geom->setUseVertexBufferObjects(true);
    geode->addDrawable(m_Geom);
    geode->getStateSet()->setMode( GL_BLEND, osg::StateAttribute::ON );
    geode->getStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    // Declare an array of vertices to create a simple pyramid.
    m_Verts = new osg::Vec3Array;
    m_Verts->push_back( osg::Vec3( -m_Length, m_Width, 0 ) ); // lower back left
    m_Verts->push_back( osg::Vec3( -m_Length,0, 0 ) );// lower front left
    m_Verts->push_back( osg::Vec3(  0,0, 0 ) );// lower front right
    m_Verts->push_back( osg::Vec3(  0, m_Width, 0 ) ); // lower back right
    m_Verts->push_back( osg::Vec3( -m_Length, m_Width,  m_Height ) ); // upper back left
    m_Verts->push_back( osg::Vec3( -m_Length,0,  m_Height ) );// upper front left
    m_Verts->push_back( osg::Vec3(  0,0,  m_Height ) );// upper front right
    m_Verts->push_back( osg::Vec3(  0, m_Width,  m_Height) ); // upper back right

    // Associate this set of vertices with the Geometry.
    m_Geom->setVertexArray(m_Verts);

    //set normals
    osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;
    normals->push_back((osg::Vec3(0.0,0.0,-1.0)));
    m_Geom->setNormalArray(normals);
    // Next, create primitive sets and add them to the Geometry.
    // Each primitive set represents a Line of the Wireframe

    //Lower Rectangle
    osg::DrawElementsUInt* line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(0);
    line->push_back(1);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(1);
    line->push_back(2);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(2);
    line->push_back(3);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(3);
    line->push_back(0);
    m_Geom->addPrimitiveSet(line);

    //UpperRectangle
    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(4);
    line->push_back(5);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(5);
    line->push_back(6);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(6);
    line->push_back(7);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(7);
    line->push_back(4);
    m_Geom->addPrimitiveSet(line);

    //Vertical Lines
    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(0);
    line->push_back(4);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(1);
    line->push_back(5);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(2);
    line->push_back(6);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(3);
    line->push_back(7);
    m_Geom->addPrimitiveSet(line);

    osg::LineWidth *lw = new osg::LineWidth(3.0);
   // stateset->setAttribute(lw);

    m_Colors = new osg::Vec4Array;
    m_Colors->push_back(m_Color);
    m_Geom->setColorArray(m_Colors);
    m_Geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    geode->getStateSet()->setAttribute(lw);
    return geode;
};

// osg::Vec3 ZoneRectangle::getVertex(int vertPos)const
// {
//     osg::Vec3 returnValue;
//     if(vertPos == 0)
//         returnValue = m_Verts->at(0);
//     else if(vertPos == 0)
//         returnValue = m_Verts->at(1);
//     else if(vertPos == 1)
//         returnValue = m_Verts->at(2);
//     else if(vertPos == 2)
//         returnValue = m_Verts->at(3);
//     else if(vertPos == 3)
//         returnValue = m_Verts->at(4);
//     else if(vertPos == 4)
//         returnValue = m_Verts->at(5);
//     else if(vertPos == 5)
//         returnValue = m_Verts->at(6);
//     else if(vertPos == 6)
//         returnValue = m_Verts->at(7);
//     else if(vertPos == 7)
    
//     return returnValue;
// }

bool ZoneRectangle::preFrame()
{
    
    m_Interactor->preFrame();
    m_SizeInteractor->preFrame();
    m_DistanceInteractor->preFrame();
    static osg::Vec3 startPos_SizeInteractor_w,startPos_SizeInteractor_o;
    static osg::Vec3 startPos_DistanceInteractor_w,startPos_DistanceInteractor_o;
    static osg::Matrix startMatrix_Interactor_to_w,startMatrix_Interactor_to_w_inverse;
    if(m_Interactor->wasStarted())
    {
        if(UI::m_DeleteStatus)
            return false;
        osg::Matrix interactor_to_w = m_Interactor->getMatrix();
        startPos_SizeInteractor_w = m_SizeInteractor->getPos();
        startPos_DistanceInteractor_w = m_DistanceInteractor->getPos();
        osg::Vec3 interactor_pos_w = interactor_to_w.getTrans();
        startPos_SizeInteractor_o = osg::Matrix::transform3x3(startPos_SizeInteractor_w-interactor_pos_w, interactor_to_w.inverse(interactor_to_w));
        startPos_DistanceInteractor_o = osg::Matrix::transform3x3(startPos_DistanceInteractor_w-interactor_pos_w, interactor_to_w.inverse(interactor_to_w));
        if(SensorZone* z = dynamic_cast<SensorZone*>(m_ZonePointer))
            z->removeAllSensors();
    }
    else if(m_Interactor->isRunning())
    {
        osg::Matrix interactor_to_w = m_Interactor->getMatrix();
        m_LocalDCS->setMatrix(interactor_to_w);
        osg::Vec3 interactor_pos_w = interactor_to_w.getTrans();
        
        //update Interactors
        osg::Vec3 sizeInteractor_pos_w = osg::Matrix::transform3x3(startPos_SizeInteractor_o, interactor_to_w);
        sizeInteractor_pos_w +=interactor_pos_w;
        m_SizeInteractor->updateTransform(sizeInteractor_pos_w);
        osg::Vec3 distanceInteractor_pos_w = osg::Matrix::transform3x3(startPos_DistanceInteractor_o, interactor_to_w);
        distanceInteractor_pos_w +=interactor_pos_w;
        m_DistanceInteractor->updateTransform(distanceInteractor_pos_w);
    }
    else if(m_Interactor->wasStopped())
    {
        if(SensorZone* z = dynamic_cast<SensorZone*>(m_ZonePointer))
            z->createSpecificNbrOfSensors();
    }
    else if(m_SizeInteractor->wasStarted())
    {
        startMatrix_Interactor_to_w = m_Interactor->getMatrix();
        startMatrix_Interactor_to_w_inverse = startMatrix_Interactor_to_w.inverse(startMatrix_Interactor_to_w); 
        osg::Vec3 interactor_pos_w = startMatrix_Interactor_to_w.getTrans();
        startPos_DistanceInteractor_w = m_DistanceInteractor->getPos();
        startPos_DistanceInteractor_o = osg::Matrix::transform3x3(startPos_DistanceInteractor_w-interactor_pos_w, startMatrix_Interactor_to_w_inverse);
        startPos_DistanceInteractor_o = osg::Vec3(std::abs(startPos_DistanceInteractor_o.x()),std::abs(startPos_DistanceInteractor_o.y()),std::abs(startPos_DistanceInteractor_o.z()));
        deleteGridPoints();
        m_DistanceInteractor->hide();
        if(SensorZone* z = dynamic_cast<SensorZone*>(m_ZonePointer))
            z->removeAllSensors();
    }
    else if(m_SizeInteractor->isRunning())
    {        
        // update vertices
        osg::Vec3 verts_o= osg::Matrix::transform3x3(m_SizeInteractor->getPos()-startMatrix_Interactor_to_w.getTrans(), startMatrix_Interactor_to_w_inverse);
        updateGeometry(verts_o);   
    }
    else if(m_SizeInteractor->wasStopped())
    {
        calcPositionOfDistanceInteractor(startPos_DistanceInteractor_o);
        m_DistanceInteractor->show();
        drawGrid();
        if(SensorZone* z = dynamic_cast<SensorZone*>(m_ZonePointer))
            z->createSpecificNbrOfSensors();
    }
    else if(m_DistanceInteractor->wasStarted())
    {
        osg::Matrix interactor_to_w = m_Interactor->getMatrix();
        startPos_DistanceInteractor_w = m_DistanceInteractor->getPos();
        startPos_DistanceInteractor_o= osg::Matrix::transform3x3(startPos_DistanceInteractor_w, interactor_to_w.inverse(interactor_to_w));
    }
    else if(m_DistanceInteractor->isRunning())      
        restrictDistanceInteractor(startPos_DistanceInteractor_o);
    else if(m_DistanceInteractor->wasStopped())
    {
        deleteGridPoints();
        drawGrid();
        if(SensorZone* z = dynamic_cast<SensorZone*>(m_ZonePointer))
            z->createSpecificNbrOfSensors();
    }
    
    
    return true;
};

void ZoneRectangle::updateGeometry(osg::Vec3& vec)
{
    //update y and z coordinate
     m_Verts->at(3) =osg::Vec3(m_Verts->at(3).x(),vec.y(),m_Verts->at(3).z());
     m_Verts->at(4) =osg::Vec3(m_Verts->at(4).x(),vec.y(),vec.z());
     m_Verts->at(7) =osg::Vec3(m_Verts->at(7).x(),vec.y(),vec.z());
     m_Verts->at(0) =osg::Vec3(m_Verts->at(0).x(),vec.y(),m_Verts->at(0).z());
    
    //update x and z coordinate
     m_Verts->at(0) =osg::Vec3(vec.x(),m_Verts->at(0).y(),m_Verts->at(0).z());
     m_Verts->at(4) =osg::Vec3(vec.x(),m_Verts->at(4).y(),m_Verts->at(4).z());
     m_Verts->at(5) =osg::Vec3(vec.x(),m_Verts->at(5).y(),vec.z());
     m_Verts->at(1) =osg::Vec3(vec.x(),m_Verts->at(1).y(),m_Verts->at(1).z());

    //update z coordinate
     m_Verts->at(6) =osg::Vec3(m_Verts->at(6).x(),m_Verts->at(6).y(),vec.z());

     m_Verts->dirty();
     m_Geom->dirtyBound();
     m_Width = std::abs(m_Verts->at(7).y()-m_Verts->at(6).y());
     m_Length = std::abs(m_Verts->at(5).x()-m_Verts->at(6).x());
     m_Height = std::abs(m_Verts->at(6).z()-m_Verts->at(2).z());
};
void ZoneRectangle::hide()
{
    m_Interactor->hide();
    m_SizeInteractor->hide();
    m_DistanceInteractor->hide(); 
}

void ZoneRectangle::show()
{
    m_Interactor->show();
    m_SizeInteractor->show();
    m_DistanceInteractor->show();
}

void ZoneShape::deleteGridPoints()
{
    for(const auto& point :m_GridPoints)
        m_LocalDCS->removeChild(point.getPoint());
    m_GridPoints.clear();
};

void ZoneSphere::createCircle()
{
    std::vector<osg::Vec3> temp;
    std::vector<osg::Vec3> verts;

    temp = circleVerts(osg::Z_AXIS, 0.70, 20, 0.0f);
    verts.insert(verts.begin(),temp.begin(), temp.end());

    temp = circleVerts(osg::Z_AXIS, 0.60, 20, 0.10f);
    verts.insert(verts.begin(),temp.begin(), temp.end());


    for(const auto& point : verts)
        addPointToVec(point);

}

std::vector<osg::Vec3> ZoneSphere::circleVerts(osg::Vec3 axis, float radius ,int approx, float height)
{
    const double angle( osg::PI * 2. / (double) approx );
    std::vector<osg::Vec3> v;
    int idx;   
    for( idx=0; idx<approx; idx++)
    {
        double cosAngle = cos(idx*angle);
        double sinAngle = sin(idx*angle);
        double x(0.), y(0.), z(0.);     
        if(axis == osg::Z_AXIS)
        {
            x = cosAngle*radius;
            y = sinAngle*radius; 
        }
        // else if(axis == osg::X_AXIS)
        // {
            // y = cosAngle*radius;
            // z = sinAngle*radius;
        // }
        // else if(axis == osg::Y_AXIS)
        // {
            // x = cosAngle*radius;
            // z = sinAngle*radius;
        // }
        //v.push_back( osg::Vec3( x, y, z ) );    
        v.push_back( osg::Vec3( x, y, height ) );    

    }
    return v;
}

void ZoneRectangle::calcPositionOfDistanceInteractor(osg::Vec3& startPos_DistanceInteractor_o)
{
    osg::Vec3 longestSide = findLongestSide();
    float posFactor = 3;

    if(longestSide.y() != 0)
    {
        float startPosY = findLargestVectorComponent(startPos_DistanceInteractor_o);
        if(std::abs(startPosY) > m_Width)
            startPos_DistanceInteractor_o.y() = m_Width / posFactor;
        else
            startPos_DistanceInteractor_o.y() = startPosY;

        startPos_DistanceInteractor_o.z() = 0;
        startPos_DistanceInteractor_o.x() = 0;
    }
    else if(longestSide.x() != 0)
    {
        float startPosX = findLargestVectorComponent(startPos_DistanceInteractor_o);
        if(std::abs(startPosX) > m_Length)   
            startPos_DistanceInteractor_o.x() =   m_Length / posFactor;
        
        else
            startPos_DistanceInteractor_o.x() = startPosX;

        startPos_DistanceInteractor_o.y() = 0;
        startPos_DistanceInteractor_o.z() = 0;
    }
    else if(longestSide.z() != 0)
    {
        float startPosZ = findLargestVectorComponent(startPos_DistanceInteractor_o);
        if(std::abs(startPosZ) > m_Height)
            startPos_DistanceInteractor_o.z() = m_Height / posFactor;
        else
            startPos_DistanceInteractor_o.z() = startPosZ;

        startPos_DistanceInteractor_o.x() = 0;
        startPos_DistanceInteractor_o.y() = 0;
    }

    osg::Vec3 sign = calcSign();
    osg::Vec3 newSign_o = osg::Vec3(startPos_DistanceInteractor_o.x()*sign.x(),startPos_DistanceInteractor_o.y()*sign.y(),startPos_DistanceInteractor_o.z()*sign.z());
    osg::Vec3 distanceInteractor_pos_w = osg::Matrix::transform3x3(newSign_o, m_Interactor->getMatrix());
    osg::Vec3 interactor_pos_w =  m_Interactor->getMatrix().getTrans();
    distanceInteractor_pos_w +=interactor_pos_w; 
    m_DistanceInteractor->updateTransform(distanceInteractor_pos_w);
}
osg::Vec3 ZoneRectangle::findLongestSide() const
{
    osg::Vec3 result;

    if (m_Width >= m_Length && m_Width >= m_Height)
        result = {0,m_Width,0};                                         // y (width) is longest side

    if (m_Length >= m_Width && m_Length >= m_Height)
        result = {m_Length,0,0};                                        // x (length) is longest side

    if (m_Height >= m_Width && m_Height >= m_Length)
        result = {0,0,m_Height};                                        // z (height) is longest side

    //std::cout<<"Longest Side" << result.x()<< " "<< result.y() << " "<< result.z() <<std::endl;
    return result;
}

float ZoneRectangle::calcGridPointDiameter()
{
   return findLargestVectorComponent(findLongestSide()/18);
}

float ZoneSphere::calcGridPointDiameter()
{
   return m_Radius/18;
}

void ZoneRectangle::restrictDistanceInteractor(osg::Vec3& startPos_DistanceInteractor_o)
{
        
    osg::Matrix interactor_to_w = m_Interactor->getMatrix();
    osg::Vec3 distanceInteractor_pos_o = osg::Matrix::transform3x3(m_DistanceInteractor->getPos(),interactor_to_w.inverse(interactor_to_w));
    osg::Vec3 sizeInteractor_pos_o = osg::Matrix::transform3x3(m_SizeInteractor->getPos(),interactor_to_w.inverse(interactor_to_w));
    osg::Vec3 interactor_pos_o = osg::Matrix::transform3x3(interactor_to_w.getTrans(),interactor_to_w.inverse(interactor_to_w));
    
    osg::Vec3 freeAxis = findLongestSide();
    osg::Vec3 sign = calcSign();
    if(freeAxis.x() != 0)
    {
        // restrict degree of freedom
        distanceInteractor_pos_o.y() = startPos_DistanceInteractor_o.y(); 
        distanceInteractor_pos_o.z() = startPos_DistanceInteractor_o.z(); 
        
        // set min and max position
        if(distanceInteractor_pos_o.x() < sizeInteractor_pos_o.x() && sign.x() < 0 )
            distanceInteractor_pos_o.x() = sizeInteractor_pos_o.x();
        else if(distanceInteractor_pos_o.x() > sizeInteractor_pos_o.x() && sign.x() > 0 )
            distanceInteractor_pos_o.x() = sizeInteractor_pos_o.x();
        else if (distanceInteractor_pos_o.x() > interactor_pos_o.x() && sign.x() < 0)  
            distanceInteractor_pos_o.x() = interactor_pos_o.x();  
        else if (distanceInteractor_pos_o.x() < interactor_pos_o.x() && sign.x() > 0)  
            distanceInteractor_pos_o.x() = interactor_pos_o.x();  
    }
    else if(freeAxis.y() != 0)
    {
        // restrict degree of freedom
        distanceInteractor_pos_o.x() = startPos_DistanceInteractor_o.x(); 
        distanceInteractor_pos_o.z() = startPos_DistanceInteractor_o.z(); 

        // set min and max position
        if(distanceInteractor_pos_o.y() > sizeInteractor_pos_o.y() && sign.y() > 0 )
            distanceInteractor_pos_o.y() = sizeInteractor_pos_o.y();
        else if(distanceInteractor_pos_o.y() < sizeInteractor_pos_o.y() && sign.y() < 0 )
            distanceInteractor_pos_o.y() = sizeInteractor_pos_o.y();
        else if (distanceInteractor_pos_o.y() < interactor_pos_o.y() && sign.y() > 0)  
            distanceInteractor_pos_o.y() = interactor_pos_o.y();  
        else if (distanceInteractor_pos_o.y() > interactor_pos_o.y() && sign.y() < 0)  
            distanceInteractor_pos_o.y() = interactor_pos_o.y();      
    }
    else if(freeAxis.z() != 0)
    {
        // restrict degree of freedom
        distanceInteractor_pos_o.y() = startPos_DistanceInteractor_o.y(); 
        distanceInteractor_pos_o.x() = startPos_DistanceInteractor_o.x(); 

        // set min and max position
        if(distanceInteractor_pos_o.z() > sizeInteractor_pos_o.z() && sign.z() > 0 )
            distanceInteractor_pos_o.z() = sizeInteractor_pos_o.z();
        else if(distanceInteractor_pos_o.z() < sizeInteractor_pos_o.z() && sign.z() < 0 )
            distanceInteractor_pos_o.z() = sizeInteractor_pos_o.z();
        else if (distanceInteractor_pos_o.z() < interactor_pos_o.z() && sign.z() > 0)  
            distanceInteractor_pos_o.z() = interactor_pos_o.z();  
        else if (distanceInteractor_pos_o.z() > interactor_pos_o.z() && sign.z() < 0)  
            distanceInteractor_pos_o.z() = interactor_pos_o.z();      
    }
    
    osg::Vec3 distanceInteractor_pos_w = osg::Matrix::transform3x3(distanceInteractor_pos_o,interactor_to_w);
    m_DistanceInteractor->updateTransform(distanceInteractor_pos_w);
}

float ZoneRectangle::calculateGridPointDistance() const
{
    float result;
    osg::Matrix interactor_to_w = m_Interactor->getMatrix();
    osg::Vec3 distanceInteractor_pos_o = osg::Matrix::transform3x3(m_DistanceInteractor->getPos(),interactor_to_w.inverse(interactor_to_w));
    osg::Vec3 interactor_pos_o = osg::Matrix::transform3x3(interactor_to_w.getTrans(),interactor_to_w.inverse(interactor_to_w));

    osg::Vec3 longestSide = findLongestSide();
    std::cout<<"calculateGridPointDistance() Longest side: " << longestSide.x()<< " "<< longestSide.y() << " "<< longestSide.z() <<std::endl;


    if(longestSide.x() != 0)
        result = std::abs(interactor_pos_o.x()-distanceInteractor_pos_o.x());
    else if(longestSide.y() != 0)
        result = std::abs(interactor_pos_o.y()-distanceInteractor_pos_o.y());
    else if(longestSide.z() != 0)
        result = std::abs(interactor_pos_o.z()-distanceInteractor_pos_o.z());
    
    std::cout <<"calculateGridPointDistance()"<<result<<".."<<std::endl;
    return result;
}

ZoneSphere::ZoneSphere(osg::Matrix matrix,float radius, osg::Vec4 color, Zone* zone)
: ZoneShape(matrix, color, zone), m_Radius(radius)
{
    m_Distance = 10;
    // m_Geode = draw();
    // m_LocalDCS->addChild(m_Geode);
    drawGrid();
}

void ZoneSphere::drawGrid()
{
    createCircle();
}

void ZoneRectangle::drawGrid()
{
    m_Distance = calculateGridPointDistance();
    float minDistance = findLargestVectorComponent(findLongestSide())  / 20 ; // minimum distance between gridpoints
    std::cout <<"draw Grid: m_Distance: "<<m_Distance <<std::endl;
    if(m_ZonePointer == nullptr)
        std::cout <<"nullptr"<<std::endl;
    if(m_Distance < minDistance)
    {
        std::cout << "Zone: No gridpoints created, distance is too small" << std::endl;

        std::cout << "distance "<<m_Distance << " minDistance "<<minDistance<<"..." << std::endl;
    }
    else
    {
        if( dynamic_cast<SensorZone*>(m_ZonePointer))
        {
            createInner3DGrid(defineStartPointForInnerGrid(),calcSign(),defineLimitsOfInnerGridPoints());
            std::cout <<"inner Grid: "<< std::endl;
        }else if(dynamic_cast<SafetyZone*>(m_ZonePointer))
        {
            std::cout <<"outer Grid: "<< std::endl;
            createOuter3DGrid(calcSign());
        }
        else if(dynamic_cast<Zone*>(m_ZonePointer))
        {
            std::cout <<"Cast to Zone successful "<< std::endl;
        }
        else
            std::cout << "no successful cast"<<std::endl;
    }
}

// void Zone::drawGrid()
// {
//     m_Distance = calculateGridPointDistance();
//     float minDistance = findLargestVectorComponent(findLongestSide())  / 20 ; // minimum distance between gridpoints

//     if(m_Distance < minDistance)
//         std::cout << "Zone: No gridpoints created, distance is too small" << std::endl;
//     else
//         createGrid();
// } 

void ZoneShape::setPosition(osg::Matrix matrix)
{
    m_LocalDCS->setMatrix(matrix);
    //updatePos of Points!!!!
};

void ZoneRectangle::setPosition(osg::Matrix matrix)
{
    ZoneShape::setPosition(matrix);
    m_Interactor->updateTransform(matrix);
}


void ZoneRectangle::createInner3DGrid(const osg::Vec3& startPoint, const osg::Vec3& sign, const osg::Vec3& limit)
{
    float incrementLength{0.0f}, incrementWidth{0.0f}, incrementHeight{0.0f};
    float widthLimit{limit.y()}, lengthLimit{limit.x()}, heightLimit{limit.z()};
    while(incrementWidth < widthLimit)
    {
        while(incrementLength < lengthLimit)
        {
            while(incrementHeight < heightLimit)
            {   
                osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
                addPointToVec(point);
                incrementHeight += m_Distance;
            }
            incrementHeight = 0.0;
            incrementLength += m_Distance;
            
        }
        incrementLength = 0.0;
        incrementWidth += m_Distance;
    }
}





void ZoneRectangle::createOuter3DGrid(const osg::Vec3& sign)
{
    float incrementLength{0.0f}, incrementWidth{0.0f}, incrementHeight{0.0f};

    // right side
    osg::Vec3 startPoint = m_Verts.get()->at(2);
    while(incrementWidth < m_Width)
    {
        while(incrementHeight < m_Height)
        {   
            osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
            addPointToVec(point);
            incrementHeight += m_Distance;
        }
        incrementHeight = 0.0;
        incrementWidth += m_Distance;
    }

    incrementWidth = incrementLength = incrementHeight = 0;

    // left side
    startPoint = m_Verts.get()->at(1);
    while(incrementWidth < m_Width)
    {
        while(incrementHeight < m_Height)
        {   
            osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
            addPointToVec(point);

            incrementHeight += m_Distance;
        }
        incrementHeight = 0.0;
        incrementWidth += m_Distance;
    }

    incrementWidth = incrementLength = incrementHeight = 0;

    // bottom
    startPoint = m_Verts.get()->at(2) + osg::Vec3(m_Distance * sign.x(),0,0);
    while(incrementLength < m_Length - m_Distance)
    {
        while(incrementWidth < m_Width )
        {   
            osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
            addPointToVec(point);

            incrementWidth += m_Distance;
        }
        incrementWidth = 0.0;
        incrementLength += m_Distance;
    }

    incrementWidth = incrementLength = incrementHeight = 0;

    // top
    startPoint = m_Verts.get()->at(6) ;
    while(incrementLength < m_Length)
    {
        while(incrementWidth < m_Width )
        {   
            osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
            addPointToVec(point);

            incrementWidth += m_Distance;
        }
        incrementWidth = 0.0;
        incrementLength += m_Distance;
    }

    incrementWidth = incrementLength = incrementHeight = 0;

    // front
    startPoint = m_Verts.get()->at(2)+osg::Vec3(m_Distance*sign.x(), 0, m_Distance*sign.z());
    while(incrementLength < m_Length -m_Distance)
    {
        while(incrementHeight < m_Height -m_Distance )
        {   
            osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
            addPointToVec(point);

            incrementHeight += m_Distance;
        }
        incrementHeight = 0.0;
        incrementLength += m_Distance; 
    }
    incrementWidth = incrementLength = incrementHeight = 0;

    // back
    startPoint = m_Verts.get()->at(3);
    while(incrementLength < m_Length )
    {
        while(incrementHeight < m_Height  )
        {   
            osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
            addPointToVec(point);

            incrementHeight += m_Distance;
        }
        incrementHeight = 0.0;
        incrementLength += m_Distance;
    }
    incrementWidth = incrementLength = incrementHeight = 0;

    //missing line from 7 to 4
    startPoint = m_Verts.get()->at(7);
    while(incrementLength < m_Length -m_Distance )
    {   
        osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
        addPointToVec(point);
        incrementLength += m_Distance;
    }
    incrementWidth = incrementLength = incrementHeight = 0;

    //missing line from 5 to 4 
    startPoint = m_Verts.get()->at(5);
    while(incrementWidth < m_Width -m_Distance )
    {   
        osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
        addPointToVec(point);
        incrementWidth += m_Distance;
    }
    incrementWidth = incrementLength = incrementHeight = 0;

    //missing line from 0 to 4
    startPoint = m_Verts.get()->at(0);
    while(incrementHeight < m_Height -m_Distance )
    {   
        osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
        addPointToVec(point);
        incrementHeight += m_Distance;
    }
    incrementWidth = incrementLength = incrementHeight = 0;
};

osg::Vec3 ZoneRectangle::calcSign() const
{
    float diffY= m_Verts.get()->at(3).y()-m_Verts.get()->at(2).y();
    float diffX = m_Verts.get()->at(2).x()-m_Verts.get()->at(1).x();
    float diffZ = m_Verts.get()->at(6).z()-m_Verts.get()->at(2).z();

    osg::Vec3 sign;

    if(diffY > 0 && diffX > 0 && diffZ > 0 )    
         sign = {-1,1,1};  
    else if(diffY < 0 && diffX > 0 && diffZ > 0 )
       sign = {-1,-1,1};     
    else if(diffY > 0 && diffX > 0 && diffZ < 0 )
       sign = {-1,1,-1};
    else if(diffY < 0 && diffX > 0 && diffZ < 0 ) 
        sign = {-1,-1,-1};      
    else if(diffY > 0 && diffX < 0 && diffZ > 0 )   
        sign = {1,1,1};      
    else if(diffY < 0 && diffX < 0 && diffZ > 0 ) 
        sign = {1,-1,1};   
    else if(diffY > 0 && diffX < 0 && diffZ < 0 )
        sign = {1,1,-1};     
    else if(diffY < 0 && diffX < 0 && diffZ < 0 )
        sign = {1,-1,-1};

    return sign;
}

osg::Vec3 ZoneRectangle::defineStartPointForInnerGrid()const
{
    osg::Vec3 corner = m_Verts.get()->at(2);
    osg::Vec3 diff = {m_Distance,m_Distance,m_Distance};
    
    if(m_Length - m_Distance < 0)
        diff[0] = m_Length / 2;  
    
    if(m_Width-m_Distance < 0)    
        diff[1] = m_Width / 2; 

    if(m_Height - m_Distance < 0)
        diff[2] = m_Height / 2;
    
    osg::Vec3 sign = calcSign();

    return  corner + osg::Vec3(diff.x()*sign.x(),diff.y()*sign.y(), diff.z()*sign.z());
}

osg::Vec3  ZoneRectangle::defineLimitsOfInnerGridPoints()const
{
    osg::Vec3 limits{m_Length-m_Distance,m_Width-m_Distance,m_Height-m_Distance};

    // if distance between Points is larger than one specific site -> position of gridpoints is in the center
    if(m_Length - m_Distance < 0)
        limits[0] = m_Length / 2; 

    if(m_Width-m_Distance < 0)    
        limits[1] = m_Width / 2;  

    if(m_Height - m_Distance < 0)
        limits[2] = m_Height / 2;

    return limits;
}


Zone::Zone(osg::Matrix matrix,osg::Vec4 color,float length, float width , float height)
    : m_Color(color)
{ 
    m_Group = new osg::Group;
};  

Zone::Zone(osg::Matrix matrix, osg::Vec4 color, float radius)
    : m_Color(color)
{
    m_Group = new osg::Group;
}

Zone::~Zone()
{
    std::cout<<" Pure virtual Zone Destructor called" << std::endl;
}


void Zone::setOriginalColor()
{
    for(auto& point : m_Shape->getGridPoints())
        point.setOriginalColor();
}

std::vector<osg::Vec3> Zone::getWorldPositionOfPoints()
{
    std::vector<osg::Vec3> result;
    result.reserve(m_Shape->getGridPoints().size());
    for(const auto& point : m_Shape->getGridPoints())
        result.push_back(point.getPosition()*m_Shape->getMatrix());
        
    return result;
}


SafetyZone::SafetyZone(osg::Matrix matrix, Priority prio, float length, float width , float height):
m_Priority(prio),Zone(matrix,calcColor(prio),length, width, height)
{   
    m_Shape = myHelpers::make_unique<ZoneRectangle>(matrix, length, width, height,calcColor(prio),this); 
    m_Group->addChild(m_Shape->getMatrixTransform());


    m_Text = new osgText::Text;
    m_Text->setColor(m_Color);
    m_Text->setCharacterSize(0.1);
    m_Text->setText(std::to_string(m_CurrentNbrOfSensors));

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    osg::ref_ptr<osg::MatrixTransform> mt = new osg::MatrixTransform();
    mt->setMatrix(osg::Matrix::rotate(osg::DegreesToRadians(90.0), osg::X_AXIS));
    mt->addChild(geode);
    geode->addDrawable(m_Text);

    m_Shape->getMatrixTransform()->addChild(mt);
}

SafetyZone::SafetyZone(osg::Matrix matrix, Priority prio, float radius):
    m_Priority(prio),Zone(matrix,calcColor(prio), radius)
{ 
    m_Shape = myHelpers::make_unique<ZoneSphere>(matrix, radius, calcColor(prio), this); 
    m_Group->addChild(m_Shape->getMatrixTransform());

  
    m_Text = new osgText::Text;
    m_Text->setColor(m_Color);
    m_Text->setCharacterSize(0.1);
    m_Text->setText(std::to_string(m_CurrentNbrOfSensors));

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    osg::ref_ptr<osg::MatrixTransform> mt = new osg::MatrixTransform();
    mt->setMatrix(osg::Matrix::rotate(osg::DegreesToRadians(90.0), osg::X_AXIS));
    mt->addChild(geode);
    geode->addDrawable(m_Text);

    m_Shape->getMatrixTransform()->addChild(mt);
}

void SafetyZone::setCurrentNbrOfSensors(int sensors)
{
    m_CurrentNbrOfSensors = sensors;
    m_Text->setText(std::to_string(m_CurrentNbrOfSensors));
}

osg::Vec4 SafetyZone::calcColor( Priority prio)const
{
    osg::Vec4 color;
    if(prio == Priority::PRIO1)
        color = osg::Vec4(1,0.5,0,1);
    else if(prio == Priority::PRIO2)
        color = osg::Vec4(1,0.75,0,1);

    return color;
}

void SafetyZone::highlitePoints(std::vector<float>& visiblePoints)
{
    if(visiblePoints.size() ==  m_Shape->getGridPoints().size())
    {
        for(auto it = visiblePoints.begin(); it != visiblePoints.end(); ++it)
        {
            if(*it != 0)
                m_Shape->getGridPoints().at(std::distance(visiblePoints.begin(), it)).highlite(osg::Vec4(0,1,0,1));
        }
    }
    else
        std::cout<<"error this shouldn't happen !"<<std::endl; //TODO Abort !!!!
}

void SafetyZone:: highlitePoints(VisibilityMatrix<float>& visiblePoints, osg::Vec4& colorVisible, osg::Vec4& colorNotVisible)
{
    if(visiblePoints.size() ==  m_Shape->getGridPoints().size())
    {
        for(auto it = visiblePoints.begin(); it != visiblePoints.end(); ++it)
        {
            if(*it != 0.0)
                m_Shape->getGridPoints().at(std::distance(visiblePoints.begin(), it)).setColor(colorVisible);
            else
                m_Shape->getGridPoints().at(std::distance(visiblePoints.begin(), it)).setColor(colorNotVisible);

        }
    }
    else
        std::cout<<"error this shouldn't happen !"<<std::endl; //TODO Abort !!!!
}

void SafetyZone::setPreviousZoneColor()
{
    for(auto& point : m_Shape->getGridPoints())
        point.setPreviousColor();
}


SensorZone::SensorZone(SensorType type, osg::Matrix matrix,float length, float width , float height)
    :Zone(matrix,osg::Vec4{1,0,1,1},length,width,height), m_SensorType(type)
{
    m_Shape = myHelpers::make_unique<ZoneRectangle>(matrix, length, width, height,osg::Vec4{1,0,1,1},this); 
    m_Group->addChild(m_Shape->getMatrixTransform());


    m_SensorGroup = new osg::Group();
    m_Group->addChild(m_SensorGroup.get());
    createSpecificNbrOfSensors();
}

SensorZone::SensorZone(SensorType type, osg::Matrix matrix, float radius)
    :Zone(matrix,osg::Vec4{1.0,0.0f,1.0f,0.4f},radius), m_SensorType(type)
{
    m_Shape = myHelpers::make_unique<ZoneSphere>(matrix, radius,osg::Vec4{1.0,0.0f,1.0f,0.4f}, this); 
    m_Group->addChild(m_Shape->getMatrixTransform());


    m_SensorGroup = new osg::Group();
    m_Group->addChild(m_SensorGroup.get());
    createSpecificNbrOfSensors();
}

bool SensorZone::preFrame()
{
    bool status = Zone::preFrame();

    for(const auto& sensor : m_Sensors)
        sensor->preFrame();
    // To Do: delete Sensor if returns false
    return status;
}


/*bool SensorZone::preFrame()
{
    bool status = Zone::preFrame();

    if(m_Interactor->wasStarted() || m_SizeInteractor->wasStarted())
        removeAllSensors();
        
    if(m_Interactor->wasStopped() || m_SizeInteractor->wasStopped() || m_DistanceInteractor->wasStopped()) 
        createSpecificNbrOfSensors();

    //check & restrict all sensors positions
    for(const auto& sensor : m_Sensors)
    {
        sensor->preFrame();
        if(sensor->getMatrix().getTrans().z()>m_SizeInteractor->getMatrix().getTrans().z())
        {  
            //std::cout<<"error"<<std::endl;
            //osg::Matrix m = sensor->getMatrix();
            //m.setTrans(m.getTrans().x(),m.getTrans().y(),m_SizeInteractor->getMatrix().getTrans().z()); 
            //sensor->setMatrix(m);
        }
    }
    return status;
}
*/

// void SensorZone::createGrid()
// {
//   createInner3DGrid(defineStartPointForInnerGrid(),calcSign(),defineLimitsOfInnerGridPoints());
  //createOuter3DGrid(calcSign());
// }

osg::Vec3 SensorZone::getFreeSensorPosition() const
{
    int nbrOfSensors = m_Sensors.size();
    return m_Shape->getGridPoints().at(m_Shape->getGridPoints().size() - nbrOfSensors -1).getPosition() ;
}

void SensorZone::addSensor(osg::Matrix matrix, bool visible)
{
    m_Sensors.push_back(Factory::createSensor(m_SensorType, matrix,visible));
    m_SensorGroup->addChild(m_Sensors.back()->getSensor());
}

void SensorZone::removeAllSensors()
{
    std::cout <<"remove all Sensors was called: Nbr of sensors before: " <<m_Sensors.size()<< std::endl;
    m_Sensors.clear();
    m_SensorGroup->removeChildren(0,m_SensorGroup->getNumChildren());
}

void SensorZone::createAllSensors()
{
    SP_PROFILE_BEGIN_SESSION("CreateSensorZone","SensorPlacement-CreateSensorZone.json");

    SP_PROFILE_FUNCTION();
    
    removeAllSensors();
  
    auto worldpositions = getWorldPositionOfPoints();
    for(const auto& worldpos : worldpositions)
        addSensor(osg::Matrix::translate(worldpos),true);
    
    std::vector<std::future<void>> futures;
    for(const auto& sensor : m_Sensors)
    {
        //sensor->calcVisibility();
        futures.push_back(std::async(std::launch::async, &SensorPosition::calcVisibility, sensor.get()));
    }
    SP_PROFILE_END_SESSION();
}

void SensorZone::createSpecificNbrOfSensors()
{
    removeAllSensors();

    if(m_NbrOfSensors <=m_Shape->getGridPoints().size())
    {
        for(size_t i{0};i<m_NbrOfSensors;i++)
        {
            osg::Matrix matrix = osg::Matrix::translate(getFreeSensorPosition()*m_Shape->getMatrix());
            osg::Quat q(osg::DegreesToRadians(50.0),osg::X_AXIS);
            matrix.setRotate(q);
            addSensor(matrix,true);
        }
    }
    else
        std::cout <<"Sensor Zone: No sensors created, not enough grid points available" << std::endl;
    
}

void SensorZone::createSensor(const osg::Matrix& matrix)
{
    addSensor(matrix, true);
}
void SensorZone::updateDoF(float dof)
{
    for(const auto& sensor : m_Sensors)
        sensor->updateDoF(dof);
}
void SensorZone::updateFoV(float fov)
{
    for(const auto& sensor : m_Sensors)
        sensor->updateFoV(fov);
}

/*void SensorZone::createSpecificNbrOfSensors(const std::vector<osg::Matrix>& sensorMatrixes)
{
    if(sensorMatrixes.size() != m_NbrOfSensors)
        throw std::invalid_argument( "received not the correct amount of sensors for sensor zone" );

    removeAllSensors();
    for(const auto& matrix : sensorMatrixes)
    {
        addSensor(matrix,true);
    }
}
*/

GridPoint::GridPoint(osg::Vec3 pos,osg::Vec4& color, float radius):m_Color(color),m_PreviousColor(color)
{
    osg::Matrix local;
    local.setTrans(pos);
    m_LocalDCS = new osg::MatrixTransform();
    m_LocalDCS->setMatrix(local);
    m_LocalDCS->setName("Translation");
    m_Sphere = new osg::Sphere(osg::Vec3(0,0,0), radius);
    m_SphereDrawable = new osg::ShapeDrawable(m_Sphere);
    m_SphereDrawable->setColor(color);
    m_Geode = new osg::Geode();
    //osg::StateSet *mystateSet = m_Geode->getOrCreateStateSet();
    //setStateSet(mystateSet);
    m_Geode->setName("Point");
    m_Geode->addDrawable(m_SphereDrawable);
    m_LocalDCS->addChild(m_Geode.get());
}
void GridPoint::highlite(const osg::Vec4& color)
{
    m_SphereDrawable->setColor(color);
}

void GridPoint::setColor(const osg::Vec4& color)
{
    m_PreviousColor = color;
    m_SphereDrawable->setColor(color);
}

void GridPoint::setOriginalColor()
{
    m_SphereDrawable->setColor(m_Color);

    m_PreviousColor = m_Color;
}


void GridPoint::setPreviousColor()
{
    m_SphereDrawable->setColor(m_PreviousColor);
}






