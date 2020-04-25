#include<osg/Geode>
#include<osg/Geometry>
#include<osg/LineWidth>

#include <cover/coVRPluginSupport.h>

#include "Helper.h"
#include "Zone.h"
using namespace opencover;
Zone::Zone(osg::Matrix matrix)
{
    m_LocalDCS = new osg::MatrixTransform(matrix);
    m_Geode = draw();
    m_LocalDCS->addChild(m_Geode);

    float _interSize = cover->getSceneSize() / 25;

    osg::Matrix startPosInterator= osg::Matrix::translate(m_Verts->at(2)*matrix);
    m_Interactor= myHelpers::make_unique<coVR3DTransRotInteractor>(startPosInterator, _interSize/2, vrui::coInteraction::ButtonA, "hand", "Interactor", vrui::coInteraction::Medium);
    m_Interactor->show();
    m_Interactor->enableIntersection();

    osg::Vec3 startPosSizeInteractor= m_Verts->at(4)*matrix;
    m_SizeInteractor = myHelpers::make_unique<coVR3DTransInteractor>(startPosSizeInteractor, _interSize/2, vrui::coInteraction::ButtonA, "hand", "sizeInteractor", vrui::coInteraction::Medium);
    m_SizeInteractor->show();
    m_SizeInteractor->enableIntersection();

    cover->getObjectsRoot()->addChild(m_LocalDCS.get());
};

osg::Geode* Zone::draw()
{
    osg::Geode* geode = new osg::Geode();
    geode->setName("Wireframe");
    m_Geom = new osg::Geometry();
    osg::StateSet *stateset = geode->getOrCreateStateSet();
   // setStateSet(stateset);
    //necessary for dynamic redraw (command:dirty)
    m_Geom->setDataVariance(osg::Object::DataVariance::DYNAMIC) ;
    m_Geom->setUseDisplayList(false);
    m_Geom->setColorBinding(osg::Geometry::BIND_OVERALL);
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
    stateset->setAttribute(lw);
    return geode;
};

void Zone::preFrame()
{
    m_Interactor->preFrame();
    m_SizeInteractor->preFrame();

    static osg::Vec3 startPos_SizeInteractor_w,startPos_SizeInteractor_o;
    static osg::Matrix startMatrix_Interactor_to_w,startMatrix_Interactor_to_w_inverse;
    if(m_Interactor->wasStarted())
    {
        osg::Matrix interactor_to_w = m_Interactor->getMatrix();
        startPos_SizeInteractor_w = m_SizeInteractor->getPos();

        osg::Vec3 interactor_pos_w = interactor_to_w.getTrans();
        startPos_SizeInteractor_o= osg::Matrix::transform3x3(startPos_SizeInteractor_w-interactor_pos_w, interactor_to_w.inverse(interactor_to_w));
    }
    else if(m_Interactor->isRunning())
    {
        //update Interactors
        osg::Matrix interactor_to_w = m_Interactor->getMatrix();
        m_LocalDCS->setMatrix(interactor_to_w);
        osg::Vec3 interactor_pos_w;
        interactor_pos_w = interactor_to_w.getTrans();

        osg::Vec3 sizeInteractor_pos_w = osg::Matrix::transform3x3(startPos_SizeInteractor_o, interactor_to_w);
        sizeInteractor_pos_w +=interactor_pos_w;
        m_SizeInteractor->updateTransform(sizeInteractor_pos_w);
    }
    else if(m_Interactor->wasStopped())
    {
        // update world pos of points ? 
    }

    else if(m_SizeInteractor->wasStarted())
    {
        startMatrix_Interactor_to_w = m_Interactor->getMatrix();
        startMatrix_Interactor_to_w_inverse = startMatrix_Interactor_to_w.inverse(startMatrix_Interactor_to_w); 
        // delete points
    }
    else if(m_SizeInteractor->isRunning())
    {        
        // update vertices
        osg::Vec3 verts_o= osg::Matrix::transform3x3(m_SizeInteractor->getPos()-startMatrix_Interactor_to_w.getTrans(), startMatrix_Interactor_to_w_inverse);
        updateGeometry(verts_o);        
    }
    else if(m_SizeInteractor->wasStopped())
    {
        // create points
    }
};
void Zone::updateGeometry(osg::Vec3& vec)
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
     m_Length= std::abs(m_Verts->at(5).x()-m_Verts->at(6).x());
};

void Zone::setPosition(osg::Matrix matrix)
{
    m_Interactor->updateTransform(matrix);
    m_LocalDCS->setMatrix(matrix);
    //updatePos of Points!!!!
};

void Zone::deleteGridPoints()
{
    //remove Drawables
    m_GridPoints.clear();
};

/*void Zone::createGridPoints()
{
    std::vector<GridPoint> firstRow;
    firstRow.push_back(GridPoint(osg::Vec3(0,0,0)));

 

    float diffY= m_Verts.get()->at(7).y()-m_Verts.get()->at(6).y();
    float diffX = m_Verts.get()->at(5).x()-m_Verts.get()->at(6).x();
    float incrementLength = 0.0;
    float incrementWidth = 0.0;
    osg::Vec3 startPoint = m_Verts.get()->at(6);
    
    //create 2D Grid
    if(diffY<0)
    {
        while(incrementWidth < m_Width)
        {
            m_GridPoints.push_back(GridPoint(startPoint+osg::Vec3(0,-incrementWidth,-m_Height/2)));
            incrementLength += m_Distance;
            while(incrementLength < m_Lenght)
            {
                m_GridPoints.push_back(GridPoint(startPoint+osg::Vec3(0,-incrementLength,-m_Height/2)));
                incrementLength += m_Distance;
            }
        }
    }
    else
    {
        
    }
    
    float yStartValue = m_Distance;
};
*/