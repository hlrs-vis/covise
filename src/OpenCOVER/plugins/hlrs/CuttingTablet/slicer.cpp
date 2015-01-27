/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCutter.h>
#include <vtkImageDataGeometryFilter.h>
#include <vtkFieldData.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPlane.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredPoints.h>
#include <vtkStructuredPointsReader.h>

#include <osg/Geode>
#include <osg/Material>
#include <osg/Texture1D>
#include <osgDB/ReadFile>

#include "slicer.h"

Slicer::Slicer(const std::string &pressure, const std::string &velocity,
               const std::string &pressureTexture, const std::string &velocityTexture)
    : dataSet(0)
{

    // Computational grid and scalar simulation data
    vtkSmartPointer<vtkStructuredPointsReader> preader = vtkSmartPointer<vtkStructuredPointsReader>::New();
    preader->SetFileName(pressure.c_str());
    preader->Update();

    vtkSmartPointer<vtkStructuredPointsReader> vreader = vtkSmartPointer<vtkStructuredPointsReader>::New();
    vreader->SetFileName(velocity.c_str());
    vreader->Update();

    // Cutting plane
    plane = vtkSmartPointer<vtkPlane>::New();

    // VTK cutter to slice dataset
    cutter[0] = vtkSmartPointer<vtkCutter>::New();
    cutter[0]->SetCutFunction(plane);
    cutter[0]->SetInputConnection(preader->GetOutputPort());

    cutter[1] = vtkSmartPointer<vtkCutter>::New();
    cutter[1]->SetCutFunction(plane);
    cutter[1]->SetInputConnection(vreader->GetOutputPort());

    osg::ref_ptr<osg::Image> image = osgDB::readImageFile(pressureTexture);
    osg::ref_ptr<osg::Texture1D> texture = new osg::Texture1D;
    texture->setWrap(osg::Texture1D::WRAP_S, osg::Texture1D::MIRROR);
    texture->setFilter(osg::Texture1D::MIN_FILTER,
                       osg::Texture1D::LINEAR_MIPMAP_LINEAR);
    texture->setFilter(osg::Texture1D::MAG_FILTER,
                       osg::Texture1D::LINEAR);
    texture->setImage(image);

    osg::ref_ptr<osg::Material> material = new osg::Material();
    state[0] = new osg::StateSet();
    state[0]->setTextureMode(0, GL_TEXTURE_1D, osg::StateAttribute::ON);
    state[0]->setAttribute(material);
    state[0]->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);
    state[0]->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    image = osgDB::readImageFile(velocityTexture);
    texture = new osg::Texture1D;
    texture->setWrap(osg::Texture1D::WRAP_S, osg::Texture1D::MIRROR);
    texture->setFilter(osg::Texture1D::MIN_FILTER,
                       osg::Texture1D::LINEAR_MIPMAP_LINEAR);
    texture->setFilter(osg::Texture1D::MAG_FILTER,
                       osg::Texture1D::LINEAR);
    texture->setImage(image);

    material = new osg::Material();
    state[1] = new osg::StateSet();
    state[1]->setTextureMode(0, GL_TEXTURE_1D, osg::StateAttribute::ON);
    state[1]->setAttribute(material);
    state[1]->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);
    state[1]->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    preader->GetOutput()->GetScalarRange(minmax);
    minmax[0] = -0.5;
    minmax[1] = 1;
    vreader->GetOutput()->GetScalarRange(minmax + 2);
    //printf("value range: [%.2f - %.2f]\n", minmax[0], minmax[1]);
}

void Slicer::setPlane(const osg::Vec3 &normal, const osg::Vec3 &origin)
{

    plane->SetNormal(normal.x(), normal.y(), normal.z());
    plane->SetOrigin(origin.x(), origin.y(), origin.z());
}

osg::ref_ptr<osg::Geode> Slicer::getGeode()
{

    cutter[dataSet]->Update();

    vtkSmartPointer<vtkCellArray> cellArray = cutter[dataSet]->GetOutput()->GetPolys();
    vtkSmartPointer<vtkPolyData> polyData = cutter[dataSet]->GetOutput(0);
    vtkSmartPointer<vtkPointData> pointData = polyData->GetPointData();
    vtkSmartPointer<vtkCellArray> verts = polyData->GetVerts();
    vtkSmartPointer<vtkCellArray> polys = polyData->GetPolys();

    int numPoly = polyData->GetNumberOfPolys();
    int numPoints = polyData->GetNumberOfPoints();
    /*
   printf("polys: %d\n", numPoly);
   printf("points: %d\n", numPoints);
   */
    // OpenSceneGraph geometry for VTK cutting plane results
    osg::Geode *geode = new osg::Geode();
    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();

    // convert VTK slice to OpenSceneGraph geometry
    for (int index = 0; index < numPoints; index++)
    {
        double vertex[3];
        polyData->GetPoint(index, vertex);
        vertices->push_back(osg::Vec3(vertex[0], vertex[1], vertex[2]));
    }
    geom->setVertexArray(vertices);
    osg::ref_ptr<osg::DrawElementsUInt> draw = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);

    vtkIdType *conn = cellArray->GetPointer();
    for (int index = 0; index < numPoly; index++)
    {
        int num = *(conn++);
        for (int i = 0; i < num; i++)
            draw->push_back(*(conn++));
    }
    geom->addPrimitiveSet(draw);
    geode->addDrawable(geom);

    osg::ref_ptr<osg::FloatArray> texCoords = new osg::FloatArray();
    for (int index = 0; index < numPoints; index++)
        texCoords->push_back((*(pointData->GetScalars()->GetTuple(index)) - minmax[0 + dataSet * 2]) / (minmax[1 + dataSet * 2] - minmax[0 + dataSet * 2]));
    geom->setTexCoordArray(0, texCoords);

    geom->setStateSet(state[dataSet]);

    return geode;
}

void Slicer::setDataSet(int data)
{

    dataSet = data;
}
