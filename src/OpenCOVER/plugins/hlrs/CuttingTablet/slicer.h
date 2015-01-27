/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SLICER_H
#define SLICER_H

#include <osg/Texture1D>
#include <osg/Vec3>
#include <vtkSmartPointer.h>
#include <vtkPlane.h>
#include <vtkCutter.h>

class Slicer
{

public:
    Slicer(const std::string &pressure, const std::string &velocity,
           const std::string &pressureTexture, const std::string &velocityTexture);

    void setPlane(const osg::Vec3 &normal, const osg::Vec3 &origin);
    void setDataSet(int data);
    osg::ref_ptr<osg::Geode> getGeode();

private:
    vtkSmartPointer<vtkPlane> plane;
    vtkSmartPointer<vtkCutter> cutter[2];

    osg::ref_ptr<osg::StateSet> state[2];

    double minmax[4];
    int dataSet;
};

#endif
