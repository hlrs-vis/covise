/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VTK_VOLUME_H_
#define _VTK_VOLUME_H_

// Virvo:
#include <vvvoldesc.h>

// VTK:
#include <vtk/vtkStructuredPoints.h>
#include <vtk/vtkContourFilter.h>
#include <vtk/vtkActor.h>

// OSG:
#include <osg/Node>

/** Creates a VTK volume dataset from a Virvo volume.
*/
class vtkVolumeGeode
{
protected:
    osg::ref_ptr<osg::Node> _node;
    vtkStructuredPoints *_points;
    vtkContourFilter *_iso;
    vvVolDesc *_vd;
    vtkActor *_isoActor;

public:
    vtkVolumeGeode(vvVolDesc *, float, float, int);
    virtual ~vtkVolumeGeode();
    void makeIsosurface(vtkStructuredPoints *, vtkContourFilter **, vtkActor **, float);
    osg::Node *getNode();
    void setSize(osg::Vec3 &);
    osg::Vec3 getSize();
    void setIsoValue(float);
    void setIsoColor(float);
};

#endif
