/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//C++ header - fIVE|Analyse - Copyright (C) 2002-2003 Michael Gronager, UNI-C
//Distributed under the terms of the GNU Library General Public License (LGPL)
//as published by the Free Software Foundation.

#ifndef VTKACTORTOOSG_H
#define VTKACTORTOOSG_H

#include <osg/Geode>
#include <osg/Geometry>

#include <vtk/vtkActor.h>
#include <vtk/vtkPolyDataMapper.h>
#include <vtk/vtkCellArray.h>

// vtkActorToOSG - translates vtkActor to osg::Node. If geode is NULL, new one
//   will be created. Optional verbose parameter prints debugging and
//   performance information.
osg::Node *vtkActorToOSG(vtkActor *actor, int verbose = 0);

osg::Geometry *processPrimitive(vtkActor *a, vtkCellArray *prims, int pType, int v);

#endif
