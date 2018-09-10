// DeskVOX - Volume Exploration Utility for the Desktop
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
// 
// This file is part of DeskVOX.
//
// DeskVOX is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the 
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef VV_OBJVIEW_H
#define VV_OBJVIEW_H

#include <vvvecmath.h>

//============================================================================
// Class Definition
//============================================================================

namespace vox
{

/** Object and viewer related functions.
  This class encapsulates all OpenGL projection and modelview matrix operations.
  @author Jurgen P. Schulze (jschulze@ucsd.edu)
*/
class vvObjView
{
  public:
    static const float DEF_CAMERA_POS_Z;          ///< default camera position on z axis [mm from origin]
    static const float DEF_IOD;                   ///< default inter-ocular distance [mm]
    static const float DEF_CLIP_NEAR;             ///< default clipping plane distance [mm]
    static const float DEF_CLIP_FAR;              ///< default clipping plane distance [mm]
    static const float DEF_FOV;                   ///< default field of view [radians]
    static const float EPSILON_NEAR;              ///< minimum distance for near plane in perspective mode [mm]
    static const float DEF_VIEWPORT_WIDTH;        ///< real-world width of OpenGL viewing canvas [mm]
    static const char* CAMERA_WINDOW_STRING;      ///< used in camera files to specify the viewing window
    static const char* CAMERA_CLIPPING_STRING;    ///< used in camera files to specify the clipping plane
    static const char* CAMERA_MATRIX_STRING;      ///< used in camera files to specify the camera matrix

    enum EyeType                                  /// eyes for stereo viewing
    { CENTER, LEFT_EYE, RIGHT_EYE };
    enum ProjectionType                           /// projection types
    { ORTHO, FRUSTUM, PERSPECTIVE };
    enum ViewType
    {
      LEFT, RIGHT, TOP, BOTTOM, FRONT, BACK
    };

  private:
    ProjectionType _projType;                     ///< projection type
    float _iod;                                   ///< inter-ocular distance for stereo mode [mm]
    float _minFOV;                                ///< minimum field of view, for perspective projection [radians]
    float _viewportWidth;                         ///< real-world width of OpenGL viewport, for parallel projection [mm]
    float _zNear;                                 ///< near viewport clipping plane
    float _zFar;                                  ///< far viewport clipping plane
    float _aspect;                                ///< viewport aspect ratio

  public:
    vvMatrix _object;                             ///< 'model' part of the OpenGL modelview matrix
    vvMatrix _camera;                             ///< 'view' part of the OpenGL modelview matrix

    vvObjView();
    void  reset();
    void  resetObject();
    void  resetCamera();
    bool  saveCamera(const char*);
    bool  loadCamera(const char*);
    void  setProjection(ProjectionType, float, float, float);
    void  setAspectRatio(float);
    void  setDepthRange(float, float);
    void  setIOD(float);
    float getIOD();
    void  setProjectionMatrix();
    void  setModelviewMatrix(EyeType = LEFT_EYE);
    float getFOV();
    float getViewportWidth();
    float getNearPlane();
    float getFarPlane();
    void  setDefaultView(ViewType);
    void  getWindowExtent(float&, float&, float&, float&, float&);
};

}
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
