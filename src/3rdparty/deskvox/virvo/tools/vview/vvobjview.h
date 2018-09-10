//****************************************************************************
// Project:         Virvo (Virtual Reality Volume Renderer)
// Copyright:       (c) 1999-2004 Jurgen P. Schulze. All rights reserved.
// Author's E-Mail: schulze@cs.brown.edu
// Affiliation:     Brown University, Department of Computer Science
//****************************************************************************

#ifndef VV_OBJVIEW_H
#define VV_OBJVIEW_H

#include <virvo/vvvecmath.h>

#include <fstream>

//============================================================================
// Class Definition
//============================================================================

/** Object and viewer related functions.
  This class encapsulates all OpenGL projection and modelview matrix operations.
  @author Juergen Schulze-Doebold (schulze@hlrs.de)
*/
class vvObjView
{
  public:
    enum EyeType                                /// eyes for stereo viewing
    { LEFT_EYE, RIGHT_EYE };
    enum ProjectionType                         /// projection types
    { ORTHO, FRUSTUM, PERSPECTIVE };

  private:
    static const float VIEWER_POS_X;            ///< default viewer position on x axis
    static const float VIEWER_POS_Y;            ///< default viewer position on y axis
    static const float VIEWER_POS_Z;            ///< default viewer position on z axis
    ProjectionType projType;                    ///< projection type
    float   eyeDist;                            ///< eye distance for stereo mode [world space]
    float   rotAngle;                           ///< rotational angle for stereo mode [degrees]
    float   fov;                                ///< minimum field of view in either world space coordinates
    ///< or degrees, depending on projection type
    float   zNear;                              ///< near viewport clipping plane
    float   zFar;                               ///< far viewport clipping plane
    float   aspect;                             ///< viewport aspect ratio
    const char* cameraString;                   ///< string to store as camera file header

  public:
    vvMatrix mv;                               ///< modelview matrix for left eye and mono mode

    vvObjView();
    void  reset();
    void  resetMV();
    bool  saveMV(const char*);
    bool  saveMV(FILE* fp);
    bool  loadMV(const char*);
    bool  loadMV(std::ifstream& file);
    void  setProjection(ProjectionType, float, float, float);
    void  setAspectRatio(float);
    void  setDepthRange(float, float);
    void  trackballRotation(int, int, int, int, int, int);
    void  setEyeDistance(float);
    void  setRotationalAngle(float);
    void  updateProjectionMatrix();
    void  updateModelviewMatrix(EyeType = LEFT_EYE);
};
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
