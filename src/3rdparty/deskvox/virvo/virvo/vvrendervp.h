//****************************************************************************
// Project Affiliation: Virvo (Virtual Reality Volume Renderer)
// Copyright:           (c) 2002 Juergen Schulze-Doebold. All rights reserved.
// Author's E-Mail:     schulze@hlrs.de
// Institution:         University of Stuttgart, Supercomputing Center (HLRS)
//****************************************************************************

#ifndef VV_RENDERVP_H
#define VV_RENDERVP_H

#ifdef HAVE_CONFIG_H
#include <vvconfig.h>
#endif

#ifdef HAVE_VOLPACK

#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <volpack.h>
#include "vvvecmath.h"
#include "vvrenderer.h"
#include "vvswitchrenderer.h"

//============================================================================
// Macro Definitions
//============================================================================

#define FATAL_VP_ERROR(str,e) { std::cerr << "Fatal Error (VolPack) in " << str <<  \
  ": " << vpGetErrorString(e) << std::endl; exit(-1); }

#define FATAL_RENDERVP_ERROR(str) { std::cerr << "Fatal Error (vvRenderVP): " << str << std::endl; exit(-1); }

//============================================================================
// Class Definitions
//============================================================================
/** Rendering engine using the VolPack API.
  The VolPack API was created by P. Lacroute from Stanford University.
  (http://graphics.stanford.EDU/software/volpack/)
  The VolPack library applies the shear-warp algorithm and does not need
  any hardware acceleration.
  @author Juergen Schulze-Doebold (schulze@hlrs.de)
  @see vvRenderer
*/
class vvRenderVP : public vvRenderer
{
  enum                              /// constants for the volume data representation
  {
    NUM_FIELDS           = 3,
    NUM_SHADE_FIELDS     = 2,
    NUM_CLASSIFY_FIELDS  = 2,
    NORM_FIELD           = 0,
    SCALAR_FIELD         = 1,
    SCALAR_MAX           = 255,
    GRAD_FIELD           = 2,
    COLOR_CHANNELS       = 3,
    DEFAULT_IMAGE_WIDTH  = 300,     ///< output image width
    DEFAULT_IMAGE_HEIGHT = 300      ///< output image height
  };

  private:
    /** Voxel data type.
        Consists of 1 byte scalar value, 2 bytes normal, and 1 byte gradient value.
    */
    typedef class         
    {
      public:
        ushort normal;    ///< encoded normal vector
        uchar  scalar;    ///< raw scalar voxel value
        uchar  gradient;  ///< gradient index
    } VoxelType;

    float color_table[VP_NORM_MAX+1][VP_MAX_MATERIAL-1][COLOR_CHANNELS];   ///< normals LUT
    float scalar_table[VP_SCALAR_MAX+1];                    ///< opacity table
    float gradient_table[VP_GRAD_MAX + 1];                  ///< gradients table
    float weight_table[VP_SCALAR_MAX+1][VP_MAX_MATERIAL-1]; ///< RGB color table
    vpContext* vpc;                                         ///< volpack context
    VoxelType* vox;                                         ///< voxel data of one animation frame
    uchar* image;                                           ///< output image data
    float  current_percentage;                              ///< for clipping plane
    bool   stickyLights;                                    ///< true=lights moving with object
    uchar* boundedData;                                     ///< volume data with bounding box
    float  left, right, bottom, top, nearPlane, farPlane;   ///< projection parameters
    bool   timing;

    void   initialize();
    void   setImageSize(int, int);
    void   makeUnclassifiedVolume(uchar*);
    void   classifyVolume();
    void   updateModelviewMatrix();
    void   updateProjectionMatrix();
    void   renderVolume(int, int);

  public:
    vvRenderVP(vvVolDesc*, vvRenderState);
    virtual ~vvRenderVP();
    void     renderVolumeGL();
    void     renderVolumeRGB(int, int, uchar*);
    void     updateTransferFunction();
    void     setCurrentFrame(size_t);
    void     setLights(int, bool);
};

class VIRVOEXPORT vvVolPack: public vvSwitchRenderer<vvRenderVP, vvRenderer>
{
public:
  vvVolPack(vvVolDesc *vd, vvRenderState rs);
};
#endif // HAVE_VOLPACK

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
