// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
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

#ifndef VV_STINGRAY_H
#define VV_STINGRAY_H

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

// Stingray:
#ifdef USE_STINGRAY
#include <StingRayCaveRenderer.hpp>
#endif

// Virvo:
#include "vvexport.h"
#include "vvvoldesc.h"
#include "vvrenderer.h"

//============================================================================
// Class Definitions
//============================================================================

/** Volume rendering engine using Peter Stephenson's Stingray volume
  rendering algorithm which is based on raytracing.
  @author Jurgen Schulze (schulze@cs.brown.de)
  @see vvRenderer
*/
class VIRVOEXPORT vvStingray : public vvRenderer
{
  public:

  private:
    float* rgbaTF;                                ///< density to RGBA conversion table, as created by TF [0..1]
    uchar* rgbaLUT;                               ///< final RGBA conversion table, as transferred to graphics hardware (includes opacity and gamma correction)
    bool interpolation;                           ///< interpolation mode: true=linear interpolation (default), false=nearest neighbor
    int  mip;                                     ///< min/maximum intensity projection (0=off, 1=max, 2=min)
    bool gammaCorrection;                         ///< true = gamma correction on
#ifdef USE_STINGRAY
    fvStingRayCaveRenderer* _stingRay;            ///< instance of StingRay renderer
#endif
    // GL state variables:
    GLboolean glsCulling;                         ///< stores GL_CULL_FACE
    GLboolean glsBlend;                           ///< stores GL_BLEND
    GLboolean glsColorMaterial;                   ///< stores GL_COLOR_MATERIAL
    GLint glsBlendSrc;                            ///< stores glBlendFunc(source,...)
    GLint glsBlendDst;                            ///< stores glBlendFunc(...,destination)
    GLboolean glsTexColTable;                     ///< stores GL_TEXTURE_COLOR_TABLE_SGI
    GLboolean glsSharedTexPal;                    ///< stores GL_SHARED_TEXTURE_PALETTE_EXT
    GLboolean glsLighting;                        ///< stores GL_LIGHTING
    GLint glsMatrixMode;                          ///< stores GL_MATRIX_MODE
    GLint glsBlendEquation;                       ///< stores GL_BLEND_EQUATION_EXT
    GLboolean glsDepthTest;                       ///< stores GL_DEPTH_TEST
    GLint glsDepthFunc;                           ///< stores glDepthFunc

    void setGLenvironment();
    void unsetGLenvironment();
    void updateLUT(float);

  public:
    vvStingray(vvVolDesc*, vvRenderState);
    virtual ~vvStingray();
    void  renderVolumeGL();
    void  updateTransferFunction();
    bool  instantClassification() const;
};
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
