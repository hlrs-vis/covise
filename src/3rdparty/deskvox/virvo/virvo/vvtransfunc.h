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

#ifndef VV_TRANSFUNC_H
#define VV_TRANSFUNC_H

#include "vvexport.h"
#include "vvinttypes.h"
#include "vvtfwidget.h"
#include "vvtoolshed.h"

#include <vector>

/** Description of a transfer function.
  @author Jurgen P. Schulze (jschulze@ucsd.edu)
  @see vvTFWidget
*/
class VIRVO_FILEIOEXPORT vvTransFunc
{
  private:
    enum                                           /// number of elements in ring buffer
    {
      BUFFER_SIZE = 20
    };
    std::vector<vvTFWidget*> _buffer[BUFFER_SIZE]; ///< ring buffer which can be used to implement Undo functionality
    int _nextBufferEntry;                          ///< index of next ring buffer entry to use for storage
    int _bufferUsed;                               ///< number of ring buffer entries used
    int _discreteColors;                           ///< number of discrete colors to use for color interpolation (0 for smooth colors)

  public:
    std::vector<vvTFWidget*> _widgets;             ///< TF widget list

    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      a.template register_type<vvTFWidget>();
      a.template register_type<vvTFBell>();
      a.template register_type<vvTFColor>();
      a.template register_type<vvTFCustom>();
      a.template register_type<vvTFPyramid>();
      a.template register_type<vvTFSkip>();
      a & _widgets;
    }

    vvTransFunc();
    vvTransFunc(const vvTransFunc &other);
    virtual ~vvTransFunc();
    vvTransFunc &operator=(vvTransFunc rhs);
    void swap(vvTransFunc &other);
    bool isEmpty();
    void clear();
    void deleteColorWidgets();
    void setDefaultColors(int, float, float);
    int  getNumDefaultColors();
    void setDefaultAlpha(int, float, float);
    int  getNumDefaultAlpha();
    int  getNumWidgets(vvTFWidget::WidgetType);
    void deleteWidgets(vvTFWidget::WidgetType);
    void computeTFTexture(size_t w, size_t h, size_t d, float* array,
                          float minX, float maxX, float minY = 0.0f, float maxY = 0.0f,
                          float minZ = 0.0f, float maxZ = 0.0f, vvToolshed::Format format = vvToolshed::VV_RGBA) const;
    vvColor computeBGColor(float, float, float) const;
    void computeTFTextureGamma(int, float*, float, float, int, float[], float[]);
    void computeTFTextureHighPass(int, float*, float, float, int, float[], float[], float[]);
    void computeTFTextureHistCDF(int, float*, float, float, int, int, uint*, float[], float[]);
    vvColor computeColor(float, float=-1.0f, float=-1.0f) const;
    float computeOpacity(float, float=-1.0f, float=-1.0f) const;
    void makeColorBar(int width, uchar* colors, float min, float max, bool invertAlpha, vvToolshed::Format format = vvToolshed::VV_RGBA);
    void makeAlphaTexture(int, int, uchar*, float, float, vvToolshed::Format format = vvToolshed::VV_RGBA);
    void make2DTFTexture(int, int, uchar*, float, float, float, float);
    void make2DTFTexture2(int, int, uchar*, float, float, float, float);
    void make8bitLUT(int, uchar*, float, float);
    void makeFloatLUT(int, float*);
    void makePreintLUTOptimized(int width, uchar *preintLUT, float thickness=1.0, float min=0.0, float max=1.0);
    void makePreintLUTCorrect(int width, uchar *preintLUT, float thickness=1.0, float min=0.0, float max=1.0);
    void makeMinMaxTable(int width, uchar *minmax, float min=0.0, float max=1.0);
    static void copy(std::vector<vvTFWidget*>*, const std::vector<vvTFWidget *> *);
    void putUndoBuffer();
    void getUndoBuffer();
    void clearUndoBuffer();
    void setDiscreteColors(int);
    int  getDiscreteColors() const;
    bool save(const std::string& filename);
    bool load(const std::string& filename);
    int  saveMeshviewer(const char*);
    int  saveBinMeshviewer(const char*);
    int  loadMeshviewer(const char*);
};
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
