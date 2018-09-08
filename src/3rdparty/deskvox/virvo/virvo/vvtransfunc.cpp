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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include <float.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "math/math.h"
#include "vvdebugmsg.h"
#include "vvtransfunc.h"
#include "vvcudatransfunc.h"
#include "vvvoldesc.h"
#include "vvtoolshed.h"

#include <algorithm>
#include <fstream>

using std::cerr;
using std::endl;
using std::list;

using virvo::vec3f;
using virvo::vec4i;


//----------------------------------------------------------------------------
/// Constructor
vvTransFunc::vvTransFunc()
{
  _nextBufferEntry = 0;
  _bufferUsed      = 0;
  _discreteColors  = 0;
}

// Copy Constructor
vvTransFunc::vvTransFunc(const vvTransFunc &tf)
{
  copy(&_widgets, &tf._widgets);

  _discreteColors  = tf._discreteColors;
  _bufferUsed      = 0;
  _nextBufferEntry = 0;
}

//----------------------------------------------------------------------------
/// Destructor
vvTransFunc::~vvTransFunc()
{
  clear();
}

vvTransFunc &vvTransFunc::operator=(vvTransFunc rhs)
{
  swap(rhs);
  return *this;
}

void vvTransFunc::swap(vvTransFunc &other)
{
  std::swap(_buffer, other._buffer);
  std::swap(_nextBufferEntry, other._nextBufferEntry);
  std::swap(_bufferUsed, other._bufferUsed);
  std::swap(_discreteColors, other._discreteColors);
  std::swap(_widgets, other._widgets);
}

//----------------------------------------------------------------------------
/** Delete all pins of given pin type from the list.
  @param wt widget type to delete
*/
void vvTransFunc::deleteWidgets(vvTFWidget::WidgetType wt)
{
  std::vector<vvTFWidget*>::iterator it = _widgets.begin();
  while (it != _widgets.end())
  {
    vvTFWidget* w = *it;
    if ((wt==vvTFWidget::TF_COLOR   && dynamic_cast<vvTFColor*>(w)) ||
      (wt==vvTFWidget::TF_PYRAMID   && dynamic_cast<vvTFPyramid*>(w)) ||
      (wt==vvTFWidget::TF_BELL      && dynamic_cast<vvTFBell*>(w)) ||
      (wt==vvTFWidget::TF_SKIP      && dynamic_cast<vvTFSkip*>(w)) ||
      (wt==vvTFWidget::TF_CUSTOM    && dynamic_cast<vvTFCustom*>(w)) ||
      (wt==vvTFWidget::TF_CUSTOM_2D && dynamic_cast<vvTFCustom2D*>(w)) ||
      (wt==vvTFWidget::TF_MAP       && dynamic_cast<vvTFCustomMap*>(w)))
    {
      it = _widgets.erase(it);
      delete w;
    }
    else
    {
      ++it;
    }
  }
}

//----------------------------------------------------------------------------
/** @return true if the transfer function contains no widgets.
 */
bool vvTransFunc::isEmpty()
{
  return _widgets.empty();
}

void vvTransFunc::clear()
{
    for (std::vector<vvTFWidget*>::const_iterator it = _widgets.begin();
         it != _widgets.end(); ++it)
    {
      delete *it;
    }
    _widgets.clear();
}

//----------------------------------------------------------------------------
/** Set default color values in the global color transfer function.
 All previous color widgets are deleted, other widgets are not affected.
 @param index color scheme
 @param min,max data range for color scheme
*/
void vvTransFunc::setDefaultColors(int index, float min, float max)
{
  deleteWidgets(vvTFWidget::TF_COLOR);
  switch (index)
  {
    case 0:                                       // bright colors
    default:
      // Set RGBA table to bright colors (range: blue->green->red):
      _widgets.push_back(new vvTFColor(vvColor(0.0f, 0.0f, 1.0f), min));
      _widgets.push_back(new vvTFColor(vvColor(0.0f, 1.0f, 1.0f), (max-min) * 0.33f + min));
      _widgets.push_back(new vvTFColor(vvColor(1.0f, 1.0f, 0.0f), (max-min) * 0.67f + min));
      _widgets.push_back(new vvTFColor(vvColor(1.0f, 0.0f, 0.0f), max));
      break;

    case 1:                                       // hue gradient
      // Set RGBA table to maximum intensity and value HSB colors:
      _widgets.push_back(new vvTFColor(vvColor(1.0f, 0.0f, 0.0f), min));
      _widgets.push_back(new vvTFColor(vvColor(1.0f, 1.0f, 0.0f), (max-min) * 0.2f + min));
      _widgets.push_back(new vvTFColor(vvColor(0.0f, 1.0f, 0.0f), (max-min) * 0.4f + min));
      _widgets.push_back(new vvTFColor(vvColor(0.0f, 1.0f, 1.0f), (max-min) * 0.6f + min));
      _widgets.push_back(new vvTFColor(vvColor(0.0f, 0.0f, 1.0f), (max-min) * 0.8f + min));
      _widgets.push_back(new vvTFColor(vvColor(1.0f, 0.0f, 1.0f), max));
      break;

    case 2:                                       // grayscale ramp
      // Set RGBA table to grayscale ramp (range: black->white).
      _widgets.push_back(new vvTFColor(vvColor(0.0f, 0.0f, 0.0f), min));
      _widgets.push_back(new vvTFColor(vvColor(1.0f, 1.0f, 1.0f), max));
      break;

    case 3:                                       // white
      // Set RGBA table to all white values:
      _widgets.push_back(new vvTFColor(vvColor(1.0f, 1.0f, 1.0f), min));
      _widgets.push_back(new vvTFColor(vvColor(1.0f, 1.0f, 1.0f), max));
      break;

    case 4:                                       // red ramp
      _widgets.push_back(new vvTFColor(vvColor(0.0f, 0.0f, 0.0f), min));
      _widgets.push_back(new vvTFColor(vvColor(1.0f, 0.0f, 0.0f), max));
      break;

    case 5:                                       // green ramp
      _widgets.push_back(new vvTFColor(vvColor(0.0f, 0.0f, 0.0f), min));
      _widgets.push_back(new vvTFColor(vvColor(0.0f, 1.0f, 0.0f), max));
      break;

    case 6:                                       // blue ramp
      _widgets.push_back(new vvTFColor(vvColor(0.0f, 0.0f, 0.0f), min));
      _widgets.push_back(new vvTFColor(vvColor(0.0f, 0.0f, 1.0f), max));
      break;

    case 7:                                       // cool to warm map
      // see http://www.cs.unm.edu/~kmorel/documents/ColorMaps/ColorMapsExpanded.pdf
      _widgets.push_back(new vvTFColor(vvColor(0.231f, 0.298f, 0.752f), min));
      _widgets.push_back(new vvTFColor(vvColor(0.552f, 0.690f, 0.996f), (max-min) * 0.25f + min));
      _widgets.push_back(new vvTFColor(vvColor(0.866f, 0.866f, 0.866f), (max-min) * 0.5f + min));
      _widgets.push_back(new vvTFColor(vvColor(0.956f, 0.603f, 0.486f), (max-min) * 0.75f + min));
      _widgets.push_back(new vvTFColor(vvColor(0.705f, 0.015f, 0.149f), max));
      break;

    case 8: {                                     // 'Fire' color table from ImageJ
#if 0
      const int r[] = {0, 0, 1, 25, 49, 73, 98,122,146,162,173,184,195,207,217,229,240,252,255,255,255,255,255,255,255,255,255,255,255,255,255,255};
      const int g[] = {0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 14, 35, 57, 79,101,117,133,147,161,175,190,205,219,234,248,255,255,255,255};
      const int b[] = {0,61,96,130,165,192,220,227,210,181,151,122, 93, 64, 35,  5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 35, 98,160,223,255};
      const int n = sizeof(r)/sizeof(r[0]);
      const float d = (max-min)/(n-1);
      float x = min;
#else
      const int r[] = {0, 0,122,195,255,255,255};
      const int g[] = {0, 0,  0,  0,117,234,255};
      const int b[] = {0,61,227, 93,  0,  0,255};
      const int k[] = {0, 1,  7, 12, 18, 26, 31}; 
      const int n = sizeof(r)/sizeof(r[0]);
      const float d = (max-min)/(k[n-1]);
#endif
      for (int i=0; i<n; ++i) {
        _widgets.push_back(new vvTFColor(vvColor(r[i]/255.f, g[i]/255.f, b[i]/255.f), d*k[i]));
      }
      break;
    }
  }
}

//----------------------------------------------------------------------------
/// Returns the number of default color schemes.
int vvTransFunc::getNumDefaultColors()
{
  return 9;
}

//----------------------------------------------------------------------------
/** Set default alpha values in the transfer function.
 The previous alpha pins are deleted, color pins are not affected.
 @param min,max data range for alpha scheme
*/
void vvTransFunc::setDefaultAlpha(int index, float min, float max)
{
  vvDebugMsg::msg(2, "vvTransFunc::setDefaultAlpha()");

  deleteWidgets(vvTFWidget::TF_PYRAMID);
  deleteWidgets(vvTFWidget::TF_BELL);
  deleteWidgets(vvTFWidget::TF_CUSTOM);
  deleteWidgets(vvTFWidget::TF_CUSTOM_2D);
  deleteWidgets(vvTFWidget::TF_MAP);
  deleteWidgets(vvTFWidget::TF_SKIP);
  switch (index)
  {
    case 0:                                       // ascending (0->1)
    default:
      _widgets.push_back(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, max, 2.0f * (max-min), 0.0f));
      break;
    case 1:                                       // descending (1->0)
      _widgets.push_back(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, min, 2.0f * (max-min), 0.0f));
      break;
    case 2:                                       // opaque (all 1)
      _widgets.push_back(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, (max-min)/2.0f+min, max-min, max-min));
      break;
  }
}

//----------------------------------------------------------------------------
/// Returns the number of default alpha schemes.
int vvTransFunc::getNumDefaultAlpha()
{
  return 3;
}

//----------------------------------------------------------------------------
/** Calculate background color for pixel in TF space by interpolating
  in RGB color space. Only TF_COLOR widgets contribute to this color.
  Currently, only the x coordinate of color widgets is considered,
  so the resulting color field varies only along the x axis.
*/
vvColor vvTransFunc::computeBGColor(float x, float, float) const
{
  vvColor col;
  vvTFColor* wBefore = NULL;
  vvTFColor* wAfter = NULL;
  vvTFColor* cw;

  for (std::vector<vvTFWidget*>::const_iterator it = _widgets.begin();
       it != _widgets.end(); ++it)
  {
    vvTFWidget* w = *it;
    if ((cw = dynamic_cast<vvTFColor*>(w)) != NULL)
    {
      if (cw->_pos[0] <= x)
      {
        if (wBefore==NULL || wBefore->_pos[0] < cw->_pos[0]) wBefore = cw;
      }
      if (cw->_pos[0] > x)
      {
        if (wAfter==NULL || wAfter->_pos[0] > cw->_pos[0]) wAfter = cw;
      }
    }
  }

  if (wBefore==NULL && wAfter==NULL) return col;
  if (wBefore==NULL) col = wAfter->_col;
  else if (wAfter==NULL) col = wBefore->_col;
  else
  {
    for (int c=0; c<3; ++c)
    {
      col[c] = vvToolshed::interpolateLinear(wBefore->_pos[0], wBefore->_col[c], wAfter->_pos[0], wAfter->_col[c], x);
    }
  }
  return col;
}

//----------------------------------------------------------------------------
/** Compute the color of a point in transfer function space. By definition
  the color is copied from the first non-TF_COLOR widget found. If no
  non-TF_COLOR widget is found, the point is colored according to the
  background color.
*/
vvColor vvTransFunc::computeColor(float x, float y, float z) const
{
  vvColor col;
  vvColor resultCol(0,0,0);
  int currentRange;
  float rangeWidth;
  bool hasOwn = false;

  if (_discreteColors>0)
  {
    rangeWidth = 1.0f / _discreteColors;
    currentRange = int(x * _discreteColors);
    if (currentRange >= _discreteColors)          // constrain range to valid ranges
    {
      currentRange = _discreteColors - 1;
    }
    x = currentRange * rangeWidth + (rangeWidth / 2.0f);
  }

  for (std::vector<vvTFWidget*>::const_iterator it = _widgets.begin();
       it != _widgets.end(); ++it)
  {
    vvTFWidget* w = *it;
    if (vvTFPyramid *pw = dynamic_cast<vvTFPyramid*>(w))
    {
      if (pw->hasOwnColor() && pw->getColor(col, x, y, z))
      {
        hasOwn = true;
        resultCol = resultCol + col;
      }
    }
    else if (vvTFBell *bw = dynamic_cast<vvTFBell*>(w))
    {
      if (bw->hasOwnColor() && bw->getColor(col, x, y, z))
      {
        hasOwn = true;
        resultCol = resultCol + col;
      }
    }
    else if (vvTFCustom2D *cw = dynamic_cast<vvTFCustom2D*>(w))
    {
      if (cw->hasOwnColor() && cw->getColor(col, x, y, z))
      {
        hasOwn = true;
        resultCol = resultCol + col;
      }
    }
    else if (vvTFCustomMap *cmw = dynamic_cast<vvTFCustomMap*>(w))
    {
      if (cmw->hasOwnColor() && cmw->getColor(col, x, y, z))
      {
        hasOwn = true;
        resultCol = resultCol + col;
      }
    }
  }
  if (!hasOwn) resultCol = computeBGColor(x, y, z);
  return resultCol;
}

//----------------------------------------------------------------------------
/** Goes through all widgets and returns the highest opacity value any widget
  has at the point.
*/
float vvTransFunc::computeOpacity(float x, float y, float z) const
{
  float opacity = 0.0f;

  for (std::vector<vvTFWidget*>::const_iterator it = _widgets.begin();
       it != _widgets.end(); ++it)
  {
    vvTFWidget* w = *it;
    if (dynamic_cast<vvTFSkip*>(w) && w->getOpacity(x, y, z)==0.0f) return 0.0f;  // skip widget is dominant
    else opacity = ts_max(opacity, w->getOpacity(x, y, z));
  }
  return opacity;
}

//----------------------------------------------------------------------------
/** Discretize transfer function values and write to a float array.
 Order of components: RGBARGBARGBA...
 @param w,h,d  number of array entries in each dimension
 @param array  _allocated_ float array in which to store computed values [0..1]
               Space for w*h*d*4 float values must be provided.
 @param min,max min/max values to create texture for               
*/
void vvTransFunc::computeTFTexture(size_t w, size_t h, size_t d, float* array, 
  float minX, float maxX, float minY, float maxY, float minZ, float maxZ,
  vvToolshed::Format format) const
{
  assert(format == vvToolshed::VV_RGBA || format == vvToolshed::VV_ARGB || format == vvToolshed::VV_BGRA);

  vec4i mask;

  if (format == vvToolshed::VV_ARGB)
  {
    mask = vec4i(1, 2, 3, 0);
  }
  else if (format == vvToolshed::VV_RGBA)
  {
    mask = vec4i(0, 1, 2, 3);
  }
  else if (format == vvToolshed::VV_BGRA)
  {
    mask = vec4i(2, 1, 0, 3);
  }
  else
  {
    assert("unhandled case for format" == 0);
    mask = vec4i(0, 1, 2, 3);
  }

  vec3f norm;    // normalized 3D position
  int index = 0;
  for (size_t z=0; z<d; ++z)
  {
    norm[2] = (d==1) ? -1.0f : ((float(z) / float(d-1)) * (maxZ - minZ) + minZ);
    for (size_t y=0; y<h; ++y)
    {
      norm[1] = (h==1) ? -1.0f : ((float(y) / float(h-1)) * (maxY - minY) + minY);
      for (size_t x=0; x<w; ++x)
      {
        norm[0] = (float(x) / float(w-1)) * (maxX - minX) + minX;
        vvColor col = computeColor(norm[0], norm[1], norm[2]);
        array[index + mask[0]] = col[0];
        array[index + mask[1]] = col[1];
        array[index + mask[2]] = col[2];
        array[index + mask[3]] = computeOpacity(norm[0], norm[1], norm[2]);
        index += 4;
      }
    }
  }
}
// 1st channel in contiguous block; last for opacity channel
void vvTransFunc::computeTFTextureGamma(int w, float* dest, float minX, float maxX, 
										int numchan, float gamma[], float offset[])
{
  int index = 0;
  for (int c = 0; c < numchan+1; c++)
  {
	  for (int i=0; i<w; ++i)
	  {
		  float x = (float(i) / float(w-1)) * (maxX - minX) + minX;	  
		  dest[index++] = ts_clamp((1-offset[c])*powf(x, gamma[c])+offset[c], 0.0f, 1.0f);
	  }
  }
}

void vvTransFunc::computeTFTextureHighPass(int w, float* dest, float minX, float maxX, 
										int numchan, float cutoff[], float order[], float offset[])
{
  int index = 0;
  for (int c = 0; c < numchan+1; c++)
  {
	  for (int i=0; i<w; ++i)
	  {
		  float x = (float(i) / float(w-1)) * (maxX - minX) + minX;
		  float filter = 0.0f;
		  if (x != 0.0f) filter = 1.0f / (1 + powf(cutoff[c]/x, 2*order[c]));
		  dest[index++] = ts_clamp((1-offset[c])*filter+offset[c], 0.0f, 1.0f);
	  }
  }
}

void vvTransFunc::computeTFTextureHistCDF(int w, float* dest, float minX, float maxX, 
										int numchan, int frame, uint* histCDF, float gamma[], float offset[])
{
  int index = 0;
  
  //int alphaCDF[256];
  //memset(alphaCDF, 0, 256*sizeof(int));
  for (int c = 0; c < numchan; c++)
  {	
	  uint* hist = histCDF + (numchan*frame + c)*256;

	  float min = float(hist[0]), max = float(hist[w-1]);
	  for (int i=0; i<w; ++i)
	  {
		  //alphaCDF[i] += hist[i];
		  float x = (float(hist[i])-min)/(max-min);
		  dest[index++] = ts_clamp((1-offset[c])*x+offset[c], 0.0f, 1.0f); 
	  }
  }	

  for (int i=0; i<w; ++i)
  {
	  float x = (float(i) / float(w-1)) * (maxX - minX) + minX;
	  //x = ts_clamp((1-offset[numchan])*float(alphaCDF[i])/float(alphaCDF[w-1])+offset[numchan], 0.0f, 1.0f); 
	  dest[index++] = ts_clamp((1-offset[numchan])*powf(x, gamma[numchan])+offset[numchan], 0.0f, 1.0f);
	  //ts_clamp((1-offset[numchan]+1)* x + (offset[numchan]-1), 0.0f, 1.0f);
  }
}

//----------------------------------------------------------------------------
/** Returns RGBA texture values for a color preview bar consisting of 3 rows:
  top: just the color, ignores opacity
  middle: color and opacity
  bottom: opacity only as grayscale
  @param width  width of color bar [pixels]
  @param colors pointer to _allocated_ memory providing space for num x 3 x 4 bytes.
               Byte quadruple 0 will be considered to correspond with scalar
                value 0.0, quadruple num-1 will be considered to correspond with
                scalar value 1.0. The resulting RGBA values are stored in the
               following order: RGBARGBARGBA...
  @param min,max data range for which color bar is to be created. Use 0..1 for integer data types.
  @param invertAlpha Setting for opacity only color bar: false=high opacity is white; true=high opacity is black
  @param format VV_RGBA, VV_ARGB or VV_BGRA
*/
void vvTransFunc::makeColorBar(int width, uchar* colors, float min, float max, bool invertAlpha,
                               vvToolshed::Format format)
{
  assert(format == vvToolshed::VV_RGBA || format == vvToolshed::VV_ARGB || format == vvToolshed::VV_BGRA);
  assert(colors);

  // Compute color components:
  std::vector<float> rgba;                    // four values per pixel
  rgba.resize(width * 4);
  computeTFTexture(width, 1, 1, &rgba[0], min, max, 0.0f, 0.0f, 0.0f, 0.0f, format);

  vec4i mask;

  if (format == vvToolshed::VV_ARGB)
  {
    mask = vec4i(1, 0, 0, 0);
  }
  else if (format == vvToolshed::VV_RGBA)
  {
    mask = vec4i(0, 0, 0, 1);
  }
  else if (format == vvToolshed::VV_BGRA)
  {
    mask = vec4i(0, 0, 0, 1);
  }

  // Convert to uchar:
  for (int x=0; x<width; ++x)
  {
    for (size_t c=0; c<4; ++c)
    {
      size_t index = x * 4 + c;
      if (mask[c] == 0) colors[index] = uchar(rgba[index] * 255.0f);
      else colors[index] = (uchar)255;
      colors[index + width * 4] = uchar(rgba[index] * 255.0f);
      float alpha = rgba[x * 4 + 3];
      if (invertAlpha) alpha = 1.0f - alpha;
      colors[index + 2 * width * 4] = (mask[c] == 0) ? (uchar(alpha * 255.0f)) : 255;
    }
  }
}

//----------------------------------------------------------------------------
/** Returns texture values for the alpha function of the 1D transfer function.
 Order of components: RGBARGBARGBA... or ARGBARGBARGB... or BGRABGRABGRA...
 @param width,height size of texture [pixels]
 @param texture  _allocated_ array in which to store texture values.
                 Space for width*height*4 bytes must be provided.
 @param min,max data range for which alpha texture is to be created. Use 0..1 for integer data types.
 @param format VV_RGBA, VV_ARGB or VV_BGRA
*/
void vvTransFunc::makeAlphaTexture(int width, int height, uchar* texture, float min, float max,
  vvToolshed::Format format)
{
  assert(format == vvToolshed::VV_RGBA || format == vvToolshed::VV_ARGB || format == vvToolshed::VV_BGRA);

  const int RGBA = 4;
  const int GRAY_LEVEL = 160;
  const int ALPHA_LEVEL = 230;
  int x, y, index1D, index2D, barHeight;

  std::vector< float > rgba(width * RGBA);
  computeTFTexture(width, 1, 1, &rgba[0], min, max);
  memset(texture, 0, width * height * RGBA); // make black and transparent

  vec4i mask;

  if (format == vvToolshed::VV_ARGB)
  {
    mask = vec4i(1, 2, 3, 0);
  }
  else if (format == vvToolshed::VV_RGBA)
  {
    mask = vec4i(0, 1, 2, 3);
  }
  else if (format == vvToolshed::VV_BGRA)
  {
    mask = vec4i(2, 1, 0, 3);
  }
  else
  {
    assert("unhandled case for format" == 0);
    mask = vec4i(0, 1, 2, 3);
  }
  

  for (x=0; x<width; ++x)
  {
    index1D = RGBA * x + 3;                          // alpha component of TF
    barHeight = int(rgba[index1D] * float(height));
    for (y=0; y<barHeight; ++y)
    {
      index2D = RGBA * (x + (height - y - 1) * width);
      texture[index2D + mask[0]] = GRAY_LEVEL;
      texture[index2D + mask[1]] = GRAY_LEVEL;
      texture[index2D + mask[2]] = GRAY_LEVEL;
      texture[index2D + mask[3]] = ALPHA_LEVEL;
    }
  }
}

//----------------------------------------------------------------------------
/** Returns RGBA texture values for the 2D transfer function.
 Order of components: RGBARGBARGBA...
 @param width,height size of texture [pixels]
 @param texture  _allocated_ array in which to store texture values.
                 Space for width*height*4 bytes must be provided.
*/
void vvTransFunc::make2DTFTexture(int width, int height, uchar* texture, float minX, float maxX, float minY, float maxY)
{
  int x, y, index;

  std::vector< float > rgba(width * height * 4);
  computeTFTexture(width, height, 1, &rgba[0], minX, maxX, minY, maxY);

  for (y=0; y<height; ++y)
  {
    for (x=0; x<width; ++x)
    {
      index = 4 * (x + y * width);
      texture[index]     = uchar(rgba[index]     * 255.0f);
      texture[index + 1] = uchar(rgba[index + 1] * 255.0f);
      texture[index + 2] = uchar(rgba[index + 2] * 255.0f);
      texture[index + 3] = uchar(rgba[index + 3] * 255.0f);
    }
  }
}

//----------------------------------------------------------------------------
/** Returns BGRA texture values for the 2D transfer function.
 Order of components: BGRABGRABGRA...
 Texture is flipped along Y axis, to be displayed on windows managers (Qt)
 @param width,height size of texture [pixels]
 @param texture  _allocated_ array in which to store texture values.
                 Space for width*height*4 bytes must be provided.
*/
void vvTransFunc::make2DTFTexture2(int width, int height, uchar* texture, float minX, float maxX, float minY, float maxY)
{
  int x, y, index1, index2;

  float* rgba = new float[width * height * 4];
  computeTFTexture(width, height, 1, rgba, minX, maxX, minY, maxY);

  for (y=0; y<height; ++y)
  {
    for (x=0; x<width; ++x)
    {
      index1 = 4 * (x + y * width);
      index2 = 4 * (x + (height - 1 - y) * width);
      texture[index1]     = uchar(rgba[index2 + 2] * 255.0f);
      texture[index1 + 1] = uchar(rgba[index2 + 1] * 255.0f);
      texture[index1 + 2] = uchar(rgba[index2]     * 255.0f);
      texture[index1 + 3] = uchar(rgba[index2 + 3] * 255.0f);
    }
  }
  delete[] rgba;
}

//----------------------------------------------------------------------------
/** Create a look-up table of 8-bit integer values from current transfer
  function.
  @param width number of LUT entries (typically 256 or 4096, depending on bpv)
  @param lut     _allocated_ space with space for entries*4 bytes
*/
void vvTransFunc::make8bitLUT(int width, uchar* lut, float min, float max)
{
  float* rgba;                                    // temporary LUT in floating point format
  int i, c;

  rgba = new float[4 * width];

  // Generate arrays from pins:
  computeTFTexture(width, 1, 1, rgba, min, max);

  // Copy RGBA values to internal array:
  for (i=0; i<width; ++i)
  {
    for (c=0; c<4; ++c)
    {
      *lut = uchar(rgba[i * 4 + c] * 255.0f);
      ++lut;
    }
  }
  delete[] rgba;
}

//----------------------------------------------------------------------------
/** Make a deep copy of the widget list.
  @param dst destination list
  @param src source list
*/
void vvTransFunc::copy(std::vector<vvTFWidget*>* dst, const std::vector<vvTFWidget*>* src)
{
  for (std::vector<vvTFWidget*>::const_iterator it = dst->begin();
       it != dst->end(); ++it)
  {
    delete *it;
  }
  dst->clear();
  for (std::vector<vvTFWidget*>::const_iterator it = src->begin();
       it != src->end(); ++it)
  {
    vvTFWidget* w = *it;
    if (vvTFPyramid *pw = dynamic_cast<vvTFPyramid*>(w))
    {
      dst->push_back(new vvTFPyramid(pw));
    }
    else if (vvTFBell *bw = dynamic_cast<vvTFBell*>(w))
    {
      dst->push_back(new vvTFBell(bw));
    }
    else if (vvTFColor *cw = dynamic_cast<vvTFColor*>(w))
    {
      dst->push_back(new vvTFColor(cw));
    }
    else if (vvTFCustom *cuw = dynamic_cast<vvTFCustom*>(w))
    {
      dst->push_back(new vvTFCustom(cuw));
    }
    else if (vvTFSkip *sw = dynamic_cast<vvTFSkip*>(w))
    {
      dst->push_back(new vvTFSkip(sw));
    }
    else if (vvTFCustomMap *cmw = dynamic_cast<vvTFCustomMap*>(w))
    {
      dst->push_back(new vvTFCustomMap(cmw));
    }
    else if (vvTFCustom2D *c2w = dynamic_cast<vvTFCustom2D*>(w))
    {
      dst->push_back(new vvTFCustom2D(c2w));
    }
    else assert(0);
  }
}

//----------------------------------------------------------------------------
/// Store the current pin list in the undo ring buffer.
void vvTransFunc::putUndoBuffer()
{
  copy(&_buffer[_nextBufferEntry], &_widgets);
  if (_bufferUsed < BUFFER_SIZE) ++_bufferUsed;
  if (_nextBufferEntry < BUFFER_SIZE-1) ++_nextBufferEntry;
  else _nextBufferEntry = 0;
}

//----------------------------------------------------------------------------
/** Restore the latest element from the undo ring buffer to the current pin list.
  If the ring buffer is empty, nothing happens.
*/
void vvTransFunc::getUndoBuffer()
{
  int bufferEntry;

  if (_bufferUsed==0) return;                     // ring buffer is empty
  if (_nextBufferEntry > 0) bufferEntry = _nextBufferEntry - 1;
  else bufferEntry = BUFFER_SIZE - 1;
  _widgets = _buffer[bufferEntry];
  _buffer[bufferEntry].clear();
  _nextBufferEntry = bufferEntry;
  --_bufferUsed;
}

//----------------------------------------------------------------------------
/// Clear the undo ring buffer.
void vvTransFunc::clearUndoBuffer()
{
  while (_bufferUsed > 0) {
    --_bufferUsed;
    int bufferEntry = BUFFER_SIZE - 1;
    if (_nextBufferEntry > 0)
      bufferEntry = _nextBufferEntry - 1;
    for (size_t i=0; i<_buffer[bufferEntry].size(); ++i) {
      vvTFWidget *w = _buffer[bufferEntry][i];
      delete w;
    }
    _buffer[bufferEntry].clear();
  }

  _bufferUsed      = 0;
  _nextBufferEntry = 0;
}

//----------------------------------------------------------------------------
/** Set the number of discrete colors to use for color interpolation.
  @param numColors number of discrete colors (use 0 for smooth colors)
*/
void vvTransFunc::setDiscreteColors(int numColors)
{
  assert(numColors >= 0);
  _discreteColors = numColors;
}

//----------------------------------------------------------------------------
/** @return the number of discrete colors used for color interpolation.
            0 means smooth colors.
*/
int vvTransFunc::getDiscreteColors() const
{
  return _discreteColors;
}

//----------------------------------------------------------------------------
/** Creates the look-up table for pre-integrated rendering.
  This version of the code runs rather slow compared to
  makeLookupTextureOptimized because it does a correct applications of
  the volume rendering integral.
  This method is
 * Copyright (C) 2001  Klaus Engel   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
@author Klaus Engel

I would like to thank Martin Kraus who helped in the adaptation
of the pre-integration method to Virvo.
@param thickness  distance of two volume slices in the direction
of the principal viewing axis (defaults to 1.0)
*/
void vvTransFunc::makePreintLUTCorrect(int width, uchar *preIntTable, float thickness, float min, float max)
{
  vvDebugMsg::msg(1, "vvTransFunc::makePreintLUTCorrect()");

  // Generate arrays from pins:
  float *rgba = new float[width * 4 + 4];
  computeTFTexture(width, 1, 1, rgba, min, max);
  rgba[width*4] = rgba[(width-1)*4];
  rgba[width*4+1] = rgba[(width-1)*4+1];
  rgba[width*4+2] = rgba[(width-1)*4+2];
  rgba[width*4+3] = rgba[(width-1)*4+3];

#if VV_HAVE_CUDA
  if(!makePreintLUTCorrectCuda(width, preIntTable, thickness, min, max, rgba))
#endif
  {
  const int minLookupSteps = 2;
  const int addLookupSteps = 1;

  // cerr << "Calculating dependent texture - Please wait ...";
  vvToolshed::initProgress(width);
  for (int sb=0;sb<width;sb++)
  {
    for (int sf=0;sf<width;sf++)
    {
      int n=minLookupSteps+addLookupSteps*abs(sb-sf);
      double stepWidth = 1./n;
      double r=0., g=0., b=0., tau=0.;
      for (int i=0;i<n;i++)
      {
        const double s = sf+(sb-sf)*(double)i/n;
        const int is = (int)s;
        const double fract_s = s-floor(s);
        const double tauc = thickness*stepWidth*(rgba[is*4+3]*fract_s+rgba[(is+1)*4+3]*(1.0-fract_s));
        const double e_tau = exp(-tau);
#ifdef STANDARD
        /* standard optical model: r,g,b densities are multiplied with opacity density */
        const double rc = e_tau*tauc*(rgba[is*4+0]*fract_s+rgba[(is+1)*4+0]*(1.0-fract_s));
        const double gc = e_tau*tauc*(rgba[is*4+1]*fract_s+rgba[(is+1)*4+1]*(1.0-fract_s));
        const double bc = e_tau*tauc*(rgba[is*4+2]*fract_s+rgba[(is+1)*4+2]*(1.0-fract_s));

#else
        /* Willhelms, Van Gelder optical model: r,g,b densities are not multiplied */
        const double rc = e_tau*stepWidth*(rgba[is*4+0]*fract_s+rgba[(is+1)*4+0]*(1.0-fract_s));
        const double gc = e_tau*stepWidth*(rgba[is*4+1]*fract_s+rgba[(is+1)*4+1]*(1.0-fract_s));
        const double bc = e_tau*stepWidth*(rgba[is*4+2]*fract_s+rgba[(is+1)*4+2]*(1.0-fract_s));
#endif

        r = r+rc;
        g = g+gc;
        b = b+bc;
        tau = tau + tauc;
      }
      if (r>1.)
        r = 1.;
      preIntTable[sf*width*4+sb*4+0] = uchar(r*255.99);
      if (g>1.)
        g = 1.;
      preIntTable[sf*width*4+sb*4+1] = uchar(g*255.99);
      if (b>1.)
        b = 1.;
      preIntTable[sf*width*4+sb*4+2] = uchar(b*255.99);
      preIntTable[sf*width*4+sb*4+3] = uchar((1.- exp(-tau))*255.99);
    }
    vvToolshed::printProgress(sb);
  }
  // cerr << "done." << endl;
  }
  delete[] rgba;
}

//----------------------------------------------------------------------------
/** Creates the look-up table for pre-integrated rendering.
  This version of the code runs much faster than makeLookupTextureCorrect
  due to some minor simplifications of the volume rendering integral.
  This method is
 * Copyright (C) 2001  Klaus Engel   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
@author Klaus Engel

I would like to thank Martin Kraus who helped in the adaptation
of the pre-integration method to Virvo.
@param thickness  distance of two volume slices in the direction
of the principal viewing axis (defaults to 1.0)
*/
void vvTransFunc::makePreintLUTOptimized(int width, uchar *preIntTable, float thickness, float min, float max)
{
  float *rInt = new float[width];
  float *gInt = new float[width];
  float *bInt = new float[width];
  float *aInt = new float[width];

  vvDebugMsg::msg(1, "vvTransFunc::makePreintLUTOptimized()");

  // Generate arrays from pins:
  float *rgba = new float[width * 4];
  computeTFTexture(width, 1, 1, rgba, min, max);

  // cerr << "Calculating optimized dependent texture" << endl;
  int rcol=0, gcol=0, bcol=0, acol=0;
  rInt[0] = 0.f;
  gInt[0] = 0.f;
  bInt[0] = 0.f;
  aInt[0] = 0.f;
  preIntTable[0] = int(rgba[0]);
  preIntTable[1] = int(rgba[1]);
  preIntTable[2] = int(rgba[2]);
  preIntTable[3] = int((1.f - expf(-rgba[3]*thickness)) * 255.99f);
  for (int i=1;i<width;i++)
  {
#ifdef STANDARD
    /* standard optical model: r,g,b densities are multiplied with opacity density */
    // accumulated values
    float tauc = (int(rgba[(i-1)*4+3]) + int(rgba[i*4+3])) * .5f;
    rInt[i] = rInt[i-1] + (int(255.99f*rgba[(i-1)*4+0]) + int(255.99f*rgba[i*4+0])) * .5f * tauc;
    gInt[i] = gInt[i-1] + (int(255.99f*rgba[(i-1)*4+1]) + int(255.99f*rgba[i*4+1])) * .5f * tauc;
    bInt[i] = bInt[i-1] + (int(255.99f*rgba[(i-1)*4+2]) + int(255.99f*rgba[i*4+2])) * .5f * tauc;
    aInt[i] = aInt[i-1] + tauc;

    // diagonal for lookup texture
    rcol = int(rgba[i*4+0] * rgba[i*4+3] * thickness * 255.99f);
    gcol = int(rgba[i*4+1] * rgba[i*4+3] * thickness * 255.99f);
    bcol = int(rgba[i*4+2] * rgba[i*4+3] * thickness * 255.99f);
    acol = int((1.f - expf(- rgba[i*4+3] * thickness)) * 255.99f);
#else
    /* Willhelms, Van Gelder optical model: r,g,b densities are not multiplied */
    // accumulated values
    rInt[i] = rInt[i-1] + (rgba[(i-1)*4+0] + rgba[i*4+0]) * .5f * 255;
    gInt[i] = gInt[i-1] + (rgba[(i-1)*4+1] + rgba[i*4+1]) * .5f * 255;
    bInt[i] = bInt[i-1] + (rgba[(i-1)*4+2] + rgba[i*4+2]) * .5f * 255;
    aInt[i] = aInt[i-1] + (rgba[(i-1)*4+3] + rgba[i*4+3]) * .5f;

    // diagonal for lookup texture
    rcol = int(255.99f*rgba[i*4+0]);
    gcol = int(255.99f*rgba[i*4+1]);
    bcol = int(255.99f*rgba[i*4+2]);
    acol = int((1.f - expf(-rgba[i*4+3] * thickness)) * 255.99f);
#endif

    preIntTable[i*width*4+i*4+0] = uchar(rcol);
    preIntTable[i*width*4+i*4+1] = uchar(gcol);
    preIntTable[i*width*4+i*4+2] = uchar(bcol);
    preIntTable[i*width*4+i*4+3] = uchar(acol);
  }

  for (int sb=0;sb<width;sb++)
  {
    for (int sf=0;sf<sb;sf++)
    {
      bool opaque = false;
      for (int s = sf; s <= sb; s++)
      {
        if (rgba[s*4+3] >= .996f)
        {
          rcol = int(rgba[s*4+0]*255.99f);
          gcol = int(rgba[s*4+1]*255.99f);
          bcol = int(rgba[s*4+2]*255.99f);
          acol = int(255);
          opaque = true;
          break;
        }
      }

      if(opaque)
      {
        preIntTable[sb*width*4+sf*4+0] = uchar(rcol);
        preIntTable[sb*width*4+sf*4+1] = uchar(gcol);
        preIntTable[sb*width*4+sf*4+2] = uchar(bcol);
        preIntTable[sb*width*4+sf*4+3] = uchar(acol);

        for (int s = sb; s >= sf; s--)
        {
          if (rgba[s*4+3] >= .996f)
          {
            rcol = int(rgba[s*4+0]*255.99f);
            gcol = int(rgba[s*4+1]*255.99f);
            bcol = int(rgba[s*4+2]*255.99f);
            acol = int(255);
            break;
          }
        }
        preIntTable[sf*width*4+sb*4+0] = uchar(rcol);
        preIntTable[sf*width*4+sb*4+1] = uchar(gcol);
        preIntTable[sf*width*4+sb*4+2] = uchar(bcol);
        preIntTable[sf*width*4+sb*4+3] = uchar(acol);
        continue;
      }

      float scale = 1.f/(sb-sf);
      rcol = int((rInt[sb] - rInt[sf])*scale);
      gcol = int((gInt[sb] - gInt[sf])*scale);
      bcol = int((bInt[sb] - bInt[sf])*scale);
      acol = int((1.f - expf(-(aInt[sb]-aInt[sf])*scale * thickness)) * 255.99f);

      if (rcol > 255)
        rcol = 255;
      if (gcol > 255)
        gcol = 255;
      if (bcol > 255)
        bcol = 255;
      if (acol > 255)
        acol = 255;

      preIntTable[sf*width*4+sb*4+0] = uchar(rcol);
      preIntTable[sf*width*4+sb*4+1] = uchar(gcol);
      preIntTable[sf*width*4+sb*4+2] = uchar(bcol);
      preIntTable[sf*width*4+sb*4+3] = uchar(acol);

      preIntTable[sb*width*4+sf*4+0] = uchar(rcol);
      preIntTable[sb*width*4+sf*4+1] = uchar(gcol);
      preIntTable[sb*width*4+sf*4+2] = uchar(bcol);
      preIntTable[sb*width*4+sf*4+3] = uchar(acol);

#if 0
      if (sb%16==0 && sf%16==0)
      {
        std::cerr << "preIntTable(" << sf << "," << sb << ") = ("
          << int(preIntTable[sf*width*4+sb*4+0]) << " "
          << int(preIntTable[sf*width*4+sb*4+1]) << " "
          << int(preIntTable[sf*width*4+sb*4+2]) << " "
          << int(preIntTable[sf*width*4+sb*4+3]) << ")" << std::endl;
      }
#endif
    }
  }

  delete[] rInt;
  delete[] gInt;
  delete[] bInt;
  delete[] aInt;
}

/** Save transfer function to ascii file
 */
bool vvTransFunc::save(const std::string& filename)
{
  std::ofstream file;
  file.open(filename.c_str(), std::ios::out);

  if (file.is_open())
  {
    for (std::vector<vvTFWidget*>::const_iterator it = _widgets.begin();
         it != _widgets.end(); ++it)
    {
      file << (*it)->toString();
    }
    file.close();
  }
  else
  {
    return false;
  }
  return true;
}

bool vvTransFunc::load(const std::string& filename)
{
  std::ifstream file;
  file.open(filename.c_str(), std::ios::in);

  if (file.is_open())
  {
    clear();

    std::string line;
    while (file.good())
    {
      getline(file, line);
      if (line.length() < 3)
      {
        continue;
      }

      std::vector<std::string> tokens = vvToolshed::split(line, " ");

      if (tokens.size() < 3)
      {
        continue;
      }
      const char* name = tokens[0].c_str();

      vvTFWidget* widget = vvTFWidget::produce(vvTFWidget::getWidgetType(name));

      if (widget)
      {
        widget->fromString(line);
        _widgets.push_back(widget);
      }
    }
    file.close();
    return true;
  }
  else
  {
    return false;
  }
}

//----------------------------------------------------------------------------
/** Save transfer function to a disk file in Meshviewer format:
  Example:
  <pre>
  ColorMapKnots: 3
  Knot:  0.0  1.0  0.0  0.0
  Knot: 50.0  0.0  1.0  0.0
  Knot: 99.0  0.0  0.0  1.0
  OpacityMapPoints: 3
  Point:  0.0   0.00
  Point: 50.0   0.05
  Point: 99.0   0.00

  Syntax:
  Knot: <float_data_value> <red 0..1> <green> <blue>
  Point: <float_data_value> <opacity 0..1>

  - only Color and Custom transfer function widgets are supported, not Bell or Pyramid!
  - numbers are floating point with any number of mantissa digits
  - '#' allowed for comments
  </pre>
  @return 1 if successful, 0 if not
*/
int vvTransFunc::saveMeshviewer(const char* filename)
{
  vvTFColor* cw;
  vvTFCustom* cuw;
  FILE* fp;

  if ( (fp = fopen(filename, "wb")) == NULL)
  {
    cerr << "Error: Cannot create file." << endl;
    return 0;
  }
  
  // Write color pins to file:
  fprintf(fp, "ColorMapKnots: %d\n", getNumWidgets(vvTFWidget::TF_COLOR));
  for (std::vector<vvTFWidget*>::const_iterator it = _widgets.begin();
       it != _widgets.end(); ++it)
  {
    vvTFWidget* w = *it;
    if ((cw=dynamic_cast<vvTFColor*>(w)))
    {
      fprintf(fp, "Knot: %f %f %f %f\n", cw->_pos[0], cw->_col[0], cw->_col[1], cw->_col[2]);
    }
  }
  
  // Write opacity pins to file:
  for (std::vector<vvTFWidget*>::const_iterator it = _widgets.begin();
       it != _widgets.end(); ++it)
  {
    vvTFWidget* w = *it;
    if ((cuw=dynamic_cast<vvTFCustom*>(w)))
    {
      fprintf(fp, "OpacityMapPoints: %d\n", (int)cuw->_points.size() + 2);   // add two points for edges of TF space
      fprintf(fp, "Point: %f %f\n", cuw->_pos[0] - cuw->_size[0]/2.0f, 0.0f);
      list<vvTFPoint*>::iterator iter;
      for(iter=cuw->_points.begin(); iter!=cuw->_points.end(); iter++) 
      {
        fprintf(fp, "Point: %f %f\n", (*iter)->_pos[0] + cuw->_pos[0], (*iter)->_opacity);
      }
      fprintf(fp, "Point: %f %f\n", cuw->_pos[0] + cuw->_size[0]/2.0f, 0.0f);
    }
  }

  // Wrap up:
  fclose(fp);
  cerr << "Wrote transfer function file: " << filename << endl;
  return 1;
}

int vvTransFunc::saveBinMeshviewer(const char* filename)
{
  vvColor col;
  float normX;    
  unsigned int index;
  const size_t nBins = vvVolDesc::NUM_HDR_BINS;
  FILE* fp;

  // open file
  if ( (fp = fopen(filename, "wb")) == NULL)
  {
	cerr << "Error: Cannot create file." << endl;
	return 0;
  }

  float* rgba = new float[nBins * 4];
  computeTFTexture(nBins, 1, 1, rgba, 0.0, 1.0);

  // save color
  fprintf(fp, "BinKnots: %d\n", (uint32_t)nBins);
  for(size_t x = 0; x < nBins; x++)
  {
	normX = float(x) / float(nBins-1);
	index = x * 4;
	fprintf(fp, "Knot: %f %f %f %f\n", normX, rgba[index], rgba[index+1], rgba[index+2]);
  }

  // save opacity
  fprintf(fp, "OpacityMapPoints: %d\n", (uint32_t)nBins);   // add two points for edges of TF space

  for (size_t x=0; x< nBins; ++x)
  {
	normX = (float(x) / float(nBins-1));
	index = x * 4 + 3;
	fprintf(fp, "Point: %f %f\n", normX, rgba[index]);
  }

  delete[] rgba;
        
  // Wrap up:
  fclose(fp);
  cerr << "Wrote transfer function file: " << filename << endl;
  return 1;
}

//----------------------------------------------------------------------------
/** Load transfer function from a disk file in Meshviewer format.
  @see vvTransFunc::saveMeshViewer
  @return 1 if successful, 0 if not
*/
int vvTransFunc::loadMeshviewer(const char* filename)
{
  vvTFColor* cw;
  vvTFCustom* cuw;
  FILE* fp;
  int i;
  int numColorWidgets, numOpacityPoints;
  float pos, col[3], opacity;

  if ( (fp = fopen(filename, "rb")) == NULL)
  {
    cerr << "Error: Cannot open file." << endl;
    return 0;
  }
  
  // Remove all existing widgets:
  for (std::vector<vvTFWidget*>::const_iterator it = _widgets.begin();
       it != _widgets.end(); ++it)
  {
    delete *it;
  }
  _widgets.clear();
  
  // Read color pins from file:
  if(fscanf(fp, "ColorMapKnots: %d\n", &numColorWidgets) != 1)
  {
    std::cerr << "vvTransFunc::loadMeshviewer: fscanf 1 failed" << std::endl;
    numColorWidgets = 0;
  }

  for (i=0; i<numColorWidgets; ++i)
  { 
    if(fscanf(fp, "Knot: %f %f %f %f\n", &pos, &col[0], &col[1], &col[2]) != 4)
       std::cerr << "vvTransFunc::loadMeshviewer: fscanf 2 failed" << std::endl;
    cw = new vvTFColor();
    cw->_pos[0] = pos;
    cw->_col[0] = col[0];
    cw->_col[1] = col[1];
    cw->_col[2] = col[2];
    _widgets.push_back(cw);
  }
  
  // Read opacity pins from file:
  if(fscanf(fp, "OpacityMapPoints: %d\n", &numOpacityPoints) != 1)
  {
    std::cerr << "vvTransFunc::loadMeshviewer: fscanf 3 failed" << std::endl;
    numOpacityPoints = 0;
  }

  if (numOpacityPoints>0) 
  {
    float begin=0., end=0.;
    cuw = new vvTFCustom(0.5f, 1.0f);
    _widgets.push_back(cuw);
    for (i=0; i<numOpacityPoints; ++i)
    { 
      if(fscanf(fp, "Point: %f %f\n", &pos, &opacity) != 2)
         std::cerr << "vvTransFunc::loadMeshviewer: fscanf 4 failed" << std::endl;
      if (i>0 && i<numOpacityPoints-1)  // skip start and end point (will be determined by widget position and width)
      {
        cuw->_points.push_back(new vvTFPoint(opacity, pos));
      }
      else 
      {
        if (i==0) begin = pos;
        else if (i==numOpacityPoints-1) end = pos;
      }
    }
    
    // Adjust widget size:
    cuw->_size[0] = end - begin;
    cuw->_pos[0] = (begin + end) / 2.0f;

    // Adjust point positions:
    list<vvTFPoint*>::iterator iter;
    for(iter=cuw->_points.begin(); iter!=cuw->_points.end(); iter++) 
    {
      (*iter)->_pos[0] -= cuw->_pos[0];
    }
  }
  
  // Wrap up:
  fclose(fp);
  cerr << "Loaded transfer function from file: " << filename << endl;
  return 1;
}

//----------------------------------------------------------------------------
/** @return the number of widgets of a given type
*/
int vvTransFunc::getNumWidgets(vvTFWidget::WidgetType wt)
{
  int num = 0;
  
  for (std::vector<vvTFWidget*>::const_iterator it = _widgets.begin();
       it != _widgets.end(); ++it)
  {
    vvTFWidget* w = *it;
    switch(wt)
    {
      case vvTFWidget::TF_COLOR:   if (dynamic_cast<vvTFColor*>(w))   ++num; break;
      case vvTFWidget::TF_PYRAMID: if (dynamic_cast<vvTFPyramid*>(w)) ++num; break;
      case vvTFWidget::TF_BELL:    if (dynamic_cast<vvTFBell*>(w))    ++num; break;
      case vvTFWidget::TF_SKIP:    if (dynamic_cast<vvTFSkip*>(w))    ++num; break;
      case vvTFWidget::TF_CUSTOM:  if (dynamic_cast<vvTFCustom*>(w))  ++num; break;

      case vvTFWidget::TF_CUSTOM_2D: if (dynamic_cast<vvTFCustom2D*>(w))  ++num; break;
      case vvTFWidget::TF_MAP:       if (dynamic_cast<vvTFCustomMap*>(w)) ++num; break;

      case vvTFWidget::TF_UNKNOWN: break;
    }
  }
  return num;
}

//----------------------------------------------------------------------------
/** Compute a min/max table for the transfer function 
 * the table contains the maximum opacity for each interval
 @param width size of one edge of the allocated 2d minmax array
 @param minmax 2d array (sized width^2) allocated by caller
 @param min,max data range for color scheme
*/
void vvTransFunc::makeMinMaxTable(int width, uchar *minmax, float min, float max)
{
  // Generate arrays from pins:
  float *rgba = new float[width * 4];
  computeTFTexture(width, 1, 1, rgba, min, max);

  for(int mn=0; mn<width; ++mn)
  {
    for(int mx=0; mx<width; ++mx)
    {
      float op = 0.f;
      for(int i=mn; i<=mx; ++i)
      {
        if(rgba[i*4+3] > op)
          op = rgba[i*4+3];
      }
      minmax[mn+width*mx] = static_cast<uchar>(op*255.99f);
    }
  }

  delete[] rgba;
}

//----------------------------------------------------------------------------
/** @set the transfer function type to LUT_1D and set LUT
 */
/*
void vvTransFunc::setLUT(int numEntries, const uchar *rgba)
{
   vvDebugMsg::msg(1, "vvTransFunc::setLUT()");
   lutEntries = numEntries;
   type = LUT_1D;
   delete[] rgbaLUT;
   rgbaLUT = new uchar[4*lutEntries];
   memcpy(rgbaLUT, rgba, 4*lutEntries);
}
*/
//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
