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

#ifndef VV_TFWIDGET_H
#define VV_TFWIDGET_H

// Boost:
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

// C++:
#include <stdio.h>
#include <fstream>
#include <list>
#include <stdexcept>
#include <string>

// Virvo:
#include "math/math.h"
#include "math/serialization.h"
#include "vvcolor.h"
#include "vvexport.h"
#include "vvinttypes.h"

/** Specifies a 3D point with an opacity.
  @see vvTFCustom
*/
class VIRVO_FILEIOEXPORT vvTFPoint
{
  public:
    virvo::vec3 _pos;
    float _opacity;   ///< opacity at this point in the TF [0..1]

    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      a & BOOST_SERIALIZATION_NVP(_pos);
      a & BOOST_SERIALIZATION_NVP(_opacity);
    }

    vvTFPoint();
    vvTFPoint(float, float, float=-1.0f, float=-1.0f);

    void setPos(virvo::vec3 const& pos);
    void setOpacity(float opacity);
    virvo::vec3 pos() const;
    float opacity() const;
};

/** Base class of transfer function widgets.
  @author Jurgen P. Schulze (jschulze@ucsd.edu)
  @see vvTransFunc
*/
class VIRVO_FILEIOEXPORT vvTFWidget
{
  protected:
    static std::string const NO_NAME;
    std::string _name;                            ///< widget name (bone, soft tissue, etc)

  public:
    enum WidgetType
    {
      TF_COLOR,
      TF_PYRAMID,
      TF_BELL,
      TF_SKIP,
      TF_CUSTOM,

      TF_CUSTOM_2D,
      TF_MAP,

      TF_UNKNOWN
    };
    static const int MAX_STR_LEN = 65535;

    virvo::vec3 _pos;                             ///< position of widget's center [volume data space]
    float _opacity;                               ///< maximum opacity [0..1]

    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      a & BOOST_SERIALIZATION_NVP(_name);
      a & BOOST_SERIALIZATION_NVP(_pos);
      a & BOOST_SERIALIZATION_NVP(_opacity);
    }

    vvTFWidget();
    vvTFWidget(float, float, float);
    vvTFWidget(vvTFWidget*);
    virtual ~vvTFWidget();

    void setOpacity(float opacity);
    float opacity() const;

    virtual void setName(std::string const& name);
    virtual std::string getName() const;
    void setPos(virvo::vec3 const& pos);
    void setPos(float x, float y, float z);
    virvo::vec3 pos() const;
    virtual void readName(std::ifstream& file);
    void write(FILE*);
    virtual std::string toString() const { throw std::runtime_error("not implemented"); }
    virtual void fromString(const std::string& /*str*/) { throw std::runtime_error("not implemented"); }
    virtual float getOpacity(float, float=-1.0f, float=-1.0f);
    virtual bool getColor(vvColor&, float, float=-1.0f, float=-1.0f);
    virtual void mapFrom01(float min, float max);
    virtual void mapTo01(float min, float max);

    static vvTFWidget* produce(WidgetType type);
    static WidgetType getWidgetType(const char* str);
};

/** Transfer function widget shaped like a Gaussian bell.
 */
class VIRVO_FILEIOEXPORT vvTFBell : public vvTFWidget
{
  protected:
    bool _ownColor;                               ///< true = use widget's own color for TF; false=use background color for TF

  public:
    vvColor _col;                                 ///< RGB color
    virvo::vec3 _size;                            ///< width, height, depth of bell's bounding box [volume data space]

    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      a.template register_type<vvTFWidget>();
      a & BOOST_SERIALIZATION_BASE_OBJECT_NVP(vvTFWidget);
      a & BOOST_SERIALIZATION_NVP(_ownColor);
      a & BOOST_SERIALIZATION_NVP(_col);
      a & BOOST_SERIALIZATION_NVP(_size);
    }

    vvTFBell();
    vvTFBell(vvTFBell*);
    vvTFBell(vvColor, bool, float, float, float, float=0.5f, float=1.0f, float=0.5f, float=1.0f);
    vvTFBell(std::ifstream& file);

    void setColor(const vvColor& col);
    void setSize(virvo::vec3 const& size);
    vvColor color() const;
    virvo::vec3 size() const;

    virtual std::string toString() const;
    virtual void fromString(const std::string& str);
    virtual bool getColor(vvColor&, float, float=-1.0f, float=-1.0f);
    virtual float getOpacity(float, float=-1.0f, float=-1.0f);
    virtual bool hasOwnColor();
    virtual void setOwnColor(bool);
};

/** Pyramid-shaped transfer function widget:
  the pyramid has four sides and its tip can be flat (frustum).
*/
class VIRVO_FILEIOEXPORT vvTFPyramid : public vvTFWidget
{
  protected:
    bool _ownColor;                               ///< true = use widget's own color for TF; false=use background color for TF

  public:
    vvColor _col;                                 ///< RGB color
    virvo::vec3 _top;                             ///< width at top [volume data space]
    virvo::vec3 _bottom;                          ///< width at bottom of pyramid [volume data space]

    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      a.template register_type<vvTFWidget>();
      a & BOOST_SERIALIZATION_BASE_OBJECT_NVP(vvTFWidget);
      a & BOOST_SERIALIZATION_NVP(_ownColor);
      a & BOOST_SERIALIZATION_NVP(_col);
      a & BOOST_SERIALIZATION_NVP(_top);
      a & BOOST_SERIALIZATION_NVP(_bottom);
    }

    vvTFPyramid();
    vvTFPyramid(vvTFPyramid*);
    vvTFPyramid(vvColor, bool, float, float, float, float, float=0.5f, float=1.0f, float=0.0f, float=0.5f, float=1.0f, float=0.0f);
    vvTFPyramid(std::ifstream& file);

    void setColor(const vvColor& col);
    void setTop(virvo::vec3 const& top);
    void setBottom(virvo::vec3 const& bottom);
    vvColor color() const;
    virvo::vec3 top() const;
    virvo::vec3 bottom() const;

    virtual std::string toString() const;
    virtual void fromString(const std::string& str);
    virtual bool getColor(vvColor&, float, float=-1.0f, float=-1.0f);
    virtual float getOpacity(float, float=-1.0f, float=-1.0f);
    virtual bool hasOwnColor();
    virtual void setOwnColor(bool);
    virtual void mapFrom01(float min, float max);
    virtual void mapTo01(float min, float max);
};

/** Transfer function widget specifying a color point in TF space.
 */
class VIRVO_FILEIOEXPORT vvTFColor : public vvTFWidget
{
  public:
    void setColor(const vvColor& col);
    vvColor color() const;

    vvColor _col;                                 ///< RGB color

    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      a.template register_type<vvTFWidget>();
      a & BOOST_SERIALIZATION_BASE_OBJECT_NVP(vvTFWidget);
      a & BOOST_SERIALIZATION_NVP(_col);
    }

    vvTFColor();
    vvTFColor(vvTFColor*);
    vvTFColor(vvColor, float, float=0.0f, float=0.0f);
    vvTFColor(std::ifstream& file);
    virtual std::string toString() const;
    virtual void fromString(const std::string& str);
};

/** Transfer function widget to skip an area of the transfer function when rendering.
 */
class VIRVO_FILEIOEXPORT vvTFSkip : public vvTFWidget
{
  public:
    void setSize(virvo::vec3 const& size);
    virvo::vec3 size() const;

    virvo::vec3 _size;         ///< width, height, depth of skipped area [volume data space]

    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      a.template register_type<vvTFWidget>();
      a & BOOST_SERIALIZATION_BASE_OBJECT_NVP(vvTFWidget);
      a & BOOST_SERIALIZATION_NVP(_size);
    }

    vvTFSkip();
    vvTFSkip(vvTFSkip*);
    vvTFSkip(float, float, float=0.5f, float=0.0f, float=0.5f, float=0.0f);
    vvTFSkip(std::ifstream& file);
    virtual std::string toString() const;
    virtual void fromString(const std::string& str);
    virtual float getOpacity(float, float=-1.0f, float=-1.0f);
    virtual void mapFrom01(float min, float max);
    virtual void mapTo01(float min, float max);
};

/** Transfer function widget to specify a custom transfer function widget with control points.
  The widget defines a rectangular area in which the user can specify control points between
  which the opacity function will be computed linearly.
 */
class VIRVO_FILEIOEXPORT vvTFCustom : public vvTFWidget
{
  public:
    virvo::vec3 _size;       ///< width, height, depth of TF area [volume data space]
    std::list<vvTFPoint*> _points; ///< list of control points; coordinates are relative to widget center
    vvTFPoint* _currentPoint;      ///< currently selected point

    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      a.template register_type<vvTFWidget>();
      a & BOOST_SERIALIZATION_BASE_OBJECT_NVP(vvTFWidget);
      a & BOOST_SERIALIZATION_NVP(_size);
      a & BOOST_SERIALIZATION_NVP(_points);
      a & BOOST_SERIALIZATION_NVP(_currentPoint);
    }

    vvTFCustom();
    vvTFCustom(vvTFCustom*);
    vvTFCustom(float, float, float=0.5f, float=0.0f, float=0.5f, float=0.0f);
    vvTFCustom(std::ifstream& file);
    virtual ~vvTFCustom();
    virtual std::string toString() const;
    virtual void fromString(const std::string& str);
    virtual float getOpacity(float, float=-1.0f, float=-1.0f);
    vvTFPoint* addPoint(float, float=-1.0f, float=-1.0f);
    void removeCurrentPoint();
    vvTFPoint* selectPoint(float, float, float, float, float=-1.0f, float=0.0f, float=-1.0f, float=0.0f);
    void setCurrentPoint(float, float, float=-1.0f, float=-1.0f);
    void moveCurrentPoint(float, float, float=0.0f, float=0.0f);
    void sortPoints();
    void setSize(float, float=-1.0f, float=-1.0f);
};


/** 06/2008 L. Dematte'
  Transfer function widget to specify a custom 2D transfer function widget with control points.
  Points will be used to create a custom "tent" or an "extruded" shape (basically a polyline
  with alpha value as height).
 */
class VIRVO_FILEIOEXPORT vvTFCustom2D : public vvTFWidget
{
protected:
    bool _ownColor;

public:
    //float _size[3];                ///< width, height, depth of TF area [volume data space]
    std::list<vvTFPoint*> _points; ///< list of control points; coordinates are relative to widget center
    bool _mapDirty;

    vvColor _col;

    float _opacity;
    bool _extrude;
    vvTFPoint* _centralPoint;      ///< central point

    vvTFCustom2D(bool extrude, float opacity, float xCenter, float yCenter);
    vvTFCustom2D(vvTFCustom2D*);
    vvTFCustom2D(std::ifstream& file);
    virtual ~vvTFCustom2D();
    virtual std::string toString() const;
    virtual void fromString(const std::string& str);
    virtual float getOpacity(float, float=-1.0f, float=-1.0f);
    vvTFPoint* addPoint(float opacity, float x, float y);
    void addPoint(vvTFPoint* newPoint);


    virtual bool getColor(vvColor&, float, float=-1.0f, float=-1.0f);
    virtual bool hasOwnColor();
    virtual void setOwnColor(bool);

private:
   virvo::vec3 _size;             // width, height, depth of TF area [volume data space]
   float* _map;
   int _dim;                      // dimension of the map

   void addMapPoint(int x, int y, float value);
   void drawFreeContour();
   void uniformFillFreeArea();

   void internalFloodFill(float* map, int x, int y, int xDim, int yDim, float oldV, float newV);
   void midPointLine(float* map, int x0, int y0, int x1, int y1, float alpha0, float alpha1);
};

/** 06/2008 L. Dematte'
  Transfer function widget to specify a custom transfer function.
  Values for each point in space are given.
 */
class VIRVO_FILEIOEXPORT vvTFCustomMap : public vvTFWidget
{
  protected:
    bool _ownColor;                // true = use widget's own color for TF; false=use background color for TF

  public:
    vvColor _col;                  // RGB color
    virvo::vec3 _size;             // width, height, depth of TF area [volume data space]
    float* _map;
    virvo::vec3i _dim;             // dimensions of the map [widget data space]

    vvTFCustomMap();
    vvTFCustomMap(float x, float w, float y=0.5f, float h=0.0f, float z=0.5f, float d=0.0f);
    vvTFCustomMap(vvColor, bool, float x, float w, float y=0.5f, float h=0.0f, float z=0.5f, float d=0.0f);
    vvTFCustomMap(vvTFCustomMap*);
    vvTFCustomMap(std::ifstream& file);
    virtual ~vvTFCustomMap();
    virtual std::string toString() const;
    virtual void fromString(const std::string& str);
    virtual float getOpacity(float, float=-1.0f, float=-1.0f);
    void setOpacity(float val, float x, float y=-1.0f, float z=-1.0f);

    virtual bool getColor(vvColor&, float, float=-1.0f, float=-1.0f);
    virtual bool hasOwnColor();
    virtual void setOwnColor(bool);

private:
    int computeIdx(float x, float y, float z);
};

BOOST_CLASS_EXPORT_KEY(vvTFBell)
BOOST_CLASS_EXPORT_KEY(vvTFColor)
BOOST_CLASS_EXPORT_KEY(vvTFCustom)
#if 0
BOOST_CLASS_EXPORT_KEY(vvTFCustom2D)
BOOST_CLASS_EXPORT_KEY(vvTFCustomMap)
#endif
BOOST_CLASS_EXPORT_KEY(vvTFPyramid)
BOOST_CLASS_EXPORT_KEY(vvTFSkip)

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
