/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// -*-c++-*-

/*
 * OpenSceneGraph - Copyright (C) 1998-2003 Robert Osfield
 *
 * This library is open source and may be redistributed and/or modified under
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * OpenSceneGraph Public License for more details.
 */

/*
 * osgFX::Outline - Copyright (C) 2004 Ulrich Hertlein
 */

#ifndef OSGFX_OUTLINE_
#define OSGFX_OUTLINE_

#include <osgFX/Export>
#include <osgFX/Effect>
#include <osgUtil/IntersectVisitor>

#include <util/coTypes.h>

namespace osgFX
{
class COVEREXPORT EnableStencilCallback;
/**
     * Outline effect.
     */
class COVEREXPORT Outline : public Effect
{
public:
    /// Constructor.
    Outline();
    Outline(const Outline &copy,
            const osg::CopyOp &op = osg::CopyOp::SHALLOW_COPY)
        : Effect(copy, op)
    {
        _width = copy._width;
        _color = copy._color;
    }
    /** custom traversal */
    virtual void traverse(osg::NodeVisitor &nv)
    {
        if (dynamic_cast<osgUtil::IntersectVisitor *>(&nv))
            osg::Group::traverse(nv);
        else
            Effect::traverse(nv);
    };

    // Effect class info
    META_Effect(osgFX, Outline, "Outline",
                "Stencil buffer based object outlining.",
                "Ulrich Hertlein <u.hertlein@sandbox.de>");

    /// Set outline width.
    void setWidth(float w)
    {
        _width = w;
        dirtyTechniques();
    }

    /// Get outline width.
    float getWidth() const
    {
        return _width;
    }

    /// Set outline color.
    void setColor(const osg::Vec4 &col)
    {
        _color = col;
        dirtyTechniques();
    }

    /// Get outline color.
    const osg::Vec4 &getColor() const
    {
        return _color;
    }

protected:
    /// Destructor.
    virtual ~Outline();

    /// Define available techniques.
    bool define_techniques();

private:
    /// Outline width.
    float _width;

    /// Outline color.
    osg::Vec4 _color;
    EnableStencilCallback *enableCallback;
};
};

#endif
