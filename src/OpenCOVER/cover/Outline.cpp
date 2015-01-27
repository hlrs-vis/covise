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

#include "Outline.h"

#include <osgFX/Registry>

#include <osg/Group>
#include <osg/Stencil>
#include <osg/CullFace>
#include <osg/PolygonMode>
#include <osg/LineWidth>
#include <osg/Material>

#include <osg/NodeCallback>
#include <osgUtil/CullVisitor>
#include <osg/PolygonOffset>

const unsigned int On = osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE;
const unsigned int Off = osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE;

namespace osgFX
{
/// Register prototype.
Registry::Proxy proxy(new Outline);

/**
     * Outline technique.
     */
class OutlineTechnique : public Technique
{
public:
    /// Constructor.
    OutlineTechnique(const Outline &outline)
        : Technique()
    {
        _outline = &outline;
    }

    /// Validate.
    bool validate(osg::State &) const
    {
        return true;
    }

protected:
    /// Define render passes.
    void define_passes()
    {

        /*
             * draw
             * - set stencil buffer to ref=1 where draw occurs
             * - clear stencil buffer to 0 where test fails
             */
        {
            osg::StateSet *state = new osg::StateSet;

            /*
                // stencil op
                osg::Stencil* stencil  = new osg::Stencil;
                stencil->setFunction(osg::Stencil::ALWAYS, 1, ~0);
                stencil->setOperation(osg::Stencil::KEEP,
                                      osg::Stencil::KEEP,
                                      osg::Stencil::REPLACE);
                state->setAttributeAndModes(stencil, On); 
                */

            osg::ref_ptr<osg::PolygonOffset> polyoffset = new osg::PolygonOffset;
            polyoffset->setFactor(1.0f);
            polyoffset->setUnits(0.5f);
            state->setAttributeAndModes(polyoffset.get(), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            addPass(state);
        }

        /*
             * post-draw
             * - only draw where draw didn't set the stencil buffer
             * - draw only back-facing polygons
             * - draw back-facing polys as lines
             * - disable depth-test, lighting & texture
             */
        {
            osg::StateSet *state = new osg::StateSet;
            /*
                // stencil op
                osg::Stencil* stencil  = new osg::Stencil;
                stencil->setFunction(osg::Stencil::NOTEQUAL, 1, ~0);
                stencil->setOperation(osg::Stencil::KEEP,
                                      osg::Stencil::KEEP,
                                      osg::Stencil::REPLACE);
                state->setAttributeAndModes(stencil, On);
                */

            // poly mode for back-facing polys
            osg::PolygonMode *pm = new osg::PolygonMode;
            pm->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
            state->setAttributeAndModes(pm, On);

            // cull front-facing polys
            osg::CullFace *cf = new osg::CullFace;
            cf->setMode(osg::CullFace::FRONT);
            state->setAttributeAndModes(cf, On);

            // outline width
            osg::LineWidth *lw = new osg::LineWidth;
            lw->setWidth(_outline->getWidth());
            state->setAttributeAndModes(lw, On);

            // outline color/material
            const osg::Material::Face face = osg::Material::FRONT_AND_BACK;
            osg::Material *mtl = new osg::Material;
            mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
            mtl->setAmbient(face, _outline->getColor());
            mtl->setDiffuse(face, _outline->getColor());
            mtl->setEmission(face, _outline->getColor());
            state->setAttributeAndModes(mtl, On);

            // disable modes
            //state->setMode(GL_BLEND, Off);
            //state->setMode(GL_DEPTH_TEST, Off);

            state->setTextureMode(0, GL_TEXTURE_1D, Off);
            state->setTextureMode(0, GL_TEXTURE_2D, Off);
            state->setTextureMode(0, GL_TEXTURE_3D, Off);

            addPass(state);
        }
    }

private:
    /// Outline effect.
    osg::ref_ptr<const Outline> _outline;
};

/**
     * Enable stencil clear callback.
     */
class EnableStencilCallback : public osg::NodeCallback
{
public:
    /// Constructor.
    EnableStencilCallback(Outline &outline)
    {
        _outline = &outline;
    }

    virtual void operator()(osg::Node *node, osg::NodeVisitor *nv)
    {

        // enable stencil clear on render stage
        osgUtil::CullVisitor *cv = dynamic_cast<osgUtil::CullVisitor *>(nv);
        if (cv)
        {
            osgUtil::RenderStage *render = cv->getRenderStage();
            unsigned int mask = render->getClearMask();
            if ((mask & GL_STENCIL_BUFFER_BIT) == 0)
            {
                render->setClearMask(mask | GL_STENCIL_BUFFER_BIT);
                render->setClearStencil(0);
                osg::notify() << "osgFX::Outline activated stencil\n";
            }
            else
                osg::notify() << "osgFX::Outline stencil is active\n";
        }

        // clear update callback
        if (node)
            node->setCullCallback(NULL);

        traverse(node, nv);
    }

private:
    /// Outline effect.
    const Outline *_outline;
};

/// Constructor.
Outline::Outline()
    : Effect()
{
    _width = 3.0f;
    _color.set(1.0f, 1.0f, 1.0f, 1.0f);
    enableCallback = new EnableStencilCallback(*this);
    enableCallback->ref();
    setCullCallback(enableCallback);
}
/// Destructor.
Outline::~Outline()
{
    if (enableCallback != NULL)
    {
        enableCallback->unref();
    }
}

/// Define available techniques.
bool Outline::define_techniques()
{
    addTechnique(new OutlineTechnique(*this));
    return true;
}
};
