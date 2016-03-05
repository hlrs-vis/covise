/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _QUICKNAV_DRAWABLE_H
#define _QUICKNAV_DRAWABLE_H

#include <osg/Drawable>

class QuickNavDrawable : public osg::Drawable
{
private:
public:
    // The constructor here does nothing. One thing that may be necessary is
    // disabling display lists. This can be done by calling
    //    setSupportsDisplayList (false);
    // Display lists should be disabled for 'Drawable's that can change over
    // time (that is, the vertices drawn change from time to time).
    QuickNavDrawable()
        : osg::Drawable()
    {
        setSupportsDisplayList(false);
        // This contructor intentionally left blank. Duh.
    };

    // I can't say much about the methods below, but OSG seems to expect
    // that we implement them.
    QuickNavDrawable(const QuickNavDrawable &pg,
                     const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY)
        : osg::Drawable(pg, copyop)
    {
        setSupportsDisplayList(false);
    };

    virtual osg::Object *cloneType() const
    {
        return new QuickNavDrawable();
    };

    virtual osg::Object *clone(const osg::CopyOp &copyop) const
    {
        return new QuickNavDrawable(*this, copyop);
    };

    // Real work is done here. THERE IS A VERY IMPORTANT THING TO NOTE HERE:
    // the 'drawImplementation()' method receives an 'osg::State' as
    // parameter. This can be used to change the OpenGL state, but changing
    // the OpenGL state here is something to be avoided as much as possible.
    // Do this *only* if it is *absolutely* necessary to make your rendering
    // algorithm work. The "right" (most efficient and flexible) way to change
    // the OpenGL state in OSG is by attaching 'StateSet's to 'Node's and
    // 'Drawable's.
    // That said, the example below shows how to change the OpenGL state in
    // these rare cases in which it is necessary. But always keep in mind:
    // *Change the OpenGL state only if strictly necessary*.
    virtual void drawImplementation(osg::RenderInfo &info) const;
};

#endif
