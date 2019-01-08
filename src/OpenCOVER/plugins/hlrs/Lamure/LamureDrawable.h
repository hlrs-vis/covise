/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LamureDrawable_H
#define _LamureDrawable_H

#include <osg/Drawable>
#include <osg/Geode>
#include <scm/core/math.h>

class management;

class LamureDrawable : public osg::Drawable
{
public:
    LamureDrawable();
    virtual ~LamureDrawable();

    int loadBVH(const char * filename);

    virtual void drawImplementation(osg::RenderInfo &renderInfo) const;
private:
    virtual osg::Object *cloneType() const
    {
        return new LamureDrawable();
    }
    virtual osg::Object *clone(const osg::CopyOp &copyop) const
    {
        return new LamureDrawable(*this, copyop);
    }

    LamureDrawable(const LamureDrawable &, const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY);
    void init();
    struct ContextState
    {
        ContextState();
        ~ContextState();
    };

    mutable std::vector<ContextState *> contextState;

    management* management_ = nullptr;

    std::vector<std::string> model_filenames;
    std::vector<scm::math::mat4f>  model_transformations;
    int numModels = 0;
};

#endif
