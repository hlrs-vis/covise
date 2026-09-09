/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AURAL_REALITY_TMTENTITY_H
#define _AURAL_REALITY_TMTENTITY_H

#include "CustomTransformInteractor.h"
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <string>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

class TmtEntity
{
public:
    TmtEntity(const std::string &id);
    ~TmtEntity() = default;

    const std::string &getId() const;

private:
    std::string id;
};

class TmtEntityTransformMixin
{
public:
    TmtEntityTransformMixin();
    ~TmtEntityTransformMixin() = default;

    void setTransform(osg::Matrix transform);
    osg::Matrix getTransform() const;
    osg::ref_ptr<osg::MatrixTransform> getTransformNode();

    void showInteractor(bool show);
    void setOffset(osg::Matrix offset_);
    bool checkTransformChanged();

protected:
    CustomTransformInteractor interactor;
    osg::ref_ptr<osg::MatrixTransform> transform;
    osg::Matrix offset;
    osg::Matrix offset_i;
};

#endif
