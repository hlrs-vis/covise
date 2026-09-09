/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AURAL_REALITY_SPEAKER_H
#define _AURAL_REALITY_SPEAKER_H

#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <string>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include "TmtEntity.h"
#include "Selection.h"

class Speaker : public Selectable, public TmtEntity, public TmtEntityTransformMixin
{
public:
    Speaker(const std::string &id);
    ~Speaker();
    void preFrame();

protected:
    virtual void updateSelection();

private:
    SelectableSensor *sensor;
};

#endif
