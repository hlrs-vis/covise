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

#include "CustomTransformInteractor.h"
#include "Selection.h"

struct SpeakerProperties
{
    float dispersion_horizontal = 0.0;
    float dispersion_vertical = 0.0;
    float cutoff_frequency_low = 0.0;
    float cutoff_frequency_high = 0.0;
    float maximum_sound_pressure_level = 0.0;
    float power_handling = 0.0;
};

class Speaker : public Selectable
{
public:
    Speaker(const std::string &id);
    ~Speaker();
    void preFrame();

    const std::string &getId() const;
    void setTransform(osg::Matrix transform);
    osg::Matrix getTransform() const;
    void setProperties(SpeakerProperties properties);
    const SpeakerProperties &getProperties() const;

protected:
    virtual void updateSelection();

private:
    std::string id;
    CustomTransformInteractor interactor;
    osg::ref_ptr<osg::MatrixTransform> transform;
    SpeakerProperties properties;

    osg::Matrix offset;
    osg::Matrix offset_i;

    SelectableSensor *sensor;
};

#endif
