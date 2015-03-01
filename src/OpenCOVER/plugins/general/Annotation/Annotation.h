/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ANNOTATION_H
#define _ANNOTATION_H

#include <osg/MatrixTransform>
#include <osg/Billboard>
#include <osg/Group>
#include <osg/Switch>
#include <osg/Material>
#include <osgText/Text>
#include <osg/AlphaFunc>
#include <osg/BlendFunc>
#include <osg/Object>
#include <osg/ShapeDrawable>

#include "AnnotationSensor.h"
#include <PluginUtil/coArrow.h>
#include <cover/coBillboard.h>
#include <cover/coVRLabel.h>

using namespace opencover;

class AnnotationSensor;

class Annotation
{
private:
    int id; ///< id of this annotation
    int owner; ///< owner ID of this annotation

    osg::Group *mainGroup; ///< osg node for this instance

    osg::MatrixTransform *pos; ///< position of this instance
    osg::MatrixTransform *rot; ///< orientation of this instance
    osg::MatrixTransform *scale; ///< scale of this instance

    AnnotationSensor *mySensor; ///< DOCUMENT ME!

    float _hue; ///< color hue value
    float _scaleVal; ///< scale

    coArrow *arrow; ///< arrow instance
    float arrowSize; ///< arrow size
    coVRLabel *label; ///< label instance

public:
    /**
    * construct a new annotation
    * \param   id          id number of this instance
    * \param   owner       collaborative owner id of this instance
    * \param   node        parent node of this instance
    * \param   initscale   initial scale of the arrow
    * \param   orientation orientation of the arrow
    */
    Annotation(int id, int owner, osg::Node *parent, float initscale,
               osg::Matrix orientation);
    ~Annotation();

    void setPos(const osg::Matrix &mat);
    osg::Matrix getPos() const;

    void getMat(osg::Matrix &) const;
    void getMat(osg::Matrix::value_type[16]) const;

    void setScale(float);
    float getScale() const;

    void setBaseSize(float);

    void setColor(float);
    float getColor() const;
    void resetColor();

    void setText(const char *text);
    const osgText::String getText() const;
    void updateLabelPosition();
    void scaleArrowToConstantSize();

    void setAmbient(osg::Vec4);

    /// sets color to local-locked state
    void setAmbientLocalLocked();

    /// sets color to remote-locked state
    void setAmbientRemoteLocked();

    /// sets color to unlocked state
    void setAmbientUnlocked();

    void setVisible(bool);

    /**
    * returns the id of this instance
    */
    int getID() const;

    /**
    * check whether this instance has the same id as the parameter
    */
    bool sameID(const int id) const;

    /// Returns the owner id; -1 in case of no owner
    int getOwnerID() const;

    /// set the owner id; set to -1 if no owner
    void setOwnerID(const int id);

    /**
    * check whether the owner ids are the same
    */
    bool sameOwnerID(const int id) const;

    /**
    * check whether manipulation is allowed for id
    */
    bool changesAllowed(const int id) const;
}; //class Annotation

#endif //_ANNOTATION_H
