/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __ADD_GEOM_H_
#define __ADD_GEOM_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS AddGeom
//
// Initial version: 2002-07-23 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "Attachable.h"
#include <iostream.h>

class coModule;
class coDistributedObject;
class coChoiceParam;
class coFloatVectorParam;

/**
 * Class modelling a single attached geometry
 *
 */
class AddGeom
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *  @param id   integer id of this AddGeom object
       *  @param mod  module to add my Parameters to
       *  must be called between paraSwitch ... paraEndSwitch !!!
       */
    AddGeom(int id, coModule *mod);

    /// Destructor : virtual in case we derive objeqcts
    virtual ~AddGeom();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // get the chosen attachable from this list
    void setAttachable(const Attachable *attachable);

    // put the object's STAR line to the given stream
    void printStarObj(ostream &str) const;

    // cleanup everything
    void clear();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // get the gometry object, including all interactions
    coDistributedObject *getObj(const char *baseName, const char *unit) const;

    // check whether this one is attached
    bool isAttached() const;

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    coFloatVectorParam *p_pos; // parameter position of the vents
    coFloatVectorParam *p_rot; // parameter for setting rotation as matrix (disabled)
    coFloatVectorParam *p_euler; // parameter for setting rotation as euler angles

    float d_pos[3]; // position of the vent
    float d_euler[3]; // rotation in euler angles
    float d_axis[3]; // channel axis, def is x-axis
    float d_coverRot[9]; // rotation matrix as vector which is set only from COVER

    Attachable *d_attachedPart; // the selected type, NULL if none

    int d_id; // my ID (integer);

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED, checked by assert
    AddGeom(const AddGeom &);

    /// Assignment operator: NOT IMPLEMENTED, checked by assert
    AddGeom &operator=(const AddGeom &);

    /// Default constructor: NOT IMPLEMENTED, checked by assert
    AddGeom();
};
#endif
