/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ATTACHABLE_H_
#define _ATTACHABLE_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS Attachable
//
/// Initial version: 2002-07-23 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

class coDistributedObject;

/**
 * Class containing all information about an additional geometry
 *
 */
class Attachable
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *  @param filename name of the given OBJ file
       */
    Attachable(const char *filename);

    /// Destructor : virtual in case we derive objects
    virtual ~Attachable();

    /// Copy-Constructor: copy everything
    Attachable(const Attachable &old);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // return the name of the OBJ file
    const char *getObjFileName() const;

    // return the choice label value
    const char *getChoiceLabel() const;

    // return the choice label value
    const char *getObjPath() const;

    // return coDoPoints with attribs for OBJ reading
    coDistributedObject *getObjDO(const char *objName, int index, const char *unit) const;

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // the OBJ filename
    char *d_filename;

    // the choice label
    char *d_choiceLabel;

    // the path to the file
    char *d_path;

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Assignment operator: NOT IMPLEMENTED, checked by assert
    Attachable &operator=(const Attachable &);

    /// Default constructor: NOT IMPLEMENTED, checked by assert
    Attachable();
};
#endif
