/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeScript.h

#ifndef _VRMLNODESCRIPT_
#define _VRMLNODESCRIPT_
#include "config.h"

#include <list>

#include "VrmlNode.h"

#include "VrmlMFString.h"
#include "VrmlSFString.h"
#include "VrmlSFBool.h"
#include "VrmlSFNode.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class Doc;
class ScriptObject;
class VrmlScene;

class VRMLEXPORT VrmlNodeScript : public VrmlNodeChild
{

public:
    // Define the fields of Script nodes
    static void initFields(VrmlNodeScript *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeScript(VrmlScene *scene = 0);
    VrmlNodeScript(const VrmlNodeScript &);
    virtual ~VrmlNodeScript();

    virtual void cloneChildren(VrmlNamespace *);
    // Copy the ROUTEs
    virtual void copyRoutes(VrmlNamespace *ns);

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    // Script processing methods
    void initialize(double timeStamp);
    void update(VrmlSFTime &now);
    void shutdown(double timeStamp);

    // Methods for adding eventIns/Outs/fields to this script
    void addEventIn(const char *name, VrmlField::VrmlFieldType type);
    void addEventOut(const char *name, VrmlField::VrmlFieldType type);
    void addExposedField(const char *name, VrmlField::VrmlFieldType typeVrml, VrmlField *defaultValue = 0);
    void addField(const char* name, VrmlField::VrmlFieldType type,
        VrmlField* defaultVal = 0, bool exposed = false);

    // Access to eventIns/Outs/fields for ScriptObjects

    // Per-script field type
    typedef struct
    {
        char *name;
        VrmlField *value;
        VrmlField::VrmlFieldType type;
        bool modified;
        bool exposed;
    } ScriptField;

    typedef std::list<ScriptField *> FieldList;

    // Tests for specific fields/events
    VrmlField::VrmlFieldType hasEventIn(const char *name) const;
    VrmlField::VrmlFieldType hasEventOut(const char *name) const;
    VrmlField::VrmlFieldType hasField(const char *name) const;
    VrmlField::VrmlFieldType hasExposedField(const char *name) const;

    // Set field/event values
    void setEventIn(const char *, const VrmlField *);
    void setEventOut(const char*, const VrmlField*);
    void setExposedField(const char*, const VrmlField*);

    // Fields and events defined for this Script
    FieldList &eventIns()
    {
        return d_eventIns;
    }
    FieldList &eventOuts()
    {
        return d_eventOuts;
    }
    FieldList &fields()
    {
        return d_fields;
    }

    // Access to browser functions for ScriptObjects
    VrmlScene *browser()
    {
        return d_scene;
    }

private:
    // Fields
    VrmlSFBool d_directOutput;
    VrmlSFBool d_mustEvaluate;
    VrmlMFString d_url;
    VrmlSFNode d_myself;

    VrmlSFString d_relativeUrl;

    // The script language-dependent part
    ScriptObject *d_script;

    // Fields and events defined for this Script
    FieldList d_eventIns;
    FieldList d_eventOuts;
    FieldList d_fields;

    // Generic field/event add/test/value methods
    FieldList::iterator add(FieldList&, const char*, VrmlField::VrmlFieldType, bool exposed = false);
    VrmlField::VrmlFieldType has(const FieldList &, const char *) const;
    VrmlField *get(const FieldList &, const char *) const;
    void set(const FieldList &, const char *, const VrmlField *);

    int d_eventsReceived;
};
}
#endif //_VRMLNODESCRIPT_
