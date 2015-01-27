/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//  Copyrights on some portions of the code are held by others as documented
//  in the code. Permission to use this code for any purpose is granted as
//  long as all other copyrights in the code are respected and this copyright
//  statement is retained in the code and accompanying documentation both
//  online and printed.
//

#ifndef _VRMLNODETYPE_
#define _VRMLNODETYPE_
/**************************************************
 * VRML 2.0 Parser
 * Copyright (C) 1996 Silicon Graphics, Inc.
 *
 * Author(s)    : Gavin Bell
 *                Daniel Woods (first port)
 **************************************************
 */

//
// The VrmlNodeType class is responsible for storing information about node
// or prototype types.
//

#include <list>

#include "config.h"

#include "VrmlField.h"
#include "VrmlMFNode.h"
#include "VrmlSFString.h"

namespace vrml
{

class Doc;
class VrmlNamespace;
class VrmlNode;
class VrmlScene;

class VRMLEXPORT VrmlNodeType
{
public:
    // Constructor.  Takes name of new type (e.g. "Transform" or "Box")
    // Copies the string given as name.
    VrmlNodeType(const char *nm,
                 VrmlNode *(*creator)(VrmlScene *scene) = 0);

    // Deallocate storage for name and PROTO implementations
    ~VrmlNodeType();

    VrmlNodeType *reference()
    {
        ++d_refCount;
        return this;
    }
    void dereference()
    {
        if (--d_refCount == 0)
            delete this;
    }

    // Create a node of this type in the specified scene
    VrmlNode *newNode(VrmlScene *s = 0) const;

    // Set/get scope of this type (namespace is NULL for builtins).
    VrmlNamespace *scope()
    {
        return d_namespace;
    }
    void setScope(VrmlNamespace *scopeDefinedIn);

    // Routines for adding/getting eventIns/Outs/fields to this type
    void addEventIn(const char *name, VrmlField::VrmlFieldType type);
    void addEventOut(const char *name, VrmlField::VrmlFieldType type);
    void addField(const char *name, VrmlField::VrmlFieldType type,
                  VrmlField *defaultVal = 0);
    void addExposedField(const char *name, VrmlField::VrmlFieldType type,
                         VrmlField *defaultVal = 0);

    void setFieldDefault(const char *name, VrmlField *value);

    VrmlField::VrmlFieldType hasEventIn(const char *name) const;
    VrmlField::VrmlFieldType hasEventOut(const char *name) const;
    VrmlField::VrmlFieldType hasField(const char *name) const;
    VrmlField::VrmlFieldType hasExposedField(const char *name) const;

    VrmlField *fieldDefault(const char *name) const;

    // Set the URL to retrieve an EXTERNPROTO implementation from.
    void setUrl(VrmlField *url, Doc *relative = 0);

    void setActualUrl(const char *url);

    // Retrieve the actual URL the PROTO was retrieved from.
    const char *url()
    {
        return d_actualUrl ? d_actualUrl->get() : 0;
    }

    // Add a node to a PROTO implementation
    void addNode(VrmlNode *implNode);

    // Add an IS linkage to one of the PROTO interface fields/events.
    void addIS(const char *isFieldName,
               const VrmlNode *implNode,
               const char *implFieldName);

    const char *getName() const
    {
        return d_name;
    }

    VrmlMFNode *getImplementationNodes();

    VrmlNode *firstNode();

    typedef struct
    {
        VrmlNode *node;
        char *fieldName;
    } NodeFieldRec;

    typedef std::list<NodeFieldRec *> ISMap;

    typedef struct
    {
        char *name;
        VrmlField::VrmlFieldType type;
        VrmlField *defaultValue;
        ISMap thisIS;
    } ProtoField;

    typedef std::list<ProtoField *> FieldList;

    ISMap *getFieldISMap(const char *fieldName);

    FieldList &eventIns()
    {
        return d_eventIns;
    }
    FieldList &eventOuts()
    {
        return d_eventOuts;
    }

private:
    // Grab the implementation of an EXTERNPROTO
    void fetchImplementation();

    void add(FieldList &, const char *, VrmlField::VrmlFieldType);
    VrmlField::VrmlFieldType has(const FieldList &, const char *) const;

    int d_refCount;

    char *d_name;

    VrmlNamespace *d_namespace;

    VrmlMFString *d_url; // Where the EXTERNPROTO could be.
    VrmlSFString *d_actualUrl; // The URL actually used.
    Doc *d_relative;

    VrmlMFNode *d_implementation; // The PROTO implementation nodes

    // Pointer to function to create instances
    VrmlNode *(*d_creator)(VrmlScene *);

    // Fields defined for this node type
    FieldList d_eventIns;
    FieldList d_eventOuts;
    FieldList d_fields;

    bool d_fieldsInitialized;
};
}
#endif // _VRMLNODETYPE_
