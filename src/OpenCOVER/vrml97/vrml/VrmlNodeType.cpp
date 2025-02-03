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

#include "VrmlNodeType.h"
#include "VrmlField.h"
#include "VrmlMFNode.h"
#include "VrmlNode.h"
#include "VrmlNamespace.h"

#include "Doc.h"
#include "VrmlNodeProto.h"
#include "VrmlMFString.h"
#include "VrmlScene.h"
#include "System.h"

using namespace vrml;

VrmlNodeType::VrmlNodeType(const char *nm, VrmlNode *(*creator)(VrmlScene *))
{
    d_refCount = 0;
    if (nm)
    {
        d_name = new char[strlen(nm) + 1];
        strcpy(d_name, nm);
    }
    else
    {
        d_name = new char[strlen("<unnamed type>") + 1];
        strcpy(d_name, "<unnamed type>");
    }
    d_namespace = 0;
    d_url = 0;
    d_actualUrl = 0;
    d_relative = 0;
    d_creator = creator;
    d_implementation = 0;
    d_fieldsInitialized = false;
}

void destructFieldList(VrmlNodeType::FieldList &f)
{
    VrmlNodeType::FieldList::iterator i;
    for (i = f.begin(); i != f.end(); ++i)
    {
        VrmlNodeType::ProtoField *r = *i;
        free(r->name);
        if (r->defaultValue)
            delete r->defaultValue;

        // free NodeFieldRec* s in r->thisIS;
        VrmlNodeType::ISMap::const_iterator j;
        for (j = r->thisIS.begin(); j != r->thisIS.end(); ++j)
        {
            VrmlNodeType::NodeFieldRec *nf = *j;
            free(nf->fieldName);
            delete nf;
        }

        delete r;
    }
}

VrmlNodeType::~VrmlNodeType()
{
    delete[] d_name;

    delete d_namespace;
    delete d_url;
    delete d_actualUrl;
    delete d_relative;

    // Free strings & defaults duplicated when fields/eventIns/eventOuts added:
    destructFieldList(d_eventIns);
    destructFieldList(d_eventOuts);
    destructFieldList(d_fields);

    delete d_implementation;
}

void VrmlNodeType::setScope(VrmlNamespace *createdIn)
{
    d_namespace = new VrmlNamespace(createdIn);
}

void VrmlNodeType::setUrl(VrmlField *url, Doc *relative)
{
    if (d_implementation)
        return; // Too late...

    if (d_url)
        delete d_url;
    d_url = url ? url->toMFString() : 0;
    if (d_relative)
        delete d_relative;
    d_relative = relative ? new Doc(relative) : 0;
}

void VrmlNodeType::setActualUrl(const char *url)
{
    if (d_actualUrl)
        delete d_actualUrl;
    d_actualUrl = url ? new VrmlSFString(url) : 0;
}

void
VrmlNodeType::addEventIn(const char *ename, VrmlField::VrmlFieldType type)
{
    add(d_eventIns, ename, type);
}

void
VrmlNodeType::addEventOut(const char *ename, VrmlField::VrmlFieldType type)
{
    add(d_eventOuts, ename, type);
}

void
VrmlNodeType::addField(const char *ename,
                       VrmlField::VrmlFieldType type,
                       VrmlField *defaultValue)
{
    add(d_fields, ename, type);
    if (defaultValue)
        setFieldDefault(ename, defaultValue);
}

void
VrmlNodeType::addExposedField(const char *ename,
                              VrmlField::VrmlFieldType type,
                              VrmlField *defaultValue)
{
    char tmp[1000];
    add(d_fields, ename, type);
    if (defaultValue)
        setFieldDefault(ename, defaultValue);

    sprintf(tmp, "set_%s", ename);
    add(d_eventIns, tmp, type);
    sprintf(tmp, "%s_changed", ename);
    add(d_eventOuts, tmp, type);
}

void
VrmlNodeType::add(FieldList &recs, const char *ename, VrmlField::VrmlFieldType type)
{
    ProtoField *r = new ProtoField;
    r->name = strdup(ename);
    r->type = type;
    r->defaultValue = 0;
    recs.push_front(r);
}

void
VrmlNodeType::setFieldDefault(const char *fname, VrmlField *defaultValue)
{
    FieldList::const_iterator i;

    for (i = d_fields.begin(); i != d_fields.end(); ++i)
        if (strcmp((*i)->name, fname) == 0)
        {
            if ((*i)->defaultValue)
            {
                System::the->error("Default for field %s of %s already set.\n",
                                   fname, getName());
                delete (*i)->defaultValue;
            }
            (*i)->defaultValue = defaultValue ? defaultValue->clone() : 0;
            return;
        }

    System::the->error("setFieldDefault for field %s of %s failed: no such field.\n",
                       fname, getName());
}

// Download the EXTERNPROTO definition

void VrmlNodeType::fetchImplementation(int parentId)
{
    // Get the PROTO def from the url (relative to original scene url).
    VrmlNodeType *proto = VrmlScene::readPROTO(d_url, d_relative, parentId);
    if (proto)
    {
        // check type of first node...
        // steal nodes without cloning
        d_implementation = proto->d_implementation;
        proto->d_implementation = 0;

        // Make sure we get all the IS maps by using the field
        // lists from the implementation rather than just the
        // interface.
        destructFieldList(d_eventIns);
        destructFieldList(d_eventOuts);
        destructFieldList(d_fields);

        d_eventIns = proto->d_eventIns;
        d_eventOuts = proto->d_eventOuts;
        d_fields = proto->d_fields;

        proto->d_eventIns.erase(proto->d_eventIns.begin(),
                                proto->d_eventIns.end());
        proto->d_eventOuts.erase(proto->d_eventOuts.begin(),
                                 proto->d_eventOuts.end());
        proto->d_fields.erase(proto->d_fields.begin(),
                              proto->d_fields.end());

        setActualUrl(proto->url());

        delete proto;
    }
}

VrmlMFNode *VrmlNodeType::getImplementationNodes(int parentId)
{

	if (!d_implementation && d_url)
        fetchImplementation(parentId);

    // Now that the nodes are here, initialize any IS'd fields
    // to the default values (could do it at instantiation...)
    // Also, any IS'd field needs a name because of the stupid
    // IS mapping implementation (when I copy the implementation
    // nodes for each instance of the PROTO, the IS mappings
    // need to copied too). So make up a unique name for those
    // without.
    if (!d_fieldsInitialized)
    {
        FieldList::iterator i;
        char buf[32];

        for (i = d_fields.begin(); i != d_fields.end(); ++i)
        {
            ISMap &ismap = (*i)->thisIS;
            ISMap::iterator j;
            for (j = ismap.begin(); j != ismap.end(); ++j)
            {
                VrmlNode *n = (*j)->node;
                if (strcmp(n->name(), "") == 0)
                {
                    sprintf(buf, "#%llx", (unsigned long long)n);
                    n->setName(buf);
                }

                if ((*i)->defaultValue)
                    n->setField((*j)->fieldName, *((*i)->defaultValue));
            }
        }

        // Set names on IS'd eventIns/Outs, too.
        for (i = d_eventIns.begin(); i != d_eventIns.end(); ++i)
        {
            ISMap &ismap = (*i)->thisIS;
            ISMap::iterator j;
            for (j = ismap.begin(); j != ismap.end(); ++j)
            {
                VrmlNode *n = (*j)->node;
                if (strcmp(n->name(), "") == 0)
                {
                    sprintf(buf, "#%llx", (unsigned long long)n);
                    n->setName(buf);
                }
            }
        }
        for (i = d_eventOuts.begin(); i != d_eventOuts.end(); ++i)
        {
            ISMap &ismap = (*i)->thisIS;
            ISMap::iterator j;
            for (j = ismap.begin(); j != ismap.end(); ++j)
            {
                VrmlNode *n = (*j)->node;
                if (strcmp(n->name(), "") == 0)
                {
                    sprintf(buf, "#%llx", (unsigned long long)n);
                    n->setName(buf);
                }
            }
        }

        d_fieldsInitialized = true;
    }

    return d_implementation;
}

// This will NOT fetch the implementation of an EXTERNPROTO.
// This method is used in VrmlNodeProto to check the type of
// SFNode fields in toXXX() node downcasts. Type checking
// of EXTERNPROTOs is deferred until the implementation is
// actually downloaded. (not actually done yet...)

VrmlNode *VrmlNodeType::firstNode()
{
    return d_implementation ? (*d_implementation)[0] : 0;
}

VrmlField::VrmlFieldType
VrmlNodeType::hasEventIn(const char *ename) const
{
    return has(d_eventIns, ename);
}

VrmlField::VrmlFieldType
VrmlNodeType::hasEventOut(const char *ename) const
{
    return has(d_eventOuts, ename);
}

VrmlField::VrmlFieldType
VrmlNodeType::hasField(const char *ename) const
{
    return has(d_fields, ename);
}

VrmlField::VrmlFieldType
VrmlNodeType::hasExposedField(const char *ename) const
{
    // Must have field "name", eventIn "set_name", and eventOut
    // "name_changed", all with same type:
    char tmp[1000];
    VrmlField::VrmlFieldType type;

    if ((type = has(d_fields, ename)) == VrmlField::NO_FIELD)
        return VrmlField::NO_FIELD;

    sprintf(tmp, "set_%s", ename);
    if (type != has(d_eventIns, tmp))
        return VrmlField::NO_FIELD;

    sprintf(tmp, "%s_changed", ename);
    if (type != has(d_eventOuts, tmp))
        return VrmlField::NO_FIELD;

    return type;
}

VrmlField::VrmlFieldType
VrmlNodeType::has(const FieldList &recs, const char *ename) const
{
    FieldList::const_iterator i;
    for (i = recs.begin(); i != recs.end(); ++i)
    {
        if (strcmp((*i)->name, ename) == 0)
            return (*i)->type;
    }
    return VrmlField::NO_FIELD;
}

VrmlField *
VrmlNodeType::fieldDefault(const char *fname) const
{
    FieldList::const_iterator i;
    for (i = d_fields.begin(); i != d_fields.end(); ++i)
    {
        if (strcmp((*i)->name, fname) == 0)
            return (*i)->defaultValue;
    }
    return 0;
}

void VrmlNodeType::addNode(VrmlNode *node)
{
    // add node to list of implementation nodes
    if (d_implementation)
        d_implementation->addNode(node);
    else
        d_implementation = new VrmlMFNode(node);
}

void VrmlNodeType::addIS(const char *isFieldName,
                         const VrmlNode *implNode,
                         const char *implFieldName)
{
    FieldList::iterator i;
    System::the->debug("%s::addIS(%s, %s::%s.%s)\n",
                       getName(),
                       isFieldName,
                       implNode->nodeType()->getName(),
                       implNode->name(),
                       implFieldName);

    // Fields
    for (i = d_fields.begin(); i != d_fields.end(); ++i)
    {
        if (strcmp((*i)->name, isFieldName) == 0)
        {
            NodeFieldRec *nf = new NodeFieldRec;
            nf->node = (VrmlNode *)implNode; // oops...
            nf->fieldName = strdup(implFieldName);
            (*i)->thisIS.push_front(nf);
            break;
        }
    }

    // EventIns
    for (i = d_eventIns.begin(); i != d_eventIns.end(); ++i)
    {
        if (strcmp((*i)->name, isFieldName) == 0)
        {
            NodeFieldRec *nf = new NodeFieldRec;
            nf->node = (VrmlNode *)implNode; // oops...
            nf->fieldName = strdup(implFieldName);
            (*i)->thisIS.push_front(nf);
            break;
        }
    }

    // EventOuts
    for (i = d_eventOuts.begin(); i != d_eventOuts.end(); ++i)
    {
        if (strcmp((*i)->name, isFieldName) == 0)
        {
            NodeFieldRec *nf = new NodeFieldRec;
            nf->node = (VrmlNode *)implNode; // oops...
            nf->fieldName = strdup(implFieldName);
            (*i)->thisIS.push_front(nf);
            break;
        }
    }
    if (strncmp(isFieldName, "set_", 4) != 0)
    {
        char *localFieldName = NULL;
        localFieldName = new char[strlen(isFieldName) + 5];
        strcpy(localFieldName, "set_");
        strcat(localFieldName, isFieldName);
        // EventIns
        for (i = d_eventIns.begin(); i != d_eventIns.end(); ++i)
        {
            if (strcmp((*i)->name, localFieldName) == 0)
            {
                NodeFieldRec *nf = new NodeFieldRec;
                nf->node = (VrmlNode *)implNode; // oops...
                nf->fieldName = strdup(implFieldName);
                (*i)->thisIS.push_front(nf);
                break;
            }
        }
        delete[] localFieldName;
    }
    if ((strlen(isFieldName) < 9) || (strncmp(isFieldName + strlen(isFieldName) - 8, "changed_", 8) != 0))
    {
        char *localFieldName = NULL;
        localFieldName = new char[strlen(isFieldName) + 9];
        strcpy(localFieldName, isFieldName);
        strcat(localFieldName, "_changed");
        // EventOuts
        for (i = d_eventOuts.begin(); i != d_eventOuts.end(); ++i)
        {
            if (strcmp((*i)->name, localFieldName) == 0)
            {
                NodeFieldRec *nf = new NodeFieldRec;
                nf->node = (VrmlNode *)implNode; // oops...
                nf->fieldName = strdup(implFieldName);
                (*i)->thisIS.push_front(nf);
                break;
            }
        }
        delete[] localFieldName;
    }
}

VrmlNodeType::ISMap *VrmlNodeType::getFieldISMap(const char *fieldName)
{
    FieldList::iterator i;
    for (i = d_fields.begin(); i != d_fields.end(); ++i)
        if (strcmp((*i)->name, fieldName) == 0)
            return &((*i)->thisIS);
    return 0;
}

// VrmlNode factory: create a new instance of a node of this type.
// Built in nodes have a creator function specified, while instances
// of PROTOs are constructed by VrmlNodeProto.

VrmlNode *
VrmlNodeType::newNode(VrmlScene *scene) const
{
    if (d_creator)
        return (*d_creator)(scene);

    return new VrmlNodeProto((VrmlNodeType *)this, scene);
}
