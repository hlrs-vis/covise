/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeScript.cpp

#include "VrmlNodeScript.h"
#include "VrmlNodeType.h"
#include "VrmlSFTime.h"
#include "VrmlSFNode.h"
#include "VrmlMFNode.h"
#include "VrmlNamespace.h"

#include "ScriptObject.h"

#include "VrmlScene.h"
#include "System.h"

#include <stdio.h>

using namespace vrml;

#if defined(WIN32)
// Disable warning messages about forcing value to bool 'true' or 'false'
#pragma warning(disable : 4800)
#endif

void VrmlNodeScript::initFields(VrmlNodeScript *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("url", node->d_url),
                     field("directOutput", node->d_directOutput),
                     field("mustEvaluate", node->d_mustEvaluate),
                     field("myself", node->d_myself));
}

const char *VrmlNodeScript::name() { return "Script"; }


VrmlNodeScript::VrmlNodeScript(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_directOutput(false)
    , d_mustEvaluate(false)
    , d_script(0)
    , d_eventsReceived(0)
{
    d_myself.set(this);
    if (d_scene)
        d_scene->addScript(this);
}

VrmlNodeScript::VrmlNodeScript(const VrmlNodeScript &n)
    : VrmlNodeChild(n)
    , d_directOutput(n.d_directOutput)
    , d_mustEvaluate(n.d_mustEvaluate)
    , d_url(n.d_url)
    , d_script(0)
    , // Force creation of a distinct script object
    d_eventIns(0)
    , d_eventOuts(0)
    , d_fields(0)
    , d_eventsReceived(0)
{
    d_myself.set(this);
    // add eventIn/eventOut/fields from source Script
    FieldList::const_iterator i;

    for (i = n.d_eventIns.begin(); i != n.d_eventIns.end(); ++i)
        addEventIn((*i)->name, (*i)->type);
    for (i = n.d_eventOuts.begin(); i != n.d_eventOuts.end(); ++i)
        addEventOut((*i)->name, (*i)->type);
    for (i = n.d_fields.begin(); i != n.d_fields.end(); ++i)
        addField((*i)->name, (*i)->type, (*i)->value, (*i)->exposed);
}

VrmlNodeScript::~VrmlNodeScript()
{
    shutdown(System::the->time());

    // removeScript ought to call shutdown...
    if (d_scene)
        d_scene->removeScript(this);

    delete d_script;

    // delete eventIn/eventOut/field ScriptField list contents
    FieldList::iterator i;
    for (i = d_eventIns.begin(); i != d_eventIns.end(); ++i)
    {
        ScriptField *r = *i;
        free(r->name);
        delete r->value;
        delete r;
    }
    for (i = d_eventOuts.begin(); i != d_eventOuts.end(); ++i)
    {
        ScriptField *r = *i;
        free(r->name);
        delete r->value;
        delete r;
    }
    for (i = d_fields.begin(); i != d_fields.end(); ++i)
    {
        ScriptField *r = *i;
        free(r->name);
        delete r->value;
        delete r;
    }
}

// Any SFNode or MFNode fields need to be cloned as well.
//

void VrmlNodeScript::cloneChildren(VrmlNamespace *ns)
{
    FieldList::iterator i;
    for (i = d_fields.begin(); i != d_fields.end(); ++i)
        if ((*i)->value)
        {
            if ((*i)->type == VrmlField::SFNODE)
            {
                const VrmlSFNode *sfn = (*i)->value->toSFNode();
                if (sfn && sfn->get() && sfn->get() != this)
                {
                    (*i)->value = new VrmlSFNode(sfn->get()->clone(ns));
                }
            }

            else if ((*i)->type == VrmlField::MFNODE && (*i)->value->toMFNode())
            {
                int nk = (*i)->value->toMFNode()->size();
                VrmlNode **kids = (*i)->value->toMFNode()->get();
                for (int k = 0; k < nk; ++k)
                    if (kids[k])
                    {
                        if (kids[k] != this)
                        {
                            VrmlNode *tmp = kids[k];
                            kids[k] = tmp->clone(ns);
                            tmp->dereference();
                            kids[k]->parentList.push_back(this);
                        }
                    }
            }
        }
}

// Copy the routes to nodes in the given namespace.

void VrmlNodeScript::copyRoutes(VrmlNamespace *ns)
{
    const char *fromName = name();
    VrmlNode *fromNode = fromName ? ns->findNode(fromName) : 0;

    if (fromNode)
        for (Route *r = d_routes; r; r = r->next())
        {
            const char *toName = r->toNode()->name();
            VrmlNode *toNode = toName ? ns->findNode(toName) : 0;
            if (toNode)
                fromNode->addRoute(r->fromEventOut(), toNode, r->toEventIn());
        }

    nodeStack.push_front(this);
    FieldList::iterator i;
    for (i = d_fields.begin(); i != d_fields.end(); ++i)
        if ((*i)->value)
        {
            if ((*i)->type == VrmlField::SFNODE)
            {
                const VrmlSFNode *sfn = (*i)->value->toSFNode();
                if (sfn && sfn->get() && !isOnStack(sfn->get()))
                {
                    sfn->get()->copyRoutes(ns);
                }
            }

            else if ((*i)->type == VrmlField::MFNODE && (*i)->value->toMFNode())
            {
                int nk = (*i)->value->toMFNode()->size();
                VrmlNode **kids = (*i)->value->toMFNode()->get();
                for (int k = 0; k < nk; ++k)
                    if (kids[k])
                    {
                        if (!isOnStack(kids[k]))
                            kids[k]->copyRoutes(ns);
                    }
            }
        }
    nodeStack.pop_front();
}

void VrmlNodeScript::addToScene(VrmlScene *s, const char *relUrl)
{
    System::the->debug("VrmlNodeScript::%s 0x%llx addToScene 0x%llx\n",
                       name(), (uint64_t)this, (uint64_t)s);

    d_relativeUrl.set(relUrl);
    if (d_scene == s)
        return;

    //addChildren to Scene as well
    nodeStack.push_front(this);
    FieldList::iterator i;
    for (i = d_fields.begin(); i != d_fields.end(); ++i)
        if ((*i)->value)
        {
            if ((*i)->type == VrmlField::SFNODE)
            {
                const VrmlSFNode *sfn = (*i)->value->toSFNode();
                if (sfn && sfn->get() && !isOnStack(sfn->get()))
                {
                    sfn->get()->addToScene(s, relUrl);
                }
            }

            else if ((*i)->type == VrmlField::MFNODE && (*i)->value->toMFNode())
            {
                int nk = (*i)->value->toMFNode()->size();
                VrmlNode **kids = (*i)->value->toMFNode()->get();
                for (int k = 0; k < nk; ++k)
                    if (kids[k])
                    {
                        if (!isOnStack(kids[k]))
                            kids[k]->addToScene(s, relUrl);
                        ;
                    }
            }
        }
    nodeStack.pop_front();

    d_scene = s;
    initialize(System::the->time());
    if (d_scene != 0)
        d_scene->addScript(this);
}

void VrmlNodeScript::initialize(double ts)
{
    System::the->debug("Script.%s::initialize\n", name());

    if (d_script)
        return; // How did I get here? Letting the days go by...
    d_eventsReceived = 0;
    if (d_url.size() > 0)
    {
        FieldList::const_iterator i;
        for (i = d_eventOuts.begin(); i != d_eventOuts.end(); ++i)
            (*i)->modified = false;
        for (i = d_fields.begin(); i != d_fields.end(); ++i)
            (*i)->modified = false;
        d_script = ScriptObject::create(this, d_url, d_relativeUrl);
        if (d_script)
            d_script->activate(ts, "initialize", 0, 0);

        // For each modified eventOut, send an event
        for (i = d_eventOuts.begin(); i != d_eventOuts.end(); ++i)
            if ((*i)->modified && (*i)->value != nullptr)
                eventOut(ts, (*i)->name, *((*i)->value));
        // For each modified exposedField, send an event
        for (i = d_fields.begin(); i != d_fields.end(); ++i)
            if ((*i)->exposed && (*i)->modified)
                eventOut(ts, (*i)->name, *((*i)->value));
    }
}

void VrmlNodeScript::shutdown(double ts)
{
    if (d_script)
        d_script->activate(ts, "shutdown", 0, 0);
}

void VrmlNodeScript::update(VrmlSFTime &timeNow)
{
    if (d_eventsReceived > 0)
    {
        //System::the->debug("Script.%s::update\n", name());
        if (d_script)
            d_script->activate(timeNow.get(), "eventsProcessed", 0, 0);
        d_eventsReceived = 0;
    }
}

//

void VrmlNodeScript::eventIn(double timeStamp,
                             const char *eventName,
                             const VrmlField *fieldValue)
{
    if (!d_script)
        initialize(timeStamp);
    if (!d_script)
        return;

    const char *origEventName = eventName;
    bool valid = hasEventIn(eventName);
    if (!valid && strncmp(eventName, "set_", 4) == 0)
    {
        eventName += 4;
        valid = hasEventIn(eventName);
    }
    if (valid)
    {
        setEventIn(eventName, fieldValue);
    }
    else
    {
        valid = hasField(eventName);
        if (!valid && strncmp(eventName, "set_", 4) == 0)
        {
            eventName += 4;
            valid = hasField(eventName);
        }
        setExposedField(eventName, fieldValue);
    }
    if (valid)
    {

        VrmlSFTime ts(timeStamp);
        const VrmlField *args[] = { fieldValue, &ts };

        FieldList::const_iterator i;
        for (i = d_eventOuts.begin(); i != d_eventOuts.end(); ++i)
            (*i)->modified = false;
        for (i = d_fields.begin(); i != d_fields.end(); ++i)
            (*i)->modified = false;

        d_script->activate(timeStamp, eventName, 2, args);

        // For each modified eventOut, send an event
        for (i = d_eventOuts.begin(); i != d_eventOuts.end(); ++i)
            if ((*i)->modified)
                eventOut(timeStamp, (*i)->name, *((*i)->value));

        // For each modified exposedField, send an event
        for (i = d_fields.begin(); i != d_fields.end(); ++i)
            if ((*i)->exposed && (*i)->modified)
                eventOut(timeStamp, (*i)->name, *((*i)->value));

        ++d_eventsReceived; // call eventsProcessed later
    }

    // Let the generic code handle the rest.
    else
        VrmlNode::eventIn(timeStamp, origEventName, fieldValue);
    // Scripts shouldn't generate redraws.
    clearModified();
}

// add events/fields

void VrmlNodeScript::addEventIn(const char *ename, VrmlField::VrmlFieldType t)
{
    add(d_eventIns, ename, t);
}

void VrmlNodeScript::addEventOut(const char *ename, VrmlField::VrmlFieldType t)
{
    add(d_eventOuts, ename, t);
}

void VrmlNodeScript::addField(const char *ename, VrmlField::VrmlFieldType t,
                              VrmlField *val, bool exposed)
{
    auto fieldIter = add(d_fields, ename, t, exposed);
    if (val)
    {
        set(d_fields, ename, val);
        auto f = field(ename, *(*fieldIter)->value);
        // initFieldsHelper(this, nullptr, field(ename, *(*fieldIter)->value), [this, fieldIter](){
        //     set(d_fields, (*fieldIter)->name, (*fieldIter)->value);
        // });
    }
}

void
VrmlNodeScript::addExposedField(const char *ename,
                                VrmlField::VrmlFieldType type,
                                VrmlField *defaultValue)
{
    char tmp[1000];
    auto fieldIter = add(d_fields, ename, type,true);
    if (defaultValue)
        set(d_fields, ename, defaultValue);

    sprintf(tmp, "set_%s", ename);
    add(d_eventIns, tmp, type);
    sprintf(tmp, "%s_changed", ename);
    add(d_eventOuts, tmp, type);
    // initFieldsHelper(this, 0, exposedField(ename, *(*fieldIter)->value), [this, fieldIter](){
    //     set(d_fields, (*fieldIter)->name, (*fieldIter)->value);
    // });
}

VrmlNodeScript::FieldList::iterator VrmlNodeScript::add(FieldList &recs,
                         const char *ename,
                         VrmlField::VrmlFieldType type,
                         bool exposed)
{
    ScriptField *r = new ScriptField;
    r->name = strdup(ename);
    r->type = type;
    r->value = 0;
    r->exposed = exposed;
    recs.push_front(r);
    return recs.begin();
}

// get event/field values
#if 0
VrmlField*
VrmlNodeScript::getEventIn(const char *fname) const
{
   return get(d_eventIns, fname);
}


VrmlField*
VrmlNodeScript::getEventOut(const char *fname) const
{
   return get(d_eventOuts, fname);
}
#endif

VrmlField *
VrmlNodeScript::get(const FieldList &recs, const char *fname) const
{
    FieldList::const_iterator i;
    for (i = recs.begin(); i != recs.end(); ++i)
    {
        if (strcmp((*i)->name, fname) == 0)
            return (*i)->value;
    }
    return 0;
}

// has

VrmlField::VrmlFieldType
VrmlNodeScript::hasEventIn(const char *ename) const
{
    return has(d_eventIns, ename);
}

VrmlField::VrmlFieldType
VrmlNodeScript::hasEventOut(const char *ename) const
{
    return has(d_eventOuts, ename);
}

VrmlField::VrmlFieldType
VrmlNodeScript::hasField(const char *ename) const
{
    return has(d_fields, ename);
}

VrmlField::VrmlFieldType
VrmlNodeScript::hasExposedField(const char *ename) const
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
VrmlNodeScript::has(const FieldList &recs, const char *ename) const
{
    FieldList::const_iterator i;
    for (i = recs.begin(); i != recs.end(); ++i)
    {
        if (strcmp((*i)->name, ename) == 0)
            return (*i)->type;
    }
    return VrmlField::NO_FIELD;
}

void
VrmlNodeScript::setEventIn(const char *fname, const VrmlField *value)
{
    set(d_eventIns, fname, value);
}

void
VrmlNodeScript::setEventOut(const char *fname, const VrmlField *value)
{
    set(d_eventOuts, fname, value);
}


void
VrmlNodeScript::setExposedField(const char* fname, const VrmlField* value)
{
    set(d_fields, fname, value);
}

void
VrmlNodeScript::set(const FieldList &recs,
                    const char *fname,
                    const VrmlField *value)
{
    FieldList::const_iterator i;
    for (i = recs.begin(); i != recs.end(); ++i)
    {
        if (strcmp((*i)->name, fname) == 0)
        {
            delete ((*i)->value);
            (*i)->value = value->clone();
            (*i)->modified = true;
            return;
        }
    }
}
