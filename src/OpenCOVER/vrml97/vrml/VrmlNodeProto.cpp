/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  VrmlNodeProto handles instances of nodes defined via PROTO
//  statements.
//
//  Instances of PROTOs clone the implementation nodes stored
//  in a VrmlNodeType object. The only tricky parts are to
//  make sure ROUTEs are properly copied (the DEF name map is
//  not available) and that fields are copied properly (the
//  MF* guys currently share data & need to be made copy-on-
//  write for this to be correct). Flags are set as each node
//  is cloned so that USEd nodes are referenced rather than
//  duplicated.
//
//  ROUTEs: Build a temp namespace as each (named) implementation
//  node is cloned, then traverse the implementation nodes again,
//  reproducing the routes in the cloned nodes using the temp ns.
//  I think that the addToScene() method is the right place to
//  download EXTERNPROTO implementations. Should also check that
//  the first node matches the expected type (need to store the
//  expected type if first node is null when one of the type
//  tests is run).
//
//  Events between nodes in the PROTO implementation are handled
//  by the ROUTE copying described above. For eventIns coming into
//  the proto, when the implementation nodes are copied, a list
//  of eventIns/exposedFields along with their IS mappings should
//  be constructed.
//  EventOuts from an implementation node to a node outside the
//  PROTO can be directly replaced at copy time.
//

#include "VrmlNodeProto.h"
#include "VrmlNamespace.h"
#include "VrmlMFNode.h"
#include "VrmlScene.h"
#include "System.h"
#include "Doc.h"
using std::list;
using namespace vrml;

VrmlNodeType *VrmlNodeProto::nodeType() const { return d_nodeType; }

void VrmlNodeProto::initFields(VrmlNodeProto *node, VrmlNodeType *t)
{

}

const char *VrmlNodeProto::name()
{
    return "PROTO";
}

VrmlNodeProto::VrmlNodeProto(VrmlNodeType *nodeDef, VrmlScene *scene)
    : VrmlNode(scene, name())
    , d_nodeType(nodeDef->reference())
    , d_instantiated(false)
    , d_scope(0)
    , d_nodes(0)
    , d_viewerObject(0)
{
}

VrmlNodeProto::VrmlNodeProto(const VrmlNodeProto &n)
    : VrmlNode(n)
    , d_nodeType(n.nodeType()->reference())
    , d_instantiated(false)
    , d_scope(0)
    , d_nodes(0)
    , d_viewerObject(0)
{
    // Copy fields.
    list<NameValueRec *>::const_iterator i;
    for (i = n.d_fields.begin(); i != n.d_fields.end(); i++)
    {
        VrmlField *value = (*i)->value;
        setField((*i)->name, *value);
    }
}

VrmlNodeProto::~VrmlNodeProto()
{
    // Free strings & values
    list<NameValueRec *>::iterator i;

    for (i = d_fields.begin(); i != d_fields.end(); i++)
    {
        NameValueRec *r = *i;
        free(r->name);
        delete r->value;
        delete r;
    }

    EventDispatchList::iterator e;
    for (e = d_eventDispatch.begin(); e != d_eventDispatch.end(); ++e)
    {
        EventDispatch *ed = *e;
        delete[] ed -> name;
        VrmlNodeType::ISMap::iterator j;
        for (j = ed->ismap.begin(); j != ed->ismap.end(); ++j)
        {
            VrmlNodeType::NodeFieldRec *nf = (*j);
            delete[] nf -> fieldName;
            delete nf;
        }
        delete ed;
    }

    delete d_nodes;
    delete d_scope;
    d_nodeType->dereference();
}

// Instantiate a local copy of the implementation nodes.
// EXTERNPROTOs are actually loaded here. We don't want
// *references* (DEF/USE) to the nodes in the PROTO definition,
// we need to actually clone new nodes...

void VrmlNodeProto::instantiate(const char* relUrl, int parentId)
{
    System::the->debug("%s::%s instantiate\n", d_nodeType->getName(),
                       name());

    if (!d_nodes)
    {
        VrmlMFNode *protoNodes = d_nodeType->getImplementationNodes(parentId);
        int nNodes = protoNodes ? protoNodes->size() : 0;
        int i;

        d_scope = new VrmlNamespace(parentId);
        // Clear all flags - encountering a set flag during cloning
        // indicates a USEd node, which should be referenced rather
        // than cloned.
        for (i = 0; i < nNodes; ++i)
            protoNodes->get(i)->clearFlags();

        // Clone nodes
        // Those squeamish about broken encapsulations shouldn't look...
        d_nodes = new VrmlMFNode(nNodes, 0);
        VrmlNode **clone = d_nodes->get();
        for (i = 0; i < nNodes; ++i)
        {
            clone[i] = protoNodes->get(i)->clone(d_scope)->reference();
            clone[i]->parentList.push_back(this);
            if (clone[i]->getTraversalForce() > 0)
            {
                forceTraversal(false, clone[i]->getTraversalForce());
            }
        }

        // Copy internal (to the PROTO implementation) ROUTEs.
        for (i = 0; i < nNodes; ++i)
            protoNodes->get(i)->copyRoutes(d_scope);

        // Collect eventIns coming from outside the PROTO.
        // A list of eventIns along with their maps to local
        // nodes/eventIns is constructed for each instance.
        VrmlNodeType::FieldList &eventIns = d_nodeType->eventIns();
        VrmlNodeType::FieldList::iterator ev;
        VrmlNodeType::ISMap *ismap;
        VrmlNodeType::ISMap::const_iterator j;

        for (ev = eventIns.begin(); ev != eventIns.end(); ++ev)
        {
            EventDispatch *ed = new EventDispatch;
            char *eventName = (*ev)->name;

            ed->name = new char[strlen(eventName) + 1];
            strcpy(ed->name, eventName);
            ismap = &(*ev)->thisIS;
            d_eventDispatch.push_front(ed);

            for (j = ismap->begin(); j != ismap->end(); ++j)
            {
                VrmlNodeType::NodeFieldRec *nf = new VrmlNodeType::NodeFieldRec;
                nf->node = d_scope->findNode((*j)->node->name());
                nf->fieldName = new char[strlen((*j)->fieldName) + 1];
                strcpy(nf->fieldName, (*j)->fieldName);
                ed->ismap.push_front(nf);
            }
        }

        // Distribute eventOuts. Each eventOut ROUTE is added
        // directly to the local nodes that have IS'd the PROTO
        // eventOut.
        VrmlNodeType::FieldList &eventOuts = d_nodeType->eventOuts();
        for (Route *r = d_routes; r; r = r->next())
            for (ev = eventOuts.begin(); ev != eventOuts.end(); ++ev)
                if (strcmp((*ev)->name, r->fromEventOut()) == 0)
                {
                    ismap = &(*ev)->thisIS;
                    for (j = ismap->begin(); j != ismap->end(); ++j)
                    {
                        VrmlNode *n = d_scope->findNode((*j)->node->name());
                        if (n)
                            n->addRoute((*j)->fieldName,
                                        r->toNode(), r->toEventIn());
                    }
                }

        // Set IS'd field values in the implementation nodes to
        // the values specified in the instantiation.

        list<NameValueRec *>::iterator ifld;
        for (ifld = d_fields.begin(); ifld != d_fields.end(); ++ifld)
        {
            VrmlField *value = (*ifld)->value;
#ifdef DEBUG
            cerr << d_nodeType->getName() << "::" << name()
                 << " setting IS field " << (*ifld)->name;
            if (value)
                cerr << " to " << *value << endl;
            else
                cerr << " to null\n";
#endif
            if (!value)
                continue;
            if ((ismap = d_nodeType->getFieldISMap((*ifld)->name)) != 0)
            {
                for (j = ismap->begin(); j != ismap->end(); ++j)
                {
                    //    cerr << (*j)->node->name() << endl;
                    VrmlNode *n = d_scope->findNode((*j)->node->name());
#ifdef DEBUG
                    cerr << " on " << n->name() << "::" << (*j)->fieldName << endl;
#endif
                    if (n)
                        n->setField((*j)->fieldName, *value);
                }
            }
        }
    }

    d_instantiated = true;
}

void VrmlNodeProto::addToScene(VrmlScene *s, const char *relUrl)
{
    System::the->debug("VrmlNodeProto::%s addToScene\n", name());
    d_scene = s;
	int parentId = -1;
	if (s)
	{
		parentId = System::the->getFileId(s->urlDoc()->url());
	}
    // Make sure my nodes are here
    if (!d_instantiated)
        instantiate(relUrl, parentId);
    System::the->debug("VrmlNodeProto::%s addToScene(%d nodes)\n",
                       name(), d_nodes ? d_nodes->size() : 0);

    // ... and add the implementation nodes to the scene.
    if (d_nodes)
    {
        const char *rel = d_nodeType->url();
        int j, n = d_nodes->size();
        for (j = 0; j < n; ++j)
            d_nodes->get(j)->addToScene(s, rel ? rel : relUrl);
    }
}

void VrmlNodeProto::accumulateTransform(VrmlNode *n)
{
    // Make sure my nodes are here
    if (!d_instantiated)
    {
        System::the->debug("VrmlNodeProto::%s accumTrans before instantiation\n",
                           name());
        instantiate();
    }

    // ... and process the implementation nodes.
    if (d_nodes)
    {
        int i, j = d_nodes->size();
        for (i = 0; i < j; ++i)
            d_nodes->get(i)->accumulateTransform(n);
    }
}

int VrmlNodeProto::size()
{
    return d_nodes ? d_nodes->size() : 0;
}

VrmlNode *VrmlNodeProto::child(int index)
{
    if (d_nodes && index >= 0 && index < d_nodes->size())
        return (*d_nodes)[index];
    return 0;
}

// Print the node type, instance vars

std::ostream &VrmlNodeProto::printFields(std::ostream &os, int) const
{
    os << "#VrmlNodeProto::printFields not implemented yet...\n";
    return os;
}

// Use the first node to check the type

VrmlNode *VrmlNodeProto::firstNode() const
{
    return ((d_nodes && d_nodes->size())
                ? d_nodes->get(0)
                : d_nodeType->firstNode());
}

VrmlNode *VrmlNodeProto::getThisProto()
{
    return firstNode();
}

const VrmlNode *VrmlNodeProto::getThisProto() const 
{
    return firstNode();
}


bool VrmlNodeProto::isModified() const
{
    if (d_modified)
        return true;

    int n = 0;
    if (d_nodes)
        n = d_nodes->size();

    for (int i = 0; i < n; ++i)
        if (d_nodes->get(i)->isModified())
            return true;

    return false;
}

void VrmlNodeProto::render(Viewer *viewer)
{
    if (!haveToRender())
        return;

    if (!d_instantiated)
    {
        System::the->debug("VrmlNodeProto::%s render before instantiation\n",
                           name());
        instantiate();
    }
    if (d_viewerObject && isModified())
    {
        viewer->removeObject(d_viewerObject);
        d_viewerObject = 0;
    }

    if (d_viewerObject)
        viewer->insertReference(d_viewerObject);

    else if (d_nodes)
    {
        d_viewerObject = viewer->beginObject(name(), 0, this);

        // render the nodes with the new values
        int n = d_nodes->size();
        for (int j = 0; j < n; ++j)
            d_nodes->get(j)->render(viewer);

        viewer->endObject();
    }

    clearModified();
}

void VrmlNodeProto::eventIn(double timeStamp,
                            const char *eventName,
                            const VrmlField *fieldValue)
{
    if (!d_instantiated)
    {
        System::the->debug("VrmlNodeProto::%s eventIn before instantiation\n",
                           name());
        instantiate();
    }
    char *localEventName = NULL;

    const char *newEventName = eventName;
    if (strncmp(eventName, "set_", 4) == 0)
        newEventName += 4;
    else
    {
        localEventName = new char[strlen(eventName) + 5];
        newEventName = localEventName;
        strcpy(localEventName, "set_");
        strcat(localEventName, eventName);
    }

#if 0
   cerr << "eventIn " << nodeType()->getName()
      << "::" << name() << "." << origEventName
      << " " << (*fieldValue) << endl;
#endif

    EventDispatchList::iterator i;
    for (i = d_eventDispatch.begin(); i != d_eventDispatch.end(); ++i)
    {
        if (strcmp(eventName, (*i)->name) == 0 || strcmp(newEventName, (*i)->name) == 0)
        {
            VrmlNodeType::ISMap *ismap = &((*i)->ismap);
            VrmlNodeType::ISMap::iterator j;
            for (j = ismap->begin(); j != ismap->end(); ++j)
                (*j)->node->eventIn(timeStamp, (*j)->fieldName, fieldValue);
            if (localEventName)
                delete[] localEventName;
            return;
        }
    }

    if (localEventName)
        delete[] localEventName;
    // Let the generic code handle errors.
    VrmlNode::eventIn(timeStamp, eventName, fieldValue);
}

VrmlNodeProto::NameValueRec *VrmlNodeProto::findField(const char *fieldName) const
{
    list<NameValueRec *>::const_iterator i;
    for (i = d_fields.begin(); i != d_fields.end(); ++i)
    {
        NameValueRec *nv = *i;
        if (nv != NULL && strcmp(nv->name, fieldName) == 0)
        {
            return nv;
        }
    }
    return NULL;
}

// Set the value of one of the node fields (creates the field if
// it doesn't exist - is that necessary?...)

void VrmlNodeProto::setField(const char *fieldName,
                             const VrmlField &fieldValue)
{
    NameValueRec *nv = findField(fieldName);

    if (!nv)
    {
        nv = new NameValueRec;
        nv->name = strdup(fieldName);
        d_fields.push_front(nv);
    }
    else
    {
        delete nv->value;
    }

    nv->value = fieldValue.clone();
}

const VrmlField *VrmlNodeProto::getField(const char *fieldName) const
{
    NameValueRec *nv = findField(fieldName);
    if (nv)
        return nv->value;

    return VrmlNode::getField(fieldName); // no other fields
}
