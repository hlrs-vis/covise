/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeInline.cpp
//

#include "config.h"
#include "VrmlNodeInline.h"

#include "VrmlNamespace.h"
#include "VrmlNodeType.h"
#include "Doc.h"
#include "MathUtils.h"
#include "VrmlScene.h"
#include "Viewer.h"
#include "System.h"
#include <errno.h>

#include <string>

using namespace vrml;

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeInline(scene);
}

// Define the built in VrmlNodeType:: "Inline" fields

VrmlNodeType *VrmlNodeInline::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;
    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("Inline", creator);
    }

    // Having Inline a subclass of Group is not right since
    // Groups have an exposedField "children" and eventIns
    // addChildren/deleteChildren that Inlines don't support...
    VrmlNodeGroup::defineType(t); // Parent class
    t->addExposedField("url", VrmlField::MFSTRING);

    return t;
}

VrmlNodeType *VrmlNodeInline::nodeType() const { return defineType(0); }

VrmlNodeInline::VrmlNodeInline(VrmlScene *scene)
    : VrmlNodeGroup(scene)
    , d_namespace(0)
    , sgObject(0)
    , d_hasLoaded(false)
{
}

VrmlNodeInline::~VrmlNodeInline()
{
    delete d_namespace;
    d_isDeletedInline = true;
}

VrmlNode *VrmlNodeInline::cloneMe() const
{
    return new VrmlNodeInline(*this);
}

VrmlNodeInline *VrmlNodeInline::toInline() const
{
    return (VrmlNodeInline *)this;
}

// Inlines are loaded during addToScene traversal

void VrmlNodeInline::addToScene(VrmlScene *s, const char *relativeUrl)
{
    d_scene = s;
    load(relativeUrl);
    VrmlNodeGroup::addToScene(s, relativeUrl);
}

std::ostream &VrmlNodeInline::printFields(std::ostream &os, int indent)
{
    if (!FPZERO(d_bboxCenter.x()) || !FPZERO(d_bboxCenter.y()) || !FPZERO(d_bboxCenter.z()))
        PRINT_FIELD(bboxCenter);

    if (!FPEQUAL(d_bboxSize.x(), -1) || !FPEQUAL(d_bboxSize.y(), -1) || !FPEQUAL(d_bboxSize.z(), -1))
        PRINT_FIELD(bboxCenter);

    if (d_url.get())
        PRINT_FIELD(url);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeInline::setField(const char *fieldName,
                              const VrmlField &fieldValue)
{
    if
        TRY_FIELD(url, MFString)
    else
        VrmlNodeGroup::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeInline::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "url") == 0)
        return &d_url;
    else if (strcmp(fieldName, "bboxCenter") == 0)
        return &d_bboxCenter;
    else if (strcmp(fieldName, "bboxSize") == 0)
        return &d_bboxSize;

    return VrmlNode::getField(fieldName); // no other fields
}

// Render each of the children

void VrmlNodeInline::render(Viewer *viewer)
{
    if (!haveToRender())
        return;

    if (d_wasCached)
        return;

    if (isModified())
    {
        if (sgObject) // we have a cached Object, so add it to the viewer
        {
            d_wasCached = System::the->getCacheMode() != System::CACHE_DISABLE;
            d_viewerObject = viewer->beginObject(name(), 0, this);
            System::the->insertObject(d_viewerObject, sgObject);
            viewer->endObject();
            clearModified();
        }
        else // render the children an store the viewerObject in the cache
        {
            VrmlNodeGroup::render(viewer);
            if (d_viewerObject && (strstr(name(), "NotCached") == NULL) && isOnlyGeometry())
            {
                d_wasCached = System::the->getCacheMode() != System::CACHE_DISABLE;
                Doc url;
                const char *pathname = NULL;
                if (d_relative.get())
                {
                    Doc relDoc(d_relative.get());
                    url.seturl(d_url.get(0), &relDoc);
                    pathname = url.localName();
                }
                d_scene->storeCachedInline(d_url.get(0), pathname, d_viewerObject);
            }
        }
    }
    else
    {
        VrmlNodeGroup::render(viewer);
    }
}

//  Load the children from the URL

void VrmlNodeInline::load(const char *relativeUrl)
{
    if (!relativeUrl)
        //URL is empty, should not occur -> return
        return;

    // Already loaded? Need to check whether Url has been modified...
    if (d_hasLoaded)
        return;

    d_hasLoaded = true; // although perhaps not successfully

    if (d_url.size() > 0)
    {

        //std::cerr << "Loading VRML inline!" << std::endl;
        //std::cerr << "parentFilename: " << relativeUrl << std::endl;
        //std::cerr << "url: " << d_url.get(0) << std::endl;

        Doc url;
        const Doc relDoc(relativeUrl);
        System::the->debug("Trying to read url '%s' (relative %s)\n",
                           d_url.get(0), d_relative.get() ? d_relative.get() : "<null>");
        url.seturl(d_url.get(0), &relDoc);

        sgObject = 0L;
        if (d_url.get(0) && strlen(d_url.get(0))>0)
        {
#if 1
            sgObject = d_scene->getCachedInline(d_url.get(0), url.localName()); // relative files in cache
            if (sgObject)
            {
                setModified();
            }
#else
            if (strstr(name(), "Cached") != NULL)
            {
                setModified();
                sgObject = d_scene->getCachedInline(d_url.get(0), url.localName()); // relative files in cache
            }
#endif
            if ((sgObject == 0L) && !VrmlScene::isWrl(url.url()))
            {
                sgObject = System::the->getInline(url.url());
                setModified();
            }
        }
        if (sgObject == 0L)
        {
            VrmlNamespace *ns = new VrmlNamespace();
            VrmlMFNode *kids = 0;
            Doc url;
            int i, n = d_url.size();
            for (i = 0; i < n; ++i)
            {
                if (!d_url.get(i) || strlen(d_url.get(i))==0)
                    continue;
                System::the->debug("Trying to read url '%s' (relative %s)\n",
                                   d_url.get(i), d_relative.get() ? d_relative.get() : "<null>");
                url.seturl(d_url.get(i), &relDoc);

                kids = VrmlScene::readWrl(&url, ns);
                if (kids)
                    break;
                else if (i < n - 1 && strncmp(d_url.get(i), "urn:", 4))
                    System::the->warn("Couldn't read url '%s': %s\n",
                                      d_url.get(i), strerror(errno));
            }

            if (kids)
            {
                delete d_namespace;
                d_namespace = ns;
                d_relative.set(url.url()); // children will be relative to this url

                removeChildren();
                addChildren(*kids); // check for nested Inlines

                delete kids;
            }
            else
            {
                System::the->warn("VRMLInline::load: couldn't load Inline %s (relative %s)\n",
                                  d_url[0],
                                  d_relative.get() ? d_relative.get() : "<null>");
                delete ns;
            }
        }
    }
}

VrmlNode *VrmlNodeInline::findInside(const char *exportName)
{
    load(d_url.get(0));
    if (d_namespace)
    {
        std::string asName = d_namespace->getExportAs(exportName);
        if (asName.size() == 0)
        {
            fprintf(stderr, "Warning: can not find EXPORT %s in %s\n",
                    exportName, d_url.get(0));
            return d_namespace->findNode(exportName);
        }
        return d_namespace->findNode(asName.data());
    }
    return NULL;
}
