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

void VrmlNodeInline::initFields(VrmlNodeInline *node, VrmlNodeType *t)
{
    VrmlNodeGroup::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("url", node->d_url));

}

const char *VrmlNodeInline::name() { return "Inline"; }

VrmlNodeInline::VrmlNodeInline(VrmlScene *scene)
    : VrmlNodeGroup(scene, name())
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

VrmlNodeInline *VrmlNodeInline::toInline() const
{
    return (VrmlNodeInline *)this;
}

// Inlines are loaded during addToScene traversal

void VrmlNodeInline::addToScene(VrmlScene *s, const char *relativeUrl)
{
    d_scene = s;
	if (s)
	{
		load(relativeUrl, System::the->getFileId(s->urlDoc()->url()));
	}
	else
	{
		load(relativeUrl);
	}
    VrmlNodeGroup::addToScene(s, relativeUrl);
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
            if (d_viewerObject && isOnlyGeometry())
            {
                d_wasCached = System::the->getCacheMode() != System::CACHE_DISABLE;
                Doc url;
                std::string pathname;
                if (d_relative.get())
                {
                    Doc relDoc(d_relative.get());
                    url.seturl(d_url.get(0), &relDoc);
                    pathname = url.localName();
                }
                d_scene->storeCachedInline(d_url.get(0), pathname.c_str(), d_viewerObject);
            }
        }
    }
    else
    {
        VrmlNodeGroup::render(viewer);
    }
}

//  Load the children from the URL

void VrmlNodeInline::load(const char *relativeUrl, int parentId)
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
            if (isOnlyGeometry())
            {
                sgObject = d_scene->getCachedInline(d_url.get(0), url.localName().c_str()); // relative files in cache
                if (sgObject)
                {
                    setModified();
                }
            }
            if ((sgObject == 0L) && !VrmlScene::isWrl(url.url()))
            {
                sgObject = System::the->getInline(url.url().c_str());
                setModified();
            }
        }
        if (sgObject == 0L)
        {
            VrmlNamespace *ns = new VrmlNamespace(parentId);
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
                d_relative.set(url.url().c_str()); // children will be relative to this url

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
