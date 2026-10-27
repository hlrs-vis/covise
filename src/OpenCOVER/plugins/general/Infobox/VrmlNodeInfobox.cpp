/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "VrmlNodeInfobox.h"

#include <filesystem>
#include <vrml97/vrml/Doc.h>
#include <vrml97/vrml/Image.h>

#include "Infobox.h"

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeInfobox(scene);
}

VrmlNodeInfobox::VrmlNodeInfobox(VrmlScene *scene)
    : VrmlNodeChild(scene, typeName())
{
}

VrmlNodeInfobox::VrmlNodeInfobox(const VrmlNodeInfobox &n)
    : VrmlNodeChild(n)
{
}

void VrmlNodeInfobox::initFields(VrmlNodeInfobox *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
        exposedField("imageUrl", &VrmlNodeInfobox::d_imageUrl, [node](auto f)
            { node->setImageUrl(f->get()); }));
}

const char *VrmlNodeInfobox::typeName()
{
    return "Infobox";
}

void VrmlNodeInfobox::setImageUrl(std::string_view url)
{
    auto infobox = Infobox::instance();
    if (!infobox)
        return;

    if (url.empty())
    {
        infobox->setImage(std::nullopt);
        return;
    }

    std::filesystem::path path(url);
    if (path.is_relative() && !std::filesystem::exists(path))
    {
        path = std::filesystem::path(m_relativeUrl).parent_path() / path;
    }

    infobox->setImage(path.string());
}
