/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRMLNODEInfobox_
#define _VRMLNODEInfobox_

#include <string>
#include <string_view>
#include <util/coTypes.h>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

using namespace vrml;

class VrmlNodeInfobox : public VrmlNodeChild
{
public:
    VrmlNodeInfobox(VrmlScene *scene = 0);
    VrmlNodeInfobox(const VrmlNodeInfobox &n);

    static void initFields(VrmlNodeInfobox *node, vrml::VrmlNodeType *t);
    static const char *typeName();

    void setImageUrl(std::string_view url);
    void updateValue();

    virtual void addToScene(VrmlScene *scene, const char *relativeUrl)
    {
        std::cout << "Add to scene " << relativeUrl << std::endl;
        m_relativeUrl = relativeUrl == NULL ? "" : std::string(relativeUrl);
        VrmlNodeChild::addToScene(scene, relativeUrl);

        std::string url = d_imageUrl.get();
        if (!url.empty())
        {
            setImageUrl(url);
        }
    }

private:
    VrmlSFString d_imageUrl { "" };
    VrmlSFFloat d_value { 0.0f };
    VrmlSFString d_unit { "" };
    VrmlSFBool d_hasValue { false };

    std::string m_relativeUrl { "" };
};
#endif
