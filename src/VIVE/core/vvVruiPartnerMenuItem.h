/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <OpenVrui/coCheckboxMenuItem.h>
#include <vsg/maths/vec3.h>

namespace osgUtil
{
class Hit;
}

namespace vive
{
/** Menu Item that reflects the state of a participant in a CVE
 */

class VVCORE_EXPORT vvVruiPartnerMenuItem : public vrui::coCheckboxMenuItem
{
protected:
    vrui::coButton *viewpoint; ///< actual button which is used for interaction

public:
    vvVruiPartnerMenuItem(const std::string &name, bool on, vrui::coCheckboxGroup * = NULL);
    ~vvVruiPartnerMenuItem() override;
    int hit(vrui::vruiHit *hit) override;
    void miss() override;
    void buttonEvent(vrui::coButton *button) override;

    /// get the Element's classname
    const char *getClassName() const override;
    /// check if the Element or any ancestor is this classname
    bool isOfClassName(const char *) const override;
};
}
