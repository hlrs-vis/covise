/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef GEO_DATA_INFOBOX_H
#define GEO_DATA_INFOBOX_H

#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coUIElement.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coTexturedBackground.h>
#include <osg/Texture2D>
#include <string>
#include <cover/coVRPlugin.h>

#include <cover/ui/Owner.h>

class Infobox : public opencover::coVRPlugin,
                public opencover::ui::Owner
{
public:
    Infobox();
    ~Infobox();

    static Infobox *instance();

    bool init();

    void setContents(std::string_view imageUrl);
    void setValue(double value);

    void show();
    void hide();

private:
    vrui::coPanel *m_panel;
    vrui::coFrame *m_frame;
    vrui::coPopupHandle *m_popupHandle;
    vrui::coTexturedBackground *m_imageBackground;

    static Infobox *s_instance;
};
#endif
