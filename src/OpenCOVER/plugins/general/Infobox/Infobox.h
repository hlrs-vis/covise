/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef GEO_DATA_INFOBOX_H
#define GEO_DATA_INFOBOX_H

#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coUIElement.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coTexturedBackground.h>
#include <osg/Texture2D>
#include <optional>
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

    void setImage(std::optional<std::string_view> imageUrl);
    void setValue(std::optional<double> value, std::optional<std::string_view> unit = std::nullopt);

private:
    void autoHideShow();

    vrui::coPanel *m_panel;
    vrui::coFrame *m_frame;
    vrui::coColoredBackground *m_background;
    vrui::coRowContainer *m_container;
    vrui::coPopupHandle *m_popupHandle;
    vrui::coTexturedBackground *m_imageBackground;

    vrui::coRowContainer *m_valueRowContainer;
    vrui::coLabel *m_valueLabelLabel;
    vrui::coLabel *m_valueLabel;

    static Infobox *s_instance;

    bool m_hasValue { false };
    bool m_hasImage { false };
};
#endif
