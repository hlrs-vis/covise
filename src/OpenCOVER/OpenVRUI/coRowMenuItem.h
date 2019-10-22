/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_ROWMENUITEM_H
#define CO_ROWMENUITEM_H

#include <OpenVRUI/coMenuItem.h>

namespace vrui
{

class coMenuContainer;
class coColoredBackground;
class coMenu;
class coLabel;

/** This is the base class of all row menu items.
  It provides a container for menu elements and a label string.
  @see coButtonMenuItem
  @see coLabelMenuItem
  @see coSubMenuItem
  @see coPotiMenuItem
  @see coSliderMenuItem
  @see coCheckboxMenuItem
*/
class OPENVRUIEXPORT coRowMenuItem : public coMenuItem
{
protected:
    static const int LEFTMARGIN; ///< size of margin at left edge of menu item
    coMenuContainer *container; ///< container to store menu elements
    coColoredBackground *background; ///< menu item background which changes its color when menu item is selected
    coLabel *label; ///< label text

public:
    coRowMenuItem(const std::string &labelString);
    coRowMenuItem(const std::string &symbolicName, const std::string &labelString);
    virtual ~coRowMenuItem();
    virtual void setLabel(coLabel *label);
    coLabel *getLabel();
    void setLabel(const std::string &labelString) override;
    coUIElement *getUIElement() override;

    /// get the Element's classname
    const char *getClassName() const override;
    /// check if the Element or any ancestor is this classname
    bool isOfClassName(const char *) const override;

    /// activates or deactivates the item
    void setActive(bool a) override;

    void selected(bool selected) override;

    void setVisible(bool visible) override;
};
}
#endif
