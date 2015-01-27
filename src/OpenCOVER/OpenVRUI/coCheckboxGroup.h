/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_CHECKBOXGROUP_H
#define CO_CHECKBOXGROUP_H

#include <util/coTypes.h>

#include <list>

namespace vrui
{

class coCheckboxMenuItem;

/** This class provides a mechanism to implement radio buttons.
  coCheckboxMenuItems can be added to the group and thus are treated
  as radio buttons (only one item can be selected at a time).
  If the constructor is called with the argument true, all checkboxes
  of the group are be deselected if the checked item is clicked on.
*/
class OPENVRUIEXPORT coCheckboxGroup
{
protected:
    std::list<coCheckboxMenuItem *> itemList; ///< list of items in the group
    coCheckboxMenuItem *selected; ///< pointer to selected checkbox
    bool allowDeselect; ///< true = all checkboxes can be deselected, false = exactly one checkbox is selected at any time

public:
    coCheckboxGroup(bool = false);
    coCheckboxMenuItem *getSelectedCheckbox();
    void toggleCheckbox(coCheckboxMenuItem *checkbox);
    void setState(coCheckboxMenuItem *checkbox, bool newState, bool generateEvent = false);
    void add(coCheckboxMenuItem *checkbox);
    void remove(coCheckboxMenuItem *checkbox);
    bool getAllowDeselect() const;
};
}
#endif
