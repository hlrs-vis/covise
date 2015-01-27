/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_LABELMENUITEM_H
#define CO_LABELMENUITEM_H

#include <OpenVRUI/coRowMenuItem.h>

#include <string>

/** This class defines a menu item which can be used as a separator.
  it does not generate events.
*/
namespace vrui
{

class OPENVRUIEXPORT coLabelMenuItem : public coRowMenuItem
{
public:
    coLabelMenuItem(const std::string &);
    virtual ~coLabelMenuItem();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;
};
}
#endif
