/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_HEIGHTFIELD_PICK_BOX_H
#define _CUI_HEIGHTFIELD_PICK_BOX_H

#include "Widget.h"
#include "PickBox.h"

namespace cui
{
class CUIEXPORT HeightFieldPickBox : public PickBox
{
public:
    HeightFieldPickBox(Interaction *, const osg::Vec4 &, const osg::Vec4 &, const osg::Vec4 &);
    virtual ~HeightFieldPickBox();
    void createHeightField(int, int, int, unsigned char *);

private:
    osg::Geode *_geode;
};
}

#endif
