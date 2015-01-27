/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SCENE_OBJECT_TYPES_H
#define SCENE_OBJECT_TYPES_H

class SceneObjectTypes
{
public:
    enum Type
    {
        NONE,
        ASSET,
        ROOM,
        SHAPE,
        GROUND,
        LIGHT,
        WINDOW
    };
};

#endif
