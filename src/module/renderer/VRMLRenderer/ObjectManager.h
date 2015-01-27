/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _OBJECT_MANAGER
#define _OBJECT_MANAGER

/**************************************************************************\
**                                                           (C)1995 RUS  **
**                                                                        **
** Description: Framework class for COVISE renderer modules               **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author:                                                                **
**                                                                        **
**                             Dirk Rantzau                               **
**                Computer Center University of Stuttgart                 **
**                            Allmandring 30                              **
**                            70550 Stuttgart                             **
**                                                                        **
** Date:  11.09.95  V1.0                                                  **
\**************************************************************************/

#include <appl/RenderInterface.h>
#include "GeometryManager.h"
#include "ObjectList.h"

namespace covise
{
class coDistributedObject;
class coDoGeometry;
}

#define MAXSETS 8000

#define CO_PER_VERTEX 0
#define CO_PER_FACE 1
#define CO_NONE 2
#define CO_OVERALL 3
#define CO_RGBA 4

//================================================================
// ObjectManager
//================================================================

class ObjectManager
{
private:
    //
    // coDoSet handling
    //
    int anzset;
    char *setnames[MAXSETS];
    int elemanz[MAXSETS];
    char **elemnames[MAXSETS];

    const char *filename;

    GeometryManager *gm;

    ObjectList *list;

    int file_writing;

    void addObjectToList(char *name, void *objPtr);
    void deleteObjectFromList(char *name);

    //void add_geometry(char *object, int doreplace,int is_timestep,
    //                 char *root,coDistributedObject *geometry,
    //		     coDistributedObject *normals,
    //	     coDistributedObject *colors);

    void add_geometry(const char *object, int is_timestep, const char *root,
                      const coDistributedObject *geometry, const coDistributedObject *normals,
                      const coDistributedObject *colors, const coDoGeometry *container);

    void remove_geometry(char *name);

public:
    ObjectManager();
    void deleteObject(char *name);
    void addObject(char *name);
    void addFeedbackButton(const char *object, const char *feedback_info);
    void set_write_file(int write_mode);
    ~ObjectManager()
    {
        delete list;
        delete gm;
    }
    void setFilename(const char *fn)
    {
        filename = fn;
    }
};

#endif
