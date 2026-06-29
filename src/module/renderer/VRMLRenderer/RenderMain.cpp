/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/RenderInterface.h>
#include "Renderer.h"

ObjectList *objlist;
int main(int argc, char *argv[])
{

    Renderer *renderer = new Renderer(argc, argv);
    objlist = new ObjectList();
    objlist->setOutputMode(renderer->outputMode);
    //renderer->run();
    renderer->start();

    return 0;
}
