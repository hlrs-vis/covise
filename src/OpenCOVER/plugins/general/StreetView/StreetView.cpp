/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2016 HLRS  **
 **                                                                          **
 ** Description: Streetview Plugin				                             **
 **                                                                          **
 **                                                                          **
 ** Author: M.Guedey		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Sep-16  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "StreetView.h"
#include "Picture.h"
#include "IndexParser.h"

#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>


using namespace opencover;

StreetView::StreetView()
{
}

bool StreetView::init()
{
   fprintf(stderr, "StreetView::init\n");

   indexParser = new IndexParser();
   indexParser->parseIndex("\\\\VISFS1\\raid\\share\\projects\\reallabor\\Herrenberg\\Daten\\vonHerrenberg\\Panorama\\P09299_Herrenberg\\EBF");
   indexParser->removeDuplicateEntries();
   //indexParser->parsePictureIndices();
   indexParser->indexList[1]->parsePictureIndex();

   
   viereckMatrixTransform = indexParser->indexList[1]->pictureList[0]->getPanelNode();



   // add root node to cover scenegraph
   cover->getObjectsRoot()->addChild(viereckMatrixTransform);

   return true;
}

// this is called if the plugin is removed at runtime
StreetView::~StreetView()
{
    fprintf(stderr, "StreetView::~StreetView\n");
	cover->getObjectsRoot()->removeChild(viereckMatrixTransform);
    delete indexParser;
}

void
StreetView::preFrame()
{
}


COVERPLUGIN(StreetView)
