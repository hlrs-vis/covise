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
#include "IndexParser.h"
#include "Picture.h"
#include "Camera.h"

#include <iostream>

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
	indexParser->parseIndex("\\\\VISFS1/raid/share/projects/reallabor/Herrenberg/Daten/vonHerrenberg/Panorama/P09299_Herrenberg/EBF");
	std::cout << "Before removing duplicate entries: " << indexParser->indexList.size() << endl;
	indexParser->removeDuplicateEntries();
	std::cout << "After removing duplicate entries: " << indexParser->indexList.size() << endl;
	indexParser->parsePictureIndices();
	//stationNode = indexParser->indexList[0]->pictureList[0]->getPanelNode();
	
	indexParser->sortIndicesPerStation();

	// generate list with cameras
	//indexParser->parseCameras();
	//std::cout << "Camera name: " << indexParser->indexList[0]->cameraList[0]->getCameraName() << endl;

	cover->getObjectsRoot()->addChild(stationNode); // add root node to cover scenegraph

	return true;
}

// this is called if the plugin is removed at runtime
StreetView::~StreetView()
{
	fprintf(stderr, "StreetView::~StreetView\n");
	cover->getObjectsRoot()->removeChild(stationNode);
	delete indexParser;
}

void
	StreetView::preFrame()
{
}


COVERPLUGIN(StreetView)
