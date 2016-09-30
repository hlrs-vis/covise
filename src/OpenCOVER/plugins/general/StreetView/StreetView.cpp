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
#include "Station.h"

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

	indexParser = new IndexParser(this);
	indexParser->parseIndex("\\\\VISFS1/raid/share/projects/reallabor/Herrenberg/Daten/vonHerrenberg/Panorama/P09299_Herrenberg/EBF");
	indexParser->removeDuplicateEntries(); // one index per directory

	indexParser->parsePicturesPerStreet("Äckerlenweg"); // later: get street name from openDRIVE
	indexParser->sortStreetPicturesPerStation();
	/*/
	osg::Matrix viewerMatrix;
	viewerMatrix.set(cover->getViewerMat());
	osg::Vec3d viewerTrans;
	viewerTrans.set(viewerMatrix.getTrans());
	viewerPosX = viewerTrans.x();
	viewerPosY = viewerTrans.y();
	viewerPosZ = viewerTrans.z();
	cout << "X: " + std::to_string(viewerPosX) + ", " + "Y: " + std::to_string(viewerPosY) + ", " "Z: " + std::to_string(viewerPosZ) << endl;
	/*/
	//stationNode = indexParser->getNearestStationNode(3493912.860, 5382153.880, 425.0);
	//stationNode = indexParser->getNearestStationNode(3494030.840, 5382145.078, 433.0); //13
	//stationNode = indexParser->getNearestStationNode(3493809.972, 5382130.307, 420.0); //152
	stationNode = indexParser->getNearestStationNode(3493990.853, 5382151.126, 429.0); //57

	if (stationNode)
	{
		cover->getObjectsRoot()->addChild(stationNode); // add root node to cover scenegraph
	}
	else
	{
		fprintf(stderr, "getting stationNode failed");
	}
	return true;
}

// this is called if the plugin is removed at runtime
StreetView::~StreetView()
{
	fprintf(stderr, "StreetView::~StreetView\n");
	cover->getObjectsRoot()->removeChild(stationNode);
	delete indexParser;
}

void StreetView::preFrame()
{
}

COVERPLUGIN(StreetView)
