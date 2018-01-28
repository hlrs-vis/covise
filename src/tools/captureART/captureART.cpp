/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * inputhdw.cpp
 *
 *  Created on: Dec 9, 2014
 *      Author: svnvlad
 */
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "captureART.h"
DTrackSDK *dt; ///ART DTrack SDK class

#include <iostream>
#include <algorithm> // for min/max

using namespace std;
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		cerr << "usage: captureART port filename\n";
		return -1;
	}
	int dtrack_port = 5000;
	char filename[1000];
	sscanf(argv[1], "%d", &dtrack_port);
	sprintf(filename, "%s%d.csv",argv[2], dtrack_port);

	FILE *fp = fopen(filename, "w");

	dt = new DTrackSDK(dtrack_port);

	if (!dt->isLocalDataPortValid())
		cout << "Cannot initialize DTrack!" << endl;

	if (!dt->receive())
		cout << "Error receiving data!" << endl;


	dt->startMeasurement();

	if (!dt)
		return false;
	while (true)
	{
		if (!dt->receive())
		{
			// error messages from example

			if (dt->getLastDataError() == DTrackSDK::ERR_TIMEOUT)
			{
				cout << "--- timeout while waiting for tracking data" << endl;
				//return -1;
			}

			if (dt->getLastDataError() == DTrackSDK::ERR_NET)
			{
				cout << "--- error while receiving tracking data" << endl;
				//return -1;
			}

			if (dt->getLastDataError() == DTrackSDK::ERR_PARSE)
			{
				cout << "--- error while parsing tracking data" << endl;
				//return -1;
			}

		}
		for (int i = 0; i < dt->getNumBody(); i++)
		{
			DTrack_Body_Type_d *b = dt->getBody(i);
			fprintf(fp, "body %d: %f %f %f %f %f %f %f %f %f %f %f %f\n",i, b->loc[0], b->loc[1], b->loc[2], b->rot[0], b->rot[1], b->rot[2], b->rot[3], b->rot[4], b->rot[5], b->rot[6], b->rot[7], b->rot[8]);
		}
	}

	fclose(fp);
}
