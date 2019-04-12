/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2002 RUS  **
 **                                                                        **
 ** Description: Read PTV data from DLR.                      **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                          Uwe Woessner                                  **
 **     High Performance Computing Center University of Stuttgart          **
 **                         Nobelstrasse 19                                **
 **                         70550 Stuttgart                                **
 **                                                                        **
 ** Cration Date: 01.02.2016                                               **
\**************************************************************************/

#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <do/coDoPoints.h>
#include <api/coModule.h>
#include <util/coFileUtil.h>
#include <alg/coChemicalElement.h>
#include <limits.h>
#include <float.h>
#include <cassert>
#include "ReadAtomProbe.h"

/// Constructor
coReadAtomProbe::coReadAtomProbe(int argc, char *argv[])
    : coModule(argc, argv, "Read atom Probbe point clouds.")
{

    // Create ports:
    poPoints = addOutputPort("Location", "Points", "particle positions");
    poTypes = addOutputPort("Number", "Int", "atom type");
    poValue = addOutputPort("Value", "Float", "Detected Value");

    // Create parameters:
	binFilename = addFileBrowserParam("binFilename", "Atom Probe bin file");
	binFilename->setValue("data/", "*.bin;*.pos;");

	rrngFilename = addFileBrowserParam("rrngFilename", "Atom Probe range file");
	rrngFilename->setValue("data/", "*.rrng;*.RRNG");


}

/// Compute routine: load checkpoint file
int coReadAtomProbe::compute(const char *)
{




	const char *path = binFilename->getValue();
	const char *rangePath = binFilename->getValue();
    const char *fileName = coDirectory::fileOf(path);
    const char *dirName = coDirectory::dirOf(path);


	int file = -1;
	file = open(path, O_RDONLY | O_BINARY);
	if (file <= 0)
	{
		sendInfo("Unable to open bin file: %s", path);
		return STOP_PIPELINE;
	}
	// read one timestep
	std::vector<int> pNumber;
	std::vector<float> px, py, pz, val;
	int res = 0;
	do
	{
		float buf[4];
		res = read(file, buf, 4 * sizeof(float));
		if (res == 4 * sizeof(float))
		{
			byteSwap(buf, 4);

			px.push_back(buf[0]);
			py.push_back(buf[1]);
			pz.push_back(buf[2]);
			val.push_back(buf[3]);
			int num = 0; // look up falue in the range file
			pNumber.push_back(num);
		}
	} while (res > 0);
	// done reading the file
	close(file);

	coDoInt *doTypes = new coDoInt(poTypes->getObjName(), int(pNumber.size()), &pNumber[0]);

	coDoPoints *doPoints = new coDoPoints(poPoints->getObjName(), int(px.size()), &px[0], &py[0], &pz[0]);

	coDoFloat *doVelos = new coDoFloat(poValue->getObjName(), int(val.size()), &val[0]);


    // Assign sets to output ports:
    poPoints->setCurrentObject(doPoints);
	poValue->setCurrentObject(doVelos);
    poTypes->setCurrentObject(doTypes);
    
    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, coReadAtomProbe)
