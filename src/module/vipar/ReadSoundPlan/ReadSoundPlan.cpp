/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                                           (C)1998 RUS  **
**                                                                        **
** Description: Read module Elmer data format         	                  **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** History:                                                               **
** Jan  03	    U. Woessner	    V1.0                                  **
**                                                                        **
**                                                                        **
*\**************************************************************************/

#include <do/coDoPoints.h>
#include <do/coDoData.h>
#include "ReadSoundPlan.h"

int main(int argc, char *argv[])
{

    ReadSoundPlan *application = new ReadSoundPlan(argc, argv);

    application->start(argc, argv);
    return 0;
}

ReadSoundPlan::ReadSoundPlan(int argc, char *argv[])
    : coModule(argc, argv, "Read Soundplan Data") // description in the module setup window
{
    // file browser parameter FIXME
    //filenameParam = addFileBrowserParam("/usr/local/covise/covise/src/application/vipar/READ_SOUNDPLAN/data/","Data file path");
    //filenameParam->setValue("/usr/local/covise/covise/src/application/vipar/READ_SOUNDPLAN/data/RREC0050.txt","*.txt*");

    filenameParam = addFileBrowserParam("Filename", "Data file path");
    filenameParam->setValue("/home/pow/covise/data", "*.txt*");

    // the output ports
    meshOutPort = addOutputPort("mesh", "Points", "point data");
    daySoundOutPort = addOutputPort("daysound", "Float", "daysound data");
    nightSoundOutPort = addOutputPort("nightsound", "Float", "nightound data");
}

ReadSoundPlan::~ReadSoundPlan()
{
}

int ReadSoundPlan::compute(const char * /*port*/)
{

    FILE *fp;
    const char *fileName;
    int num_elem, num_field;
    int i;
    char buf[300], tmp[400];

    float *x_coord, *y_coord, *z_coord; // coordinate lists
    float *dBDay, *dBNight; // data lists

    // names of the COVISE output objects
    const char *meshName;
    const char *daySoundName;
    const char *nightSoundName;

    //temp data
    num_elem = 0;
    num_field = 0;
    char *pch;

    // read the file browser parameter
    fileName = filenameParam->getValue();

    // open the file
    if ((fp = fopen(fileName, "r")) == NULL)
    {
        sendError("ERROR: Can't open file >> %s", fileName);
        return STOP_PIPELINE;
    }

    // the COVISE output objects (located in shared memory)
    coDoPoints *meshObj;
    coDoFloat *daySoundObj;
    coDoFloat *nightSoundObj;

    // get the ouput object names from the controller
    // the output object names have to be assigned by the controller
    meshName = meshOutPort->getObjName();
    daySoundName = daySoundOutPort->getObjName();
    nightSoundName = nightSoundOutPort->getObjName();

    //check for number of objects
    while (fgets(tmp, sizeof(tmp), fp) != NULL)
    {
        if (num_elem == 0)
        {
            pch = strtok(tmp, ";");
            while (pch != NULL)
            {
                printf("%s\n", pch);
                pch = strtok(NULL, " ;");
                num_field++;
            }
        }
        num_elem++;
    }
    rewind(fp);
    printf("%i\n", num_field);

    // create the unstructured grid object for the mesh
    if (meshName != NULL)
    {
        meshObj = new coDoPoints(meshName, num_elem);
        // the last parameters needs to be 1
        meshOutPort->setCurrentObject(meshObj);

        // create the scalar data object for daySound
        if (daySoundName != NULL)
        {
            daySoundObj = new coDoFloat(daySoundName, num_elem);
            daySoundOutPort->setCurrentObject(daySoundObj);

            if (daySoundObj->objectOk())
            {
                daySoundObj->getAddress(&dBDay);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'daySoundObj' failed");
                return STOP_PIPELINE;
            }
        }
        else
        {
            Covise::sendError("ERROR: object name not correct for 'daysound'");
            return STOP_PIPELINE;
        }

        // create the scalar data object for nightSound
        if (nightSoundName != NULL)
        {
            nightSoundObj = new coDoFloat(nightSoundName, num_elem);
            nightSoundOutPort->setCurrentObject(nightSoundObj);

            if (nightSoundObj->objectOk())
            {
                nightSoundObj->getAddress(&dBNight);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'nightSoundObj' failed");
                return STOP_PIPELINE;
            }
        }
        else
        {
            Covise::sendError("ERROR: object name not correct for 'nightsound'");
            return STOP_PIPELINE;
        }

        if (meshObj->objectOk())
        {
            // get pointers to the element, vertex and coordinate lists
            meshObj->getAddresses(&x_coord, &y_coord, &z_coord);

            // read the coordinate lines
            for (i = 0; i < num_elem; i++)
            {
                // read the line which contains the coordinates and scan it
                if (fgets(buf, 300, fp) != NULL)
                {
                    //sscanf(buf,"%s%f%f%f%f%f\n",tmp, x_coord, y_coord, z_coord,dBDay, dBNight);
                    sscanf(buf, "%f%f%f%f%f\n", x_coord, y_coord, z_coord, dBDay, dBNight);
                    x_coord++;
                    y_coord++;
                    z_coord++;
                    dBDay++;
                    dBNight++;
                }
                else
                {
                    sendError("ERROR: unexpected end of file");
                    return STOP_PIPELINE;
                }
            }
        }
        else
        {
            Covise::sendError("ERROR: object name not correct for 'mesh'");
            return STOP_PIPELINE;
        }

        // close the file
    }
    fclose(fp);
    return CONTINUE_PIPELINE;
}
