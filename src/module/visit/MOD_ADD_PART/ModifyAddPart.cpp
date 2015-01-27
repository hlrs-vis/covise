/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// checkin Tue Nov 13 10:26:40 MEZ 2001   Fehler in euler winkeln behoben
// checkin Mon Nov  5 19:49:08 MET 2001
// eingecheckt Mon Nov  5 15:15:39 MET 2001
#include <iostream.h>
#include <fstream.h>
#include <strstream.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/dirent.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <netdb.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <math.h>
#include "ModifyAddPart.h"
#include "element.h"
#include "domain.h"

#include <api/coFeedback.h>

#include <Performer/pr/pfLinMath.h>

char buf[4000];
char *action[3] = { "Create_Surface", "Create_Projected_Surface", "Create_Grid" };
int currAction = 0, nAction = 3;
char *Nothing = { "---" };

int pj_covise_tetin(DO_BinData *obj);
int pj_covise_tetin(coTetin *obj);

int ICEM_openDomain(char *);
int ICEM_closeDomain(int, char *);
int ICEM_projectSurfaceGrid(int, char *, float *direct);
void ICEM_transformDomain(int, float pos[3], float rotMat[3][3]);
coDoPolygons *ICEM_getDomainSurface(int, char *);

/*
 structure of vent directory

 data/visit/icem/vent/                        // ventFilePath
                     - vent_rund/             // ventDirs[0]
                                 - domains/   // domaun file name
                                 - mesh/      // boco file name
                                 - transfer/
                                       - vent1.*
                                       - vent2.*

- vent_eckig/            // ventDirs[1]
- domains/
- mesh/
- transfer/
- vent3.*

- oval/                  // ventDirs[2]
- domains/
- mesh/
- transfer/
- vent4.*
...
...

*/

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ModifyAddPart::ModifyAddPart()
{
    int i;

    // no. of attached vents to cabin
    numVents = 0;

    // no. of read directories for vent description
    // default nothing
    numVentDirs = 1;

    for (i = 0; i < MAX_NAMES; i++)
        ventdirs[i] = NULL;

    // init some defaults
    for (i = 0; i < MAX_VENTS; i++)
    {
        fileid[i] = 0;
        exist[i] = 0;
    }

    // declare the name of our module
    set_module_description("Add parts to the Cabin");

    // paramter for vent directory structure
    p_ventDir = addStringParam("ventPath", "Path for vent descriptions");
    p_ventDir->setValue("data/visit/icem/test");
    ventFilePath = "data/visit/icem/test";

    // get the directory structure for all vents in given path
    getVentDirs();

    // create the COVISE parameter
    // user can set a subdirectory for vents
    createVentParam();

    // what do you want to calculate
    p_action = addChoiceParam("Set Action", "select the action");
    p_action->setValue(nAction, action, currAction);

    // tetin object describing the cabin
    inTetin = addInputPort("tetinObj", "DO_Tetin", "coTetin object");
    inTetin->setRequired(1);

    // output objects for transfer directory and vent polygons
    solverText = addOutputPort("solverText", "coDoText", "Command for StarCD");
    outPolygon = addOutputPort("polygon_set", "coDoPolygons", "Geometry output");
    prostarData = addOutputPort("prostarData", "coDoText", "Part data for Prostar");
}

/////////////// Create Vent parameters ////////////////////////////////////

void ModifyAddPart::createVentParam()
{
    int i, j, k;

    p_vent = paraSwitch("Vent", "Select a vent");
    for (i = 0; i < MAX_VENTS; i++)
    {
        // create description and name
        sprintf(buf, "Vent %d", i);

        // case for the vent switching
        paraCase(buf);

        sprintf(buf, "Vent_%d:Name", i);
        currVentFile[i] = 0;
        p_name[i] = addChoiceParam(buf, "Select a vent type substructure");
        p_name[i]->setValue(numVentDirs, ventdirs, currVentFile[i]);

        sprintf(buf, "Vent_%d:Pos", i);
        p_pos[i] = addFloatVectorParam(buf, "Position");
        p_pos[i]->setImmediate(1);
        p_pos[i]->setValue(0.0, 0.0, 0.0);
        pos[i][0] = pos[i][1] = pos[i][2] = 0.0;

        sprintf(buf, "Vent_%d:Euler", i);
        p_euler[i] = addFloatVectorParam(buf, "Euler Angles");
        p_euler[i]->setImmediate(1);
        p_euler[i]->setValue(0.0, 0.0, 0.0);
        euler[i][0] = euler[i][1] = euler[i][2] = 0.0;

        sprintf(buf, "Vent_%d:Rot", i);
        p_rot[i] = addFloatVectorParam(buf, "Rotation Matrix");
        p_rot[i]->setImmediate(1);

        for (j = 0; j < 3; j++)
        {
            for (k = 0; k < 3; k++)
            {
                if (j == k)
                    coverRot[i][3 * k + j] = 1.0;
                else
                    coverRot[i][3 * k + j] = 0;
            }
        }

        p_rot[i]->setValue(9, coverRot[i]);
        axis[i][0] = 1;
        axis[i][1] = 0;
        axis[i][2] = 0;
        /// vent case ends here
        paraEndCase();
    }
    paraEndSwitch();
}

/////////////// Read the vent substructures ////////////////////////////////////

void ModifyAddPart::getVentDirs()
{
    int i;
    DIR *dirp;
    struct dirent *dp;

    // look if given directory exist
    Covise::getname(buf, ventFilePath);
    if (strlen(buf) == 0)
    {
        sprintf(buf, "Directory %s doesn't exist", ventFilePath);
        sendError(buf);
        return;
    }
    else
    {
        dirp = opendir(buf);
        if (dirp == NULL)
        {
            sprintf(buf, "Directory %s doesn't exist", ventFilePath);
            sendError(buf);
            return;
        }
    }
    // set duumy directory noting as first one
    i = 0;
    ventdirs[0] = new char[strlen(Nothing) + 1];
    strcpy(ventdirs[0], Nothing);
    i++;

    // read all subdirectories
    while ((dp = readdir(dirp)) != NULL)
    {
        if (!strcmp(dp->d_name, ".") == NULL && !strcmp(dp->d_name, "..") == NULL)
        {
            if (ventdirs[i])
                delete ventdirs[i];
            ventdirs[i] = new char[strlen(dp->d_name) + 1];
            strcpy(ventdirs[i], dp->d_name);
            //cerr << "Vent substructure is  " << ventdirs[i] <<endl;
            i++;
            if (i == MAX_NAMES)
                break;
        }
    }
    numVentDirs = i;
}

/////////////// Get No. of set vents ////////////////////////////////////

void ModifyAddPart::getNumOfVents()
{
    int i;

    numVents = 0;
    for (i = 0; i < MAX_VENTS; i++)
    {
        if (exist[i] == 1)
            numVents++;
    }
    sprintf(buf, ":::::::::::::Detected %d attached vents\n", numVents);
    sendInfo(buf);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Compute callback: Called when the module is executed
// ++++
// ++++  NEVER use input/output ports or distributed objects anywhere
// ++++        else than inside this function
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ModifyAddPart::compute()
{
    int i, j, k, ierr;
    const char *name, *name2;
    coDoSet *setObject;
    DO_BinData *binobj;
    coTetin *tetin = 0;
    char *tmp;
    float icemRot[3][3]; // rotation matrix which is input for icemTransformDomain

    // read input data objects describing the cabin
    coDistributedObject *objTetin = inTetin->getCurrentObject();

    // check read object
    if (!objTetin)
    {
        sendError("Can't open Tetin object");
        return STOP_PIPELINE;
    }

    if (!objTetin->isType("COVBIN"))
    {
        sendError("Object isn't of type COVBIN");
        return STOP_PIPELINE;
    }

    binobj = (DO_BinData *)objTetin;

    // send coTetin object to the projection library
    ierr = pj_covise_tetin(binobj);
    if (ierr != 0)
    {
        sendError("Error in pj_covise_tetin");
        return STOP_PIPELINE;
    }

    // get no. of selected (attached) vents
    getNumOfVents();

    // get name for polygon output object
    name = outPolygon->getObjName();

    // get name for DO_TEXT output object
    name2 = solverText->getObjName();

    // get name for ProStar DO_TEXT output object
    const char *prostarName = prostarData->getObjName();

    // Prostar object creation
    ostrstream *prostarTextStream = NULL;
    coDoText *prostarText = NULL;

    if (currAction == 2)
    {
        prostarText = new coDoText(prostarName, 512 * numVents + 64);
        char *textAddr;
        prostarText->getAddress(&textAddr);
        *textAddr = '\0';
        prostarTextStream = new ostrstream(textAddr, 512 * numVents);
        (*prostarTextStream) << "NUMADDPART " << numVents << endl;
    }

    // loop over all selected (attached) vents
    // make sets
    //==============================================================
    for (i = 0; i < MAX_VENTS; i++)
    {
        if (exist[i])
        {
            // get bocofile (family_boco)
            tmp = new char[strlen(ventFilePath) + strlen(ventdirs[currVentFile[i]]) + 100];
            strcpy(tmp, ventFilePath);
            strcat(tmp, "/");
            strcat(tmp, ventdirs[currVentFile[i]]);
            strcat(tmp, "/mesh/family_boco");
            Covise::getname(buf, tmp);
            if (!buf)
            {
                sprintf(buf, "ERROR: boco file %s doesn't exist", tmp);
                sendError(buf);
                return STOP_PIPELINE;
            }
            sendInfo(":::::::::::::boco file is   : %s\n", buf);
            coTetin__bocoFile *bf = new coTetin__bocoFile(buf);
            delete[] tmp;

            // get configuration directory
            tmp = new char[strlen(ventFilePath) + 100];
            strcpy(tmp, ventFilePath);
            strcat(tmp, "/");
            strcat(tmp, ventdirs[currVentFile[i]]);
            Covise::getname(buf, tmp);
            if (!buf)
            {
                sprintf(buf, "ERROR: configdir %s doesn't exist", tmp);
                sendError(buf);
                return STOP_PIPELINE;
            }
            sendInfo(":::::::::::::configDir is   : %s\n", buf);
            coTetin__configDir *cfd = new coTetin__configDir(buf);
            delete[] tmp;

            // create solver and case name
            tmp = new char[strlen("star") + strlen("vent") + 10];
            sprintf(tmp, "star#vent%d", i);
            sendInfo(":::::::::::::Solver#Case name is is   : %s\n", tmp);
            coTetin__OutputInterf *outintinf = new coTetin__OutputInterf(tmp);
            delete[] tmp;

            // create tetin object for projection library
            tetin = new coTetin();
            if (tetin)
            {
                tetin->append(bf);
                tetin->append(cfd);
                tetin->append(outintinf);
                ierr = pj_covise_tetin(tetin);
                delete tetin;
                if (ierr != 0)
                {
                    sendError("Error in pj_covise_tetin");
                    return STOP_PIPELINE;
                }
            }

            // open ICEM domain
            // domainfilename : filepathname of domain file
            // is a domainfile given
            tmp = new char[strlen(ventFilePath) + strlen(ventdirs[currVentFile[i]]) + 100];
            strcpy(tmp, ventFilePath);
            strcat(tmp, "/");
            strcat(tmp, ventdirs[currVentFile[i]]);
            strcat(tmp, "/domains/hexa.unstruct");
            Covise::getname(buf, tmp);
            if (!buf)
            {
                sprintf(buf, "ERROR: domainfile %s doesn't exist", tmp);
                sendError(buf);
                return STOP_PIPELINE;
            }
            sendInfo(":::::::::::::domain file is : %s\n", buf);
            delete[] tmp;
            fileid[i] = ICEM_openDomain(buf);
            if (fileid[i] < 0)
            {
                sendError("Error in opening domain file");
                return STOP_PIPELINE;
            }

            // make rot matrix
            for (j = 0; j < 3; j++)
            {
                for (k = 0; k < 3; k++)
                {
                    icemRot[j][k] = coverRot[i][3 * k + j];
                }
            }

            // do something
            switch (currAction)
            {
            case 0:
            {
                // create surface
                // get boundary surface of current domain as coDoPolygons
                // objectname : name of coDoPolygons object
                ICEM_transformDomain(fileid[i], &pos[i][0], icemRot);
                polynames[i] = new char[strlen(name) + 100];
                sprintf(polynames[i], "%s_%d", name, i);
                sendInfo(":::::::::::::output name is : %s\n", polynames[i]);
                polyobj[i] = ICEM_getDomainSurface(fileid[i], (char *)polynames[i]);

                coFeedback feedback("ModifyAddPart");
                char str[30];
                sprintf(str, "VENT_%d", i);
                feedback.addString(str);
                feedback.addPara(p_pos[i]);
                feedback.addPara(p_rot[i]);
                feedback.addPara(p_euler[i]);
                feedback.apply(polyobj[i]);
            }
            break;

            case 1:
            {
                // project grid
                // transform work on origonal
                // translation + rotate
                // project selected surface elements of current domain onto
                // geometry (nearest point)
                // family_oberflaeche : family name of surface elements to project

                ICEM_transformDomain(fileid[i], &pos[i][0], icemRot);
                /////ierr = ICEM_projectSurfaceGrid(fileid[i], "VENTL_DUMY", axis[i]);
                ierr = ICEM_projectSurfaceGrid(fileid[i], "VENTL_DUMY", NULL);
                if (ierr != 0)
                {
                    sendError("Error in ICEM_projectSurfaceGrid");
                    return STOP_PIPELINE;
                }
                polynames[i] = new char[strlen(name) + 100];
                sprintf(polynames[i], "%s_%d", name, i);
                sendInfo(":::::::::::::output name is : %s\n", polynames[i]);
                polyobj[i] = ICEM_getDomainSurface(fileid[i], (char *)polynames[i]);

                coFeedback feedback("ModifyAddPart");
                char str[30];
                sprintf(str, "VENT_%d", i);
                feedback.addString(str);
                feedback.addPara(p_pos[i]);
                feedback.addPara(p_rot[i]);
                feedback.addPara(p_euler[i]);
                feedback.apply(polyobj[i]);
            }

            break;

            case 2:
            {

                ICEM_transformDomain(fileid[i], &pos[i][0], icemRot);
                /////ierr = ICEM_projectSurfaceGrid(fileid[i], "VENTL_DUMY", axis[i]);
                ierr = ICEM_projectSurfaceGrid(fileid[i], "VENTL_DUMY", NULL);
                if (ierr != 0)
                {
                    sendError("Error in ICEM_projectSurfaceGrid");
                    return STOP_PIPELINE;
                }
                polynames[i] = new char[strlen(name) + 100];
                sprintf(polynames[i], "%s_%d", name, i);
                sendInfo(":::::::::::::output name is : %s\n", polynames[i]);
                polyobj[i] = ICEM_getDomainSurface(fileid[i], (char *)polynames[i]);

                coFeedback feedback("ModifyAddPart");
                char str[30];
                sprintf(str, "VENT_%d", i);
                feedback.addString(str);
                feedback.addPara(p_pos[i]);
                feedback.addPara(p_rot[i]);
                feedback.addPara(p_euler[i]);
                feedback.apply(polyobj[i]);

                // create grid
                // close (temporary) ICEM domain and call output interface
                // output_name : type of output interface # case name
                // type of output interface can be star, fluent or fenfloss
                // objectname : name of coDoText object
                fprintf(stderr, "SOLVER TEX=[%s]\n", name2);
                ICEM_closeDomain(fileid[i], (char *)name2);

                // Add an ADDPART line to StarCD Command Text Input
                (*prostarTextStream) << "ADDPART ";

                // fully resolve the path from COVISE abbreviations
                char fullpath[MAXPATHLEN];
                Covise::getname(fullpath, ventFilePath);

                (*prostarTextStream) << fullpath << "/"
                                     << ventdirs[currVentFile[i]]
                                     << "/transfer"
                                     << " "
                                     << "vent" << i << " ";
                int j;
                for (j = 0; j < 3; j++)
                    (*prostarTextStream) << euler[i][j] << " ";
                for (j = 0; j < 3; j++)
                    (*prostarTextStream) << pos[i][j] << " ";
                (*prostarTextStream) << endl;
            }
            break;
            }
        } // end of exist check
    } // end of vent loop

    // create the set object
    //if(currAction != 2)
    //{
    setObject = new coDoSet(name, numVents, (coDistributedObject **)polyobj);

    coFeedback feedback("ModifyAddPart");
    feedback.addString("SET");

    int l = 0; // str len
    //int i;
    char *str;
    for (i = 1; i < numVentDirs; i++) // ommit choice o:"---"
    {
        l += strlen(ventdirs[i]) + 1; // " vent_box vent_cyl"
    }
    l += 100; // "VENTDIRS 2 vent_box vent_cyl"
    str = new char[l + 1];
    sprintf(str, "VENTDIRS %d", numVentDirs - 1);
    for (i = 1; i < numVentDirs; i++)
    {
        strcat(str, " ");
        strcat(str, ventdirs[i]);
    }
    fprintf(stderr, "str=[%s]\n", str);
    feedback.addString(str);
    for (i = 0; i < MAX_VENTS; i++)
        feedback.addPara(p_name[i]);

    feedback.apply(setObject);

    outPolygon->setCurrentObject(setObject);

    for (i = 0; i < numVents; i++)
        delete polynames[i];
    //}

    if (currAction == 2)
    {
        //  close stream for Prostar commands
        if (prostarTextStream)
        {
            prostarTextStream->write("\0", 1);
            delete prostarTextStream;
            prostarData->setCurrentObject(prostarText);
        }
    }

    return CONTINUE_PIPELINE;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Parameter callback: This one is called whenever an immediate
// ++++                      mode parameter is changed, but NOT for
// ++++                      non-immediate ports
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void ModifyAddPart::param(const char *name)
{
    const char *text;

    // cerr << "Paramname is " << name << endl;

    // parameter order:
    // ventPath
    // Vent
    // Vent_0:Name
    // Vent_0:Pos
    // Vent_0:Dir
    // Vent_1:Name
    // Vent_1:Pos
    // Vent_1:Dir
    // ...
    // Set Action

    // check user action
    if (strcmp(name, p_action->getName()) == 0)
    {
        currAction = p_action->getValue();
        return;
    }

    // check vent directory name=ventPath
    if (strcmp(name, p_ventDir->getName()) == 0)
    {
        text = p_ventDir->getValue();
        if (strcmp(text, ventFilePath) == NULL) // nothing changed
            return;

        if (ventFilePath)
            delete[] ventFilePath;
        ventFilePath = new char[strlen(text) + 1];
        strcpy(ventFilePath, text);
        numVentDirs = 1;
        getVentDirs();
        return;
    }

    int k;
    for (int i = 0; i < MAX_VENTS; i++)
    {
        // check vent_? domainfile
        if (strcmp(name, p_name[i]->getName()) == 0)
        {
            currVentFile[i] = p_name[i]->getValue();
            //cerr << "currStructure" << ventdirs[currVentFile[i]] << endl;
            if (strcmp(ventdirs[currVentFile[i]], Nothing) == NULL)
                exist[i] = 0;
            else
                exist[i] = 1;

            //cerr << "Vent "<< i << " set to  " << ventdirs[currVentFile[i]] << endl;
            //cerr << "Vent "<< i << " state   " << exist[i] << endl;
            return;
        }

        // check vent_? position
        if (strcmp(name, p_pos[i]->getName()) == 0)
        {
            for (k = 0; k < 3; k++)
            {
                pos[i][k] = p_pos[i]->getValue(k);
            }
            return;
        }

        // the matrix can be set only be feedback from COVER
        // because it is deactivated in the Control Panel
        // if COVER sends the matrix
        // compute the euler angles
        if (strcmp(name, p_rot[i]->getName()) == 0)
        {
            for (k = 0; k < 9; k++)
            {
                coverRot[i][k] = p_rot[i]->getValue(k);
            }

            computeEulerAngles(i);
            return;
        }

        // if the euler angles are set through the Control Panel
        // compute the matrix
        if (strcmp(name, p_euler[i]->getName()) == 0)
        {
            for (k = 0; k < 3; k++)
            {
                euler[i][k] = p_euler[i]->getValue(k);
            }

            computeCoverRot(i);

            return;
        }
    }
}

void
ModifyAddPart::computeEulerAngles(int i)
{
    pfMatrix m;
    pfCoord coord;

    m[0][0] = coverRot[i][0];
    m[0][1] = coverRot[i][1];
    m[0][2] = coverRot[i][2];
    m[1][0] = coverRot[i][3];
    m[1][1] = coverRot[i][4];
    m[1][2] = coverRot[i][5];
    m[2][0] = coverRot[i][6];
    m[2][1] = coverRot[i][7];
    m[2][2] = coverRot[i][8];

    m.getOrthoCoord(&coord);

    euler[i][0] = coord.hpr[0];
    euler[i][1] = coord.hpr[1];
    euler[i][2] = coord.hpr[2];

    p_euler[i]->setValue(euler[i][0], euler[i][1], euler[i][2]);

    pfVec3 a(1, 0, 0);
    a.xformVec(a, m);
    axis[i][0] = a[0];
    axis[i][1] = a[1];
    axis[i][2] = a[2];
}

/*
void
ModifyAddPart::computeEulerAngles_daniela(int i)
{
   // resort dir
   float m[16];
   m[0]=coverRot[i][0];
   m[1]=coverRot[i][3];
   m[2]=coverRot[i][6];
   m[3]=0;
   m[4]=coverRot[i][1];
m[5]=coverRot[i][4];
m[6]=coverRot[i][7];
m[7]=0;
m[8]=coverRot[i][2];
m[9]=coverRot[i][5];
m[10]=coverRot[i][8];
m[11]=0;
m[12]=0;
m[13]=0;
m[14]=0;
m[15]=1;

//http://www.cs.ualberta.ca/~andreas/math/matrfaq_latest.html#Q37
float C, angle_x, angle_y, angle_z, tsrx, tsry;

float radians=180/M_PI;

angle_y = asin( m[2]);        // Calculate Y-axis angle
C  =  cos( angle_y );
angle_y    *=  radians;

if ( fabs( C ) > 0.005 )             // Gimball lock?
{
tsrx      =  m[10] / C;           //No, so get X-axis angle
tsry      = -m[6]  / C;

angle_x  = atan2( tsry, tsrx ) * radians;

tsrx      =  m[0] / C;            // Get Z-axis angle
tsry      = -m[1] / C;

angle_z  = atan2( tsry, tsrx ) * radians;
}
else                                 // Gimball lock has occurred
{
angle_x  = 0;                      // Set X-axis angle to zero

tsrx      =  m[5];                 // And calculate Z-axis angle
tsry      =  m[4];

angle_z  = atan2( tsry, tsrx ) * radians;
}

// return only positive angles in [0,360]
////if (angle_x < 0) angle_x += 360;
////if (angle_y < 0) angle_y += 360;
////if (angle_z < 0) angle_z += 360;

euler[i][0]=angle_z;
euler[i][1]=angle_x;
euler[i][2]=angle_y;

p_euler[i]->setValue( euler[i][0], euler[i][1], euler[i][2]);
//fprintf(stderr,"euler angles: [%f %f %f]\n", angle_x, angle_y, angle_z);
}
*/

/*
void
ModifyAddPart::computeCoverRot_daniela(int i)
{
   // update the rotation matrix (in the vector) for cover
   //http://www.cs.ualberta.ca/~andreas/math/matrfaq_latest.html#Q36
   float m[16];
   float A, B, C, D, E, F, AD, BD, angle_x, angle_y, angle_z;
   angle_x=euler[i][1]*M_PI/180.0;
   angle_y=euler[i][2]*M_PI/180.0;
   angle_z=euler[i][0]*M_PI/180.0;

A = cos(angle_x);
B = sin(angle_x);
C = cos(angle_y);
D = sin(angle_y);
E = cos(angle_z);
F = sin(angle_z);

AD =   A * D;
BD =   B * D;

m[0]  =   C * E;
m[1]  =  -C * F;
m[2]  =   D;
m[4]  =  BD * E + A * F;
m[5]  = -BD * F + A * E;
m[6]  =  -B * C;
m[8]  = -AD * E + B * F;
m[9]  =  AD * F + B * E;
m[10] =   A * C;

m[3]  =  m[7] = m[11] = m[12] = m[13] = m[14] = 0;
m[15] =  1;

coverRot[i][0]= m[0];
coverRot[i][1]= m[4];
coverRot[i][2]= m[8];

coverRot[i][3]= m[1];
coverRot[i][4]= m[5];
coverRot[i][5]= m[9];

coverRot[i][6]= m[2];
coverRot[i][7]= m[6];
coverRot[i][8]= m[10];

p_rot[i]->setValue(9, coverRot[i]);

}

*/

void
ModifyAddPart::computeCoverRot(int i)
{
    pfMatrix m;
    float h, p, r;
    h = euler[i][0];
    p = euler[i][1];
    r = euler[i][2];
    m.makeEuler(h, p, r);

    coverRot[i][0] = m[0][0];
    coverRot[i][1] = m[0][1];
    coverRot[i][2] = m[0][2];

    coverRot[i][3] = m[1][0];
    coverRot[i][4] = m[1][1];
    coverRot[i][5] = m[1][2];

    coverRot[i][6] = m[2][0];
    coverRot[i][7] = m[2][1];
    coverRot[i][8] = m[2][2];

    p_rot[i]->setValue(9, coverRot[i]);

    pfVec3 a(1, 0, 0);
    a.xformVec(a, m);
    axis[i][0] = a[0];
    axis[i][1] = a[1];
    axis[i][2] = a[2];
}

void ModifyAddPart::quit()
{
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  postInst() is called once after we contacted Covise, but before
// ++++             getting into the main loop
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void ModifyAddPart::postInst()
{
    p_ventDir->show();
    p_vent->show();
    p_action->show();

    for (int i = 0; i < MAX_VENTS; i++)
    {
        p_rot[i]->disable();
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  What's left to do for the Main program:
// ++++                                    create the module and start it
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int main(int argc, char *argv[])

{
    // create the module
    ModifyAddPart *application = new ModifyAddPart;

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
