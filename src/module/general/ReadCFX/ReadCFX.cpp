/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2002 OSR  **
 **                                                                        **
 **     Description: Read module CFX-11 data format                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 *\**************************************************************************/
#include <config/CoviseConfig.h>
#include <api/coStepFile.h>
#include <do/coDoData.h>
#include <cfxExport.h>
#include <getargs.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/stat.h>
#ifdef _WIN32
#define MAXPATHLEN 1024
#else
#include <unistd.h>
#endif

#include "ReadCFX.h"

#undef strlen
#undef strcpy

#ifdef MINGW
extern "C" long _ftol2_sse(double x) { return static_cast<long>(x); }
extern "C" long long _allmul(long long a, long long b) { return a * b; }
extern "C" long long _alldiv(long long a, long long b) { return a / b; }
#endif

// check whether this is a valid CFX file
//   return   >0 : errno
//            =0 : may be correct
//            -1 : too small
//            -2 : wrong start magic
int checkFile(const char *filename)
{
    const int MIN_FILE_SIZE = 1024; // minimal size for .res files

    const int MACIC_LEN = 5; // "magic" at the start
    const char *magic = "*INFO";
    char magicBuf[MACIC_LEN];

    struct stat statRec;
    off_t fileSize;
#ifdef WIN32
    HANDLE hFile = CreateFile(filename, GENERIC_READ,
                              FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE)
    {
        fprintf(stderr, "could not open file %s\n", filename);
        return -1; // error condition, could call GetLastError to find out more
    }

    LARGE_INTEGER size;
    if (!GetFileSizeEx(hFile, &size))
    {
        CloseHandle(hFile);
        fprintf(stderr, "could not get filesize for %s\n", filename);
        return -1; // error condition, could call GetLastError to find out more
    }
    fileSize = size.QuadPart;

    CloseHandle(hFile);
#else
    if (stat(filename, &statRec) < 0)
        return errno;
    fileSize = statRec.st_size;
#endif

    FILE *fi = fopen(filename, "r");
    if (!fi)
        return errno;

    if (fileSize < MIN_FILE_SIZE)
        return -1;

    size_t iret = fread(magicBuf, 1, MACIC_LEN, fi);
    if (iret != MACIC_LEN)
        cerr << "checkFile :: error reading MACIC_LEN " << endl;

    if (strncasecmp(magicBuf, magic, MACIC_LEN) != 0)
        return -2;

    fclose(fi);

    return 0;
}

CFX::CFX(int argc, char **argv)
    : coSimpleModule(argc, argv, "Read CFX")
    , currenttimestep(0)
    , _dir("")
    , _theFile("")
{
    pScalChoiceVal = NULL;
    pVectChoiceVal = NULL;
    psVarIndex = NULL;
    pvVarIndex = NULL;
    //char *ChoiceInitVal[] = {"none"};
    //const char *initRegion[] = { "No Region" };
    const char *initDomain[] = { "No Domain" };
    const char *initScalar[] = { "No Scalar data" };
    const char *initVector[] = { "No Vector data" };
    const char *initParticle[] = { "No particletype" };

    // file browser parameter
    p_result = addFileBrowserParam("result", "Result file path");
    //gridfileParam->setValue("data","data *.dat*");
    char defaultfile[MAXPATHLEN];
    sprintf(defaultfile, "%s/nofile.res", getenv("COVISEDIR"));
    p_result->setValue(defaultfile, "*.res");

    p_zone = addChoiceParam("domains", "domains");
    p_zone->setValue(1, initDomain, 0);

    p_region = addStringParam("RegionsSelection", "selection of regions");
    p_region->setValue("1-9999999");

    p_boundary = addStringParam("BoundarySelection", "selection of boundaries, can be numbers or WALL, INLET, OUTLET, OPENING, CFX_INTERFACE, SYMMETRY");
    p_boundary->setValue("WALL");

    p_scalar = addChoiceParam("scalar_data", "Scalar data");
    p_scalar->setValue(1, initScalar, 0);

    p_vector = addChoiceParam("vector_data", "Vector data");
    p_vector->setValue(1, initVector, 0);

    // the output ports
    p_outPort1 = addOutputPort("GridOut0", "UnstructuredGrid", "unstructured grid");
    p_outPort2 = addOutputPort("DataOut0", "Float", "scalar data");
    p_outPort3 = addOutputPort("DataOut1", "Vec3", "vector data");
    p_outPort4 = addOutputPort("GridOut1", "Polygons", "region grid");
    p_outPort5 = addOutputPort("DataOut2", "Float", "region scalar data");
    p_outPort6 = addOutputPort("GridOut2", "Polygons", "boundary grid");
    p_outPort7 = addOutputPort("DataOut3", "Float", "boundary scalar data");
    p_outPort8 = addOutputPort("DataOut4", "Vec3", "boundary vector data");
    p_outPort9 = addOutputPort("GridOut3", "Points", "particle points");
    p_outPort10 = addOutputPort("DataOut5", "Float", "particle scalar data");
    p_outPort11 = addOutputPort("DataOut6", "Vec3", "particle vector data");

    p_readGrid = addBooleanParam("readGrid", "read Grid and volume data");
    p_readGrid->setValue(true);
    p_readRegions = addBooleanParam("readRegions", "read surfaces and surface data");
    p_readRegions->setValue(false);
    p_readBoundaries = addBooleanParam("readBoundaries", "read boundary surfaces");
    p_readBoundaries->setValue(false);

    p_scalarFactor = addFloatParam("scalar_factor", "factor to change scalar variable (3D and boundary)");
    p_scalarFactor->setValue(1.0f);

    p_scalarDelta = addFloatParam("scalar_delta", "value to add to change scalar variable level (3D and boundary)");
    p_scalarDelta->setValue(0.0f);

    p_boundScalar = addChoiceParam("boundary_scalar_data", "Boundary Scalar data");
    p_boundScalar->setValue(1, initScalar, 0);

    p_boundVector = addChoiceParam("boundary_vector_data", "Boundary Vector data");
    p_boundVector->setValue(1, initVector, 0);

    p_particleType = addChoiceParam("particle_type", "Particle Type to read");
    p_particleType->setValue(1, initParticle, 0);

    p_particleScalar = addChoiceParam("particle_scalar_data", "Particle Scalar data");
    p_particleScalar->setValue(1, initScalar, 0);

    p_particleVector = addChoiceParam("particle_vector_data", "Particle Vector data");
    p_particleVector->setValue(1, initVector, 0);

    p_maxTimeSteps = addInt32Param("maxTimeSteps", "maxTimeSteps");
    p_maxTimeSteps->setValue(500);

    p_skipParticles = addInt32Param("skipParticles", "skipParticles");
    p_skipParticles->setValue(0);

    p_skipTimeSteps = addInt32Param("skipTimeSteps", "skipTimeSteps");
    p_skipTimeSteps->setValue(0);

    p_timesteps = addInt32Param("timesteps", "timesteps");
    p_timesteps->setValue(0);

    p_firststep = addInt32Param("first_timestep", "index of the first timestep to read (0..[n-1])");
    p_firststep->setValue(0);

    p_readGridForAllTimeSteps = addBooleanParam("grid_is_time_dependent", "use this if you have a time-dependent grid");
    p_timeDependentZone = addInt32Param("zone_with_time_dependent_grid", "zone in which we need to read all timesteps. Zone (0..[n-1]) is used for rotation as well, -1 reads all timesteps in all zones");
    /*   p_initialRotation = addFloatParam("initial_rotation","workaround for time dependent grids. Give angle of first timestep to read relative to timestep 0");
      p_initialRotation->setValue(0.);*/
    p_rotAxis = addFloatVectorParam("rotAxis", "rotation axis for grid rotation");
    p_rotAxis->setValue(0.0, 0.0, 1.0);
    p_rotVertex = addFloatVectorParam("point_on_rotAxis", "one point on the rotation axis");
    p_rotVertex->setValue(0.0, 0.0, 0.0);
    p_rotAngle = addFloatParam("rot_Angle_per_timestep", "rotation angle between read timesteps");
    p_rotAngle->setValue(12.);

    p_relabs = addBooleanParam("transform_velocity", "transform velocity (relative->absolute or reverse)");
    p_relabsDirection = addChoiceParam("transform_direction", "Rel2Abs or Abs2Rel");
    s_direction[0] = strdup("none");
    s_direction[1] = strdup("abs2rel");
    s_direction[2] = strdup("rel2abs");
    p_relabsDirection->setValue(3, s_direction, 2);

    p_relabsAxis = addChoiceParam("rotation_axis", "rotation axis for relabs transformation");
    s_rotaxis[0] = strdup("x");
    s_rotaxis[1] = strdup("y");
    s_rotaxis[2] = strdup("z");
    p_relabsAxis->setValue(3, s_rotaxis, 2);

    p_relabs_zone = addInt32Param("zone_to_transform_velocity", "Zone (1..n]) in which we want to tranform the velocity");

    p_rotSpeed = addFloatParam("angular_velocity", "omega, set to 0 for automatic settings [1/min]");
    p_rotSpeed->setValue(0.);

    p_rotVectors = addBooleanParam("rotate_velocity", "rotate velocity vectors according to timestep and angle / timestep");
    p_rotVectors->setValue(false);

    // nothing read so far
    nzones = 0;

    // not yet in compute call
    computeRunning = false;

    // user level defaults to 0
    usrLevel_ = 0;
    usrLevel_ = coCoviseConfig::getInt("Module.ReadCFX.UserLevel", 0);
}

CFX::~CFX()
{
}

string
ExtractDirectory(const char *path)
{
#ifndef WIN32
    const char *DIR_SEP = "/";
#else
    const char *DIR_SEP = "\\";
#endif
    size_t pos = string(path).rfind(DIR_SEP);
    if (pos >= strlen(path))
    {
        return string("./");
    }
    pos += strlen(DIR_SEP);
    return string(path, pos);
}

string
ExtractFile(const char *path)
{
#ifndef WIN32
    const char *DIR_SEP = "/";
#else
    const char *DIR_SEP = "\\";
#endif
    size_t pos = string(path).rfind(DIR_SEP);
    if (pos >= strlen(path))
    {
        return path;
    }
    pos += strlen(DIR_SEP);
    return string(path + pos);
}

void CFX::param(const char *paramName, bool in_map_loading)
{
    inMapLoading = in_map_loading;
    if (0 == strcmp(p_result->getName(), paramName)
        && (!computeRunning))
    {
        // @@@@@@@@@@ safety block
        /*
            if( nzones>0 && !getenv("CFX_NO_PROTECT") )
            {
               sendError("Sorry, but you may not modify this browser parameter");
               return;
            }
      */
        resultfileName = p_result->getValue();

        int checkValue = checkFile(resultfileName);
        if (checkValue != 0)
        {
            if (checkValue > 0)
                sendError("'%s':%s", resultfileName, strerror(checkValue));
            else
                switch (checkValue)
                {

                case -1:
                    sendError("'%s': too small to be a real result file",
                              resultfileName);
                    break;
                case -2:
                    sendError("'%s':does not start with '*INFO'",
                              resultfileName);
                }
            return;
        }

        static char *resultName = NULL;
        if (resultName == NULL || strcmp(resultName, resultfileName) != 0)
        {
            resultName = new char[strlen(resultfileName) + 1];
            strcpy(resultName, resultfileName);

            sendInfo("Please wait...");

            if (nzones > 0)
                cfxExportDone();

            nzones = cfxExportInit(resultName, NULL);

            _dir = ExtractDirectory(resultfileName);
            _theFile = ExtractFile(resultfileName);

            if (nzones < 0)
            {
                cfxExportDone();
                sendError("cfxExportInit could not open %s", resultfileName);
                return;
            }

            /*if (cfxExportZoneSet (0, counts) < 0)
           {
            }*/
            int timeStepNum = cfxExportTimestepNumGet(1);
            if (timeStepNum < 0)
            {
                printf("  no timesteps\n");
            }

            int iteration = cfxExportTimestepNumGet(1);
            if (cfxExportTimestepSet(iteration) < 0)
            {
                printf("  oops\n");
            }

            sendInfo("Found %d zones", nzones);

            get_choice_fields(1);

            // @@@ cfxExportDone();
            sendInfo("The initialisation was successfully done");

            if (!in_map_loading)
            {
                /*
                        p_zone->updateValue(nzones+1,ZoneChoiceVal,0);
                        p_scalar->updateValue(nscalars+1,ScalChoiceVal,1);
                        p_vector->updateValue(nvectors+1,VectChoiceVal,1);
            */
                p_zone->updateValue(nzones + 1, ZoneChoiceVal, p_zone->getValue());

                p_scalar->updateValue(nscalars + 1, ScalChoiceVal, p_scalar->getValue());
                p_vector->updateValue(nvectors + 1, VectChoiceVal, p_vector->getValue());

                p_boundScalar->updateValue(nscalars + 1, ScalChoiceVal, p_boundScalar->getValue());
                p_boundVector->updateValue(nvectors + 1, VectChoiceVal, p_boundVector->getValue());

                p_particleScalar->updateValue(npscalars + 1, pScalChoiceVal, p_particleScalar->getValue());
                p_particleVector->updateValue(npvectors + 1, pVectChoiceVal, p_particleVector->getValue());
            }
        }
        else
        {
            resultName = new char[strlen(resultfileName) + 1];
            strcpy(resultName, resultfileName);
        }
    }
    else if (strcmp(p_zone->getName(), paramName) == 0 && !in_map_loading)
    {
        if (nzones > 0)
        {
            p_zone->updateValue(nzones + 1, ZoneChoiceVal, p_zone->getValue());
            get_choice_fields(p_zone->getValue());
            p_scalar->updateValue(nscalars + 1, ScalChoiceVal, p_scalar->getValue());
            p_vector->updateValue(nvectors + 1, VectChoiceVal, p_vector->getValue());
            p_boundScalar->updateValue(nscalars + 1, ScalChoiceVal, p_boundScalar->getValue());
            p_boundVector->updateValue(nvectors + 1, VectChoiceVal, p_boundVector->getValue());
            p_particleScalar->updateValue(npscalars + 1, pScalChoiceVal, p_particleScalar->getValue());
            p_particleVector->updateValue(npvectors + 1, pVectChoiceVal, p_particleVector->getValue());
        }
    }
    else if (strcmp(p_particleType->getName(), paramName) == 0 && !in_map_loading)
    {
        setParticleVarChoice(p_particleType->getValue());
    }
    else if (strcmp(p_timesteps->getName(), paramName) == 0)
    {
        if (nzones > 0)
        {
            p_zone->updateValue(nzones + 1, ZoneChoiceVal, p_zone->getValue());
            //get_choice_fields(p_zone->getValue());
            p_scalar->updateValue(nscalars + 1, ScalChoiceVal, p_scalar->getValue());
            p_vector->updateValue(nvectors + 1, VectChoiceVal, p_vector->getValue());
            p_boundScalar->updateValue(nscalars + 1, ScalChoiceVal, p_boundScalar->getValue());
            p_boundVector->updateValue(nvectors + 1, VectChoiceVal, p_boundVector->getValue());
            p_particleScalar->updateValue(npscalars + 1, pScalChoiceVal, p_particleScalar->getValue());
            p_particleVector->updateValue(npvectors + 1, pVectChoiceVal, p_particleVector->getValue());
        }
    }
}

int CFX::compute(const char *)
{

    int i;
    char buf[8][300];
    int startzone, endzone;
    int counts[cfxCNT_SIZE];

    currenttimestep = -1;

    scalarFactor = p_scalarFactor->getValue();
    scalarDelta = p_scalarDelta->getValue();

    if (nzones == 0)
    {
        computeRunning = true;
        param(p_result->getName(), false);
        computeRunning = false;
        if (nzones == 0)
        {
            sendError("Input file is not correct");
            return STOP_PIPELINE;
        }
    }

    // read the file browser parameter
    resultfileName = p_result->getValue();

    int zone = p_zone->getValue();

    int timesteps = p_timesteps->getValue();

    // the COVISE output objects (located in shared memory)

    coDistributedObject **outputgrid;
    coDistributedObject **outputscalar;
    coDistributedObject **outputvector;
    coDistributedObject **outputregion;
    coDistributedObject **outputregscal;
    coDistributedObject **outputboundary;
    coDistributedObject **outputboundscal;
    coDistributedObject **outputboundvec;

    // get the ouput object names from the controller
    // the output object names have to be assigned by the controller

    //////////////////////////////////////////////////////

    char *resultname = new char[strlen(resultfileName) + 1];
    strcpy(resultname, resultfileName);

    //cerr << " @@@ cfxExportInit(\"" << resultname << "\",counts);" << endl;
    // @@@ nzones = cfxExportInit(resultname,counts);

    if (nzones < 0)
    {
        sendError("cfxExportInit could not read '%s'", resultname);
        return STOP_PIPELINE;
    }

    float *omega = new float[nzones + 1]; // cfx counts zones from 1 to n, so we do as well
    memset(omega, 0, (nzones + 1) * sizeof(float));

    if (cfxExportZoneSet(zone, counts) < 0)
        cfxExportFatal((char *)"invalid zone number");

    //cerr<<"maximum timesteps = "<<cfxExportTimestepCount()<<endl;

    //cerr<<"1 timesteps = "<<cfxExportTimestepNumGet(1)<<endl;
    if (timesteps < 1 || timesteps > cfxExportTimestepCount())
    {
        timesteps = cfxExportTimestepCount();
        p_timesteps->setValue(timesteps);
    }

    if (timesteps == 0)
        timesteps = 1;

    cerr << "timesteps to read = " << timesteps << endl;

    outputgrid = new coDistributedObject *[nzones + 1];
    outputgrid[nzones] = NULL;

    outputscalar = new coDistributedObject *[nzones + 1];
    outputscalar[nzones] = NULL;

    outputvector = new coDistributedObject *[nzones + 1];
    outputvector[nzones] = NULL;

    outputregion = new coDistributedObject *[nzones + 1];
    outputregion[nzones] = NULL;

    outputregscal = new coDistributedObject *[nzones + 1];
    outputregscal[nzones] = NULL;

    outputboundary = new coDistributedObject *[nzones + 1];
    outputboundary[nzones] = NULL;

    outputboundscal = new coDistributedObject *[nzones + 1];
    outputboundscal[nzones] = NULL;

    outputboundvec = new coDistributedObject *[nzones + 1];
    outputboundvec[nzones] = NULL;

    for (i = 0; i <= nzones; i++)
    {
        outputgrid[i] = NULL;
        outputvector[i] = NULL;
        outputscalar[i] = NULL;
        outputregion[i] = NULL;
        outputregscal[i] = NULL;
        outputboundary[i] = NULL;
        outputboundscal[i] = NULL;
        outputboundvec[i] = NULL;
    }

    //cerr<<"timsetep set 1..";
    if (timesteps > 1)
    {
        int timeStepNum = cfxExportTimestepNumGet(1);
        if (timeStepNum < 0)
        {
            return STOP_PIPELINE;
        }
    }
    //cerr<<" was successful!"<<endl;

    // DEBUG
    // print available timesteps
    int nt;
    int n = cfxExportZoneCount();
    cerr << "number of domains:" << n << endl;
    for (i = 0; i < n; i++)
    {
        cerr << "zone Nr. " << i + 1 << ", Zone Name: " << cfxExportZoneName(i + 1) << endl;
        ;
    }
    nt = cfxExportTimestepCount();
    cerr << "number of timesteps=" << nt << endl;
    /*   if(nt)
      {
        for(i = 0; i < nt; i++)
        {
          cerr << "TimestepNumGet("<< i+1 << ")= " << cfxExportTimestepNumGet(i+1) << endl;
        }
      }
   */

    selRegion.clear();
    if (p_region->getValue() != NULL)
    {
        selRegion.add(p_region->getValue());
    }

    selBoundary.clear();
    if (p_boundary->getValue() != NULL)
    {
        selBoundary.add(p_boundary->getValue());
    }

    if (zone == 0) // read all zones!
    {
        startzone = 0;
        endzone = nzones - 1;
    }
    else // read just one zone!
    {
        startzone = zone - 1;
        endzone = zone - 1;
    }

    for (i = startzone; i <= endzone; i++)
    {
        int res = cfxExportZoneSet(i + 1, NULL);
        //cerr << "cfxExportZoneSet returns " << res << endl;
        if ((res < 1) || (res > cfxExportZoneCount()))
        {
            cerr << "Error in compute. cfxExportZoneSet returns bad value (" << res << ")!" << endl;
            return STOP_PIPELINE;
        }

        // check whether the current zone is rotating, axis and rotation speed

        double rotationAxis[2][3];
        double angularVelocity;

        if (p_rotSpeed->getValue() == 0.)
        {
            cfxExportZoneIsRotating(rotationAxis, &angularVelocity);
            omega[i - startzone + 1] = angularVelocity;
        }
        else
        {
            // cfx lib crashes in cfxExportZoneIsRotating on ia64 and several other platforms
            omega[i - startzone + 1] = p_rotSpeed->getValue() * M_PI / 30.;
        }

        fprintf(stderr, "\nzone %d: omega = %8.3lf [1/s] = %8.3lf [1/min]\n", i + 1, omega[i - startzone + 1], omega[i - startzone + 1] * 30. / M_PI);

        sprintf(buf[0], "%s_%d", p_outPort1->getObjName(), i);
        sprintf(buf[1], "%s_%d", p_outPort2->getObjName(), i);
        sprintf(buf[2], "%s_%d", p_outPort3->getObjName(), i);
        sprintf(buf[3], "%s_%d", p_outPort4->getObjName(), i);
        sprintf(buf[4], "%s_%d", p_outPort5->getObjName(), i);
        sprintf(buf[5], "%s_%d", p_outPort6->getObjName(), i);
        sprintf(buf[6], "%s_%d", p_outPort7->getObjName(), i);
        sprintf(buf[7], "%s_%d", p_outPort8->getObjName(), i);

        outputgrid[i] = NULL;
        outputvector[i] = NULL;
        outputscalar[i] = NULL;
        outputregion[i] = NULL;
        outputregscal[i] = NULL;
        outputboundary[i] = NULL;
        outputboundscal[i] = NULL;
        outputboundvec[i] = NULL;

        // check if variables are only defined at boundary
        bool scal_boundaryonly = false;
        bool vect_boundaryonly = false;
        if (p_scalar->getValue())
        {
            char *scalname = ScalChoiceVal[p_scalar->getValue()];
            int len = strlen(scalname);
            if (len > 14 && !strcmp(scalname + (len - 14) * sizeof(char), "_boundary_only"))
            {
                scal_boundaryonly = true;
                sendError("Scalar Variable %s is only defined at boundaries\n", ScalChoiceVal[p_scalar->getValue()]);
            }
        }
        if (p_vector->getValue())
        {
            char *vectname = VectChoiceVal[p_vector->getValue()];
            //cerr << "vectname=" << vectname << "," << vectname+(strlen(vectname)-14)*sizeof(char) << endl;
            if (!strcmp(vectname + (strlen(vectname) - 14) * sizeof(char), "_boundary_only"))
            {
                vect_boundaryonly = true;
                sendError("Variable %s is only defined at boundaries\n", VectChoiceVal[p_vector->getValue()]);
            }
        }

        float stepduration = 0.;

        if (((fabs(omega[i - startzone + 1])) > 0.) && (nt > 1) && ((p_timesteps->getValue()) > 1) && ((p_firststep->getValue()) < nt) && ((p_timeDependentZone->getValue() == (i - startzone + 1)) || (p_timeDependentZone->getValue() == -1)))
        {
            int t1 = cfxExportTimestepNumGet(2);
            int t0 = cfxExportTimestepNumGet(1);
            if (cfxExportTimestepTimeGet(t1) != -1) // cfxExportTimestepTimeGet doesn' seem to work ... maybe it does in CFX-12?
            {
                stepduration = cfxExportTimestepTimeGet(t1) - cfxExportTimestepTimeGet(t0);
                fprintf(stderr, "stepduration=%10.6lf=%10.6lf-%10.6lf. t0=%d, t1=%d\n", stepduration, cfxExportTimestepTimeGet(t1), cfxExportTimestepTimeGet(t0), t0, t1);
            }
            else
            {
                stepduration = fabs((p_rotAngle->getValue() / 360.) * (2 * M_PI / omega[i - startzone + 1]));
                fprintf(stderr, "stepduration=%10.6lf\n", stepduration);
            }
        }

        if (p_readGrid->getValue())
        {
            if (readZone(buf[0], NULL, NULL, timesteps, GRID, i + 1, omega, stepduration, &outputgrid[i], NULL, NULL))
                return STOP_PIPELINE;
        }
        if (p_readRegions->getValue())
        {
            readZone(buf[3], buf[4], NULL, timesteps, REGION, i + 1, omega, 0., &outputregion[i], &outputregscal[i], NULL);
        }
        if (p_readBoundaries->getValue())
        {
            readZone(buf[5], buf[6], buf[7], timesteps, BOUNDARY, i + 1, omega, 0., &outputboundary[i], &outputboundscal[i], &outputboundvec[i]);
        }

        // read data (also without grid)
        if ((p_scalar->getValue() != 0) && (!scal_boundaryonly))
        {
            readZone(buf[1], NULL, NULL, timesteps, SCALAR, i + 1, omega, stepduration, &outputscalar[i], NULL, NULL);
        }
        if ((p_vector->getValue() != 0) && (!vect_boundaryonly))
        {
            readZone(buf[2], NULL, NULL, timesteps, VECTOR, i + 1, omega, stepduration, &outputvector[i], NULL, NULL);
        }
    }

    outputgrid[nzones] = NULL;
    outputvector[nzones] = NULL;
    outputscalar[nzones] = NULL;
    outputregion[nzones] = NULL;
    outputregscal[nzones] = NULL;
    outputboundary[nzones] = NULL;
    outputboundscal[nzones] = NULL;
    outputboundvec[nzones] = NULL;

    coDoSet *multiblockgridset = NULL;
    coDoSet *multiblockvectorset = NULL;
    coDoSet *multiblockscalarset = NULL;
    coDoSet *multiblockregionset = NULL;
    coDoSet *multiblockregscalset = NULL;
    coDoSet *multiblockboundaryset = NULL;
    coDoSet *multiblockboundscalset = NULL;
    coDoSet *multiblockboundvecset = NULL;

    if (timesteps > 1)
    {
        int numTimesteps = timesteps;
        char ts[100];

        sprintf(ts, "1 %d", timesteps);

        if (((zone == 0) && outputgrid[0]) || ((zone != 0) && outputgrid[zone - 1]))
        {
            multiblockgridset = normalizeSet(&numTimesteps, outputgrid, coObjInfo(p_outPort1->getObjName()));
            multiblockgridset->addAttribute("TIMESTEP", ts);
        }
        if (((zone == 0) && outputvector[0]) || ((zone != 0) && outputvector[zone - 1]))
        {
            multiblockvectorset = normalizeSet(&numTimesteps, outputvector, coObjInfo(p_outPort3->getObjName()));
            multiblockvectorset->addAttribute("TIMESTEP", ts);
        }
        if (((zone == 0) && outputscalar[0]) || ((zone != 0) && outputscalar[zone - 1]))
        {
            multiblockscalarset = normalizeSet(&numTimesteps, outputscalar, coObjInfo(p_outPort2->getObjName()));
            multiblockscalarset->addAttribute("TIMESTEP", ts);
        }
        if (((zone == 0) && outputregion[0]) || ((zone != 0) && outputregion[zone - 1]))
        {
            multiblockregionset = normalizeSet(&numTimesteps, outputregion, coObjInfo(p_outPort4->getObjName()));
            multiblockregionset->addAttribute("TIMESTEP", ts);
        }
        if (((zone == 0) && outputregscal[0]) || ((zone != 0) && outputregscal[zone - 1]))
        {
            multiblockregscalset = normalizeSet(&numTimesteps, outputregscal, coObjInfo(p_outPort5->getObjName()));
            multiblockregscalset->addAttribute("TIMESTEP", ts);
        }
        if (((zone == 0) && outputboundary[0]) || ((zone != 0) && outputboundary[zone - 1]))
        {
            multiblockboundaryset = normalizeSet(&numTimesteps, outputboundary, coObjInfo(p_outPort6->getObjName()));
            multiblockboundaryset->addAttribute("TIMESTEP", ts);
        }
        if (((zone == 0) && outputboundscal[0]) || ((zone != 0) && outputboundscal[zone - 1]))
        {
            multiblockboundscalset = normalizeSet(&numTimesteps, outputboundscal, coObjInfo(p_outPort7->getObjName()));
            multiblockboundscalset->addAttribute("TIMESTEP", ts);
        }
        if (((zone == 0) && outputboundvec[0]) || ((zone != 0) && outputboundvec[zone - 1]))
        {
            multiblockboundvecset = normalizeSet(&numTimesteps, outputboundvec, coObjInfo(p_outPort8->getObjName()));
            multiblockboundvecset->addAttribute("TIMESTEP", ts);
        }
    }
    else
    {
        if (zone == 0) // all zones -> create a set
        {
            if (outputgrid[0])
                multiblockgridset = new coDoSet(coObjInfo(p_outPort1->getObjName()), outputgrid);
            if (outputscalar[0])
                multiblockscalarset = new coDoSet(coObjInfo(p_outPort2->getObjName()), outputscalar);
            if (outputvector[0])
                multiblockvectorset = new coDoSet(coObjInfo(p_outPort3->getObjName()), outputvector);
            if (outputregion[0])
                multiblockregionset = new coDoSet(coObjInfo(p_outPort4->getObjName()), outputregion);
            if (outputregscal[0])
                multiblockregscalset = new coDoSet(coObjInfo(p_outPort5->getObjName()), outputregscal);
            if (outputboundary[0])
                multiblockboundaryset = new coDoSet(coObjInfo(p_outPort6->getObjName()), outputboundary);
            if (outputboundscal[0])
                multiblockboundscalset = new coDoSet(coObjInfo(p_outPort7->getObjName()), outputboundscal);
            if (outputboundvec[0])
                multiblockboundvecset = new coDoSet(coObjInfo(p_outPort8->getObjName()), outputboundvec);
        }
        else // only 1 zone selected -> no set, clone
        {
        }
    }

    if (zone == 0)
    {
        if (multiblockgridset)
            p_outPort1->setCurrentObject(multiblockgridset);
        if (multiblockscalarset)
            p_outPort2->setCurrentObject(multiblockscalarset);
        if (multiblockvectorset)
            p_outPort3->setCurrentObject(multiblockvectorset);
        if (multiblockregionset)
            p_outPort4->setCurrentObject(multiblockregionset);
        if (multiblockregscalset)
            p_outPort5->setCurrentObject(multiblockregscalset);
        if (multiblockboundaryset)
            p_outPort6->setCurrentObject(multiblockboundaryset);
        if (multiblockboundscalset)
            p_outPort7->setCurrentObject(multiblockboundscalset);
        if (multiblockboundvecset)
            p_outPort8->setCurrentObject(multiblockboundvecset);
    }
    else
    {
        if (outputgrid[zone - 1])
        {
            p_outPort1->setCurrentObject(outputgrid[zone - 1]->clone(coObjInfo(p_outPort1->getObjName())));
        }
        if (outputscalar[zone - 1])
        {
            p_outPort2->setCurrentObject(outputscalar[zone - 1]->clone(coObjInfo(p_outPort2->getObjName())));
        }
        if (outputvector[zone - 1])
        {
            p_outPort3->setCurrentObject(outputvector[zone - 1]->clone(coObjInfo(p_outPort3->getObjName())));
        }
        if (outputregion[zone - 1])
        {
            p_outPort4->setCurrentObject(outputregion[zone - 1]->clone(coObjInfo(p_outPort4->getObjName())));
        }
        if (outputregscal[zone - 1])
        {
            p_outPort5->setCurrentObject(outputregscal[zone - 1]->clone(coObjInfo(p_outPort5->getObjName())));
        }
        if (outputboundary[zone - 1])
        {
            p_outPort6->setCurrentObject(outputboundary[zone - 1]->clone(coObjInfo(p_outPort6->getObjName())));
        }
        if (outputboundscal[zone - 1])
        {
            p_outPort7->setCurrentObject(outputboundscal[zone - 1]->clone(coObjInfo(p_outPort7->getObjName())));
        }
        if (outputboundvec[zone - 1])
        {
            p_outPort8->setCurrentObject(outputboundvec[zone - 1]->clone(coObjInfo(p_outPort8->getObjName())));
        }
    }
    // Read Particles

    int nparticleTypes = cfxExportGetNumberOfParticleTypes();
    int typeToRead = p_particleType->getValue();
    if (nparticleTypes >= typeToRead && typeToRead > 0)
    {
        numTracks = cfxExportGetNumberOfTracks(typeToRead);
        int maxTimesteps = p_maxTimeSteps->getValue();
        int incParticles = p_skipParticles->getValue() + 1;
        int incTimeSteps = p_skipTimeSteps->getValue() + 1;
        int numTrackPoints;
        //float maxTime=0;
        int maxPoints = 0;
        std::vector<int> numParticlesPerStep;
        for (int i = 1; i <= numTracks; i += incParticles)
        {
            //cfxExportGetParticleTrackCoordinatesByTrack(typeToRead, i, &numTrackPoints);
            numTrackPoints = cfxExportGetNumberOfPointsOnParticleTrack(typeToRead, i);
            if (numTrackPoints > maxPoints)
            {
                for (int i = maxPoints; i < numTrackPoints; i++)
                {
                    numParticlesPerStep.push_back(0);
                }
                maxPoints = numTrackPoints;
            }
            int n = 0;
            for (int j = 0; j < numTrackPoints; j += incTimeSteps)
            {
                numParticlesPerStep[n]++;
                n++;
            }

            /*float *time = cfxExportGetParticleTrackTimeByTrack(typeToRead, i, &numTrackPoints);
          if(numTrackPoints > maxPoints)
             maxPoints = numTrackPoints;
          if(numTrackPoints > 0)
          {
             if(time[numTrackPoints-1] > maxTime)
                maxTime = time[numTrackPoints-1];
          }*/
        }
        int numTimesteps = maxPoints / incTimeSteps;
        if (numTimesteps > maxTimesteps)
            numTimesteps = maxTimesteps;

        coDoPoints **particleObjects = new coDoPoints *[numTimesteps + 1];
        coDoFloat **particleSObjects = NULL;
        coDoVec3 **particleVObjects = NULL;
        particleObjects[numTimesteps] = NULL;
        float **xc = new float *[numTimesteps];
        float **yc = new float *[numTimesteps];
        float **zc = new float *[numTimesteps];
        float **pS = NULL;
        float **pVx = NULL;
        float **pVy = NULL;
        float **pVz = NULL;
        if (p_particleScalar->getValue() != 0)
        {
            pS = new float *[numTimesteps];
            particleSObjects = new coDoFloat *[numTimesteps + 1];
            particleSObjects[numTimesteps] = NULL;
        }
        if (p_particleScalar->getValue() != 0)
        {
            pVx = new float *[numTimesteps];
            pVy = new float *[numTimesteps];
            pVz = new float *[numTimesteps];
            particleVObjects = new coDoVec3 *[numTimesteps + 1];
            particleVObjects[numTimesteps] = NULL;
        }

        char objectName[1000];
        for (int i = 0; i < numTimesteps; i++)
        {
            sprintf(objectName, "%s_%d", p_outPort9->getObjName(), i);
            particleObjects[i] = new coDoPoints(objectName, numParticlesPerStep[i]);
            particleObjects[i]->getAddresses(xc + i, yc + i, zc + i);
            numParticlesPerStep[i] = 0;
            if (particleSObjects)
            {
                sprintf(objectName, "%s_%d", p_outPort10->getObjName(), i);
                particleSObjects[i] = new coDoFloat(objectName, numParticlesPerStep[i]);
            }
            if (particleVObjects)
            {
                sprintf(objectName, "%s_%d", p_outPort11->getObjName(), i);
                particleVObjects[i] = new coDoVec3(objectName, numParticlesPerStep[i]);
            }
        }

        sprintf(objectName, "1 %d", numTimesteps);
        coDoSet *particleSet = new coDoSet(p_outPort9->getObjName(), (coDistributedObject **)particleObjects);
        particleSet->addAttribute("TIMESTEP", objectName);
        p_outPort9->setCurrentObject(particleSet);

        for (int i = 1; i <= numTracks; i += incParticles)
        {
            float *coord = cfxExportGetParticleTrackCoordinatesByTrack(typeToRead, i, &numTrackPoints);
            float *scalar = NULL;
            float *vector = NULL;
            if (pS)
            {
                scalar = cfxExportGetParticleTypeVar(typeToRead, p_particleScalar->getValue(), i);
            }
            if (pVx)
            {
                vector = cfxExportGetParticleTypeVar(typeToRead, p_particleVector->getValue(), i);
            }
            int ts = 0;
            for (int j = 0; j < numTrackPoints; j += incTimeSteps)
            {
                int coordNum = numParticlesPerStep[ts];
                xc[ts][coordNum] = coord[(j * 3) + 0];
                yc[ts][coordNum] = coord[(j * 3) + 1];
                zc[ts][coordNum] = coord[(j * 3) + 2];
                if (scalar)
                {
                    pS[ts][coordNum] = scalar[j];
                }
                if (vector)
                {
                    pVx[ts][coordNum] = vector[(j * 3) + 0];
                    pVy[ts][coordNum] = vector[(j * 3) + 1];
                    pVz[ts][coordNum] = vector[(j * 3) + 2];
                }
                numParticlesPerStep[ts]++;
                ts++;
                if (ts >= numTimesteps)
                    break;
            }
        }

        delete[] xc;
        delete[] yc;
        delete[] zc;
        delete[] particleObjects;
        if (particleSObjects)
            delete[] particleSObjects;
        if (particleVObjects)
            delete[] particleVObjects;
        if (pS)
        {
            delete[] pS;
        }
        if (pVx)
        {
            delete[] pVx;
            delete[] pVy;
            delete[] pVz;
        }
    }

    delete[] outputgrid;
    delete[] outputvector;
    delete[] outputscalar;
    delete[] outputregion;
    delete[] outputregscal;
    delete[] outputboundary;
    delete[] outputboundscal;
    delete[] outputboundvec;

    delete[] omega;

    return CONTINUE_PIPELINE;
}

// sort timesteps into top hierarchy level
coDoSet *
CFX::normalizeSet(int *numTimesteps, coDistributedObject **list, const coObjInfo &objInfo)
{

    std::vector<std::vector<const coDistributedObject *> > blocks;

    int numBlocks = 0;
    int i = 0;
    // we have either one single block / zone or all blocks / zones!
    // so we start with the first zone that is not NULL
    while (list[i] == NULL)
        i++;

    while (const coDistributedObject *elem = list[i])
    {
        numBlocks++;
        const coDoSet *subSet = dynamic_cast<const coDoSet *>(elem);
        if (!subSet)
            return NULL;
        std::vector<const coDistributedObject *> empty;
        blocks.push_back(empty);
        std::vector<const coDistributedObject *> &steps = blocks[blocks.size() - 1];
        if (subSet->getNumElements() != *numTimesteps)
        {
            if (*numTimesteps >= 0)
                return NULL;
            *numTimesteps = subSet->getNumElements();
        }
        for (int j = 0; j < subSet->getNumElements(); ++j)
        {
            steps.push_back(subSet->getElement(j));
        }
        i++;
    }

    const coDistributedObject **timesteps = new const coDistributedObject *[*numTimesteps + 1];
    for (int i = 0; i < *numTimesteps; ++i)
    {
        const coDistributedObject **block = new const coDistributedObject *[numBlocks + 1];
        for (int j = 0; j < numBlocks; ++j)
        {
            block[j] = blocks[j][i];
        }
        block[numBlocks] = NULL;

        char suffix[20];
        snprintf(suffix, sizeof(suffix), "_ts_%d", i);
        std::string name(objInfo.getName());
        name.append(suffix);
        coDoSet *timestep = new coDoSet(name.c_str(), block);
        timesteps[i] = timestep;
        delete[] block;
    }
    timesteps[*numTimesteps] = NULL;
    coDoSet *newSet = new coDoSet(objInfo, timesteps);
    delete[] timesteps;

    return newSet;
}

int CFX::readZone(char *Name, char *Name2, char *Name3, int timesteps, int var, int zone, float *omega, float stepduration, coDistributedObject **zoneToRead, coDistributedObject **zoneScalToRead, coDistributedObject **zoneVectToRead)
{

    char buf[3][300];
    char ts[100];
    coDoSet *timeset;
    int i, n, j;

    //cerr << "Name=" << Name << endl;
    int firststep = p_firststep->getValue();

    if (((firststep + timesteps) > cfxExportTimestepCount()) && (cfxExportTimestepCount() != 0))
    {
        cerr << "timesteps (" << timesteps << ") > number of timesteps (" << cfxExportTimestepCount() << "). timesteps resetted to ";
        timesteps = cfxExportTimestepCount() - firststep;
        cerr << timesteps << endl;
    }

    if (var == GRID)
    {
        xcoords = new float *[timesteps];
        ycoords = new float *[timesteps];
        zcoords = new float *[timesteps];

        //cerr << "entering readZone, var=GRID" << endl;
        coDistributedObject **timegrd;
        timegrd = new coDistributedObject *[timesteps + 1];

        // rotating grid! -> read all timesteps
        if ((p_readGridForAllTimeSteps->getValue()) && ((p_timeDependentZone->getValue() == (zone)) || (p_timeDependentZone->getValue() == -1)))
        {
            for (i = 0; i < timesteps; i++)
            {
                cfxExportZoneMotionAction(zone, cfxMOTION_IGNORE);
                timegrd[i] = NULL;
                sprintf(buf[0], "%s_%d", Name, i + firststep);
                currenttimestep = i;
                if (readTimeStep(buf[0], NULL, NULL, i + firststep + 1, var, zone, omega, stepduration, &timegrd[i], NULL))
                    return STOP_PIPELINE;
            }
        }
        else // static grid
        {
            cfxExportZoneMotionAction(zone, cfxMOTION_IGNORE);
            firststep = 0;
            sprintf(buf[0], "%s_%d", Name, firststep);
            currenttimestep = 0;
            timegrd[0] = NULL;
            if (readTimeStep(buf[0], NULL, NULL, 1, var, zone, omega, stepduration, &timegrd[0], NULL))
                return STOP_PIPELINE;
            for (i = 1; i < timesteps; i++)
            {
                sprintf(buf[0], "%s_%d", Name, i + firststep);
                timegrd[i] = timegrd[0];
                timegrd[i]->incRefCount();
            }
        }
        timegrd[timesteps] = NULL;
        timeset = new coDoSet(coObjInfo(Name), timegrd);

        if (timesteps > 1)
        {
            sprintf(ts, "1 %d", timesteps);
            timeset->addAttribute("TIMESTEP", ts);
        }

        *zoneToRead = timeset;

        return CONTINUE_PIPELINE;
    }
    else if (var == VECTOR)
    {
        coDistributedObject **timevec;
        timevec = new coDistributedObject *[timesteps + 1];
        for (i = 0; i < timesteps; i++)
        {
            timevec[i] = NULL;
            sprintf(buf[0], "%s_%d", Name, i + firststep);
            currenttimestep = i;
            if (readTimeStep(buf[0], NULL, NULL, i + firststep + 1, var, zone, omega, 0., &timevec[i], NULL))
                return STOP_PIPELINE;
        }
        timevec[timesteps] = NULL;
        timeset = new coDoSet(coObjInfo(Name), timevec);
        delete[] timevec;

        if (timesteps > 1)
        {
            sprintf(ts, "1 %d", timesteps);
            timeset->addAttribute("TIMESTEP", ts);
        }
        int varnum = VectIndex[p_vector->getValue() - 1];
        timeset->addAttribute("SPECIES", cfxExportVariableName(varnum, 0));
        //cerr << "vector data object: setting attribute SPECIES to " << cfxExportVariableName(varnum, 0) << endl;
        *zoneToRead = timeset;

        return CONTINUE_PIPELINE;
    }
    else if (var == SCALAR)
    {
        coDistributedObject **timescal;
        timescal = new coDistributedObject *[timesteps + 1];
        for (i = 0; i < timesteps; i++)
        {
            timescal[i] = NULL;
            sprintf(buf[0], "%s_%d", Name, i + firststep);
            if (readTimeStep(buf[0], NULL, NULL, i + firststep + 1, var, zone, NULL, 0., &timescal[i], NULL))
                return STOP_PIPELINE;
        }
        timescal[timesteps] = NULL;
        timeset = new coDoSet(coObjInfo(Name), timescal);
        delete[] timescal;

        if (timesteps > 1)
        {
            sprintf(ts, "1 %d", timesteps);
            timeset->addAttribute("TIMESTEP", ts);
        }
        int varnum = ScalIndex[p_scalar->getValue() - 1];
        timeset->addAttribute("SPECIES", cfxExportVariableName(varnum, 0));
        //cerr << "scalar data object: setting attribute SPECIES to " << cfxExportVariableName(varnum, 0) << endl;
        *zoneToRead = timeset;

        return CONTINUE_PIPELINE;
    }
    else if (var == REGION)
    {
        coDistributedObject **timereg;
        timereg = new coDistributedObject *[timesteps + 1];
        coDistributedObject **timeregscal;
        timeregscal = new coDistributedObject *[timesteps + 1];
        for (i = 0; i < timesteps; i++)
        {
            timereg[i] = NULL;
            timeregscal[i] = NULL;
            sprintf(buf[0], "%s_%d", Name, i + firststep);
            sprintf(buf[1], "%s_%d", Name2, i + firststep);
            sprintf(buf[2], "%s_%d", Name3, i + firststep);
            int res = cfxExportTimestepSet(cfxExportTimestepNumGet(firststep + i + 1));
            if ((res < 0) || (res > cfxExportTimestepCount()))
            {
                cerr << "Error in compute. cfxExportTimeStepSet returns bad value (" << res << ")!" << endl;
                return STOP_PIPELINE;
            }

            if (cfxExportZoneSet(zone, counts) < 0)
                cfxExportFatal((char *)"invalid zone number");

            if ((nregions = cfxExportRegionCount()) > 0)
            {
                coDistributedObject **regset;
                regset = new coDistributedObject *[nregions + 1];
                coDistributedObject **regsetscal;
                regsetscal = new coDistributedObject *[nregions + 1];

                coDoPolygons *reg_gridObj = NULL;
                coDoFloat *reg_scalarObj = NULL;
                int numread = 0;
                for (j = 0; j < nregions; j++)
                {
                    char buf2[3][300];
                    sprintf(buf2[0], "%s_%d", buf[0], j);
                    sprintf(buf2[1], "%s_%d", buf[1], j);
                    if (selRegion(j + 1))
                    {
                        reg_gridObj = NULL;
                        reg_scalarObj = NULL;
                        //cerr << "reading Region "<< j+1 << " timestep " << i << " (" << cfxExportRegionName(j+1) << ")" << endl;
                        if (readRegion(buf2[0], buf2[1], NULL, j + 1, p_scalar->getValue(), &reg_gridObj, &reg_scalarObj) == STOP_PIPELINE)
                            return STOP_PIPELINE;

                        regset[numread] = reg_gridObj;
                        regsetscal[numread] = reg_scalarObj;
                        regset[numread + 1] = NULL;
                        regsetscal[numread + 1] = NULL;
                        numread++;
                    }
                }

                coDoSet *regSetObj = new coDoSet(coObjInfo(buf[0]), regset);
                for (n = 0; n < numread && regset[n]; n++)
                {
                    delete regset[n];
                }
                delete[] regset;

                timereg[i] = regSetObj;

                coDoSet *regSetScalObj = new coDoSet(coObjInfo(buf[1]), regsetscal);
                for (n = 0; n < numread && regsetscal[n]; n++)
                {
                    delete regsetscal[n];
                }
                delete[] regsetscal;
                timeregscal[i] = regSetScalObj;
            }
        }
        timereg[timesteps] = NULL;
        timeset = new coDoSet(coObjInfo(Name), timereg);
        delete[] timereg;

        if (timesteps > 1)
        {
            sprintf(ts, "1 %d", timesteps);
            timeset->addAttribute("TIMESTEP", ts);
        }

        *zoneToRead = timeset;

        timeregscal[timesteps] = NULL;
        coDoSet *timeset2 = new coDoSet(coObjInfo(Name2), timeregscal);
        delete[] timeregscal;

        if (timesteps > 1)
        {
            sprintf(ts, "1 %d", timesteps);
            timeset->addAttribute("TIMESTEP", ts);
        }

        *zoneScalToRead = timeset2;

        return CONTINUE_PIPELINE;
    }

    else if (var == BOUNDARY)
    {
        //cerr << "readZone BOUNDARY" << endl;
        coDistributedObject **timebound;
        timebound = new coDistributedObject *[timesteps + 1];
        coDistributedObject **timeboundscal;
        coDistributedObject **timeboundvect;
        timeboundscal = new coDistributedObject *[timesteps + 1];
        timeboundvect = new coDistributedObject *[timesteps + 1];

        for (i = 0; i < timesteps; i++)
        {
            timebound[i] = NULL;
            timeboundscal[i] = NULL;
            timeboundvect[i] = NULL;
            sprintf(buf[0], "%s_%d", Name, i + firststep);
            sprintf(buf[1], "%s_%d", Name2, i + firststep);
            sprintf(buf[2], "%s_%d", Name3, i + firststep);

            int res = cfxExportTimestepSet(cfxExportTimestepNumGet(i + 1 + firststep));
            if ((res < 0) || (res > cfxExportTimestepCount()))
            {
                cerr << "Error in compute. cfxExportTimeStepSet returns bad value (" << res << ")!" << endl;
                return STOP_PIPELINE;
            }

            if (cfxExportZoneSet(zone, counts) < 0)
                cfxExportFatal((char *)"invalid zone number");

            int nboundaries = cfxExportBoundaryCount();
            if (nboundaries > 0)
            {
                coDistributedObject **boundset;
                boundset = new coDistributedObject *[nboundaries + 1];
                boundset[nboundaries] = NULL;
                coDistributedObject **boundsetscal;
                boundsetscal = new coDistributedObject *[nboundaries + 1];
                boundsetscal[nboundaries] = NULL;
                coDistributedObject **boundsetvect;
                boundsetvect = new coDistributedObject *[nboundaries + 1];
                boundsetvect[nboundaries] = NULL;

                coDoPolygons *bound_gridObj = NULL;
                coDoFloat *bound_scalarObj = NULL;
                coDoVec3 *bound_vectorObj = NULL;

                int numread = 0;

                // some debug output
                /*
            for (int k=1;k<=nboundaries_total;k++)
            {
               cerr << "boundary_zone_list[" << k << "]=" << boundary_zone_list[k] << endl;
            }
            */

                // get global boundary number!
                int boundnrglobstart = 0;
                while (boundary_zone_list[boundnrglobstart + 1] != zone)
                    boundnrglobstart++; // first appearance of zone

                for (j = 1; j <= nboundaries; j++) // these are just the boundaries in this zone!
                {
                    char buf2[3][300];
                    sprintf(buf2[0], "%s_%d", buf[0], j - 1);
                    sprintf(buf2[1], "%s_%d", buf[1], j - 1);
                    sprintf(buf2[2], "%s_%d", buf[2], j - 1);

                    char *boundtype;
                    int boundnrglob = j + boundnrglobstart;
                    if (boundary_type_list[boundnrglob] == WALL)
                    {
                        boundtype = strdup("WALL");
                    }
                    else if (boundary_type_list[boundnrglob] == INLET)
                    {
                        boundtype = strdup("INLET");
                    }
                    else if (boundary_type_list[boundnrglob] == OUTLET)
                    {
                        boundtype = strdup("OUTLET");
                    }
                    else if (boundary_type_list[boundnrglob] == CFX_INTERFACE)
                    {
                        boundtype = strdup("INTERFACE");
                    }
                    else if (boundary_type_list[boundnrglob] == SYMMETRY)
                    {
                        boundtype = strdup("SYMMETRY");
                    }
                    else if (boundary_type_list[boundnrglob] == OPENING)
                    {
                        boundtype = strdup("OPENING");
                    }
                    else
                    {
                        cerr << "Unknown boundary type (" << boundary_type_list[boundnrglob] << "). Please contact Covise support! " << endl;
                        //cerr << "j=" << j << ", boundnrglobstart=" << boundnrglobstart << endl;
                        return STOP_PIPELINE;
                    }

                    if (selBoundary(boundnrglob) || !(strcmp(p_boundary->getValue(), boundtype)))
                    {
                        bound_gridObj = NULL;
                        bound_scalarObj = NULL;
                        bound_vectorObj = NULL;

                        //cerr << "reading boundary "<< boundnrglob << " timestep " << i << " (" << cfxExportBoundaryName(j) << ")" << endl;

                        if (readBoundary(buf2[0],
                                         buf2[1],
                                         buf2[2],
                                         j,
                                         zone,
                                         p_boundScalar->getValue(),
                                         p_boundVector->getValue(),
                                         &bound_gridObj,
                                         &bound_scalarObj,
                                         &bound_vectorObj) == STOP_PIPELINE)
                            return STOP_PIPELINE;

                        boundset[numread] = bound_gridObj;
                        boundsetscal[numread] = bound_scalarObj;
                        boundsetvect[numread] = bound_vectorObj;
                        boundset[numread + 1] = NULL;
                        boundsetscal[numread + 1] = NULL;
                        boundsetvect[numread + 1] = NULL;
                        numread++;
                    }
                }

                // NULL-stop for set. Even if nothing was read (numRead == 0)
                boundset[numread] = NULL;
                boundsetscal[numread] = NULL;
                boundsetvect[numread] = NULL;

                coDoSet *boundSetObj = new coDoSet(coObjInfo(buf[0]), boundset);
                for (n = 0; n < numread && boundset[n]; n++)
                {
                    delete boundset[n];
                }
                delete[] boundset;

                timebound[i] = boundSetObj;

                coDoSet *boundSetScalObj = new coDoSet(coObjInfo(buf[1]), boundsetscal);

                for (n = 0; n < numread && boundsetscal[n]; n++)
                {
                    delete boundsetscal[n];
                }
                delete[] boundsetscal;

                timeboundscal[i] = boundSetScalObj;

                coDoSet *boundSetVectObj = new coDoSet(coObjInfo(buf[2]), boundsetvect);

                for (n = 0; n < numread && boundsetvect[n]; n++)
                {
                    delete boundsetvect[n];
                }
                delete[] boundsetvect;

                timeboundvect[i] = boundSetVectObj;
            }
        }

        timebound[timesteps] = NULL;
        timeset = new coDoSet(coObjInfo(Name), timebound);
        for (i = 0; i < timesteps && timebound[i]; i++)
        {
            delete timebound[i];
        }
        delete[] timebound;

        if (timesteps > 1)
        {
            sprintf(ts, "1 %d", timesteps);
            timeset->addAttribute("TIMESTEP", ts);
        }

        *zoneToRead = timeset;

        timeboundscal[timesteps] = NULL;
        coDoSet *timesetscal = new coDoSet(coObjInfo(Name2), timeboundscal);
        for (i = 0; i < timesteps && timeboundscal[i]; i++)
        {
            delete timeboundscal[i];
        }
        delete[] timeboundscal;

        if (timesteps > 1)
        {
            sprintf(ts, "1 %d", timesteps);
            timesetscal->addAttribute("TIMESTEP", ts);
        }

        *zoneScalToRead = timesetscal;

        timeboundvect[timesteps] = NULL;
        coDoSet *timesetvect = new coDoSet(coObjInfo(Name2), timeboundvect);
        for (i = 0; i < timesteps && timeboundvect[i]; i++)
        {
            delete timeboundvect[i];
        }
        delete[] timeboundvect;

        if (timesteps > 1)
        {
            sprintf(ts, "1 %d", timesteps);
            timesetvect->addAttribute("TIMESTEP", ts);
        }

        *zoneVectToRead = timesetvect;

        return CONTINUE_PIPELINE;
    }

    else
    {
        cerr << "Error in readZone. None-existing value for var" << endl;
        return STOP_PIPELINE;
    }

    return STOP_PIPELINE;
}

int CFX::readTimeStep(char *Name, char *Name2, char *Name3, int timestep, int var, int zone, float *omega, float stepduration, coDistributedObject **timeStepToRead, coDistributedObject **regScal)
{
    //int counts[cfxCNT_SIZE]; declared globally
    cerr << "CFX::readTimeStep. var=" << var << ", zone=" << zone << ", timestep=" << timestep << endl;
    int res = cfxExportTimestepSet(cfxExportTimestepNumGet(timestep));
    //cerr << "cfxExportTimestepSet returns " << res << endl;
    if ((res < 0) || (res > cfxExportTimestepCount()))
    {
        cerr << "Error in compute. cfxExportTimeStepSet returns bad value (" << res << ")!" << endl;
        return STOP_PIPELINE;
    }

    if (cfxExportZoneSet(zone, counts) < 0)
        cfxExportFatal((char *)"invalid zone number");

    int nelems = 0;
    int nconn = 0;
    int nnodes = 0;
    nelems = cfxExportElementCount();
    nconn = counts[cfxCNT_TET] * 4 + counts[cfxCNT_WDG] * 6 + counts[cfxCNT_HEX] * 8 + counts[cfxCNT_PYR] * 5;
    nnodes = cfxExportNodeCount();

    if (var == GRID)
    {
        coDoUnstructuredGrid *gridObj = NULL;

        float omega_zone;
        if (omega)
        {
            omega_zone = omega[zone];
        }
        else
        {
            omega_zone = 0.;
        }

        if (readGrid(Name, nelems, nconn, nnodes, timestep, zone, omega_zone, stepduration, &gridObj) == STOP_PIPELINE)
            return STOP_PIPELINE;

        *timeStepToRead = gridObj;
        return CONTINUE_PIPELINE;
    }

    else if (var == VECTOR)
    {
        coDoVec3 *vectorObj = NULL;

        t = timestep;

        if (readVector(Name, nnodes, p_vector->getValue(), zone, omega, &vectorObj) == STOP_PIPELINE)
            return STOP_PIPELINE;

        *timeStepToRead = vectorObj;
        return CONTINUE_PIPELINE;
    }

    else if (var == SCALAR)
    {
        coDoFloat *scalarObj = NULL;

        if (readScalar(Name, nnodes, p_scalar->getValue(), &scalarObj) == STOP_PIPELINE)
            return STOP_PIPELINE;

        *timeStepToRead = scalarObj;
        return CONTINUE_PIPELINE;
    }

    else if ((var == REGION) || (var == REGSCAL))
    {
        coDoPolygons *reg_gridObj = NULL;
        coDoFloat *reg_scalarObj = NULL;

        if (readRegion(Name, Name2, Name3, 0, p_scalar->getValue(), &reg_gridObj, &reg_scalarObj) == STOP_PIPELINE)
            return STOP_PIPELINE;

        *timeStepToRead = reg_gridObj;
        *regScal = reg_scalarObj;
        return CONTINUE_PIPELINE;
    }

    else
    {
        cerr << "Error in readTimeStep. None-existing value for var" << endl;
        return STOP_PIPELINE;
    }

    return STOP_PIPELINE;
}

int CFX::readGrid(const char *gridName, int nelems, int nconn, int nnodes, int tstep, int /* zone */, float omega, float stepduration, coDoUnstructuredGrid **gridObj)
{
    float *x_coord, *y_coord, *z_coord; // coordinate lists
    int *el, *cl, *tl;
    int i;
    cfxElement *elems;
    //bool rotate=false;
    //Matrix *rotMatrix = new Matrix();
    //float angleDEG = 0.0f;

    int iteration = cfxExportTimestepNumGet(tstep);

    if (cfxExportTimestepSet(iteration) < 0)
    {
        return STOP_PIPELINE;
    }

    // read grid just for 1st timestep or if it contains rotating parts
    if ((tstep > 1) && (!p_readGridForAllTimeSteps->getValue()))
    {
        return CONTINUE_PIPELINE;
    }

    // create the structured grid object for the mesh
    if (gridName != NULL)
    {
        //cerr<< "line 318"<<endl;
        *gridObj = new coDoUnstructuredGrid(gridName, nelems, nconn, nnodes, 1);
        if ((*gridObj)->objectOk())
        {
            // get pointers to the element, vertex and coordinate lists
            (*gridObj)->getAddresses(&el, &cl, &x_coord, &y_coord, &z_coord);
            // remember pointers to coordinate arrays
            xcoords[currenttimestep] = x_coord;
            ycoords[currenttimestep] = y_coord;
            zcoords[currenttimestep] = z_coord;

            (*gridObj)->getTypeList(&tl);

            //cerr<<"line 324"<<endl;
            //nodes = cfxExportNodeList();
            //cerr<<"nodes= "<<nodes;
            typedef struct
            {
                double x, y, z;
            } corr_cfxNode;
            corr_cfxNode *nodes = (corr_cfxNode *)cfxExportNodeList();
            for (i = 0; i < nnodes; i++, nodes++)
            {
                //printf("%8d %12.5e %12.5e %12.5e\n", i + 1, nodes->x,
                //  nodes->y, nodes->z );
                // cfxExportNodeGet(i+1,&x, &y, &z);
                x_coord[i] = (float)nodes->x;
                y_coord[i] = (float)nodes->y;
                z_coord[i] = (float)nodes->z;
            }

            //float time = cfxExportTimestepTimeGet(tstep);
            //cerr << "time=" << time << endl;
            /*
                  if ( (rotate) && (fabs(angleDEG)>0.0001) )
                  {
                     //rotate!
                  cerr << "rotating grid with angle (deg) " << angleDEG << endl;
                     rotMatrix->transformCoordinates(nnodes,x_coord,y_coord,z_coord);
                }
         */
            //cerr<<"line 337"<<endl;
            cfxExportNodeFree(); // @@@
            //cerr<<"line 339"<<endl;
            elems = cfxExportElementList();
            //cerr<<"line 341"<<endl;
            int crt_index = 0;
            for (i = 0; i < nelems; i++, elems++)
            {
                el[i] = crt_index;
                for (int j = 0; j < elems->type; j++)
                {
                    cl[crt_index++] = elems->nodeid[j] - 1;
                }
                switch (elems->type)
                {
                case cfxELEM_TET:
                    tl[i] = TYPE_TETRAHEDER;
                    break;
                case cfxELEM_PYR:
                    tl[i] = TYPE_PYRAMID;
                    break;
                case cfxELEM_WDG:
                    tl[i] = TYPE_PRISM;
                    break;
                case cfxELEM_HEX:
                    int aux = cl[crt_index - 1];
                    cl[crt_index - 1] = cl[crt_index - 2];
                    cl[crt_index - 2] = aux;
                    aux = cl[crt_index - 5];
                    cl[crt_index - 5] = cl[crt_index - 6];
                    cl[crt_index - 6] = aux;
                    tl[i] = TYPE_HEXAEDER;
                    break;
                }
            }
            cfxExportElementFree();
            //cerr<<"line 360"<<endl;
        }
        else
        {
            sendError("Failed to create the object '%s' for the port '%s'", gridName, p_outPort1->getName());
            return STOP_PIPELINE;
        }
    }
    else
    {
        return STOP_PIPELINE;
    }

    if (fabs(stepduration) > 0.)
    {
        char rotaxis[100];
        char rotaxispoint[100];
        char rotspeed[100];
        char tsteplength[100];

        sprintf(rotaxis, "%5.2f %5.2f %5.2f", p_rotAxis->getValue(0), p_rotAxis->getValue(1), p_rotAxis->getValue(2));
        sprintf(rotaxispoint, "%5.2f %5.2f %5.2f", p_rotVertex->getValue(0), p_rotVertex->getValue(1), p_rotVertex->getValue(2));
        sprintf(rotspeed, "%9.6f", omega);
        sprintf(tsteplength, "%16.10f", stepduration);

        (*gridObj)->addAttribute("ROT_AXIS", rotaxis);
        (*gridObj)->addAttribute("ROT_AXIS_POINT", rotaxispoint);
        (*gridObj)->addAttribute("ROT_SPEED", rotspeed);
        (*gridObj)->addAttribute("STEPDURATION", tsteplength);
    }

    return CONTINUE_PIPELINE;
}

int CFX::readScalar(const char *scalarName, int n_values, int scal, coDoFloat **scalarObj)
{
    float *sdata;
    if (scalarName != NULL)
    {
        if (scal > 0)
        {
            float *var = cfxExportVariableList(ScalIndex[scal - 1], 0);
            if (var)
            {
                *scalarObj = new coDoFloat(scalarName, n_values);

                if ((*scalarObj)->objectOk())
                {
                    (*scalarObj)->getAddress(&sdata);
                    for (int i = 0; i < n_values; i++)
                    {
                        sdata[i] = var[i] * scalarFactor + scalarDelta;
                        //if (i==0) fprintf(stderr, "********* var[i] = %f, sdata[i] = %f, scalarDelta = %f\n", var[i], sdata[i], scalarDelta);
                    }
                }
                else
                {
                    sendError("Failed to create the object '%s' for the port '%s'", scalarName, p_outPort2->getName());
                    return STOP_PIPELINE;
                }
            }
            cfxExportVariableFree(ScalIndex[scal - 1]);
        }
        else
        {
            *scalarObj = new coDoFloat(scalarName, 1);
        }
    }
    else
    {
        return STOP_PIPELINE;
    }

    return CONTINUE_PIPELINE;
}

int CFX::readVector(const char *vectorName, int n_values, int vect, int zone, float *omega, coDoVec3 **vectorObj)
{
    float *vdata1, *vdata2, *vdata3, r = 0.0f;

    if (vectorName != NULL)
    {
        if (vect > 0)
        {
            float *var = cfxExportVariableList(VectIndex[vect - 1], 0);

            if (var)
            {
                *vectorObj = new coDoVec3(vectorName, n_values);

                if ((*vectorObj)->objectOk())
                {
                    (*vectorObj)->getAddresses(&vdata1, &vdata2, &vdata3);
                    for (int i = 0, j = 0; i <= n_values * 3; i += 3, j++)
                    {
                        vdata1[j] = var[i];
                        vdata2[j] = var[i + 1];
                        vdata3[j] = var[i + 2];
                    }
                }
                else
                {
                    sendError("Failed to create the object '%s' for the port '%s'", vectorName, p_outPort3->getName());
                    return STOP_PIPELINE;
                }

                // rotating velocity vectors according to timestep and angle / timestep
                float alpha = ((t - 1) * p_rotAngle->getValue() * M_PI / 180.);
                float ca = cos(alpha);
                float sa = sin(alpha);
                float x, y, z;
                if ((p_rotVectors->getValue()) && (zone == p_relabs_zone->getValue()))
                {
                    fprintf(stderr, "\trotating velocity vectors with alpha = %7.3lf\n", (t - 1) * p_rotAngle->getValue());
                    if (p_relabsAxis->getValue() == 0) // x-axis
                    {
                        for (int i = 0; i < n_values; i++)
                        {
                            y = vdata2[i];
                            z = vdata3[i];
                            vdata2[i] = y * ca + z * sa;
                            vdata3[i] = y * -sa + z * ca;
                        }
                    }
                    if (p_relabsAxis->getValue() == 1) // y-axis
                    {
                        for (int i = 0; i < n_values; i++)
                        {
                            x = vdata1[i];
                            z = vdata3[i];
                            vdata1[i] = x * ca + z * -sa;
                            vdata3[i] = x * sa + z * ca;
                        }
                    }
                    if (p_relabsAxis->getValue() == 2) // z-axis
                    {
                        for (int i = 0; i < n_values; i++)
                        {
                            x = vdata1[i];
                            y = vdata2[i];
                            vdata1[i] = (x * ca) + (y * sa);
                            vdata2[i] = (x * -sa) + (y * ca);
                        }
                    }
                }

                if ((p_relabs->getValue()) && (p_relabs_zone->getValue() == zone))
                {
                    // transform vector from absolute to relative system or vice versa
                    cerr << "  ----- transforming velocity in zone " << zone << ", axis is " << p_relabsAxis->getValue() << endl;

                    // transform from relative to abolute
                    if (p_relabsDirection->getValue() == 1)
                    {
                        r = -1;
                    }
                    // transform from relative to abolute
                    if (p_relabsDirection->getValue() == 2)
                    {
                        r = 1;
                    }

                    // transform from relative to absolute system
                    if (p_relabsAxis->getValue() == 0) // x-axis
                    {
                        for (int i = 0; i < n_values; i++)
                        {
                            // abs2rel: r== 1
                            // rel2abs: r==-1
                            vdata1[i] = vdata1[i];
                            vdata2[i] = vdata2[i] - r * omega[zone] * zcoords[currenttimestep][i];
                            vdata3[i] = vdata3[i] + r * omega[zone] * ycoords[currenttimestep][i];
                        }
                    }
                    if (p_relabsAxis->getValue() == 1) // y-axis
                    {
                        for (int i = 0; i < n_values; i++)
                        {
                            // abs2rel: r== 1
                            // rel2abs: r==-1
                            vdata1[i] = vdata1[i] - r * omega[zone] * zcoords[currenttimestep][i];
                            vdata2[i] = vdata2[i];
                            vdata3[i] = vdata3[i] + r * omega[zone] * xcoords[currenttimestep][i];
                        }
                    }
                    if (p_relabsAxis->getValue() == 2) // z-axis
                    {
                        for (int i = 0; i < n_values; i++)
                        {
                            // abs2rel: r== 1
                            // rel2abs: r==-1
                            vdata1[i] = vdata1[i] - r * omega[zone] * ycoords[currenttimestep][i];
                            vdata2[i] = vdata2[i] + r * omega[zone] * xcoords[currenttimestep][i];
                            vdata3[i] = vdata3[i];
                        }
                    }
                }
            }
            cfxExportVariableFree(VectIndex[vect - 1]);
        }
        else
        {
            *vectorObj = new coDoVec3(vectorName, 1);
        }
    }
    else
    {
        return STOP_PIPELINE;
    }

    return CONTINUE_PIPELINE;
}

int CFX::readRegion(const char *gridName, const char *scalarName, const char *, int regnum, int scal, coDoPolygons **gridObj, coDoFloat **scalarObj)
{
    int *pl, *cl;
    float *x_coord, *y_coord, *z_coord; // coordinate lists
    float *sdata;
    int i, id;
    int nelems = 0, nconn = 0, nnodes = 0;
    int nodelist[4], psize;

    if (gridName != NULL && scalarName != NULL)
    {
        if (regnum > 0)
        {
            nelems = cfxExportRegionSize(regnum, cfxREG_FACES);
            nnodes = cfxExportRegionSize(regnum, cfxREG_NODES);
            for (i = 1; i <= nelems; i++)
            {
                cfxExportRegionGet(regnum, cfxREG_FACES, i, &id);
                psize = cfxExportFaceNodes(id, nodelist);
                nconn += psize;
                //cerr<<psize<<" ";
            }
            //cerr<<"nelems= "<<nelems<<" nconn= "<<nconn<<"nnodes= "<<nnodes<<endl;
            *gridObj = new coDoPolygons(gridName, nnodes, nconn, nelems);
            //cerr << "Region gridName=" << gridName << endl;
            if ((*gridObj)->objectOk())
            {
                *scalarObj = new coDoFloat(scalarName, nnodes);
                if (!(*scalarObj)->objectOk())
                {
                    sendError("Failed to create the object '%s' for the port '%s'", gridName, p_outPort5->getName());
                    return STOP_PIPELINE;
                }
                (*gridObj)->getAddresses(&x_coord, &y_coord, &z_coord, &cl, &pl);
                (*scalarObj)->getAddress(&sdata);

                float *svar = cfxExportVariableList(ScalIndex[scal], 0);
                int cl_index = 0, nl_index = 0;
                if (svar)
                {
                    for (i = 1; i <= nelems; i++)
                    {
                        pl[i - 1] = cl_index;
                        cfxExportRegionGet(regnum, cfxREG_FACES, i, &id);
                        //cerr<<"face_id = "<<id<<endl;
                        psize = cfxExportFaceNodes(id, nodelist);

                        for (int j = 0; j < psize; j++)
                        {
                            //cerr<<" "<<nodecoDoPolygonslist[j];
                            int k = 0, found = 0;
                            double x, y, z;
                            cfxExportNodeGet(nodelist[j], &x, &y, &z);
                            float xf, yf, zf;
                            xf = (float)x;
                            yf = (float)y;
                            zf = (float)z;

                            while (k < nl_index && !found)
                            {
                                if (xf == x_coord[k] && yf == y_coord[k] && zf == z_coord[k])
                                    found = 1;
                                k++;
                            }
                            if (!found)
                            {
                                if (nl_index < nnodes)
                                {
                                    x_coord[nl_index] = xf;
                                    y_coord[nl_index] = yf;
                                    z_coord[nl_index] = zf;
                                    sdata[nl_index] = svar[nodelist[j]] * scalarFactor + scalarDelta;
                                }
                                else
                                {
                                    cerr << "too many nodes" << endl;
                                }
                                if (cl_index < nconn)
                                {
                                    cl[cl_index++] = nl_index++;
                                }
                                else
                                {
                                    cerr << "too many vertices" << endl;
                                }
                            }
                            else
                            {
                                if (cl_index < nconn)
                                {
                                    cl[cl_index++] = k - 1;
                                }
                                else
                                {
                                    cerr << "too many vertices" << endl;
                                }
                            }
                            //cerr<<" "<<cl[cl_index-1];
                        }
                    }
                }
                //cerr<<"cl_index = "<<cl_index-1<<"nl_index = "<< nl_index-1;
                //for (i=0;i<=nnodes;i++)
                //cerr<<x_coord[i]<<"  "<<y_coord[i]<<"  "<<z_coord[i]<<"  "<<endl;
            }
            else
            {
                sendError("Failed to create the object '%s' for the port '%s'", gridName, p_outPort4->getName());
                return STOP_PIPELINE;
            }
        }
        else
        {
            //cerr<<"gridName: "<<gridName<<endl;
            *gridObj = new coDoPolygons(gridName, 0, 0, 0);
            //(*gridObj)->getAddresses(&el,&cl,&x_coord, &y_coord, &z_coord);
            //x_coord[0] = 0;y_coord[0] = 0; z_coord[0] = 0;el[0] = 0; cl[0] = 0;
            *scalarObj = new coDoFloat(scalarName, 0);
            (*scalarObj)->getAddress(&sdata);
            //sdata[0]=0;
        }
    }

    else
        return STOP_PIPELINE;

    return CONTINUE_PIPELINE;
}

int CFX::readBoundary(const char *gridName, const char *scalarName, const char *vectorName, int boundnr, int zone, int scal, int vect, coDoPolygons **gridObj, coDoFloat **scalarObj, coDoVec3 **vectorObj)
{

    float *sdata;
    float *vdata_x, *vdata_y, *vdata_z;
    int i, id;
    int nelems = 0, nconn = 0, nnodes = 0;
    int nodelist[4], psize;

    cfxExportZoneSet(zone, NULL);

    int readScal = 1;
    int readVect = 1;

    if (scal == 0)
        readScal = 0;
    if (vect == 0)
        readVect = 0;

    if (gridName != NULL)
    {
        if (boundnr > 0)
        {
            nelems = cfxExportBoundarySize(boundnr, cfxREG_FACES);
            nnodes = cfxExportBoundarySize(boundnr, cfxREG_NODES);

            // find out length of connectivity list ...
            for (i = 1; i <= nelems; i++)
            {
                cfxExportBoundaryGet(boundnr, cfxREG_FACES, i, &id);
                psize = cfxExportFaceNodes(id, nodelist);
                nconn += psize;
            }
            cerr << "    nelems= " << nelems << ", nconn= " << nconn << ", nnodes= " << nnodes << endl;

            std::vector<float> x_coord;
            std::vector<float> y_coord;
            std::vector<float> z_coord;
            std::vector<int> cl;
            std::vector<int> pl;

            std::vector<float> scalars;
            std::vector<float> vectors_x;
            std::vector<float> vectors_y;
            std::vector<float> vectors_z;

            float *svar = NULL, *vvar = NULL;
            if (readScal)
            {
                svar = cfxExportVariableList(ScalIndex[scal - 1], 0); //bugfix ohne wissen dahinter
            }
            if (readVect)
            {
                vvar = cfxExportVariableList(VectIndex[vect - 1], 0);
            }
            if (vvar == NULL)
            {
                readVect = 0;
            }
            if (svar == NULL)
            {
                readScal = 0;
            }

            int cl_index = 0, nl_index = 0;

            int *foundnode = new int[cfxExportNodeCount()];
            memset(foundnode, 0, cfxExportNodeCount() * sizeof(int));

            for (i = 1; i <= nelems; i++)
            {
                pl.push_back(cl_index);
                cfxExportBoundaryGet(boundnr, cfxREG_FACES, i, &id);
                psize = cfxExportFaceNodes(id, nodelist);

                if (psize > 4)
                {
                    cerr << "error in readBoundary (psize>4). enlarge nodelist array. Please call Covise support." << endl;
                }

                for (int j = 0; j < psize; j++)
                {
                    double x, y, z;
                    cfxExportNodeGet(nodelist[j], &x, &y, &z);
                    float xf, yf, zf;
                    xf = (float)x;
                    yf = (float)y;
                    zf = (float)z;

                    if (!foundnode[nodelist[j] - 1]) // node appears for the first time
                    {
                        if (nl_index < nnodes)
                        {
                            x_coord.push_back(xf);
                            y_coord.push_back(yf);
                            z_coord.push_back(zf);
                            if (readScal)
                            {
                                scalars.push_back(svar[nodelist[j] - 1] * scalarFactor + scalarDelta);
                            }
                            if (readVect)
                            {
                                vectors_x.push_back(vvar[3 * (nodelist[j] - 1) + 0]);
                                vectors_y.push_back(vvar[3 * (nodelist[j] - 1) + 1]);
                                vectors_z.push_back(vvar[3 * (nodelist[j] - 1) + 2]);
                            }
                        }
                        else
                        {
                            cerr << "too many nodes, nl_index = " << nl_index << endl;
                        }
                        if (cl_index < nconn)
                        {
                            cl.push_back(nl_index);
                            cl_index++;
                            nl_index++;
                        }
                        else
                        {
                            cerr << "too many vertices" << endl;
                        }
                        // we store our "replacement" node and add 1, so that it never can be =0 (array is used to check for new nodes)
                        foundnode[nodelist[j] - 1] = nl_index;
                    }
                    else // we already know about coordinates and scalar value of node
                    {
                        if (cl_index < nconn)
                        {
                            //k-1; // subtract 1
                            cl.push_back(foundnode[nodelist[j] - 1] - 1);
                            cl_index++;
                        }
                        else
                        {
                            cerr << "too many vertices" << endl;
                        }
                    }
                }
            }
            delete[] foundnode;

            *gridObj = new coDoPolygons(gridName, nl_index, &x_coord[0], &y_coord[0], &z_coord[0], nconn, &cl[0], nelems, &pl[0]);

            if (readScal)
            {
                *scalarObj = new coDoFloat(scalarName, nl_index, &scalars[0]);
                if (!(*scalarObj)->objectOk())
                {
                    sendError("Failed to create the object '%s' for the port '%s'", scalarName, p_outPort7->getName());
                    return STOP_PIPELINE;
                }
            }
            if (readVect)
            {
                *vectorObj = new coDoVec3(vectorName, nl_index, &vectors_x[0], &vectors_y[0], &vectors_z[0]);
                if (!(*vectorObj)->objectOk())
                {
                    sendError("Failed to create the object '%s' for the port '%s'", vectorName, p_outPort8->getName());
                    return STOP_PIPELINE;
                }
            }
        }
        else
        {
            //cerr<<"gridName: "<<gridName<<endl;
            *gridObj = new coDoPolygons(gridName, 0, 0, 0);
            //(*gridObj)->getAddresses(&el,&cl,&x_coord, &y_coord, &z_coord);
            //x_coord[0] = 0;y_coord[0] = 0; z_coord[0] = 0;el[0] = 0; cl[0] = 0;
            if (readScal)
            {
                *scalarObj = new coDoFloat(scalarName, 0);
                (*scalarObj)->getAddress(&sdata);
            }
            if (readVect)
            {
                *vectorObj = new coDoVec3(vectorName, 0);
                (*vectorObj)->getAddresses(&vdata_x, &vdata_y, &vdata_z);
            }
        }
    }
    else
        return STOP_PIPELINE;

    return CONTINUE_PIPELINE;
}

void CFX::setParticleVarChoice(int type)
{
    npscalars = 0;
    npvectors = 0;
    if (type > 0)
    {
        int npvars = cfxExportGetParticleTypeVarCount(type);
        delete[] psVarIndex;
        delete[] pvVarIndex;
        psVarIndex = new int[npvars + 1];
        pvVarIndex = new int[npvars + 1];
        for (int i = 1; i <= npvars; i++)
        {
            int dim = cfxExportGetParticleTypeVarDimension(type, i);
            if (dim == 1)
            {
                npscalars++;
                psVarIndex[npscalars] = i;
            }
            else
            {
                npvectors++;
                pvVarIndex[npvectors] = i;
            }
        }

        delete[] pScalChoiceVal;
        delete[] pVectChoiceVal;
        pScalChoiceVal = new char *[npscalars + 2];
        pVectChoiceVal = new char *[npvectors + 2];
        pScalChoiceVal[0] = (char *)"none";
        pVectChoiceVal[0] = (char *)"none";
        for (int i = 1; i <= npscalars; i++)
        {
            const char *tmpName = cfxExportGetParticleTypeVarName(type, psVarIndex[i], 0);
            pScalChoiceVal[i] = new char[strlen(tmpName) + 1];
            strcpy(pScalChoiceVal[i], tmpName);
        }
        pScalChoiceVal[npscalars + 1] = NULL;
        for (int i = 1; i <= npvectors; i++)
        {
            const char *tmpName = cfxExportGetParticleTypeVarName(type, pvVarIndex[i], 0);
            pVectChoiceVal[i] = new char[strlen(tmpName) + 1];
            strcpy(pVectChoiceVal[i], tmpName);
        }
        pVectChoiceVal[npvectors + 1] = NULL;
    }
    else
    {
        delete[] pScalChoiceVal;
        delete[] pVectChoiceVal;
        pScalChoiceVal = new char *[npscalars + 2];
        pVectChoiceVal = new char *[npvectors + 2];
        pScalChoiceVal[0] = (char *)"none";
        pVectChoiceVal[0] = (char *)"none";
        pScalChoiceVal[npscalars + 1] = NULL;
        pVectChoiceVal[npvectors + 1] = NULL;
    }
    if (!inMapLoading)
    {
        p_particleScalar->updateValue(npscalars + 1, pScalChoiceVal, p_particleScalar->getValue());
        p_particleVector->updateValue(npvectors + 1, pVectChoiceVal, p_particleVector->getValue());
    }
}
void
CFX::get_choice_fields(int zone)
{
    int nvalues;

    if (zone > 0 && cfxExportZoneSet(zone, NULL) < 0)
    {
        sendError("invalid zone number");
        return;
    }

    npscalars = npvectors = 0;

    int nparticleTypes = cfxExportGetNumberOfParticleTypes();
    if (nparticleTypes > 0)
    {
        ParticleTypeNames = new char *[nparticleTypes + 2];
        ParticleTypeNames[0] = (char *)"none";
        for (int i = 1; i <= nparticleTypes; i++)
        {
            const char *tmpName = cfxExportGetParticleName(i);
            ParticleTypeNames[i] = new char[strlen(tmpName) + 1];
            strcpy(ParticleTypeNames[i], tmpName);
        }
        ParticleTypeNames[nparticleTypes + 1] = NULL;
        if (!inMapLoading)
        {
            p_particleType->updateValue(nparticleTypes + 1, ParticleTypeNames, p_particleType->getValue());
        }
        //setParticleVarChoice(1);
    }

    nscalars = nvectors = 0;
    if ((nvalues = cfxExportVariableCount(usrLevel_)) > 0)
    {
        int dim, length, bndflag;
        for (int i = 1; i <= nvalues; i++)
        {
            cfxExportVariableSize(i, &dim, &length, &bndflag);
            if (dim == 1)
                nscalars++;
            else if (dim == 3)
                nvectors++;
        }
        //cerr<<"nzones = "<<nzones<<"nscalars= "<<nscalars<<", nvectors= "<<nvectors<<", length= "<<length<<endl;

        ScalChoiceVal = new char *[nscalars + 2];
        VectChoiceVal = new char *[nvectors + 2];
        ZoneChoiceVal = new char *[nzones + 2];
        ScalChoiceVal[0] = (char *)"none";
        VectChoiceVal[0] = (char *)"none";
        ZoneChoiceVal[0] = (char *)"all";

        for (int i = 1; i <= nzones; i++)
        {
            const char *zname = cfxExportZoneName(i);
            ZoneChoiceVal[i] = new char[strlen(zname) + 1];
            strcpy(ZoneChoiceVal[i], zname);
        }

        ScalIndex = new int[nscalars + 1];
        VectIndex = new int[nvectors + 1];

        const char **varname = new const char *[nvalues + 1];

        // remember full variable names
        for (int i = 1; i <= nvalues; i++)
        {
            varname[i] = new char[300];
            varname[i] = strdup(cfxExportVariableName(i, 1));
        }

        int sindex = 1, vindex = 1;
        int choice_length;
        cerr << "\n"
             << "Variable names" << endl;
        cerr << "==============" << endl;
        for (int i = 1; i <= nvalues; i++)
        {
            cfxExportVariableSize(i, &dim, &length, &bndflag);
            if (dim == 1)
            {
                const char *sname = cfxExportVariableName(i, 0);
                if (length == 1) // variable only defined at boundary nodes
                {
                    choice_length = strlen(sname) + 15;
                    ScalChoiceVal[sindex] = new char[choice_length];
                    sprintf(ScalChoiceVal[sindex], "%s_boundary_only", sname);
                }
                else
                {
                    choice_length = strlen(sname) + 1;
                    ScalChoiceVal[sindex] = new char[choice_length];
                    strcpy(ScalChoiceVal[sindex], sname);
                }
                ScalIndex[sindex - 1] = i;
                //cerr << "ScalIndex[" << sindex-1 << "]=" << cfxExportVariableName(i,0) << "(" << cfxExportVariableName(i,1) << ")" << endl;
                //cerr << "scalar variable " << sindex << ": " << cfxExportVariableName(i,0) << endl;
                cerr << "scalar[" << sindex << "]=" << cfxExportVariableName(i, 0) << " (" << varname[i] << ")"
                     << ", number of nodes=" << length << endl;
                sindex++;
            }
            else if (dim == 3)
            {
                const char *vname = cfxExportVariableName(i, 0);
                if (length == 1) // variable only defined at boundary nodes
                {
                    choice_length = strlen(vname) + 15;
                    VectChoiceVal[vindex] = new char[choice_length];
                    sprintf(VectChoiceVal[vindex], "%s_boundary_only", vname);
                }
                else
                {
                    choice_length = strlen(vname) + 1;
                    VectChoiceVal[vindex] = new char[strlen(vname) + 1];
                    strcpy(VectChoiceVal[vindex], vname);
                }
                VectIndex[vindex - 1] = i;
                //cerr << "VectIndex[" << vindex-1 << "]=" << cfxExportVariableName(i,0) << "(" << cfxExportVariableName(i,1) << ")" << endl;
                //cerr << "vector variable " << vindex << ": " << cfxExportVariableName(i,0) << endl;
                cerr << "vector[" << vindex << "]=" << cfxExportVariableName(i, 0) << " (" << varname[i] << ")"
                     << ", number of nodes=" << length << endl;
                vindex++;
            }
        }
        for (int i = 1; i <= nvalues; i++)
        {
            free((void *)varname[i]);
        }
        delete[] varname;
    }

    cerr << endl;
    cerr << "Regions (List of number, name and type)" << endl;
    cerr << "==========================================" << endl;

    if ((nregions = cfxExportRegionCount()) > 0)
    {
        RegionChoiceVal = new char *[nregions + 2];
        RegionChoiceVal[0] = (char *)"none";
        for (int i = 1; i <= nregions; i++)
        {
            const char *rname = cfxExportRegionName(i);
            RegionChoiceVal[i] = new char[strlen(rname) + 1];
            strcpy(RegionChoiceVal[i], rname);

            //cerr<<"Region "<<cfxExportRegionName(i)<<" has " << cfxExportRegionSize(i, cfxREG_FACES) <<" and "<<cfxExportRegionSize (i, cfxREG_NODES)<<endl;
        }
    }

    // boundaries
    int nboundaries; // per zone
    nboundaries_total = 0; // total

    cerr << endl;
    cerr << "Boundaries (List of number, name and type)" << endl;
    cerr << "==========================================" << endl;

    for (int j = 1; j <= nzones; j++)
    {
        cfxExportZoneSet(j, NULL);
        nboundaries = cfxExportBoundaryCount();
        nboundaries_total += nboundaries;
    }

    //cerr << "nboundaries_total=" << nboundaries_total << endl;

    // gives boundary type for each boundary
    boundary_type_list = new int[nboundaries_total + 1];
    // gives zone that contains boundary
    boundary_zone_list = new int[nboundaries_total + 1];

    int pos = 0;
    for (int j = 1; j <= nzones; j++)
    {
        cfxExportZoneSet(j, NULL);
        nboundaries = cfxExportBoundaryCount();

        const char *zoneName = cfxExportZoneName(j);
        fprintf(stderr, "Zone %d (%s): %d boundaries\n", j, zoneName, nboundaries);
        Covise::sendInfo("Zone %d %s %d boundaries\n", j, zoneName, nboundaries);
        for (int i = 1; i <= nboundaries; i++)
        {
            if (cfxExportBoundaryType(i) == NULL)
            {
                cerr << "cfxExportBoundaryType is NULL! There might be problems with your boundaries!" << endl;
                cerr << "switching off readBoundaries" << endl;
                p_readBoundaries->setValue(false);
                return;
            }
            fprintf(stderr, "%2d: %s (%s)\n", pos + i, cfxExportBoundaryName(i), cfxExportBoundaryType(i));
            Covise::sendInfo("%s - %d %s \n", zoneName, pos + i, cfxExportBoundaryName(i));
            boundary_zone_list[pos + i] = j;
            //cerr << "boundary_zone_list[" << pos+i << "]=" << j << endl;

            if (!strcmp(cfxExportBoundaryType(i), "WALL"))
            {
                boundary_type_list[pos + i] = WALL;
            }
            else if (!strcmp(cfxExportBoundaryType(i), "INLET"))
            {
                boundary_type_list[pos + i] = INLET;
            }
            else if (!strcmp(cfxExportBoundaryType(i), "OUTLET"))
            {
                boundary_type_list[pos + i] = OUTLET;
            }
            else if (!strcmp(cfxExportBoundaryType(i), "INTERFACE"))
            {
                boundary_type_list[pos + i] = CFX_INTERFACE;
            }
            else if (!strcmp(cfxExportBoundaryType(i), "SYMMETRY"))
            {
                boundary_type_list[pos + i] = SYMMETRY;
            }
            else if (!strcmp(cfxExportBoundaryType(i), "OPENING"))
            {
                boundary_type_list[pos + i] = OPENING;
            }

            else
            {
                cerr << "boundary type " << cfxExportBoundaryType(i) << "not supported yet. Can't read boundaries. Please contact Covise support! " << endl;
                p_readBoundaries->setValue(false);
            }
        }
        cerr << endl;
        pos += nboundaries;
    }
    Covise::sendInfo("Finished read boundaries");

    // @@@ cfxExportZoneFree();
}

MODULE_MAIN(IO, CFX)
