/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: Filter program in COVISE API                           ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

// this includes our own class's headers
#include "ReadStarDrop.h"
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoSet.h>
#include <config/CoviseConfig.h>
#include <errno.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif
#include <float.h>

#undef VERBOSE

#ifdef VERBOSE
FILE *debug;
#endif

const char *ReadStarDrop::s_species[6] = {
    "Droplet Location", "Drop Velocity", "Drop Temperature", "Drop Diameter",
    "Drop Mass", "Drop count"
};

ReadStarDrop::ReadStarDrop(int argc, char *argv[])
    : coModule(argc, argv, "Star Droplet file reader")
{
#ifdef VERBOSE
    debug = fopen("DEBUG", "w");
#endif

    // the LoopThrough data
    p_grid_out = addOutputPort("grid_out", "UnstructuredGrid", "multiplexed grid");
    p_grid_in = addInputPort("grid_in", "UnstructuredGrid", "grid set with REALTIME attribute");

    int i;
    char buf[16];
    for (i = 0; i < NUM_DATA_PORTS; i++)
    {
        sprintf(buf, "dataOut_%d", i);
        p_dataOut[i] = addOutputPort(buf, "Vec3|Float|Polygons|Lines|IntArr|Geometry", "data set to multiplex");
        sprintf(buf, "dataIn_%d", i);
        p_dataIn[i] = addInputPort(buf, "Vec3|Float|Polygons|Lines|IntArr|Geometry", "data set to multiplex");
        p_dataIn[i]->setRequired(0);
        p_dataOut[i]->setDependencyPort(p_dataIn[i]);
    }

    // the output ports
    p_dropOut[LOCA] = addOutputPort("location", "Points", "Droplet Location");
    p_dropOut[VELO] = addOutputPort("velocity", "Vec3", "Droplet velocity");
    p_dropOut[TEMP] = addOutputPort("temp", "Float", "Droplet Temperature");
    p_dropOut[DIAM] = addOutputPort("diameter", "Float", "Droplet Diameter");
    p_dropOut[MASS] = addOutputPort("mass", "Float", "Droplet Mass");
    p_dropOut[COUN] = addOutputPort("count", "Float", "Droplet Count");

    // hand out the mapping for additional loop-thru multiplier modules
    p_mapping = addOutputPort("mapping", "IntArr", "Grid multiplication mapping");

    // select the Droplet file name with a file browser
    p_filename = addFileBrowserParam("filename", "Droplet File");
    p_filename->setValue("data/track.trk", "*.trk;*.33");

    // how often to multiply the input data
    p_maxDrops = addInt32Param("maxPart", "Max. no. of Particles to show, 0=all");
    p_maxDrops->setValue(0);

    // how often to multiply the input data
    p_numSteps = addInt32Param("numSteps", "Number of steps to use, 0=auto");
    p_numSteps->setValue(0);

    // whether to do artificial termination or not
    p_stickHandling = addChoiceParam("stickHandling", "How to handle sticked droplets");
    static const char *labels[] = { "No stuck Droplets", "Both", "Only stuck" };
    p_stickHandling->setValue(3, labels, 0);

    // init local data
    d_lastFileName = NULL;
    d_dropArr = NULL;
    d_numDrops = 0;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++ qsort sorting function for droplets
int ReadStarDrop::dropCompare(const void *d1, const void *d2)
{
    DropRec *drop1 = (DropRec *)d1;
    DropRec *drop2 = (DropRec *)d2;

    if (drop1->t > drop2->t)
        return +1;
    if (drop1->t < drop2->t)
        return -1;

    return 0;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++ Read a droplet file, sort and analyse
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadStarDrop::readFile(const char *fileName)
{
    // open the file
    int fd = Covise::open((char *)fileName, O_RDONLY);
    if (fd < 0)
    {
        sendError("Error opening '%s': %s", fileName, strerror(errno));
        return FAIL; // still leave old droplet set intact
    }

    // check its length
    struct stat status;
    if (fstat(fd, &status) < 0)
    {
        sendError("Could not stat '%s': %s", fileName, strerror(errno));
        return FAIL; // still leave old droplet set intact
    }

    // oops, this is strange
    int fileSize = status.st_size;
    if (fileSize % sizeof(DropRec))
    {
        sendError("File length incorrect: '%s' truncated or not star33 file",
                  fileName);
        fileSize -= fileSize % sizeof(DropRec); // skip the rest
    }

    // ok, get rid of old stuff, if any exist. otherwise d_numDrops==0
    if (d_dropArr)
    {
        int i;
        for (i = 0; i < d_numDrops; i++)
            delete[] d_dropArr[i].drop;
        delete[] d_dropArr;
    }

    // we erased, so make sure we have valid state if anything after goes wrong
    d_dropArr = NULL;
    d_numDrops = 0;
    delete[] d_lastFileName;
    d_lastFileName = NULL;

    // now allocate buffer for reading : LATER: allow two-pass reading
    int numDropRecs = fileSize / sizeof(DropRec);
    DropRec *drop = new DropRec[numDropRecs];

    // read the file at once...
    sendInfo("read file");
    int bytesRead = read(fd, (void *)drop, fileSize);
    if (bytesRead != fileSize)
    {
        if (bytesRead <= 0)
            sendError("Could not read '%s'", fileName);
        else
            sendError("Error reading '%s': incomplete read", fileName);
        delete[] drop;
        return FAIL;
    }

    // count droplets and check consistency
    int i;
    bytesRead = sizeof(DropRec) - 8; // this is what it should be....
    int maxidx = 0;
    for (i = 1; i < numDropRecs; i++)
    {
        if ((drop[i].origIxd != bytesRead) || (drop[i].length2 != bytesRead))
        {
            sendError("error reading '%s': not a Star 33 file", fileName);
            delete[] drop;
            return FAIL;
        }
        if (drop[i].num > maxidx)
            maxidx = drop[i].num;
    }
    d_numDrops = maxidx + 1; // Drop #0 usually empty

    // allocate and pre-set DropInfo fields
    d_dropArr = new DropInfo[d_numDrops];
    for (i = 0; i < d_numDrops; i++)
    {
        d_dropArr[i].startTime = FLT_MIN;
        d_dropArr[i].endTime = FLT_MIN;
        d_dropArr[i].endFly = FLT_MIN;
        d_dropArr[i].numTraces = 0;
        d_dropArr[i].firstStuck = -1;
    }

    // count traces per drop and allocate buffers
    sendInfo("count traces per drop");
    for (i = 1; i < numDropRecs; i++)
        d_dropArr[abs(drop[i].num)].numTraces++;

    for (i = 0; i < d_numDrops; i++)
    {
        if (d_dropArr[i].numTraces)
            d_dropArr[i].drop = new DropRec[d_dropArr[i].numTraces];
        else
            d_dropArr[i].drop = NULL;
        // we need this as a counter when filling
        d_dropArr[i].numTraces = 0;
    }

    // now fill in the data and delete read buffer
    sendInfo("fill in the data");
    for (i = 1; i < numDropRecs; i++)
    {
        int dropNo = abs(drop[i].num);
        int &traceCount = d_dropArr[dropNo].numTraces;
        d_dropArr[dropNo].drop[traceCount] = drop[i];
        d_dropArr[dropNo].drop[traceCount].origIxd = i;
        traceCount++;
    }
    delete[] drop;

    // now we sort and find start- and endtime for each drop
    sendInfo("sort droplets");
    for (i = 0; i < d_numDrops; i++)
        if (d_dropArr[i].numTraces)
        {
            // check whether we have to sort
            int j = 0;
            while (j < d_dropArr[i].numTraces - 1
                   && d_dropArr[i].drop[j].t < d_dropArr[i].drop[j + 1].t)
                j++;

            // we do have to sort
            if (j < d_dropArr[i].numTraces - 1)
                qsort(d_dropArr[i].drop, d_dropArr[i].numTraces,
                      sizeof(DropRec), dropCompare);

            // start time
            d_dropArr[i].startTime = d_dropArr[i].drop[0].t;

            // end time
            int maxIdx = d_dropArr[i].numTraces - 1;
            d_dropArr[i].endTime = d_dropArr[i].drop[maxIdx].t;

            // check stuck ones
            while (maxIdx && d_dropArr[i].drop[maxIdx].num < 0)
                maxIdx--;

            // time of last 'flying' one
            d_dropArr[i].endFly = d_dropArr[i].drop[maxIdx].t;

            // index of first sticking one
            if (maxIdx < d_dropArr[i].numTraces)
                d_dropArr[i].firstStuck = maxIdx + 1;
        }

#ifdef VERBOSE
    fprintf(debug, "Droplet Info dump\n=================\n");
    for (i = 0; i < d_numDrops; i++)
    {
        int j;
        fprintf(debug, "\n\nDroplet # %d\n", i);
        fprintf(debug, "   startTime = %f\n", d_dropArr[i].startTime);
        fprintf(debug, "   endTime   = %f\n", d_dropArr[i].endTime);
        fprintf(debug, "   endFly    = %f\n", d_dropArr[i].endFly);
        fprintf(debug, "   numTraces = %d\n", d_dropArr[i].numTraces);
        fprintf(debug, "   firstStuck= %d\n", d_dropArr[i].firstStuck);

        fprintf(debug, "\n   Indices");
        for (j = 0; j < d_dropArr[i].numTraces; j++)
        {
            if ((j % 8) == 0)
                fprintf(debug, "\n%7d :", j);
            fprintf(debug, " %8d", d_dropArr[i].drop[j].origIxd);
        }

        fprintf(debug, "\n\n   Timesteps");

        for (j = 0; j < d_dropArr[i].numTraces; j++)
        {
            if ((j % 8) == 0)
                fprintf(debug, "\n%7d :", j);
            fprintf(debug, " %8.6f", d_dropArr[i].drop[j].t);
        }

        int n = -1;
        fprintf(debug, "\n\n   Stuck droplet times:");
        for (j = 0; j < d_dropArr[i].numTraces; j++)
        {
            if (d_dropArr[i].drop[j].num < 0)
            {
                n++;
                if ((n % 8) == 0)
                    fprintf(debug, "\n%7d :", j);
                fprintf(debug, " %8.6f", d_dropArr[i].drop[j].t);
            }
        }
    }
#endif

    d_lastFileName = strcpy(new char[strlen(fileName) + 1], fileName);
    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++ Read the meshes from the port and prepare time arrays
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadStarDrop::retrieveMeshes(const coDistributedObject *const *&meshSet,
                                 int &numGridSteps, float *&gridTime)
{
    // Mesh must be a time set or a single grid
    const coDistributedObject *obj = p_grid_in->getCurrentObject();

    // retrieve possible scaling attribute from the grid object
    const char *scale = obj->getAttribute("STAR_SCALE8");
    if (scale)
        d_scale = 1.0f / (float)atof(scale);
    else
        d_scale = 1.0f;

    if (!obj->isType("SETELE"))
    {
        if (!obj->isType("UNSGRD"))
        {
            sendError("Object at port %s must be UNSGRD or set of UNSGRD",
                      p_grid_in->getName());
            return FAIL;
        }

        // ok: we don't have a moving grid here.
        static const coDistributedObject *grid[] = { obj };
        meshSet = grid;
        gridTime = new float[1];
        numGridSteps = 1;
        *gridTime = 0.0;
        return SUCCESS;
    }
    else
    {
        /// this is a set, but is it a timestep series with REALTIME attrib?
        meshSet = ((coDoSet *)obj)->getAllElements(&numGridSteps);
        if (numGridSteps < 1)
        {
            sendError("Object at port %s is set with 0 elements", p_grid_in->getName());
            return FAIL;
        }
    }

    gridTime = new float[numGridSteps];
    int i;
    for (i = 0; i < numGridSteps; i++)
    {
        const char *time = meshSet[i]->getAttribute("REALTIME");
        if (!time)
        {
            delete[] gridTime;
            delete[] meshSet;
            sendError("Set at port %s lacks REALTIME attributes",
                      p_grid_in->getName());
            return FAIL;
        }
        gridTime[i] = (float)atof(time);
    }

    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++ Create the timesteps
int ReadStarDrop::timeArray(int numGridSteps, const float *gridTime,
                            int &numSteps, float *&timeFrame, int *&meshIdx)
{
    // legal numbering for steps: 0=automatic or as many steps as req'ed
    numSteps = p_numSteps->getValue();
    if (numSteps < 0)
    {
        p_numSteps->setValue(0);
        sendError("Illegal number of steps: set to automatic (0)");
        return FAIL;
    }

    // numSteps=0 means automatic numbering
    int i;
    if (numSteps == 0 || numSteps == numGridSteps)
    {
        numSteps = numGridSteps;
        timeFrame = new float[numSteps + 1];

        meshIdx = new int[numSteps];
        for (i = 0; i < numSteps; i++)
            meshIdx[i] = i;

        if (numSteps == 1)
        {
            timeFrame[0] = FLT_MIN;
            timeFrame[1] = FLT_MAX;
        }
        else
        {
            timeFrame[0] = 1.5f * gridTime[0] - 0.5f * gridTime[1];
            timeFrame[numSteps] = 1.5f * gridTime[numSteps - 1] - 0.5f * gridTime[numSteps - 2];
            for (i = 1; i < numSteps; i++)
                timeFrame[i] = 0.5f * (gridTime[i - 1] + gridTime[i]);
        }
    }

    // just a single grid: use droplets only
    else if (numSteps == 1)
    {
        timeFrame = new float[2];
        meshIdx = new int[1];
        timeFrame[0] = FLT_MIN; // this shoul be first and last droplet time
        timeFrame[1] = FLT_MAX;
        meshIdx[0] = 0;
    }

    // different stepping: constant time-frames and 'best' grid per frame
    else
    {
        timeFrame = new float[numSteps + 1];
        meshIdx = new int[numSteps];

        // we extend the interval between the first and last timestep by this factor
        // of the first/last interval
        float ext = 0.5;

        // first and last step interval
        timeFrame[0] = (1 + ext) * gridTime[0] - ext * gridTime[1];
        timeFrame[numSteps] = (1 + ext) * gridTime[numGridSteps - 1] - ext * gridTime[numGridSteps - 2];

        // fill interval boundaries between
        double delta = (timeFrame[numSteps] - timeFrame[0]) / numSteps;
        for (i = 1; i < numSteps; i++)
        {
            timeFrame[i] = timeFrame[0] + (float)delta * i;
        }

        // select 'best' grids per step
        for (i = 0; i < numSteps; i++)
        {
            // 'middle' time of interval
            float midTime = 0.5f * (timeFrame[i] + timeFrame[i + 1]);

            // find first grid after midTime
            int actGrid = 0;
            while (actGrid < numGridSteps && midTime > gridTime[actGrid])
                actGrid++;

            // if this isn't the first grid: compare whether one before is closer
            if (actGrid == numGridSteps)
            {
                meshIdx[i] = actGrid - 1;
            }
            else if (actGrid != 0)
            {
                if ((gridTime[actGrid] - midTime) < (midTime - gridTime[actGrid - 1]))
                    meshIdx[i] = actGrid;
                else
                    meshIdx[i] = actGrid - 1;
            }
            else
                meshIdx[i] = actGrid;
        }
    }

    // Create the 'mapping' object
    const char *mapName = p_mapping->getObjName();
    int size = numSteps;
    coDoIntArr *map = new coDoIntArr(mapName, 1, &size, meshIdx);
    p_mapping->setCurrentObject(map);

    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++ Create the LoopThrough Output objects
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

static void copyAllAttrButTimesteps(coDistributedObject *tgt, const coDistributedObject *src)
{
    int n;
    const char **name, **value;

    if (src && tgt)
    {
        n = src->getAllAttributes(&name, &value);
        if (n > 0)
        {
            int i, num = 0;
            const char **oName = name, **oValu = value;
            const char **nPtr = name, **vPtr = value;
            for (i = 0; i < n; i++)
            {
                if (strcmp(*nPtr, "TIMESTEP") != 0) // copy attrib != TIMESTEP
                {
                    *oName = *nPtr;
                    oName++;
                    *oValu = *vPtr;
                    oValu++;
                    num++;
                }
                nPtr++;
                vPtr++;
            }
            tgt->addAttributes(num, name, value);
        }
    }
    return;
}

// read a set at a certain point
const coDistributedObject *const *ReadStarDrop::readSet(coInputPort *port, int numElem)
{
    const coDistributedObject *dataObj = port->getCurrentObject();
    if (dataObj)
    {
        if (dataObj->isType("SETELE"))
        {
            int numObj;
            const coDistributedObject *const *objs = ((const coDoSet *)dataObj)->getAllElements(&numObj);
            if (numObj == numElem)
                return objs;
            else
            {
                sendWarning("Object at port '%s' has wrong number of steps", port->getName());
                return NULL;
            }
        }
        else
        {
            sendWarning("Object at port '%s' is not a timestep series", port->getName());
            return NULL;
        }
    }
    else
        return NULL;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++ Create all 'Loop-Through' output objects
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadStarDrop::createLoopThrough(const coDistributedObject *const *meshSet, int numGridSteps,
                                    int numSteps, int *meshIdx)
{
    int step, port;

    // array for output meshes
    const coDistributedObject **newMeshArr = new const coDistributedObject *[numSteps + 1];
    newMeshArr[numSteps] = NULL;

    const coDistributedObject **newDataArr[NUM_DATA_PORTS],
        *const *oldDataObj[NUM_DATA_PORTS];

    for (port = 0; port < NUM_DATA_PORTS; port++)
    {
        oldDataObj[port] = readSet(p_dataIn[port], numGridSteps);
        if (oldDataObj[port])
        {
            newDataArr[port] = new const coDistributedObject *[numSteps + 1];
            newDataArr[port][numSteps] = NULL;
        }
        else
            newDataArr[port] = NULL;
    }

    for (step = 0; step < numSteps; step++)
    {
        int useIdx = meshIdx[step];

        newMeshArr[step] = meshSet[useIdx];
        meshSet[useIdx]->incRefCount();

        for (port = 0; port < NUM_DATA_PORTS; port++)
        {
            if (oldDataObj[port])
            {
                newDataArr[port][step] = oldDataObj[port][useIdx];
                oldDataObj[port][useIdx]->incRefCount();
            }
        }
    }

    // make the output grid set
    char timestepAtt[64];
    sprintf(timestepAtt, "0 %d", numSteps - 1);

    // make the grid: we MUST NOT copy the old 'TIMESTEP' attribute, because we might have a different
    coDoSet *outSet = new coDoSet(p_grid_out->getObjName(), newMeshArr);
    if (!outSet || !outSet->objectOk())
        return FAIL;
    copyAllAttrButTimesteps(outSet, p_grid_in->getCurrentObject());
    outSet->addAttribute("TIMESTEP", timestepAtt);

    // make the grid: we MUST NOT copy the old 'TIMESTEP' attribute, because we might have a different

    for (port = 0; port < NUM_DATA_PORTS; port++)
    {
        if (oldDataObj[port])
        {
            coDoSet *outSet = new coDoSet(p_dataOut[port]->getObjName(), newDataArr[port]);
            outSet->copyAllAttributes(p_dataIn[port]->getCurrentObject());
            p_dataOut[port]->setCurrentObject(outSet);
        }
    }

    p_grid_out->setCurrentObject(outSet);

    // clean up
    for (step = 0; step < numSteps; step++)
        delete newMeshArr[step];
    delete[] newMeshArr;

    for (port = 0; port < NUM_DATA_PORTS; port++)
        if (newDataArr[port])
        {
            for (step = 0; step < numSteps; step++)
                delete newDataArr[port][numSteps];
            delete[] newDataArr[port];
        }
    // do NOT delete [] newDataArr - it's a static array ;-(

    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++ get a Droplet for a given time : return NULL if none
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ReadStarDrop::DropRec *ReadStarDrop::getDroplet(int dropNo,
                                                float tMin, float tMax,
                                                int stickHandling)
{
    // Check input
    if (dropNo < 0 || dropNo >= d_numDrops)
        return NULL;

    // the current droplet
    DropInfo &info = d_dropArr[dropNo];
    int idx;

    float t = (tMax + tMin) / 2.0f;

    static DropRec interpol;

    switch (stickHandling)
    {
    /////////////////////////////////////////////////////////////
    case NO_STUCK: // only droplets in flow at given time
        // not in this timeframe - quit here
        if (tMax < info.startTime || tMin > info.endFly)
            return NULL;

        if (info.numTraces == 1)
            return &info.drop[0];

        // find first drop after 'center' time of timestep
        idx = 0;
        while (idx < info.firstStuck && info.drop[idx].t < t)
            idx++;

        // if this is the last flying drop, we return it
        if (idx == info.firstStuck)
        {
            if (idx == 0) // this should never happen : stuckOnly
                return &info.drop[0]; // was sorted out before
        }
        else
            return &info.drop[idx - 1];

        if (idx == 0)
            return &info.drop[0];

        // Interpolate between drop[idx] and drop[idx-1]
        break;

    /////////////////////////////////////////////////////////////
    case WITH_STUCK: // allow sticking ones, too

        // not in this timeframe and not sticking- quit here
        if (tMax < info.startTime || (tMin > info.endTime && info.firstStuck < 0))
            return NULL;

        // If we have only one step, we use it
        if (info.numTraces == 1)
            return &info.drop[0];

        // find first drop after 'center' time of timestep
        idx = 0;
        while (idx < info.numTraces && info.drop[idx].t < t)
            idx++;

        // if we are behind the last drop, we return it
        if (idx == info.numTraces)
            return &info.drop[info.numTraces - 1];

        // if we are before the first droplet, we return this one
        if (idx == 0)
            return &info.drop[0];

        // Interpolate between drop[idx] and drop[idx-1]
        break;

    case ONLY_STUCK: // just show Droplets 'after normal life'

        // not in this timeframe or no stuck ones - quit here
        if (info.firstStuck < 0 || tMax < info.drop[info.firstStuck].t)
            return NULL;

        // find first drop in the interval
        idx = info.firstStuck;
        while (idx < info.numTraces && info.drop[idx].t < tMax)
            idx++;

        if (idx == info.numTraces)
            idx--;

        return &info.drop[idx];
    // we don't interpolate here
    default:
        cerr << "Unhandled value for stickHandling: " << stickHandling << endl;
        return NULL;
        break;
    }

    // droplet interpolation - use references for easier writing
    DropRec &d0 = info.drop[idx - 1];
    DropRec &d1 = info.drop[idx];

    // interpolation factor
    float f0 = (t - d0.t) / (d1.t - d0.t);
    float f1 = 1.0f - f0;

    // do NOT extrapolate
    if (f0 > 1)
        return &d0;
    if (f1 > 1)
        return &d1;

    // interpolate values
    interpol.x = f0 * d0.x + f1 * d1.x;
    interpol.y = f0 * d0.y + f1 * d1.y;
    interpol.z = f0 * d0.z + f1 * d1.z;
    interpol.u = f0 * d0.u + f1 * d1.u;
    interpol.v = f0 * d0.v + f1 * d1.v;
    interpol.w = f0 * d0.w + f1 * d1.w;
    interpol.temp = f0 * d0.temp + f1 * d1.temp;
    interpol.diam = f0 * d0.diam + f1 * d1.diam;
    interpol.mass = f0 * d0.mass + f1 * d1.mass;
    interpol.coun = f0 * d0.coun + f1 * d1.coun;

    if (interpol.temp < 250)
        cerr << "Temp low" << endl;

    return &interpol; // if we get here, we HAVE to give a droplet
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++ find the first and last step of all droplets
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void ReadStarDrop::getSteps(int numSteps, float *timeFrame,
                            int *dropStart, int *dropEnd, int stickHandling)
{
    float tMin = timeFrame[0];
    float tMax = timeFrame[numSteps];
    int dropNo;

    switch (stickHandling)
    {
    ////////////////////////////////////////////////////////////////////////
    case NO_STUCK: // given interval must touch droplet lifetime

        for (dropNo = 0; dropNo < d_numDrops; dropNo++)
        {
            DropInfo &info = d_dropArr[dropNo];

            // it never appears
            if (info.numTraces == 0 || tMax < info.startTime || tMin > info.endFly)
            {
                dropStart[dropNo] = INT_MAX;
                dropEnd[dropNo] = -1;
            }

            // we got one
            else
            {
                int step = 0;
                // start
                while (step < numSteps && timeFrame[step] < info.startTime)
                    step++;
                dropStart[dropNo] = step - 1;
                if (step == numSteps)
                    step--;

                // end
                while (step < numSteps && timeFrame[step] < info.endFly)
                    step++;

                dropEnd[dropNo] = step - 1;

                if (dropStart[dropNo] < 0)
                    dropStart[dropNo] = 0;
                if (dropEnd[dropNo] < 0)
                    dropEnd[dropNo] = 0;
            }
        }
        return;

    ////////////////////////////////////////////////////////////////////////
    case WITH_STUCK: //show all, always

        for (dropNo = 0; dropNo < d_numDrops; dropNo++)
        {
            DropInfo &info = d_dropArr[dropNo];

            // Checke whether the drop never appears
            if (info.numTraces == 0 || tMax < info.startTime || tMin > info.endTime)
            {
                dropStart[dropNo] = INT_MAX;
                dropEnd[dropNo] = -1;
            }
            else
            {
                int step = 0;
                // start
                while (step < numSteps && timeFrame[step] < info.startTime)
                    step++;
                dropStart[dropNo] = step - 1;
                if (step == numSteps)
                    step--;

                // end
                while (step < numSteps && timeFrame[step] < info.endTime)
                    step++;

                dropEnd[dropNo] = step - 1;
                if (dropStart[dropNo] < 0)
                    dropStart[dropNo] = 0;
                if (dropEnd[dropNo] < 0)
                    dropEnd[dropNo] = 0;
            }
        }
        return;

    ////////////////////////////////////////////////////////////////////////
    case ONLY_STUCK: // just show Droplets 'after normal life'

        for (dropNo = 0; dropNo < d_numDrops; dropNo++)
        {
            DropInfo &info = d_dropArr[dropNo];

            if (info.numTraces == 0 || info.firstStuck < 0)
            {
                dropStart[dropNo] = INT_MAX;
                dropEnd[dropNo] = -1;
                continue;
            }

            // Check whether the drop never appears
            float startStuckTime = info.drop[info.firstStuck].t;

            if (tMax < startStuckTime)
            {
                dropStart[dropNo] = INT_MAX;
                dropEnd[dropNo] = -1;
            }
            else
            {
                int step = 0;
                // start
                while (step < numSteps && timeFrame[step] < startStuckTime)
                    step++;
                dropStart[dropNo] = step - 1;
                dropEnd[dropNo] = numSteps - 1;
            }
        }
        return;
    }

    return;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++ Create Droplet data
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadStarDrop::createDropletObj(int numSteps, float *timeFrame)
{
    // object arrays : create all in a loop
    coDistributedObject **outObj[6];
    char nameMask[6][256];

    int portNo;
    for (portNo = 0; portNo < 6; portNo++)
    {
        outObj[portNo] = new coDistributedObject *[numSteps + 1];
        outObj[portNo][numSteps] = NULL;
        sprintf(nameMask[portNo], "%s_%%d", p_dropOut[portNo]->getObjName());
    }

    int step, drop;
    char buf[128];

    // 1=no stuck, 2=all, 3=stuck only
    int stickHandling = p_stickHandling->getValue();

    // get the point size from covise.config
    std::string pointSize = coCoviseConfig::getEntry("Module.ReadStarDrop.PointSize");

    // we already calculated the start/end Times, but now for the STEPS
    int *dropStart = new int[d_numDrops];
    int *dropEnd = new int[d_numDrops];

    getSteps(numSteps, timeFrame, dropStart, dropEnd, stickHandling);

#ifdef VERBOSE
    {
        int i;
        fprintf(debug, "\n\nSteps\n======\n\n");
        for (i = 0; i < d_numDrops; i++)
            if (dropStart[i] != INT_MAX || dropEnd[i] != -1)
                fprintf(debug, "Drop %8d : %5d - %5d\n", i, dropStart[i], dropEnd[i]);
            else
                fprintf(debug, "Drop %8d : Never appears\n", i);
    }
#endif

    ///// This may be too many droplets to show, so thin out here

    int useDrops = p_maxDrops->getValue();
    if (useDrops && useDrops < d_numDrops)
    {
        int i = 0;

        // we increase useDrops instead of decreasing d_numDrops
        while (useDrops < d_numDrops)
        {
            i = (i + 786433) % (d_numDrops - 1); // loop with prime -> reach all elements once
            dropStart[i + 1] = INT_MAX;
            dropEnd[i + 1] = -1;
            useDrops++;
        }
    }

///// loop over time steps

#ifdef VERBOSE
    fprintf(debug, "\nTimeframes used:\n---------------\n\n");
#endif

    for (step = 0; step < numSteps; step++)
    {
        // count droplets in step
        int numDrops = 0;
        for (drop = 1; drop < d_numDrops; drop++)
        {
            if (step >= dropStart[drop] && step <= dropEnd[drop])
                numDrops++;
        }
#ifdef VERBOSE
        fprintf(debug, "Step %5d: %5d Drops  Time %5f - %5f\n-----------\n",
                step, numDrops, timeFrame[step], timeFrame[step + 1]);
#endif
        // alocate output objects : create 0-sized objects as well !!!

        sprintf(buf, nameMask[LOCA], step);
        coDoPoints *locaObj = new coDoPoints(buf, numDrops);
        outObj[LOCA][step] = locaObj;
        if (!pointSize.empty())
            locaObj->addAttribute("POINTSIZE", pointSize.c_str());

        sprintf(buf, nameMask[VELO], step);
        coDoVec3 *veloObj = new coDoVec3(buf, numDrops);
        outObj[VELO][step] = veloObj;

        sprintf(buf, nameMask[TEMP], step);
        coDoFloat *tempObj = new coDoFloat(buf, numDrops);
        outObj[TEMP][step] = tempObj;

        sprintf(buf, nameMask[DIAM], step);
        coDoFloat *diamObj = new coDoFloat(buf, numDrops);
        outObj[DIAM][step] = diamObj;

        sprintf(buf, nameMask[MASS], step);
        coDoFloat *massObj = new coDoFloat(buf, numDrops);
        outObj[MASS][step] = massObj;

        sprintf(buf, nameMask[COUN], step);
        coDoFloat *counObj = new coDoFloat(buf, numDrops);
        outObj[COUN][step] = counObj;

        if (numDrops)
        {
            float *x, *y, *z, *u, *v, *w;
            float *temp, *diam, *mass, *coun;

            locaObj->getAddresses(&x, &y, &z);
            veloObj->getAddresses(&u, &v, &w);
            tempObj->getAddress(&temp);
            diamObj->getAddress(&diam);
            massObj->getAddress(&mass);
            counObj->getAddress(&coun);

            for (drop = 1; drop < d_numDrops; drop++)
            {
                DropRec *dropRec;
                if (step >= dropStart[drop]
                    && step <= dropEnd[drop])
                {
                    dropRec = getDroplet(drop, timeFrame[step], timeFrame[step + 1],
                                         stickHandling);
                    if (dropRec)
                    {
                        numDrops--;

                        *x = d_scale * dropRec->x;
                        x++;
                        *y = d_scale * dropRec->y;
                        y++;
                        *z = d_scale * dropRec->z;
                        z++;

                        *u = dropRec->u;
                        u++;
                        *v = dropRec->v;
                        v++;
                        *w = dropRec->w;
                        w++;

                        *temp = dropRec->temp;
                        temp++;
                        *diam = dropRec->diam;
                        diam++;
                        *mass = dropRec->mass;
                        mass++;
                        *coun = dropRec->coun;
                        coun++;
                    }
                    else
                    {
#ifdef VERBOSE
                        fprintf(debug, "Step %4d : Drop %7d not in Step\n", step, drop);
                        getDroplet(drop, timeFrame[step], timeFrame[step + 1], stickHandling);
#endif
                    }
                }
            }
            if (numDrops != 0)
                cerr << "Step " << step << " left " << numDrops << " empty Drops" << endl;
        }
    }

    // content for timestep attrib
    sprintf(buf, "0 %d", numSteps - 1);

    // Create all Sets
    for (portNo = 0; portNo < 6; portNo++)
    {
        coDoSet *set = new coDoSet(p_dropOut[portNo]->getObjName(), outObj[portNo]);
        set->addAttribute("TIMESTEP", buf);
        set->addAttribute("DataObjectName", "Droplet");
        if (portNo > 0)
            set->addAttribute("SPECIES", s_species[portNo]);
        p_dropOut[portNo]->setCurrentObject(set);
    }

    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++ The Compute Call-Back
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadStarDrop::compute(const char *)
{
    int numSteps; // number of steps we use : Param or automatic
    float *timeFrame; // starting Time for this step index
    int *meshIdx; // we use grid index [] first for the given step

    // retrieve filename parameter
    const char *fileName = p_filename->getValue();

    // Retrieve and test Mesh obj: Must be a set of USGs with REALTIME attribs
    const coDistributedObject *const *meshSet;
    int numGridSteps;
    float *gridTime;
    if (retrieveMeshes(meshSet, numGridSteps, gridTime) == FAIL)
        return FAIL;

    // if already read, do not read again
    if (!d_lastFileName || strcmp(fileName, d_lastFileName) || d_dropArr == NULL)
    {
        if (readFile(fileName) == FAIL)
            return FAIL;
    }

    // get numSteps, build timeFrame and meshIdx arrays
    if (timeArray(numGridSteps, gridTime, numSteps, timeFrame, meshIdx) == FAIL)
        return FAIL;

    // this is the loppThrough data: meshes are akready open, rest opened inside
    if (createLoopThrough(meshSet, numGridSteps, numSteps, meshIdx) == FAIL)
        return FAIL;

    if (createDropletObj(numSteps, timeFrame) == FAIL)
        return FAIL;

    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++ Destructor: never called since mainloop ends with exit()
ReadStarDrop::~ReadStarDrop()
{
    if (d_dropArr)
    {
        int i;
        for (i = 0; i < d_numDrops; i++)
            delete[] d_dropArr[i].drop;
        delete[] d_dropArr;
    }
}

MODULE_MAIN(Reader, ReadStarDrop)
