/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                        **
 **                                                (C)2002 VirCinity GmbH  **
 ** Description:   PipelineCollect                                         **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Sven Kufer		                                          **
 **         (C)  VirCinity IT- Consulting GmbH                             **
 **         Nobelstrasse 15                               		  **
 **         D- 70569 Stuttgart    			       		  **
 **                                                                        **
 **  17.10.2001                                                            **
 **                                                                        **
\**************************************************************************/

#include <do/coDoTriangleStrips.h>
#include <do/coDoIntArr.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoSet.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#include "PipelineCollect.h"

/**************************************************************************\ 
 ** Dieses Modul wird ueber die Attribute an den Input-Objekten gesteuert  **
 ** diese sind:                                                            **
 **    BLOCK_FEEDBACK  -  wie ein Feedback, zum Aufrufen des Readers       **
 **    NEXT_STEP_PARAM -  param message to chance step                     **
 **    NEXT_STEP       -  string containing the number of the next step    **
 **    LAST_STEP/LAST_BLOCK - gambling finished                            **
\**************************************************************************/

PipelineCollect::PipelineCollect(int argc, char *argv[])
    : coModule(argc, argv, "gamble parts of a set")
{
    int i;
    char portname[20];
    for (i = 0; i < NUM_PORTS; i++)
    {
        sprintf(portname, "inport_%d", i);
        p_inport[i] = addInputPort(portname, "coDistributedObject", "input object");
        sprintf(portname, "outport_%d", i);
        if (i > 0)
            p_inport[i]->setRequired(0);
        p_outport[i] = addOutputPort(portname, "coDistributedObject", "output object");
        p_outport[i]->setDependencyPort(p_inport[i]);
    }

    param_skip = addInt32Param("skip", "skip n timesteps after every step");
    param_skip->setValue(0);

    first_step = num_timesteps = 0;
}

//////
////// this is our compute-routine
//////

int PipelineCollect::compute(const char *)
{
    int i;

    const coDistributedObject *tmp_obj;
    const coDistributedObject *real_obj;

    coDoSet *set_output;

    const char **modules = new const char *[NUM_PORTS];
    int num_mods = 0;

    int skip = param_skip->getValue();
    if (skip < 0)
    {
        sendWarning("skip<0 not allowed. Please correct.");
        skip *= -1;
    }

    real_obj = p_inport[0]->getCurrentObject();
    if (real_obj == NULL)
    {
        sendError("Object at inport_0 not found");
        first_step = num_timesteps = 0;
        return STOP_PIPELINE;
    }

    if (real_obj->getAttribute("NEXT_STEP") == NULL && real_obj->getAttribute("LAST_STEP") == NULL && real_obj->getAttribute("LAST_BLOCK") == NULL)
    {
        sendError("Please use a module that controls PipelineCollect (e.g. GetSetelem)");
        first_step = num_timesteps = 0;
        return STOP_PIPELINE;
    }

    else
    {
        char objname[1280];
        coDistributedObject *newObj = NULL;

        std::string next;

        if (real_obj->getAttribute("NEXT_STEP_PARAM"))
        {
            int next_step = 0;
            // calcalute next step
            if (sscanf(real_obj->getAttribute("NEXT_STEP"), "%d", &next_step) != 1)
            {
                fprintf(stderr, "PipelineCollect: sscanf failed\n");
            }
            // store starting param
            if (first_step == 0)
            {
                first_step = next_step - 1;
            }
            if (!real_obj->getAttribute("LAST_STEP") && !real_obj->getAttribute("LAST_BLOCK"))
            {
                next_step += skip;
            }
            else
            { // reset parameter to value of first execution
                next_step = first_step;
            }

            next = std::string(real_obj->getAttribute("NEXT_STEP_PARAM"));
            char number[64];
            sprintf(number, "%d", next_step);
            next += number;
        }

        // add objects to internal list
        for (i = 0; i < NUM_PORTS; i++)
        {
            tmp_obj = p_inport[i]->getCurrentObject();
            if (tmp_obj && tmp_obj->objectOk())
            {
                //store copy of current object because objects are deleted after finishing the pipeline
                sprintf(objname, "%s_%d", p_outport[i]->getObjName(), num_timesteps);
                num_timesteps++;
                newObj = createNewObj(objname, tmp_obj);
                newObj->copyAllAttributes(tmp_obj);
                store[i].add(newObj);

                //send param msg
                if (tmp_obj->getAttribute("BLOCK_FEEDBACK"))
                {
                    const char *fbAttrib = tmp_obj->getAttribute("BLOCK_FEEDBACK");
                    Covise::set_feedback_info(fbAttrib);

                    /// check existing modules:
                    int idx;
                    for (idx = 0; idx < num_mods; idx++)
                    {
                        if (0 == strcmp(fbAttrib, modules[idx]))
                        {
                            break;
                        }
                    }
                    if (idx == num_mods)
                    {
                        modules[num_mods++] = tmp_obj->getAttribute("BLOCK_FEEDBACK");
                    }
                    Covise::send_feedback_message("PARAM", next.c_str());
                }
            }
        }

        if (real_obj->getAttribute("LAST_STEP") || real_obj->getAttribute("LAST_BLOCK"))
        { //finish
            for (i = 0; i < NUM_PORTS; i++)
            {
                int factor = 1;
                while (factor > 0)
                {
                    int cnt = 0;
                    const coDistributedObject **outobjs = new const coDistributedObject *[factor * 1024];
                    store[i].reset();
                    while ((tmp_obj = store[i].next()) && cnt < factor * 1023)
                    {
                        outobjs[cnt++] = tmp_obj;
                    }
                    outobjs[cnt] = NULL;

                    if (cnt >= factor * 1023)
                    {
                        factor++;
                        delete[] outobjs;
                    }
                    else
                    {
                        set_output = new coDoSet(p_outport[i]->getObjName(), outobjs);
                        if (real_obj->getAttribute("LAST_STEP"))
                        {
                            char step_range[100];
                            sprintf(step_range, "1 %d", cnt);
                            set_output->addAttribute("TIMESTEP", step_range);
                        }
                        p_outport[i]->setCurrentObject(set_output);
                        delete[] outobjs;
                        store[i].clear();
                        first_step = num_timesteps = 0;
                        factor = 0;
                    }
                }
            }
        }
        else
        { // go one pipeline
            for (int i = 0; i < num_mods; i++)
            {
                Covise::set_feedback_info(modules[i]);
                Covise::send_feedback_message("EXEC", "");
            }
            return STOP_PIPELINE;
        }
    }

    return CONTINUE_PIPELINE;
}

coDistributedObject *
PipelineCollect::createNewObj(const char *name, const coDistributedObject *obj) const
{
    const int DIM = 3;

    // return NULL for empty input
    if (!(obj))
        return NULL;
    if (!(name))
        return NULL;

    coDistributedObject *outObj = NULL;

    // SETELE
    if (obj->isType("SETELE"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) SETELE" << endl;
        coDoSet *shmSubSet;
        if (!(shmSubSet = (coDoSet *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::compute( ) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        int numSubSets = shmSubSet->getNumElements();
        const coDistributedObject **subSetElements = new const coDistributedObject *[numSubSets + 1];
        int i;
        for (i = 0; i < numSubSets; ++i)
        {
            const coDistributedObject *subSetEle = shmSubSet->getElement(i);
            subSetEle->incRefCount();
            subSetElements[i] = subSetEle;
        }
        subSetElements[numSubSets] = NULL;
        outObj = new coDoSet(name, subSetElements);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // Int array
    else if (obj->isType("INTARR"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) UNIGRD" << endl;
        coDoIntArr *iArr;
        if (!(iArr = (coDoIntArr *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }
        int numDim = iArr->getNumDimensions();
        const int *sizeArr = iArr->getDimensionPtr();
        const int *values = iArr->getAddress();

        outObj = new coDoIntArr(name, numDim, sizeArr, values);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // UNIFORM GRID
    else if (obj->isType("UNIGRD"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) UNIGRD" << endl;
        coDoUniformGrid *uGrd;
        if (!(uGrd = (coDoUniformGrid *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }
        int nX, nY, nZ;
        uGrd->getGridSize(&nX, &nY, &nZ);

        float xMin, xMax, yMin, yMax, zMin, zMax;
        uGrd->getMinMax(&xMin, &xMax, &yMin, &yMax, &zMin, &zMax);

        outObj = new coDoUniformGrid(name, nX, nY, nZ, xMin, xMax, yMin, yMax, zMin, zMax);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }

    // RECTILINEAR GRID
    else if (obj->isType("RCTGRD"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) RCTGRD" << endl;
        coDoRectilinearGrid *rGrd;
        if (!(rGrd = (coDoRectilinearGrid *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }
        int nX, nY, nZ;
        rGrd->getGridSize(&nX, &nY, &nZ);

        float *X, *Y, *Z;
        rGrd->getAddresses(&X, &Y, &Z);

        outObj = new coDoRectilinearGrid(name, nX, nY, nZ, X, Y, Z);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // STRUCTURED GRID
    else if (obj->isType("STRGRD"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) STRGRD" << endl;
        coDoStructuredGrid *sGrd;
        if (!(sGrd = (coDoStructuredGrid *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }
        int nX, nY, nZ;
        sGrd->getGridSize(&nX, &nY, &nZ);
        float *X, *Y, *Z;
        sGrd->getAddresses(&X, &Y, &Z);

        outObj = new coDoStructuredGrid(name, nX, nY, nZ, X, Y, Z);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // UNSGRD
    else if (obj->isType("UNSGRD"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) UNSGRD" << endl;

        coDoUnstructuredGrid *unsGrd;
        if (!(unsGrd = (coDoUnstructuredGrid *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        // get dimensions
        int nElem, nConn, nCoords;
        unsGrd->getGridSize(&nElem, &nConn, &nCoords);

        // create new arrays
        int *el, *cl, *tl;
        el = NULL;
        cl = NULL;
        tl = NULL;
        float *coords[DIM];
        int i;
        for (i = 0; i < DIM; ++i)
            coords[i] = NULL;

        unsGrd->getAddresses(&el, &cl, &coords[0], &coords[1], &coords[2]);
        if (unsGrd->hasTypeList())
        {
            unsGrd->getTypeList(&tl);
        }

        // create new DO
        outObj = new coDoUnstructuredGrid(name, nElem, nConn, nCoords, el, cl,
                                          coords[0], coords[1], coords[2], tl);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // POINTS
    else if (obj->isType("POINTS"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) POINTS" << endl;
        coDoPoints *points;
        if (!(points = (coDoPoints *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        // get dimensions
        int nPoints = points->getNumPoints();

        // new pointers
        float *coords[DIM];
        int i;
        for (i = 0; i < DIM; ++i)
            coords[i] = NULL;

        points->getAddresses(&coords[0], &coords[1], &coords[2]);

        // create new DO
        outObj = new coDoPoints(name, nPoints, coords[0], coords[1], coords[2]);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // LINES
    else if (obj->isType("LINES"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) LINES" << endl;
        coDoLines *lines;
        if (!(lines = (coDoLines *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        // get dimensions
        int nPoints = lines->getNumPoints();
        int nCorners = lines->getNumVertices();
        int nLines = lines->getNumLines();

        // create new arrays
        int *cl, *ll;
        ll = NULL;
        cl = NULL;
        float *coords[DIM];
        int i;
        for (i = 0; i < DIM; ++i)
            coords[i] = NULL;

        lines->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &ll);

        // create new DO
        outObj = new coDoLines(name, nPoints, coords[0], coords[1], coords[2],
                               nCorners, cl, nLines, ll);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // POLYGONS
    else if (obj->isType("POLYGN"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) POLYGN" << endl;
        coDoPolygons *polygons;
        if (!(polygons = (coDoPolygons *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        // get dimensions
        int nPoints = polygons->getNumPoints();
        int nCorners = polygons->getNumVertices();
        int nPolygons = polygons->getNumPolygons();

        // create new arrays
        int *cl, *pl;
        pl = NULL;
        cl = NULL;
        float *coords[DIM];
        int i;
        for (i = 0; i < DIM; ++i)
            coords[i] = NULL;

        polygons->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &pl);

        // create new DO
        outObj = new coDoPolygons(name, nPoints, coords[0], coords[1], coords[2],
                                  nCorners, cl, nPolygons, pl);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // TRIANGLE STRIPS
    else if (obj->isType("TRIANG"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) TRIANG" << endl;
        coDoTriangleStrips *triangleStrips;
        if (!(triangleStrips = (coDoTriangleStrips *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        // get dimensions
        int nPoints = triangleStrips->getNumPoints();
        int nCorners = triangleStrips->getNumVertices();
        int nStrips = triangleStrips->getNumStrips();

        // create new arrays
        int *cl, *pl;
        pl = NULL;
        cl = NULL;
        float *coords[DIM];
        int i;
        for (i = 0; i < DIM; ++i)
            coords[i] = NULL;

        triangleStrips->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &pl);

        // create new DO
        outObj = new coDoTriangleStrips(name, nPoints, coords[0], coords[1], coords[2],
                                        nCorners, cl, nStrips, pl);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // VOLUMES
    else if (obj->isType("VOLUME"))
    {
        cerr << "GetSetElem::createNewSimpleObj(..) VOLUME not supported" << endl;
    }
    // UNSTRUCTURED SCALAR DATA
    else if (obj->isType("USTSDT"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) USTSDT" << endl;
        coDoFloat *sData;
        if (!(sData = (coDoFloat *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }
        int n = sData->getNumPoints();
        float *dat;
        sData->getAddress(&dat);

        outObj = new coDoFloat(name, n, dat);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // UNSTRUCTURED VECTOR DATA
    else if (obj->isType("USTVDT"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) USTVDT" << endl;
        coDoVec3 *vData;
        if (!(vData = (coDoVec3 *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        int n = vData->getNumPoints();
        float *dat[DIM];

        vData->getAddresses(&dat[0], &dat[1], &dat[2]);

        outObj = new coDoVec3(name, n, dat[0], dat[1], dat[2]);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // RGBA DATA
    else if (obj->isType("RGBADT"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) USTVDT" << endl;
        coDoRGBA *cData;
        if (!(cData = (coDoRGBA *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        int n = cData->getNumPoints();
        int *dat;

        cData->getAddress(&dat);

        outObj = new coDoRGBA(name, n, dat);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }

    return outObj;
}

MODULE_MAIN(Tools, PipelineCollect)
