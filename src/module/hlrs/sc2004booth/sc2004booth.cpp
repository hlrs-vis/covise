/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <config/CoviseConfig.h>

#ifndef _WIN32
#include <strings.h>
#endif

#include <math.h>

#include "sc2004booth.h"
#include "include/booth.h"
#include <api/coFeedback.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>

#define RAD(x) ((x)*M_PI / 180.0)
#define GRAD(x) ((x)*180.0 / M_PI)

sc2004booth::sc2004booth(int argc, char *argv[])
    : coModule(argc, argv, "sc2004booth")
{
    geo = NULL;

    fprintf(stderr, "sc2004booth::sc2004booth()\n");

#ifdef USE_STARTFILE
    // start file param
    fprintf(stderr, "sc2004booth::sc2004booth() Init of StartFile\n");
    startFile = addFileBrowserParam("startFile", "Start file");
    startFile->setValue(coCoviseConfig::getEntry("value", "Module.SC2004Booth.DataPath", getenv("HOME")), "*.txt");
#endif

    // We build the User-Menue ...
    sc2004booth::CreateUserMenu();

    // the output ports
    fprintf(stderr, "sc2004booth::sc2004booth() SetOutPort\n");
    grid = addOutputPort("grid", "UnstructuredGrid", "Computation Grid");
    surface = addOutputPort("surface", "Polygons", "Surface Polygons");
    bcin = addOutputPort("bcin", "Polygons", "Cells at entry");
    bcout = addOutputPort("bcout", "Polygons", "Cells at exit");
    bcwall = addOutputPort("bcwall", "Polygons", "Cells at walls");
    boco = addOutputPort("boco", "USR_FenflossBoco", "Boundary Conditions");

    inpoints = addOutputPort("InbcNodes", "Points", "inbc nodes");
    airpoints = addOutputPort("AirbcNodes", "Points", "airbc nodes");
    venpoints = addOutputPort("VenbcNodes", "Points", "venbc nodes");
    feedback_info = addOutputPort("FeedbackInfo", "Points", "Feedback Info");

    booth = NULL;
    sg = NULL;

    isInitialized = 0;
}

void sc2004booth::postInst()
{
#ifdef USE_STARTFILE
    startFile->show();
#endif
    p_makeGrid->show();
    p_lockmakeGrid->show();
    p_openDoor->show();
    p_gridSpacing->show();
    p_nobjects->show();
    p_booth_size->show();
    p_v_in->show();
    p_v_aircond_front->show();
    p_v_aircond_middle->show();
    p_v_aircond_back->show();
    p_v_ven->show();
    p_geofile->show();
    p_rbfile->show();
}

void sc2004booth::param(const char *portname, bool)
{

    //	char buf[255];

    fprintf(stderr, "sc2004booth::param = %s\n", portname);

#ifdef USE_STARTFILE
    if (strcmp(portname, "startFile") == 0)
    {
        if (isInitialized)
        {
            sendError("We Had an input file before...");
            return;
        }
        Covise::getname(buf, startFile->getValue());
        if (strlen(buf) == 0)
        {
            Covise::sendError("startFile parameter incorrect");
        }
        else
        {
            fprintf(stderr, "sc2004booth::param = ReadGeometry(%s) ...", buf);
            booth = ReadStartfile(buf);
            if (booth == NULL)
            {
                Covise::sendError("ReadStartfile error");
                return;
            }
        }
    }
#endif

    //selfExec();
}

void sc2004booth::quit()
{
    // :-)
}

int sc2004booth::compute(const char *)
{

    coDoPolygons *poly;
    int i;

    char name[256];

    fprintf(stderr, "sc2004booth::compute(const char *) entering... \n");

#ifdef USE_STARTFILE
    char buf[256];
    Covise::getname(buf, startFile->getValue());
    fprintf(stderr, "sc2004booth::param = ReadGeometry(%s) ...", buf);
#endif
    if (booth != NULL)
        FreeBooth(booth);
    booth = AllocBooth();

#ifdef USE_STARTFILE
    ReadStartfile(buf, booth);
#else
    GetParamsFromControlPanel(booth);
#endif

    if (!booth)
    {
        sendError("Please select a parameter file first!!");
        return FAIL;
    }

    //// Cover plugin information object is created here
    createFeedbackObjects();

    /////////////////////////////
    // create geometry for COVISE
    if ((ci = CreateGeometry4Covise(booth)))
    {
        fprintf(stderr, "sc2004booth::compute(const char *): Geometry created\n");

        poly = new coDoPolygons(surface->getObjName(),
                                ci->p->nump,
                                ci->p->x, ci->p->y, ci->p->z,
                                ci->vx->num, ci->vx->list,
                                ci->pol->num, ci->pol->list);
        poly->addAttribute("MATERIAL", "metal metal.30");
        poly->addAttribute("vertexOrder", "1");
        surface->setCurrentObject(poly);
    }
    else
        fprintf(stderr, "Error in CreateGeometry4Covise (%s, %d)\n", __FILE__, __LINE__);

    //////////////// This creates the volume grid ////////////////////

    ////////////////////////
    // if button is pushed --> create computation grid
    if (p_makeGrid->getValue())
    {
        int size[2];

        if (p_lockmakeGrid->getValue() == 0)
            p_makeGrid->setValue(0); // push off button

        if (booth == NULL)
        {
            sendError("Cannot create grid because booth is NULL!");
            return (1);
        }
        booth->spacing = p_gridSpacing->getValue();
        if ((sg = CreateSC2004Grid(booth)) == NULL)
        {
            fprintf(stderr, "Error in CreateSC2004Grid!\n");
            return -1;
        }
        sg->bc_inval = 100;
        sg->bc_outval = 110;

        fprintf(stderr, "sc2004booth: Grid created\n");

        coDoUnstructuredGrid *unsGrd = new coDoUnstructuredGrid(grid->getObjName(), // name of USG object
                                                                sg->e->nume, // number of elements
                                                                8 * sg->e->nume, // number of connectivities
                                                                sg->p->nump, // number of coordinates
                                                                1); // does type list exist?

        int *elem, *conn, *type;
        float *xc, *yc, *zc;
        unsGrd->getAddresses(&elem, &conn, &xc, &yc, &zc);
        unsGrd->getTypeList(&type);

        printf("nelem  = %d\n", sg->e->nume);
        printf("nconn  = %d\n", 8 * sg->e->nume);
        printf("nccord = %d\n", sg->p->nump);

        int **GgridConn = sg->e->e;
        for (i = 0; i < sg->e->nume; i++)
        {
            *elem = 8 * i;
            elem++;

            *conn = (*GgridConn)[0];
            conn++;
            *conn = (*GgridConn)[1];
            conn++;
            *conn = (*GgridConn)[2];
            conn++;
            *conn = (*GgridConn)[3];
            conn++;
            *conn = (*GgridConn)[4];
            conn++;
            *conn = (*GgridConn)[5];
            conn++;
            *conn = (*GgridConn)[6];
            conn++;
            *conn = (*GgridConn)[7];
            conn++;

            *type = TYPE_HEXAGON;
            type++;

            GgridConn++;
        }

        // copy geometry coordinates to unsgrd
        memcpy(xc, sg->p->x, sg->p->nump * sizeof(float));
        memcpy(yc, sg->p->y, sg->p->nump * sizeof(float));
        memcpy(zc, sg->p->z, sg->p->nump * sizeof(float));

        // set out port
        grid->setCurrentObject(unsGrd);

        // boundary condition lists
        // 1. Cells at walls
        poly = new coDoPolygons(bcwall->getObjName(),
                                sg->p->nump,
                                sg->p->x, sg->p->y, sg->p->z,
                                sg->bcwall->num, sg->bcwall->list,
                                sg->bcwallpol->num, sg->bcwallpol->list);
        //poly->addAttribute("MATERIAL","metal metal.30");
        poly->addAttribute("vertexOrder", "1");
        bcwall->setCurrentObject(poly);

        // 2. Cells at outlet
        poly = new coDoPolygons(bcout->getObjName(),
                                sg->p->nump,
                                sg->p->x, sg->p->y, sg->p->z,
                                sg->bcout->num, sg->bcout->list,
                                sg->bcoutpol->num, sg->bcoutpol->list);
        //poly->addAttribute("MATERIAL","metal metal.30");
        poly->addAttribute("vertexOrder", "1");
        bcout->setCurrentObject(poly);

        // 3. Cells at inlet
        poly = new coDoPolygons(bcin->getObjName(),
                                sg->p->nump,
                                sg->p->x, sg->p->y, sg->p->z,
                                sg->bcin->num, sg->bcin->list,
                                sg->bcinpol->num, sg->bcinpol->list);
        //poly->addAttribute("MATERIAL","metal metal.30");
        poly->addAttribute("vertexOrder", "1");
        bcin->setCurrentObject(poly);

        // we had several additional info, we should send to the
        // Domaindecomposition:
        //   0. number of columns per info
        //   1. type of node
        //   2. type of element
        //   3. list of nodes with bc (a node may appear more than one time)
        //   4. corresponding type to 3.
        //   5. wall
        //   6. balance
        //   7. pressure
        //   8. NULL

        coDistributedObject *partObj[10];
        int *data;
        float *bPtr;
        const char *basename = boco->getObjName();

        //   0. number of columns per info
        sprintf(name, "%s_colinfo", basename);
        size[0] = 6;
        size[1] = 0;
        coDoIntArr *colInfo = new coDoIntArr(name, 1, size);
        data = colInfo->getAddress();
        data[0] = SG_COL_NODE; // (=2)
        data[1] = SG_COL_ELEM; // (=2)
        data[2] = SG_COL_DIRICLET; // (=2)
        data[3] = SG_COL_WALL; // (=6)
        data[4] = SG_COL_BALANCE; // (=6)
        data[5] = SG_COL_PRESS; // (=6)
        partObj[0] = colInfo;

        //   1. type of node
        sprintf(name, "%s_nodeinfo", basename);
        size[0] = SG_COL_NODE;
        size[1] = sg->p->nump;
        coDoIntArr *nodeInfo = new coDoIntArr(name, 2, size);
        data = nodeInfo->getAddress();
        for (i = 0; i < sg->p->nump; i++)
        {
            *data++ = i + 1; // may be, that we later do it really correct
            *data++ = 0; // same comment ;-)
        }
        partObj[1] = nodeInfo;

        //   2. type of element
        sprintf(name, "%s_eleminfo", basename);
        size[0] = 2;
        size[1] = sg->e->nume * SG_COL_ELEM;
        coDoIntArr *elemInfo = new coDoIntArr(name, 2, size);
        data = elemInfo->getAddress();
        for (i = 0; i < sg->e->nume; i++)
        {
            *data++ = i + 1; // may be, that we later do it really corect
            *data++ = 0; // same comment ;-)
        }
        partObj[2] = elemInfo;

        //   3. list of nodes with bc (a node may appear more than one time)
        //      and its types
        sprintf(name, "%s_diricletNodes", basename);
        int num_diriclet = sg->bcin_nodes->num + sg->bcair_nodes->num + sg->bcven_nodes->num;

        size[0] = SG_COL_DIRICLET;
        size[1] = 6 * (num_diriclet);
        coDoIntArr *diricletNodes = new coDoIntArr(name, 2, size);
        data = diricletNodes->getAddress();

        //   4. corresponding value to 3.
        sprintf(name, "%s_diricletValue", basename);
        coDoFloat *diricletValues = new coDoFloat(name, 6 * num_diriclet);
        diricletValues->getAddress(&bPtr);

        for (i = 0; i < sg->bcin_nodes->num; i++)
        {
            *data++ = sg->bcin_nodes->list[i] + 1; // node-number
            *data++ = 1; // type of node
            *bPtr++ = sg->bcin_velos->list[5 * i + 0]; // u

            *data++ = sg->bcin_nodes->list[i] + 1; // node-number
            *data++ = 2; // type of node
            *bPtr++ = sg->bcin_velos->list[5 * i + 1]; // v

            *data++ = sg->bcin_nodes->list[i] + 1; // node-number
            *data++ = 3; // type of node
            *bPtr++ = sg->bcin_velos->list[5 * i + 2]; // w

            *data++ = sg->bcin_nodes->list[i] + 1; // node-number
            *data++ = 4; // type of node
            *bPtr++ = sg->bcin_velos->list[5 * i + 4]; // epsilon

            *data++ = sg->bcin_nodes->list[i] + 1; // node-number
            *data++ = 5; // type of node
            *bPtr++ = sg->bcin_velos->list[5 * i + 3]; // k

            *data++ = sg->bcin_nodes->list[i] + 1; // node-number
            *data++ = 6; // type of node
            *bPtr++ = 0.0; // temperature = 0.
        }

        for (i = 0; i < sg->bcven_nodes->num; i++)
        {
            *data++ = sg->bcven_nodes->list[i] + 1; // node-number
            *data++ = 1; // type of node
            *bPtr++ = sg->bcven_velos->list[5 * i + 0]; // u

            *data++ = sg->bcven_nodes->list[i] + 1; // node-number
            *data++ = 2; // type of node
            *bPtr++ = sg->bcven_velos->list[5 * i + 1]; // v

            *data++ = sg->bcven_nodes->list[i] + 1; // node-number
            *data++ = 3; // type of node
            *bPtr++ = sg->bcven_velos->list[5 * i + 2]; // w

            *data++ = sg->bcven_nodes->list[i] + 1; // node-number
            *data++ = 4; // type of node
            *bPtr++ = sg->bcven_velos->list[5 * i + 4]; // epsilon

            *data++ = sg->bcven_nodes->list[i] + 1; // node-number
            *data++ = 5; // type of node
            *bPtr++ = sg->bcven_velos->list[5 * i + 3]; // k

            *data++ = sg->bcven_nodes->list[i] + 1; // node-number
            *data++ = 6; // type of node
            *bPtr++ = 0.0; // temperature = 0.
        }
        for (i = 0; i < sg->bcair_nodes->num; i++)
        {
            *data++ = sg->bcair_nodes->list[i] + 1; // node-number
            *data++ = 1; // type of node
            *bPtr++ = sg->bcair_velos->list[5 * i + 0]; // u

            *data++ = sg->bcair_nodes->list[i] + 1; // node-number
            *data++ = 2; // type of node
            *bPtr++ = sg->bcair_velos->list[5 * i + 1]; // v

            *data++ = sg->bcair_nodes->list[i] + 1; // node-number
            *data++ = 3; // type of node
            *bPtr++ = sg->bcair_velos->list[5 * i + 2]; // w

            *data++ = sg->bcair_nodes->list[i] + 1; // node-number
            *data++ = 4; // type of node
            *bPtr++ = sg->bcair_velos->list[5 * i + 4]; // epsilon

            *data++ = sg->bcair_nodes->list[i] + 1; // node-number
            *data++ = 5; // type of node
            *bPtr++ = sg->bcair_velos->list[5 * i + 3]; // k

            *data++ = sg->bcair_nodes->list[i] + 1; // node-number
            *data++ = 6; // type of node
            *bPtr++ = 0.0; // temperature = 0.
        }

        partObj[3] = diricletNodes;
        partObj[4] = diricletValues;

        //   5. wall
        sprintf(name, "%s_wall", basename);
        size[0] = SG_COL_WALL;
        size[1] = sg->bcwallpol->num;
        coDoIntArr *faces = new coDoIntArr(name, 2, size);
        data = faces->getAddress();
        for (i = 0; i < sg->bcwallpol->num; i++) // Achtung bcwall->pol->num != bcwall->vol->num
        {
            *data++ = sg->bcwall->list[4 * i + 0] + 1;
            *data++ = sg->bcwall->list[4 * i + 1] + 1;
            *data++ = sg->bcwall->list[4 * i + 2] + 1;
            *data++ = sg->bcwall->list[4 * i + 3] + 1;
            *data++ = sg->bcwallvol->list[i] + 1;
            *data++ = 0; // wall: moving | standing. here: always standing!
        }
        partObj[5] = faces;

        //   6. balance
        sprintf(name, "%s_balance", basename);
        size[0] = SG_COL_BALANCE;
        size[1] = sg->bcinvol->num + sg->bcoutvol->num;

        coDoIntArr *balance = new coDoIntArr(name, 2, size);
        data = balance->getAddress();
        for (i = 0; i < sg->bcinvol->num; i++)
        {
            *data++ = sg->bcin->list[4 * i + 0] + 1;
            *data++ = sg->bcin->list[4 * i + 1] + 1;
            *data++ = sg->bcin->list[4 * i + 2] + 1;
            *data++ = sg->bcin->list[4 * i + 3] + 1;
            *data++ = sg->bcinvol->list[i] + 1;
            *data++ = sg->bc_inval;
        }
        for (i = 0; i < sg->bcoutvol->num; i++)
        {
            *data++ = sg->bcout->list[4 * i + 0] + 1;
            *data++ = sg->bcout->list[4 * i + 1] + 1;
            *data++ = sg->bcout->list[4 * i + 2] + 1;
            *data++ = sg->bcout->list[4 * i + 3] + 1;
            *data++ = sg->bcoutvol->list[i] + 1;
            *data++ = sg->bc_outval;
        }
        partObj[6] = balance;

        //  7. pressure bc: outlet elements
        sprintf(name, "%s_pressElems", basename);
        size[0] = 6;
        size[1] = 0;
        coDoIntArr *pressElems = new coDoIntArr(name, 2, size);
        data = pressElems->getAddress();

        //  8. pressure bc: value for outlet elements
        sprintf(name, "%s_pressVal", basename);
        coDoFloat *pressValues
            = new coDoFloat(name, 0);
        pressValues->getAddress(&bPtr);

        partObj[7] = pressElems;
        partObj[8] = pressValues;

        partObj[9] = NULL;

        coDoSet *set = new coDoSet((char *)basename, (coDistributedObject **)partObj);

        boco->setCurrentObject(set);

        float *xinp = new float[sg->bcin_nodes->num];
        float *yinp = new float[sg->bcin_nodes->num];
        float *zinp = new float[sg->bcin_nodes->num];
        for (i = 0; i < sg->bcin_nodes->num; i++)
        {
            xinp[i] = sg->p->x[sg->bcin_nodes->list[i]];
            yinp[i] = sg->p->y[sg->bcin_nodes->list[i]];
            zinp[i] = sg->p->z[sg->bcin_nodes->list[i]];
        }
        coDoPoints *in_points;
        in_points = new coDoPoints(inpoints->getObjName(), sg->bcin_nodes->num, xinp, yinp, zinp);
        inpoints->setCurrentObject(in_points);

        float *xairp = new float[sg->bcair_nodes->num];
        float *yairp = new float[sg->bcair_nodes->num];
        float *zairp = new float[sg->bcair_nodes->num];
        for (i = 0; i < sg->bcair_nodes->num; i++)
        {
            xairp[i] = sg->p->x[sg->bcair_nodes->list[i]];
            yairp[i] = sg->p->y[sg->bcair_nodes->list[i]];
            zairp[i] = sg->p->z[sg->bcair_nodes->list[i]];
        }
        coDoPoints *air_points;
        air_points = new coDoPoints(airpoints->getObjName(), sg->bcair_nodes->num, xairp, yairp, zairp);
        airpoints->setCurrentObject(air_points);

        float *xvenp = new float[sg->bcven_nodes->num];
        float *yvenp = new float[sg->bcven_nodes->num];
        float *zvenp = new float[sg->bcven_nodes->num];
        for (i = 0; i < sg->bcven_nodes->num; i++)
        {
            xvenp[i] = sg->p->x[sg->bcven_nodes->list[i]];
            yvenp[i] = sg->p->y[sg->bcven_nodes->list[i]];
            zvenp[i] = sg->p->z[sg->bcven_nodes->list[i]];
        }
        coDoPoints *ven_points;
        ven_points = new coDoPoints(venpoints->getObjName(), sg->bcven_nodes->num, xvenp, yvenp, zvenp);
        venpoints->setCurrentObject(ven_points);

        const char *geofile = p_geofile->getValue();
        const char *rbfile = p_rbfile->getValue();

        if (p_createGeoRbFile->getValue())
        {
            CreateGeoRbFile(sg, geofile, rbfile);
        }
    }

    ///////////////////////// Free everything ////////////////////////////////

    return SUCCESS;
}

void sc2004booth::CreateUserMenu(void)
{
    fprintf(stderr, "Entering CreateUserMenu()\n");

    char *tmp;
    int i;
    //Generates C4189
    //const char *buf;
    char path[512];

    p_makeGrid = addBooleanParam("make_grid", "make grid?");
    p_makeGrid->setValue(0);

    p_lockmakeGrid = addBooleanParam("lock_make_grid_button", "lock make grid button?");
    p_lockmakeGrid->setValue(0);

    p_createGeoRbFile = addBooleanParam("create_geo_or_rb_file", "create geo/rb file?");
    p_createGeoRbFile->setValue(0);

    p_openDoor = addBooleanParam("open_cube_door", "open cube door?");
    p_openDoor->setValue(0);

    p_gridSpacing = addFloatParam("grid_spacing", "grid spacing");
    p_gridSpacing->setValue(0.1f);

    p_booth_size = addFloatVectorParam("booth_size", "booth size");
    p_booth_size->setValue(8.8f, 6.0, 4.0);

    p_nobjects = addInt32Param("n_objects", "make grid?");
    p_nobjects->setValue(13);

    p_v_in = addFloatSliderParam("vx_inlet", "vx inlet");
    p_v_in->setValue(0., 10., 1.0);

    p_v_aircond_front = addFloatSliderParam("vz_aircond_front", "vz aircond. front");
    p_v_aircond_front->setValue(-10., 10., 1.0);

    p_v_aircond_middle = addFloatSliderParam("vz_aircond_middle", "vz aircond. middle");
    p_v_aircond_middle->setValue(-10., 10., -1.0);

    p_v_aircond_back = addFloatSliderParam("vz_aircond_back", "vz aircond. back");
    p_v_aircond_back->setValue(-10., 10., 1.0);

    p_v_ven = addFloatSliderParam("vz_ven_on_cube", "vz ven on cub");
    p_v_ven->setValue(-10., 10., 3.0);

    sprintf(path, "%s/geofile.geo", coCoviseConfig::getEntry("value", "Module.sc2005booth.GeorbPath", "c:/temp").c_str());
    p_geofile = addStringParam("GeofilePath", "geofile path");
    p_geofile->setValue(path);

    sprintf(path, "%s/rbfile.geo", coCoviseConfig::getEntry("value", "Module.sc2005booth.GeorbPath", "c:/temp").c_str());
    p_rbfile = addStringParam("RbfilePath", "rbfile path");
    p_rbfile->setValue(path);

    m_Geometry = paraSwitch("Geometry", "Select Geometry");
    geo_labels = (char **)calloc(MAX_CUBES, sizeof(char *));

    for (i = 0; i < MAX_CUBES; i++)
    {
        // create description and name
        geo_labels[i] = IndexedParameterName(GEO_SEC, i);
        paraCase(geo_labels[i]); // Geometry section

        tmp = IndexedParameterName("pos_cube", i);
        p_cubes_pos[i] = addFloatVectorParam(tmp, tmp);
        p_cubes_pos[i]->setValue(0.0, 0.0, 0.0);

        tmp = IndexedParameterName("size_cube", i);
        p_cubes_size[i] = addFloatVectorParam(tmp, tmp);
        p_cubes_size[i]->setValue(0.0, 0.0, 0.0);
        free(tmp);

        paraEndCase(); // Geometry section
    }

    paraEndSwitch(); // "GeCross section", "Select GeCross section"

#ifndef USE_STARTFILE
    setGeoParamsStandardForSC2004();
#endif
}

char *sc2004booth::IndexedParameterName(const char *name, int index)
{
    char buf[255];
    sprintf(buf, "%s_%d", name, index + 1);
    return strdup(buf);
}

void sc2004booth::createFeedbackObjects()
{

    cerr << "createFeedbackObjects()" << endl;

    coDoPoints *feedinfo;
    feedinfo = new coDoPoints(feedback_info->getObjName(), 0);
    ////////////////////////////////////////////////////////////////////
    // add the current parameter values as feedback parameters
    int i = 0;

    coFeedback feedback("TangiblePosition");
    feedback.addString("Cubes Geometry");

    // the others only if used
    for (i = 0; i < p_nobjects->getValue(); i++)
    {
        feedback.addPara(p_cubes_pos[i]);
        feedback.addPara(p_cubes_size[i]);
    }
    feedback.apply(feedinfo);
    feedback_info->setCurrentObject(feedinfo);
}

int sc2004booth::GetParamsFromControlPanel(struct sc_booth *booth)
{
    int i;

    fprintf(stderr, "entering GetParamsFromControlPanel\n");

    booth->nobjects = p_nobjects->getValue();

    booth->bcin_velo = p_v_in->getValue();
    booth->bcair_velo_front = p_v_aircond_front->getValue();
    booth->bcair_velo_middle = p_v_aircond_middle->getValue();
    booth->bcair_velo_back = p_v_aircond_back->getValue();
    booth->bcven_velo = p_v_ven->getValue();

    // alloc cubes
    if ((booth->cubes = (struct cubus **)calloc(booth->nobjects, sizeof(struct cubus *))) == NULL)
    {
        fprintf(stderr, "Not enough space!\n");
        return -1;
    }
    for (i = 0; i < booth->nobjects; i++)
    {
        if ((booth->cubes[i] = (struct cubus *)calloc(1, sizeof(struct cubus))) == NULL)
        {
            fprintf(stderr, "Not enough space!\n");
            return -1;
        }
    }

    // total size
    booth->size[0] = p_booth_size->getValue(0);
    booth->size[1] = p_booth_size->getValue(1);
    booth->size[2] = p_booth_size->getValue(2);

    for (i = 0; i < booth->nobjects; i++)
    {
        booth->cubes[i]->size[0] = p_cubes_size[i]->getValue(0);
        booth->cubes[i]->size[1] = p_cubes_size[i]->getValue(1);
        booth->cubes[i]->size[2] = p_cubes_size[i]->getValue(2);

        booth->cubes[i]->pos[0] = p_cubes_pos[i]->getValue(0);
        booth->cubes[i]->pos[1] = p_cubes_pos[i]->getValue(1);
        booth->cubes[i]->pos[2] = p_cubes_pos[i]->getValue(2);

        // relative position values
        // table bottom 1 - depends on 4
        // table bottom 2 - depends on 5
        // table bottom 3 - depends on 6
        // cafe table middle - depends on 7
        // cafe table bottom - depends on 7
        if (i == 8)
        {
            booth->cubes[i]->pos[0] += p_cubes_pos[4]->getValue(0);
            booth->cubes[i]->pos[1] += p_cubes_pos[4]->getValue(1);
            booth->cubes[i]->pos[2] += p_cubes_pos[4]->getValue(2);
        }
        if (i == 9)
        {
            booth->cubes[i]->pos[0] += p_cubes_pos[5]->getValue(0);
            booth->cubes[i]->pos[1] += p_cubes_pos[5]->getValue(1);
            booth->cubes[i]->pos[2] += p_cubes_pos[5]->getValue(2);
        }
        if (i == 10)
        {
            booth->cubes[i]->pos[0] += p_cubes_pos[6]->getValue(0);
            booth->cubes[i]->pos[1] += p_cubes_pos[6]->getValue(1);
            booth->cubes[i]->pos[2] += p_cubes_pos[6]->getValue(2);
        }
        if (i == 11)
        {
            booth->cubes[i]->pos[0] += p_cubes_pos[7]->getValue(0);
            booth->cubes[i]->pos[1] += p_cubes_pos[7]->getValue(1);
            booth->cubes[i]->pos[2] += p_cubes_pos[7]->getValue(2);
        }
        if (i == 12)
        {
            booth->cubes[i]->pos[0] += p_cubes_pos[7]->getValue(0);
            booth->cubes[i]->pos[1] += p_cubes_pos[7]->getValue(1);
            booth->cubes[i]->pos[2] += p_cubes_pos[7]->getValue(2);
        }
    }

    return (0);
}

int sc2004booth::setGeoParamsStandardForSC2004()
{
    int i;

    float sc_cubes_pos[13][3];
    float sc_cubes_size[13][3];

    //  0: cube
    //  1: long table 1
    //  2: long table 2
    //  3: long table 2
    //  4: table top 1
    //  5: table top 2
    //  6: table top 3
    //  7: cafe table top
    //  8: table bottom 1     - depends on 4
    //  9: table bottom 2     - depends on 5
    // 10: table bottom 3     - depends on 6
    // 11: cafe table middle  - depends on 7
    // 12: cafe table bottom  - depends on 7

    fprintf(stderr, "entering setGeoParamsStandardForSC2004 ...\n");
    // cube
    sc_cubes_pos[0][0] = 0.000f;
    sc_cubes_size[0][0] = 2.380f;
    sc_cubes_pos[0][1] = 0.000f;
    sc_cubes_size[0][1] = 2.380f;
    sc_cubes_pos[0][2] = 1.500f;
    sc_cubes_size[0][2] = 3.000f;

    // long table 1
    sc_cubes_pos[1][0] = -3.500f;
    sc_cubes_size[1][0] = 1.800f;
    sc_cubes_pos[1][1] = -2.675f;
    sc_cubes_size[1][1] = 0.660f;
    sc_cubes_pos[1][2] = 0.380f;
    sc_cubes_size[1][2] = 0.760f;

    // long table 2
    sc_cubes_pos[2][0] = 0.000f;
    sc_cubes_size[2][0] = 1.800f;
    sc_cubes_pos[2][1] = -2.675f;
    sc_cubes_size[2][1] = 0.660f;
    sc_cubes_pos[2][2] = 0.380f;
    sc_cubes_size[2][2] = 0.760f;

    // long table 2
    sc_cubes_pos[3][0] = 3.500f;
    sc_cubes_size[3][0] = 1.800f;
    sc_cubes_pos[3][1] = -2.675f;
    sc_cubes_size[3][1] = 0.660f;
    sc_cubes_pos[3][2] = 0.380f;
    sc_cubes_size[3][2] = 0.760f;

    // table top 1
    sc_cubes_pos[4][0] = -2.000f;
    sc_cubes_size[4][0] = 0.660f;
    sc_cubes_pos[4][1] = 0.000f;
    sc_cubes_size[4][1] = 0.660f;
    sc_cubes_pos[4][2] = 0.920f;
    sc_cubes_size[4][2] = 0.200f;

    // table top 2
    sc_cubes_pos[5][0] = -3.000f;
    sc_cubes_size[5][0] = 0.660f;
    sc_cubes_pos[5][1] = 0.000f;
    sc_cubes_size[5][1] = 0.660f;
    sc_cubes_pos[5][2] = 0.920f;
    sc_cubes_size[5][2] = 0.200f;

    // table top 3
    sc_cubes_pos[6][0] = 2.000f;
    sc_cubes_size[6][0] = 0.660f;
    sc_cubes_pos[6][1] = 0.000f;
    sc_cubes_size[6][1] = 0.660f;
    sc_cubes_pos[6][2] = 0.920f;
    sc_cubes_size[6][2] = 0.200f;

    // cafe table top
    sc_cubes_pos[7][0] = 3.000f;
    sc_cubes_size[7][0] = 0.760f;
    sc_cubes_pos[7][1] = 0.000f;
    sc_cubes_size[7][1] = 0.760f;
    sc_cubes_pos[7][2] = 0.670f;
    sc_cubes_size[7][2] = 0.050f;

    // relative position values
    // table bottom 1 - depends on 4
    sc_cubes_pos[8][0] = 0.000f;
    sc_cubes_size[8][0] = 0.580f;
    sc_cubes_pos[8][1] = 0.000f;
    sc_cubes_size[8][1] = 0.580f;
    sc_cubes_pos[8][2] = -0.510f;
    sc_cubes_size[8][2] = 0.820f;

    //table bottom 2 - depends on 5
    sc_cubes_pos[9][0] = 0.000f;
    sc_cubes_size[9][0] = 0.580f;
    sc_cubes_pos[9][1] = 0.000f;
    sc_cubes_size[9][1] = 0.580f;
    sc_cubes_pos[9][2] = -0.510f;
    sc_cubes_size[9][2] = 0.820f;

    // table bottom 3 - depends on 6
    sc_cubes_pos[10][0] = 0.000f;
    sc_cubes_size[10][0] = 0.580f;
    sc_cubes_pos[10][1] = 0.000f;
    sc_cubes_size[10][1] = 0.580f;
    sc_cubes_pos[10][2] = -0.510f;
    sc_cubes_size[10][2] = 0.820f;

    // cafe table middle - depends on 7
    sc_cubes_pos[11][0] = 0.000f;
    sc_cubes_size[11][0] = 0.180f;
    sc_cubes_pos[11][1] = 0.000f;
    sc_cubes_size[11][1] = 0.180f;
    sc_cubes_pos[11][2] = -0.345f;
    sc_cubes_size[11][2] = 0.640f;

    // cafe table bottom - depends on 7
    sc_cubes_pos[12][0] = 0.000f;
    sc_cubes_size[12][0] = 0.440f;
    sc_cubes_pos[12][1] = 0.000f;
    sc_cubes_size[12][1] = 0.440f;
    sc_cubes_pos[12][2] = -0.645f;
    sc_cubes_size[12][2] = 0.050f;

    /*
      // absolute vlaues
      // table bottom 1 - depends on 4
       sc_cubes_pos[ 8][0]=-2.000;  sc_cubes_size[ 8][0]= 0.580;
       sc_cubes_pos[ 8][1]= 0.000;  sc_cubes_size[ 8][1]= 0.580;
       sc_cubes_pos[ 8][2]= 0.410;  sc_cubes_size[ 8][2]= 0.820;

      //table bottom 2 - depends on 5
       sc_cubes_pos[ 9][0]=-3.000;  sc_cubes_size[ 9][0]= 0.580;
       sc_cubes_pos[ 9][1]= 0.000;  sc_cubes_size[ 9][1]= 0.580;
       sc_cubes_pos[ 9][2]= 0.410;  sc_cubes_size[ 9][2]= 0.820;

   // table bottom 3 - depends on 6
   sc_cubes_pos[10][0]= 2.000;  sc_cubes_size[10][0]= 0.580;
   sc_cubes_pos[10][1]= 0.000;  sc_cubes_size[10][1]= 0.580;
   sc_cubes_pos[10][2]= 0.410;  sc_cubes_size[10][2]= 0.820;

   // cafe table middle - depends on 7
   sc_cubes_pos[11][0]= 3.000;  sc_cubes_size[11][0]= 0.180;
   sc_cubes_pos[11][1]= 0.000;  sc_cubes_size[11][1]= 0.180;
   sc_cubes_pos[11][2]= 0.325;  sc_cubes_size[11][2]= 0.640;

   // cafe table bottom - depends on 7
   sc_cubes_pos[12][0]= 3.000;  sc_cubes_size[12][0]= 0.440;
   sc_cubes_pos[12][1]= 0.000;  sc_cubes_size[12][1]= 0.440;
   sc_cubes_pos[12][2]= 0.025;  sc_cubes_size[12][2]= 0.050;
   */
    for (i = 0; i < 13; i++)
    {
        p_cubes_pos[i]->setValue(sc_cubes_pos[i][0], sc_cubes_pos[i][1], sc_cubes_pos[i][2]);
        p_cubes_size[i]->setValue(sc_cubes_size[i][0], sc_cubes_size[i][1], sc_cubes_size[i][2]);
    }

    return 0;
}

MODULE_MAIN(UnderDev, sc2004booth)
