/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>

#include <christmas06.h>
#include <include/model.h>
#include <api/coFeedback.h>

#include <config/CoviseConfig.h>

#define RAD(x) ((x)*M_PI / 180.0)
#define GRAD(x) ((x)*180.0 / M_PI)

#define NPERSONS 9
#define OBJSPERPERSON 4
#define NOBJECTS NPERSONS *OBJSPERPERSON

christmas06::christmas06(int argc, char *argv[])
    : coModule(argc, argv, "christmas06")
{
    geo = NULL;

    fprintf(stderr, "christmas06::christmas06()\n");

#ifdef USE_STARTFILE
    // start file param
    fprintf(stderr, "christmas06::christmas06() Init of StartFile\n");
    startFile = addFileBrowserParam("startFile", "Start file");
    startFile->setValue(coCoviseConfig::getEntry("value", "Module.ChristMas06.DataPath", getenv("HOME")), "*.txt");
#endif

    // We build the User-Menue ...
    christmas06::CreateUserMenu();

    // the output ports
    fprintf(stderr, "christmas06::christmas06() SetOutPort\n");
    grid = addOutputPort("grid", "UnstructuredGrid", "Computation Grid");
    surface = addOutputPort("surface", "Polygons", "Surface Polygons");
    bcin = addOutputPort("bcin", "Polygons", "Cells at entry");
    bcout = addOutputPort("bcout", "Polygons", "Cells at exit");
    bcwall = addOutputPort("bcwall", "Polygons", "Cells at walls");
    boco = addOutputPort("boco", "USR_FenflossBoco", "Boundary Conditions");

    inpoints = addOutputPort("inbc nodes", "Points", "inbc nodes");
    feedback_info = addOutputPort("feedback info", "Points", "Feedback Info");

    model = NULL;
    ag = NULL;

    isInitialized = 0;
}

void christmas06::postInst()
{
#ifdef USE_STARTFILE
    startFile->show();
#endif
    p_makeGrid->show();
    p_lockmakeGrid->show();
    p_openDoor->show();
    p_gridSpacing->show();
    p_nobjects->show();
    p_model_size->show();
    p_v_in->show();
    p_zScale->show();
    p_geofile->show();
    p_rbfile->show();
}

void christmas06::param(const char *portname)
{

    //	char buf[255];

    fprintf(stderr, "christmas06::param = %s\n", portname);

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
            fprintf(stderr, "christmas06::param = ReadGeometry(%s) ...", buf);
            model = ReadStartfile(buf);
            if (model == NULL)
            {
                Covise::sendError("ReadStartfile error");
                return;
            }
        }
    }
#endif

    //selfExec();
}

void christmas06::quit()
{
    // :-)
}

int christmas06::compute(const char *)
{

    coDoPolygons *poly;
    int i;

    char name[256];

    fprintf(stderr, "christmas06::compute(const char *) entering... \n");

#ifdef USE_STARTFILE
    char buf[256];
    Covise::getname(buf, startFile->getValue());
    fprintf(stderr, "christmas06::param = ReadGeometry(%s) ...", buf);
#endif
    if (model != NULL)
        FreeModel(model);
    model = AllocModel();

#ifdef USE_STARTFILE
    ReadStartfile(buf, model);
#else
    GetParamsFromControlPanel(model);
#endif

    if (!model)
    {
        sendError("Please select a parameter file first!!");
        return FAIL;
    }

    //// Cover plugin information object is created here
    createFeedbackObjects();

    /////////////////////////////
    // create geometry for COVISE
    if ((ci = CreateGeometry4Covise(model)))
    {
        fprintf(stderr, "christmas06::compute(const char *): Geometry created\n");

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

        if (model == NULL)
        {
            sendError("Cannot create grid because model is NULL!");
            return (1);
        }
        model->spacing = p_gridSpacing->getValue();
        if ((ag = CreateChristGrid(model)) == NULL)
        {
            fprintf(stderr, "Error in CreateSC2004Grid!\n");
            return -1;
        }
        ag->bc_inval = 100;
        ag->bc_outval = 110;

        fprintf(stderr, "christmas06: Grid created\n");

        coDoUnstructuredGrid *unsGrd = new coDoUnstructuredGrid(grid->getObjName(), // name of USG object
                                                                ag->e->nume, // number of elements
                                                                8 * ag->e->nume, // number of connectivities
                                                                ag->p->nump, // number of coordinates
                                                                1); // does type list exist?

        int *elem, *conn, *type;
        float *xc, *yc, *zc;
        unsGrd->getAddresses(&elem, &conn, &xc, &yc, &zc);
        unsGrd->getTypeList(&type);

        printf("nelem  = %d\n", ag->e->nume);
        printf("nconn  = %d\n", 8 * ag->e->nume);
        printf("nccord = %d\n", ag->p->nump);

        int **GgridConn = ag->e->e;
        for (i = 0; i < ag->e->nume; i++)
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
        memcpy(xc, ag->p->x, ag->p->nump * sizeof(float));
        memcpy(yc, ag->p->y, ag->p->nump * sizeof(float));
        memcpy(zc, ag->p->z, ag->p->nump * sizeof(float));

        // set out port
        grid->setCurrentObject(unsGrd);

        // boundary condition lists
        // 1. Cells at walls
        poly = new coDoPolygons(bcwall->getObjName(),
                                ag->p->nump,
                                ag->p->x, ag->p->y, ag->p->z,
                                ag->bcwall->num, ag->bcwall->list,
                                ag->bcwallpol->num, ag->bcwallpol->list);
        //poly->addAttribute("MATERIAL","metal metal.30");
        poly->addAttribute("vertexOrder", "1");
        bcwall->setCurrentObject(poly);

        // 2. Cells at outlet
        poly = new coDoPolygons(bcout->getObjName(),
                                ag->p->nump,
                                ag->p->x, ag->p->y, ag->p->z,
                                ag->bcout->num, ag->bcout->list,
                                ag->bcoutpol->num, ag->bcoutpol->list);
        //poly->addAttribute("MATERIAL","metal metal.30");
        poly->addAttribute("vertexOrder", "1");
        bcout->setCurrentObject(poly);

        // 3. Cells at inlet
        poly = new coDoPolygons(bcin->getObjName(),
                                ag->p->nump,
                                ag->p->x, ag->p->y, ag->p->z,
                                ag->bcin->num, ag->bcin->list,
                                ag->bcinpol->num, ag->bcinpol->list);
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
        data[0] = AG_COL_NODE; // (=2)
        data[1] = AG_COL_ELEM; // (=2)
        data[2] = AG_COL_DIRICLET; // (=2)
        data[3] = AG_COL_WALL; // (=6)
        data[4] = AG_COL_BALANCE; // (=6)
        data[5] = AG_COL_PRESS; // (=6)
        partObj[0] = colInfo;

        //   1. type of node
        sprintf(name, "%s_nodeinfo", basename);
        size[0] = AG_COL_NODE;
        size[1] = ag->p->nump;
        coDoIntArr *nodeInfo = new coDoIntArr(name, 2, size);
        data = nodeInfo->getAddress();
        for (i = 0; i < ag->p->nump; i++)
        {
            *data++ = i + 1; // may be, that we later do it really correct
            *data++ = 0; // same comment ;-)
        }
        partObj[1] = nodeInfo;

        //   2. type of element
        sprintf(name, "%s_eleminfo", basename);
        size[0] = 2;
        size[1] = ag->e->nume * AG_COL_ELEM; // uwe: hier wird 4*nume allociert aber nur 2*nume mit Werten gefüllt
        coDoIntArr *elemInfo = new coDoIntArr(name, 2, size);
        data = elemInfo->getAddress();
        for (i = 0; i < ag->e->nume; i++)
        {
            *data++ = i + 1; // may be, that we later do it really corect
            *data++ = 0; // same comment ;-)
        }
        partObj[2] = elemInfo;

        //   3. list of nodes with bc (a node may appear more than one time)
        //      and its types
        sprintf(name, "%s_diricletNodes", basename);
        int num_diriclet = ag->bcin_nodes->num;

        size[0] = AG_COL_DIRICLET;
        size[1] = 5 * (num_diriclet);
        coDoIntArr *diricletNodes = new coDoIntArr(name, 2, size);
        data = diricletNodes->getAddress();

        //   4. corresponding value to 3.
        sprintf(name, "%s_diricletValue", basename);
        coDoFloat *diricletValues = new coDoFloat(name, 5 * num_diriclet);
        diricletValues->getAddress(&bPtr);

        for (i = 0; i < ag->bcin_nodes->num; i++)
        {
            *data++ = ag->bcin_nodes->list[i] + 1; // node-number
            *data++ = 1; // type of node
            *bPtr++ = ag->bcin_velos->list[5 * i + 0]; // u

            *data++ = ag->bcin_nodes->list[i] + 1; // node-number
            *data++ = 2; // type of node
            *bPtr++ = ag->bcin_velos->list[5 * i + 1]; // v

            *data++ = ag->bcin_nodes->list[i] + 1; // node-number
            *data++ = 3; // type of node
            *bPtr++ = ag->bcin_velos->list[5 * i + 2]; // w

            *data++ = ag->bcin_nodes->list[i] + 1; // node-number
            *data++ = 4; // type of node
            *bPtr++ = ag->bcin_velos->list[5 * i + 4]; // epsilon

            *data++ = ag->bcin_nodes->list[i] + 1; // node-number
            *data++ = 5; // type of node
            *bPtr++ = ag->bcin_velos->list[5 * i + 3]; // k

            //*data++ = ag->bcin_nodes->list[i]+1;     // node-number
            //*data++ = 6;                             // type of node
            //*bPtr++ = 0.0;                           // temperature = 0.
        }

        partObj[3] = diricletNodes;
        partObj[4] = diricletValues;

        //   5. wall
        sprintf(name, "%s_wallValue", basename);
        coDoFloat *wallValues = new coDoFloat(name, ag->bcwallvol->num);
        wallValues->getAddress(&bPtr);
        size[0] = AG_COL_WALL;
        size[1] = ag->bcwallpol->num;
        sprintf(name, "%s_wall", basename);
        coDoIntArr *faces = new coDoIntArr(name, 2, size);
        data = faces->getAddress();
        for (i = 0; i < ag->bcwallpol->num; i++) // Achtung bcwall->pol->num != bcwall->vol->num
        {
            *data++ = ag->bcwall->list[4 * i + 0] + 1;
            *data++ = ag->bcwall->list[4 * i + 1] + 1;
            *data++ = ag->bcwall->list[4 * i + 2] + 1;
            *data++ = ag->bcwall->list[4 * i + 3] + 1;
            *data++ = ag->bcwallvol->list[i] + 1;
            *data++ = 55; // wall: moving | standing. here: always standing!
            *data++ = 0;
        }
        partObj[5] = faces;

        //   6. balance
        sprintf(name, "%s_balance", basename);
        size[0] = AG_COL_BALANCE;
        size[1] = ag->bcinvol->num + ag->bcoutvol->num;

        coDoIntArr *balance = new coDoIntArr(name, 2, size);
        data = balance->getAddress();
        for (i = 0; i < ag->bcinvol->num; i++)
        {
            *data++ = ag->bcin->list[4 * i + 0] + 1;
            *data++ = ag->bcin->list[4 * i + 1] + 1;
            *data++ = ag->bcin->list[4 * i + 2] + 1;
            *data++ = ag->bcin->list[4 * i + 3] + 1;
            *data++ = ag->bcinvol->list[i] + 1;
            *data++ = ag->bc_inval;
            *data++ = 0;
        }
        for (i = 0; i < ag->bcoutvol->num; i++)
        {
            *data++ = ag->bcout->list[4 * i + 0] + 1;
            *data++ = ag->bcout->list[4 * i + 1] + 1;
            *data++ = ag->bcout->list[4 * i + 2] + 1;
            *data++ = ag->bcout->list[4 * i + 3] + 1;
            *data++ = ag->bcoutvol->list[i] + 1;
            *data++ = ag->bc_outval;
            *data++ = 0;
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
        partObj[8] = NULL;

        coDoSet *set = new coDoSet((char *)basename, (coDistributedObject **)partObj);

        boco->setCurrentObject(set);

        float *xinp = new float[ag->bcin_nodes->num];
        float *yinp = new float[ag->bcin_nodes->num];
        float *zinp = new float[ag->bcin_nodes->num];
        for (i = 0; i < ag->bcin_nodes->num; i++)
        {
            xinp[i] = ag->p->x[ag->bcin_nodes->list[i]];
            yinp[i] = ag->p->y[ag->bcin_nodes->list[i]];
            zinp[i] = ag->p->z[ag->bcin_nodes->list[i]];
        }
        coDoPoints *in_points;
        in_points = new coDoPoints(inpoints->getObjName(), ag->bcin_nodes->num, xinp, yinp, zinp);
        inpoints->setCurrentObject(in_points);
        /*
      float *xairp = new float[ag->bcair_nodes->num];
      float *yairp = new float[ag->bcair_nodes->num];
      float *zairp = new float[ag->bcair_nodes->num];
      for (i=0;i<ag->bcair_nodes->num;i++)
      {
         xairp[i]=ag->p->x[ag->bcair_nodes->list[i]];
         yairp[i]=ag->p->y[ag->bcair_nodes->list[i]];
         zairp[i]=ag->p->z[ag->bcair_nodes->list[i]];
      }

      coDoPoints *air_points;
      air_points = new coDoPoints(airpoints->getObjName(), ag->bcair_nodes->num, xairp, yairp, zairp);
      airpoints->setCurrentObject(air_points);

      float *xvenp = new float[ag->bcven_nodes->num];
      float *yvenp = new float[ag->bcven_nodes->num];
      float *zvenp = new float[ag->bcven_nodes->num];
      for (i=0;i<ag->bcven_nodes->num;i++)
      {
         xvenp[i]=ag->p->x[ag->bcven_nodes->list[i]];
         yvenp[i]=ag->p->y[ag->bcven_nodes->list[i]];
         zvenp[i]=ag->p->z[ag->bcven_nodes->list[i]];
      }
      coDoPoints *ven_points;
      ven_points = new coDoPoints(venpoints->getObjName(), ag->bcven_nodes->num, xvenp, yvenp, zvenp);
      venpoints->setCurrentObject(ven_points);
*/
        const char *geofile = p_geofile->getValue();
        const char *rbfile = p_rbfile->getValue();

        if (p_createGeoRbFile->getValue())
        {
            CreateGeoRbFile(ag, geofile, rbfile);
        }
    }

    ///////////////////////// Free everything ////////////////////////////////

    return SUCCESS;
}

void christmas06::CreateUserMenu(void)
{
    fprintf(stderr, "Entering CreateUserMenu()\n");

    char *tmp;
    int i;
    const char *buf;
    char path[512];

    p_makeGrid = addBooleanParam("make_grid", "make grid?");
    p_makeGrid->setValue(1);

    p_lockmakeGrid = addBooleanParam("lock_make_grid_button", "lock make grid button?");
    p_lockmakeGrid->setValue(1);

    p_createGeoRbFile = addBooleanParam("create_geo/rb_file", "create geo/rb file?");
    p_createGeoRbFile->setValue(0);

    p_openDoor = addBooleanParam("open_cube_door", "open cube door?");
    p_openDoor->setValue(0);

    p_gridSpacing = addFloatParam("grid_spacing", "grid spacing");
    p_gridSpacing->setValue(0.55);

    p_model_size = addFloatVectorParam("model_size", "model size");
    p_model_size->setValue(60.0, 100.0, 45.0);

    p_nobjects = addInt32Param("n_objects", "make grid?");
    p_nobjects->setValue(NOBJECTS);

    p_v_in = addFloatVectorParam("v", "v inlet");
    p_v_in->setValue(2., 0., 0.);

    p_zScale = addFloatParam("zScale", "scale Velo according to z value");
    p_zScale->setValue(0.0);

    sprintf(path, "%s/geofile.geo", coCoviseConfig::getEntry("value", "Module.ChristMas06.GeorbPath", "c:/temp").c_str());
    p_geofile = addStringParam("geofile path", "geofile path");
    p_geofile->setValue(path);

    sprintf(path, "%s/rbfile.geo", coCoviseConfig::getEntry("value", "Module.ChristMas06.GeorbPath", "c:/temp").c_str());
    p_rbfile = addStringParam("rbfile path", "rbfile path");
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
    setGeoParamsStandardForChristMas06();
#endif
}

char *christmas06::IndexedParameterName(const char *name, int index)
{
    char buf[255];
    sprintf(buf, "%s_%d", name, index + 1);
    return strdup(buf);
}

void christmas06::createFeedbackObjects()
{

    cerr << "createFeedbackObjects()" << endl;

    coDoPoints *feedinfo;
    feedinfo = new coDoPoints(feedback_info->getObjName(), 0);
    ////////////////////////////////////////////////////////////////////
    // add the current parameter values as feedback parameters
    int i = 0;

    coFeedback feedback("TangiblePosition");
    feedback.addString("Christmas06");

    // the others only if used
    for (i = 0; i < p_nobjects->getValue(); i++)
    {
        feedback.addPara(p_cubes_pos[i]);
        feedback.addPara(p_cubes_size[i]);
    }
    feedback.apply(feedinfo);
    feedback_info->setCurrentObject(feedinfo);
}

int christmas06::GetParamsFromControlPanel(struct christ_model *model)
{
    int i;

    fprintf(stderr, "entering GetParamsFromControlPanel\n");

    model->nobjects = p_nobjects->getValue();

    p_v_in->getValue(model->bcin_velo[0], model->bcin_velo[1], model->bcin_velo[2]);

    // alloc cubes
    if ((model->cubes = (struct cubus **)calloc(model->nobjects, sizeof(struct cubus *))) == NULL)
    {
        fprintf(stderr, "Not enough space!\n");
        return -1;
    }
    for (i = 0; i < model->nobjects; i++)
    {
        if ((model->cubes[i] = (struct cubus *)calloc(1, sizeof(struct cubus))) == NULL)
        {
            fprintf(stderr, "Not enough space!\n");
            return -1;
        }
    }

    // total size
    model->size[0] = p_model_size->getValue(0);
    model->size[1] = p_model_size->getValue(1);
    model->size[2] = p_model_size->getValue(2);
    model->zScale = p_zScale->getValue() / model->size[2];

    for (i = 0; i < model->nobjects; i++)
    {
        model->cubes[i]->size[0] = p_cubes_size[i]->getValue(0);
        model->cubes[i]->size[1] = p_cubes_size[i]->getValue(1);
        model->cubes[i]->size[2] = p_cubes_size[i]->getValue(2);

        model->cubes[i]->pos[0] = p_cubes_pos[i]->getValue(0);
        model->cubes[i]->pos[1] = p_cubes_pos[i]->getValue(1);
        model->cubes[i]->pos[2] = p_cubes_pos[i]->getValue(2);
        /*
      // relative position values
      if (i==7)
      {
         model->cubes[i]->pos[0] += p_cubes_pos[0]->getValue(0);
         model->cubes[i]->pos[1] += p_cubes_pos[0]->getValue(1);
         model->cubes[i]->pos[2] += p_cubes_pos[0]->getValue(2);
      }
      if (i==8)
      {
         model->cubes[i]->pos[0] += p_cubes_pos[1]->getValue(0);
         model->cubes[i]->pos[1] += p_cubes_pos[1]->getValue(1);
         model->cubes[i]->pos[2] += p_cubes_pos[1]->getValue(2);
      }
      if (i==9)
      {
         model->cubes[i]->pos[0] += p_cubes_pos[2]->getValue(0);
         model->cubes[i]->pos[1] += p_cubes_pos[2]->getValue(1);
         model->cubes[i]->pos[2] += p_cubes_pos[2]->getValue(2);
      }
      if (i==10)
      {
         model->cubes[i]->pos[0] += p_cubes_pos[1]->getValue(0);
         model->cubes[i]->pos[1] += p_cubes_pos[1]->getValue(1);
         model->cubes[i]->pos[2] += p_cubes_pos[1]->getValue(2);
      }
      if (i==11)
      {
         model->cubes[i]->pos[0] += p_cubes_pos[1]->getValue(0);
         model->cubes[i]->pos[1] += p_cubes_pos[1]->getValue(1);
         model->cubes[i]->pos[2] += p_cubes_pos[1]->getValue(2);
      }
      if (i==12)
      {
         model->cubes[i]->pos[0] += p_cubes_pos[3]->getValue(0);
         model->cubes[i]->pos[1] += p_cubes_pos[3]->getValue(1);
         model->cubes[i]->pos[2] += p_cubes_pos[3]->getValue(2);
      }
*/
    }

    return (0);
}

int christmas06::setGeoParamsStandardForChristMas06()
{
    int i;

    float christ_cubes_pos[NOBJECTS][3];
    float christ_cubes_size[NOBJECTS][3];

    fprintf(stderr, "entering setGeoParamsStandardForChristMas06 ...\n");

    float base = 20.; // base unit size

    // define standards
    float def_width = 0.3;
    float def_depth = 0.1;
    float def_leg_length = 0.6;
    float def_body_length = 0.7;
    float def_neck_length = 0.2;
    float def_neck_width = 0.3 * def_width;
    //float def_head_length = 0.15;
    //float def_head_width = 0.37*def_width;

    // factors for individuals to take care of individual proportions
    float personsize[NPERSONS];
    float leg_length[NPERSONS];
    float body_length[NPERSONS];
    float neck_length[NPERSONS];
    float neck_width[NPERSONS];
    //float head_length[NPERSONS];
    //float head_width[NPERSONS];
    float width[NPERSONS];
    float depth[NPERSONS];

    float pos[NPERSONS][3];

    int person;

    // default distance between persons
    //float distance=0.1*base;

    // #0: Andreas
    person = 0;
    personsize[person] = 1.90;
    leg_length[person] = def_leg_length * 1.;
    body_length[person] = def_body_length * 1.;
    neck_length[person] = def_neck_length * 1.;
    //head_length[person]=def_head_length*1.;
    width[person] = personsize[person] * def_width * base * 0.7;
    neck_width[person] = personsize[person] * def_neck_width * base * 1.;
    //head_width[person]=personsize[person]*def_head_width*base*1.;
    depth[person] = personsize[person] * def_depth * base * 1.;
    pos[person][0] = 6.;
    pos[person][1] = -47;
    pos[person][2] = 0.;

    // #1: Uwe W.
    person = 1;
    personsize[person] = 1.85;
    leg_length[person] = def_leg_length * 0.98;
    body_length[person] = def_body_length * 0.95;
    neck_length[person] = def_neck_length * 1.02;
    //head_length[person]=def_head_length*1.;
    width[person] = personsize[person] * def_width * base * 1.;
    neck_width[person] = personsize[person] * def_neck_width * base * 1.;
    //head_width[person]=personsize[person]*def_head_width*base*1.;
    depth[person] = personsize[person] * def_depth * base * 1.;
    pos[person][0] = -9.;
    pos[person][1] = -28.; //pos[person-1][1]+width[person-1]+width[person]/2.+distance;
    pos[person][2] = 0.;

    // #2: Blasius
    person = 2;
    personsize[person] = 1.74;
    leg_length[person] = def_leg_length * 1.02;
    body_length[person] = def_body_length * 0.98;
    neck_length[person] = def_neck_length * 0.96;
    //head_length[person]=def_head_length*1.;
    width[person] = personsize[person] * def_width * base * 1.;
    neck_width[person] = personsize[person] * def_neck_width * base * 1.;
    //head_width[person]=personsize[person]*def_head_width*base*1.;
    depth[person] = personsize[person] * def_depth * base * 1.;
    pos[person][0] = 8.;
    pos[person][1] = -15.5; //pos[person-1][1]+width[person-1]+width[person]/2.+distance;
    pos[person][2] = -18.;

    // #3: Florian
    person = 3;
    personsize[person] = 1.84;
    leg_length[person] = def_leg_length * 0.96;
    body_length[person] = def_body_length * 1.;
    neck_length[person] = def_neck_length * 1.;
    //head_length[person]=def_head_length*1.;
    width[person] = personsize[person] * def_width * base * 1.1;
    neck_width[person] = personsize[person] * def_neck_width * base * 1.;
    //head_width[person]=personsize[person]*def_head_width*base*1.;
    depth[person] = personsize[person] * def_depth * base * 1.;
    pos[person][0] = -10.;
    pos[person][1] = -7.; //pos[person-1][1]+width[person-1]+width[person]/2.+distance;
    pos[person][2] = 0.;

    // #4: Uwe Z.
    person = 4;
    personsize[person] = 1.72;
    leg_length[person] = def_leg_length * 1.;
    body_length[person] = def_body_length * 1.;
    neck_length[person] = def_neck_length * 0.94;
    //head_length[person]=def_head_length*1.;
    width[person] = personsize[person] * def_width * base * 1.;
    neck_width[person] = personsize[person] * def_neck_width * base * 1.;
    //head_width[person]=personsize[person]*def_head_width*base*1.;
    depth[person] = personsize[person] * def_depth * base * 1.;
    pos[person][0] = 22.;
    pos[person][1] = 12.; //pos[person-1][1]+width[person-1]+width[person]/2.+distance;
    pos[person][2] = -15.;

    // #5: Braitmaik
    person = 5;
    personsize[person] = 1.72;
    leg_length[person] = def_leg_length * 0.98;
    body_length[person] = def_body_length * 1.;
    neck_length[person] = def_neck_length * 0.96;
    //head_length[person]=def_head_length*1.;
    width[person] = personsize[person] * def_width * base * 1.;
    neck_width[person] = personsize[person] * def_neck_width * base * 1.;
    //head_width[person]=personsize[person]*def_head_width*base*1.;
    depth[person] = personsize[person] * def_depth * base * 1.;
    pos[person][0] = -10.;
    pos[person][1] = 15.5; //pos[person-1][1]+width[person-1]+width[person]/2.+distance;
    pos[person][2] = 0.;

    // #6: Mario
    person = 6;
    personsize[person] = 1.60;
    leg_length[person] = def_leg_length * 1.0;
    body_length[person] = def_body_length * 1.04;
    neck_length[person] = def_neck_length * 0.96;
    //head_length[person]=def_head_length*1.;
    width[person] = personsize[person] * def_width * base * 1.;
    neck_width[person] = personsize[person] * def_neck_width * base * 1.;
    //head_width[person]=personsize[person]*def_head_width*base*1.;
    depth[person] = personsize[person] * def_depth * base * 1.;
    pos[person][0] = 2.;
    pos[person][1] = 33.5; //pos[person-1][1]+width[person-1]+width[person]/2.+distance;
    pos[person][2] = 0.;

    // #7: Jutta
    person = 7;
    personsize[person] = 1.60;
    leg_length[person] = def_leg_length * 0.97;
    body_length[person] = def_body_length * 1.0;
    neck_length[person] = def_neck_length * 0.96;
    //head_length[person]=*def_head_length*1.;
    width[person] = personsize[person] * def_width * base * 1.;
    neck_width[person] = personsize[person] * def_neck_width * base * 1.;
    //head_width[person]=personsize[person]*def_head_width*base*1.;
    depth[person] = personsize[person] * def_depth * base * 1.;
    pos[person][0] = -9.;
    pos[person][1] = 51.5; //pos[person-1][1]+width[person-1]+width[person]/2.+distance;
    pos[person][2] = 0.;

    // #8: Martin
    person = 8;
    personsize[person] = 1.84;
    leg_length[person] = def_leg_length * 0.96;
    body_length[person] = def_body_length * 1.;
    neck_length[person] = def_neck_length * 1.0;
    //head_length[person]*def_head_length*1.;
    width[person] = personsize[person] * def_width * base * 1.2;
    neck_width[person] = personsize[person] * def_neck_width * base * 1.;
    //head_width[person]=personsize[person]*def_head_width*base*1.;
    depth[person] = personsize[person] * def_depth * base * 1.;
    pos[person][0] = 22.;
    pos[person][1] = 49.; //pos[person-1][1]+width[person-1]+width[person]/2.+distance;
    pos[person][2] = -16.;

    // norm lengths to achieve correct personsize
    float totallength;
    for (i = 0; i < NPERSONS; i++)
    {
        totallength = 0.;
        totallength += leg_length[i];
        totallength += body_length[i];
        totallength += neck_length[i];
        cerr << "totallength[" << i << "]=" << totallength << endl;
        leg_length[i] = leg_length[i] / totallength * personsize[i] * base;
        body_length[i] = body_length[i] / totallength * personsize[i] * base;
        neck_length[i] = neck_length[i] / totallength * personsize[i] * base;
        pos[i][0] *= 0.7;
        pos[i][1] *= 0.7;
        pos[i][2] *= 0.7;
    }

    person = 0;
    i = person * OBJSPERPERSON;
    // left leg
    christ_cubes_pos[i + 0][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 0][1] = pos[person][1] + width[person] / 2.; // changed from default!
    christ_cubes_pos[i + 0][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 0][0] = depth[person];
    christ_cubes_size[i + 0][1] = width[person] / 2.;
    christ_cubes_size[i + 0][2] = leg_length[person];

    // right leg
    christ_cubes_pos[i + 1][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 1][1] = pos[person][1] + width[person] / 2.; // changed from default!
    christ_cubes_pos[i + 1][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 1][0] = depth[person];
    christ_cubes_size[i + 1][1] = width[person] / 2.;
    christ_cubes_size[i + 1][2] = leg_length[person];

    //body
    christ_cubes_pos[i + 2][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 2][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 2][2] = pos[person][2] + leg_length[person] + body_length[person] / 2.;
    christ_cubes_size[i + 2][0] = depth[person];
    christ_cubes_size[i + 2][1] = width[person];
    christ_cubes_size[i + 2][2] = body_length[person];

    //neck
    christ_cubes_pos[i + 3][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 3][1] = pos[person][1] + width[person] * 3. / 4.; // changed from default!
    christ_cubes_pos[i + 3][2] = pos[person][2] + leg_length[person] + body_length[person] + neck_length[person] / 2.;
    christ_cubes_size[i + 3][0] = depth[person];
    christ_cubes_size[i + 3][1] = neck_width[person];
    christ_cubes_size[i + 3][2] = neck_length[person];
    /*   
   // head
   christ_cubes_pos [i+4][0]=pos[person][0]+depth[person]/2.;							 
   christ_cubes_pos [i+4][1]=pos[person][1]+width[person]/2.;							 
   christ_cubes_pos [i+4][2]=pos[person][2]+leg_length[person]+body_length[person]+neck_length[person]+head_length[person]/2.;   
   christ_cubes_size[i+4][0]=depth[person];
   christ_cubes_size[i+4][1]=head_width[person];
   christ_cubes_size[i+4][2]=head_length[person];
*/
    person = 1;
    i = person * OBJSPERPERSON;
    // left leg
    christ_cubes_pos[i + 0][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 0][1] = pos[person][1] + width[person] / 6.;
    christ_cubes_pos[i + 0][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 0][0] = depth[person];
    christ_cubes_size[i + 0][1] = width[person] / 3.;
    christ_cubes_size[i + 0][2] = leg_length[person];

    // right leg
    christ_cubes_pos[i + 1][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 1][1] = pos[person][1] + 5 * width[person] / 6.;
    christ_cubes_pos[i + 1][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 1][0] = depth[person];
    christ_cubes_size[i + 1][1] = width[person] / 3.;
    christ_cubes_size[i + 1][2] = leg_length[person];

    // body
    christ_cubes_pos[i + 2][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 2][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 2][2] = pos[person][2] + leg_length[person] + body_length[person] / 2.;
    christ_cubes_size[i + 2][0] = depth[person];
    christ_cubes_size[i + 2][1] = width[person];
    christ_cubes_size[i + 2][2] = body_length[person];

    // neck
    christ_cubes_pos[i + 3][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 3][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 3][2] = pos[person][2] + leg_length[person] + body_length[person] + neck_length[person] / 2.;
    christ_cubes_size[i + 3][0] = depth[person];
    christ_cubes_size[i + 3][1] = neck_width[person];
    christ_cubes_size[i + 3][2] = neck_length[person];
    /*
   // head
   christ_cubes_pos [i+4][0]=pos[person][0]+depth[person]/2.;							 
   christ_cubes_pos [i+4][1]=pos[person][1]+width[person]/2.;							 
   christ_cubes_pos [i+4][2]=pos[person][2]+leg_length[person]+body_length[person]+neck_length[person]+head_length[person]/2.;   
   christ_cubes_size[i+4][0]=depth[person];
   christ_cubes_size[i+4][1]=head_width[person];
   christ_cubes_size[i+4][2]=head_length[person];
*/
    person = 2;
    i = person * OBJSPERPERSON;
    // left leg
    christ_cubes_pos[i + 0][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 0][1] = pos[person][1] + width[person] / 6.;
    christ_cubes_pos[i + 0][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 0][0] = depth[person];
    christ_cubes_size[i + 0][1] = width[person] / 3.;
    christ_cubes_size[i + 0][2] = leg_length[person];

    // right leg
    christ_cubes_pos[i + 1][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 1][1] = pos[person][1] + 5 * width[person] / 6.;
    christ_cubes_pos[i + 1][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 1][0] = depth[person];
    christ_cubes_size[i + 1][1] = width[person] / 3.;
    christ_cubes_size[i + 1][2] = leg_length[person];

    // body
    christ_cubes_pos[i + 2][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 2][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 2][2] = pos[person][2] + leg_length[person] + body_length[person] / 2.;
    christ_cubes_size[i + 2][0] = depth[person];
    christ_cubes_size[i + 2][1] = width[person];
    christ_cubes_size[i + 2][2] = body_length[person];

    // neck
    christ_cubes_pos[i + 3][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 3][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 3][2] = pos[person][2] + leg_length[person] + body_length[person] + neck_length[person] / 2.;
    christ_cubes_size[i + 3][0] = depth[person];
    christ_cubes_size[i + 3][1] = neck_width[person];
    christ_cubes_size[i + 3][2] = neck_length[person];
    /*
   // head
   christ_cubes_pos [i+4][0]=pos[person][0]+depth[person]/2.;							 
   christ_cubes_pos [i+4][1]=pos[person][1]+width[person]/2.;							 
   christ_cubes_pos [i+4][2]=pos[person][2]+leg_length[person]+body_length[person]+neck_length[person]+head_length[person]/2.;   
   christ_cubes_size[i+4][0]=depth[person];
   christ_cubes_size[i+4][1]=head_width[person];
   christ_cubes_size[i+4][2]=head_length[person];
*/
    person = 3;
    i = person * OBJSPERPERSON;
    // left leg
    christ_cubes_pos[i + 0][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 0][1] = pos[person][1] + width[person] / 6.;
    christ_cubes_pos[i + 0][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 0][0] = depth[person];
    christ_cubes_size[i + 0][1] = width[person] / 3.;
    christ_cubes_size[i + 0][2] = leg_length[person];

    // right leg
    christ_cubes_pos[i + 1][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 1][1] = pos[person][1] + 5 * width[person] / 6.;
    christ_cubes_pos[i + 1][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 1][0] = depth[person];
    christ_cubes_size[i + 1][1] = width[person] / 3.;
    christ_cubes_size[i + 1][2] = leg_length[person];

    // body
    christ_cubes_pos[i + 2][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 2][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 2][2] = pos[person][2] + leg_length[person] + body_length[person] / 2.;
    christ_cubes_size[i + 2][0] = depth[person];
    christ_cubes_size[i + 2][1] = width[person];
    christ_cubes_size[i + 2][2] = body_length[person];

    // neck
    christ_cubes_pos[i + 3][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 3][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 3][2] = pos[person][2] + leg_length[person] + body_length[person] + neck_length[person] / 2.;
    christ_cubes_size[i + 3][0] = depth[person];
    christ_cubes_size[i + 3][1] = neck_width[person];
    christ_cubes_size[i + 3][2] = neck_length[person];
    /*
   // head
   christ_cubes_pos [i+4][0]=pos[person][0]+depth[person]/2.;							 
   christ_cubes_pos [i+4][1]=pos[person][1]+width[person]/2.;							 
   christ_cubes_pos [i+4][2]=pos[person][2]+leg_length[person]+body_length[person]+neck_length[person]+head_length[person]/2.;   
   christ_cubes_size[i+4][0]=depth[person];
   christ_cubes_size[i+4][1]=head_width[person];
   christ_cubes_size[i+4][2]=head_length[person];
*/
    person = 4;
    i = person * OBJSPERPERSON;
    // left leg
    christ_cubes_pos[i + 0][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 0][1] = pos[person][1] + width[person] / 6.;
    christ_cubes_pos[i + 0][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 0][0] = depth[person];
    christ_cubes_size[i + 0][1] = width[person] / 3.;
    christ_cubes_size[i + 0][2] = leg_length[person];

    // right leg
    christ_cubes_pos[i + 1][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 1][1] = pos[person][1] + 5 * width[person] / 6.;
    christ_cubes_pos[i + 1][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 1][0] = depth[person];
    christ_cubes_size[i + 1][1] = width[person] / 3.;
    christ_cubes_size[i + 1][2] = leg_length[person];

    // body
    christ_cubes_pos[i + 2][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 2][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 2][2] = pos[person][2] + leg_length[person] + body_length[person] / 2.;
    christ_cubes_size[i + 2][0] = depth[person];
    christ_cubes_size[i + 2][1] = width[person];
    christ_cubes_size[i + 2][2] = body_length[person];

    // neck
    christ_cubes_pos[i + 3][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 3][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 3][2] = pos[person][2] + leg_length[person] + body_length[person] + neck_length[person] / 2.;
    christ_cubes_size[i + 3][0] = depth[person];
    christ_cubes_size[i + 3][1] = neck_width[person];
    christ_cubes_size[i + 3][2] = neck_length[person];
    /*
   // head
   christ_cubes_pos [i+4][0]=pos[person][0]+depth[person]/2.;							 
   christ_cubes_pos [i+4][1]=pos[person][1]+width[person]/2.;							 
   christ_cubes_pos [i+4][2]=pos[person][2]+leg_length[person]+body_length[person]+neck_length[person]+head_length[person]/2.;   
   christ_cubes_size[i+4][0]=depth[person];
   christ_cubes_size[i+4][1]=head_width[person];
   christ_cubes_size[i+4][2]=head_length[person];
*/
    person = 5;
    i = person * OBJSPERPERSON;
    // left leg
    christ_cubes_pos[i + 0][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 0][1] = pos[person][1] + width[person] / 6.;
    christ_cubes_pos[i + 0][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 0][0] = depth[person];
    christ_cubes_size[i + 0][1] = width[person] / 3.;
    christ_cubes_size[i + 0][2] = leg_length[person];

    // right leg
    christ_cubes_pos[i + 1][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 1][1] = pos[person][1] + 5 * width[person] / 6.;
    christ_cubes_pos[i + 1][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 1][0] = depth[person];
    christ_cubes_size[i + 1][1] = width[person] / 3.;
    christ_cubes_size[i + 1][2] = leg_length[person];

    // body
    christ_cubes_pos[i + 2][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 2][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 2][2] = pos[person][2] + leg_length[person] + body_length[person] / 2.;
    christ_cubes_size[i + 2][0] = depth[person];
    christ_cubes_size[i + 2][1] = width[person];
    christ_cubes_size[i + 2][2] = body_length[person];

    // neck
    christ_cubes_pos[i + 3][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 3][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 3][2] = pos[person][2] + leg_length[person] + body_length[person] + neck_length[person] / 2.;
    christ_cubes_size[i + 3][0] = depth[person];
    christ_cubes_size[i + 3][1] = neck_width[person];
    christ_cubes_size[i + 3][2] = neck_length[person];
    /*
   // head
   christ_cubes_pos [i+4][0]=pos[person][0]+depth[person]/2.;							 
   christ_cubes_pos [i+4][1]=pos[person][1]+width[person]/2.;							 
   christ_cubes_pos [i+4][2]=pos[person][2]+leg_length[person]+body_length[person]+neck_length[person]+head_length[person]/2.;   
   christ_cubes_size[i+4][0]=depth[person];
   christ_cubes_size[i+4][1]=head_width[person];
   christ_cubes_size[i+4][2]=head_length[person];
*/
    person = 6;
    i = person * OBJSPERPERSON;
    // left leg
    christ_cubes_pos[i + 0][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 0][1] = pos[person][1] + width[person] / 6.;
    christ_cubes_pos[i + 0][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 0][0] = depth[person];
    christ_cubes_size[i + 0][1] = width[person] / 3.;
    christ_cubes_size[i + 0][2] = leg_length[person];

    // right leg
    christ_cubes_pos[i + 1][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 1][1] = pos[person][1] + 5 * width[person] / 6.;
    christ_cubes_pos[i + 1][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 1][0] = depth[person];
    christ_cubes_size[i + 1][1] = width[person] / 3.;
    christ_cubes_size[i + 1][2] = leg_length[person];

    // body
    christ_cubes_pos[i + 2][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 2][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 2][2] = pos[person][2] + leg_length[person] + body_length[person] / 2.;
    christ_cubes_size[i + 2][0] = depth[person];
    christ_cubes_size[i + 2][1] = width[person];
    christ_cubes_size[i + 2][2] = body_length[person];

    // neck
    christ_cubes_pos[i + 3][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 3][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 3][2] = pos[person][2] + leg_length[person] + body_length[person] + neck_length[person] / 2.;
    christ_cubes_size[i + 3][0] = depth[person];
    christ_cubes_size[i + 3][1] = neck_width[person];
    christ_cubes_size[i + 3][2] = neck_length[person];
    /*
   // head
   christ_cubes_pos [i+4][0]=pos[person][0]+depth[person]/2.;							 
   christ_cubes_pos [i+4][1]=pos[person][1]+width[person]/2.;							 
   christ_cubes_pos [i+4][2]=pos[person][2]+leg_length[person]+body_length[person]+neck_length[person]+head_length[person]/2.;   
   christ_cubes_size[i+4][0]=depth[person];
   christ_cubes_size[i+4][1]=head_width[person];
   christ_cubes_size[i+4][2]=head_length[person];
*/
    person = 7;
    i = person * OBJSPERPERSON;
    // left leg
    christ_cubes_pos[i + 0][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 0][1] = pos[person][1] + width[person] / 6.;
    christ_cubes_pos[i + 0][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 0][0] = depth[person];
    christ_cubes_size[i + 0][1] = width[person] / 3.;
    christ_cubes_size[i + 0][2] = leg_length[person];

    // right leg
    christ_cubes_pos[i + 1][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 1][1] = pos[person][1] + 5 * width[person] / 6.;
    christ_cubes_pos[i + 1][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 1][0] = depth[person];
    christ_cubes_size[i + 1][1] = width[person] / 3.;
    christ_cubes_size[i + 1][2] = leg_length[person];

    // body
    christ_cubes_pos[i + 2][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 2][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 2][2] = pos[person][2] + leg_length[person] + body_length[person] / 2.;
    christ_cubes_size[i + 2][0] = depth[person];
    christ_cubes_size[i + 2][1] = width[person];
    christ_cubes_size[i + 2][2] = body_length[person];

    // neck
    christ_cubes_pos[i + 3][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 3][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 3][2] = pos[person][2] + leg_length[person] + body_length[person] + neck_length[person] / 2.;
    christ_cubes_size[i + 3][0] = depth[person];
    christ_cubes_size[i + 3][1] = neck_width[person];
    christ_cubes_size[i + 3][2] = neck_length[person];
    /*
   // head
   christ_cubes_pos [i+4][0]=pos[person][0]+depth[person]/2.;							 
   christ_cubes_pos [i+4][1]=pos[person][1]+width[person]/2.;							 
   christ_cubes_pos [i+4][2]=pos[person][2]+leg_length[person]+body_length[person]+neck_length[person]+head_length[person]/2.;   
   christ_cubes_size[i+4][0]=depth[person];
   christ_cubes_size[i+4][1]=head_width[person];
   christ_cubes_size[i+4][2]=head_length[person];
*/
    person = 8;
    i = person * OBJSPERPERSON;
    // left leg
    christ_cubes_pos[i + 0][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 0][1] = pos[person][1] + width[person] / 6.;
    christ_cubes_pos[i + 0][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 0][0] = depth[person];
    christ_cubes_size[i + 0][1] = width[person] / 3.;
    christ_cubes_size[i + 0][2] = leg_length[person];

    // right leg
    christ_cubes_pos[i + 1][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 1][1] = pos[person][1] + 5 * width[person] / 6.;
    christ_cubes_pos[i + 1][2] = pos[person][2] + leg_length[person] / 2.;
    christ_cubes_size[i + 1][0] = depth[person];
    christ_cubes_size[i + 1][1] = width[person] / 3.;
    christ_cubes_size[i + 1][2] = leg_length[person];

    // body
    christ_cubes_pos[i + 2][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 2][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 2][2] = pos[person][2] + leg_length[person] + body_length[person] / 2.;
    christ_cubes_size[i + 2][0] = depth[person];
    christ_cubes_size[i + 2][1] = width[person];
    christ_cubes_size[i + 2][2] = body_length[person];

    // neck
    christ_cubes_pos[i + 3][0] = pos[person][0] + depth[person] / 2.;
    christ_cubes_pos[i + 3][1] = pos[person][1] + width[person] / 2.;
    christ_cubes_pos[i + 3][2] = pos[person][2] + leg_length[person] + body_length[person] + neck_length[person] / 2.;
    christ_cubes_size[i + 3][0] = depth[person];
    christ_cubes_size[i + 3][1] = neck_width[person];
    christ_cubes_size[i + 3][2] = neck_length[person];
    /*
   // head
   christ_cubes_pos [i+4][0]=pos[person][0]+depth[person]/2.;							 
   christ_cubes_pos [i+4][1]=pos[person][1]+width[person]/2.;							 
   christ_cubes_pos [i+4][2]=pos[person][2]+leg_length[person]+body_length[person]+neck_length[person]+head_length[person]/2.;   
   christ_cubes_size[i+4][0]=depth[person];
   christ_cubes_size[i+4][1]=head_width[person];
   christ_cubes_size[i+4][2]=head_length[person];
*/
    // relative position values
    // we use that in christmas06::GetParamsFromControlPanel
    /*
   // station tower - relative to station (0)
   christ_cubes_pos[ 7][0]= 91.500;  christ_cubes_size[ 7][0]= 15.000;
   christ_cubes_pos[ 7][1]=  0.000;  christ_cubes_size[ 7][1]= 15.000;
   christ_cubes_pos[ 7][2]= 15.000;  christ_cubes_size[ 7][2]= 55.000;

   // L-building part 2 - relative to L-building part 1 (1)
   christ_cubes_pos[ 8][0]= 27.500;  christ_cubes_size[ 8][0]= 13.000;
   christ_cubes_pos[ 8][1]= 27.500;  christ_cubes_size[ 8][1]= 50.000;
   christ_cubes_pos[ 8][2]=  0.000;  christ_cubes_size[ 8][2]= 25.000;

   // roof long building 1 - relative to long building 1 (2)
   christ_cubes_pos[ 9][0]=  0.000;  christ_cubes_size[ 9][0]= 47.000;
   christ_cubes_pos[ 9][1]=  0.000;  christ_cubes_size[ 9][1]= 10.000;
   christ_cubes_pos[ 9][2]= 15.000;  christ_cubes_size[ 9][2]=  5.000;

   // roof L-building part 1 - relative to L-building part 1 (1)
   christ_cubes_pos[10][0]=  0.000;  christ_cubes_size[10][0]= 47.000;
   christ_cubes_pos[10][1]=  0.000;  christ_cubes_size[10][1]= 10.000;
   christ_cubes_pos[10][2]= 15.000;  christ_cubes_size[10][2]=  5.000;

   // roof L-building part 2 - relative to L-building part 1 (1)
   christ_cubes_pos[11][0]= 27.500;  christ_cubes_size[11][0]= 10.000;
   christ_cubes_pos[11][1]= 27.500;  christ_cubes_size[11][1]= 47.000;
   christ_cubes_pos[11][2]= 15.000;  christ_cubes_size[11][2]=  5.000;

   // roof long building 2 - relative to long building 2 (3)
   christ_cubes_pos[12][0]=  0.000;  christ_cubes_size[12][0]= 10.000;
   christ_cubes_pos[12][1]=  0.000;  christ_cubes_size[12][1]= 47.000;
   christ_cubes_pos[12][2]= 15.000;  christ_cubes_size[12][2]=  5.000;
*/
    for (i = 0; i < NOBJECTS; i++)
    {
        p_cubes_pos[i]->setValue(christ_cubes_pos[i][0], christ_cubes_pos[i][1], christ_cubes_pos[i][2]);
        p_cubes_size[i]->setValue(christ_cubes_size[i][0], christ_cubes_size[i][1], christ_cubes_size[i][2]);
    }

    return 0;
}

MODULE_MAIN(UnderDev, christmas06)
