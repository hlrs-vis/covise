/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Read module for VECTIS Files                              **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Andreas Wierse                              **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  30.07.98  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadVectis.h"
#include "VectisFile.h"
#include <string.h>

//macros
#define ERR0(cond, text, action)     \
    {                                \
        if (cond)                    \
        {                            \
            Covise::sendError(text); \
            {                        \
                action               \
            }                        \
        }                            \
    }

#define ERR1(cond, text, arg1, action) \
    {                                  \
        if (cond)                      \
        {                              \
            sprintf(buf, text, arg1);  \
            Covise::sendError(buf);    \
            {                          \
                action                 \
            }                          \
        }                              \
    }

#define ERR2(cond, text, arg1, arg2, action) \
    {                                        \
        if (cond)                            \
        {                                    \
            sprintf(buf, text, arg1, arg2);  \
            Covise::sendError(buf);          \
            {                                \
                action                       \
            }                                \
        }                                    \
    }

static const int NDATA = 3; // number of results data fields

int main(int argc, char *argv[])
{
    ReadVectis *application = new ReadVectis(argc, argv);
    application->run();
    return 0;
}

//
// static stub callback functions calling the real class
// member functions
//

void ReadVectis::quitCallback(void *userData, void *callbackData)
{
    ReadVectis *thisApp = (ReadVectis *)userData;
    thisApp->quit(callbackData);
}

void ReadVectis::computeCallback(void *userData, void *callbackData)
{
    ReadVectis *thisApp = (ReadVectis *)userData;
    thisApp->compute(callbackData);
}

void
ReadVectis::paramCallback(void *userData, void *callbackData)
{
    ReadVectis *thisApp = (ReadVectis *)userData;
    thisApp->paramChange(callbackData);
}

/*********************************
 *                               *
 *     C O N S T R U C T O R     *
 *                               *
 *********************************/

ReadVectis::ReadVectis(int argc, char *argv[])
{
    vectisfile = 0L;
    vdata = new VectisData;
    file_name = 0L;
    pts = NULL;

    Covise::set_module_description("Read VECTIS Files");

    // File Name
    Covise::add_port(PARIN, "file_name", "Browser", "File path");
    Covise::set_port_default("file_name", "./*.*");
    Covise::set_port_immediate("file_name", 1);

    Covise::add_port(PARIN, "field1", "Choice", "Field to be read");
    Covise::set_port_default("field1", "1 ---");

    Covise::add_port(PARIN, "field2", "Choice", "Field to be read");
    Covise::set_port_default("field2", "1 ---");

    Covise::add_port(PARIN, "field3", "Choice", "Field to be read");
    Covise::set_port_default("field3", "1 ---");

    // Output
    Covise::add_port(OUTPUT_PORT, "mesh", "DO_CellGrid", "Mesh output");
    Covise::add_port(OUTPUT_PORT, "surface", "coDoPolygons", "Patch Output");
    Covise::add_port(OUTPUT_PORT, "data1", "coDoVec3 | coDoFloat", "Data Field 1 output");
    Covise::add_port(OUTPUT_PORT, "data2", "coDoVec3 | coDoFloat", "Data Field 2 output");
    Covise::add_port(OUTPUT_PORT, "data3", "coDoVec3 | coDoFloat", "Data Field 3 output");

    // Do the setup
    Covise::init(argc, argv);
    Covise::set_quit_callback(ReadVectis::quitCallback, this);
    Covise::set_start_callback(ReadVectis::computeCallback, this);
    Covise::set_param_callback(ReadVectis::paramCallback, this);

    // Set internal object pointers to Files and Filenames
}

/*******************************
 *                             *
 *     D E S T R U C T O R     *
 *                             *
 *******************************/

ReadVectis::~ReadVectis()
{
}

void ReadVectis::quit(void *)
{
    //
    // ...... delete your data here .....
    //
}

void ReadVectis::paramChange(void *)
{

    const char *tmp;
    char *pname, *new_file_name;

    // get watchdir parameter
    pname = Covise::get_reply_param_name();

    if (strcmp("file_name", pname) == NULL)
    {
        Covise::get_reply_browser(&tmp);

        new_file_name = (char *)new char[strlen(tmp) + 1];
        strcpy(new_file_name, tmp);

        if (new_file_name != NULL)
        {
            if (file_name == 0 || strcmp(file_name, new_file_name) != 0)
            {
                delete file_name;
                pts = NULL;
                file_name = new_file_name;
                if (vectisfile != 0)
                    delete vectisfile;
                vectisfile = new VectisFile(file_name);

                ReadLinkageData();
                ResetChoiceList();
                ReadTimeStepData(SKIP);
                UpdateChoiceList();
            }
        }
        else
        {
            Covise::sendError("ERROR:file_name is NULL");
        }
    }
}

void ReadVectis::compute(void *)
{

    // ======================== Input parameters ======================

    char *new_file_name, *tmp_name, buf[256];
    int i;
    int fieldNo[NDATA];

    Covise::get_browser_param("file_name", &tmp_name);
    new_file_name = (char *)new char[strlen(tmp_name) + 1];
    strcpy(new_file_name, tmp_name);

    //     int ende = 0;
    //     while(!ende)
    //         ;
    if (new_file_name != NULL)
    {
        if (file_name == 0L || strcmp(file_name, new_file_name) != 0)
        {
            delete file_name;
            file_name = new_file_name;
            if (vectisfile != 0L)
                delete vectisfile;
            vectisfile = new VectisFile(file_name);
            ReadLinkageData();
            ResetChoiceList();
            ReadTimeStepData(SKIP);
            UpdateChoiceList();
        }
    }
    else
    {
        Covise::sendError("ERROR:file_name is NULL");
    }

    for (i = 0; i < NDATA; i++)
    {
        sprintf(buf, "field%i", i + 1);
        Covise::get_choice_param(buf, fieldNo + i);
    }
    no1 = choicelist->get_orig_num(fieldNo[0]);
    no2 = choicelist->get_orig_num(fieldNo[1]);
    no3 = choicelist->get_orig_num(fieldNo[2]);

    cout << "Chosen: " << no1 << ", " << no2 << ", " << no3 << endl;

    WriteVectisMesh();
    WriteVectisPatch();
    WriteVectisData(no1, no2, no3);
}

int ReadVectis::ReadLinkageData()
{

    char buf[512];
    //int i, *iptr, len;
    //float *fptr;

    sprintf(buf, "Reading linkage data for %s", vectisfile->get_filename());
    Covise::sendInfo(buf);

    //     int ende = 0;
    //     while(!ende)
    //         ;

    // ident
    vectisfile->read_record(ident);
    if (ident != 1)
    {
        ERR0((ident != 99), "wrong block type for global mesh information or parallel data set", return (0);)
    }
    if (ident == 99)
    {
        ReadParallelBlock(ident);
        vectisfile->read_record(ident);
        ERR0((ident != 1), "wrong block type for global mesh information", return (0);)
    }
    ReadGlobalMeshDimensions(ident);
    // ident
    vectisfile->read_record(ident);
    ERR1((ident != 24 && ident != 25), "wrong block type for scalar cell information: %d", ident, return (0);)
    ReadScalarCellInformation(ident);
    // ident
    vectisfile->read_record(ident);
    ERR1((ident != 8), "wrong block type for velocity face information: %d", ident, return (0);)
    ReadVelocityFaceInformation(ident);

    ComputeNeighbourList();

    // ident
    vectisfile->read_record(ident);
    ERR1((ident != 45 && ident != 46), "wrong block type for patch information: %d", ident, return (0);)
    ReadPatchInformation(ident);

    Covise::sendInfo("Finished reading linkage data");

    return 1;
}

int ReadVectis::ComputeNeighbourList()
{

    return 0;
}

int ReadVectis::ReadParallelBlock(int ident)
{
    int idrand;
    int *iptr, len;

    vectisfile->read_record(idrand);
    cerr << "reading parallel file id " << idrand << endl;
    vectisfile->read_record(len, (char **)&iptr);
    cerr << "ncells:   " << iptr[0] << endl;
    cerr << "nts:      " << iptr[1] << endl;
    cerr << "nnode:    " << iptr[9] << endl;
    vectisfile->read_record(len, (char **)&iptr); // domain specific info (ncelss, etc.)
    cerr << "d_ncells: " << iptr[0] << endl;
    cerr << "d_nts:    " << iptr[1] << endl;
    cerr << "d_nnode:  " << iptr[9] << endl;
    nnode = iptr[9];
    vectisfile->skip_record(); // map_s
    vectisfile->skip_record(); // map_u
    vectisfile->skip_record(); // map_v
    vectisfile->skip_record(); // map_w
    vectisfile->skip_record(); // map_p
    vectisfile->skip_record(); // map_n

    return 1;
}

int ReadVectis::ReadGlobalMeshDimensions(int ident)
{

    int i, *iptr, len;
    float *fptr;

    // xmin, xmax, ymin, ymax, zmin, zmax
    vectisfile->read_record(len, (char **)&fptr);
    xmin = fptr[0];
    xmax = fptr[1];
    ymin = fptr[2];
    ymax = fptr[3];
    zmin = fptr[4];
    zmax = fptr[5];
    delete[] fptr;

    // ncells, nts, nbwss + 5 to be ignored
    vectisfile->read_record(len, (char **)&iptr);
    ncells = iptr[0];
    nts = iptr[1];
    delete[] iptr;

    // icube, jcube, kcube, ni, nj, nk;
    vectisfile->read_record(len, (char **)&iptr);
    icube = iptr[0];
    jcube = iptr[1];
    kcube = iptr[2];
    ni = iptr[3];
    nj = iptr[4];
    nk = iptr[5];
    delete[] iptr;

    cout << "icube: " << icube << endl;
    cout << "jcube: " << icube << endl;
    cout << "kcube: " << icube << endl;

    // *xndim, *yndim, *zndim;
    xndim = new float[ni + 1];
    yndim = new float[nj + 1];
    zndim = new float[nk + 1];
    for (i = 0; i <= ni; i++)
        vectisfile->read_record(xndim[i]);
    for (i = 0; i <= nj; i++)
        vectisfile->read_record(yndim[i]);
    for (i = 0; i <= nk; i++)
        vectisfile->read_record(zndim[i]);

    return 0;
}

int ReadVectis::ReadScalarCellInformation(int ident)
{
    int i, len; //*iptr,
    // float *fptr;

    // iglobe, jglobe, kglobe
    iglobe = new int[nts];
    jglobe = new int[nts];
    kglobe = new int[nts];
    vectisfile->read_record(len, (char **)&iglobe);
    ERR0((len != nts * sizeof(int)), "wrong number of iglobe entries", return (0);)
    vectisfile->read_record(len, (char **)&jglobe);
    ERR0((len != nts * sizeof(int)), "wrong number of jglobe entries", return (0);)
    vectisfile->read_record(len, (char **)&kglobe);
    ERR0((len != nts * sizeof(int)), "wrong number of kglobe entries", return (0);)

    switch (ident)
    {
    case 24:
        ilpack = new int[nts];
        itpack = new int[nts];
        vectisfile->read_record(len, (char **)&ilpack);
        ERR0((len != nts * sizeof(int)), "wrong number of ilpack entries", return (0);)
        vectisfile->read_record(len, (char **)&itpack);
        ERR0((len != nts * sizeof(int)), "wrong number of itpack entries", return (0);)
        ils = new char[nts];
        ile = new char[nts];
        jls = new char[nts];
        jle = new char[nts];
        kls = new char[nts];
        kle = new char[nts];
        itypew = new char[nts];
        itypee = new char[nts];
        itypes = new char[nts];
        itypen = new char[nts];
        itypel = new char[nts];
        itypeh = new char[nts];
        for (i = 0; i < nts; i++)
        {
            ils[i] = 0x0000001f & ilpack[i];
            ile[i] = 0x0000001f & ilpack[i] >> 5;
            jls[i] = 0x0000001f & ilpack[i] >> 10;
            jle[i] = 0x0000001f & ilpack[i] >> 15;
            kls[i] = 0x0000001f & ilpack[i] >> 20;
            kle[i] = 0x0000001f & ilpack[i] >> 25;
            itypew[i] = 0x0000001f & itpack[i];
            itypee[i] = 0x0000001f & itpack[i] >> 5;
            itypes[i] = 0x0000001f & itpack[i] >> 10;
            itypen[i] = 0x0000001f & itpack[i] >> 15;
            itypel[i] = 0x0000001f & itpack[i] >> 20;
            itypeh[i] = 0x0000001f & itpack[i] >> 25;
        }
        voln = new float[nts];
        vectisfile->read_record(len, (char **)&voln);
        ERR0((len != nts * sizeof(int)), "wrong number of voln entries", return (0);)
        break;
    case 25:
        ERR0(1, "block 25 for scalar cell information not yet implemented", return (0);)
        break;
    }

    return 0;
}

int ReadVectis::ReadVelocityFaceInformation(int ident)
{
    int i, *iptr, len;
    //float *fptr;

    // ncellu, ntu, ncellv, ntv, ncellw, ntw
    vectisfile->read_record(len, (char **)&iptr);
    //	cout << "ncellu:  " << iptr[0] << endl;
    //	cout << "ntu:  " << iptr[1] << endl;
    //	cout << "ncellv:  " << iptr[2] << endl;
    //	cout << "ntv:     " << iptr[3] << endl;
    //	cout << "ncellw:     " << iptr[4] << endl;
    //	cout << "ntw:     " << iptr[5] << endl;
    ncellu = iptr[0];
    ntu = iptr[1];
    ncellv = iptr[2];
    ntv = iptr[3];
    ncellw = iptr[4];
    ntw = iptr[5];

    // iafactor, jafactor, kafactor
    vectisfile->read_record(len, (char **)&iptr);
    //	cout << "iafactor: " << iptr[0] << endl;
    //	cout << "jafactor: " << iptr[1] << endl;
    //	cout << "kafactor: " << iptr[2] << endl;
    iafactor = iptr[0];
    jafactor = iptr[1];
    kafactor = iptr[2];

    areau = new float[ntu];
    areav = new float[ntv];
    areaw = new float[ntw];

    lwus = new int[ntu];
    leus = new int[ntu];
    lsvs = new int[ntv];
    lnvs = new int[ntv];
    llws = new int[ntw];
    lhws = new int[ntw];

    vectisfile->read_record(len, (char **)&areau);
    vectisfile->read_record(len, (char **)&lwus);
    vectisfile->read_record(len, (char **)&leus);
    vectisfile->read_record(len, (char **)&areav);
    vectisfile->read_record(len, (char **)&lsvs);
    vectisfile->read_record(len, (char **)&lnvs);
    vectisfile->read_record(len, (char **)&areaw);
    vectisfile->read_record(len, (char **)&llws);
    vectisfile->read_record(len, (char **)&lhws);

    global_cell2face = new CellPointer[nts];

    for (i = 0; i < nts; i++)
    {
        global_cell2face[i].e = -1;
        global_cell2face[i].w = -1;
        global_cell2face[i].s = -1;
        global_cell2face[i].n = -1;
        global_cell2face[i].l = -1;
        global_cell2face[i].h = -1;
    }

    for (i = 0; i < ntu; i++)
    {
        global_cell2face[lwus[i] - 1].e = i;
        global_cell2face[leus[i] - 1].w = i;
    }
    for (i = 0; i < ntv; i++)
    {
        global_cell2face[lsvs[i] - 1].n = i;
        global_cell2face[lnvs[i] - 1].s = i;
    }
    for (i = 0; i < ntw; i++)
    {
        global_cell2face[llws[i] - 1].h = i;
        global_cell2face[lhws[i] - 1].l = i;
    }

    vectisfile->read_record(nfpadr);
    nfpol = new int[ntu + ntv + ntw];
    lbfpol = new int[ntu + ntv + ntw];
    lfpol = new int[nfpadr];
    vectisfile->read_record(len, (char **)&nfpol);
    vectisfile->read_record(len, (char **)&lbfpol);
    vectisfile->read_record(len, (char **)&lfpol);
    return 0;
}

int ReadVectis::ReadPatchInformation(int ident)
{
    int *iptr, len; // i,
    //float *fptr;

    // nbpatch, nbound
    vectisfile->read_record(len, (char **)&iptr);
    nbpatch = iptr[0];
    nbound = iptr[1];
    nnode = iptr[2];
    cerr << "nnode : " << nnode << endl;
    nnodref = iptr[3];

    ncpactual = new int[nbpatch];
    mpatch = new int[nbpatch];
    nodspp = new int[nbpatch];
    lbnod = new int[nbpatch];
    nodlist = new int[nnodref];
    ltype = new int[nbound];
    vectisfile->read_record(len, (char **)&ncpactual);
    vectisfile->read_record(len, (char **)&mpatch);
    vectisfile->read_record(len, (char **)&ltype);
    vectisfile->read_record(len, (char **)&nodspp);
    vectisfile->read_record(len, (char **)&lbnod);
    vectisfile->read_record(len, (char **)&nodlist);
    if (pts == NULL)
    {
        pts = new float[nnode * 3];
        vectisfile->read_record(len, (char **)&pts);
    }
    else
    {
        vectisfile->skip_record();
    }

    if (ident == 46)
    {
        int dummy; // these two block should be empty
        vectisfile->read_record(len, (char **)&dummy);
        ERR0((len != 0), "in Block 46 non empty block found", return (0);)
        vectisfile->read_record(len, (char **)&dummy);
        ERR0((len != 0), "in Block 46 non empty block found", return (0);)
    }
    return 0;
}

int ReadVectis::ReadTimeStepData(int skip)
{
    char buf[512];
    char *tmp_text;
    int len;

    Covise::sendInfo("Reading timestepdata");

    // ident
    vectisfile->read_record(ident);
    ERR1((ident != 600), "wrong block type for time step data: %d", ident, return (0);)

    vectisfile->read_textrecord(&tmp_text);
    //	cout << "Time step text: " << tmp_text << endl;

    while (strcmp(tmp_text, "END_DATA"))
    {
        if (!strcmp(tmp_text, "REFERENCE_PRESSURE"))
        {
            vectisfile->read_record(vdata->pref);
            cout << "Ref pressure: " << vdata->pref << endl;
        }
        else if (!strcmp(tmp_text, "TIME"))
        {
            vectisfile->read_record(vdata->time);
            cout << "Time:         " << vdata->time << endl;
        }
        else if (!strcmp(tmp_text, "CRANKANGLE"))
        {
            vectisfile->read_record(vdata->cangle);
            cout << "Crank angle:  " << vdata->cangle << endl;
        }
        else if (!strcmp(tmp_text, "U_VELOCITY"))
        {
            if (skip)
            {
                vdata->o_uvel = vectisfile->set_lseek();
                vectisfile->skip_record();
            }
            else
            {
                if (vdata->uvel)
                    delete vdata->uvel;
                vdata->uvel = new float[ncells];
                vectisfile->read_record(len, (char **)&vdata->uvel);
            }
            cout << "U Velocity" << endl;
        }
        else if (!strcmp(tmp_text, "V_VELOCITY"))
        {
            if (skip)
            {
                vdata->o_vvel = vectisfile->set_lseek();
                vectisfile->skip_record();
            }
            else
            {
                if (vdata->vvel)
                    delete vdata->vvel;
                vdata->vvel = new float[ncells];
                vectisfile->read_record(len, (char **)&vdata->vvel);
            }
            cout << "V Velocity" << endl;
        }
        else if (!strcmp(tmp_text, "W_VELOCITY"))
        {
            if (skip)
            {
                vdata->o_wvel = vectisfile->set_lseek();
                vectisfile->skip_record();
            }
            else
            {
                if (vdata->wvel)
                    delete vdata->wvel;
                vdata->wvel = new float[ncells];
                vectisfile->read_record(len, (char **)&vdata->wvel);
            }
            cout << "W Velocity" << endl;
        }
        else if (!strcmp(tmp_text, "PRESSURE"))
        {
            if (skip)
            {
                vdata->o_p = vectisfile->set_lseek();
                vectisfile->skip_record();
            }
            else
            {
                if (vdata->p)
                    delete vdata->p;
                vdata->p = new float[ncells];
                vectisfile->read_record(len, (char **)&vdata->p);
            }
            cout << "Pressure" << endl;
        }
        else if (!strcmp(tmp_text, "TEMPERATURE"))
        {
            if (skip)
            {
                vdata->o_t = vectisfile->set_lseek();
                vectisfile->skip_record();
            }
            else
            {
                if (vdata->t)
                    delete vdata->t;
                vdata->t = new float[ncells];
                vectisfile->read_record(len, (char **)&vdata->t);
            }
            cout << "Temperature" << endl;
        }
        else if (!strcmp(tmp_text, "PASSIVE_SCALAR"))
        {
            if (skip)
            {
                vdata->o_ps1 = vectisfile->set_lseek();
                vectisfile->skip_record();
            }
            else
            {
                if (vdata->ps1)
                    delete vdata->ps1;
                vdata->ps1 = new float[ncells];
                vectisfile->read_record(len, (char **)&vdata->ps1);
            }
            cout << "Passive Scalar" << endl;
        }
        else if (!strcmp(tmp_text, "TURBULENCE_ENERGY"))
        {
            if (skip)
            {
                vdata->o_te = vectisfile->set_lseek();
                vectisfile->skip_record();
            }
            else
            {
                if (vdata->te)
                    delete vdata->te;
                vdata->te = new float[ncells];
                vectisfile->read_record(len, (char **)&vdata->te);
            }
            cout << "Turbulence Energy" << endl;
        }
        else if (!strcmp(tmp_text, "TURBULENCE_DISSIPATION"))
        {
            if (skip)
            {
                vdata->o_ed = vectisfile->set_lseek();
                vectisfile->skip_record();
            }
            else
            {
                if (vdata->ed)
                    delete vdata->ed;
                vdata->ed = new float[ncells];
                vectisfile->read_record(len, (char **)&vdata->ed);
            }
            cout << "Turbulence Dissipation" << endl;
        }
        else if (!strcmp(tmp_text, "DENSITY"))
        {
            if (skip)
            {
                vdata->o_den = vectisfile->set_lseek();
                vectisfile->skip_record();
            }
            else
            {
                if (vdata->den)
                    delete vdata->den;
                vdata->den = new float[ncells];
                vectisfile->read_record(len, (char **)&vdata->den);
            }
            cout << "Density" << endl;
        }
        else if (!strcmp(tmp_text, "PATCH_TEMPERATURE"))
        {
            if (skip)
            {
                vdata->o_tpatch = vectisfile->set_lseek();
                vectisfile->skip_record();
            }
            else
            {
                if (vdata->tpatch)
                    delete vdata->tpatch;
                vdata->tpatch = new float[nbpatch];
                vectisfile->read_record(len, (char **)&vdata->tpatch);
            }
            cout << "Patch Temperature" << endl;
        }
        else if (!strcmp(tmp_text, "PATCH_FLUID_TEMP"))
        {
            if (skip)
            {
                vdata->o_tflpatch = vectisfile->set_lseek();
                vectisfile->skip_record();
            }
            else
            {
                if (vdata->tflpatch)
                    delete vdata->tflpatch;
                vdata->tflpatch = new float[nbpatch];
                vectisfile->read_record(len, (char **)&vdata->tflpatch);
            }
            cout << "Patch Fluid Temp" << endl;
        }
        else if (!strcmp(tmp_text, "PATCH_HTC"))
        {
            if (skip)
            {
                vdata->o_gpatch = vectisfile->set_lseek();
                vectisfile->skip_record();
            }
            else
            {
                if (vdata->gpatch)
                    delete vdata->gpatch;
                vdata->gpatch = new float[nbpatch];
                vectisfile->read_record(len, (char **)&vdata->gpatch);
            }
            cout << "Patch HTC" << endl;
        }
        else if (!strcmp(tmp_text, "PATCH_SHEAR"))
        {
            if (skip)
            {
                vdata->o_taupatch = vectisfile->set_lseek();
                vectisfile->skip_record();
            }
            else
            {
                if (vdata->taupatch)
                    delete vdata->taupatch;
                vdata->taupatch = new float[nbpatch];
                vectisfile->read_record(len, (char **)&vdata->taupatch);
            }
            cout << "Patch Shear" << endl;
        }
        else if (!strcmp(tmp_text, "NODE_COORDINATES"))
        {
            if (skip)
            {
                vdata->o_noco = vectisfile->set_lseek();
                vectisfile->skip_record();
            }
            else
            {
                if (pts)
                    delete pts;
                pts = new float[nnode * 3];
                vectisfile->read_record(len, (char **)&pts);
            }
            cout << "Density" << endl;
        }
        else
        {
            sprintf(buf, "Unknown variable %s in Vectis file", tmp_text);
            ERR0((1), buf, return (0););
        }

        vectisfile->read_textrecord(&tmp_text);
        //		cout << "Next text: " << tmp_text << endl;
    }

    Covise::sendInfo("Finished reading timestepdata");

    return 1;
}

int ReadVectis::ResetChoiceList(void)
{

    delete choicelist;

    choicelist = new ChoiceList("---", 0);

    return 1;
}

int ReadVectis::UpdateChoiceList(void)
{

    if (vdata->o_p)
    {
        choicelist->add("pressure", V_PRESSURE);
        choicelist->add("patch_pressure", V_PATCH_PRESSURE);
    }
    if (vdata->o_den)
    {
        choicelist->add("density", V_DENSITY);
        choicelist->add("patch_density", V_PATCH_DENSITY);
    }
    if (vdata->o_t)
    {
        choicelist->add("temperature", V_TEMPERATURE);
        choicelist->add("patch_temperature_cellbased", V_PATCH_TEMPERATURE_CELLBASED);
    }
    if (vdata->o_ps1)
    {
        choicelist->add("passive_scalar", V_PASSIVE_SCALAR);
        choicelist->add("patch_passive_scalar", V_PATCH_PASSIVE_SCALAR);
    }
    if (vdata->o_te)
    {
        choicelist->add("turbulence_energy", V_TURBULENCE_ENERGY);
        choicelist->add("patch_turbulence_energy", V_PATCH_TURBULENCE_ENERGY);
    }
    if (vdata->o_ed)
    {
        choicelist->add("turbulence_dissipation", V_TURBULENCE_DISSIPATION);
        choicelist->add("patch_turbulence_dissipation", V_PATCH_TURBULENCE_DISSIPATION);
    }
    if (vdata->o_uvel)
    {
        choicelist->add("velocity", V_VELOCITY);
        choicelist->add("patch_velocity", V_PATCH_VELOCITY);
    }
    if (vdata->o_tpatch)
        choicelist->add("patch_temperature_direct", V_PATCH_TEMPERATURE_DIRECT);
    if (vdata->o_tflpatch)
        choicelist->add("patch_fluid_temp", V_PATCH_FLUID_TEMP);
    if (vdata->o_gpatch)
        choicelist->add("patch_HTC", V_PATCH_HTC);
    if (vdata->o_taupatch)
        choicelist->add("patch_shear", V_PATCH_SHEAR);

    Covise::update_choice_param(
        "field1", choicelist->get_num(), (char **)choicelist->get_strings(), 1);
    Covise::update_choice_param(
        "field2", choicelist->get_num(), (char **)choicelist->get_strings(), 1);
    Covise::update_choice_param(
        "field3", choicelist->get_num(), (char **)choicelist->get_strings(), 1);

    return 1;
}

int ReadVectis::WriteVectisMesh()
{
    char *Mesh = Covise::get_object_name("mesh");
    int i, j;
    int *c_l, *v_l, *t_l;
    int cell_count, coord_count, node_count;
    int *dummy; // iile, iils,
    int *gci_l, *gcsc_l, *cigc_l;
    float *x, *y, *z;
    float *xm, *ym, *zm;
    // 	float txmin, txmax, tymin, tymax, tzmin, tzmax;
    // 	float x_part_width, y_part_width, z_part_width;

    ERR0((Mesh == NULL), "Error getting name 'mesh'", return (0);)

    DO_CellGrid *mesh = new DO_CellGrid(Mesh, ncells * 8,
                                        ncells * 8, 0, 0, 0, TYPE_HEXAEDER, ncells, ni, nj, nk);

    ERR0((mesh->objectOk() != 1), "Error creating Vectis object", return (0);)

    mesh->getAddresses(&x, &y, &z, &v_l,
                       &dummy, &dummy, &dummy, &c_l, &t_l,
                       &xm, &ym, &zm,
                       &gci_l, &gcsc_l, &cigc_l);

    // initialize mesh coordinates
    for (i = 0; i < ni + 1; i++)
        xm[i] = xmin + (xmax - xmin) * xndim[i];

    for (i = 0; i < nj + 1; i++)
        ym[i] = ymin + (ymax - ymin) * yndim[i];

    for (i = 0; i < nk + 1; i++)
        zm[i] = zmin + (zmax - zmin) * zndim[i];

    // initialize global cell index list
    for (i = 0; i < ni * nj * nk + 1; i++)
        gci_l[i] = 0;

    for (i = 0; i < ncells; i++)
    {
        // compute the global cell number this cell lies in
        cigc_l[i] = iglobe[i] - 1 + (jglobe[i] - 1) * ni
                    + (kglobe[i] - 1) * ni * nj;
        // increment counter that counts how many subcells a global cell has
        gci_l[cigc_l[i]]++;
        // initialize "subcell per global cell" list
        gcsc_l[i] = -1;
    }

    // compute index list for the array that holds the subcells
    // that a global cell has
    int tmp_count = 0, tmpi;
    for (i = 0; i < ni * nj * nk; i++)
    {
        tmpi = gci_l[i];
        gci_l[i] = tmp_count;
        tmp_count += tmpi;
    }
    gci_l[i] = tmp_count;

    // fill the list that holds the subcells that a global cell has
    int gci;
    for (i = 0; i < ncells; i++)
    {
        gci = gci_l[cigc_l[i]];
        j = 0;
        while (gcsc_l[gci + j] != -1)
            j++;
        gcsc_l[gci + j] = i;
    }

    //  x = new float[ncells * 8];
    // 	y = new float[ncells * 8];
    // 	z = new float[ncells * 8];

    coord_count = cell_count = 0;

    for (node_count = 0; node_count < nnode; node_count++)
    {
        x[node_count] = pts[3 * node_count + 0];
        y[node_count] = pts[3 * node_count + 1];
        z[node_count] = pts[3 * node_count + 2];
    }

    for (cell_count = 0; cell_count < ncells; cell_count++)
    {
        v_l[cell_count * 8 + 0] = lfpol[lbfpol[global_cell2face[i].w + 0]];
        v_l[cell_count * 8 + 1] = lfpol[lbfpol[global_cell2face[i].w + 1]];
        v_l[cell_count * 8 + 2] = lfpol[lbfpol[global_cell2face[i].w + 2]];
        v_l[cell_count * 8 + 3] = lfpol[lbfpol[global_cell2face[i].w + 3]];
        v_l[cell_count * 8 + 4] = lfpol[lbfpol[global_cell2face[i].e + 3]];
        v_l[cell_count * 8 + 5] = lfpol[lbfpol[global_cell2face[i].e + 2]];
        v_l[cell_count * 8 + 6] = lfpol[lbfpol[global_cell2face[i].e + 1]];
        v_l[cell_count * 8 + 7] = lfpol[lbfpol[global_cell2face[i].e + 0]];

        // set cell list (points to vertex list, also simple)
        c_l[cell_count] = cell_count * 8;

        t_l[cell_count] = TYPE_HEXAGON;
    }
}

// int ReadVectis::WriteVectisMesh() {
// 	int i, j;
// 	int *c_l, *v_l, *t_l;
// 	int cell_count, coord_count;
// 	int *dummy;  // iile, iils,
//     int *gci_l, *gcsc_l, *cigc_l;
//     float *x, *y, *z;
//     float *xm, *ym, *zm;
// 	float txmin, txmax, tymin, tymax, tzmin, tzmax;
// 	float x_part_width, y_part_width, z_part_width;
// 	char *Mesh = Covise::get_object_name("mesh");
//
// 	ERR0((Mesh==NULL),"Error getting name 'mesh'", return(0); )
//
// 	DO_CellGrid *mesh = new DO_CellGrid(Mesh, ncells * 8,
// 				ncells * 8, 0, 0, 0, TYPE_HEXAEDER, ncells, ni, nj, nk);
//
// 	ERR0((mesh->objectOk() != 1),"Error creating Vectis object", return(0); )
//
//     mesh->getAddresses(&x, &y, &z, &v_l,
// 			  &dummy, &dummy, &dummy, &c_l, &t_l,
//               &xm, &ym, &zm,
//               &gci_l, &gcsc_l, &cigc_l);
//
//     // initialize mesh coordinates
//     for(i = 0 ; i < ni + 1; i++)
//         xm[i] = xmin + (xmax - xmin) * xndim[i];
//
//     for(i = 0 ; i < nj + 1; i++)
//         ym[i] = ymin + (ymax - ymin) * yndim[i];
//
//     for(i = 0 ; i < nk + 1; i++)
//         zm[i] = zmin + (zmax - zmin) * zndim[i];
//
//     // initialize global cell index list
//     for(i = 0 ; i < ni * nj * nk + 1 ; i++)
//         gci_l[i] = 0;
//
//     for(i = 0 ; i < ncells ; i++ ) {
//     // compute the global cell number this cell lies in
//         cigc_l[i] = iglobe[i] - 1 + (jglobe[i] - 1) * ni
//                     + (kglobe[i] - 1) * ni * nj;
//     // increment counter that counts how many subcells a global cell has
//         gci_l[cigc_l[i]]++;
//     // initialize "subcell per global cell" list
//         gcsc_l[i] = -1;
//     }
//
//     // compute index list for the array that holds the subcells
//     // that a global cell has
//     int tmp_count = 0, tmpi;
//     for(i = 0 ; i < ni * nj * nk ; i++) {
//         tmpi = gci_l[i];
//         gci_l[i] = tmp_count;
//         tmp_count += tmpi;
//     }
//     gci_l[i] = tmp_count;
//
//     // fill the list that holds the subcells that a global cell has
//     int gci;
//     for(i = 0 ; i < ncells ; i++ ) {
//         gci = gci_l[cigc_l[i]];
//         j = 0;
//         while(gcsc_l[gci + j] != -1)
//             j++;
//         gcsc_l[gci + j] = i;
//     }
//
// //  x = new float[ncells * 8];
// // 	y = new float[ncells * 8];
// // 	z = new float[ncells * 8];
//
// 	coord_count = cell_count = 0;
//
// 	for(cell_count = 0;cell_count < ncells;cell_count++) {
// 		txmin = xmin + (xmax - xmin) * xndim[iglobe[cell_count]-1];
// 		txmax = xmin + (xmax - xmin) * xndim[iglobe[cell_count]-1+1];
// 		x_part_width = (txmax - txmin) / icube;
// 		x[coord_count+0] = x[coord_count+3] =
// 		x[coord_count+4] = x[coord_count+7] =
// 		txmin + ((ils[cell_count] - 1) / 2) * x_part_width;
// 		x[coord_count+1] = x[coord_count+2] =
// 		x[coord_count+5] = x[coord_count+6] =
// 		txmin + (ile[cell_count] / 2) * x_part_width;
//
// /*
//                 if(cell_count % 20000 == 0) {
//                     iile = ile[cell_count];
//                     iils = ils[cell_count];
//
//                     cout << "ils[" << cell_count << "]: " << iils << endl;
//                     cout << "ile[" << cell_count << "]: " << iile << endl;
//                  }
// */
//
// 		tymin = ymin + (ymax - ymin) * yndim[jglobe[cell_count]-1];
// 		tymax = ymin + (ymax - ymin) * yndim[jglobe[cell_count]-1+1];
// 		y_part_width = (tymax - tymin) / jcube;
// 		y[coord_count+0] = y[coord_count+1] =
// 		y[coord_count+2] = y[coord_count+3] =
// 		tymin + ((jls[cell_count] - 1) / 2) * y_part_width;
// 		y[coord_count+4] = y[coord_count+5] =
// 		y[coord_count+6] = y[coord_count+7] =
// 		tymin + (jle[cell_count] / 2) * y_part_width;
//
// 		tzmin = zmin + (zmax - zmin) * zndim[kglobe[cell_count]-1];
// 		tzmax = zmin + (zmax - zmin) * zndim[kglobe[cell_count]-1+1];
// 		z_part_width = (tzmax - tzmin) / kcube;
// 		z[coord_count+0] = z[coord_count+1] =
// 		z[coord_count+4] = z[coord_count+5] =
// 		tzmin + ((kls[cell_count] - 1) / 2) * z_part_width;
// 		z[coord_count+2] = z[coord_count+3] =
// 		z[coord_count+6] = z[coord_count+7] =
// 		tzmin + (kle[cell_count] / 2) * z_part_width;
//
//     // set vertex list (simple since always hexaeders)
//         v_l[cell_count * 8 + 0] = coord_count + 0;
//         v_l[cell_count * 8 + 1] = coord_count + 1;
//         v_l[cell_count * 8 + 2] = coord_count + 2;
//         v_l[cell_count * 8 + 3] = coord_count + 3;
//         v_l[cell_count * 8 + 4] = coord_count + 4;
//         v_l[cell_count * 8 + 5] = coord_count + 5;
//         v_l[cell_count * 8 + 6] = coord_count + 6;
//         v_l[cell_count * 8 + 7] = coord_count + 7;
//
//     // set cell list (points to vertex list, also simple)
//         c_l[cell_count] = cell_count * 8;
//
//         t_l[cell_count] = TYPE_HEXAGON;
//
// 		coord_count+=8;
// 	}
//
// 	return 1;
// }

int ReadVectis::WriteVectisPatch()
{
    float *x_c, *y_c, *z_c;
    int *v_l, *l_l;
    int i, j, np, tri_count, patch_count, curr_tri; //, no_vert
    int *vert_check, node_count, vert_count, curr_node;

    char *Surface = Covise::get_object_name("surface");
    ERR0((Surface == NULL), "Error getting name 'surface'", return (0);)

    vert_check = new int[nnode];
    for (i = 0; i < nnode; i++)
        vert_check[i] = 0;

    //no_vert = 0;
    node_count = 0;
    patch_count = 0;
    tri_count = 0;
    for (i = 0; i < nbpatch; i++)
    {
        patch_count++;
        tri_count += nodspp[i] - 2;
        for (j = 0; j < nodspp[i]; j++)
        {
            np = nodlist[lbnod[i] + j] - 1;
            vert_check[np]++;
            node_count++;
        }
    }

    vert_count = 0;
    for (i = 0; i < nnode; i++)
    {
        if (vert_check[i] != 0)
        {
            vert_check[i] = vert_count;
            vert_count++;
        }
    }

    cerr << "node_count: " << node_count << endl;
    cerr << "patch_count: " << patch_count << endl;
    cerr << "tri_count: " << tri_count << endl;
    cerr << "vert_count: " << vert_count << endl;

    coDoPolygons *patch = new coDoPolygons(Surface, vert_count,
                                           tri_count * 3, tri_count);

    patch->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
    polygon2patch = new int[tri_count];

    curr_node = 0;
    curr_tri = 0;
    for (i = 0; i < nbpatch; i++)
    {
        //    for(i = 0;i < 100;i++) {
        for (j = 0; j < nodspp[i]; j++)
        {
            np = nodlist[lbnod[i] + j] - 1; // -1 due to Fortran arrays
            //            if(vert_check[np] != -1) {
            x_c[vert_check[np]] = pts[3 * np];
            y_c[vert_check[np]] = pts[3 * np + 1];
            z_c[vert_check[np]] = pts[3 * np + 2];
            //                vert_check[np] = -1;
            //            }
            if (j < nodspp[i] - 2)
            {
                l_l[curr_tri] = curr_node;
                polygon2patch[curr_tri] = i;
                curr_tri++;
                v_l[curr_node++] = vert_check[nodlist[lbnod[i] + j + 2] - 1];
                v_l[curr_node++] = vert_check[nodlist[lbnod[i] + j + 1] - 1];
                v_l[curr_node++] = vert_check[nodlist[lbnod[i]] - 1];
            }
        }
    }
    nbtris = curr_tri;

    delete patch;
    /*
      for(i = 0;i < 3 * nbpatch;i+=3) {
         x_c[i] = xpatch[i+2];
         y_c[i] = ypatch[i+2];
         z_c[i] = zpatch[i+2];
         x_c[i+1] = xpatch[i+1];
         y_c[i+1] = ypatch[i+1];
         z_c[i+1] = zpatch[i+1];
         x_c[i+2] = xpatch[i];
         y_c[i+2] = ypatch[i];
         z_c[i+2] = zpatch[i];
   v_l[i]   = i;
   v_l[i+1] = i+1;
   v_l[i+2] = i+2;
   }
   for(i = 0;i < nbpatch;i++)
   l_l[i] = 3 * i;
   */
    return 1;
}

int ReadVectis::WriteVectisData(int no1, int no2, int no3)
{
    int no_list[3];
    int field, i, j;
    char objName[16];
    char *Name, buf[256];
    float *S, mint = 1000000, maxt = -1000000;
    float *U, *V, *W;
    // temp patch data to be read in and distribute to triangles
    float *read_data = new float[nbpatch];

    no_list[0] = no1;
    no_list[1] = no2;
    no_list[2] = no3;

    for (i = 0; i < 3; i++)
    {
        if (no_list[i] > 0)
        {
            field = no_list[i];
            sprintf(objName, "data%i", i + 1);
            Name = Covise::get_object_name(objName);
            ERR1((Name == NULL), "Error getting name '%s'", objName, return 0;)

            if (field == V_VELOCITY) // Displacements -> Vector
            {
                coDoVec3 *data
                    = new coDoVec3(Name, ncells);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;);
                data->getAddresses(&U, &V, &W);
                vectisfile->goto_lseek(vdata->o_uvel);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)U);
                vectisfile->goto_lseek(vdata->o_vvel);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)V);
                vectisfile->goto_lseek(vdata->o_wvel);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)W);
                delete data;
            }
            else if (field == V_PRESSURE)
            {
                coDoFloat *data
                    = new coDoFloat(Name, ncells);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;);
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_p);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)S);
                for (j = 0; j < ncells; j++)
                {
                    if (S[j] > maxt)
                        maxt = S[j];
                    else if (S[j] < mint)
                        mint = S[j];
                }
                cout << "Pressure between " << mint << " and " << maxt << endl;
                delete data;
            }
            else if (field == V_DENSITY)
            {
                coDoFloat *data
                    = new coDoFloat(Name, ncells);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;);
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_den);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)S);
                for (j = 0; j < ncells; j++)
                {
                    if (S[j] > maxt)
                        maxt = S[j];
                    else if (S[j] < mint)
                        mint = S[j];
                }
                cout << "Density between " << mint << " and " << maxt << endl;
                delete data;
            }
            else if (field == V_TEMPERATURE)
            {
                coDoFloat *data
                    = new coDoFloat(Name, ncells);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;);
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_t);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)S);
                for (j = 0; j < ncells; j++)
                {
                    if (S[j] > maxt)
                        maxt = S[j];
                    else if (S[j] < mint)
                        mint = S[j];
                }
                cout << "Temperature between " << mint << " and " << maxt << endl;
                delete data;
            }
            else if (field == V_PASSIVE_SCALAR)
            {
                coDoFloat *data
                    = new coDoFloat(Name, ncells);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;);
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_ps1);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)S);
                for (j = 0; j < ncells; j++)
                {
                    if (S[j] > maxt)
                        maxt = S[j];
                    else if (S[j] < mint)
                        mint = S[j];
                }
                cout << "Passive scalar between " << mint << " and " << maxt << endl;
                delete data;
            }
            else if (field == V_TURBULENCE_ENERGY)
            {
                coDoFloat *data
                    = new coDoFloat(Name, ncells);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;);
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_te);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)S);
                for (j = 0; j < ncells; j++)
                {
                    if (S[j] > maxt)
                        maxt = S[j];
                    else if (S[j] < mint)
                        mint = S[j];
                }
                cout << "Turbulence energy between " << mint << " and " << maxt << endl;
                delete data;
            }
            else if (field == V_TURBULENCE_DISSIPATION)
            {
                coDoFloat *data
                    = new coDoFloat(Name, ncells);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;);
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_ed);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)S);
                for (j = 0; j < ncells; j++)
                {
                    if (S[j] > maxt)
                        maxt = S[j];
                    else if (S[j] < mint)
                        mint = S[j];
                }
                cout << "Turbulence dissipation between " << mint << " and " << maxt << endl;
                delete data;
            }
            else if (field == V_PATCH_TEMPERATURE_DIRECT)
            {
                coDoFloat *data
                    = new coDoFloat(Name, nbtris);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;)
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_tpatch);
                vectisfile->read_record(nbpatch * sizeof(float),
                                        (char *)read_data);
                for (j = 0; j < nbtris; j++)
                {
                    S[j] = read_data[polygon2patch[j]];
                    if (S[j] > maxt)
                        maxt = S[j];
                    else if (S[j] < mint)
                        mint = S[j];
                }
                cout << "Patch temperature between " << mint << " and " << maxt << endl;
                delete data;
            }
            else if (field == V_PATCH_FLUID_TEMP)
            {
                coDoFloat *data
                    = new coDoFloat(Name, nbtris);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;)
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_tflpatch);
                vectisfile->read_record(nbpatch * sizeof(float),
                                        (char *)read_data);
                for (j = 0; j < nbtris; j++)
                {
                    S[j] = read_data[polygon2patch[j]];
                    if (S[j] > maxt)
                        maxt = S[j];
                    else if (S[j] < mint)
                        mint = S[j];
                }
                cout << "Patch fluid temp between " << mint << " and " << maxt << endl;
                delete data;
            }
            else if (field == V_PATCH_HTC)
            {
                coDoFloat *data
                    = new coDoFloat(Name, nbpatch);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;)
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_gpatch);
                vectisfile->read_record(nbpatch * sizeof(float),
                                        (char *)read_data);
                for (j = 0; j < nbtris; j++)
                {
                    S[j] = read_data[polygon2patch[j]];
                    if (S[j] > maxt)
                        maxt = S[j];
                    else if (S[j] < mint)
                        mint = S[j];
                }
                cout << "Patch HTC between " << mint << " and " << maxt << endl;
                delete data;
            }
            else if (field == V_PATCH_SHEAR)
            {
                coDoFloat *data
                    = new coDoFloat(Name, nbtris);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;)
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_taupatch);
                vectisfile->read_record(nbpatch * sizeof(float),
                                        (char *)read_data);
                for (j = 0; j < nbtris; j++)
                {
                    S[j] = read_data[polygon2patch[j]];
                    if (S[j] > maxt)
                        maxt = S[j];
                    else if (S[j] < mint)
                        mint = S[j];
                }
                cout << "Patch shear between " << mint << " and " << maxt << endl;
                delete data;
            }
            else if (field == V_PATCH_PRESSURE)
            {
                float *tmp_data = new float[ncells];
                coDoFloat *data
                    = new coDoFloat(Name, nbtris);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;)
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_p);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)tmp_data);
                int internal = 0;
                int external = 0;
                for (j = 0; j < nbtris; j++)
                {
                    if ((ncpactual[polygon2patch[j]] - 1) < ncells)
                    {
                        S[j] = tmp_data[ncpactual[polygon2patch[j]] - 1];
                        internal++;
                    }
                    else
                    {
                        int tmp_index = FindInternalCell(polygon2patch[j]);
                        S[j] = tmp_data[tmp_index];
                        //S[j] = 0.0;
                        external++;
                    }
                }
                cerr << "i: " << internal << " e: " << external << endl;
                delete data;
                delete[] tmp_data;
            }
            else if (field == V_PATCH_DENSITY)
            {
                float *tmp_data = new float[ncells];
                coDoFloat *data
                    = new coDoFloat(Name, nbtris);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;)
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_den);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)tmp_data);
                int internal = 0;
                int external = 0;
                for (j = 0; j < nbtris; j++)
                {
                    if ((ncpactual[polygon2patch[j]] - 1) < ncells)
                    {
                        S[j] = tmp_data[ncpactual[polygon2patch[j]] - 1];
                        internal++;
                    }
                    else
                    {
                        int tmp_index = FindInternalCell(polygon2patch[j]);
                        S[j] = tmp_data[tmp_index];
                        //S[j] = 0.0;
                        external++;
                    }
                }
                cerr << "i: " << internal << " e: " << external << endl;
                delete data;
                delete[] tmp_data;
            }
            else if (field == V_PATCH_TEMPERATURE_CELLBASED)
            {
                float *tmp_data = new float[ncells];
                coDoFloat *data
                    = new coDoFloat(Name, nbtris);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;)
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_t);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)tmp_data);
                int internal = 0;
                int external = 0;
                for (j = 0; j < nbtris; j++)
                {
                    if ((ncpactual[polygon2patch[j]] - 1) < ncells)
                    {
                        S[j] = tmp_data[ncpactual[polygon2patch[j]] - 1];
                        internal++;
                    }
                    else
                    {
                        int tmp_index = FindInternalCell(polygon2patch[j]);
                        S[j] = tmp_data[tmp_index];
                        //S[j] = 0.0;
                        external++;
                    }
                }
                cerr << "i: " << internal << " e: " << external << endl;
                delete data;
                delete[] tmp_data;
            }
            else if (field == V_PATCH_PASSIVE_SCALAR)
            {
                float *tmp_data = new float[ncells];
                coDoFloat *data
                    = new coDoFloat(Name, nbtris);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;)
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_ps1);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)tmp_data);
                int internal = 0;
                int external = 0;
                for (j = 0; j < nbtris; j++)
                {
                    if ((ncpactual[polygon2patch[j]] - 1) < ncells)
                    {
                        S[j] = tmp_data[ncpactual[polygon2patch[j]] - 1];
                        internal++;
                    }
                    else
                    {
                        int tmp_index = FindInternalCell(polygon2patch[j]);
                        S[j] = tmp_data[tmp_index];
                        //S[j] = 0.0;
                        external++;
                    }
                }
                cerr << "i: " << internal << " e: " << external << endl;
                delete data;
                delete[] tmp_data;
            }
            else if (field == V_PATCH_TURBULENCE_ENERGY)
            {
                float *tmp_data = new float[ncells];
                coDoFloat *data
                    = new coDoFloat(Name, nbtris);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;)
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_te);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)tmp_data);
                int internal = 0;
                int external = 0;
                for (j = 0; j < nbtris; j++)
                {
                    if ((ncpactual[polygon2patch[j]] - 1) < ncells)
                    {
                        S[j] = tmp_data[ncpactual[polygon2patch[j]] - 1];
                        internal++;
                    }
                    else
                    {
                        int tmp_index = FindInternalCell(polygon2patch[j]);
                        S[j] = tmp_data[tmp_index];
                        //S[j] = 0.0;
                        external++;
                    }
                }
                cerr << "i: " << internal << " e: " << external << endl;
                delete data;
                delete[] tmp_data;
            }
            else if (field == V_PATCH_TURBULENCE_DISSIPATION)
            {
                float *tmp_data = new float[ncells];
                coDoFloat *data
                    = new coDoFloat(Name, nbtris);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;)
                data->getAddress(&S);
                vectisfile->goto_lseek(vdata->o_ed);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)tmp_data);
                int internal = 0;
                int external = 0;
                for (j = 0; j < nbtris; j++)
                {
                    if ((ncpactual[polygon2patch[j]] - 1) < ncells)
                    {
                        S[j] = tmp_data[ncpactual[polygon2patch[j]] - 1];
                        internal++;
                    }
                    else
                    {
                        int tmp_index = FindInternalCell(polygon2patch[j]);
                        S[j] = tmp_data[tmp_index];
                        //S[j] = 0.0;
                        external++;
                    }
                }
                cerr << "i: " << internal << " e: " << external << endl;
                delete data;
                delete[] tmp_data;
            }
            else if (field == V_PATCH_VELOCITY)
            {
                float *tmp_udata = new float[ncells];
                float *tmp_vdata = new float[ncells];
                float *tmp_wdata = new float[ncells];
                coDoVec3 *data
                    = new coDoVec3(Name, nbtris);
                ERR1((data == NULL), "Error allocating '%s'", objName, return 0;)
                data->getAddresses(&U, &V, &W);
                vectisfile->goto_lseek(vdata->o_uvel);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)tmp_udata);
                vectisfile->goto_lseek(vdata->o_vvel);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)tmp_vdata);
                vectisfile->goto_lseek(vdata->o_wvel);
                vectisfile->read_record(ncells * sizeof(float),
                                        (char *)tmp_wdata);
                int internal = 0;
                int external = 0;
                for (j = 0; j < nbtris; j++)
                {
                    if ((ncpactual[polygon2patch[j]] - 1) < ncells)
                    {
                        U[j] = tmp_udata[ncpactual[polygon2patch[j]] - 1];
                        V[j] = tmp_vdata[ncpactual[polygon2patch[j]] - 1];
                        W[j] = tmp_wdata[ncpactual[polygon2patch[j]] - 1];
                        internal++;
                    }
                    else
                    {
                        int tmp_index = FindInternalCell(polygon2patch[j]);
                        U[j] = tmp_udata[tmp_index];
                        V[j] = tmp_vdata[tmp_index];
                        W[j] = tmp_wdata[tmp_index];
                        external++;
                    }
                }
                cerr << "i: " << internal << " e: " << external << endl;
                delete data;
                delete[] tmp_udata;
                delete[] tmp_vdata;
                delete[] tmp_wdata;
            }
        }
    }
    delete[] read_data;
    return 1;
}

int ReadVectis::FindInternalCell(int patch_no)
{
    /*
      float p1[3], p2[3], p3[3];
      float v1[3], v2[3];
      float n[3];
      int dir;

      p1[0] = pts[3 * (nodlist[lbnod[patch_no] + 0] - 1) + 0];
      p1[1] = pts[3 * (nodlist[lbnod[patch_no] + 0] - 1) + 1];
      p1[2] = pts[3 * (nodlist[lbnod[patch_no] + 0] - 1) + 2];
      p2[0] = pts[3 * (nodlist[lbnod[patch_no] + 1] - 1) + 0];
      p2[1] = pts[3 * (nodlist[lbnod[patch_no] + 1] - 1) + 1];
   p2[2] = pts[3 * (nodlist[lbnod[patch_no] + 1] - 1) + 2];
   p3[0] = pts[3 * (nodlist[lbnod[patch_no] + 2] - 1) + 0];
   p3[1] = pts[3 * (nodlist[lbnod[patch_no] + 2] - 1) + 1];
   p3[2] = pts[3 * (nodlist[lbnod[patch_no] + 2] - 1) + 2];

   v1[0] = p2[0] - p1[0];
   v1[1] = p2[1] - p1[1];
   v1[2] = p2[2] - p1[2];
   v2[0] = p2[0] - p3[0];
   v2[1] = p2[1] - p3[1];
   v2[2] = p2[2] - p3[2];

   n[0] = v1[1] * v2[2] - v2[1] * v1[2];
   n[1] = v1[2] * v2[0] - v2[2] * v1[0];
   n[2] = v1[0] * v2[1] - v2[0] * v1[1];

   if(fabsf(n[0]) >= fabsf(n[1]) && fabsf(n[0]) >= fabsf(n[2])) {
   if(n[0] <= 0.0)
   dir = 0;
   else
   dir = 1;
   } else if(fabsf(n[0]) <= fabsf(n[1]) && fabsf(n[1]) >= fabsf(n[2])) {
   if(n[1] <= 0.0)
   dir = 2;
   else
   dir = 3;
   } else {
   if(n[2] <= 0.0)
   dir = 4;
   else
   dir = 5;
   }
   */

    // take the easy way: ignore normals and just take a neighbour cell that is an internal cell

    int retval;

    retval = lwus[global_cell2face[ncpactual[patch_no] - 1].w] - 1;
    if (retval >= 0 && retval < ncells)
    {
        return retval;
    }

    retval = leus[global_cell2face[ncpactual[patch_no] - 1].e] - 1;
    if (retval >= 0 && retval < ncells)
    {
        return retval;
    }

    retval = lsvs[global_cell2face[ncpactual[patch_no] - 1].s] - 1;
    if (retval >= 0 && retval < ncells)
    {
        return retval;
    }

    retval = lnvs[global_cell2face[ncpactual[patch_no] - 1].n] - 1;
    if (retval >= 0 && retval < ncells)
    {
        return retval;
    }

    retval = llws[global_cell2face[ncpactual[patch_no] - 1].l] - 1;
    if (retval >= 0 && retval < ncells)
    {
        return retval;
    }

    retval = lhws[global_cell2face[ncpactual[patch_no] - 1].h] - 1;
    if (retval >= 0 && retval < ncells)
    {
        return retval;
    }

    cerr << "did not find anything matching\n";

    return -1;
}
