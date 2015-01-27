/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <iostream.h>
#include <fstream.h>
#include <strstream.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <netdb.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "ModifyCabin.h"
#include <api/coFeedback.h>

// #include "PJcovise.h"

char buf[256], buf1[256], buf2[256], buf3[256];
char tetin_buf[256];

static int n_reset = 2;
static const char *reset_Names[] = { "ResetTrans", "ResetProj" };

static int n_solver = 3;
static const char *solver_buf[] = { "Star", "Fluent", "Fenfloss" };
static const char *outp_intf[] = { "star", "fluent", "fenfloss" };

void pj_covise_get_projected_points(float **points_x, float **points_y,
                                    float **points_z, int *n_points);
void pj_covise_free_projected_points(void);
void pj_covise_get_apprx_curve_pnts(float **points_x, float **points_y,
                                    float **points_z, int *n_points);
void pj_covise_init_apprx_curve_pnts(void);
int pj_covise_tetin(coTetin *obj);
void pj_covise_init_prescr_pnts();
void pj_covise_get_prescr_pnts(float **points_x, float **points_y,
                               float **points_z, char ***pntNames,
                               int *n_points);

// this one will be in coModule of next API release

void ModifyCabin::selfExec()
{
    sprintf(buf, "T%s\n%s\n%s\n", Covise::get_module(),
            Covise::get_instance(),
            Covise::get_host());
    Covise::set_feedback_info(buf);

    // send execute message
    Covise::send_feedback_message("EXEC", "");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ModifyCabin::ModifyCabin()
{

    // declare the name of our module
    set_module_description("Modify the Cabin cases");

    p_baseTetin = addFileBrowserParam("baseTetin", "Name of basic Tetin file");
    p_baseTetin->setValue("data/visit/icem/cabin/mesh/tetin1_reg_cabin", "tetin*");
    p_baseTetin->setImmediate(1);
    baseTetinName = NULL;

    p_replayFile = addFileBrowserParam("replayFile", "Name of replay file");
    p_replayFile->setValue("data/visit/icem/cabin/hexa/replay_3_07", "repl*");
    p_replayFile->setImmediate(1);
    replayFile = NULL;

    p_configFile = addFileBrowserParam("configFile", "Name of config file");
    p_configFile->setValue("data/visit/icem/cabin/ModifyCabin.config", "*");
    p_configFile->setImmediate(1);
    configFileName = NULL;

    p_configDir = addStringParam("configDir", "Name of configuration directory");
    p_configDir->setValue("data/visit/icem/cabin");
    p_configDir->setImmediate(1);
    configDir = "data/visit/icem/cabin";

    p_tolerance = addFloatParam("tolerance", "Tolerance for tessalate");
    p_tolerance->setValue(2.0);
    p_tolerance->setImmediate(1);
    p_tolerance->show();
    tolerance = 2.0;

    p_caseName = addStringParam("CaseName", "set the casename");
    p_caseName->setValue("VISiT");
    caseName = "VISiT";

    p_trans_value = addFloatSliderParam("trans_values", "Set the translation values");
    p_trans_value->setValue(moveList[0].dmin, moveList[0].dmax, moveList[0].value);

    p_direction = addFloatSliderParam("proj_values", "Set the projection values");
    p_direction->setValue(projList[0].dmin, projList[0].dmax, projList[0].value);
    p_direction->setImmediate(1);

    p_solver = addChoiceParam("SelectSolver", "select the solver");
    p_solver->setValue(n_solver, solver_buf, 0);
    solverName = outp_intf[0];

    p_reset = addChoiceParam("ResetActions", "reset translation or projection");
    p_reset->setValue(n_reset, reset_Names, 0);
    resetName = 0;

    moveList[0].name = "NONE";
    moveList[0].dmin = 0.;
    moveList[0].dmax = 1.;
    moveList[0].value = 0.;
    moveList[0].oldvalue = 0.;
    moveNames[0] = moveList[0].name;
    p_moveName = addChoiceParam("MoveNames", "Select a name for moving");
    p_moveName->setValue(1, moveNames, 0);
    currMoveName = 0;
    numMoveNames = 1;

    projList[0].name = "NONE";
    projList[0].dmin = 0.;
    projList[0].dmax = 1.;
    projList[0].value = 0.;
    projList[0].oldvalue = 0.;
    projNames[0] = projList[0].name;
    p_projName = addChoiceParam("ProjNames", "Select a name for projection");
    p_projName->setValue(1, projNames, 0);
    currProjName = 0;
    numProjNames = 1;

    p_tetinObject = addOutputPort("tetinObj", "DO_Tetin", "Tetin commands for tesselate");
    p_hexaObject = addOutputPort("hexaObj", "DO_Tetin", "Tetin commands for hexa");
    p_addPartObject = addOutputPort("partObj", "DO_Tetin", "Tetin commands for hexa");

    p_feedback = addOutputPort("feedbackObj", "coDoPoints", "Dummy Point for Attaching Feedback");

    // defaults
    MAX_MOVE = 20;
    command = SEND_BASE_TETIN;
    newGeom = false;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// show base tetin file
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int ModifyCabin::sendBaseInfo()
{
    coTetin *tetin1 = 0, *tetin2 = 0, *tetin3 = 0;
    coTetin__tetinFile *tetinf1, *tetinf2;
    coTetin__trianTol *trian_tol;
    coTetin__trianFam *trian_fam;
    coTetin__Hexa *hexa;
    char *outp_intf_str;

    //===========================================================================
    // create output object for tesselate modul
    // Send filename, triangulation tolerance and triangulation command
    //===========================================================================
    Covise::getname(buf, baseTetinName);
    tetinf1 = new coTetin__tetinFile(buf);
    trian_tol = new coTetin__trianTol(tolerance);
    trian_fam = new coTetin__trianFam((int)0, (char **)0);
    strcpy(tetin_buf, buf);

    tetin1 = new coTetin();
    tetin1->append(trian_tol);
    tetin1->append(tetinf1);
    tetin1->append(trian_fam);

    const char *name = p_tetinObject->getObjName();
    DO_BinData *binObj = new DO_BinData((char *)name, tetin1);
    p_tetinObject->setCurrentObject(binObj);

    //===========================================================================
    // create output object for hexa modul
    // Send tetin, replay, configDir , solver name and case name
    //===========================================================================
    if (replayFile && configDir && solverName && caseName)
    {
        outp_intf_str = new char[strlen(solverName) + strlen(caseName) + 2];
        strcpy(outp_intf_str, solverName);
        strcat(outp_intf_str, "#");
        strcat(outp_intf_str, caseName);
        Covise::getname(buf, baseTetinName);
        Covise::getname(buf1, replayFile);
        Covise::getname(buf2, configDir);
        if (strlen(buf2) == 0)
        {
            sendWarning("configDir %s doesn't exist ", configDir);
            strcpy(buf2, configDir);
        }

        hexa = new coTetin__Hexa(buf, buf1, buf2, outp_intf_str);
        if (hexa)
        {
            tetin2 = new coTetin();
            tetin2->append(hexa);

            const char *name2 = p_hexaObject->getObjName();
            DO_BinData *binObj2 = new DO_BinData((char *)name2, tetin2);
            p_hexaObject->setCurrentObject(binObj2);
        }
        else
        {
            sendWarning("Can't create data object for HEXA");
            return STOP_PIPELINE;
        }
    }

    else
    {
        sendWarning("replayFile, configDir, solverName or caseName wrong");
        return STOP_PIPELINE;
    }

    //===========================================================================
    // create output object for modifyaddpart modul
    //===========================================================================
    tetinf2 = new coTetin__tetinFile(buf);
    //cerr << "Base tetin file named    : " << buf << endl;
    //cerr << "Solver & Case name is    : " << outp_intf_str << endl;
    sendInfo("Base tetin file named    :  %s\n", buf);
    sendInfo("Solver & Case name is    :  %s\n", outp_intf_str);
    coTetin__OutputInterf *outintinf = new coTetin__OutputInterf(outp_intf_str);
    tetin3 = new coTetin();
    tetin3->append(tetinf2);
    tetin3->append(outintinf);

    const char *name3 = p_addPartObject->getObjName();
    DO_BinData *binObj3 = new DO_BinData((char *)name3, tetin3);
    p_addPartObject->setCurrentObject(binObj3);

    delete[] outp_intf_str;

    return CONTINUE_PIPELINE;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// translate families
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int ModifyCabin::transGeom()
{
    coTetin *tetin1 = 0, *tetin2 = 0, *tetin3 = 0;
    coTetin__tetinFile *tetinf1, *tetinf2;
    coTetin__trianTol *trian_tol;
    coTetin__trianFam *trian_fam;
    coTetin__transGeom *trans_geom, *trans_geom2, *trans_geom3;
    coTetin__Hexa *hexa;
    char *outp_intf_str;
    float trans[3];

    float rot_matrix[3][3];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            rot_matrix[i][j] = ((i == j) ? 1.0 : 0.0);
        }
    }

    Covise::getname(buf, baseTetinName);

    //===========================================================================
    // create output object for tesselate modul
    // Send filename, triangulation tolerance and triangulation command
    //===========================================================================
    tetin1 = new coTetin();

    tetinf1 = new coTetin__tetinFile(buf);
    tetin1->append(tetinf1);

    trian_tol = new coTetin__trianTol(tolerance);
    tetin1->append(trian_tol);

    strcpy(tetin_buf, buf);
    for (i = 1; i < numMoveNames; i++)
    {
        float va = moveList[i].oldvalue;
        float vn = moveList[i].value;

        trans[0] = moveList[i].trans_vec[0] * (vn - va);
        trans[1] = moveList[i].trans_vec[1] * (vn - va);
        trans[2] = moveList[i].trans_vec[2] * (vn - va);
        trans_geom = new coTetin__transGeom(coTetin__transGeom::ALL_TRANS,
                                            1,
                                            moveList[i].numFamilies,
                                            moveList[i].familyNames,
                                            trans, rot_matrix);

        if (trans_geom)
            tetin1->append(trans_geom);
    }

    trian_fam = new coTetin__trianFam((int)0, (char **)0);
    tetin1->append(trian_fam);

    const char *name = p_tetinObject->getObjName();
    DO_BinData *binObj = new DO_BinData((char *)name, tetin1);
    p_tetinObject->setCurrentObject(binObj);

    fprintf(stderr, "ModifyCabin::transGeom TESS_COMMANDS CREATED\n");
    //===========================================================================
    // create output object for hexa modul
    //===========================================================================
    if (replayFile && configDir && solverName && caseName)
    {
        outp_intf_str = new char[strlen(solverName) + strlen(caseName) + 2];
        strcpy(outp_intf_str, solverName);
        strcat(outp_intf_str, "#");
        strcat(outp_intf_str, caseName);
        Covise::getname(buf, baseTetinName);
        Covise::getname(buf1, replayFile);
        Covise::getname(buf2, configDir);
        if (strlen(buf2) == 0)
        {
            sendWarning("configDir %s doesn't exist ", configDir);
            strcpy(buf2, configDir);
        }

        tetin2 = new coTetin();

        for (i = 1; i < numMoveNames; i++)
        {
            float va = moveList[i].oldvalue;
            float vn = moveList[i].value;

            trans[0] = moveList[i].trans_vec[0] * (vn - va);
            trans[1] = moveList[i].trans_vec[1] * (vn - va);
            trans[2] = moveList[i].trans_vec[2] * (vn - va);

            trans_geom2 = new coTetin__transGeom(coTetin__transGeom::ALL_TRANS,
                                                 1,
                                                 moveList[i].numFamilies,
                                                 moveList[i].familyNames,
                                                 trans, rot_matrix);

            if (trans_geom2)
                tetin2->append(trans_geom2);
        }

        hexa = new coTetin__Hexa(buf, buf1, buf2, outp_intf_str);
        if (hexa)
        {
            tetin2->append(hexa);

            const char *name2 = p_hexaObject->getObjName();
            DO_BinData *binObj2 = new DO_BinData((char *)name2, tetin2);
            p_hexaObject->setCurrentObject(binObj2);
            fprintf(stderr, "ModifyCabin::transGeom HEXA_COMMANDS CREATED\n");
        }

        else
        {
            sendWarning("Can't create data object for HEXA");
            return STOP_PIPELINE;
        }
    }

    else
    {
        sendWarning("replayFile, configDir, solverName or caseName wrong");
        return STOP_PIPELINE;
    }

    //===========================================================================
    // create output object for modifyaddpart modul
    //===========================================================================
    tetinf2 = new coTetin__tetinFile(buf);
    //cerr << "Base tetin file named    : " << buf << endl;
    //cerr << "Solver & Case name is    : " << outp_intf_str << endl;
    sendInfo("Base tetin file named    :  %s\n", buf);
    sendInfo("Solver & Case name is    :  %s\n", outp_intf_str);
    coTetin__OutputInterf *outintinf = new coTetin__OutputInterf(outp_intf_str);
    tetin3 = new coTetin();
    tetin3->append(tetinf2);
    tetin3->append(outintinf);
    for (i = 1; i < numMoveNames; i++)
    {
        float va = moveList[i].oldvalue;
        float vn = moveList[i].value;

        trans[0] = moveList[i].trans_vec[0] * (vn - va);
        trans[1] = moveList[i].trans_vec[1] * (vn - va);
        trans[2] = moveList[i].trans_vec[2] * (vn - va);

        trans_geom3 = new coTetin__transGeom(coTetin__transGeom::ALL_TRANS,
                                             1,
                                             moveList[i].numFamilies,
                                             moveList[i].familyNames,
                                             trans, rot_matrix);

        if (trans_geom3)
            tetin3->append(trans_geom3);

        // reset old values
        moveList[i].oldvalue = moveList[i].value;
    }
    const char *name3 = p_addPartObject->getObjName();
    DO_BinData *binObj3 = new DO_BinData((char *)name3, tetin3);
    p_addPartObject->setCurrentObject(binObj3);
    fprintf(stderr, "ModifyCabin::transGeom ADDPART_COMMANDS CREATED\n");

    delete[] outp_intf_str;

    return CONTINUE_PIPELINE;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// reset translations
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int ModifyCabin::resetGeom()
{
    coTetin *tetin = 0, *tetin2 = 0, *tetin3 = 0;
    coTetin__tetinFile *tetinf, *tetinf2;
    coTetin__trianTol *trian_tol;
    coTetin__trianFam *trian_fam;
    coTetin__transGeom *trans_geom, *trans_geom2, *trans_geom3;
    coTetin__Hexa *hexa;
    char *outp_intf_str;
    int k;

    float trans[3];
    float rot_matrix[3][3];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            rot_matrix[i][j] = ((i == j) ? 1.0 : 0.0);
        }
    }

    Covise::getname(buf, baseTetinName);
    tetinf = new coTetin__tetinFile(buf);
    trian_tol = new coTetin__trianTol(tolerance);
    trian_fam = new coTetin__trianFam((int)0, (char **)0);
    strcpy(tetin_buf, buf);

    //===========================================================================
    // make output object for tesselate
    //===========================================================================
    tetin = new coTetin();

    // set tol for all following commands
    tetin->append(trian_tol);

    // set tetin file
    tetin->append(tetinf);

    for (k = 0; k < numMoveNames; k++)
    {
        if (moveList[k].value != 0.0)
        {
            trans[0] = -moveList[k].trans_vec[0] * moveList[k].value;
            trans[1] = -moveList[k].trans_vec[1] * moveList[k].value;
            trans[2] = -moveList[k].trans_vec[2] * moveList[k].value;
            trans_geom = new coTetin__transGeom(coTetin__transGeom::ALL_TRANS,
                                                1,
                                                moveList[k].numFamilies,
                                                moveList[k].familyNames,
                                                trans, rot_matrix);

            if (trans_geom)
                tetin->append(trans_geom);
        }
    }

    // set triangulation command
    tetin->append(trian_fam);

    const char *name = p_tetinObject->getObjName();
    DO_BinData *binObj = new DO_BinData((char *)name, tetin);
    p_tetinObject->setCurrentObject(binObj);

    //===========================================================================
    // create output object for hexa modul
    //===========================================================================
    outp_intf_str = new char[strlen(solverName) + strlen(caseName) + 2];
    strcpy(outp_intf_str, solverName);
    strcat(outp_intf_str, "#");
    strcat(outp_intf_str, caseName);
    Covise::getname(buf, baseTetinName);
    Covise::getname(buf1, replayFile);
    Covise::getname(buf2, configDir);
    if (strlen(buf2) == 0)
    {
        sendWarning("configDir %s doesn't exist ", configDir);
        strcpy(buf2, configDir);
    }

    hexa = new coTetin__Hexa(buf, buf1, buf2, outp_intf_str);

    if (hexa)
    {
        tetin2 = new coTetin();

        for (k = 0; k < numMoveNames; k++)
        {
            if (moveList[k].value != 0.0)
            {
                trans[0] = -moveList[k].trans_vec[0] * moveList[k].value;
                trans[1] = -moveList[k].trans_vec[1] * moveList[k].value;
                trans[2] = -moveList[k].trans_vec[2] * moveList[k].value;
                trans_geom2 = new coTetin__transGeom(coTetin__transGeom::ALL_TRANS,
                                                     1,
                                                     moveList[k].numFamilies,
                                                     moveList[k].familyNames,
                                                     trans, rot_matrix);

                if (trans_geom2)
                    tetin->append(trans_geom2);
            }
        }

        tetin2->append(hexa);

        const char *name2 = p_hexaObject->getObjName();
        DO_BinData *binObj2 = new DO_BinData((char *)name2, tetin2);
        p_hexaObject->setCurrentObject(binObj2);
    }

    else
    {
        sendWarning("Can't create data object for HEXA");
        return STOP_PIPELINE;
    }

    //===========================================================================
    // create output object for modifyaddpart modul
    //===========================================================================
    tetinf2 = new coTetin__tetinFile(buf);
    //cerr << "Base tetin file named    : " << buf << endl;
    //cerr << "Solver & Case name is    : " << outp_intf_str << endl;
    sendInfo("Base tetin file named    :  %s\n", buf);
    sendInfo("Solver & Case name is    :  %s\n", outp_intf_str);
    coTetin__OutputInterf *outintinf = new coTetin__OutputInterf(outp_intf_str);

    tetin3 = new coTetin();
    tetin3->append(tetinf2);
    tetin3->append(outintinf);

    for (k = 0; k < numMoveNames; k++)
    {
        if (moveList[k].value != 0.0)
        {
            trans[0] = -moveList[k].trans_vec[0] * moveList[k].value;
            trans[1] = -moveList[k].trans_vec[1] * moveList[k].value;
            trans[2] = -moveList[k].trans_vec[2] * moveList[k].value;

            trans_geom3 = new coTetin__transGeom(coTetin__transGeom::ALL_TRANS,
                                                 1,
                                                 moveList[currMoveName].numFamilies,
                                                 moveList[currMoveName].familyNames,
                                                 trans, rot_matrix);
            if (trans_geom3)
                tetin3->append(trans_geom3);
        }
    }

    const char *name3 = p_addPartObject->getObjName();
    DO_BinData *binObj3 = new DO_BinData((char *)name3, tetin3);
    p_addPartObject->setCurrentObject(binObj3);

    delete[] outp_intf_str;

    for (k = 0; k < numMoveNames; k++)
    {
        moveList[k].value = 0.;
        moveList[k].oldvalue = 0.;
    }
    return CONTINUE_PIPELINE;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// project curves
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int ModifyCabin::projCurves(coTetin *tetinAll, char *tetin_name)
{
    coTetin *tetin = 0;
    coTetin *tetinPM = 0;
    coTetin__Proj *Proj = 0;
    float *points_x, *points_y, *points_z;
    int Res = 0, n_points;
    int i, j;

    float rot_matrix[3][3];
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            rot_matrix[i][j] = ((i == j) ? 1.0 : 0.0);
        }
    }

    float trans[3];
    float va = projList[currProjName].oldvalue;
    float vn = projList[currProjName].value;

    trans[0] = projList[currProjName].trans_vec[0] * (vn - va);
    trans[1] = projList[currProjName].trans_vec[1] * (vn - va);
    trans[2] = projList[currProjName].trans_vec[2] * (vn - va);
    cerr << "************** translate curves "
         << trans[0] << " "
         << trans[1] << " "
         << trans[2] << " " << endl;

    for (i = 0; i < projList[currProjName].numCurves; i++)
    {
        // translate curves
        coTetin__transGeom *trans_geom2 = new coTetin__transGeom(coTetin__transGeom::CURVE_TRANS,
                                                                 0,
                                                                 1,
                                                                 &projList[currProjName].curveNames[i],
                                                                 trans, rot_matrix);

        //get approximated curves
        coTetin__apprxCurve *apprxCurve = 0;
        apprxCurve = new coTetin__apprxCurve(0, 1, &projList[currProjName].curveNames[i]);
        coTetin__tetinFile *tetinf = new coTetin__tetinFile(tetin_name);
        if (apprxCurve && tetinf)
        {
            tetin = new coTetin();
            tetin->append(tetinf);
            if (trans_geom2)
                tetin->append(trans_geom2);
            tetin->append(apprxCurve);
        }

        else
            return STOP_PIPELINE;

        if (tetin)
        {
            Res = pj_covise_tetin(tetin);
            delete tetin;
            if (Res != 0)
            {
                if (Res == -999)
                    sendWarning(" No Projection license");

                else
                    sendError("Error in projection library");

                return STOP_PIPELINE;
            }
        }

        else
            return STOP_PIPELINE;

        // get the points of the approx. curves
        pj_covise_get_apprx_curve_pnts(&points_x, &points_y,
                                       &points_z, &n_points);

        // project approximated points
        // take always nearest point
        float *direct = 0;

        // point feld muss gemacht werden
        float *points = new float[3 * n_points];
        int ip = 0;
        for (int k = 0; k < n_points; k++)
        {
            points[ip] = points_x[k];
            points[ip + 1] = points_y[k];
            points[ip + 2] = points_z[k];
            ip += 3;
        }

        pj_covise_init_apprx_curve_pnts();

        Proj = new coTetin__Proj(n_points, points, direct, 1, &projList[currProjName].projName);
        tetinf = new coTetin__tetinFile(tetin_name);
        if (Proj && tetinf)
        {
            tetin = new coTetin();
            tetin->append(tetinf);
            tetin->append(Proj);
            delete[] points;
        }

        else
            return STOP_PIPELINE;

        if (tetin)
        {
            Res = pj_covise_tetin(tetin);
            delete tetin;
            if (Res != 0)
            {
                if (Res == -999)
                    sendWarning(" No Projection license");

                else
                    sendError("Error in projection library");

                return STOP_PIPELINE;
            }
        }

        else
            return STOP_PIPELINE;

        // get projected points
        pj_covise_get_projected_points(&points_x, &points_y,
                                       &points_z, &n_points);

        // replace points in TETIN file
        char *cname = new char[strlen(projList[currProjName].curveNames[i]) + 1];
        strcpy(cname, projList[currProjName].curveNames[i]);
        coTetin__defCurve *curvesAll = new coTetin__defCurve(points_x, points_y, points_z,
                                                             n_points, cname);
        coTetin__defCurve *curves = new coTetin__defCurve(points_x, points_y, points_z,
                                                          n_points, cname);

        pj_covise_free_projected_points();
        delete[] cname;

        if (curvesAll)
            tetinAll->append(curvesAll);

        else
            return STOP_PIPELINE;

        if (curves)
        {
            if (!tetinPM)
                tetinPM = new coTetin();

            tetinPM->append(curves);
        }
        else
            return STOP_PIPELINE;
    }

    if (tetinPM)
    {
        Res = pj_covise_tetin(tetinPM);
        delete tetinPM;
        if (Res != 0)
        {
            if (Res == -999)
                sendWarning(" No Projection license");

            else
                sendError("Error in projection library");

            return STOP_PIPELINE;
        }
    }

    return CONTINUE_PIPELINE;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// project precribed points
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int ModifyCabin::projPoints(coTetin *tetinAll, char *tetin_name)
{
    coTetin *tetin = 0;
    coTetin *tetinPM = 0;
    coTetin__Proj *Proj = 0;
    float *points_x, *points_y, *points_z;
    int Res = 0, n_points;
    int i, j;

    float rot_matrix[3][3];
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            rot_matrix[i][j] = ((i == j) ? 1.0 : 0.0);
        }
    }

    float trans[3];
    float va = projList[currProjName].oldvalue;
    float vn = projList[currProjName].value;

    trans[0] = projList[currProjName].trans_vec[0] * (vn - va);
    trans[1] = projList[currProjName].trans_vec[1] * (vn - va);
    trans[2] = projList[currProjName].trans_vec[2] * (vn - va);

    cerr << "************** translate prescribed points "
         << trans[0] << " "
         << trans[1] << " "
         << trans[2] << " " << endl;

    // translate prescribed points
    coTetin__transGeom *trans_geom2 = new coTetin__transGeom(coTetin__transGeom::PPOINT_TRANS,
                                                             1,
                                                             1,
                                                             &projList[currProjName].familyName,
                                                             trans, rot_matrix);

    //get prescribed points
    coTetin__getprescPnt *prescPoints = 0;
    prescPoints = new coTetin__getprescPnt(1, 1, &projList[currProjName].familyName);

    coTetin__tetinFile *tetinf = new coTetin__tetinFile(tetin_name);
    if (prescPoints && tetinf)
    {
        tetin = new coTetin();
        tetin->append(tetinf);
        if (trans_geom2)
            tetin->append(trans_geom2);
        tetin->append(prescPoints);
    }

    else
        return STOP_PIPELINE;

    if (tetin)
    {
        Res = pj_covise_tetin(tetin);
        delete tetin;
        if (Res != 0)
        {
            if (Res == -999)
                sendWarning(" No Projection license");

            else
                sendError("Error in projection library");

            return STOP_PIPELINE;
        }
    }

    else
        return STOP_PIPELINE;

    // get the points
    char **pntNames;
    pj_covise_get_prescr_pnts(&points_x, &points_y, &points_z,
                              &pntNames, &n_points);

    // project prescribed points
    // take always nearest point
    float *direct = 0;

    // point feld muss gemacht werden
    float *points = new float[3 * n_points];
    int ip = 0;
    for (int k = 0; k < n_points; k++)
    {
        points[ip] = points_x[k];
        points[ip + 1] = points_y[k];
        points[ip + 2] = points_z[k];
        ip += 3;
    }

    Proj = new coTetin__Proj(n_points, points, direct, 1, &projList[currProjName].projName);
    tetinf = new coTetin__tetinFile(tetin_name);
    if (Proj && tetinf)
    {
        tetin = new coTetin();
        tetin->append(tetinf);
        tetin->append(Proj);
        delete[] points;
    }

    else
        return STOP_PIPELINE;

    if (tetin)
    {
        Res = pj_covise_tetin(tetin);
        delete tetin;
        if (Res != 0)
        {
            if (Res == -999)
                sendWarning(" No Projection license");

            else
                sendError("Error in projection library");

            return STOP_PIPELINE;
        }
    }

    else
        return STOP_PIPELINE;

    // get projected points and replace them
    pj_covise_get_projected_points(&points_x, &points_y,
                                   &points_z, &n_points);

    // replace prescribed points
    for (i = 0; i < n_points; i++)
    {
        coTetin__prescPnt *pntsAll = new coTetin__prescPnt(points_x[i], points_y[i], points_z[i], pntNames[i]);
        coTetin__prescPnt *pnts = new coTetin__prescPnt(points_x[i], points_y[i], points_z[i], pntNames[i]);
        if (pntsAll)
            tetinAll->append(pntsAll);

        else
            return STOP_PIPELINE;

        if (pnts)
        {
            if (!tetinPM)
                tetinPM = new coTetin();

            tetinPM->append(pnts);
        }
        else
            return STOP_PIPELINE;
    }

    pj_covise_init_prescr_pnts();
    pj_covise_free_projected_points();

    if (tetinPM)
    {
        Res = pj_covise_tetin(tetinPM);
        delete tetinPM;
        if (Res != 0)
        {
            if (Res == -999)
                sendWarning(" No Projection license");

            else
                sendError("Error in projection library");

            return STOP_PIPELINE;
        }
    }

    return CONTINUE_PIPELINE;
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

int ModifyCabin::compute()
{
    coTetin *tetin = 0;
    int ierr;

    //===================================================================
    // show basis tetin file
    //===================================================================
    if (currMoveName == 0 && currProjName == 0)
        command = SEND_BASE_TETIN;

    //fprintf(stderr,"command = [%d]\n", command);

    if (command == SEND_BASE_TETIN)
    {
        if (!baseTetinName)
        {
            sendError("Need tetin base file ");
            return STOP_PIPELINE;
        }

        ierr = sendBaseInfo();
        if (ierr == STOP_PIPELINE)
            return STOP_PIPELINE;
    }

    //===================================================================
    // show basis tetin file with one translation of a family
    //===================================================================
    if (command == TRANSLATE_GEOM)
    {
        if (!baseTetinName || !configFileName)
        {
            sendError("Need base and config file before translation");
            return STOP_PIPELINE;
        }

        ierr = transGeom();
        if (ierr == STOP_PIPELINE)
            return STOP_PIPELINE;
    }

    //===================================================================
    // show basis tetin file
    //===================================================================
    if (command == PROJ_CURVES)
    {
        if (!baseTetinName || !configFileName)
        {
            sendError("Need base and config file before translation");
            return STOP_PIPELINE;
        }

        // make a approximate output for tesselate
        // Send filename, triangulation tolerance and triangulation command
        //=========================================================================
        float rot_matrix[3][3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                rot_matrix[i][j] = ((i == j) ? 1.0 : 0.0);
            }
        }

        getname(buf, baseTetinName);
        coTetin__tetinFile *tetinf = new coTetin__tetinFile(buf);
        coTetin__trianTol *trian_tol = new coTetin__trianTol(tolerance);
        coTetin__trianFam *trian_fam = new coTetin__trianFam((int)0, (char **)0);
        strcpy(tetin_buf, buf);

        float trans[3];
        float va = projList[currProjName].oldvalue;
        float vn = projList[currProjName].value;

        trans[0] = projList[currProjName].trans_vec[0] * (vn - va);
        trans[1] = projList[currProjName].trans_vec[1] * (vn - va);
        trans[2] = projList[currProjName].trans_vec[2] * (vn - va);

        cerr << "************** translate surface "
             << trans[0] << " "
             << trans[1] << " "
             << trans[2] << " " << endl;

        coTetin__transGeom *trans_geom = new coTetin__transGeom(coTetin__transGeom::ALL_TRANS,
                                                                1, 1,
                                                                &projList[currProjName].familyName,
                                                                trans, rot_matrix);

        // create output object for tesselate modul
        if (tetinf)
        {
            tetin = new coTetin();
            tetin->append(trian_tol);
            tetin->append(tetinf);
            if (trans_geom)
                tetin->append(trans_geom);
            tetin->append(trian_fam);

            const char *name = p_tetinObject->getObjName();
            DO_BinData *binObj = new DO_BinData((char *)name, tetin);
            p_tetinObject->setCurrentObject(binObj);
        }

        else
        {
            sendWarning("Can't create data object for TESSELATE");
            return STOP_PIPELINE;
        }

        // make a the output objects for hexa
        //=========================================================================

        char *outp_intf_str = new char[strlen(solverName) + strlen(caseName) + 2];
        strcpy(outp_intf_str, solverName);
        strcat(outp_intf_str, "#");
        strcat(outp_intf_str, caseName);
        Covise::getname(buf, baseTetinName);
        Covise::getname(buf1, replayFile);
        Covise::getname(buf2, configDir);
        if (strlen(buf2) == 0)
        {
            sendWarning("configDir %s doesn't exist ", configDir);
            strcpy(buf2, configDir);
        }

        coTetin__Hexa *hexa = new coTetin__Hexa(buf, buf1, buf2, outp_intf_str);
        delete[] outp_intf_str;

        Covise::getname(buf, baseTetinName);
        tetinf = new coTetin__tetinFile(buf);
        strcpy(tetin_buf, buf);

        // loop over all curves that have to be projected
        coTetin *tetinAll = new coTetin();
        tetinAll->append(tetinf);

        ierr = projCurves(tetinAll, tetin_buf);
        if (ierr == STOP_PIPELINE)
            return STOP_PIPELINE;

        ierr = projPoints(tetinAll, tetin_buf);
        if (ierr == STOP_PIPELINE)
            return STOP_PIPELINE;

        tetinAll->append(hexa);
        const char *name2 = p_hexaObject->getObjName();
        DO_BinData *binObj2 = new DO_BinData((char *)name2, tetinAll);
        p_hexaObject->setCurrentObject(binObj2);

        // reset old values
        projList[currProjName].oldvalue = projList[currProjName].value;
    }

    //===================================================================
    // reset all translations
    //===================================================================
    if (command == TRANSLATE_RESET)
    {
        if (!baseTetinName || !configFileName)
        {
            sendError("Need base and config file before translation");
            return STOP_PIPELINE;
        }

        ierr = resetGeom();
        currMoveName = 0;
        if (ierr == STOP_PIPELINE)
            return STOP_PIPELINE;
    }

    // in any case create the feedback object
    const char *feedbackPortName = p_feedback->getObjName();
    if (feedbackPortName)
    {
        float px[1], py[1], pz[1];
        px[0] = py[0] = pz[0] = 0.0;
        coDistributedObject *dummyPoint = new coDoPoints(feedbackPortName, 1, px, py, pz);

        coFeedback feedback("ModifyCabin");

        if (newGeom)
            feedback.addString("NEWGEOM");
        else
            feedback.addString("OLDGEOM");
        for (int i = 0; i < numMoveNames; i++)
        {
            char str[1024];
            sprintf(str, "%s %f %f %f %f %f %f %d", moveNames[i],
                    moveList[i].trans_vec[0], moveList[i].trans_vec[1], moveList[i].trans_vec[2],
                    moveList[i].dmin, moveList[i].dmax, moveList[i].value, moveList[i].numFamilies);
            for (int j = 0; j < moveList[i].numFamilies; j++)
            {
                char tmp[300];
                sprintf(tmp, " %s", moveList[i].familyNames[j]);
                strcat(str, tmp);
            }
            //fprintf(stderr,"feedback Str=[%s]\n", str);
            feedback.addString(str);
        }

        feedback.addPara(p_moveName);
        feedback.apply(dummyPoint);
        p_feedback->setCurrentObject(dummyPoint);
    }
    newGeom = false;
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

void ModifyCabin::param(const char *name)
{
    FILE *fp;
    char *token[20];
    int irest, tmax = 20, ip, i;
    const char *text;

    // check whether p_baseTetin
    if (strcmp(name, p_baseTetin->getName()) == 0)
    {
        text = p_baseTetin->getValue();
        if (baseTetinName)
            delete[] baseTetinName;
        baseTetinName = new char[strlen(text) + 1];
        strcpy(baseTetinName, text);
        command = SEND_BASE_TETIN;
        newGeom = true;

        return;
    }

    // check whether p_configDir
    if (strcmp(name, p_configDir->getName()) == 0)
    {
        text = p_configDir->getValue();
        if (configDir)
            delete[] configDir;
        configDir = new char[strlen(text) + 1];
        strcpy(configDir, text);
        return;
    }

    // check whether p_replayFile
    if (strcmp(name, p_replayFile->getName()) == 0)
    {
        text = p_replayFile->getValue();
        if (replayFile)
            delete[] replayFile;
        replayFile = new char[strlen(text) + 1];
        strcpy(replayFile, text);
        return;
    }

    // read a new tolerance
    if (strcmp(name, p_tolerance->getName()) == 0)
    {
        tolerance = p_tolerance->getValue();
        return;
    }

    // we select another solver
    if (strcmp(name, p_solver->getName()) == 0)
    {
        int pp = p_solver->getValue();
        solverName = outp_intf[pp];
        return;
    }

    // we set a new casename
    if (strcmp(name, p_caseName->getName()) == 0)
    {
        text = p_caseName->getValue();
        if (caseName)
            delete[] caseName;
        caseName = new char[strlen(text) + 1];
        strcpy(caseName, text);
        return;
    }

    // we reset an action
    if ((strcmp(name, p_reset->getName()) == 0) && (!Covise::in_map_loading()))
    {
        if (configFileName)
        {
            resetName = p_reset->getValue();
            if (resetName == 0)
            {
                command = TRANSLATE_RESET;
                p_moveName->setValue(numMoveNames, moveNames, 0);
                p_moveName->show();
            }

            else
            {
                command = PROJ_RESET;
                sendError("The reset of  projection is not yet implemented\n");
                currProjName = 0;
                p_projName->setValue(numProjNames, projNames, 0);
            }
        }

        else
            sendWarning("Please select an configFile first\n");

        return;
    }

    // we select another moveable family
    if (strcmp(name, p_moveName->getName()) == 0)
    {
        if (configFileName)
        {
            // find out which family we chose
            currMoveName = p_moveName->getValue();

            // set the proper values for it
            p_trans_value->setValue(moveList[currMoveName].dmin,
                                    moveList[currMoveName].dmax,
                                    moveList[currMoveName].value);
            if (currMoveName != 0)
                command = TRANSLATE_GEOM;
        }

        else
            sendWarning("Please select an configFile first %s\n");

        return;
    }

    // we select another projection familyconfigFile
    if (strcmp(name, p_projName->getName()) == 0)
    {
        if (configFileName)
        {
            // find out which family we chose
            currProjName = p_projName->getValue();

            // set the proper values for it
            p_direction->setValue(projList[currProjName].dmin,
                                  projList[currProjName].dmax,
                                  projList[currProjName].value);
            if (currProjName != 0)
                command = PROJ_CURVES;
        }

        else
            sendWarning("Please select an configFile first %s\n");

        return;
    }

    // we select another translation value
    if (strcmp(name, p_trans_value->getName()) == 0)
    {
        if (configFileName)
        {
            moveList[currMoveName].dmin = p_trans_value->getMin();
            moveList[currMoveName].dmax = p_trans_value->getMax();
            moveList[currMoveName].value = p_trans_value->getValue();
            //fprintf(stderr,"ModifyCabin::param update trans of [%d] to [%f]\n", currMoveName, p_trans_value->getValue());
            if (currMoveName != 0)
                command = TRANSLATE_GEOM;
        }

        else
        {
            sendWarning("Please select an configFile first %s\n");
        }

        return;
    }

    // we select another projection translation value
    if (strcmp(name, p_direction->getName()) == 0)
    {
        if (configFileName)
        {
            projList[currProjName].dmin = p_direction->getMin();
            projList[currProjName].dmax = p_direction->getMax();
            projList[currProjName].value = p_direction->getValue();
            if (currProjName != 0)
                command = PROJ_CURVES;
        }

        else
            sendWarning("Please select an configFile first %s\n");

        return;
    }

    // config file tells us which families can be modified or projected
    if (strcmp(name, p_configFile->getName()) == 0)
    {
        newGeom = true;
        // if a new config is read destroy old lists
        if (configFileName != NULL)
        {
            for (i = 1; i < numMoveNames; i++)
            {
                delete[] moveList[i].name;
                for (int j = 0; j < moveList[i].numFamilies; j++)
                    delete[] moveList[i].familyNames[j];
            }
            numMoveNames = 1;

            for (i = 1; i < numProjNames; i++)
            {
                delete[] projList[i].name;
                delete[] projList[i].projName;
                delete[] projList[i].familyName;
                for (int j = 0; j < projList[i].numCurves; j++)
                    delete[] projList[i].curveNames[j];
            }
            numProjNames = 1;

            delete[] configFileName;
        }

        text = p_configFile->getValue();
        configFileName = new char[strlen(text) + 1];
        strcpy(configFileName, text);

        // open file
        if ((fp = fopen(configFileName, "r")) == NULL)
        {
            sendWarning("Could not open file %s", configFileName);
            configFileName = NULL;
            return;
        }

        while (fgets(buf, sizeof(buf), fp) != NULL)
        {
            int ntok = parser(buf, token, tmax, " 	;,\n");
            if (ntok != 0)
            {
                if (strcmp(token[0], "MOVE") == 0)
                {
                    if (ntok < 6)
                    {
                        sendWarning("Too few items in config file  %s\n");
                    }

                    else
                    {
                        // list name
                        ip = 1;
                        moveList[numMoveNames].name = new char[strlen(token[ip]) + 1];
                        strcpy((char *)moveList[numMoveNames].name, token[ip]);
                        ip++;

                        // set the translation vector
                        // fixed translation in x direction
                        moveList[numMoveNames].trans_vec[0] = 1.;
                        moveList[numMoveNames].trans_vec[1] = 0.;
                        moveList[numMoveNames].trans_vec[2] = 0.;

                        // slider values;
                        moveList[numMoveNames].dmin = atof(token[ip]);
                        ip++;
                        moveList[numMoveNames].dmax = atof(token[ip]);
                        ip++;
                        moveList[numMoveNames].value = atof(token[ip]);
                        ip++;
                        moveList[numMoveNames].oldvalue = moveList[numMoveNames].value;

                        // family names
                        irest = ntok - 5;
                        moveList[numMoveNames].numFamilies = irest;
                        moveList[numMoveNames].familyNames = new const char *[irest];

                        for (i = 0; i < irest; i++)
                        {
                            moveList[numMoveNames].familyNames[i] = new char[strlen(token[ip]) + 1];
                            strcpy((char *)moveList[numMoveNames].familyNames[i], token[ip]);
                            ip++;
                        }

                        // new value
                        numMoveNames++;
                    }
                }

                else if (strcmp(token[0], "PROJECT") == 0)
                {
                    if (ntok < 8)
                    {
                        sendWarning("Too few items in config file  %s\n");
                    }

                    else
                    {
                        // list name
                        ip = 1;
                        projList[numProjNames].name = new char[strlen(token[ip]) + 1];
                        strcpy((char *)projList[numProjNames].name, token[ip]);
                        //cerr << "Button name:  " << projList[numProjNames].name << endl;
                        ip++;

                        // projection family name
                        projList[numProjNames].familyName = new char[strlen(token[ip]) + 1];
                        strcpy((char *)projList[numProjNames].familyName, token[ip]);
                        //cerr << "Family name:  " << projList[numProjNames].familyName << endl;
                        ip++;

                        // to projection family name
                        projList[numProjNames].projName = new char[strlen(token[ip]) + 1];
                        strcpy((char *)projList[numProjNames].projName, token[ip]);
                        //cerr << "Proj.  name:  " << projList[numProjNames].projName << endl;
                        ip++;

                        // set the translation vector
                        // fixed translation in y direction
                        projList[numProjNames].trans_vec[0] = 0.;
                        projList[numProjNames].trans_vec[1] = 1.;
                        projList[numProjNames].trans_vec[2] = 0.;

                        // slider values;
                        projList[numProjNames].dmin = atof(token[ip]);
                        ip++;
                        projList[numProjNames].dmax = atof(token[ip]);
                        ip++;
                        projList[numProjNames].value = atof(token[ip]);
                        ip++;
                        projList[numProjNames].oldvalue = projList[numProjNames].value;

                        // curve names
                        irest = ntok - 7;
                        projList[numProjNames].numCurves = irest;
                        projList[numProjNames].curveNames = new const char *[irest];

                        for (i = 0; i < irest; i++)
                        {
                            projList[numProjNames].curveNames[i] = new char[strlen(token[ip]) + 1];
                            strcpy((char *)projList[numProjNames].curveNames[i], token[ip]);
                            //cerr << "Curve name:  " << projList[numProjNames].curveNames[i] << endl;
                            ip++;
                        }

                        // new value
                        numProjNames++;
                    }
                }

                else
                {
                    //sendWarning("Unknown keyword in config file %s", token[0]);
                }
            }
        }

        if (fp != NULL)
            fclose(fp);

        // set the default family name and variables
        int i;
        for (i = 0; i < numMoveNames; i++)
        {
            moveNames[i] = moveList[i].name;
        }
        p_moveName->setValue(numMoveNames, moveNames, 0);
        p_moveName->show();
        p_trans_value->show();

        // set the default family name and variables
        for (i = 0; i < numProjNames; i++)
        {
            projNames[i] = projList[i].name;
        }
        p_projName->setValue(numProjNames, projNames, 0);
        p_projName->show();
        p_direction->show();

        p_reset->show();

        return;
    }
}

int ModifyCabin::parser(char *line, char *token[], int tmax, char *sep)
{
    char *tp;
    int count;

    count = 0;
    tp = strtok(line, sep);
    for (count = 0; count < tmax && tp != NULL;)
    {
        token[count] = tp;
        tp = strtok(NULL, sep);
        count++;
    }
    token[count] = NULL;
    return count;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Quit callback: as the name tells...
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void ModifyCabin::quit()
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

void ModifyCabin::postInst()
{
    p_baseTetin->show();
    p_configFile->show();
    p_replayFile->show();
    p_configDir->show();
    p_solver->show();
    p_tolerance->show();
    p_caseName->show();
    //p_moveName->hide();
    //p_projName->hide();
    //p_reset->hide();
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
    ModifyCabin *application = new ModifyCabin;

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
