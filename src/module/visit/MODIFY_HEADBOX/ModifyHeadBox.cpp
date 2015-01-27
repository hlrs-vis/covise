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
#include <api/coFeedback.h>
#include "ModifyHeadBox.h"

void close_group();
int open_group();
int read_string();
void read_dims();
void read_nodes();
void read_faces();
void read_spaces();
int read_hex();

ifstream ifile;
float *xs, *ys, *zs;
int *pl, *cl, pls, cls, xss, dims, nps, ncs, nxs;

float xCoords[4] = { -1.0e-3, 1.0e-3, 1.0e-3, -1.0e-3 };
float yCoords[4] = { -1.0, -1.0, 1.0, 1.0 };
float zCoords[4] = { 0.0, 0.0, 0.0, 0.0 };
int vertexList[4] = { 0, 1, 2, 3 };
int polygonList[1] = { 0 };

char buf[256], buf1[256], buf2[256], buf3[256];

// this one will be in coModule of next API release

void ModifyHeadBox::selfExec()
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

ModifyHeadBox::ModifyHeadBox()
{

    // declare the name of our module
    set_module_description("Modify Valmet's Headbox");

    p_user = addStringParam("user", "User name");
    p_user->setValue("auj");
    user = NULL;

    p_host = addStringParam("host", "Host");
    p_host->setValue("cfd.ux.phys.jyu.fi");
    host = NULL;

    p_dir = addStringParam("dir", "Directory");
    p_dir->setValue("/users/auj/wrk0/visit/fluent/visit/modify_headbox");
    dir = NULL;

    p_run = addStringParam("run", "shell script");
    p_run->setValue("./modifygambit");
    run = NULL;

    p_geom = addOutputPort("polygon_set", "coDoGeometry", "faces of the mesh as a set of polygons");

    pl = NULL;
    cl = NULL;
    xs = NULL;
    ys = NULL;
    zs = NULL;
    pls = 0;
    cls = 0;
    xss = 0;
    nps = 0;
    ncs = 0;
    nxs = 0;

    surface = NULL;

    flag = -1;

    // MUUTA: kullekin liulle muutujat:
    // trn : nykyinen arvo
    // otrn : vanha arvo
    dmin1 = 300;
    dmax1 = 1200;
    trn1 = 561;
    otrn1 = 561;

    dmin2 = 300;
    dmax2 = 1200;
    trn2 = 561;
    otrn2 = 561;

    dmin3 = 300;
    dmax3 = 1200;
    trn3 = 519;
    otrn3 = 519;

    dmin4 = 300;
    dmax4 = 1200;
    trn4 = 416;
    otrn4 = 416;

    dmin5 = 300;
    dmax5 = 1200;
    trn5 = 416;
    otrn5 = 416;

    dmin6 = 300;
    dmax6 = 1200;
    trn6 = 374;
    otrn6 = 374;

    dmin7 = 100;
    dmax7 = 1000;
    trn7 = 230;
    otrn7 = 230;

    dmin8 = 100;
    dmax8 = 1000;
    trn8 = 230;
    otrn8 = 230;

    dmin9 = 100;
    dmax9 = 1000;
    trn9 = 230;
    otrn9 = 230;

    dmin10 = 10;
    dmax10 = 500;
    trn10 = 90;
    otrn10 = 90;

    dmin11 = 10;
    dmax11 = 500;
    trn11 = 90;
    otrn11 = 90;

    dmin12 = 10;
    dmax12 = 500;
    trn12 = 90;
    otrn12 = 90;

    p_trans_value1 = addFloatSliderParam("point1_x", "Set the translation values");
    p_trans_value1->setImmediate(0);
    p_trans_value1->setValue(dmin1, dmax1, trn1);

    p_trans_value2 = addFloatSliderParam("point2_x", "Set the translation values");
    p_trans_value2->setImmediate(0);
    p_trans_value2->setValue(dmin2, dmax2, trn2);

    p_trans_value3 = addFloatSliderParam("point2_z", "Set the translation values");
    p_trans_value3->setImmediate(0);
    p_trans_value3->setValue(dmin3, dmax3, trn3);

    p_trans_value4 = addFloatSliderParam("point3_x", "Set the translation values");
    p_trans_value4->setImmediate(0);
    p_trans_value4->setValue(dmin4, dmax4, trn4);

    p_trans_value5 = addFloatSliderParam("point4_x", "Set the translation values");
    p_trans_value5->setImmediate(0);
    p_trans_value5->setValue(dmin5, dmax5, trn5);

    p_trans_value6 = addFloatSliderParam("point4_z", "Set the translation values");
    p_trans_value6->setImmediate(0);
    p_trans_value6->setValue(dmin6, dmax6, trn6);

    p_trans_value7 = addFloatSliderParam("point5_x", "Set the translation values");
    p_trans_value7->setImmediate(0);
    p_trans_value7->setValue(dmin7, dmax7, trn7);

    p_trans_value8 = addFloatSliderParam("point6_x", "Set the translation values");
    p_trans_value8->setImmediate(0);
    p_trans_value8->setValue(dmin8, dmax8, trn8);

    p_trans_value9 = addFloatSliderParam("point6_z", "Set the translation values");
    p_trans_value9->setImmediate(0);
    p_trans_value9->setValue(dmin9, dmax9, trn9);

    p_trans_value10 = addFloatSliderParam("point7_z", "Set the translation values");
    p_trans_value10->setImmediate(0);
    p_trans_value10->setValue(dmin10, dmax10, trn10);

    p_trans_value11 = addFloatSliderParam("point8_z", "Set the translation values");
    p_trans_value11->setImmediate(0);
    p_trans_value11->setValue(dmin11, dmax11, trn11);

    p_trans_value12 = addFloatSliderParam("point9_z", "Set the translation values");
    p_trans_value12->setImmediate(0);
    p_trans_value12->setValue(dmin12, dmax12, trn12);

    p_recalc = addBooleanParam("generate", "recalculate the box");
    p_recalc->setValue(0);
    p_recalc->setImmediate(1);
    orec = 0;

    p_mode = addBooleanParam("preview", "mode of calculation");
    p_mode->setValue(0);
    p_mode->setImmediate(1);
    ocmode = 0;

    p_command = addOutputPort("command", "coDoText", "Flag for fluent");

    dmin = 0.0;
    dmax = 1.0;
    red = 1.0;
    green = 1.0;
    blue = 1.0;
    opq = 1.0;
    p_red = addFloatSliderParam("previoew_red", "Set the red value");
    p_red->setValue(dmin, dmax, red);
    p_red->setImmediate(0);
    p_green = addFloatSliderParam("preview_green", "Set the green value");
    p_green->setValue(dmin, dmax, green);
    p_green->setImmediate(0);
    p_blue = addFloatSliderParam("preview_blue", "Set the blue value");
    p_blue->setValue(dmin, dmax, blue);
    p_blue->setImmediate(0);
    p_opaque = addFloatSliderParam("preview_alpha", "Set the opaque value");
    p_opaque->setValue(dmin, dmax, opq);
    p_opaque->setImmediate(0);

    p_points = addOutputPort("modifiable_points", "coDoPoints", "modifiable points");
}

int ModifyHeadBox::compute()
{
    fprintf(stderr, "\nModifyHeadBox::compute\n");

    int npols, i, ired, igreen, iblue, iopq, mode, surf;
    unsigned long *cols;
    char comtext[1000], runtext[1000];
    const char *x, *oname;

    trn1 = p_trans_value1->getValue();
    trn2 = p_trans_value2->getValue();
    trn3 = p_trans_value3->getValue();
    trn4 = p_trans_value4->getValue();
    trn5 = p_trans_value5->getValue();
    trn6 = p_trans_value6->getValue();
    trn7 = p_trans_value7->getValue();
    trn8 = p_trans_value8->getValue();
    trn9 = p_trans_value9->getValue();
    trn10 = p_trans_value10->getValue();
    trn11 = p_trans_value11->getValue();
    trn12 = p_trans_value12->getValue();
    rec = p_recalc->getValue();
    cmode = p_mode->getValue();

    if (((((fabs(trn1 - otrn1) > 1) || (fabs(trn2 - otrn2) > 1) || (fabs(trn3 - otrn3) > 1) || (fabs(trn4 - otrn4) > 1) || (fabs(trn5 - otrn5) > 1) || (fabs(trn6 - otrn6) > 1) || (fabs(trn7 - otrn7) > 1) || (fabs(trn8 - otrn8) > 1) || (fabs(trn9 - otrn9) > 1) || (fabs(trn10 - otrn10) > 1) || (fabs(trn11 - otrn11) > 1) || (fabs(trn12 - otrn12) > 1)) && (cmode != 0)) || ((fabs(trn1 - sotrn1) > 1) || (fabs(trn2 - sotrn2) > 1) || (fabs(trn3 - sotrn3) > 1) || (fabs(trn4 - sotrn4) > 1) || (fabs(trn5 - sotrn5) > 1) || (fabs(trn6 - sotrn6) > 1) || (fabs(trn7 - sotrn7) > 1) || (fabs(trn8 - sotrn8) > 1) || (fabs(trn9 - sotrn9) > 1) || (fabs(trn10 - sotrn10) > 1) || (fabs(trn11 - sotrn11) > 1) || (fabs(trn12 - sotrn12) > 1))) && (rec != 0))
    {

        mode = cmode;

        if (!((fabs(trn1 - otrn1) > 1) || (fabs(trn2 - otrn2) > 1) || (fabs(trn3 - otrn3) > 1) || (fabs(trn4 - otrn4) > 1) || (fabs(trn5 - otrn5) > 1) || (fabs(trn6 - otrn6) > 1) || (fabs(trn7 - otrn7) > 1) || (fabs(trn8 - otrn8) > 1) || (fabs(trn9 - otrn9) > 1) || (fabs(trn10 - otrn10) > 1) || (fabs(trn11 - otrn11) > 1) || (fabs(trn12 - otrn12) > 1)))
            mode = 0;

        x = p_user->getValue();
        delete[] user;
        user = new char[strlen(x) + 1];
        strcpy(user, x);

        x = p_host->getValue();
        delete[] host;
        host = new char[strlen(x) + 1];
        strcpy(host, x);

        x = p_dir->getValue();
        delete[] dir;
        dir = new char[strlen(x) + 1];
        strcpy(dir, x);

        x = p_run->getValue();
        delete[] run;
        run = new char[strlen(x) + 1];
        strcpy(run, x);

        delete[] hexa_run;
        hexaconf = new CoviseConfig;
        hexa_run = hexaconf->get_entry("ModifyHeadBox.RUN");

        if (!hexa_run)
            hexa_run = "ssh -l %s %s 'cd %s ; %s %d %f %f %f %f %f %f %f %f %f %f %f %f'; scp %s@%s:%s/surface.msh .";
        //hexa_run = "ssh -l %s %s 'cd %s ; %s %d %f %f %f %f %f %f %f %f %f %f %f %f'";

        sprintf(runtext, hexa_run, user, host, dir, run, mode, -trn1 / 1000.0, -trn2 / 1000.0, -trn3 / 1000.0, -trn4 / 1000.0, -trn5 / 1000.0, -trn6 / 1000.0, -trn7 / 1000.0, -trn8 / 1000.0, -trn9 / 1000.0, -trn10 / 1000.0, -trn11 / 1000.0, -trn12 / 1000.0, user, host, dir);

        fprintf(stderr, "\truntext=[%s]\n", runtext);

        system(runtext);
        if (mode != 0)
        {
            flag = -flag;
            runmode = 2;
        }
        else
            runmode = 0;

        if (mode == 1)
        {
            otrn1 = trn1;
            otrn2 = trn2;
            otrn3 = trn3;
            otrn4 = trn4;
            otrn5 = trn5;
            otrn6 = trn6;
            otrn7 = trn7;
            otrn8 = trn8;
            otrn9 = trn9;
            otrn10 = trn10;
            otrn11 = trn11;
            otrn12 = trn12;
        }
        sotrn1 = trn1;
        sotrn2 = trn2;
        sotrn3 = trn3;
        sotrn4 = trn4;
        sotrn5 = trn5;
        sotrn6 = trn6;
        sotrn7 = trn7;
        sotrn8 = trn8;
        sotrn9 = trn9;
        sotrn10 = trn10;
        sotrn11 = trn11;
        sotrn12 = trn12;

        nps = 0;
    }
    else if ((rec != 0) || (cmode != ocmode))
        runmode = 0;
    else
    {
        flag = -flag;
        runmode = 1;
    }

    ocmode = cmode;

    sprintf(comtext, "%d\n", flag * runmode);

    fprintf(stderr, "\t output command text = [%s]\n", comtext);
    oname = p_command->getObjName();
    coDoText *command = new coDoText((char *)oname, strlen(comtext), comtext);
    p_command->setCurrentObject(command);

    if (nps == 0)
        createCube();

    if (nps == 0)
        nps = -1;

    surf = 1;
    if ((nps > 0) && (cmode == 0))
        surface = new coDoPolygons("surface", nxs,
                                   xs, ys, zs,
                                   ncs, cl,
                                   nps, pl);
    else
    {
        surf = 0;
        surface = new coDoPolygons("square", 4,
                                   xCoords, yCoords, zCoords,
                                   4, vertexList,
                                   1, polygonList);
    }

    surface->addAttribute("TRANSPARENCY", "1.0");

    red = p_red->getValue();
    green = p_green->getValue();
    blue = p_blue->getValue();
    opq = p_opaque->getValue();
    if (surf == 0)
    {
        opq = 1.0;
    }
    npols = surface->getNumPolygons();
    oname = p_geom->getObjName();
    coDoRGBA *co = new coDoRGBA("colors", npols);
    co->getAddress((int **)&cols);
    ired = 255 * red;
    igreen = 255 * green;
    iblue = 255 * blue;
    iopq = 255 * opq;
    for (i = 0; i < npols; i++)
        cols[i] = (ired << 24) + (igreen << 16) + (iblue << 8) + (iopq);
    coDoGeometry *com = new coDoGeometry(oname, surface);
    com->setColor(PER_FACE, co);
    p_geom->setCurrentObject(com);

    p_recalc->setValue(0);

    float *px, *py, *pz;
    const char *pointsObjectName = p_points->getObjName();
    coDoPoints *points = new coDoPoints(pointsObjectName, 9);
    points->getAddresses(&px, &py, &pz);

    // vertex 1
    px[0] = -p_trans_value1->getValue() / 1000.0;
    py[0] = 3334.84 / 1000.0;
    pz[0] = 0.0;

    // vertex 2
    px[1] = -p_trans_value2->getValue() / 1000.0;
    py[1] = 3334.84 / 1000.0;
    pz[1] = -p_trans_value3->getValue() / 1000.0;

    // vertex 3
    px[2] = -p_trans_value4->getValue() / 1000.0;
    py[2] = 6668.84 / 1000.0;
    pz[2] = 0.0;

    //vertex 4
    px[3] = -p_trans_value5->getValue() / 1000.0;
    py[3] = 6668.84 / 1000.0;
    pz[3] = -p_trans_value6->getValue() / 1000.0;

    // vertex 5
    px[4] = -p_trans_value7->getValue() / 1000.0;
    py[4] = 10000.0 / 1000.0;
    pz[4] = 0.0;

    // vertex 6
    px[5] = -p_trans_value8->getValue() / 1000.0;
    py[5] = 10000.0 / 1000.0;
    pz[5] = -p_trans_value9->getValue() / 1000.0;

    //vertex 7
    px[6] = 1500.0 / 1000.0;
    py[6] = 0.0;
    pz[6] = -p_trans_value10->getValue() / 1000.0;

    // vertex 8
    px[7] = 1500.0 / 1000.0;
    py[7] = 5000.0 / 1000.0;
    pz[7] = -p_trans_value11->getValue() / 1000.0;

    // vertex 9
    px[8] = 1500.0 / 1000.0;
    py[8] = 10000.0 / 1000.0;
    pz[8] = -p_trans_value12->getValue() / 1000.0;

    // add feedback attribute
    coFeedback feedback("ModifyHeadBox");
    feedback.addPara(p_trans_value1);
    feedback.addPara(p_trans_value2);
    feedback.addPara(p_trans_value3);
    feedback.addPara(p_trans_value4);
    feedback.addPara(p_trans_value5);
    feedback.addPara(p_trans_value6);
    feedback.addPara(p_trans_value7);
    feedback.addPara(p_trans_value8);
    feedback.addPara(p_trans_value9);
    feedback.addPara(p_trans_value10);
    feedback.addPara(p_trans_value11);
    feedback.addPara(p_trans_value12);

    feedback.addPara(p_recalc);
    feedback.addPara(p_mode);

    feedback.apply(points);

    p_points->setCurrentObject(points);
    return CONTINUE_PIPELINE;
}

void ModifyHeadBox::param(const char *name)
{
    if (strcmp(name, p_recalc->getName()) == 0)
        if (p_recalc->getValue() != 0)
            selfExec();
    return;
}

void ModifyHeadBox::quit()
{
}

void ModifyHeadBox::postInst()
{

    p_user->show();
    p_host->show();
    p_dir->show();
    p_run->show();
    p_trans_value1->show();
    p_trans_value2->show();
    p_trans_value3->show();
    p_trans_value4->show();
    p_trans_value5->show();
    p_trans_value6->show();
    p_trans_value7->show();
    p_trans_value8->show();
    p_trans_value9->show();
    p_trans_value10->show();
    p_trans_value11->show();
    p_trans_value12->show();
    otrn1 = p_trans_value1->getValue();
    otrn2 = p_trans_value2->getValue();
    otrn3 = p_trans_value3->getValue();
    otrn4 = p_trans_value4->getValue();
    otrn5 = p_trans_value5->getValue();
    otrn6 = p_trans_value6->getValue();
    otrn7 = p_trans_value7->getValue();
    otrn8 = p_trans_value8->getValue();
    otrn9 = p_trans_value9->getValue();
    otrn10 = p_trans_value10->getValue();
    otrn11 = p_trans_value11->getValue();
    otrn12 = p_trans_value12->getValue();
    sotrn1 = p_trans_value1->getValue();
    sotrn2 = p_trans_value2->getValue();
    sotrn3 = p_trans_value3->getValue();
    sotrn4 = p_trans_value4->getValue();
    sotrn5 = p_trans_value5->getValue();
    sotrn6 = p_trans_value6->getValue();
    sotrn7 = p_trans_value7->getValue();
    sotrn8 = p_trans_value8->getValue();
    sotrn9 = p_trans_value9->getValue();
    sotrn10 = p_trans_value10->getValue();
    sotrn11 = p_trans_value11->getValue();
    sotrn12 = p_trans_value12->getValue();

    p_recalc->show();
    p_mode->show();

    p_red->show();
    p_green->show();
    p_blue->show();
    p_opaque->show();
}

void ModifyHeadBox::createCube()
{
    int ind;

    nps = 0;
    ncs = 0;
    nxs = 0;
    ifile.open("surface.msh");
    if (!ifile)
        return;
    while (open_group() == 0)
    {
        ind = -1;
        ifile >> ind;
        //    // cerr << ind << endl;
        switch (ind)
        {
        case 0:
        case 1:
            read_string();
            break;
        case 2:
            read_dims();
            break;
        case 10:
            read_nodes();
            break;
        case 13:
            read_faces();
            break;
        default:
            break;
        }
        close_group();
    }
    ifile.close();
}

int open_group()
{
    char ch;

    read_spaces();
    ifile.get(ch);
    if (ch != '(')
        return -1;
    return 0;
}

void close_group()
{
    char ch;
    while (ifile.get(ch))
    {
        if (ch == '"')
        {
            ifile.putback(ch);
            read_string();
        }
        if (ch == '(')
            close_group();
        if (ch == ')')
        {
            return;
        }
    }
}

void read_spaces()
{
    char ch;
    while (ifile.get(ch))
        if ((ch != ' ') && (ch != '\t') && (ch != '\n'))
        {
            ifile.putback(ch);
            return;
        }
}

int read_string()
{
    char ch;
    read_spaces();
    ifile.get(ch);
    if (ch != '"')
        return -1;
    while (ifile.get(ch))
        if (ch == '"')
            return 0;
    return -2;
}

void read_dims()
{
    ifile >> dims;
}

void read_nodes()
{
    int i, first, last, etype, zone_id;

    if (open_group() != 0)
        return;
    ifile >> zone_id;
    first = read_hex();
    last = read_hex();
    ifile >> etype;
    read_spaces();
    if (ifile.peek() != ')')
        ifile >> dims;
    close_group();
    if (zone_id == 0)
        return;
    if (open_group() != 0)
        return;
    if (last > xss - 1)
    {
        xs = (float *)realloc(xs, (last + 1000) * sizeof(float));
        ys = (float *)realloc(ys, (last + 1000) * sizeof(float));
        zs = (float *)realloc(zs, (last + 1000) * sizeof(float));
        xss = last + 1000;
    }
    for (i = first - 1; i < last; i++)
    {
        nxs++;
        ifile >> xs[i];
        ifile >> ys[i];
        if (dims > 2)
            ifile >> zs[i];
    }
    close_group();
}

void read_faces()
{
    int i, j, first, last, btype, etype, ttype, zone_id, nid;

    if (open_group() != 0)
        return;
    ifile >> zone_id;
    first = read_hex();
    last = read_hex();
    ifile >> btype;
    etype = 0;
    read_spaces();
    if (ifile.peek() != ')')
        ifile >> etype;
    close_group();
    if (zone_id == 0)
        return;
    if (open_group() != 0)
        return;
    if (last > pls - 1)
    {
        pl = (int *)realloc(pl, (last + 1000) * sizeof(int));
        pls = last + 1000;
    }
    for (i = first; i <= last; i++)
    {
        ttype = etype;
        if (ttype == 0)
            ifile >> ttype;
        if ((ncs + ttype) > cls - 1)
        {
            cl = (int *)realloc(cl, (cls + 1000) * sizeof(int));
            cls = cls + 1000;
        }
        pl[nps] = ncs;
        nps++;
        for (j = 1; j <= ttype; j++)
        {
            nid = read_hex();
            cl[ncs] = nid - 1;
            ncs++;
        }
        ifile.ignore(10000, '\n');
    }
    close_group();
}

int read_hex()
{
    char hex[100];
    char ch;
    int val, i;

    read_spaces();
    i = 0;
    while (ifile.get(ch))
    {
        switch (ch)
        {
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
        case 'a':
        case 'A':
        case 'b':
        case 'B':
        case 'c':
        case 'C':
        case 'd':
        case 'D':
        case 'e':
        case 'E':
        case 'f':
        case 'F':
            hex[i++] = ch;
            break;
        default:
            ifile.putback(ch);
            hex[i] = 0;
            sscanf(hex, "%x", &val);
            return val;
        }
    }
    return -1;
}

int main(int argc, char *argv[])

{
    ModifyHeadBox *application = new ModifyHeadBox;
    application->start(argc, argv);
    return 0;
}
