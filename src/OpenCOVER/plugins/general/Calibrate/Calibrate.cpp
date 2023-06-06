/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)1999 RUS   **
**                                                                          **
** Description: Module for Calibrating VEs                                  **
**                                                                          **
**                                                                          **
** Author: U. Woessner                                                      **
**                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <config/CoviseConfig.h>
#include <util/unixcompat.h>

#include "Calibrate.h"
//#define MyNAN(X) *((int *)(&(X)))=-1
//#define IsMyNAN(X) (*((int *)(&(X)))==-1)

#define MyNAN 10000000.0

Calibrate::Calibrate()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    position = NULL;
    orientation = NULL;
    CubeMax[0] = 1400;
    CubeMax[1] = 1400;
    CubeMax[2] = 1250;
    CubeMin[0] = -1400;
    CubeMin[1] = -1400;
    CubeMin[2] = -1250;
    fp = NULL;
    //text = NULL;
    //string = NULL;
    nx = ny = nz = 0;
    cI = cJ = cK = 0;
    cerr << "initialize Callibration" << endl;
    // create menu entries for this plugin
    createMenuEntry();
    readFile();

    // Start on top and stepDown
    cK = nz;
}

// this is called if the plugin is removed at runtime
Calibrate::~Calibrate()
{
    // remove the menu entries for this plugin
    removeDisplay();
    removeMenuEntry();
    delete[] position;
    delete[] orientation;
}
void Calibrate::readFile()
{

    int i, j, k, n = 0, line = 0;
    bool hasDim = false;
    bool hasMin = false;
    bool hasMax = false;
    fp = fopen("calib.asc", "r");
    if (fp == NULL)
    {
        return;
    }
    char buf[1000];
    // read header
    while ((!feof(fp)) && (hasDim == false || hasMin == false || hasMax == false))
    {
        if (fgets(buf, 1000, fp) == NULL)
            cerr << "Calibrate::readFile fgets failed " << endl;
        line++;
        if ((buf[0] != '%') && (strlen(buf) > 5))
        {
            if (strncasecmp(buf, "DIM", 3) == 0)
            {
                int iret = sscanf(buf + 3, "%d %d %d", &nx, &ny, &nz);
                if (iret != 3)
                    cerr << "Calibrate::readFile  sscanf failed: read " << iret << endl;
                hasDim = true;
            }
            else if (strncasecmp(buf, "MIN", 3) == 0)
            {
                int iret = sscanf(buf + 3, "%f %f %f", &min[0], &min[1], &min[2]);
                if (iret != 3)
                    cerr << "Calibrate::readFile  sscanf failed: read " << iret << endl;
                hasMin = true;
            }
            else if (strncasecmp(buf, "MAX", 3) == 0)
            {
                int iret = sscanf(buf + 3, "%f %f %f", &max[0], &max[1], &max[2]);
                if (iret != 3)
                    cerr << "Calibrate::readFile  sscanf failed: read " << iret << endl;
                hasMax = true;
            }
            else
            {
                cerr << "Unknown statement in line " << line << endl;
            }
        }
    }

    if (!(nx == 0 || ny == 0 || nz == 0))
    {
        int i, num = nx * ny * nz;
        position = new pos[num];
        orientation = new ori[num];
        for (i = 0; i < num; i++)
        {
            position[i].x = MyNAN; // make x Not a Number, so that we know wich values are set
        }
        while (!feof(fp))
        {
            if (fgets(buf, 1000, fp) == NULL)
                cerr << "Calibrate::readFile fgets2 failed " << endl;
            line++;
            if ((buf[0] != '%') && (strlen(buf) > 5))
            {
                int ntok = sscanf(buf, "%d %d %d", &i, &j, &k);
                if (ntok == 3) // all three numbers parsed
                {
                    n = i * ny * nz + j * nz + k;
                    int iret = sscanf(buf, "%d %d %d    %f %f %f    %f %f %f  %f %f %f  %f %f %f",
                                      &i, &j, &k,
                                      &position[n].x, &position[n].y, &position[n].z,
                                      &orientation[n].o[0], &orientation[n].o[1], &orientation[n].o[2],
                                      &orientation[n].o[3], &orientation[n].o[4], &orientation[n].o[5],
                                      &orientation[n].o[6], &orientation[n].o[7], &orientation[n].o[8]);
                    if (iret != 15)
                        cerr << "Calibrate::readFile  sscanf4 failed: read " << iret << endl;
                }
            }
        }
        makeDisplay();
        updateDisplay();
    }

    int numSet = 0, num = nx * ny * nz;
    for (i = 0; i < num; i++)
    {
        if (position[i].x != MyNAN)
            numSet++;
    }

    sprintf(buf, "%3d of %3d positions captured", numSet, num);
    statusLine->setLabel(buf);
    fclose(fp);
}

// Check Buttons.
void Calibrate::preFrame()
{
    if (nx * ny * nz == 0)
        return;
    osg::Matrix m = cover->getViewerMat();
    osg::Vec3 pos;
    pos = m.getTrans();
    char buf[1000];
    sprintf(buf, "Index: %d,%d,%d", cI, cJ, cK);
    currentIndex->setLabel(buf);
    sprintf(buf, "Soll: %6.2f,%6.2f,%6.2f", getX(cI), getY(cJ), getZ(cK));
    currentSoll->setLabel(buf);
    sprintf(buf, "Ist: %6.2f,%6.2f,%6.2f", pos[0], pos[1], pos[2]);
    currentIst->setLabel(buf);
    if (cover->getPointerButton()->wasPressed())
    {
        fprintf(stderr, "\a"); // beep
        int n = cI * ny * nz + cJ * nz + cK;
        if (position[n].x != MyNAN)
            fprintf(stderr, "\a"); // beep
        position[n].x = pos[0];
        position[n].y = pos[1];
        position[n].z = pos[2];
        orientation[n].o[0] = m(0, 0);
        orientation[n].o[1] = m(0, 1);
        orientation[n].o[2] = m(0, 2);
        orientation[n].o[3] = m(1, 0);
        orientation[n].o[4] = m(1, 1);
        orientation[n].o[5] = m(1, 2);
        orientation[n].o[6] = m(2, 0);
        orientation[n].o[7] = m(2, 1);
        orientation[n].o[8] = m(2, 2);
        stepDown();
    }
}

// update position Display on the screen
void Calibrate::updateDisplay()
{
    /*   pfString *s;
   while(text->getNumStrings())
   {
   s = text->getString(0);
   text->removeString(s);
   pfDelete(s);
   }
   int n = cI*ny*nz+cJ*nz+cK;
   if(position[n].x != MyNAN)
   {
   marker->setVal(0);
   fprintf(stderr,"\a");                       // beep
   }
   else
   {
   marker->setVal(1);
   }*/
    int i, numSet = 0, num = nx * ny * nz;
    for (i = 0; i < num; i++)
    {
        if (position[i].x != MyNAN)
            numSet++;
    }
    char buf[1000];
    sprintf(buf, "%d of %d positions captured)", numSet, num);
    statusLine->setLabel(buf);
    fprintf(stderr, "%s\n", buf);
    /* string = new pfString();
   string->setMode( PFSTR_JUSTIFY, PFSTR_CENTER);
   string->setColor(1.0, 1.0, 1.0,1.0);
   string->setGState(geoState);
   pfGeoSet* chr = (pfGeoSet*)string->getCharGSet(0);
   chr->setGState(geoState);
   string->setFont( font);
   string->setString(buf);
   string->setMat( textMat);

   text->addString(string);

   markers[0]->setTrans(getX(cI),CubeMax[1]+0.1,CubeMin[2]);
   markers[1]->setTrans(CubeMin[0],CubeMax[1]+0.1,getZ(cK));
   markers[2]->setTrans(getX(cI),CubeMax[1]+0.1,CubeMin[2]-0.1);
   markers[3]->setTrans(CubeMin[0],getY(cJ),CubeMin[2]-0.1);
   markers[4]->setTrans(CubeMin[0],CubeMax[1]+0.1,getZ(cK));
   markers[5]->setTrans(CubeMax[0],CubeMax[1]+0.1,getZ(cK));
   markers[6]->setTrans(CubeMin[0],getY(cJ),CubeMin[2]-0.1);
   markers[7]->setTrans(CubeMax[0],getY(cJ),CubeMin[2]-0.1);*/
}

void Calibrate::makeDisplay()
{
    /* float size = 100;

   // create Text
   char    *buf;

   buf = (char *)cover->getname("share/covise/fonts/Helvetica.mf");
   buf [strlen(buf) -3] = 0;
   //PFDFONT_FILLED  PFDFONT_TEXTURED
   font = pfdLoadFont("type1", buf, PFDFONT_FILLED);
   font->ref();

   text = new pfText;

   osg::Matrix scale,translation;
   //rotation.makeRot(-90,1,0,0);
   translation.makeTrans(0,CubeMax[1],0);
   scale.makeScale(size,size,size);
   textMat =  scale * translation;
   pfMaterial *mtl;

   mtl = new pfMaterial;
   mtl->setSide(PFMTL_BOTH);
   mtl->setColorMode(PFMTL_BOTH,PFMTL_CMODE_AMBIENT_AND_DIFFUSE);
   mtl->setColor( PFMTL_AMBIENT, 1,1,1);
   mtl->setColor( PFMTL_DIFFUSE, 1,1,1);
   mtl->setColor( PFMTL_SPECULAR, 1.0f, 0.0f, 0.0f);
   mtl->setColor( PFMTL_EMISSION, 1,1,1);
   mtl->setShininess(16.0f);

   geoState = new pfGeoState();
   geoState->makeBasic();
   geoState->setAttr(PFSTATE_FRONTMTL, mtl);
   geoState->setAttr(PFSTATE_BACKMTL, mtl);
   geoState->setMode(PFSTATE_ENLIGHTING, PF_ON);
   geoState->setMode(PFSTATE_TRANSPARENCY, PFTR_OFF);
   geoState->ref();

   string = new pfString();
   string->setMode( PFSTR_JUSTIFY, PFSTR_CENTER);
   string->setColor(1.0, 1.0, 1.0,1.0);

   string->setGState(geoState);

   pfGeoSet* chr = (pfGeoSet*)string->getCharGSet(0);
   chr->setGState(geoState);

   string->setFont( font);
   string->setString("Covise Tracker Calibration");
   string->setMat( textMat);

   text->addString(string);

   fixNode(text);
   cover->getScene()->addChild(text);

   //
   // create Grid lines
   //

   void* arena = pfGetSharedArena();
   pfGeoSet *geoset = new(arena) pfGeoSet();
   pfVec4 *color = (pfVec4*)pfMalloc(sizeof(pfVec4), arena);
   pfVec3 *coord = (pfVec3*)pfMalloc(sizeof(pfVec3)*(4*nx+6*ny+6*nz), arena);

   int i,n=0;
   for(i=0;i<nx;i++)
   {
   coord[n].set(getX(i), CubeMax[1] , CubeMin[2]);
   n++;
   coord[n].set(getX(i), CubeMax[1] , CubeMax[2]);
   n++;
   }
   for(i=0;i<ny;i++)
   {
   coord[n].set(CubeMax[0], getY(i) , CubeMin[2]);
   n++;
   coord[n].set(CubeMax[0], getY(i) , CubeMax[2]);
   n++;
   }
   for(i=0;i<ny;i++)
   {
   coord[n].set(CubeMin[0], getY(i) , CubeMin[2]);
   n++;
   coord[n].set(CubeMin[0], getY(i) , CubeMax[2]);
   n++;
   }
   for(i=0;i<nz;i++)
   {
   coord[n].set(CubeMin[0], CubeMin[1], getZ(i));
   n++;
   coord[n].set(CubeMin[0], CubeMax[1], getZ(i));
   n++;
   }
   for(i=0;i<nz;i++)
   {
   coord[n].set(CubeMax[0], CubeMin[1], getZ(i));
   n++;
   coord[n].set(CubeMax[0], CubeMax[1], getZ(i));
   n++;
   }
   for(i=0;i<nz;i++)
   {
   coord[n].set(CubeMin[0], CubeMax[1], getZ(i));
   n++;
   coord[n].set(CubeMax[0], CubeMax[1], getZ(i));
   n++;
   }
   for(i=0;i<nx;i++)
   {
   coord[n].set(getX(i), CubeMin[1] , CubeMin[2]);
   n++;
   coord[n].set(getX(i), CubeMax[1] , CubeMin[2]);
   n++;
   }
   for(i=0;i<ny;i++)
   {
   coord[n].set(CubeMax[0], getY(i) , CubeMin[2]);
   n++;
   coord[n].set(CubeMin[0], getY(i) , CubeMin[2]);
   n++;
   }

   color[0].set(1.0f, 1.0f, 1.0f, 1.0f);

   geoset->setAttr(PFGS_COLOR4, PFGS_OVERALL, color, NULL);
   geoset->setAttr(PFGS_COORD3, PFGS_PER_PRIM, coord, NULL);

   geoset->setNumPrims(nx*2+3*ny+3*nz);
   geoset->setPrimType(PFGS_LINES);
   geoset->setLineWidth(1.0f);

   geoset->setGState(geoState);

   grid = new pfGeode();
   grid->setName("Grid");
   grid->addGSet(geoset);
   cover->getScene()->addChild(grid);

   //
   // create Marker
   //

   geoset = new(arena) pfGeoSet();
   color = (pfVec4*)pfMalloc(sizeof(pfVec4), arena);
   coord = (pfVec3*)pfMalloc(sizeof(pfVec3)*4, arena);
   pfMaterial *redMtl;

   redMtl = new pfMaterial;
   redMtl->setSide(PFMTL_BOTH);
   redMtl->setColorMode(PFMTL_BOTH,PFMTL_CMODE_AMBIENT_AND_DIFFUSE);
   redMtl->setColor( PFMTL_AMBIENT, 1,0,0);
   redMtl->setColor( PFMTL_DIFFUSE, 1,0,0);
   redMtl->setColor( PFMTL_SPECULAR, 1.0f, 0.0f, 0.0f);
   redMtl->setColor( PFMTL_EMISSION, 1,0,0);
   redMtl->setShininess(16.0f);

   pfGeoState* redGeoState = new pfGeoState();
   redGeoState->makeBasic();
   redGeoState->setAttr(PFSTATE_FRONTMTL, redMtl);
   redGeoState->setAttr(PFSTATE_BACKMTL, redMtl);
   redGeoState->setMode(PFSTATE_ENLIGHTING, PF_ON);
   redGeoState->setMode(PFSTATE_TRANSPARENCY, PFTR_OFF);
   redGeoState->ref();

   coord[0].set(-size/5, 0, 0);
   coord[1].set( size/5, 0, 0);
   coord[2].set( size/5, 1, 0);
   coord[3].set(-size/5, 1, 0);

   color[0].set(1.0f, 0.0f, 0.0f, 1.0f);

   geoset->setAttr(PFGS_COLOR4, PFGS_OVERALL, color, NULL);
   geoset->setAttr(PFGS_COORD3, PFGS_PER_VERTEX, coord, NULL);

   geoset->setNumPrims(1);
   geoset->setPrimType(PFGS_QUADS);

   geoset->setGState(redGeoState);

   pfGeode *g1 = new pfGeode();
   g1->setName("Marker Red");
   g1->addGSet(geoset);

   //
   // create Marker
   //

   geoset = new(arena) pfGeoSet();
   color = (pfVec4*)pfMalloc(sizeof(pfVec4), arena);
   coord = (pfVec3*)pfMalloc(sizeof(pfVec3)*4, arena);
   pfMaterial *greenMtl;

   greenMtl = new pfMaterial;
   greenMtl->setSide(PFMTL_BOTH);
   greenMtl->setColorMode(PFMTL_BOTH,PFMTL_CMODE_AMBIENT_AND_DIFFUSE);
   greenMtl->setColor( PFMTL_AMBIENT, 0,1,0);
   greenMtl->setColor( PFMTL_DIFFUSE, 0,1,0);
   greenMtl->setColor( PFMTL_SPECULAR, 0.0f, 1.0f, 0.0f);
   greenMtl->setColor( PFMTL_EMISSION, 0,1,0);
   greenMtl->setShininess(16.0f);

   pfGeoState* greenGeoState = new pfGeoState();
   greenGeoState->makeBasic();
   greenGeoState->setAttr(PFSTATE_FRONTMTL, greenMtl);
   greenGeoState->setAttr(PFSTATE_BACKMTL, greenMtl);
   greenGeoState->setMode(PFSTATE_ENLIGHTING, PF_ON);
   greenGeoState->setMode(PFSTATE_TRANSPARENCY, PFTR_OFF);
   greenGeoState->ref();

   coord[0].set(-size/5, 0, 0);
   coord[1].set( size/5, 0, 0);
   coord[2].set( size/5, 1, 0);
   coord[3].set(-size/5, 1, 0);

   color[0].set(0.0f, 1.0f, 0.0f, 1.0f);

   geoset->setAttr(PFGS_COLOR4, PFGS_OVERALL, color, NULL);
   geoset->setAttr(PFGS_COORD3, PFGS_PER_VERTEX, coord, NULL);

   geoset->setNumPrims(1);
   geoset->setPrimType(PFGS_QUADS);

   geoset->setGState(greenGeoState);

   pfGeode *g2 = new pfGeode();
   g2->setName("Marker Green");
   g2->addGSet(geoset);

   marker = new pfSwitch();
   marker->addChild(g1);
   marker->addChild(g2);

   pfDCS *dcs = new pfDCS();
   dcs->setScale(1,(CubeMax[2]-CubeMin[2]),1);
   dcs->setRot(0,90,0);
   dcs->addChild(marker);
   markers[0] = dcs;

   dcs = new pfDCS();
   dcs->setScale(1,(CubeMax[0]-CubeMin[0]),1);
   dcs->setRot(-90,0,90);
   dcs->addChild(marker);
   markers[1] = dcs;

   dcs = new pfDCS();
   dcs->setScale(1,(CubeMax[1]-CubeMin[1]),1);
   dcs->setRot(180,0,0);
   dcs->addChild(marker);
   markers[2] = dcs;

   dcs = new pfDCS();
   dcs->setScale(1,(CubeMax[0]-CubeMin[0]),1);
   dcs->setRot(-90,0,0);
   dcs->addChild(marker);
   markers[3] = dcs;

   dcs = new pfDCS();
   dcs->setScale(1,(CubeMax[1]-CubeMin[1]),1);
   dcs->setRot(180,0,90);
   dcs->addChild(marker);
   markers[4] = dcs;

   dcs = new pfDCS();
   dcs->setScale(1,(CubeMax[1]-CubeMin[1]),1);
   dcs->setRot(180,0,90);
   dcs->addChild(marker);
   markers[5] = dcs;

   dcs = new pfDCS();
   dcs->setScale(1,(CubeMax[2]-CubeMin[2]),1);
   dcs->setRot(0,90,90);
   dcs->addChild(marker);
   markers[6] = dcs;

   dcs = new pfDCS();
   dcs->setScale(1,(CubeMax[2]-CubeMin[2]),1);
   dcs->setRot(0,90,90);
   dcs->addChild(marker);
   markers[7] = dcs;

   cover->getScene()->addChild(markers[0]);
   cover->getScene()->addChild(markers[1]);
   cover->getScene()->addChild(markers[2]);
   cover->getScene()->addChild(markers[3]);
   cover->getScene()->addChild(markers[4]);
   cover->getScene()->addChild(markers[5]);
   cover->getScene()->addChild(markers[6]);
   cover->getScene()->addChild(markers[7]);
   cover->getScene()->addChild(marker);
   */
}

void Calibrate::removeDisplay()
{
    /* cover->getScene()->removeChild(text);
   cover->getScene()->removeChild(grid);
   pfDelete(text);
   pfDelete(grid);*/
}

void Calibrate::createMenuEntry()
{

    calibTab = new coTUITab("Calibrate", coVRTui::instance()->mainFolder->getID());
    calibTab->setPos(0, 0);

    currentIndex = new coTUILabel("currentIndex", calibTab->getID());
    currentSoll = new coTUILabel("currentSoll", calibTab->getID());
    currentIst = new coTUILabel("currentIst", calibTab->getID());
    statusLine = new coTUILabel("no values captured", calibTab->getID());
    Next = new coTUIButton("Next", calibTab->getID());
    StepX = new coTUIButton("Step X", calibTab->getID());
    StepY = new coTUIButton("Step Y", calibTab->getID());
    StepZ = new coTUIButton("Step Z", calibTab->getID());
    Save = new coTUIButton("Save", calibTab->getID());
    Load = new coTUIButton("Load", calibTab->getID());
    Capture = new coTUIButton("Capture", calibTab->getID());

    Next->setEventListener(this);
    StepX->setEventListener(this);
    StepY->setEventListener(this);
    StepZ->setEventListener(this);
    Save->setEventListener(this);
    Load->setEventListener(this);
    Capture->setEventListener(this);

    currentIndex->setPos(1, 0);
    currentSoll->setPos(1, 1);
    currentIst->setPos(1, 2);
    statusLine->setPos(0, 1);
    statusLine->setSize(200, 20);
    Next->setPos(0, 2);
    StepX->setPos(0, 3);
    StepY->setPos(0, 4);
    StepZ->setPos(0, 5);
    Save->setPos(0, 6);
    Load->setPos(0, 7);
    Capture->setPos(0, 8);
}

void
Calibrate::removeMenuEntry()
{
    delete Save;
    delete StepZ;
    delete StepY;
    delete StepX;
    delete Next;
    delete Load;
    delete Capture;
    delete statusLine;
    delete currentIst;
    delete currentSoll;
    delete currentIndex;
    delete calibTab;
}

void
Calibrate::save()
{
    if (!coVRMSController::instance()->isMaster())
        return; // don´t save on slaves
    int i, j, k, n = 0;
    fp = fopen("calib.asc", "w");

    fprintf(fp, "DIM %d %d %d\n", nx, ny, nz);
    fprintf(fp, "MIN %f %f %f\n", min[0], min[1], min[2]);
    fprintf(fp, "MAX %f %f %f\n", max[0], max[1], max[2]);
    for (i = 0; i < nx; i++)
    {
        for (j = 0; j < ny; j++)
        {
            for (k = 0; k < nz; k++)
            {
                if (position[n].x != MyNAN)
                {
                    fprintf(fp, "%d %d %d    %f %f %f    %f %f %f  %f %f %f  %f %f %f\n",
                            i, j, k,
                            position[n].x, position[n].y, position[n].z,
                            orientation[n].o[0], orientation[n].o[1], orientation[n].o[2],
                            orientation[n].o[3], orientation[n].o[4], orientation[n].o[5],
                            orientation[n].o[6], orientation[n].o[7], orientation[n].o[8]);
                }
                n++;
            }
        }
    }

    fclose(fp);

    fp = fopen("calib.pos", "w");

    fprintf(fp, "%d %d %d 1\n", nx, ny, nz);
    fprintf(fp, "\n");
    n = 0;
    for (i = 0; i < nx; i++)
    {
        for (j = ny - 1; j >= 0; j--)
        {
            for (k = 0; k < nz; k++)
            {
                if (position[n].x != MyNAN)
                {
                    fprintf(fp, " %f %f %f    %f %f %f  %d\n",
                            position[n].x / 10.0, position[n].y / 10.0, position[n].z / 10.0, getX(i) / 10.0, getY(j) / 10.0, getZ(k) / 10.0, n);
                }
                else
                {
                    fprintf(fp, " %f %f %f    %f %f %f  %d\n",
                            getX(i) / 10.0, getY(j) / 10.0, getZ(k) / 10.0, getX(i) / 10.0, getY(j) / 10.0, getZ(k) / 10.0, n);
                }
                n++;
            }
            fprintf(fp, "\n");
        }
    }

    fclose(fp);

    fp = fopen("calib.or_x", "w");

    fprintf(fp, "%d %d %d 1\n", nx, ny, nz);
    fprintf(fp, "\n");
    n = 0;
    for (i = 0; i < nx; i++)
    {
        for (j = ny - 1; j >= 0; j--)
        {
            for (k = 0; k < nz; k++)
            {
                if (position[n].x != MyNAN)
                {
                    fprintf(fp, " %f %f %f    %f %f %f  %d\n",
                            position[n].x / 10.0, position[n].y / 10.0, position[n].z / 10.0,
                            orientation[n].o[0], orientation[n].o[1], orientation[n].o[2],
                            n);
                }
                else
                {
                    fprintf(fp, " %f %f %f    %f %f %f  %d\n",
                            getX(i) / 10.0, getY(j) / 10.0, getZ(k) / 10.0, 1.0, 0.0, 0.0, n);
                }
                n++;
            }
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
    fp = fopen("calib.or_y", "w");

    fprintf(fp, "%d %d %d 1\n", nx, ny, nz);
    fprintf(fp, "\n");
    n = 0;
    for (i = 0; i < nx; i++)
    {
        for (j = ny - 1; j >= 0; j--)
        {
            for (k = 0; k < nz; k++)
            {
                if (position[n].x != MyNAN)
                {
                    fprintf(fp, " %f %f %f    %f %f %f  %d\n",
                            position[n].x / 10.0, position[n].y / 10.0, position[n].z / 10.0,
                            orientation[n].o[3], orientation[n].o[4], orientation[n].o[5],
                            n);
                }
                else
                {
                    fprintf(fp, " %f %f %f    %f %f %f  %d\n",
                            getX(i) / 10.0, getY(j) / 10.0, getZ(k) / 10.0, 0.0, 1.0, 0.0, n);
                }
                n++;
            }
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
    fp = fopen("calib.or_z", "w");

    fprintf(fp, "%d %d %d 1\n", nx, ny, nz);
    fprintf(fp, "\n");
    n = 0;
    for (i = 0; i < nx; i++)
    {
        for (j = ny - 1; j >= 0; j--)
        {
            for (k = 0; k < nz; k++)
            {
                if (position[n].x != MyNAN)
                {
                    fprintf(fp, " %f %f %f    %f %f %f  %d\n",
                            position[n].x / 10.0, position[n].y / 10.0, position[n].z / 10.0,
                            orientation[n].o[6], orientation[n].o[7], orientation[n].o[8],
                            n);
                }
                else
                {
                    fprintf(fp, " %f %f %f    %f %f %f  %d\n",
                            getX(i) / 10.0, getY(j) / 10.0, getZ(k) / 10.0, 0.0, 0.0, 1.0, n);
                }
                n++;
            }
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
}

void Calibrate::stepX()
{
    cI++;
    if (cI >= nx)
    {
        cI = 0;
    }
    updateDisplay();
}

void Calibrate::stepY()
{
    cJ++;
    if (cJ >= ny)
    {
        cJ = 0;
    }
    updateDisplay();
}

void Calibrate::stepZ()
{
    cK++;
    if (cK >= nz)
    {
        cK = 0;
    }
    updateDisplay();
}

void Calibrate::step()
{
    cK++;
    if (cK >= nz)
    {
        cK = 0;
        cI++;
        if (cI >= nx)
        {
            cI = 0;
            cJ++;
            if (cJ >= ny)
            {
                cJ = 0;
            }
        }
    }
    updateDisplay();
}

void Calibrate::stepDown()
{
    cK--;
    if (cK < 0)
    {
        cK = nz;
        cI++;
        if (cI >= nx)
        {
            cI = 0;
            cJ++;
            if (cJ >= ny)
            {
                cJ = 0;
            }
        }
    }
    updateDisplay();
}

void Calibrate::stepNext()
{
    int i, j, k = 0;
    for (j = 0; j < ny; j++)
    {
        for (i = 0; i < nx; i++)
        {
            for (k = 0; k < nz; k++)
            {
                if (position[i * ny * nz + j * nz + k].x == MyNAN)
                {
                    break;
                    cI = i;
                    cJ = j;
                    cK = k;
                }
            }
            if (position[i * ny * nz + j * nz + k].x == MyNAN)
                break;
        }
        if (position[i * ny * nz + j * nz + k].x == MyNAN)
            break;
    }
    updateDisplay();
}

void Calibrate::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == StepX)
    {
        stepX();
    }
    else if (tUIItem == StepY)
    {
        stepY();
    }
    else if (tUIItem == StepZ)
    {
        stepZ();
    }
    else if (tUIItem == Next)
    {
        stepNext();
    }
    else if (tUIItem == Save)
    {
        save();
    }
    else if (tUIItem == Load)
    {
        readFile();
    }
    else if (tUIItem == Capture)
    {
        osg::Matrix m = cover->getViewerMat();
        osg::Vec3 pos;
        pos = m.getTrans();
        fprintf(stderr, "\a"); // beep
        int n = cI * ny * nz + cJ * nz + cK;
        if (position)
        {
            if (position[n].x != MyNAN)
                fprintf(stderr, "\a"); // beep
            position[n].x = pos[0];
            position[n].y = pos[1];
            position[n].z = pos[2];
        }
        if (orientation)
        {
            orientation[n].o[0] = m(0, 0);
            orientation[n].o[1] = m(0, 1);
            orientation[n].o[2] = m(0, 2);
            orientation[n].o[3] = m(1, 0);
            orientation[n].o[4] = m(1, 1);
            orientation[n].o[5] = m(1, 2);
            orientation[n].o[6] = m(2, 0);
            orientation[n].o[7] = m(2, 1);
            orientation[n].o[8] = m(2, 2);
        }
        stepDown();
    }
}

COVERPLUGIN(Calibrate)
