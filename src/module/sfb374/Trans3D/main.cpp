/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          main.cpp  -  main program
                             -------------------
    begin                : Wed Mar 29 07:37:14 CEST 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

/***************************************************************************
 *     PROGRAM NAME: TRANS3D  (NON-CONSERVATIVE FORM)                      *
 *     TRANSIENT 3-D CARTESIAN COORDINATE CONDUCTION CODE                  *
 *     WITH BOUNDARY-FITTED COORDINATE TRANSFORMATION.                     *
 *     IMPLICIT IN ZETA; EXPLICIT IN XI, ETA;                              *
 *     FIRST ORDER IN TAU; 2ND ORDER IN ZETA, ETA, XI.                     *
 ***************************************************************************/

#ifdef WIN32
#pragma warning(disable : 4786) // disable debug warning for long names
#endif

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#ifdef __sgi
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#else
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <cstdio>
#endif

#include "main.h"
#include "fortran.h"
#include "raytrace.h"
#include "classext.h"
#include "grid.h"
#include "laser.h"
#include "material.h"
#include "simul.h"
#include "solid.h"
#include "solve.h"

#ifdef COVISE
#include "coviseInterface.h"
#endif

#ifdef GUI
#include <qfont.h>

#include "trans3d_gui.h"
#include "trans3d_guiview.h"
#endif

using namespace std;

// typedef int  (*PZIPFN) (char*);

#ifdef GUI
QApplication *papp; // application
TextView *pout; // default output stream
TextView *perr; // default error stream
#else
ostream *pout; // default output stream
ostream *perr; // default error stream
#endif
ostream *plog; // default log stream
ostream *pdebug; // default debug stream

istream *pin; // default input stream

string inname; // name of input file
string outname; // name of output file
string abortname; // name of abort file

TScript script; // script class

// ***************************************************************************
// global variables
// ***************************************************************************

prec RefLength; // length scale
prec RefTime; // time scale
prec RefVelocity; // velocity scale

// ***************************************************************************
// compare stream content with string
// ***************************************************************************

istream &operator>>(istream &is, checkstring cs)
{
    is >> ws;
    int i = 0;
    char c;
    while (cs.pstr[i] != '\0')
    {
        c = is.get();
        if (!is || tolower(c) != tolower(cs.pstr[i]))
        {
            is.clear();
            if (c != char(EOF))
                ;
            is.putback(c);
            while (i > 0)
            {
                i--;
                is.putback(cs.pstr[i]);
            }
            *cs.pb = false;
            return is;
        }
        i++;
    }
    *cs.pb = true;
    return is;
}

// ***************************************************************************
// compare stream content with strings
// ***************************************************************************

istream &operator>>(istream &is, checkstrings cs)
{
    int i = 0;
    bool b;
    while (cs.slist[i][0] != '\0')
    {
        is >> checkstring(cs.slist[i], &b);
        if (b)
        {
            *cs.pi = i;
            return is;
        }
        i++;
    }
    *cs.pi = -1;
    return is;
}

// ***************************************************************************
// compare next line in file with header
// does not read next character like endl!
// ***************************************************************************

bool CheckHeader(istream &ps, char *header)
{
    char c;
    char *pstr = header;

    ps >> ws;
    while (*pstr != '\0')
    {
        c = ps.get();
        if (c != *pstr++)
        {
            ps.putback(c);
            return false;
        }
    }
    return true;
}

// ***************************************************************************
// compare strings w/o capital letters and maximum length
// ***************************************************************************

#ifndef strnicmp
int strnicmp(const char *p1, const char *p2, size_t maxlen)
{
    size_t i = 0;
    int di;

    while (i < maxlen)
    {
        di = tolower(p1[i]) - tolower(p2[i]);
        if (di != 0)
            return di;
        i++;
    }
    return 0;
}
#endif

// ***************************************************************************
// find entry in list of strings
// ***************************************************************************

int find_entry(char *&pentry, const char *elist[])
{
    int i = 0;

    while (*pentry == ' ')
        pentry++;
    while (elist[i][0] != '\0')
    {
        if (strnicmp(pentry, elist[i], strlen(elist[i])) == 0)
        {
            pentry += strlen(elist[i]);
            break;
        }
        i++;
    }
    return i;
}

int strfind_entry(istrstream &sstr, const char *elist[])
{
    int i = 0;
    string s;

    sstr >> ws;

#ifdef __sgi // ugly workaround for broken sgi istrstream
    sstr >> s;
    int m, n = s.length();
    for (m = 0; m < n; m++)
        sstr.putback(s[n - m - 1]);

#else
    long ipos;
    ipos = sstr.tellg();
    sstr >> s;
    //sstr.clear();
    sstr.seekg(ipos, ios::beg);
#endif
    while (elist[i][0] != '\0')
    {
        if (strnicmp(s.c_str(), elist[i], strlen(elist[i])) == 0)
        {
#ifdef __sgi
            int m, n = strlen(elist[i]);
            char c;
            for (m = 0; m < n; m++)
                sstr >> c;
#else
            sstr.seekg(ipos + strlen(elist[i]));
#endif
            break;
        }
        i++;
        //sstr.seekg(ipos);
    }
    return i;
}

const char *argslist[] = { "-in", "-out", "-h", "-?", "/h", "/?", "" };

// ***************************************************************************
// parse program arguments
// ***************************************************************************

int parse_args(int argc, char *argv[])
{
    int i = 1;
    char *pstr;

    while (i < argc)
    {
        pstr = argv[i];
        switch (find_entry(pstr, argslist))
        {
        default:
            break;

        case 0: // -in
            if (*pstr == '\0')
            {
                i++;
                pstr = argv[i];
            }
            inname = pstr;
            break;

        case 1: // -out
            if (*pstr == '\0')
            {
                i++;
                pstr = argv[i];
            }
            outname = pstr;
            break;

        case 2: // -h
        case 3: // -?
        case 4: // /h
        case 5: // /?
            cout << "program usage:" << endl;
            cout << "trans3d [-in input file][-out output file]" << endl;
            return -1;
        }
        i++;
    }

    if (outname.length() == 0)
    {
        if (inname.length() == 0)
            outname = "trans3d.spj";
        else
        {
            outname = inname;
            if ((i = outname.find('.')) >= 0)
                outname.erase(i);
            outname += ".spj";
        }
    }
    if (abortname.length() == 0)
    {
        abortname = outname;
        if ((i = outname.find('.')) >= 0)
            abortname.erase(i);
        abortname += ".end";
    }
    return 0;
}

// ***************************************************************************
// updating all variables
// ***************************************************************************

void UpdateVariables()
{
    Material.Update();
    Laser.Update();
    Grid.Update();
    Simulation.Update();
}

// ***************************************************************************
// switch all variables to non-dimensional form
// reference scales should be set earlier
// ***************************************************************************

void MakeNonDimensional()
{
    Material.MakeNonDimensional();
    Laser.MakeNonDimensional();
    Grid.MakeNonDimensional();
    Simulation.MakeNonDimensional();
}

// ***************************************************************************
// switch all variables to dimensional form
// ***************************************************************************

void MakeDimensional()
{
    Material = Material.GetDimensional();
    Laser = Laser.GetDimensional();
    Grid = Grid.GetDimensional();
    Simulation = Simulation.GetDimensional();
}

// ***************************************************************************
// initialize streams
// ***************************************************************************

void InitStream()
{
    ofstream *pof;
#ifndef GUI
    pout = &cout;
    perr = &cerr;
#endif
    pof = new ofstream("protocol.txt");
    //  pof->close();
    plog = pof;
    pof = new ofstream("debug.txt");
    pof->close();
    pdebug = pof;
    pin = &cin;
}

// ***************************************************************************
// read default values from trinput.fil
// ***************************************************************************

void ReadInputFile(const char *filename)
{
    prec xr, yr, zr, xi, yi, zi, cirk, cirn, terdif, cs1, cze;

    *pout << "reading trinput.fil..." << endl;

    ifstream fs(filename);

    if (!fs)
    {
        *perr << "ERROR: couldn't open trinput.fil" << endl;
        return;
    }
    fs >> fc('=') >> Grid.iVolNodes;
    fs >> fc('=') >> Grid.jVolNodes;
    fs >> fc('=') >> Grid.kVolNodes;
    fs >> fc('=') >> Simulation.iTimeSteps;
    fs >> fc('=') >> Simulation.ndTimeEnd;
    fs >> fc('=') >> Simulation.iAveRays;
    fs >> fc('=') >> Simulation.bMultReflec;
    fs >> fc('=') >> Simulation.bSpecReflec;
    fs >> fc('=') >> Simulation.iStatus;
    fs >> fc('='); // igraph
    fs >> fc('='); // ipltgrd
    fs >> fc('='); // iscreen
    fs >> fc('='); // icross
    fs >> fc('='); // jcross
    fs >> fc('='); // iwrite
    fs >> fc('=') >> Grid.iLeftBound;
    fs >> fc('=') >> Grid.iRightBound;
    fs >> fc('=') >> Grid.iFrontBound;
    fs >> fc('=') >> Grid.iBackBound;
    fs >> fc('='); // uvel
    fs >> fc('='); // rtrep
    fs >> fc('=') >> Material.LatentVapor;
    fs >> fc('=') >> Laser.RefIntensity;
    fs >> fc('=') >> Material.AbsCoeff;
    fs >> fc('=') >> Laser.ndInitialPos.z;
    fs >> fc('=') >> Laser.Divergence;
    fs >> fc('=') >> Simulation.iReflection;
    fs >> fc('=') >> cirn;
    fs >> fc('=') >> cirk;
    fs >> fc('=') >> xr;
    fs >> fc(',') >> yr;
    fs >> fc(',') >> zr;
    fs >> fc('=') >> xi;
    fs >> fc(',') >> yi;
    fs >> fc(',') >> zi;
    fs >> fc('='); // idmatrl
    fs >> fc('='); // cza
    fs >> fc('=') >> cze;
    fs >> fc('='); // czi
    fs >> fc('=') >> Laser.ndWavelength;
    fs >> fc('=') >> Grid.ndWpLength;
    fs >> fc('=') >> Grid.ndWpWidth;
    fs >> fc('=') >> Laser.ndInitialPos.x;
    fs >> fc('=') >> Laser.ndInitialPos.y;
    fs >> fc('=') >> Grid.ndXFront;
    fs >> fc('=') >> Grid.ndXBack;
    fs >> fc('=') >> Grid.ndYSide;
    fs >> fc('=') >> Grid.ndSpacing;
    fs >> fc('=') >> Grid.ndDepth;
    fs >> fc('='); // xgmin
    fs >> fc('='); // xgmax
    fs >> fc('='); // ygmin
    fs >> fc('='); // ygmax
    fs >> fc('='); // zgmax
    fs >> fc('=') >> cs1;
    fs >> fc('=') >> Simulation.ndRefRadius;
    fs >> fc('=') >> Grid.TangentWarp;
    fs >> fc('=') >> Grid.TangentDir;
    fs >> fc('=') >> Grid.TangentBound;
    fs >> fc('=') >> Grid.iWarpPower;
    fs >> fc('=') >> Grid.iWarpAverage;
    fs >> fc('=') >> Simulation.ndMaxSurfMove;
    fs >> fc('=') >> Simulation.ndTempTol;
    fs >> fc('=') >> Simulation.ndMinVapVelo;
    fs >> fc('=') >> Simulation.ndMaxGridMove;
    fs >> fc('=') >> Simulation.ndTempOff;
    fs >> fc('=') >> Simulation.ndMinTemp;
    fs >> fc('=') >> Simulation.ndMaxZMove;
    fs >> fc('=') >> Simulation.ndMaxXYMove;
    fs >> fc('=') >> Simulation.ndDeltatOn;
    fs >> fc('=') >> Simulation.ndDeltatOff;
    fs >> fc('=') >> Laser.Mode;
    fs >> fc('=') >> Laser.Pulselength;
    fs >> fc('=') >> Laser.iPulseForm;
    fs >> fc('=') >> terdif;
    fs >> fc('=') >> Material.RefConductivity;
    fs >> fc('=') >> Laser.Radius;
    fs >> fc('=') >> Material.VaporTemp;
    fs >> fc('=') >> Material.RoomTemp;
    fs >> fc('=') >> Material.RefSpecHeat;
    fs >> fc('=') >> Laser.ndHeight;
    if (!fs)
        *perr << "ERROR in reading trinput.fil" << endl;

    Laser.EField = TCPoint3D(cmplx(xr, xi), cmplx(yr, yi), cmplx(zr, zi));
    Grid.bGridMove = (cs1 == 1);

    RefLength = Laser.Radius; // non-dimensional
    RefVelocity = terdif / RefLength;
    RefTime = RefLength / RefVelocity;

    Simulation.ndTimeEnd /= RefTime;
    Laser.ndWavelength /= RefLength;
    Simulation.ndDeltatOn /= RefTime;
    Simulation.ndDeltatOff /= RefTime;

    cze *= Material.VaporTemp;
    Material.CriticalTemp = 2 * cze;
    Material.SpecGasConst = Material.LatentVapor / cze;
    Material.RefDensity = Material.RefConductivity / (terdif * Material.RefSpecHeat);
    Material.RefIndex = cmplx(cirn, cirk);
    Grid.iIniNodes = Grid.iVolNodes;
    Grid.jIniNodes = Grid.jVolNodes;
    Grid.kIniNodes = Grid.kVolNodes;

    UpdateVariables();
}

// ***************************************************************************
// read input file (temporary)
// ***************************************************************************

void read_file(const char *pname)
{
    CmFileStruct fs;
    int i, j, k, is, js, ih, jh;
    prec h1, ga1, xt, yt, zt, qd, qr, qi;
    char buffer[200], buffer2[200], *pstr, c;
    string s;

    if (*pname == '\0')
        return;
    cout << "reading " << pname << "..." << endl;
    cout.flush();
    ResetVariables();

    fs.unit = 14;
    if (unitopen(fs.unit, pname) < 0)
        return;
    c = readchar(fs.unit);
    unitclose(fs.unit);
    if (c == 0x1f) // komprimierte Textdatei
    {
        strcpy(buffer2, pname);
        if ((pstr = strrchr(buffer2, '.')) != 0)
            *pstr = '\0';
        /*    strcat(buffer2,".tpj.gz");
      //    rename(pname,buffer2);
      //    sprintf(buffer,"%sgzip.dll",AppPath);
      //    HINSTANCE   hInst = LoadLibrary(buffer);
      //    PZIPFN      pZipFn = (PZIPFN) GetProcAddress(hInst,"_Decompress");
      //    pZipFn(buffer2);
      //    FreeLibrary(hInst);
          if((pstr = strrchr(buffer2,'.'))!=0)
            *pstr = '\0';*/
        if (unitopen(fs.unit, buffer2) < 0)
            return;
    }
    else if (unitopen(fs.unit, pname) < 0)
        return;
    unitreadln(fs.unit, buffer); // Dateibezeichnung

    if (strncmp(buffer, "Trans3D Simulationsprojekt",
                strlen("Trans3D Simulationsprojekt")) != 0)
        ErrorFunction("wrong inputfile format");
    pstr = buffer + strlen("Trans3D Simulationsprojekt");
    if (*pstr != '\0')
        sscanf(pstr, " Version %f", &fs.vers);
    else
        fs.vers = 1.0;
    unitreadln(fs.unit, buffer); // /n
    unitreadln(fs.unit, buffer); // /n
    unitreadln(fs.unit, buffer); // Projekteinstellungen

    Simulation.iStatus = TSimulation::loaded;
    Laser.Read(fs.unit, fs.vers);
    Material.Read(fs.unit, fs.vers);
    Laser.Update(); // correct reference values
    Grid.Read(fs.unit, fs.vers);
    Simulation.Read(fs.unit, fs.vers);

    unitreadln(fs.unit, buffer);
    unitreadln(fs.unit, buffer);
    unitreadln(fs.unit, buffer);
    unitreadln(fs.unit, buffer);
    unitreadln(fs.unit, buffer);
    unitreadln(fs.unit, buffer);
    Simulation.ndTime = readreal(fs.unit);
    Simulation.ndPulseTime = readreal(fs.unit);
    Laser.ndRelPower = readreal(fs.unit);
    Grid.iSurNodes = readint(fs.unit);
    Grid.jSurNodes = readint(fs.unit);
    Grid.iVolMin = readint(fs.unit);
    Grid.iVolMax = readint(fs.unit);
    Grid.jVolMin = readint(fs.unit);
    Grid.jVolMax = readint(fs.unit);

    Grid.ResizeArrays();
    UpdateVariables();

    unitreadln(fs.unit, buffer);
    unitreadln(fs.unit, buffer);
    unitreadln(fs.unit, buffer);
    for (i = 1; i <= Grid.iVolNodes; i++)
    {
        readint(fs.unit, false);
        Solid.jBegin(i) = readint(fs.unit);
        Solid.jEnd(i) = readint(fs.unit);
    }

    unitreadln(fs.unit, buffer);
    unitreadln(fs.unit, buffer);
    unitreadln(fs.unit, buffer);
    for (j = 0; j <= Grid.jVolNodes + 1; j++)
        for (i = 0; i <= Grid.iVolNodes + 1; i++)
            for (k = 1; k <= Grid.kVolNodes; k++)
            {
                Solid(i, j, k).ndNode.x = readreal(fs.unit);
                Solid(i, j, k).ndNode.y = readreal(fs.unit);
                Solid(i, j, k).ndNode.z = readreal(fs.unit);
                Solid(i, j, k).ndTemp = readreal(fs.unit);
                Solid(i, j, k).ndHeat = readreal(fs.unit);
            }

    unitreadln(fs.unit, buffer);
    unitreadln(fs.unit, buffer);
    unitreadln(fs.unit, buffer);

    Solid.ndMinSurZ = 1e10;
    for (js = 0; js <= Grid.jSurNodes + 1; js++)
    {
        jh = js + 1 - Grid.jVolMin;
        for (is = 0; is <= Grid.iSurNodes + 1; is++)
        {
            ih = is + 1 - Grid.iVolMin;
            Surface(is, js).x = readreal(fs.unit);
            Surface(is, js).y = readreal(fs.unit);
            Surface(is, js).z = readreal(fs.unit);
            h1 = readreal(fs.unit);
            xt = readreal(fs.unit);
            yt = readreal(fs.unit);
            zt = readreal(fs.unit);
            qd = readreal(fs.unit);
            qr = readreal(fs.unit);
            ga1 = readreal(fs.unit);
            switch ((int)(fs.vers * 100 + 0.1))
            {
            default:
                qi = readreal(fs.unit);
                break;

            case 103:
                qi = 0;
                break;
            }
            if (!(ih < 1 || ih > Grid.iVolNodes || jh < 1 || jh > Grid.jVolNodes))
            {
                SolSurf(ih, jh).ndDirAbs = qd;
                SolSurf(ih, jh).ndMultAbs = qr;
                SolSurf(ih, jh).ndIncident = qi;
                Solid(ih, jh, 1).ndHeat = ga1;
                Solid(ih, jh, 1).ndTemp = h1;
                SolSurf(ih, jh).xtau.x = xt;
                SolSurf(ih, jh).xtau.y = yt;
                SolSurf(ih, jh).xtau.z = zt;
                if (Solid(ih, jh, 1).ndNode.z < Solid.ndMinSurZ)
                    Solid.ndMinSurZ = Solid(ih, jh, 1).ndNode.z;
            }
        }
    }

    Solid.ndVapRate = readreal(fs.unit);
    Solid.ndMaxTemp = readreal(fs.unit);
    SolSurf.ndTotalIn = readreal(fs.unit);
    SolSurf.ndTotalDirAbs = readreal(fs.unit);
    SolSurf.ndTotalMultAbs = readreal(fs.unit);
    readreal(fs.unit); // reflected

    Grid.xDistance = Surface(0, 1).x - Surface(1, 1).x;
    Grid.yDistance = Surface(1, 1).y - Surface(1, 0).y;
    do
    {
        getline(*fp[fs.unit], s, '\n');
#ifndef WIN32 // get rid of tailing \n and \rs
        char c = s[s.length() - 1];
        while ((c == '\r') || (c == '\n'))
        {
            s.erase(s.end() - 1, s.end());
            if (s.length() == 0)
                break;
            c = s[s.length() - 1];
        }
#endif
        if (s == "Skriptfenster:")
        {
            getline(*fp[fs.unit], s, '\n');
            getline(*fp[fs.unit], s, '\n');
            getline(*fp[fs.unit], s, '\n');
            getline(*fp[fs.unit], s, '\n');
            script.Read(fs.unit);
        }
    } while (fp[fs.unit]->good());
    //  ReadAll(&fs);

    unitclose(fs.unit);
    if (c == 0x1f) // Textdatei wieder komprimieren
    {
        sprintf(buffer, "%sgzip.dll", AppPath);
        //    HINSTANCE   hInst = LoadLibrary(buffer);
        //    PZIPFN      pZipFn = (PZIPFN) GetProcAddress(hInst,"_Compress");
        //    pZipFn(buffer2);
        //    FreeLibrary(hInst);
        strcat(buffer2, ".gz");
        rename(buffer2, pname);
    }

    Laser.bChanged = false;
    Material.bChanged = false;
    Grid.bChanged = false;
    Simulation.bChanged = false;
    Simulation.Initialize();
}

// ***************************************************************************
// write output file (temporary)
// ***************************************************************************

void write_file(const char *pname)
{
    CmFileStruct fs;
    int i, j, k, is, js, ih, jh;
    prec h1, ga1, xt, yt, zt, qd, qr, qi;
    const char *pstr; //, buffer[200];

    cout << "writing " << pname << "..." << endl;
    cout.flush();
    fs.unit = 14;
#ifdef GUI
    sscanf(VERSION, "%f", &fs.vers);
#else
    fs.vers = (float)VERSION;
#endif

    /*  if()
       {
       strcpy(buffer,pname);
       if((pstr = strrchr(buffer,'.'))!=0)
         *pstr = '\0';
       strcat(buffer,".tpj");
       pstr = buffer;
       }
     else*/
    pstr = pname;
    if (unitopen(fs.unit, pstr, ios::out) < 0)
        return;
    unitwriteln(fs.unit, "Trans3D Simulationsprojekt Version %f\n\n", VERSION);
    unitwriteln(fs.unit, "Projekteinstellungen:");

    Laser.Save(fs.unit);
    Material.Save(fs.unit);
    Grid.Save(fs.unit);
    Simulation.Save(fs.unit);

    unitwriteln(fs.unit, "\n\nSimulationsergebnis:\n");

    unitwriteln(fs.unit, "\ninterne Variablen:");
    unitwrite(fs.unit, Simulation.ndTime, "Rechenzeit =\t%le\n");
    unitwrite(fs.unit, Simulation.ndPulseTime, "Zeit ab Pulsbeginn =\t%le\n");
    unitwrite(fs.unit, Laser.ndRelPower, "aktuelle Pulsstärke =\t%le\n");
    unitwrite(fs.unit, Grid.iSurNodes, "x-Oberflächengitter =\t%i\n");
    unitwrite(fs.unit, Grid.jSurNodes, "y-Oberflächengitter =\t%i\n");
    unitwrite(fs.unit, Grid.iVolMin, "xmin Volumengitter =\t%i\n");
    unitwrite(fs.unit, Grid.iVolMax, "xmax Volumengitter =\t%i\n");
    unitwrite(fs.unit, Grid.jVolMin, "ymin Volumengitter =\t%i\n");
    unitwrite(fs.unit, Grid.jVolMax, "ymax Volumengitter =\t%i\n");

    unitwriteln(fs.unit, "\nWärmeeinflußzone:");
    unitwriteln(fs.unit, "\tymin\tymax");
    for (i = 1; i <= Grid.iVolNodes; i++)
    {
        unitwrite(fs.unit, i, "Knoten %i:\t");
        unitwrite(fs.unit, Solid.jBegin(i), "%i\t");
        unitwrite(fs.unit, Solid.jEnd(i), "%i\n");
    }

    unitwriteln(fs.unit, "\nVolumengitter:");
    unitwriteln(fs.unit, "x\ty\tz\tT\tQl");
    for (j = 0; j <= Grid.jVolNodes + 1; j++)
        for (i = 0; i <= Grid.iVolNodes + 1; i++)
            for (k = 1; k <= Grid.kVolNodes; k++)
            {
                unitwrite(fs.unit, Solid(i, j, k).ndNode.x, "%le\t");
                unitwrite(fs.unit, Solid(i, j, k).ndNode.y, "%le\t");
                unitwrite(fs.unit, Solid(i, j, k).ndNode.z, "%le\t");
                unitwrite(fs.unit, Solid(i, j, k).ndTemp, "%le\t");
                unitwrite(fs.unit, Solid(i, j, k).ndHeat, "%le\n");
            }

    if (Simulation.ndDeltat == 0)
        Simulation.ndDeltat = Simulation.ndDeltatOn;
    unitwriteln(fs.unit, "\nOberflächengitter:");
    unitwriteln(fs.unit, "x\ty\tz\tT\txt\tyt\tzt\tQdir\tQref\tQl");
    for (js = 0; js <= Grid.jSurNodes + 1; js++)
    {
        jh = js + 1 - Grid.jVolMin;
        for (is = 0; is <= Grid.iSurNodes + 1; is++)
        {
            ih = is + 1 - Grid.iVolMin;
            if (ih < 1 || ih > Grid.iVolNodes || jh < 1 || jh > Grid.jVolNodes)
            {
                qi = 0;
                qd = 0;
                qr = 0;
                ga1 = 0;
                h1 = 0;
                xt = 0;
                yt = 0;
                zt = 0;
            }
            else
            {
                qi = SolSurf(ih, jh).ndIncident;
                qd = SolSurf(ih, jh).ndDirAbs;
                qr = SolSurf(ih, jh).ndMultAbs;
                ga1 = Solid(ih, jh, 1).ndHeat;
                h1 = Solid(ih, jh, 1).ndTemp;
                xt = SolSurf(ih, jh).xtau.x;
                yt = SolSurf(ih, jh).xtau.y;
                zt = SolSurf(ih, jh).xtau.z;
            }
            unitwrite(fs.unit, (double)Surface(is, js).x, "%le\t");

            unitwrite(fs.unit, (double)Surface(is, js).y, "%le\t");
            unitwrite(fs.unit, (double)Surface(is, js).z, "%le\t");
            unitwrite(fs.unit, (double)h1, "%le\t");
            unitwrite(fs.unit, (double)xt, "%le\t");
            unitwrite(fs.unit, (double)yt, "%le\t");
            unitwrite(fs.unit, (double)zt, "%le\t");
            unitwrite(fs.unit, (double)qd, "%le\t");
            unitwrite(fs.unit, (double)qr, "%le\t");
            unitwrite(fs.unit, (double)ga1, "%le\n");
            unitwrite(fs.unit, (double)qi, "%le\n");
        }
    }

    unitwrite(fs.unit, Solid.ndVapRate, "Verdampfungsrate =\t%le\n");
    unitwrite(fs.unit, Solid.ndMaxTemp, "Maximaltemperatur =\t%le\n");
    unitwrite(fs.unit, SolSurf.ndTotalIn, "Leistung einfallend =\t%le\n");
    unitwrite(fs.unit, SolSurf.ndTotalDirAbs, "Absorbiert direkt =\t%le\n");
    unitwrite(fs.unit, SolSurf.ndTotalMultAbs, "Absorbiert indirekt =\t%le\n");
    unitwrite(fs.unit, 0., "Reflektiert =\t%le\n");

    script.Save(fs.unit);
    //  SaveAll(&fs);

    unitclose(fs.unit);

    Laser.bChanged = false;
    Material.bChanged = false;
    Grid.bChanged = false;
    Simulation.bChanged = false;

    /*  if(ProjectFile->FilterIndex!=1)
       return;

   // Komprimieren und umbenennen

     char  buffer2[200];
     sprintf(buffer2,"%sgzip.dll",AppPath);
     HINSTANCE   hInst = LoadLibrary(buffer2);
     PZIPFN      pZipFn = (PZIPFN) GetProcAddress(hInst,"_Compress");
     pZipFn(buffer);
     FreeLibrary(hInst);
   remove(ProjectFile->FileName);
   strcat(buffer,".gz");
   rename(buffer,ProjectFile->FileName);*/
}

// ***************************************************************************
// write global variables to temporary output file (for debug uses)
// ***************************************************************************

void WriteGlobalVariables()
{
    ofstream os("global.txt");

    os << Simulation.ndTime << ' ' << 0 << ' ' << Solid.ndMaxTemp << ' '
       << Laser.ndPulselength << ' ' << Solid.ndVapRate << ' '
       << Laser.iRays << ' ' << Laser.iBadRays << endl;
    os << "cza: " << Material.ndVapRate << " cze: " << Material.ndLatent
       << " czi: " << 0 << endl;
    os << "dnk: " << 1. / Laser.ndPower << " dste: " << Material.RefStefan
       << " dnw: " << Laser.ndPosition.z << endl;
    os << "il: " << Grid.iVolNodes << " jl: " << Grid.jVolNodes << " kl: "
       << Grid.kVolNodes << endl;
    os << "br: " << Laser.Radius << " cirk: " << Material.RefIndex.imag()
       << " cirn: " << Material.RefIndex.real() << endl;
    os << "depth: " << Grid.ndDepth << " opthick: " << 0 << " ptotal: "
       << Laser.ndPulselength << endl;
    os << "spheat: " << Material.RefSpecHeat << " tercon: "
       << Material.RefConductivity << "terdif: " << Material.RefDiffusivity
       << endl;
    os << "vratec: " << 1 / Laser.ndRate << " qbundlneg: "
       << Simulation.ndMinPower << " pon: " << Laser.ndPulseOn << endl;
    os << "ck1: " << Grid.TangentWarp << " ck2: " << Grid.TangentDir
       << " ck3: " << Grid.TangentBound << endl;
    os << "cs1: " << Grid.bGridMove << " cs2: " << Simulation.ndRefRadius
       << " grdmvtol: " << Simulation.ndMaxSurfMove << endl;
    os << "gtol: " << Simulation.ndMaxGridMove << " Simulation.hnegtol: "
       << Simulation.ndMinTemp << " hofftol: " << Simulation.ndTempOff << endl;
    os << "htol: " << Simulation.ndTempTol << " smin: " << Grid.ndSpacing
       << " depth: " << Grid.ndDepth;
    os << "xymvtol: " << Simulation.ndMaxXYMove << " ztamvtol: "
       << Simulation.ndMaxZMove << endl;
}

// ***************************************************************************
// test for user break
// ***************************************************************************

bool ShouldAbort()
{
    ifstream abstream(abortname.c_str());
    return (abstream.is_open() != 0);
}

// ***************************************************************************
// run simulation
// ***************************************************************************

int CalculateAblation()
{
    int i, nstep, iAutoSave, rv;
    char buffer[200];
    prec ress, resd, dtmax, dimtime;

    if (ShouldAbort())
        return ERR_ABORTED;

    *pout << "starting calculation..." << endl;
    try
    {
        if ((rv = Simulation.Initialize()) < 0)
            return rv;
        WriteGlobalVariables();

        srand((unsigned)time(NULL));
        *pout << "    Time     Step#    Max dT      Max T     AvePower"
                 "   RemRate  # Rays Bad Rays" << endl;
        *plog << "    Time     Step#    Max dT      Max T     AvePower"
                 "   RemRate  # Rays Bad Rays" << endl;
        iAutoSave = (int)(Simulation.ndTime / Simulation.ndAutoSaveTime + 1e-12);

        nstep = 0;
        while ((Simulation.iTimeSteps > 0 && nstep < Simulation.iTimeSteps) || (Simulation.iTimeSteps == 0 && Simulation.ndTime / Simulation.ndTimeEnd - 1.0 < -1.0e-12))
        {
            nstep++;
            do
            {
                if ((rv = Simulation.NextTimeStep()) < 0)
                    return rv;
                //        Grid.iGridMove = TGrid::no_update;
                //        WriteGlobalVariables();
                if ((rv = Solid.Solve(ress, resd, dtmax)) < 0)
                    return rv;
                Grid.UpdateGrid();
            } while (Grid.iGridMove == TGrid::reduced_step);

            dimtime = Simulation.ndPulseTime * RefTime;
#ifdef WIN32
            sprintf(buffer, "%10.6le %4i  %10.3le %10.3le %10.3le %10.3le %7i %7i",
#else
            sprintf(buffer, "%10.6e %4i  %10.3e %10.3e %10.3e %10.3e %7i %7i",
#endif
                    dimtime, nstep, dtmax, Solid.ndMaxTemp,
                    Laser.ndAvePower, Solid.ndVapRate, Laser.iRays, Laser.iBadRays);
            *pout << string(buffer) << endl;
            *plog << string(buffer) << endl;
#ifdef GUI
            papp->processEvents(100);
#endif
            if (ShouldAbort())
                return ERR_ABORTED;

            if (Simulation.bAutoSave)
            {
                if (Simulation.iAutoSaveSteps == 0)
                {
                    if ((i = (int)(Simulation.ndTime / Simulation.ndAutoSaveTime + 1e-12)) > iAutoSave)
                        iAutoSave = i;
                }
                else if (nstep % Simulation.iAutoSaveSteps == 0)
                    iAutoSave++;
            }
        }
    }
    catch (TException &ex)
    {
        Simulation.RestoreLastStep();
        ex.Display();
        return -1;
    }
    catch (...)
    {
        Simulation.RestoreLastStep();
        ErrorFunction("Unbekannter Programmfehler");
        return -1;
    }

    return 0;
}

// ***************************************************************************
// set global variables at start
// ***************************************************************************

void InitVarsOnce(void)
{
    const int viadj[] = { 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, 0, 1 };
    const int vjadj[] = { -1, 0, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1 };
    const int vic[] = { 1, 1, 0, 0 };
    const int vjc[] = { 0, 1, 1, 0 };
    const int vidface[] = { 0, 2, 0, 3, 1, 7, 1, 6, 4, 7, 5, 6, 2, 7, 5, 4, 6, 5, 3, 3, 1, 0, 2, 4 };
    int i, j;

    for (i = 0; i < 3; i++)
        for (j = 0; j < 4; j++)
        {
            AdjPatches(j, i).x = viadj[i + j * 3];
            AdjPatches(j, i).y = vjadj[i + j * 3];
        }

    for (i = 0; i < 4; i++)
    {
        ptCorners(i).x = vic[i];
        ptCorners(i).y = vjc[i];
    }

    for (i = 0; i < 6; i++)
        for (j = 0; j < 4; j++)
            FaceEdges(i, j) = vidface[i + j * 6];
}

// ***************************************************************************
// reset global variables
// ***************************************************************************

void ResetVariables()
{
    int i;

    Solid.ResetTemperature();
    idum = -217579;
    iy = 0;
    for (i = 0; i <= ntab; i++)
        iv(i) = 0;

    for (i = 0; i < MAXPREV; i++)
        ptPrev(i) = 0;

    Grid.Reset();
    Laser.Reset();
    Material.Reset();
    Simulation.Reset();
}

// ***************************************************************************
// program initialization
// ***************************************************************************

void init_program(const char *filename)
{
    init_stream();
    InitVarsOnce();
    ResetVariables();
    Simulation.iStatus = TSimulation::new_grid;
    ReadInputFile(filename);
}

// ***************************************************************************
// script interpreter
// ***************************************************************************

int TScript::execute()
{
    string sLine;
    int i, iPosition;

    if (iLine < 0)
        iLine = 0;
    iPosition = getLineIndex(iLine); // find offset of line in array
    while (iPosition < int(sScript.length()))
    {
        i = sScript.find('\n', iPosition);
        if (i == int(string::npos))
            i = sScript.length();
        sLine = sScript.substr(iPosition, i - iPosition);
        iPosition = ++i;
        i = executeCommand(sLine);
        if (i < 0)
        {
            if (i == SyntaxError)
                ErrorFunction("ERROR: syntax error in script line %i", iLine);
            break;
        }
        iLine++;
        if (i == START)
        {
            return i;
        }
        if (i == JumpBracket)
            getBracketStart(iPosition);
    }
    if (iPosition >= int(sScript.length())) // end of script
        iLine = -1;
    return i;
}

// ***************************************************************************
// line interpreter
// ***************************************************************************

// list of script commands

const char *CommandList[] = {
    "start", "//", "for", "int", "real", "timesteps",
    "timeend", "to", "{", "}", "deltaton", "deltatoff",
    "break", ""
};

int TScript::executeCommand(string &s)
{
    map<int, int>::iterator it1, it2;
    int i, j;
    prec p;
    istrstream sstr(s.c_str());

    sstr >> ws;
    if (sstr.peek() == -1)
        return 0;
    switch (strfind_entry(sstr, CommandList))
    {
    default: // unknown command
        ErrorFunction("ERROR: unknown command in script line %i", iLine);
        return Error;

    case 0: // start calculation
#ifdef COVISE
        return START;
#else
        i = CalculateAblation();
        Simulation.iStatus = TSimulation::simulated;
        return i;
#endif

    case 1: // comment
        return 0;

    case 2: // for loop
        if (strfind_entry(sstr, CommandList) != 3) // != int
            return SyntaxError;
        sstr >> i; // number of integer variable
        sstr >> fc('=');
        sstr >> j; // value
        if (sstr.fail())
            return SyntaxError;
        ivariables[i] = j;
        if (strfind_entry(sstr, CommandList) != 7) // != to
            return SyntaxError;
        sstr >> j;
        if (sstr.fail())
            return SyntaxError;
        iloopvar[iBracket] = i; // loop variable
        iloopend[iBracket] = j; // end of loop
        break;

    case 3: // integer variable
        sstr >> i; // number of variable
        sstr >> fc('=');
        j = int(getExpression(sstr));
        if (sstr.fail())
            return SyntaxError;
        ivariables[i] = j;
        break;

    case 4: // prec variable
        sstr >> i; // number of variable
        sstr >> fc('=');
        p = getExpression(sstr);
        if (sstr.fail())
            return SyntaxError;
        pvariables[i] = p;
        break;

    case 5: // number of time steps
        sstr >> fc('=');
        j = int(getExpression(sstr));
        if (j >= 0)
            Simulation.iTimeSteps = j;
        break;

    case 6: // time for simulation end
        sstr >> fc('=');
        p = getExpression(sstr);
        if (p >= 0)
            Simulation.ndTimeEnd = p / RefTime;
        break;

    case 8: // {
        iBracket++;
        break;

    case 9: // }
        if (iBracket == 0)
        {
            ErrorFunction("ERROR: too many closed brackets in script line %i",
                          iLine);
            return Error;
        }
        iBracket--;
        // for loop
        if ((it1 = iloopvar.find(iBracket)) != iloopvar.end())
        {
            i = (*it1).second;
            it2 = iloopend.find(iBracket);
            j = (*it2).second;

            ivariables[i]++;
            if (ivariables[i] > j) // end of loop
            {
                iloopvar.erase(it1);
                iloopend.erase(it2);
            }
            else // jump to loop start
                return JumpBracket;
        }
        break;

    case 10: // time step pulse on
        sstr >> fc('=');
        p = getExpression(sstr);
        if (p >= 0)
            Simulation.ndDeltatOn = p / RefTime;
        break;

    case 11: // time step pulse off
        sstr >> fc('=');
        p = getExpression(sstr);
        if (p >= 0)
            Simulation.ndDeltatOff = p / RefTime;
        break;

    case 12: // break
        iLine++; // start again at next line
        return Aborted;
    }

    return Ok;
}

// get number of lines

int TScript::getNumLines() const
{
    int i = 1, j = sScript.length();

    if (j < 1)
        return 0;
    j--;
    j--;
    while (j >= 0)
    {
        if (sScript[j] == '\n')
            i++;
        j--;
    }

    return i;
}

// find position of line i in sScript

int TScript::getLineIndex(int i)
{
    int ipos = 0, j = 0, k;

    if (i <= 0)
        return 0;
    while ((k = sScript.find('\n', ipos)) >= 0)
    {
        ipos = k + 1;
        j++;
        if (j == i)
            break;
    }
    return ipos;
}

// find position and line of opening bracket iBracket

void TScript::getBracketStart(int &ipos)
{
    int i, k = iBracket;

    while (ipos > 0)
    {
        ipos--;
        if (sScript[ipos] == '\n')
            iLine--;
        else if (sScript[ipos] == '}')
            k++;
        else if (sScript[ipos] == '{')
        {
            k--;
            if (k == iBracket) // starting bracket found
            {
                i = sScript.rfind('\n', ipos);
                if (i >= 0)
                    ipos = i + 1;
                else
                {
                    iLine = 0;
                    ipos = 0;
                }
                return;
            }
        }
    }
    iLine = -1; // not found
    ipos = sScript.length();
}

// read number

prec TScript::getValue(istrstream &sstr)
{
    prec d;
    int i;
    char ch;

    sstr >> ws;
    ch = sstr.peek();
    switch (ch | 0x20)
    {
    case 'i':
        if (strfind_entry(sstr, CommandList) != 3) // int
            return 0;
        sstr >> i;
        d = ivariables[i];
        break;

    case 'r':
        if (strfind_entry(sstr, CommandList) != 4) // real
            return 0;
        sstr >> i;
        d = pvariables[i];
        break;

    default:
        sstr >> d;
    }
    return d;
}

// evaluate expression

prec TScript::getExpression(istrstream &sstr)
{
    prec d1, d2;
    char ch;

    d1 = getValue(sstr);
    sstr >> ws;
    sstr >> ch;
    if (!sstr)
        return d1;
    d2 = getValue(sstr);
    switch (ch)
    {
    case '+': // +
        return d1 + d2;

    case '-': // -
        return d1 - d2;

    case '*': // *
        return d1 * d2;

    case '/': // /
        return d1 / d2;
    }
    return 0;
}

// write script

ostream &operator<<(ostream &os, const TScript &scr)
{
    int imax = scr.getNumLines();

    os << endl;
    os << "script:" << endl;
    os << "number of lines:\t" << imax << endl;
    os << scr.sScript;
    if (scr.sScript[scr.sScript.length() - 1] != '\n')
        os << endl;
    os << "current line:\t" << scr.iLine << endl;
    os << "current bracket:\t" << scr.iBracket << endl;
    os << "integer variables:" << endl;
    os << scr.ivariables;
    os << "real variables:" << endl;
    os << scr.pvariables;
    os << "loop variables:" << endl;
    os << scr.iloopvar;
    os << "loop end:" << endl;
    os << scr.iloopend;
    return os << endl;
}

// read script

istream &operator>>(istream &is, TScript &scr)
{
    bool b;
    int i, imax;
    string s;

    is >> checkstring("script:", &b);
    if (!b)
    {
        is.setstate(ios::failbit);
        return is;
    }
    is >> tab >> imax >> endl;
    if (imax > 0)
    {
        i = 1;
        getline(is, scr.sScript);
        while (i < imax)
        {
            scr.sScript += '\n';
            getline(is, s);
            scr.sScript += s;
            i++;
        }
    }
    else
        scr.sScript = "START";
    is >> tab >> scr.iLine;
    is >> tab >> scr.iBracket >> endl;
    is >> endl;
    is >> scr.ivariables;
    is >> endl;
    is >> scr.pvariables;
    is >> endl;
    is >> scr.iloopvar;
    is >> endl;
    is >> scr.iloopend >> endl;
    return is;
}

// Inhalt Schreiben (temporär)

void TScript::Save(int unit)
{
    int i, imax;
    //  string  s;

    imax = getNumLines();

    unitwriteln(unit, "Skriptfenster:\n\n\n\n");
    unitwrite(unit, imax, "Zeilenanzahl:\t%i\n");
    *fp[unit] << sScript;
    unitwrite(unit, iLine, "aktuelle Zeile:\t%i\n");
    unitwrite(unit, iBracket, "Klammern:\t%i\n");

    for (i = 0; i < iBracket; i++)
    {
        *fp[unit] << iloopvar[i] << '\t' << iloopend[i] << '\t' << 0 << endl;
    }
    for (i = 0; i < 10; i++)
    {
        *fp[unit] << ivariables[i] << '\t' << pvariables[i] << endl;
    }
    /*  *fp[unit] << "integer variables";
   //  *fp[unit] << ivariables;
     *fp[unit] << "real variables";
   //  *fp[unit] << pvariables;
     *fp[unit] << "loop variables";
   //  *fp[unit] << iloopvar;
     *fp[unit] << "loop end";
   //  *fp[unit] << iloopend;*/
}

// Inhalt lesen (temporär)

void TScript::Read(int unit)
{
    int i, imax;
    string s;

    sScript = "";
    imax = readint(unit);
    i = 0;
    while (i < imax)
    {
        getline(*fp[unit], s, '\n');
#ifndef WIN32 // get rid of tailing \n and \rs
        char c = s[s.length() - 1];
        while ((c == '\r') || (c == '\n'))
        {
            s.erase(s.end() - 1, s.end());
            if (s.length() == 0)
                break;
            c = s[s.length() - 1];
        }
#endif
        sScript += s + '\n';
        i++;
    }
    iLine = readint(unit);
    iBracket = readint(unit);
    for (i = 0; i < iBracket; i++)
    {
        iloopvar[i] = readint(unit);
        iloopend[i] = readint(unit);
        readint(unit);
    }
    for (i = 0; i < 10; i++)
    {
        ivariables[i] = readint(unit);
        pvariables[i] = readreal(unit);
    }
    //  iLine = -1;
    //  iBracket = 0;
}

// ***************************************************************************
// MAIN PROGRAM
// ***************************************************************************

int main(int argc, char *argv[])
{
    int iret = 0;

    InitStream();
#ifdef COVISE
    if ((argc == 7) || (argc == 8))
    {
        covise.run(argc, argv);
    }
    else
#endif
    {
#ifdef GUI
        QApplication app(argc, argv);
        papp = &app;
        app.setFont(QFont("helvetica", 12));
        Trans3D_GUIApp *trans3d_gui = new Trans3D_GUIApp();
        app.setMainWidget(trans3d_gui);
        pout = trans3d_gui->getTextView();
        perr = pout;
#endif
        if (parse_args(argc, argv) < 0)
        {
            pin->get();
            return 0;
        }
        try
        {
#ifdef GUI
            trans3d_gui->show();
            if (inname.length() == 0)
            {
                trans3d_gui->openDocumentFile();
                init_program("trinput.fil");
            }
            else
                trans3d_gui->openDocumentFile(inname.c_str());
            iret = app.exec();
#else
            init_program("trinput.fil");
            read_file(inname.c_str()); // read input file

            if (script.execute() == ERR_ABORTED)
                remove(abortname.c_str()); // delete abort file

            write_file(outname.c_str()); // write output file
#endif
        }
        catch (TException &ex)
        {
            ex.Display();
            iret = -1;
        }
    }
    cout << "done!" << endl;
    pin->get();
    return iret;
}
