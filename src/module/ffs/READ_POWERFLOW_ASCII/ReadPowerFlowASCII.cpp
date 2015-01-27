/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 2000					*
 *                 VirCinity IY-Consulting GmbH				*
 *                         Nobelstrasse 15				*
 *                        D-70569 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			ReadTecplot.C	 	                *
 *									*
 *	Description	        Tecplot file reader			*
 *									*
 *	Author			Tobias Schweickhardt 			*
 *									*
 *	Date			14. 4. 1998				*
 *									*
 *	Status			finished as specified	 		*
 *									*
 ************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <appl/ApplInterface.h>
#include "ReadPowerFlowASCII.h"

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

void main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
}

Application::Application(int argc, char *argv[])
{
    Covise::set_module_description("Tecplot file reader V 1.0b");

    // module parameter filename as a browser
    Covise::add_port(PARIN, "from_ts", "Scalar", "Start Timestep");
    Covise::set_port_default("from_ts", "1");
    Covise::set_port_immediate("from_ts", 1);
    Covise::add_port(PARIN, "to_ts", "Scalar", "End Timestep");
    Covise::set_port_default("to_ts", "1");
    Covise::add_port(PARIN, "fullpath", "Browser", "filename");
    Covise::set_port_default("fullpath", "~");
    Covise::set_port_immediate("fullpath", 1);

    Covise::add_port(PARIN, "grid_x", "Choice", "Select Grid Data (x-axis)");
    Covise::set_port_default("grid_x", "1 (none)");
    Covise::add_port(PARIN, "grid_y", "Choice", "Select Grid Data (y-axis)");
    Covise::set_port_default("grid_y", "1 (none)");
    Covise::add_port(PARIN, "grid_z", "Choice", "Select Grid Data (z-axis)");
    Covise::set_port_default("grid_z", "1 (none)");
    Covise::add_port(PARIN, "data1", "Choice", "Select Vector_x Data");
    Covise::set_port_default("data1", "1 (none)");
    Covise::add_port(PARIN, "data2", "Choice", "Select Vector_y Data");
    Covise::set_port_default("data2", "1 (none)");
    Covise::add_port(PARIN, "data3", "Choice", "Select Vector_z Data");
    Covise::set_port_default("data3", "1 (none)");
    Covise::add_port(PARIN, "data4", "Choice", "Select Output Data");
    Covise::set_port_default("data4", "1 (none)");
    Covise::add_port(PARIN, "data5", "Choice", "Select Output Data");
    Covise::set_port_default("data5", "1 (none)");

    // define module output ports
    Covise::add_port(OUTPUT_PORT, "grid", "Set_StructuredGrid|Set_UnstructuredGrid", "grid");

    Covise::add_port(OUTPUT_PORT, "dataout1", "Set_Float|Set_Float|Set_Vec3|Set_Vec3", "dataout1");
    Covise::add_port(OUTPUT_PORT, "dataout2", "Set_Float|Set_Float", "dataout2");
    Covise::add_port(OUTPUT_PORT, "dataout3", "Set_Float|Set_Float", "dataout3");

    // covise setup
    Covise::init(argc, argv);

    // define callbacks
    Covise::set_quit_callback(Application::quitCallback, this);
    Covise::set_start_callback(Application::executeCallback, this);
    Covise::set_param_callback(Application::paramCallback, this);

    init_Vars();
}

Application::~Application()
{
    // ...
}

void
Application::run()
{

    Covise::main_loop();
}

//..........................................................................

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::executeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->execute(callbackData);
}

void Application::paramCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->paramChange(callbackData);
}

//..........................................................................

void Application::init_Vars()
{
    preInitOK = FALSE; // IMM-Parameter-Bearbeitung noch nicht erfolgt
    filename = NULL;
    data = NULL; // noch keine Daten
    zone = NULL; // noch keine Zones gelesen
    nused = 0; // Anzahl der benutzten Daten
    isused = 0; // Zuordnungstabelle
    nZones = 0; // noch keine Zonenanzahl bekannt
    nVars = 0; // noch keine Variablen bekannt
    VarNames[0] = new char[10]; // die gibt's immer
    strcpy(VarNames[0], "(none)\n"); //
}

void Application::paramChange(void *)
{
    char msg[200];

    const char *paramname = Covise::get_reply_param_name();

    if (strcmp("from_ts", paramname) == 0)
    {
        Covise::get_reply_int_scalar(&from_ts);
    }
    else if (strcmp("fullpath", paramname) == 0)
    {

        // clear data
        delete_data();

        // read data file name parameter
        if (!getFilename())
            return;

        // open the file
        if (!openFile())
            return;

        // read the file header
        if (!readFileHeader(fp, Title, &nVars, VarNames))
        {
            fclose(fp);
            return;
        }
        fclose(fp);
        sprintf(msg, "TITLE: \"%s\"", Title);
        Covise::sendInfo(msg);
        sprintf(msg, "%i VARIABLES", nVars);
        Covise::sendInfo(msg);
        /*
      for(int i = 0; i < nVars;i++) {
         sprintf(msg, "[%d] %s",i,VarNames[i]);
         Covise::sendInfo (msg);
      }
      */

        // copy variable names to choice param
        Covise::update_choice_param("grid_x", nVars + 1, VarNames, (nVars > 0) ? 2 : 1);
        Covise::update_choice_param("grid_y", nVars + 1, VarNames, (nVars > 1) ? 3 : 1);
        Covise::update_choice_param("grid_z", nVars + 1, VarNames, (nVars > 2) ? 4 : 1);
        Covise::update_choice_param("data1", nVars + 1, VarNames, (nVars > 3) ? 5 : 1);
        Covise::update_choice_param("data2", nVars + 1, VarNames, (nVars > 4) ? 6 : 1);
        Covise::update_choice_param("data3", nVars + 1, VarNames, (nVars > 5) ? 7 : 1);
        Covise::update_choice_param("data4", nVars + 1, VarNames, (nVars > 6) ? 8 : 1);
        Covise::update_choice_param("data5", nVars + 1, VarNames, (nVars > 7) ? 9 : 1);

        preInitOK = TRUE;
    }
    return;
}

void Application::execute(void *)
{
    char msg[200];
    // File Header read without error ?
    if (!preInitOK)
    {
        Covise::sendError("Can't execute because of earlier errors or no file selected");
        return;
    }

    // get names of the output objects
    if (!getOutputObjectNames())
        return;

    // get input parameters
    if (!getInputParams())
        return;

    if (time_dependent)
    {
        for (current_ts = from_ts; current_ts <= to_ts; current_ts++)
        {
            // read data file name parameter
            if (!getTSFilename(current_ts))
                return;

            sprintf(msg, "Reading %s", filename);
            Covise::sendInfo(msg);

            // open the file
            if (!openFile())
                return;

            // get the data, close the file
            if (!readFile())
            {
                fclose(fp);
                return;
            }
            fclose(fp);

            // create ouput data objects
            createOutputObjects();
        }
        delete GRID_TS_Set;
    }
    else
    {

        // open the file
        if (!openFile())
            return;

        sprintf(msg, "Reading %s", filename);
        Covise::sendInfo(msg);

        // get the data, close the file
        if (!readFile())
        {
            fclose(fp);
            return;
        }
        fclose(fp);

        // create ouput data objects
        createOutputObjects();
    }
}

void Application::quit(void *)
{
    delete_data();
}

void Application::delete_chain(DynZoneElement *element)
{
    while (element->next != NULL)
    {
        element = element->next;
        delete element->prev;
        if (element->value != NULL)
            delete[] element -> value;
    }
    delete element;
}

void Application::delete_chain(DynZoneDescr *element)
{
    while (element->next != NULL)
    {
        element = element->next;
        delete element->prev;
    }
    delete element;
}

void Application::delete_data()
{
    int i;

    if (data != NULL)
    {
        for (i = 0; i < nused; i++)
            delete_chain(data[i].next);
        delete[] data;
        data = NULL;
    }
    if (zone != NULL)
    {
        delete_chain(zone);
        zone = NULL;
    }
    if (isused != NULL)
    {
        delete[] isused;
        isused = NULL;
    }
    for (i = 0; i <= nVars; i++)
        delete[] VarNames[i];
    init_Vars();
}

int Application::getFilename()
{
    static const char *tmp;

    Covise::get_reply_browser(&tmp);

    if (tmp == NULL)
    {
        Covise::sendError("ERROR: filename is NULL");
        return (FALSE);
    }

    if (strchr(tmp, '%'))
    {
        filename = (char *)new char[strlen(tmp) + 1];
        sprintf(filename, tmp, from_ts);
        base_filename = new char[strlen(tmp) + 1];
        strcpy(base_filename, tmp);
        //	cerr << "base_filename:   " << base_filename << endl;
        //	cerr << "%-filename:      " << filename << endl;
        return (TRUE);
    }
    else
    {
        if (time_dependent)
        {
            Covise::sendError("ERROR: from_ts and to_ts differ, but filename does not");
            Covise::sendError("ERROR: contain wildcard for timestep number.");
            Covise::sendError("ERROR: Please add %04d or similar to filename!!");
            return (FALSE);
        }
        filename = (char *)new char[strlen(tmp) + 1];
        strcpy(filename, tmp);
        //	cerr << "std-filename:    " << filename << endl;
        return (TRUE);
    }
}

int Application::getTSFilename(int ts)
// ts == -1 : create first timestep filename
// ts >= 0  : compose filename for timestep ts
// ts < -1  : error
{
    if (ts < -1)
        return (FALSE);

    if (ts == -1)
        ts = from_ts;

    // 10 should be sufficient
    filename = (char *)new char[strlen(base_filename) + 10];
    sprintf(filename, base_filename, ts);
    //    cerr << "ts-filename:     " << filename << endl;
    return (TRUE);
}

int Application::openFile()
{
    char line[1000]; // error message

    if ((fp = Covise::fopen(filename, "r")) == NULL)
    {
        strcpy(line, "ERROR: Can't open file >> '");
        strcat(line, filename);
        strcat(line, "'");
        Covise::sendError(line);
        return (FALSE);
    }
    else
    {
        return (TRUE);
    }
}

int Application::fnextstring(long *fpos, char *str)
{
    char LFtest[200]; // Hilfs-Strings zum Einlesen

    do
    {
        strcpy(LFtest, "");
        fscanf(fp, "%*[\x1-\x9\xb- =,]");
        fscanf(fp, "%[\n]", LFtest);
        if (feof(fp))
            return (FALSE);
        line += strlen(LFtest);
    } while (strlen(LFtest) != 0);
    (*fpos) = ftell(fp);
    if (fscanf(fp, "%[^\x1- =,]", str) == 1)
        return (TRUE);
    else
        return (FALSE);
}

char *Application::upStr(char *strParam)
{
    int i;

    for (i = 0; i < strlen(strParam); i++)
        if (('a' <= strParam[i]) && (strParam[i] <= 'z'))
            strParam[i] -= 'a' - 'A';

    return strParam;
}

void Application::fileformaterror(char *msg)
{
    char hstr[1000];

    sprintf(hstr, "error in '%s' in line %i: %s.", filename, line, msg);
    Covise::sendError(hstr);
}

int Application::isVarName(char *hstr)
{
    float f;

    if ((strcmp(upStr(hstr), "TITLE") != 0)
        && (strcmp(upStr(hstr), "ZONE") != 0)
        && (strcmp(upStr(hstr), "TEXT") != 0)
        && (strcmp(upStr(hstr), "GEOMETRY") != 0)
        && (strcmp(upStr(hstr), "CUSTOMLABELS") != 0)
        && (sscanf(hstr, "%f", &f) != 1))
        return TRUE;
    else
        return FALSE;
}

int Application::readFileHeader(FILE *fp, char *Title, int *n, char **VarNames)
{
    char hstr[200], LFtest[200]; // Hilfs-String zum Einlesen
    long dummy;
    float f;

    (*n) = 0;
    strcpy(Title, "");

    rewind(fp);
    fpos = ftell(fp);
    line = 1;

    if (!fnextstring(&fpos, hstr))
    {
        fileformaterror("unexpected EOF");
        return (FALSE);
    }

    if (strcmp(upStr(hstr), "TITLE") == 0)
    {
        fscanf(fp, "%*[^\"]%*[\"]%[^\"]%*[^\n]", Title);
        if (!fnextstring(&fpos, hstr))
        {
            fileformaterror("unexpected EOF");
            return (FALSE);
        }
    }

    if (strcmp(upStr(hstr), "VARIABLES") == 0)
    {
        if (!fnextstring(&fpos, hstr))
        {
            fileformaterror("unexpected EOF");
            return (FALSE);
        }
        for ((*n) = 0; isVarName(hstr); (*n)++)
        {
            if (hstr[0] == '"')
            {
                fseek(fp, fpos, SEEK_SET);
                fscanf(fp, "%*[\"]%[^\"\n]", hstr);
                fscanf(fp, "%[\n]", LFtest);
                if (strlen(LFtest) != 0)
                {
                    fileformaterror("unexpected EOL (>\"< expected)");
                    return (FALSE);
                }
                fscanf(fp, "%*[\"]");
            }
            VarNames[(*n) + 1] = new char[strlen(hstr) + 2];
            sprintf(VarNames[(*n) + 1], "%s\n", hstr);
            if (!fnextstring(&fpos, hstr))
            {
                fileformaterror("unexpected EOF");
                return (FALSE);
            }
        }
        fseek(fp, fpos, SEEK_SET);
        if ((*n) == 0)
        {
            fileformaterror("incomplete variable definition");
            return (FALSE);
        }
    }
    else
    {
        fseek(fp, fpos, SEEK_SET);
        if (!fnextfloat(&fpos, &f))
        {
            fseek(fp, fpos, SEEK_SET);
            do
            {
                if (!fnextstring(&dummy, hstr))
                {
                    fileformaterror("unexpected EOF ('ZONE' expected)");
                    return (FALSE);
                }
            } while (strcmp(upStr(hstr), "ZONE") != 0);
            fscanf(fp, "%*[^\n]%*[\n]");
            line++;
        }
        else
            fseek(fp, fpos, SEEK_SET);

        fscanf(fp, "%*[\x1-\x9\xb- ]");
        for ((*n) = 0; fscanf(fp, "%[\n]", LFtest) == 0; (*n)++)
        {
            fscanf(fp, "%s%*[\x1-\x9\xb- ]", hstr);
            if (sscanf(hstr, "%f", &f) != 1)
            {
                fileformaterror("numeric value expected");
                return (FALSE);
            }
            VarNames[(*n) + 1] = new char[10];
            sprintf(VarNames[(*n) + 1], "v%i\n", (*n) + 1);
        }

        if ((*n) == 0)
        {
            fileformaterror("number of variables == 0");
            return (FALSE);
        }
    }
    return (TRUE);
}

int Application::getOutputObjectNames()
{
    int i;
    char name[100];

    grid_name = Covise::get_object_name("grid");
    if (grid_name == NULL)
    {
        Covise::sendError("ERROR: object name not correct");
        return (FALSE);
    }

    for (i = 0; i < 3; i++)
    {
        sprintf(name, "dataout%i", i + 1);
        data_name[i] = Covise::get_object_name(name);
        if (data_name[i] == NULL)
        {
            Covise::sendError("ERROR: object name not correct");
            return (FALSE);
        }
    }
    return (TRUE);
}

int Application::getInputParams()
{
    // get parameter value
    Covise::get_choice_param("grid_x", &usedVars[0]);
    Covise::get_choice_param("grid_y", &usedVars[1]);
    Covise::get_choice_param("grid_z", &usedVars[2]);
    Covise::get_choice_param("data1", &usedVars[3]);
    Covise::get_choice_param("data2", &usedVars[4]);
    Covise::get_choice_param("data3", &usedVars[5]);
    Covise::get_choice_param("data4", &usedVars[6]);
    Covise::get_choice_param("data5", &usedVars[7]);
    Covise::get_scalar_param("from_ts", &from_ts);
    Covise::get_scalar_param("to_ts", &to_ts);

    time_dependent = to_ts - from_ts;

    int i;
    for (i = 0; i < 8; i++)
        usedVars[i]--;

    if ((usedVars[0] && usedVars[1] && usedVars[2]) == 0)
    {
        Covise::sendError("Can't handle 1D or 2D grid.");
        return (FALSE);
    }

    return (TRUE);
}

int Application::readFile()
{
    char hstr[200]; //, msg[200];
    int cnt;
    DynZoneElement **curZone;
    DynZoneDescr *curDescr;

    rewind(fp);
    fpos = ftell(fp);
    line = 1;

    isused = new int[nVars];
    for (cnt = 0; cnt < nVars; cnt++)
        isused[cnt] = -1;
    for (nused = 0, cnt = 0; cnt < 8; cnt++)
        if (usedVars[cnt] != 0)
        {
            isused[usedVars[cnt] - 1] = nused;
            nused++;
        }

    data = new DynZoneElement[nused];
    for (cnt = 0; cnt < nused; cnt++)
    {
        data[cnt].next = NULL;
        data[cnt].prev = NULL;
        data[cnt].value = NULL;
    }
    curZone = new DynZoneElement *[nused];
    for (cnt = 0; cnt < nused; cnt++)
        curZone[cnt] = &data[cnt];
    zone = new DynZoneDescr;
    zone->next = NULL;
    zone->prev = NULL;
    curDescr = zone;

    while (fnextstring(&fpos, hstr))
    {
        if (strcmp(upStr(hstr), "TITLE") == 0)
        {
            fscanf(fp, "%*[^\n]");
        }
        else if (strcmp(upStr(hstr), "VARIABLES") == 0)
        {
            fscanf(fp, "%*[^\n]");
        }
        else if (strcmp(upStr(hstr), "ZONE") == 0)
        {
            if (!readZoneRecord(curZone, &curDescr))
            {
                delete[] curZone;
                return (FALSE);
            }
        }
        else if (strcmp(upStr(hstr), "TEXT") == 0)
        {
            // readTextRecord ();
        }
        else if (strcmp(upStr(hstr), "GEOMETRY") == 0)
        {
            // readGeometryRecord ();
        }
        else if (strcmp(upStr(hstr), "CUSTOMLABELS") == 0)
        {
            // readCustomLabelsRecord ();
        }
        else
        {
            // Fehler, falls ALLE Records von Routinen bedient werden
            // sprintf (msg, "unknown record type >>%s<<", hstr);
            // fileformaterror (msg);
            // return (FALSE);
        }
    }
    delete[] curZone; // geloescht werden nur Verweise, NICHT die Daten

    return (TRUE);
}

int Application::readZoneHeader(DynZoneDescr **curDescrP)
{
    char msg[200], hstr[200];
    float f;
    DynZoneDescr *curDescr;

    curDescr = *curDescrP;
    if (curDescr->prev == NULL)
        nZones = 0;
    // curDescr Element hinzufuegen
    curDescr->next = new DynZoneDescr;
    curDescr->next->prev = curDescr;
    curDescr = curDescr->next;
    *curDescrP = curDescr;
    curDescr->next = NULL;

    // curDescr initialisieren
    strcpy(curDescr->Title, "");
    strcpy(curDescr->Color, "");
    curDescr->Format = POINT;
    curDescr->ndup = 0;
    strcpy(curDescr->DataTypeList, "");
    curDescr->i = 0;
    curDescr->j = 1;
    curDescr->k = 1;
    curDescr->n = 0;
    curDescr->e = 0;
    curDescr->et = 0;
    curDescr->nv = 0;

    while (!fnextfloat(&fpos, &f))
    {
        if (feof(fp))
        {
            fileformaterror("unexpected EOF");
            return (FALSE);
        }
        fseek(fp, fpos, SEEK_SET);
        fscanf(fp, "%[^\x1-\x9\xb- =]%*[ =]", hstr);
        if (strcmp(upStr(hstr), "T") == 0) // Zonentitel
        {
            fscanf(fp, "%[^\"]", hstr);
            fscanf(fp, "\"%[^\"]\"", curDescr->Title);
        }
        else if (strcmp(upStr(hstr), "C") == 0) // Zonenfarbe
            fscanf(fp, "%[^\x1- ,]", curDescr->Color);
        else if (strcmp(upStr(hstr), "F") == 0) // Format
        {
            fscanf(fp, "%[^\x1- ,]", hstr);
            if (strcmp(upStr(hstr), "POINT") == 0)
                curDescr->Format = POINT;
            else if (strcmp(upStr(hstr), "BLOCK") == 0)
                curDescr->Format = BLOCK;
            else if (strcmp(upStr(hstr), "FEPOINT") == 0)
                curDescr->Format = FEPOINT;
            else if (strcmp(upStr(hstr), "FEBLOCK") == 0)
                curDescr->Format = FEBLOCK;
            else
            {
                sprintf(msg, "unknown zone format >>%s<<", hstr);
                fileformaterror(msg);
                return (FALSE);
            }
        }
        else if (strcmp(upStr(hstr), "D") == 0) // duplist
        {
            fscanf(fp, "%*[^(]");
            fscanf(fp, "(");
            for (curDescr->ndup = 0; fscanf(fp, "%[)]", hstr) == 0; curDescr->ndup++)
                if (fscanf(fp, "%i%*[ ,]", &(curDescr->duplist[curDescr->ndup])) != 1)
                {
                    fileformaterror("numeric value expected");
                    return (FALSE);
                }
            if ((nZones == 0) && (curDescr->ndup != 0))
            {
                fileformaterror("can't dup values in first zone record");
                return (FALSE);
            }
        }
        else if (strcmp(upStr(hstr), "DT") == 0) // datatypelist
        {
            fscanf(fp, "%[^(]", hstr);
            fscanf(fp, "(%[^)])", curDescr->DataTypeList);
        }
        else if (strcmp(upStr(hstr), "I") == 0) // fuer i,j,k-Daten
            fscanf(fp, "%i", &curDescr->i);
        else if (strcmp(upStr(hstr), "J") == 0)
            fscanf(fp, "%i", &curDescr->j);
        else if (strcmp(upStr(hstr), "K") == 0)
            fscanf(fp, "%i", &curDescr->k);
        else
            // numnodes, ab hier fuer FE-Zones
            if (strcmp(upStr(hstr), "N") == 0)
        {
            fscanf(fp, "%[^\x1- ,]", hstr);
            int i = 0;
            while (hstr[i] == '0' && i < strlen(hstr))
                i++;
            sscanf(&hstr[i], "%i", &curDescr->n);
            sprintf(msg, "%d Nodes", curDescr->n);
            Covise::sendInfo(msg);
        }
        else if (strcmp(upStr(hstr), "E") == 0) // numelement
        {
            fscanf(fp, "%[^\x1- ,]", hstr);
            int i = 0;
            while (hstr[i] == '0' && i < strlen(hstr))
                i++;
            sscanf(&hstr[i], "%i", &curDescr->e);
            sprintf(msg, "%d Elements", curDescr->e);
            Covise::sendInfo(msg);
        }
        else if (strcmp(upStr(hstr), "ET") == 0) // elementtype -> not yet
        {
            //        fscanf (fp, "%*[^\x1-\x9\xb- ,]");
            fscanf(fp, "%[^\x1- ,]", hstr);
            if (strcmp(upStr(hstr), "BRICK") == 0)
                curDescr->et = BRICK;
            else if (strcmp(upStr(hstr), "QUADRILATERAL") == 0)
                curDescr->et = QUADRILATERAL;
            else
            {
                sprintf(msg, "unknown zone format >>%s<<", hstr);
                fileformaterror(msg);
                return (FALSE);
            }
        }
        else if (strcmp(upStr(hstr), "NV") == 0) // nodevariable -> not yet
            fscanf(fp, "%*[^\"]\"%*[^\"]\"");
    }
    fseek(fp, fpos, SEEK_SET);
    if (curDescr->i == 0 && curDescr->n == 0)
    {
        fileformaterror("no grid dimension information");
        return (FALSE);
    }
    return (TRUE);
}

int Application::fnextfloat(long *fpos, float *f)
{
    char LFtest[200], hstr[200]; // Hilfs-Strings zum Einlesen

    do
    {
        strcpy(LFtest, "");
        fscanf(fp, "%*[\x1-\x9\xb- ,]");
        fscanf(fp, "%[\n]", LFtest);
        if (feof(fp))
            return (FALSE);
        line += strlen(LFtest);
    } while (strlen(LFtest) != 0);
    (*fpos) = ftell(fp);
    if (fscanf(fp, "%[^\x1- ,]", hstr) != 1)
        return (FALSE);
    if (sscanf(hstr, "%f", f) != 1)
        return (FALSE);
    return (TRUE);
}

int Application::fnextint(long *fpos, int *i)
{
    char LFtest[200], hstr[200]; // Hilfs-Strings zum Einlesen

    do
    {
        strcpy(LFtest, "");
        fscanf(fp, "%*[\x1-\x9\xb- ,]");
        fscanf(fp, "%[\n]", LFtest);
        if (feof(fp))
            return (FALSE);
        line += strlen(LFtest);
    } while (strlen(LFtest) != 0);
    (*fpos) = ftell(fp);
    if (fscanf(fp, "%[^\x1- ,]", hstr) != 1)
        return (FALSE);
    if (sscanf(hstr, "%d", i) != 1)
        return (FALSE);
    return (TRUE);
}

int Application::readStructVal(DynZoneElement **curZone, int Var, int Val)
{
    float dummy;

    if (!fnextfloat(&fpos, &dummy))
    {
        fileformaterror("numeric value expected");
        return (FALSE);
    }
    if (isused[Var] != -1)
        curZone[isused[Var]]->value[Val] = dummy;

    return (TRUE);
}

int Application::readElement(int *Var, int no_of_elements)
{
    int dummy, i;

    for (i = 0; i < no_of_elements; i++)
    {
        if (!fnextint(&fpos, &dummy))
        {
            fileformaterror("numeric value expected");
            return (FALSE);
        }
        Var[i] = dummy - 1;
    }
    return (TRUE);
}

int Application::readZoneRecord(DynZoneElement **curZone,
                                DynZoneDescr **curDescrP)
{
    int v, cnt, isdubbed[MAX_VARS], curdubbed[8];
    DynZoneDescr *curDescr;

    if (!readZoneHeader(curDescrP))
        return (FALSE);
    curDescr = *curDescrP;

    // TRUE: soll dubliziert werden, sonst FALSE
    for (cnt = 0; cnt < nused; cnt++)
        curdubbed[cnt] = FALSE;
    for (v = 0; v < nVars; v++)
        for (isdubbed[v] = FALSE, cnt = 0; cnt < curDescr->ndup; cnt++)
            if (curDescr->duplist[cnt] == v + 1)
            {
                isdubbed[v] = TRUE;
                if (isused[v] != -1)
                    curdubbed[isused[v]] = TRUE;
            }

    // Neue Zone fuer alle Var
    for (cnt = 0; cnt < nused; cnt++)
    {
        curZone[cnt]->next = new DynZoneElement;
        curZone[cnt]->next->prev = curZone[cnt];
        curZone[cnt] = curZone[cnt]->next;
        curZone[cnt]->next = NULL;
        if (curdubbed[cnt])
            curZone[cnt]->value = curZone[cnt]->prev->value;
        else
        {
            if (curDescr->Format == FEBLOCK || curDescr->Format == FEPOINT)
                curZone[cnt]->value = new float[(curDescr->n)];
            else
                curZone[cnt]->value = new float[(curDescr->i) * (curDescr->j) * (curDescr->k)];
            curZone[cnt]->connectivity = NULL;
        }
        if (curZone[cnt]->value == NULL)
        {
            Covise::sendError("memory error.");
            return FALSE;
        }
    }

    // Daten einlesen
    if (curDescr->Format == POINT)
    {
        for (cnt = 0; cnt < (curDescr->i) * (curDescr->j) * (curDescr->k); cnt++)
            for (v = 0; v < nVars; v++)
                if (!isdubbed[v])
                    if (!readStructVal(curZone, v, cnt))
                        return (FALSE);
    }
    else if (curDescr->Format == BLOCK)
    {
        for (v = 0; v < nVars; v++)
            if (!isdubbed[v])
                for (cnt = 0; cnt < (curDescr->i) * (curDescr->j) * (curDescr->k); cnt++)
                    if (!readStructVal(curZone, v, cnt))
                        return (FALSE);
    }
    else if (curDescr->Format == FEPOINT)
    {
        for (cnt = 0; cnt < (curDescr->n); cnt++)
            for (v = 0; v < nVars; v++)
                if (!isdubbed[v])
                    if (!readStructVal(curZone, v, cnt))
                        return (FALSE);
        if (curDescr->et == BRICK)
        {
            curZone[0]->connectivity = new int[8 * curDescr->e];
            for (v = 0; v < curDescr->e; v++)
                readElement(&(curZone[0]->connectivity[v * 8]), 8);
        }
        else if (curDescr->et == QUADRILATERAL)
        {
            curZone[0]->connectivity = new int[4 * curDescr->e];
            for (v = 0; v < curDescr->e; v++)
                readElement(&(curZone[0]->connectivity[v * 4]), 4);
        }
    }
    else if (curDescr->Format == FEBLOCK)
    {
        for (v = 0; v < nVars; v++)
            if (!isdubbed[v])
                for (cnt = 0; cnt < (curDescr->n); cnt++)
                    if (!readStructVal(curZone, v, cnt))
                        return (FALSE);
        if (curDescr->et == BRICK)
        {
            curZone[0]->connectivity = new int[8 * curDescr->e];
            for (v = 0; v < curDescr->e; v++)
                readElement(&(curZone[0]->connectivity[v * 8]), 8);
        }
        else if (curDescr->et == QUADRILATERAL)
        {
            curZone[0]->connectivity = new int[4 * curDescr->e];
            for (v = 0; v < curDescr->e; v++)
                readElement(&(curZone[0]->connectivity[v * 4]), 4);
        }
    }
    nZones++;

    return (TRUE);
}

void Application::createOutputObjects()
{
    int z, v, i, cnt;
    char hstr[200];
    char *tmp_grid_name;
    char attr[200];
    DynZoneElement **curZone;
    DynZoneDescr *curDescr;

    curZone = new DynZoneElement *[nVars];

    nZones = 1; // only for PowerFlow ASCII files in TECPLOT format

    //  fprintf(stderr, "usedVars[]: %d %d %d %d %d %d\n", usedVars[0], usedVars[1], usedVars[2], usedVars[3], usedVars[4], usedVars[5]);

    for (cnt = 0; cnt < 8; cnt++)
    {
        if (usedVars[cnt] != 0)
        {
            curZone[cnt] = &data[isused[usedVars[cnt] - 1]];
        }
        else
            curZone[cnt] = NULL;
    }

    //    if(time_dependent) {
    //   	 tmp_grid_name = new char[strlen(grid_name) + 1 + 3];
    // 	 sprintf(tmp_grid_name, "%s_ts", grid_name);
    //       	 if(current_ts == from_ts) {
    // 	     GRID_TS_Set = new coDoSet(grid_name, SET_CREATE);
    // 	     sprintf(attr, "1 %d", to_ts - from_ts + 1);
    // 	     GRID_TS_Set->addAttribute( "TIMESTEP", attr );
    // 	 }
    //    } else {
    //   	 tmp_grid_name = new char[strlen(grid_name) + 1];
    // 	 strcpy(tmp_grid_name, grid_name);
    //    }

    curDescr = zone;
    if (!time_dependent || current_ts == from_ts)
    {
        GRID_Set = new coDoSet(grid_name, SET_CREATE);
    }
    for (z = 0; z < nZones; z++)
    {
        curZone[0] = curZone[0]->next;
        curZone[1] = curZone[1]->next;
        curZone[2] = curZone[2]->next;
        curDescr = curDescr->next;
        // structured grid
        if (curDescr->k * curDescr->j * curDescr->i > 0)
        {
            if (curDescr->k == 1)
            {
                curDescr->k = curDescr->j;
                curDescr->j = curDescr->i;
                curDescr->i = 1;
            }
            if (curDescr->k == 1)
            {
                curDescr->k = curDescr->i;
                curDescr->j = 1;
            }

            sprintf(hstr, "%s_grid%i", grid_name, z);
            if (!time_dependent || current_ts == from_ts)
            {
                GRID = new coDoStructuredGrid(hstr, curDescr->k,
                                              curDescr->j,
                                              curDescr->i,
                                              curZone[0]->value,
                                              curZone[1]->value,
                                              curZone[2]->value);
                GRID_Set->addElement(GRID);
                if (time_dependent)
                {
                    for (i = from_ts; i < to_ts; i++)
                        GRID_Set->addElement(GRID);
                    sprintf(attr, "1 %d", to_ts - from_ts + 1);
                    GRID_Set->addAttribute("TIMESTEP", attr);
                }
                delete GRID;
            }
        } // unstructured grid
        else
        {

            sprintf(hstr, "%s_grid%i", grid_name, z);
            int *tl = new int[curDescr->e];
            int *el = new int[curDescr->e];
            switch (curDescr->et)
            {
            case BRICK:
                for (i = 0; i < curDescr->e; i++)
                {
                    el[i] = i * 8;
                    tl[i] = TYPE_HEXAEDER;
                }
                if (!time_dependent || current_ts == from_ts)
                {
                    uGRID = new coDoUnstructuredGrid(hstr, curDescr->e, curDescr->e * 8, curDescr->n,
                                                     el, curZone[0]->connectivity,
                                                     curZone[0]->value,
                                                     curZone[1]->value,
                                                     curZone[2]->value, tl);
                }
                break;
            case QUADRILATERAL:
                for (i = 0; i < curDescr->e; i++)
                {
                    el[i] = i * 4;
                    tl[i] = TYPE_QUAD;
                }
                if (!time_dependent || current_ts == from_ts)
                {
                    uGRID = new coDoUnstructuredGrid(hstr, curDescr->e, curDescr->e * 4, curDescr->n,
                                                     el, curZone[0]->connectivity,
                                                     curZone[0]->value,
                                                     curZone[1]->value,
                                                     curZone[2]->value, tl);
                }
                break;
            }
            if (!time_dependent || current_ts == from_ts)
            {
                GRID_Set->addElement(uGRID);
                if (time_dependent)
                {
                    for (i = from_ts; i < to_ts; i++)
                        GRID_Set->addElement(uGRID);
                    sprintf(attr, "1 %d", to_ts - from_ts + 1);
                    GRID_Set->addAttribute("TIMESTEP", attr);
                }
                delete uGRID;
            }
        }
    }

    if (curZone[3] != NULL)
    {
        curDescr = zone;
        char *tmp_data_name;
        if (time_dependent)
        {
            tmp_data_name = new char[strlen(data_name[0]) + 20];
            sprintf(tmp_data_name, "%s_ts%d", data_name[0], current_ts);
            if (current_ts == from_ts)
            {
                DATA_TS_Set[0] = new coDoSet(data_name[0], SET_CREATE);
                sprintf(attr, "1 %d", to_ts - from_ts + 1);
                DATA_TS_Set[0]->addAttribute("TIMESTEP", attr);
            }
        }
        else
        {
            tmp_data_name = new char[strlen(data_name[0]) + 1];
            strcpy(tmp_data_name, data_name[0]);
            DATA_Set = new coDoSet(data_name[0], SET_CREATE);
        }

        //     DATA_Set->addAttribute("DATA_NAME","velocity");

        for (z = 0; z < nZones; z++)
        {
            curZone[3] = curZone[3]->next;
            curZone[4] = curZone[4]->next;
            curZone[5] = curZone[5]->next;
            curDescr = curDescr->next;
            sprintf(hstr, "%s_zone%i", tmp_data_name, z);
            if (curDescr->k * curDescr->j * curDescr->i > 0)
            {
                VDATA = new coDoVec3(hstr, curDescr->k,
                                     curDescr->j,
                                     curDescr->i,
                                     curZone[3]->value,
                                     curZone[4]->value,
                                     curZone[5]->value);
                if (time_dependent)
                    DATA_TS_Set[0]->addElement(VDATA);
                else
                    DATA_Set->addElement(VDATA);

                delete VDATA;
            }
            else
            {
                uVDATA = new coDoVec3(hstr, curDescr->n,
                                      curZone[3]->value,
                                      curZone[4]->value,
                                      curZone[5]->value);
                if (time_dependent)
                    DATA_TS_Set[0]->addElement(uVDATA);
                else
                    DATA_Set->addElement(uVDATA);
                delete uVDATA;
            }
        }
        if (!time_dependent)
            delete DATA_Set;
        if (current_ts == to_ts)
            delete DATA_TS_Set[0];
        delete[] tmp_data_name;
    }

    for (v = 0; v < 2; v++)
        if (curZone[v + 6] != NULL)
        {
            curDescr = zone;
            char *tmp_data_name;
            if (time_dependent)
            {
                tmp_data_name = new char[strlen(data_name[1 + v]) + 20];
                sprintf(tmp_data_name, "%s_ts%d", data_name[1 + v], current_ts);
                if (current_ts == from_ts)
                {
                    DATA_TS_Set[1 + v] = new coDoSet(data_name[1 + v], SET_CREATE);
                    sprintf(attr, "1 %d", to_ts - from_ts + 1);
                    DATA_TS_Set[1 + v]->addAttribute("TIMESTEP", attr);
                }
            }
            else
            {
                tmp_data_name = new char[strlen(data_name[1 + v]) + 1];
                strcpy(tmp_data_name, data_name[1 + v]);
                DATA_Set = new coDoSet(data_name[1 + v], SET_CREATE);
            }

            //     DATA_Set->addAttribute("DATA_NAME",VarNames[usedVars[v+6]]);

            for (z = 0; z < nZones; z++)
            {
                curZone[v + 6] = curZone[v + 6]->next;
                curDescr = curDescr->next;
                sprintf(hstr, "%s_zone%i", tmp_data_name, z);
                if (curDescr->k * curDescr->j * curDescr->i > 0)
                {
                    DATA = new coDoFloat(hstr, curDescr->k,
                                         curDescr->j,
                                         curDescr->i,
                                         curZone[v + 6]->value);
                    if (time_dependent)
                        DATA_TS_Set[1 + v]->addElement(DATA);
                    else
                        DATA_Set->addElement(DATA);
                    delete DATA;
                }
                else
                {
                    uDATA = new coDoFloat(hstr, curDescr->n,
                                          curZone[v + 6]->value);
                    if (time_dependent)
                        DATA_TS_Set[1 + v]->addElement(uDATA);
                    else
                        DATA_Set->addElement(uDATA);
                    delete uDATA;
                }
            }
            if (!time_dependent)
                delete DATA_Set;
            if (current_ts == to_ts)
                delete DATA_TS_Set[1 + v];
            delete[] tmp_data_name;
        }

    delete[] curZone;
}
