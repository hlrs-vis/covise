/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 1997					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30a				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			Flower.C	 	                *
 *									*
 *	Description	        FLower file reader			*
 *									*
 *	Author			Tobias Schweickhardt 			*
 *									*
 *	Date			7. 8. 1997				*
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
#include "ReadFlower.h"

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
    Covise::set_module_description("Read Flower Surf Files");

    // module parameter filename as a browser
    Covise::add_port(PARIN, "fullpath", "Browser", "filename");
    Covise::set_port_default("fullpath", "data/spock/flower/surf");
    Covise::set_port_immediate("fullpath", 1);

    //    Covise::add_port(PARIN,"datax","Choice","Select Output Data");
    //    Covise::set_port_default ("datax","1 <none> ");
    Covise::add_port(PARIN, "gridselect", "String", "Select Grid Data");
    Covise::set_port_default("gridselect", "x/y/z");
    Covise::add_port(PARIN, "dataselect", "String", "Select Output Data");
    Covise::set_port_default("dataselect", "rho");
    Covise::add_port(PARIN, "allVars", "String", "possible choices");
    Covise::set_port_default("allVars", "(none)");

    // define module output ports
    Covise::add_port(OUTPUT_PORT, "grid", "Set_StructuredGrid", "grid");

    Covise::add_port(OUTPUT_PORT, "data1", "Set_Float", "data1");
    Covise::add_port(OUTPUT_PORT, "data2", "Set_Float", "data2");
    Covise::add_port(OUTPUT_PORT, "data3", "Set_Float", "data3");

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
    VarNames[0] = new char[MAX_N_VARS]; // die gibt's immer
    strcpy(VarNames[0], "(none)"); //
}

void Application::paramChange(void *)
{
    //fprintf(stderr,"param change\n");
    int i;
    char msg[200];

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

    // copy variable names to choice param
    //  Covise::update_choice_param ("datax", nVars+1, VarNames, 1);
    strcpy(msg, "(none)");
    if (nVars >= MAX_N_VARS)
    {
        Covise::sendError("Limit of maximum number of variables exceeded in ReadFlower");
        return;
    }
    for (i = 1; i < nVars + 1; i++)
    {
        strcat(msg, " ");
        strcat(msg, VarNames[i]);
    }
    if (!Covise::update_string_param("allVars", msg))
    {
        Covise::sendError("error at call 'update_string_param ()' in function 'paramChange'.");
        return;
    }

    strcpy(msg, VarNames[1]);
    for (i = 2; i < 4; i++)
    {
        strcat(msg, "/");
        strcat(msg, VarNames[i]);
    }
    Covise::update_string_param("gridselect", msg);

    preInitOK = TRUE;
}

void Application::execute(void *)
{
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
    const char *tmp;

    Covise::get_reply_browser(&tmp);

    if (tmp == NULL)
    {
        Covise::sendError("ERROR: filename is NULL");
        return (FALSE);
    }
    else
    {
        filename = (char *)new char[strlen(tmp) + 1];
        strcpy(filename, tmp);
        return (TRUE);
    }
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
        //fprintf(stderr,"filename is: %s\n", filename);
        return (TRUE);
    }
}

int Application::fnextstring(long *fpos, char *str)
{
    char LFtest[200]; // Hilfs-Strings zum Einlesen

    do
    {
        strcpy(LFtest, "");
        fscanf(fp, "%*[\x1-\x9\xb- ]");
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
        strcpy(LFtest, "");
        fscanf(fp, "%*[^\"\n]");
        fscanf(fp, "%[\n]", LFtest);
        for ((*n) = 0; strlen(LFtest) == 0; (*n)++)
        {
            fscanf(fp, "%*[\"]%[^\"\n]%[\n]", hstr, LFtest);
            if (strlen(LFtest) != 0)
            {
                fileformaterror("unexpected EOL (>\"< expected)");
                return (FALSE);
            }
            VarNames[(*n) + 1] = new char[strlen(hstr) + 1];
            strcpy(VarNames[(*n) + 1], hstr);
            fscanf(fp, "%*[\"]%*[^\"\n]");
            fscanf(fp, "%[\n]", LFtest);
        }
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
            sprintf(VarNames[(*n) + 1], "v%i", (*n) + 1);
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
        sprintf(name, "data%i", i + 1);
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
    int i, v;
    char *param, hstr[300], *h[6];

    for (i = 0; i < 6; i++)
    {
        h[i] = &hstr[i * 50];
        strcpy(h[i], "");
    }

    if (!Covise::get_string_param("gridselect", &param))
    {
        Covise::sendError("unable to get input parameter 'gridselect'.");
        return (FALSE);
    }
    sscanf(param, "%[^/ ]/%[^/ ]/%[^/ ]", h[0], h[1], h[2]);
    if (!Covise::get_string_param("dataselect", &param))
    {
        Covise::sendError("unable to get input parameter 'dataselect'.");
        return (FALSE);
    }
    sscanf(param, "%[^/ ]/%[^/ ]/%[^/ ]", h[3], h[4], h[5]);

    for (i = 0; i < 6; i++)
    {
        usedVars[i] = 0;
        if (usedVars[i] == 0 && nVars >= MAX_N_VARS)
        {
            Covise::sendError("Limit of maximum number of variables exceeded in ReadFlower");
            return FALSE;
        }
        for (v = 1; (v <= nVars) && (usedVars[i] == 0); v++)
            if (strcmp(VarNames[v], h[i]) == 0)
                usedVars[i] = v;
    }

    if (usedVars[2] == 0)
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
    for (nused = 0, cnt = 0; cnt < 6; cnt++)
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
    char msg[200], hstr[200], LFtest[200];
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

    strcpy(LFtest, "");
    fscanf(fp, "%*[\x1-\x9\xb- ,]");
    fscanf(fp, "%[\n]", LFtest);
    while (strlen(LFtest) == 0)
    {
        fscanf(fp, "%[^\x1-\x9\xb- =]%*[ =]", hstr);
        if (strcmp(upStr(hstr), "T") == 0) // Zonentitel
            fscanf(fp, "%[^\"]\"%*[^\"]\"", curDescr->Title);
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
            fscanf(fp, "%[^(](%*[^)])", curDescr->DataTypeList);
        else if (strcmp(upStr(hstr), "I") == 0) // fuer i,j,k-Daten
            fscanf(fp, "%i", &curDescr->i);
        else if (strcmp(upStr(hstr), "J") == 0)
            fscanf(fp, "%i", &curDescr->j);
        else if (strcmp(upStr(hstr), "K") == 0)
            fscanf(fp, "%i", &curDescr->k);
        else
            // numnodes, ab hier fuer FE-Zones
            if (strcmp(upStr(hstr), "N") == 0)
            fscanf(fp, "%i", &curDescr->n);
        else
            // numelement
            if (strcmp(upStr(hstr), "E") == 0)
            fscanf(fp, "%i", &curDescr->e);
        else
            // elementtype -> not yet
            if (strcmp(upStr(hstr), "ET") == 0)
            fscanf(fp, "%*[^\x1-\x9\xb- ,]");
        else
            // nodevariable -> not yet
            if (strcmp(upStr(hstr), "NV") == 0)
            fscanf(fp, "%*[^\"]\"%*[^\"]\"");

        fpos = ftell(fp);
        strcpy(LFtest, "");
        fscanf(fp, "%*[\x1-\x9\xb- ,]");
        fscanf(fp, "%[\n]", LFtest);
        if (feof(fp))
        {
            fileformaterror("unexpected EOF");
            return (FALSE);
        }
    }
    line += strlen(LFtest);
    if (curDescr->i == 0)
    {
        fileformaterror("i==0");
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

int Application::readZoneRecord(DynZoneElement **curZone,
                                DynZoneDescr **curDescrP)
{
    int v, cnt, isdubbed[MAX_VARS], curdubbed[6];
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
            curZone[cnt]->value = new float[(curDescr->i) * (curDescr->j) * (curDescr->k)];
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
        fileformaterror("can't handle unstructured data yet");
        return (FALSE);
    }
    else if (curDescr->Format == FEBLOCK)
    {
        fileformaterror("can't handle unstructured data yet");
        return (FALSE);
    }
    nZones++;

    return (TRUE);
}

void Application::createOutputObjects()
{
    int z, v, cnt;
    char hstr[100];
    DynZoneElement **curZone;
    DynZoneDescr *curDescr;

    curZone = new DynZoneElement *[nVars];

    //fprintf(stderr, "usedVars[]: %d %d %d %d %d %d\n", usedVars[0], usedVars[1], usedVars[2], usedVars[3], usedVars[4], usedVars[5]);

    for (cnt = 0; cnt < 6; cnt++)
    {
        if (usedVars[cnt] != 0)
        {
            curZone[cnt] = &data[isused[usedVars[cnt] - 1]];
        }
        else
            curZone[cnt] = NULL;
    }

    curDescr = zone;
    GRID_Set = new coDoSet(grid_name, SET_CREATE);
    for (z = 0; z < nZones; z++)
    {
        curZone[0] = curZone[0]->next;
        curZone[1] = curZone[1]->next;
        curZone[2] = curZone[2]->next;
        curDescr = curDescr->next;
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
        GRID = new coDoStructuredGrid(hstr, curDescr->k,
                                      curDescr->j,
                                      curDescr->i,
                                      curZone[0]->value,
                                      curZone[1]->value,
                                      curZone[2]->value);
        GRID_Set->addElement(GRID);
        delete GRID;
    }
    delete GRID_Set;

    for (v = 0; v < 3; v++)
        if (curZone[v + 3] != NULL)
        {
            curDescr = zone;
            DATA_Set = new coDoSet(data_name[v], SET_CREATE);
            //fprintf(stderr,"Data Set Name: %s", VarNames[usedVars[v+3]]);
            DATA_Set->addAttribute("DATA_NAME", VarNames[usedVars[v + 3]]);

            for (z = 0; z < nZones; z++)
            {
                curZone[v + 3] = curZone[v + 3]->next;
                curDescr = curDescr->next;
                sprintf(hstr, "%s_zone%i", data_name[v], z);
                DATA = new coDoFloat(hstr, curDescr->k,
                                     curDescr->j,
                                     curDescr->i,
                                     curZone[v + 3]->value);

                DATA_Set->addElement(DATA);
                delete DATA;
            }
            delete DATA_Set;
        }

    delete[] curZone;
}
