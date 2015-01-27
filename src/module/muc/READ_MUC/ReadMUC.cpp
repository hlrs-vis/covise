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
 *      Modified by Ralph Bruckschen to read in timedependent data	*
 *	with every timestep from different files			*
 *	Date 6. 10. 99							*
 *									*
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <appl/ApplInterface.h>
#include "ReadMUC.h"

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
    Covise::set_module_description("Tecplot file reader V 1.1b");

    // module parameter filename as a browser
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
    Covise::add_port(PARIN, "timedep", "Choice", "Timedependent Data");
    Covise::set_port_default("timedep", "1 Timedependent Stationary");

    // define module output ports
    Covise::add_port(OUTPUT_PORT, "grid", "Set_StructuredGrid", "grid");
    Covise::add_port(OUTPUT_PORT, "ugrid", "Set_UnstructuredGrid", "ugrid");

    Covise::add_port(OUTPUT_PORT, "dataout1", "Set_Vec3", "dataout1");
    Covise::add_port(OUTPUT_PORT, "dataout2", "Set_Float", "dataout2");
    Covise::add_port(OUTPUT_PORT, "dataout3", "Set_Float", "dataout3");
    Covise::add_port(OUTPUT_PORT, "udataout1", "Set_Vec3", "udataout1");
    Covise::add_port(OUTPUT_PORT, "udataout2", "Set_Float", "udataout2");
    Covise::add_port(OUTPUT_PORT, "udataout3", "Set_Float", "udataout3");

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
    for (int i = 0; i < 2; i++)
        DATA_Set[i] = NULL;
}

void Application::paramChange(void *)
{
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

void Application::execute(void *)
{
    // File Header read without error ?
    if (!preInitOK)
    {
        Covise::sendError("Can't execute because of earlier errors or no file selected");
        return;
    }

    Covise::get_choice_param("timedep", &is_timedependent);
    printf("timedep=%d\n", is_timedependent);

    if (is_timedependent == 1)
    {
        // get names of the output objects
        if (!getOutputObjectNames())
            return;

        // get input parameters
        if (!getInputParams())
            return;
        // extract rumpf & tail from filename
        // get first timestep
        // filename is supposed to be xxxtimestep.extension
        char rumpf[128], tail[128];
        int i = 0;
        while (filename[i] != '.')
        {
            rumpf[i] = filename[i];
            i++;
        }
        int j = i - 1;
        while (rumpf[j] >= '0' && rumpf[j] <= '9')
        {
            j--;
        }
        sscanf(filename + j + 1, "%d", &timestep);
        rumpf[j + 1] = '\0';
        j = 0;
        while (filename[i])
        {
            tail[j] = filename[i];
            i++;
            j++;
        }
        tail[j] = '\0';
        char old_filename[256];
        strcpy(old_filename, filename);
        sprintf(filename, "%s%d%s", rumpf, timestep, tail);
        FILE *dummy;
        for (i = 0; i < 2; i++)
            TIME_DATA_Set[i] = NULL;
        TIME_GRID_Set = NULL;
        TIME_VECTOR_Set = NULL;
        TIME_uGRID_Set = NULL;
        TIME_uVECTOR_Set = NULL;
        for (i = 0; i < 2; i++)
            TIME_uSCALAR_Set[i] = NULL;
        // Extract Number of files
        filesets = 0;
        int temp_timestep = timestep;
        char temp_filename[256];
        sprintf(temp_filename, "%s%d%s", rumpf, temp_timestep, tail);
        while (dummy = Covise::fopen(temp_filename, "r"))
        {
            filesets++;
            fclose(dummy);
            temp_timestep++;
            sprintf(temp_filename, "%s%d%s", rumpf, temp_timestep, tail);
        }
        //initialize sets for timedependent output

        char attr_str[32];
        sprintf(attr_str, "%d %d", timestep, temp_timestep - 1);

        GRID_sets = new coDistributedObject *[filesets + 1];
        GRID_sets[0] = NULL;

        uGRID_sets = new coDistributedObject *[filesets + 1];
        uGRID_sets[0] = NULL;

        VECTOR_sets = new coDistributedObject *[filesets + 1];
        VECTOR_sets[0] = NULL;

        uVECTOR_sets = new coDistributedObject *[filesets + 1];
        uVECTOR_sets[0] = NULL;

        for (i = 0; i < 2; i++)
        {
            DATA_sets[i] = new coDistributedObject *[filesets + 1];
            DATA_sets[i][0] = NULL;
            uSCALAR_sets[i] = new coDistributedObject *[filesets + 1];
            uSCALAR_sets[i][0] = NULL;
        }

        current_file = 0;
        //read in all data
        while (dummy = Covise::fopen(filename, "r"))
        {
            char msg[256];
            sprintf(msg, "Reading %s", filename);
            Covise::sendInfo(msg);

            fclose(dummy);

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
            create_time_OutputObjects();

            timestep++;
            current_file++;
            sprintf(filename, "%s%d%s", rumpf, timestep, tail);
        }
        // Build output data from collected sets
        TIME_GRID_Set = new coDoSet(grid_name, GRID_sets);
        TIME_GRID_Set->addAttribute("TIMESTEP", attr_str);

        TIME_uGRID_Set = new coDoSet(ugrid_name, uGRID_sets);
        TIME_uGRID_Set->addAttribute("TIMESTEP", attr_str);

        TIME_VECTOR_Set = new coDoSet(data_name[0], VECTOR_sets);
        TIME_uVECTOR_Set = new coDoSet(udata_name[0], uVECTOR_sets);

        for (i = 0; uGRID_sets[i]; i++)
            delete uGRID_sets[i];

        for (i = 0; GRID_sets[i]; i++)
            delete GRID_sets[i];

        for (i = 0; VECTOR_sets[i]; i++)
            delete VECTOR_sets[i];

        for (i = 0; i < 2; i++)
            TIME_DATA_Set[i] = new coDoSet(data_name[i + 1], DATA_sets[i]);

        for (i = 0; i < 2; i++)
            TIME_uSCALAR_Set[i] = new coDoSet(udata_name[i + 1], uSCALAR_sets[i]);

        for (i = 0; i < 2; i++)
            for (j = 0; DATA_sets[i][j]; j++)
                delete DATA_sets[i][j];

        for (i = 0; i < 2; i++)
            for (j = 0; uSCALAR_sets[i][j]; j++)
                delete uSCALAR_sets[i][j];

        //delete TIME_GRID_Set;

        //for(i=0;i<3;i++) if(TIME_DATA_Set[i]) delete TIME_DATA_Set[i];

        strcpy(filename, old_filename);
    }
    else
    { // not timedependent - do as the old code did
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

    ugrid_name = Covise::get_object_name("ugrid");
    if (ugrid_name == NULL)
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

    for (i = 0; i < 3; i++)
    {
        sprintf(name, "udataout%i", i + 1);
        udata_name[i] = Covise::get_object_name(name);
        if (udata_name[i] == NULL)
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
    }
    fseek(fp, fpos, SEEK_SET);
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
            curZone[cnt]->value = new float[(curDescr->i) * (curDescr->j) * (curDescr->k)];
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

void Application::create_time_OutputObjects()
{
    int z, v, cnt;
    char hstr[100];
    DynZoneElement **curZone;
    DynZoneDescr *curDescr;

    curZone = new DynZoneElement *[nVars];

    //fprintf(stderr, "usedVars[]: %d %d %d %d %d %d\n", usedVars[0], usedVars[1], usedVars[2], usedVars[3], usedVars[4], usedVars[5]);

    for (cnt = 0; cnt < 8; cnt++)
    {
        if (usedVars[cnt] != 0)
        {
            curZone[cnt] = &data[isused[usedVars[cnt] - 1]];
        }
        else
            curZone[cnt] = NULL;
    }

    curDescr = zone;
    char dummy[128];
    sprintf(dummy, "%s_%d", grid_name, timestep);
    GRID_Set = new coDoSet(dummy, SET_CREATE);
    sprintf(dummy, "%s_%d", ugrid_name, timestep);
    uGRID_Set = new coDoSet(dummy, SET_CREATE);
    coDoStructuredGrid **grid_array;
    grid_array = new coDoStructuredGrid *[nZones];
    coDoVec3 **vector_array;
    vector_array = new coDoVec3 *[nZones];
    coDoFloat **scalar_array;
    scalar_array = new coDoFloat *[nZones];

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
            cerr << "shifting a\n";
        }
        if (curDescr->k == 1)
        {
            curDescr->k = curDescr->i;
            curDescr->j = 1;
            cerr << "shifting b\n";
        }

        sprintf(hstr, "%s_grid%i_%d", grid_name, z, timestep);
        GRID = new coDoStructuredGrid(hstr, curDescr->k,
                                      curDescr->j,
                                      curDescr->i,
                                      curZone[0]->value,
                                      curZone[1]->value,
                                      curZone[2]->value);
        GRID_Set->addElement(GRID);
        grid_array[z] = GRID;
        //      delete GRID;
    }

    int num_coord = 0;
    int num_elem = 0;
    int num_conn = 0;
    int i, j, k;
    int *elem, *conn, *tl;
    float *x_c, *y_c, *z_c;
    float *x_sc, *y_sc, *z_sc;

    for (z = 0; z < nZones; z++)
    {
        grid_array[z]->getGridSize(&i, &j, &k);
        num_coord += i * j * k;
        num_conn += 8 * (i - 1) * (j - 1) * (k - 1);
        num_elem += (i - 1) * (j - 1) * (k - 1);
    }

    cerr << "num_elem:  " << num_elem << endl;
    cerr << "num_coord: " << num_coord << endl;
    cerr << "num_conn:  " << num_conn << endl;

    sprintf(hstr, "%s_ugrid_%d", ugrid_name, timestep);

    //	uGRID = new coDoUnstructuredGrid(hstr, 8, 64, 27, 8);
    uGRID = new coDoUnstructuredGrid(hstr, num_elem, num_conn,
                                     num_coord, 1);
    if (!uGRID->objectOk())
    {
        Covise::sendInfo("new for UnstructuredGrid failed");
        return;
    }

    uGRID->getAddresses(&elem, &conn, &x_c, &y_c, &z_c);
    uGRID->getTypeList(&tl);
    for (i = 0; i < num_elem; i++)
        tl[i] = TYPE_HEXAEDER;

    int start_elem = 0;
    int start_coord = 0;
    int start_conn = 0;
    int si, sj, sk;
    //int delta;

    //    for(z = 0;z < nZones;z++) {  // SIEHE AUCH UNTEN!!!!
    for (z = 0; z < nZones; z++)
    {
        cerr << "Zone " << z << endl;
        cerr << "start_elem: " << start_elem << endl;
        cerr << "start_coord: " << start_coord << endl;
        cerr << "start_conn: " << start_conn << endl;
        grid_array[z]->getGridSize(&si, &sj, &sk);
        cerr << "gridsize: " << si << ", " << sj << ", " << sk << endl;
        grid_array[z]->getAddresses(&x_sc, &y_sc, &z_sc);

        //         for(i = 0;i < si;i++)
        // 			for(j = 0;j < sj;j++)
        //                 for(k = 0;k < sk;k++)
        //                     grid_array[z]->getPointCoordinates(i, &x_c[start_coord + i * sj * sk + j * sk + k],
        //                                                          j, &y_c[start_coord + i * sj * sk + j * sk + k],
        //                                                          k, &z_c[start_coord + i * sj * sk + j * sk + k]);
        //         for(i = 0;i < 3;i++)
        // 			for(j = 0;j < 3;j++)
        //                 for(k = 0;k < 3;k++)
        //                     grid_array[z]->getPointCoordinates(i, &x_c[start_coord + i * 9 + j * 3 + k],
        //                                                          j, &y_c[start_coord + i * 9 + j * 3 + k],
        //                                                          k, &z_c[start_coord + i * 9 + j * 3 + k]);

        memcpy(&(x_c[start_coord]), &x_sc[0], si * sj * sk * sizeof(int));
        memcpy(&(y_c[start_coord]), &y_sc[0], si * sj * sk * sizeof(int));
        memcpy(&(z_c[start_coord]), &z_sc[0], si * sj * sk * sizeof(int));

        for (i = 0; i < (si - 1) * (sj - 1) * (sk - 1); i++)
            elem[start_elem + i] = start_conn + 8 * i;

        //         conn[0] = 1;
        //         conn[1] = 0;
        //         conn[2] = 4;
        //         conn[3] = 5;
        //         conn[4] = 3;
        //         conn[5] = 2;
        //         conn[6] = 6;
        //         conn[7] = 7;
        int os;
        //         for(i = 0;i < 2;i++)
        //             for(j = 0;j < 2;j++)
        //                 for(k = 0;k < 2;k++) {
        //                     os = i * 4 + j * 2 + k;
        //                     conn[os * 8 + 0] = i * 9 + j * 3 + k + 1;
        //                     conn[os * 8 + 1] = i * 9 + j * 3 + k;
        //                     conn[os * 8 + 2] = (i + 1) * 9 + j * 3 + k;
        //                     conn[os * 8 + 3] = (i + 1) * 9 + j * 3 + k + 1;
        //                     conn[os * 8 + 4] = i * 9 + (j + 1) * 3 + k + 1;
        //                     conn[os * 8 + 5] = i * 9 + (j + 1) * 3 + k;
        //                     conn[os * 8 + 6] = (i + 1) * 9 + (j + 1) * 3 + k;
        //                     conn[os * 8 + 7] = (i + 1) * 9 + (j + 1) * 3 + k + 1;
        //                 }

        int last_conn;

        for (i = 0; i < si - 1; i++)
            for (j = 0; j < sj - 1; j++)
                for (k = 0; k < sk - 1; k++)
                {
                    os = i * (sj - 1) * (sk - 1) + j * (sk - 1) + k;

                    conn[start_conn + os * 8 + 0] = start_coord + i * sj * sk + j * sk + k + 1;
                    conn[start_conn + os * 8 + 1] = start_coord + i * sj * sk + j * sk + k;
                    conn[start_conn + os * 8 + 2] = start_coord + (i + 1) * sj * sk + j * sk + k;
                    conn[start_conn + os * 8 + 3] = start_coord + (i + 1) * sj * sk + j * sk + k + 1;
                    conn[start_conn + os * 8 + 4] = start_coord + i * sj * sk + (j + 1) * sk + k + 1;
                    conn[start_conn + os * 8 + 5] = start_coord + i * sj * sk + (j + 1) * sk + k;
                    conn[start_conn + os * 8 + 6] = start_coord + (i + 1) * sj * sk + (j + 1) * sk + k;
                    conn[start_conn + os * 8 + 7] = start_coord + (i + 1) * sj * sk + (j + 1) * sk + k + 1;
                    last_conn = start_conn + os * 8 + 8;
                    //                     delta = i * (sj - 1) * (sk - 1) + j * (sk - 1) + k;
                    //                     int pi = k;
                    //                     int pj = j;
                    //                     int pk = i;
                    //                     int psi = sk;
                    //                     int psj = sj;
                    //                     int psk = si;
                    //                     conn[start_conn + delta + 0] =
                    //                         start_coord + pi * psj * psk + pj * psk + pk;
                    //                     conn[start_conn + delta + 1] =
                    //                         start_coord + (pi + 1) * psj * psk + pj * psk + pk;
                    //                     conn[start_conn + delta + 2] =
                    //                         start_coord + pi * psj * psk + (pj + 1) * psk + pk;
                    //                     conn[start_conn + delta + 3] =
                    //                         start_coord + (pi + 1) * psj * psk + (pj + 1) * psk + pk;
                    //                     conn[start_conn + delta + 4] =
                    //                         start_coord + pi * psj * psk + pj * psk + (pk + 1);
                    //                     conn[start_conn + delta + 5] =
                    //                         start_coord + (pi + 1) * psj * psk + pj * psk + (pk + 1);
                    //                     conn[start_conn + delta + 6] =
                    //                         start_coord + pi * psj * psk + (pj + 1) * psk + (pk + 1);
                    //                     conn[start_conn + delta + 7] =
                    //                         start_coord + (pi + 1) * psj * psk + (pj + 1) * psk + (pk + 1);
                }
        start_coord += si * sj * sk;
        cerr << "start_conn: " << start_conn << endl;
        start_conn += 8 * (si - 1) * (sj - 1) * (sk - 1);
        cerr << "last_conn: " << last_conn << endl;

        start_elem += (si - 1) * (sj - 1) * (sk - 1);

        delete grid_array[z];
    }

    cerr << "End Zones\n";
    cerr << "start_elem: " << start_elem << endl;
    cerr << "start_coord: " << start_coord << endl;
    cerr << "start_conn: " << start_conn << endl;

    uGRID_sets[current_file] = uGRID;
    uGRID_sets[current_file + 1] = NULL;

    GRID_sets[current_file] = GRID_Set;
    GRID_sets[current_file + 1] = NULL;
    if (curZone[3] && curZone[4] && curZone[5])
    {
        curDescr = zone;
        char dummy[128];
        sprintf(dummy, "%s_%d", data_name[0], timestep);
        VECTOR_Set = new coDoSet(dummy, SET_CREATE);
        VECTOR_Set->addAttribute("DATA_NAME", VarNames[usedVars[3]]);
        for (z = 0; z < nZones; z++)
        {
            curZone[3] = curZone[3]->next;
            curZone[4] = curZone[4]->next;
            curZone[5] = curZone[5]->next;
            curDescr = curDescr->next;
            sprintf(hstr, "%s_zone%i_%d", data_name[0], z, timestep);
            vector_array[z] = new coDoVec3(hstr, curDescr->k,
                                           curDescr->j,
                                           curDescr->i,
                                           curZone[3]->value,
                                           curZone[4]->value,
                                           curZone[5]->value);

            VECTOR_Set->addElement(vector_array[z]);
            //          delete VECTOR;
        }
        VECTOR_sets[current_file] = VECTOR_Set;
        VECTOR_sets[current_file + 1] = NULL;

        start_elem = 0;
        start_coord = 0;
        start_conn = 0;

        float *uu, *vu, *wu;
        float *us, *vs, *ws;
        int no_of_points;

        sprintf(dummy, "%s_%d", udata_name[0], timestep);
        uVECTOR = new coDoVec3(dummy, num_coord);
        uVECTOR->getAddresses(&uu, &vu, &wu);
        uVECTOR->addAttribute("DATA_NAME", VarNames[usedVars[3]]);
        //    for (z=0; z<nZones; z++) {
        for (z = 0; z < nZones; z++)
        {
            vector_array[z]->getGridSize(&i, &j, &k);
            no_of_points = i * j * k;
            vector_array[z]->getAddresses(&us, &vs, &ws);
            memcpy(&(uu[start_coord]), &us[0], sizeof(float) * no_of_points);
            memcpy(&(vu[start_coord]), &vs[0], sizeof(float) * no_of_points);
            memcpy(&(wu[start_coord]), &ws[0], sizeof(float) * no_of_points);
            delete vector_array[z];
            start_coord += no_of_points;
        }
        uVECTOR_sets[current_file] = uVECTOR;
        uVECTOR_sets[current_file + 1] = NULL;
    }

    for (v = 0; v < 2; v++)
        if (curZone[v + 6] != NULL)
        {
            curDescr = zone;
            char dummy[128];
            sprintf(dummy, "%s_%d", data_name[v + 1], timestep);
            DATA_Set[v] = new coDoSet(dummy, SET_CREATE);
            DATA_Set[v]->addAttribute("DATA_NAME", VarNames[usedVars[v + 6]]);
            for (z = 0; z < nZones; z++)
            {
                curZone[v + 6] = curZone[v + 6]->next;
                curDescr = curDescr->next;
                sprintf(hstr, "%s_zone%i_%d", data_name[v + 1], z, timestep);
                scalar_array[z] = new coDoFloat(hstr, curDescr->k,
                                                curDescr->j,
                                                curDescr->i,
                                                curZone[v + 6]->value);

                DATA_Set[v]->addElement(scalar_array[z]);
            }
            DATA_sets[v][current_file] = DATA_Set[v];
            DATA_sets[v][current_file + 1] = NULL;

            start_elem = 0;
            start_coord = 0;
            start_conn = 0;

            float *su;
            float *ss;
            int no_of_points;

            sprintf(dummy, "%s_%d", udata_name[v + 1], timestep);
            uSCALAR = new coDoFloat(dummy, num_coord);
            uSCALAR->getAddress(&su);
            uSCALAR->addAttribute("DATA_NAME", VarNames[usedVars[v + 6]]);
            //    for (z=0; z<nZones; z++) {
            for (z = 0; z < nZones; z++)
            {
                int si, sj, sk;
                scalar_array[z]->getGridSize(&si, &sj, &sk);
                no_of_points = i * j * k;
                //        for(i = 0;i < si;i++)
                //            for(j = 0;j < sj;j++)
                //                for(k = 0;k < sk;k++)
                //                    scalar_array[z]->getPointValue(i, j, k, &(su[start_coord + si * j * k + sj * k + sk]));
                scalar_array[z]->getAddress(&ss);
                memcpy(&(su[start_coord]), &ss[0], sizeof(float) * no_of_points);
                delete scalar_array[z];
                start_coord += no_of_points;
            }
            uSCALAR_sets[v][current_file] = uSCALAR;
            uSCALAR_sets[v][current_file + 1] = NULL;
        }
    delete[] curZone;
}

void Application::createOutputObjects()
{
    int z, v, cnt;
    char hstr[100];
    DynZoneElement **curZone;
    DynZoneDescr *curDescr;

    curZone = new DynZoneElement *[nVars];

    //fprintf(stderr, "usedVars[]: %d %d %d %d %d %d\n", usedVars[0], usedVars[1], usedVars[2], usedVars[3], usedVars[4], usedVars[5]);

    for (cnt = 0; cnt < 8; cnt++)
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

    if (curZone[3] && curZone[4] && curZone[5])
    {
        curDescr = zone;
        VECTOR_Set = new coDoSet(data_name[0], SET_CREATE);
        VECTOR_Set->addAttribute("DATA_NAME", VarNames[usedVars[3]]);
        for (z = 0; z < nZones; z++)
        {
            curZone[3] = curZone[3]->next;
            curZone[4] = curZone[4]->next;
            curZone[5] = curZone[5]->next;
            curDescr = curDescr->next;
            sprintf(hstr, "%s_zone%i", data_name[0], z);
            VECTOR = new coDoVec3(hstr, curDescr->k,
                                  curDescr->j,
                                  curDescr->i,
                                  curZone[3]->value,
                                  curZone[4]->value,
                                  curZone[5]->value);

            VECTOR_Set->addElement(VECTOR);
            delete VECTOR;
        }
        delete VECTOR_Set;
    }

    for (v = 0; v < 2; v++)
        if (curZone[v + 6] != NULL)
        {
            curDescr = zone;
            DATA_Set[v] = new coDoSet(data_name[v + 1], SET_CREATE);
            DATA_Set[v]->addAttribute("DATA_NAME", VarNames[usedVars[v + 6]]);
            for (z = 0; z < nZones; z++)
            {
                curZone[v + 6] = curZone[v + 6]->next;
                curDescr = curDescr->next;
                sprintf(hstr, "%s_zone%i", data_name[v + 1], z);
                DATA = new coDoFloat(hstr, curDescr->k,
                                     curDescr->j,
                                     curDescr->i,
                                     curZone[v + 6]->value);

                DATA_Set[v]->addElement(DATA);
                delete DATA;
            }
            delete DATA_Set[v];
        }
    delete[] curZone;
}
