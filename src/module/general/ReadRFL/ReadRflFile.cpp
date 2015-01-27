/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// =============================================================================
// READRFL Klasse zum lesen von ANSYS RFL-Ergebnisfiles (FLOWTRAN)
// -----------------------------------------------------------------------------
// 17.9.2001  Björn Sander
// =============================================================================

#include "ReadRflFile.h"

const char *dofname[32] = {
    "UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ",
    "AX", "AY", "AZ", "VX", "VY", "VZ",
    "unused1", "unused2", "unused3", "unused4", "unused5", "unused6",
    "PRES", "TEMP", "VOLT", "MAG", "ENKE", "ENDS", "EMF",
    "CURR", "SP01", "SP02", "SO03", "SP04", "SP05", "SP06"
};

const char *exdofname[28] = {
    "DENS", "VISC", "EVIS", "COND", "ECON", "LMD1", "LMD2", "LMD3",
    "LMD4", "LMD5", "LMD6", "EMD1", "EMD2", "EMD3", "EMD4", "EMD5",
    "EMD6", "PTOT", "TTOT", "PCOE", "MACH", "STRM", "HFLU", "HFLM",
    "YPLU", "TAUW", "SPHT", "CMUV"
};

// =============================================================================
// Konstruktor / Destruktor
// =============================================================================

// -----------------------------------------------
// Konstruktor
// -----------------------------------------------

READRFL::READRFL(void)
{
    dofroot = NULL;
    rfp = NULL;
    ety = NULL;
    node = NULL;
    element = NULL;
    anznodes = 0;
    anzety = 0;
    anzelem = 0;
    nodeindex = NULL;
    elemindex = NULL;
    timetable = NULL;
}

// -----------------------------------------------
// Destruktor
// -----------------------------------------------
READRFL::~READRFL()
{
    Reset();
}

// =============================================================================
// Interne Methoden
// =============================================================================

// -----------------------------------------------
// Reset
// -----------------------------------------------
int READRFL::Reset(void)
{
    DOFROOT *act;
    int i;

    // DOF-Liste löschen
    while (dofroot != NULL)
    {
        act = dofroot->next;
        delete dofroot->data;
        delete dofroot;
        dofroot = act;
    }
    if (node != NULL)
    {
        delete node;
        node = NULL;
    }
    anznodes = 0;
    if (ety != NULL)
    {
        delete ety;
        ety = NULL;
    }
    anzety = 0;
    if (element != NULL)
    {
        for (i = 0; i < anzelem; ++i)
        {
            delete element[i].nodes;
        }
        delete element;
        element = NULL;
    }
    anzelem = 0;
    memset(&header, 0, sizeof(BINHEADER));
    memset(&rstheader, 0, sizeof(RSTHEADER));
    if (rfp != NULL)
    {
        fclose(rfp);
        rfp = NULL;
    }
    delete nodeindex;
    delete elemindex;
    delete timetable;

    return (0);
}

// -----------------------------------------------
// OpenFile
// -----------------------------------------------
// Öffnet den Ergebnisfile, liest den header und setzt
// den Read-File-Pointer rfp;
// Rückgabe:
// 0    : alles OK
// 1    : File not found
// 2    : could not read header
// 3    : Read Error Nodal equivalence
// 4    : Read Error Element equivalence
// 5    : Read Error Time table
int READRFL::OpenFile(char *filename)
{
    int *buf;
    int offset, i;

    if ((rfp = fopen(filename, "rb")) == NULL)
    {
        return (1);
    }

    // File offen und OK
    // erst mal zwei INTs lesen (lead-in):
    // int 1 ergibt die Länge des folgenden Records in bytes
    // int 2 ergibt den Typ des Records (z.B. 0x64==Integer)
    buf = new int[1024];
    if (fread(buf, sizeof(int), 103, rfp) != 103)
    {
        return (2);
    }

    // alle INT-Elemente umdrehen
    header.filenum = SwitchEndian(buf[2]);
    header.format = SwitchEndian(buf[3]);
    header.time = SwitchEndian(buf[4]);
    header.date = SwitchEndian(buf[5]);
    header.unit = SwitchEndian(buf[10]);
    header.version = SwitchEndian(buf[11]);
    header.ansysdate = SwitchEndian(buf[12]);
    memcpy(header.machine, &buf[13], 3 * sizeof(int));
    header.machine[11] = '\0';
    memcpy(header.jobname, &buf[16], 2 * sizeof(int));
    header.jobname[7] = '\0';
    memcpy(header.product, &buf[18], 2 * sizeof(int));
    header.product[7] = '\0';
    memcpy(header.label, &buf[20], 1 * sizeof(int));
    header.label[3] = '\0';
    memcpy(header.user, &buf[21], 3 * sizeof(int));
    header.user[11] = '\0';
    memcpy(header.machine2, &buf[24], 3 * sizeof(int));
    header.machine2[11] = '\0';
    header.recordsize = SwitchEndian(buf[27]);
    header.maxfilelen = SwitchEndian(buf[28]);
    header.maxrecnum = SwitchEndian(buf[29]);
    header.cpus = SwitchEndian(buf[30]);
    memcpy(header.title, &buf[42], 20 * sizeof(int));
    header.title[79] = '\0';
    memcpy(header.subtitle, &buf[62], 20 * sizeof(int));
    header.subtitle[79] = '\0';
    // File Pointer steht jetzt auf Position (100+3)*4=412 Bytes

    // jetzt den RST-File Header reinbasteln
    if (fread(buf, sizeof(int), 43, rfp) != 43)
    {
        return (2);
    }
    // Werte zuweisen
    rstheader.fun12 = SwitchEndian(buf[2]);
    rstheader.maxnodes = SwitchEndian(buf[3]);
    rstheader.usednodes = SwitchEndian(buf[4]);
    rstheader.maxres = SwitchEndian(buf[5]);
    rstheader.numdofs = SwitchEndian(buf[6]);
    rstheader.maxelement = SwitchEndian(buf[7]);
    rstheader.numelement = SwitchEndian(buf[8]);
    rstheader.analysis = SwitchEndian(buf[9]);
    rstheader.numsets = SwitchEndian(buf[10]);
    rstheader.ptr_eof = SwitchEndian(buf[11]);
    rstheader.ptr_dsi = SwitchEndian(buf[12]);
    rstheader.ptr_time = SwitchEndian(buf[13]);
    rstheader.ptr_load = SwitchEndian(buf[14]);
    rstheader.ptr_elm = SwitchEndian(buf[15]);
    rstheader.ptr_node = SwitchEndian(buf[16]);
    rstheader.ptr_geo = SwitchEndian(buf[17]);
    rstheader.units = SwitchEndian(buf[21]);
    rstheader.numsectors = SwitchEndian(buf[22]);
    rstheader.ptr_end = 0;

#ifdef BYTESWAP
    rstheader.ptr_end = (long long)(buf[23]) << 32 | buf[24];
#else
    rstheader.ptr_end = (long long)(SwitchEndian(buf[24])) << 32 | SwitchEndian(buf[23]);
#endif
    // Header fertig gelesen

    // Jetzt noch die Indextabellen laden
    offset = (rstheader.ptr_node + 2) * 4;
    fseek(rfp, offset, SEEK_SET);

    delete buf;
    buf = new int[rstheader.usednodes];
    if (fread(buf, sizeof(int), rstheader.usednodes, rfp) != rstheader.usednodes)
    {
        return (3);
    }
    // Jetzt zuweisen
    nodeindex = new int[rstheader.usednodes];
    for (i = 0; i < rstheader.usednodes; ++i)
    {
        nodeindex[i] = SwitchEndian(buf[i]);
    }

    // das selbe für die Elemente
    offset = (rstheader.ptr_elm + 2) * 4;
    fseek(rfp, offset, SEEK_SET);

    delete buf;
    buf = new int[rstheader.numelement];
    if (fread(buf, sizeof(int), rstheader.numelement, rfp) != rstheader.numelement)
    {
        return (4);
    }
    // Jetzt zuweisen
    elemindex = new int[rstheader.numelement];
#ifdef BYTESWAP
    for (i = 0; i < rstheader.numelement; ++i)
    {
        elemindex[i] = SwitchEndian(buf[i]);
    }
#endif
    delete buf;

    // Timetable lesen
    timetable = new double[rstheader.maxres];
    offset = (rstheader.ptr_time + 2) * 4;
    fseek(rfp, offset, SEEK_SET);
    if (fread(timetable, sizeof(double), rstheader.maxres, rfp) != rstheader.maxres)
    {
        return (6);
    }
// nochmal Endian umsetzen
#ifdef BYTESWAP
    for (i = 0; i < rstheader.maxres; ++i)
    {
        timetable[i] = SwitchEndian(timetable[i]);
    }
#endif

    // That's all folks
    return (0);
}

// -----------------------------------------------
// SwitchEndian
// -----------------------------------------------
int READRFL::SwitchEndian(int value)
{
#ifdef BIG_ENDIAN
    return (value);
#else
    int ret = 0;
    int bytes[4];

    bytes[0] = (value & 0x000000FF) << 24;
    bytes[1] = (value & 0x0000FF00) << 8;
    bytes[2] = (value & 0x00FF0000) >> 8;
    bytes[3] = (value & 0xFF000000) >> 24;

    ret = bytes[0] | bytes[1] | bytes[2] | bytes[3];
    return (ret);
#endif
}

double READRFL::SwitchEndian(double dval)
{
#ifdef BIG_ENDIAN
    return (dval);
#else
    int tmp[2], rettmp[2];
    double value = dval;
    memcpy(tmp, &value, sizeof(double));
    rettmp[1] = SwitchEndian(tmp[0]);
    rettmp[0] = SwitchEndian(tmp[1]);
    memcpy(&value, rettmp, sizeof(double));
    return (value);
#endif
}

// -----------------------------------------------
// CreateNewDOFList
// -----------------------------------------------
DOFROOT *READRFL::CreateNewDOFList(void)
{
    DOFROOT *act;

    if (dofroot == NULL)
    {
        dofroot = new DOFROOT;
        act = dofroot;
    }
    else
    {
        act = dofroot;
        while (act->next != NULL)
            act = act->next;
        act->next = new DOFROOT;
        act = act->next;
    }
    act->next = NULL;

    return (act);
}

// =============================================================================
// externe Methoden
// =============================================================================

// -----------------------------------------------
// Read
// -----------------------------------------------
int READRFL::Read(char *filename, int num)
{
    Reset(); // alles löschen, wenn nötig
    // Mal keine Fehlerabfrage, FIXME!
    switch (OpenFile(filename))
    {
    // 0    : alles OK
    // 1    : File not found
    // 2    : could not read header
    // 3    : Read Error Nodal equivalence
    // 4    : Read Error Element equivalence
    // 5    : Read Error Time table
    case 0:
        printf("Open file OK\n");
        break;

    case 1:
        printf("Open file: file not found \n");
        break;

    case 2:
        printf("Open file: Read Error, header \n");
        break;

    case 3:
        printf("Open file: Read Error, nodal equ tabular \n");
        break;

    case 4:
        printf("Open file: Read Error, element equi tabular \n");
        break;

    case 5:
        printf("Open file: Read Error, time table \n");
        break;
    }
    switch (GetDataset(num))
    {
    // 1        : File ist nicht offen/initialisiert
    // 2        : Read Error DSI-Tabelle
    // 3        : Num ist nicht im Datensatz
    // 4        : Read Error Solution Header
    // 5        : Read Error DOFs
    // 6        : Read Error exDOFs
    case 0:
        printf("GetData : OK\n");
        break;

    case 1:
        printf("GetData : file not open\n");
        break;

    case 2:
        printf("GetData : read error: DSI\n");
        break;

    case 3:
        printf("GetData : num exeeds limits\n");
        break;

    case 4:
        printf("GetData : read error solution header\n");
        break;

    case 5:
        printf("GetData : read error DOFs\n");
        break;

    case 6:
        printf("GetData : read error exDOF\n");
        break;
    }
    switch (GetNodes())
    {
    // 1        : Read Error Geometrieheader
    // 2        : Read Error Nodes
    // 3        : Read Error Elementbeschreibung
    // 4        : Read Error ETYs
    // 5        : Read Error Elementtabelle
    // 6        : Read Error Elemente
    case 0:
        printf("GetNodes : ok\n");
        break;

    case 1:
        printf("GetNodes : read error geo\n");
        break;

    case 2:
        printf("GetNodes : read error nodes\n");
        break;

    case 3:
        printf("GetNodes : read error element description\n");
        break;

    case 4:
        printf("GetNodes : read error ety\n");
        break;

    case 5:
        printf("GetNodes : read error element tabular\n");
        break;

    case 6:
        printf("GetNodes : read error elements\n");
        break;
    }
    //  fclose(rfp);
    //  rfp=NULL;
    return (0);
}

// -----------------------------------------------
// ReadSHDR
// -----------------------------------------------
int READRFL::ReadSHDR(int num)
{
    int *buf = NULL, solbuf[103], *exdofbuf = NULL;
    int size, i;
    long long offset;
    double dsol[100];

    //  SOLUTIONHEADER shdr;

    // File sollte offen sein
    if (rfp == NULL)
        return (1);

    // Out of range check
    if (num > rstheader.numsets)
        return (3);

    // Springe erst mal zu der DSI-Tabelle
    offset = rstheader.ptr_dsi * 4;
    fseek(rfp, (long)offset, SEEK_SET); // im pointer steht die Anzahl der int-Elemente vom Anfang

    // Jetzt sollte man die Tabelle Lesen können. Die ist mitunter aber recht groß
    // gewöhnlich beinhaltet sie 2*1000 Einträge (besser 1000 64-bit Pointer)
    // dazu kommt dann immer noch der Lead-in (2 Ints) und er Lead-out (1 int)
    size = 2 * rstheader.maxres + 3;
    buf = new int[size];

    if (fread(buf, sizeof(int), size, rfp) != size)
    {
        return (2);
    }
    // jetzt mal diese Tabelle auswerten
    solheader.offset = 0;
// Hi/Lo lesen umdrehen und einfügen
#ifdef BIG_ENDIAN
    solheader.offset = ((long long)(buf[num + 2 + rstheader.maxres]) << 32 | buf[num + 1]) * 4;
#else
    solheader.offset = ((long long)(SwitchEndian(buf[num + 2 + rstheader.maxres])) << 32 | SwitchEndian(buf[num + 1])) * 4;
#endif
    // jetzt da hin springen und dort einen Solution Header lesen
    fseek(rfp, (long)solheader.offset, SEEK_SET);
    if (fread(solbuf, sizeof(int), 103, rfp) != 103)
    {
        return (4);
    }
    // Jetzt die Werte decodieren und zuweisen
    solheader.numelements = SwitchEndian(solbuf[3]);
    solheader.numnodes = SwitchEndian(solbuf[4]);
    solheader.mask = SwitchEndian(solbuf[5]);
    solheader.loadstep = SwitchEndian(solbuf[6]);
    solheader.iteration = SwitchEndian(solbuf[7]);
    solheader.sumiteration = SwitchEndian(solbuf[8]);
    solheader.numreact = SwitchEndian(solbuf[9]);
    solheader.maxesz = SwitchEndian(solbuf[10]);
    solheader.nummasters = SwitchEndian(solbuf[11]);
    solheader.ptr_nodalsol = SwitchEndian(solbuf[12]);
    solheader.ptr_elemsol = SwitchEndian(solbuf[13]);
    solheader.ptr_react = SwitchEndian(solbuf[14]);
    solheader.ptr_masters = SwitchEndian(solbuf[15]);
    solheader.ptr_bc = SwitchEndian(solbuf[16]);
    solheader.extrapolate = SwitchEndian(solbuf[17]);
    solheader.mode = SwitchEndian(solbuf[18]);
    solheader.symmetry = SwitchEndian(solbuf[19]);
    solheader.complex = SwitchEndian(solbuf[20]);
    solheader.numdofs = SwitchEndian(solbuf[21]);
    // jetzt Titel und Subtitel lesen
    memcpy(solheader.title, &solbuf[52], sizeof(int) * 20);
    solheader.title[79] = '\0';
    memcpy(solheader.subtitle, &solbuf[72], sizeof(int) * 20);
    solheader.subtitle[79] = '\0';
    // weiter gehts
    solheader.changetime = SwitchEndian(solbuf[92]);
    solheader.changedate = SwitchEndian(solbuf[93]);
    solheader.changecount = SwitchEndian(solbuf[94]);
    solheader.soltime = SwitchEndian(solbuf[95]);
    solheader.soldate = SwitchEndian(solbuf[96]);
    solheader.ptr_onodes = SwitchEndian(solbuf[97]);
    solheader.ptr_oelements = SwitchEndian(solbuf[98]);
    solheader.numexdofs = SwitchEndian(solbuf[99]);
    solheader.ptr_extra_a = SwitchEndian(solbuf[100]);
    solheader.ptr_extra_t = SwitchEndian(solbuf[101]);

    // DOFs einlesen
    solheader.dof = new int[solheader.numdofs];
    // erst mal die normalen DOFs kopieren
    for (i = 0; i < solheader.numdofs; ++i)
        solheader.dof[i] = SwitchEndian(solbuf[22 + i]);
    // exdofs reinhauen
    solheader.exdof = new int[solheader.numexdofs];

    // Jetzt die TIME Varioable aus dem folgenden Daten besorgen
    // Zwei ints überspringen
    fseek(rfp, 2 * 4, SEEK_CUR);
    if (fread(dsol, sizeof(double), 100, rfp) != 100)
    {
        return (7);
    }
// Time Variable sollte an erster Stelle stehen
#ifdef BIG_ENDIAN
    solheader.time = dsol[0];
#else
    solheader.time = SwitchEndian(dsol[0]);
#endif

    offset = solheader.offset + (solheader.ptr_extra_a + 2) * 4;
    fseek(rfp, (long)offset, SEEK_SET);
    exdofbuf = new int[64 * sizeof(int)];
    if (fread(exdofbuf, sizeof(int), 64, rfp) != 64)
    {
        return (6);
    }
    for (i = 0; i < solheader.numexdofs; ++i)
        solheader.exdof[i] = SwitchEndian(exdofbuf[i]);

    delete buf;

    return (0);
}

// -----------------------------------------------
// GetDataset
// -----------------------------------------------
// Liest die Ergebnisdaten für einen Datensatz aus dem File
// Rückgabe:
// 0        : alles OK
// 1        : File ist nicht offen/initialisiert
// 2        : Read Error DSI-Tabelle
// 3        : Num ist nicht im Datensatz
// 4        : Read Error Solution Header
// 5        : Read Error DOFs
// 6        : Read Error exDOFs
int READRFL::GetDataset(int num)
{
    //  int *buf=NULL;
    int size, i, j, sumdof;
    long long offset;
    double *dof;
    DOFROOT *act;

    // File sollte offen sein
    if (rfp == NULL)
        return (1);

    // Out of range check
    if (num > rstheader.numsets)
        return (3);

    /*
     // Springe erst mal zu der DSI-Tabelle
     offset = rstheader.ptr_dsi*4;
     fseek(rfp,offset,SEEK_SET); // im pointer steht die Anzahl der int-Elemente vom Anfang

     // Jetzt sollte man die Tabelle Lesen können. Die ist mitunter aber recht groß
     // gewöhnlich beinhaltet sie 2*1000 Einträge (besser 1000 64-bit Pointer)
     // dazu kommt dann immer noch der Lead-in (2 Ints) und er Lead-out (1 int)
     size = 2*rstheader.maxres+3;
     buf = new int[size];

   if (fread(buf,sizeof(int),size,rfp)!=size)
   {
   return(2);
   }
   // jetzt mal diese Tabelle auswerten
   soloffset=0;
   // Hi/Lo lesen umdrehen und einfügen
   #ifdef BIG_ENDIAN
   soloffset = ((long long)(buf[num+2+rstheader.maxres])<<32 | buf[num+1])*4;
   #else
   soloffset = ((long long)(SwitchEndian(buf[num+2+rstheader.maxres])) << 32 | SwitchEndian(buf[num+1]))*4;
   #endif
   // jetzt da hin springen und dort einen Solution Header lesen

   fseek(rfp,soloffset,SEEK_SET);
   if (fread(solbuf,sizeof(int),103,rfp)!=103)
   {
   return(4);
   }
   // Jetzt die Werte decodieren und zuweisen
   shdr.numelements    = SwitchEndian(solbuf[3]);
   shdr.numnodes       = SwitchEndian(solbuf[4]);
   shdr.mask           = SwitchEndian(solbuf[5]);
   shdr.loadstep       = SwitchEndian(solbuf[6]);
   shdr.iteration      = SwitchEndian(solbuf[7]);
   shdr.sumiteration   = SwitchEndian(solbuf[8]);
   shdr.numreact       = SwitchEndian(solbuf[9]);
   shdr.maxesz         = SwitchEndian(solbuf[10]);
   shdr.nummasters     = SwitchEndian(solbuf[11]);
   shdr.ptr_nodalsol   = SwitchEndian(solbuf[12]);
   shdr.ptr_elemsol    = SwitchEndian(solbuf[13]);
   shdr.ptr_react      = SwitchEndian(solbuf[14]);
   shdr.ptr_masters    = SwitchEndian(solbuf[15]);
   shdr.ptr_bc         = SwitchEndian(solbuf[16]);
   shdr.extrapolate    = SwitchEndian(solbuf[17]);
   shdr.mode           = SwitchEndian(solbuf[18]);
   shdr.symmetry       = SwitchEndian(solbuf[19]);
   shdr.complex        = SwitchEndian(solbuf[20]);
   shdr.numdofs        = SwitchEndian(solbuf[21]);
   // jetzt Titel und Subtitel lesen
   memcpy(shdr.title,&solbuf[52],sizeof(int)*20);
   shdr.title[79]='\0';
   memcpy(shdr.subtitle,&solbuf[72],sizeof(int)*20);
   shdr.subtitle[79]='\0';
   // weiter gehts
   shdr.time           = SwitchEndian(solbuf[92]);
   shdr.date           = SwitchEndian(solbuf[93]);
   shdr.changecount    = SwitchEndian(solbuf[94]);
   shdr.soltime        = SwitchEndian(solbuf[95]);
   shdr.soldate        = SwitchEndian(solbuf[96]);
   shdr.ptr_onodes     = SwitchEndian(solbuf[97]);
   shdr.ptr_oelements  = SwitchEndian(solbuf[98]);
   shdr.numexdofs      = SwitchEndian(solbuf[99]);
   shdr.ptr_extra_a    = SwitchEndian(solbuf[100]);
   shdr.ptr_extra_t    = SwitchEndian(solbuf[101]);

   // DOFs einlesen
   shdr.dof = new int[shdr.numdofs];
   // erst mal die normalen DOFs kopieren
   for (i=0;i<shdr.numdofs;++i)
   shdr.dof[i] = SwitchEndian(solbuf[22+i]);
   // exdofs reinhauen
   shdr.exdof = new int[shdr.numexdofs];
   offset = soloffset+(shdr.ptr_extra_a+2)*4;
   fseek(rfp,offset,SEEK_SET);
   exdofbuf = new int[64*sizeof(int)];
   if (fread(exdofbuf,sizeof(int),64,rfp)!=64)
   {
   return(6);
   }
   for (i=0;i<shdr.numexdofs;++i)
   shdr.exdof[i] = SwitchEndian(exdofbuf[i]);

   */

    // Eventuell alte Daten Löschen
    // DOF-Liste löschen
    while (dofroot != NULL)
    {
        act = dofroot->next;
        delete dofroot->data;
        delete dofroot;
        dofroot = act;
    }
    if (node != NULL)
    {
        delete node;
        node = NULL;
    }
    anznodes = 0;
    if (ety != NULL)
    {
        delete ety;
        ety = NULL;
    }
    anzety = 0;
    if (element != NULL)
    {
        for (i = 0; i < anzelem; ++i)
        {
            delete element[i].nodes;
        }
        delete element;
        element = NULL;
    }
    anzelem = 0;
    // so, alles gelöscht

    ReadSHDR(num);

    // jetzt Daten einlesen
    offset = solheader.offset + (solheader.ptr_nodalsol + 2) * 4;
    fseek(rfp, (long)offset, SEEK_SET);
    sumdof = solheader.numdofs + solheader.numexdofs;
    size = solheader.numnodes * (sumdof);
    dof = new double[size];

    // Her damit!
    if (fread(dof, sizeof(double), size, rfp) != size)
    {
        return (5);
    }
    // Lead out wird gekickt!

    // Erst mal pro DOF einen double-Array erstellen
    for (i = 0; i < sumdof; ++i)
    {
        act = CreateNewDOFList();
        act->dataset = num;
        act->typ = solheader.dof[i];
        if (i < solheader.numdofs)
        {
            act->exdof = false;
            sprintf(act->name, "%s", dofname[solheader.dof[i] - 1]);
        }
        else
        {
            act->exdof = true;
            sprintf(act->name, "%s", exdofname[solheader.exdof[i - solheader.numdofs] - 1]);
        }
        act->anz = solheader.numnodes;
        act->data = new double[act->anz];
        if (!(solheader.mask & 0x200)) // Nur Teilbereich der Knoten wurde verwendet
        {
            for (j = 0; j < solheader.numnodes; ++j)
            {
                act->data[j] = SwitchEndian(dof[j * sumdof + i]);
            }
        }
        else
            memset(act->data, 0, sizeof(double) * solheader.numnodes);
        // ACHTUNG: Extradofs werden falsch bezeichnet!
        // ACHTUNG: Bei Teilbereich sind alle Daten Null!
    }
    delete dof;
    // Alle dofs gelesen und gespeichert!

    return (0);
}

// -----------------------------------------------
// GetNodes
// -----------------------------------------------
// Liest die Koordinaten der Knoten ein
// Rückgabe:
// 0        : alles ok
// 1        : Read Error Geometrieheader
// 2        : Read Error Nodes
// 3        : Read Error Elementbeschreibung
// 4        : Read Error ETYs
// 5        : Read Error Elementtabelle
// 6        : Read Error Elemente
int READRFL::GetNodes(void)
{
    GEOMETRYHEADER ghdr;
    long long offset;
    int size, *buf, i, j, *etybuf;

    // Springe erst mal zu der Geometrie-Tabelle
    offset = rstheader.ptr_geo * 4;
    fseek(rfp, (long)offset, SEEK_SET); // im pointer steht die Anzahl der int-Elemente vom Anfang

    size = 23;
    buf = new int[size];

    if (fread(buf, sizeof(int), size, rfp) != size)
    {
        return (1);
    }

    // Werte zuweisen
    ghdr.maxety = SwitchEndian(buf[3]);
    ghdr.maxrl = SwitchEndian(buf[4]);
    ghdr.nodes = SwitchEndian(buf[5]);
    ghdr.elements = SwitchEndian(buf[6]);
    ghdr.maxcoord = SwitchEndian(buf[7]);
    ghdr.ptr_ety = SwitchEndian(buf[8]);
    ghdr.ptr_rel = SwitchEndian(buf[9]);
    ghdr.ptr_nodes = SwitchEndian(buf[10]);
    ghdr.ptr_sys = SwitchEndian(buf[11]);
    ghdr.ptr_eid = SwitchEndian(buf[12]);
    ghdr.ptr_mas = SwitchEndian(buf[17]);
    ghdr.coordsize = SwitchEndian(buf[18]);
    ghdr.elemsize = SwitchEndian(buf[19]);
    ghdr.etysize = SwitchEndian(buf[20]);
    ghdr.rcsize = SwitchEndian(buf[21]);

    // Jetzt zu den KNoten springen und diese lesen (Lead in überspringen)
    offset = (ghdr.ptr_nodes + 2) * 4;
    fseek(rfp, (long)offset, SEEK_SET);

    // Jetzt die NODES definieren
    node = new NODE[ghdr.nodes];
    for (i = 0; i < ghdr.nodes; ++i)
    {
        if (fread(&node[i], sizeof(NODE), 1, rfp) != 1)
            return (2);

        // Werte jetzt umdrehen
        node[i].id = SwitchEndian(node[i].id);
        node[i].x = SwitchEndian(node[i].x);
        node[i].y = SwitchEndian(node[i].y);
        node[i].z = SwitchEndian(node[i].z);
        node[i].thxy = SwitchEndian(node[i].thxy);
        node[i].thyz = SwitchEndian(node[i].thyz);
        node[i].thzx = SwitchEndian(node[i].thzx);
    }
    // das war's!
    anznodes = ghdr.nodes;

    // Jetzt die Elemente lesen: zuerst mal zu ETY um die Elementbeschreibungen zu laden
    offset = (ghdr.ptr_ety + 2) * 4;
    fseek(rfp, (long)offset, SEEK_SET);

    delete buf;

    buf = new int[ghdr.maxety];
    if (fread(buf, sizeof(int), ghdr.maxety, rfp) != ghdr.maxety)
    {
        return (3);
    }

    // ETYs im Objekt erstellen
    ety = new ETY[ghdr.maxety];
    anzety = ghdr.maxety;
    etybuf = new int[ghdr.etysize];

    for (i = 0; i < ghdr.maxety; ++i)
    {
        offset = (SwitchEndian(buf[i]) + 2) * 4;
        fseek(rfp, (long)offset, SEEK_SET);
        if (fread(etybuf, sizeof(int), ghdr.etysize, rfp) != ghdr.etysize)
        {
            return (4);
        }
        // Daten jetzt in den ety bringen
        ety[i].id = SwitchEndian(etybuf[0]);
        ety[i].routine = SwitchEndian(etybuf[1]);
        ety[i].keyops[0] = SwitchEndian(etybuf[2]);
        ety[i].keyops[1] = SwitchEndian(etybuf[3]);
        ety[i].keyops[2] = SwitchEndian(etybuf[4]);
        ety[i].keyops[3] = SwitchEndian(etybuf[5]);
        ety[i].keyops[4] = SwitchEndian(etybuf[6]);
        ety[i].keyops[5] = SwitchEndian(etybuf[7]);
        ety[i].keyops[6] = SwitchEndian(etybuf[8]);
        ety[i].keyops[7] = SwitchEndian(etybuf[9]);
        ety[i].keyops[8] = SwitchEndian(etybuf[10]);
        ety[i].keyops[9] = SwitchEndian(etybuf[11]);
        ety[i].keyops[10] = SwitchEndian(etybuf[12]);
        ety[i].keyops[11] = SwitchEndian(etybuf[13]);
        ety[i].dofpernode = SwitchEndian(etybuf[33]);
        ety[i].nodes = SwitchEndian(etybuf[60]);
        ety[i].nodeforce = SwitchEndian(etybuf[62]);
        ety[i].nodestress = SwitchEndian(etybuf[93]);
    }
    delete buf;
    delete etybuf;
    // Passt schon!
    // Jetzt die Elemente selber einlesen
    element = new ELEMENT[ghdr.elements];

    anzelem = ghdr.elements;

    buf = new int[ghdr.elements];

    // hinsurfen und lesen
    offset = (ghdr.ptr_eid + 2) * 4;
    fseek(rfp, (long)offset, SEEK_SET);
    if (fread(buf, sizeof(int), ghdr.elements, rfp) != ghdr.elements)
    {
        return (5);
    }

    etybuf = new int[100]; // ist grosszügig bemessen! Länge ist variabel zur Laufzeit
    // Jetzt mit Schleife alle Elemente packen
    for (i = 0; i < ghdr.elements; ++i)
    {
        offset = (SwitchEndian(buf[i]) + 2) * 4;
        fseek(rfp, (long)offset, SEEK_SET);
        if (fread(etybuf, sizeof(int), 100, rfp) != 100)
        {
            return (6);
        }
        // Jetzt Daten zuweisen
        element[i].material = SwitchEndian(etybuf[0]);
        element[i].type = SwitchEndian(etybuf[1]);
        element[i].real = SwitchEndian(etybuf[2]);
        element[i].section = SwitchEndian(etybuf[3]);
        element[i].coord = SwitchEndian(etybuf[4]);
        element[i].death = SwitchEndian(etybuf[5]);
        element[i].solidmodel = SwitchEndian(etybuf[6]);
        element[i].shape = SwitchEndian(etybuf[7]);
        element[i].num = SwitchEndian(etybuf[8]);
        for (j = 0; j < anzety; ++j)
        {
            if (ety[j].id == element[i].type)
                element[i].anznodes = ety[j].nodes;
        }
        element[i].nodes = new int[element[i].anznodes];
        for (j = 0; j < element[i].anznodes; ++j)
            element[i].nodes[j] = SwitchEndian(etybuf[10 + j]);
    }

    delete buf;
    return (0);
}

// -----------------------------------------------
// WriteData
// -----------------------------------------------
// Return:
// 0        : alles OK
// 1        : schreibfehler file
int READRFL::WriteData(char *filename)
{
    const char *names[7] = { "TRI", "QUA", "HEX", "PYR", "TET", "PRI", "ERR" };
    FILE *fp;
    DOFROOT *act = dofroot;
    int i, j, k, numdofs, geo, runs;
    int connect = 0;

    if ((fp = fopen(filename, "wt")) == NULL)
        return (1);

    fprintf(fp, "! nodes = %d\n", anznodes);
    fprintf(fp, "! elements = %d\n", anzelem);
    fprintf(fp, "! number of results = %d\n", rstheader.numsets);

    act = dofroot;
    numdofs = 0;
    while (act != NULL)
    {
        numdofs++;
        act = act->next;
    }
    fprintf(fp, "! DOFs = %d\n", numdofs);

    act = dofroot;
    while (act != NULL)
    {
        fprintf(fp, "! NAME = %s\n", act->name);
        act = act->next;
    }

    fprintf(fp, "# Koordinaten: id x y z + DOFs\n");
    for (i = 0; i < anznodes; ++i)
    {
        fprintf(fp, "%6d %+e %+e %+e ", (int)(node[i].id), node[i].x, node[i].y, node[i].z);
        act = dofroot;
        while (act != NULL)
        {
            fprintf(fp, "%+e ", act->data[i]);
            act = act->next;
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "# Geometriedaten/Elemente\n");
    //  fprintf(fp,"# id, typ, anz.knoten, knotenliste\n");

    // geordnet nach Elementtypen

    //  const char *names[6]={"TRI","QUA","HEX","PYR","TET","PRI"};

    for (runs = 0; runs < 3; ++runs)
    {
        switch (runs)
        {
        case 0:
            fprintf(fp, "# TYPENLISTE\n");
            break;

        case 1:
            fprintf(fp, "# ELEMENTLISTE\n");
            connect = 0;
            break;

        case 2:
            fprintf(fp, "# Connectivity (Nummerierung gegen den Uhrzeigersinn!)\n");
            fprintf(fp, "! %d # Anzahl der verwendeten Punkte (numConn)\n", connect);
            break;
        }
        for (j = 0; j < anzety; ++j)
        {
            for (i = 0; i < anzelem; ++i)
            {
                if (element[i].type == ety[j].id) // nach Art sortieren
                {

                    switch (runs)
                    {
                    case 0: // Elementtypen raus schreiben
                        // Typ Erkennung
                        switch (element[i].anznodes)
                        {
                        case 4: // 2D
                            if (element[i].nodes[3] == element[i].nodes[2])
                            {
                                // Dreieck
                                geo = 0;
                                element[i].anznodes = 3;
                                // Nodes selber ist ok, die hinteren beiden sind ja doppelt
                            }
                            else
                            {
                                // viereck
                                geo = 1;
                            }
                            break;

                        case 8: // 3d Fluid
                            if (element[i].nodes[4] != element[i].nodes[5])
                            {
                                if (element[i].nodes[6] == element[i].nodes[7])
                                {
                                    geo = 5; // Prism
                                    element[i].anznodes = 6;
                                    // lösche knoten 3, rest eins vor
                                    element[i].nodes[3] = element[i].nodes[4];
                                    element[i].nodes[4] = element[i].nodes[5];
                                    element[i].nodes[5] = element[i].nodes[6];
                                }
                                else
                                {
                                    geo = 2; // Hex
                                    // alles ok
                                }
                            }
                            else
                            {
                                if (element[i].nodes[2] == element[i].nodes[3])
                                {
                                    geo = 4; // Tetraeder
                                    element[i].anznodes = 4;
                                    element[i].nodes[3] = element[i].nodes[4];
                                }
                                else
                                {
                                    geo = 3; // Pyramid
                                    element[i].anznodes = 5;
                                    // Knoten sind OK
                                }
                            }
                            break;

                        default: // FIXME: macht Fehler bei Mittelknoten
                            geo = 6; //ERROR
                            break;
                        }
                        // Typ raus schreiben
                        fprintf(fp, "%s\n", names[geo]);

                        // Nebenjob: Knoten zählen
                        connect += element[i].anznodes;
                        break;

                    case 1: // Elementliste
                        fprintf(fp, "%d\n", connect);
                        connect += element[i].anznodes;
                        break;

                    case 2: // Connectivity
                        fprintf(fp, "%d ", element[i].anznodes);
                        for (k = 0; k < element[i].anznodes; ++k)
                            fprintf(fp, "%d ", element[i].nodes[k]);
                        fprintf(fp, "\n");
                        break;
                    }
                }
            }
        }
    }

    fprintf(fp, "#EOF\n");
    fclose(fp);
    return (0);
}

// -----------------------------------------------
// GetTime
// -----------------------------------------------
// Return:
// <0   Fehler und zwar:
// -1.0     : Index out of Range
// -2.0     : Keine Zeittafel vorhanden
double READRFL::GetTime(int pos)
{
    if (pos > rstheader.maxres) // index ausserhalb
        return (-1.0);

    if (timetable == NULL) // keine Zeittabelle gelesen
        return (-2.0);

    return (timetable[pos]); // Zeitwert zurück (in Sekunden)
}

// =============================================================================
// EOF
// =============================================================================
