/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/ApplInterface.h>
#include "ReadLat.h"

#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>

// Prototypes

bool readlat(char *Dateiname); // reads ascii latice-file
// returns TRUE if successful
void ErrorDisplay(const char *String, int Token, int SubToken);
void ProcessAt(FILE *FileDescr, _cxData *cxData, _cxCoord *cxCoord, int Index);
void ProcessCoord(FILE *FileDescr, _cxCoord *cxCoord);
void ProcessData(FILE *FileDescr, _cxData *cxData);
void ProcessLattice(FILE *FileDescr, _lattice *lattice);
int Decode(char *String, int *Value, int Lng);
void GetToken(FILE *FileDescr, int *Token, int *SubToken);
void GetBlock(int Index, FILE *FileDescr);
void ErrorPrint(const char *String, int Code);

// Globale Variablen

int Level = 0;
_PtrList PtrList[PTRLISTSIZ];
int Sizes[5] = { sizeof(char), sizeof(short), sizeof(int), sizeof(float), sizeof(double) };
char Item[256];
int ItemLng = 0;
int ClosePar = 0;

int main(int argc, char *argv[])
{

    Application *C_ReadLat = new Application(argc, argv);

    C_ReadLat->run();

    return 0;
    //  readlat("/awssg6/s/zr/0390/zrgw/test.lat");
}

void Application::compute(void *, void *)
{
    char *Dateiname;

    //int i;
    //const int no_out = 2;
    //char *Out_ObjName[2];
    //char *Out_PortName[2] = {"Mesh","Scalar"};

    // parse the START information
    //	get output data objects	name
    //for (i=0; i<no_out; i++)
    //{
    //  Out_ObjName[i]  = Covise::get_object_name(Out_PortName[i]);
    //}

    //	get string parameter (Filename)
    Covise::get_browser_param("Filename", &Dateiname);

    // start the calcaulation
    if (!readlat(Dateiname))
    {
        Covise::send_stop_pipeline();
    }
}

bool readlat(char *Dateiname)
{
    int LatticeType;
    int Token, SubToken;
    _lattice lattice;
    _cxData cxData;
    _cxCoord cxCoord;
    //float bBox[6];
    int i, Leave = 0;
    float MinExtent[10];
    //float MaxExtent[10];
    float *FPtr, *x_coord, *y_coord, *z_coord;
    int *IPtr;
    char Line[4096];
    //char Label[30];
    char Meldung[256], *tmpstr;
    char *Mesh, *Daten; // Out Object Names
    FILE *FileDescr = NULL; // yn : error msg. bugfix
    coDoFloat *daten = NULL;
    coDoStructuredGrid *gridstruct = NULL;
    coDoUniformGrid *griduni = NULL;
    coDoRectilinearGrid *gridrect = NULL;
    Level = 0;
    Item[0] = '\0';
    ItemLng = 0;
    ClosePar = 0;

    /********************************************************************
    * stat knows nothing about covise paths...
    *
    if( stat(Dateiname,&buf)<0 || !S_ISREG(buf.st_mode))
    {
    sprintf (Meldung,"The data file %s doesn't exists !!\n",Dateiname);
    Covise::sendError(Meldung);
    return false;
    }
    *
    ********************************************************************/
    FileDescr = Covise::fopen(Dateiname, "r");
    if (FileDescr == NULL)
    {
        sprintf(Meldung, "Fehler beim Oeffnen der Datei %s\n", Dateiname);
        print_comment(__LINE__, __FILE__, "%s", Meldung);
        Covise::sendError("%s", Meldung);
        return false;
    }

    if (fgets(Line, sizeof(Line), FileDescr) == NULL)
        printf("fgets_1 failed in ReadLat.cpp");
    if ((tmpstr = strstr(Line, "cxLattice")) == 0)
    {
        sprintf(Meldung, "%s ist keine Lattice Datei\n", Dateiname);
        print_comment(__LINE__, __FILE__, "%s", Meldung);
        Covise::sendError("%s", Meldung);
        return false;
    }
    LatticeType = 0;
    if (!strncasecmp(tmpstr + 10, "ascii", 5))
        LatticeType = ASCII;
    if (!strncasecmp(tmpstr + 10, "binary", 6))
        LatticeType = BIN;
    if (!LatticeType)
    {
        sprintf(Meldung, "%s: falscher Lattice-Typ\n", Dateiname);
        print_comment(__LINE__, __FILE__, "%s", Meldung);
        Covise::sendError("%s", Meldung);
        return false;
    }

    for (i = 0; i < PTRLISTSIZ; i++)
    {
        PtrList[i].Size = -1;
        PtrList[i].Type = -1;
        PtrList[i].Ptr = (char *)0;
    }

    if (LatticeType == ASCII)
    {
        while (!Leave)
        {
            GetToken(FileDescr, &Token, &SubToken);
            switch (Token)
            {
            case SKIP:
                break;
            case LATTICE:
                ProcessLattice(FileDescr, &lattice);
                break;
            case CXDATA:
                ProcessData(FileDescr, &cxData);
                break;
            case CXCOORD:
                ProcessCoord(FileDescr, &cxCoord);
                break;
            case AT:
                ProcessAt(FileDescr, &cxData, &cxCoord, SubToken);
                break;
            case EOF:
                Leave = 1;
                break;
            default:
                ErrorDisplay((char *)"Unknown Block Type", Token, SubToken);
                break;
            }
        }
    }
    else // Binary Lattice-File
    {
        if (fread(&lattice.nDim, sizeof(int), 1, FileDescr) <= sizeof(int))
            printf("ffgets_1 failed in ReadLat.cpp");
        if (fread(&lattice.dims, sizeof(int), 1, FileDescr) <= sizeof(int))
            printf("ffgets_2 failed in ReadLat.cpp");
        if (fread(&lattice.data, sizeof(int), 1, FileDescr) <= sizeof(int))
            printf("ffgets_3 failed in ReadLat.cpp");
        if (fread(&lattice.coord, sizeof(int), 1, FileDescr) <= sizeof(int))
            printf("ffgets_4 failed in ReadLat.cpp");
        PtrList[lattice.dims].Type = LONG;
        GetBlock(lattice.dims, FileDescr);
        if (fread(&cxData.nDim, sizeof(int), 1, FileDescr) <= sizeof(int))
            printf("ffgets_5 failed in ReadLat.cpp");
        if (fread(&cxData.dims, sizeof(int), 1, FileDescr) <= sizeof(int))
            printf("ffgets_6 failed in ReadLat.cpp");
        if (fread(&cxData.nDataVar, sizeof(int), 1, FileDescr) <= sizeof(int))
            printf("ffgets_7 failed in ReadLat.cpp");
        if (fread(&cxData.primType, sizeof(int), 1, FileDescr) <= sizeof(int))
            printf("ffgets_8 failed in ReadLat.cpp");
        /**** Just skip 4 bytes ****/
        if (fread(&MinExtent[0], sizeof(int), 1, FileDescr) <= sizeof(int))
            printf("ffgets_9 failed in ReadLat.cpp");

        if (fread(&cxData.data, sizeof(int), 1, FileDescr) <= sizeof(int))
            printf("ffgets_10 failed in ReadLat.cpp");
        if (fread(&cxCoord.nDim, sizeof(int), 1, FileDescr) <= sizeof(int))
            printf("ffgets_11 failed in ReadLat.cpp");
        if (fread(&cxCoord.dims, sizeof(int), 1, FileDescr) <= sizeof(int))
            printf("ffgets_12 failed in ReadLat.cpp");
        if (fread(&cxCoord.coordType, sizeof(int), 1, FileDescr) <= sizeof(int))
            printf("ffgets_13 failed in ReadLat.cpp");
        /**** Just skip 4 bytes ****/
        if (fread(&MinExtent[0], sizeof(int), 1, FileDescr) <= sizeof(int))
            printf("ffgets_14 failed in ReadLat.cpp");
        switch (cxCoord.coordType)
        {
        case 0:
            cxCoord.perimCoord = -1;
            cxCoord.values = -1;
            if (fread(&cxCoord.bBox, sizeof(int), 1, FileDescr) <= sizeof(int))
                printf("ffgets_15 failed in ReadLat.cpp");
            break;
        case 1:
            cxCoord.bBox = -1;
            cxCoord.values = -1;
            if (fread(&cxCoord.sumCoord, sizeof(int), 1, FileDescr) <= sizeof(int))
                printf("ffgets_16 failed in ReadLat.cpp");
            if (fread(&cxCoord.perimCoord, sizeof(int), 1, FileDescr) <= sizeof(int))
                printf("ffgets_17 failed in ReadLat.cpp");
            break;
        case 2:
            cxCoord.bBox = -1;
            cxCoord.perimCoord = -1;
            if (fread(&cxCoord.nCoordVar, sizeof(int), 1, FileDescr) <= sizeof(int))
                printf("ffgets_18 failed in ReadLat.cpp");
            if (fread(&cxCoord.values, sizeof(int), 1, FileDescr) <= sizeof(int))
                printf("ffgets_19 failed in ReadLat.cpp");
            break;
        default:
            print_comment(__LINE__, __FILE__, "Falsche Koordinaten");
            break;
        }

        PtrList[cxData.dims].Type = LONG;
        GetBlock(cxData.dims, FileDescr);

        PtrList[cxData.data].Type = cxData.primType;
        GetBlock(cxData.data, FileDescr);

        PtrList[cxCoord.dims].Type = LONG;
        GetBlock(cxCoord.dims, FileDescr);

        if (cxCoord.bBox != -1)
        {
            PtrList[cxCoord.bBox].Type = FLOAT;
            GetBlock(cxCoord.bBox, FileDescr);
        }

        if (cxCoord.perimCoord != -1)
        {
            PtrList[cxCoord.perimCoord].Type = FLOAT;
            GetBlock(cxCoord.perimCoord, FileDescr);
        }

        if (cxCoord.values != -1)
        {
            PtrList[cxCoord.values].Type = FLOAT;
            GetBlock(cxCoord.values, FileDescr);
        }
    }

    fclose(FileDescr);

    // Rausschreiben der Daten

    Mesh = Covise::get_object_name("Gitter");
    Daten = Covise::get_object_name("Daten");

    switch (cxCoord.coordType)
    {
    case UNIFORM:
        IPtr = (int *)(PtrList[cxCoord.dims].Ptr);
        FPtr = (float *)(PtrList[cxCoord.bBox].Ptr);
        griduni = new coDoUniformGrid(Mesh, IPtr[0], IPtr[1], IPtr[2],
                                      FPtr[0], FPtr[1], FPtr[2], FPtr[3], FPtr[4], FPtr[5]);
        break;
    case RECTILINEAR:
        IPtr = (int *)(PtrList[cxCoord.dims].Ptr);
        FPtr = (float *)(PtrList[cxCoord.perimCoord].Ptr);
        gridrect = new coDoRectilinearGrid(Mesh, IPtr[0], IPtr[1], IPtr[2], FPtr, FPtr + IPtr[0], FPtr + IPtr[0] + IPtr[1]);
        break;
    case IRREGULAR:
        IPtr = (int *)(PtrList[cxCoord.dims].Ptr);
        FPtr = (float *)(PtrList[cxCoord.values].Ptr);
        gridstruct = new coDoStructuredGrid(Mesh, IPtr[0], IPtr[1], IPtr[2]);
        gridstruct->getAddresses(&x_coord, &y_coord, &z_coord);
        for (i = 0; i < IPtr[0] * IPtr[1] * IPtr[2]; i++)
        {
            *(x_coord++) = FPtr[i * 3];
            *(y_coord++) = FPtr[(i * 3) + 1];
            *(z_coord++) = FPtr[(i * 3) + 2];
        }
        break;
    default:
        ErrorPrint((char *)"Unknown Coord Type", -1);
        break;
    }

    IPtr = (int *)(PtrList[cxData.dims].Ptr);
    daten = new coDoFloat(Daten, IPtr[0] * IPtr[1] * IPtr[2], (float *)PtrList[cxData.data].Ptr);
    // Speicher freigeben

    for (i = 0; i < PTRLISTSIZ; i++)
    {
        if (PtrList[i].Ptr)
            free(PtrList[i].Ptr);
    }

    delete daten;
    delete griduni;
    delete gridrect;
    delete gridstruct;

    return true;
}

void GetBlock(int Index, FILE *FileDescr)
{
    if (fread(&PtrList[Index].Size, 1, sizeof(int), FileDescr) <= sizeof(int))
        printf("ffgets_20 failed in ReadLat.cpp");
    PtrList[Index].Ptr = (char *)malloc((int)(PtrList[Index].Size * Sizes[PtrList[Index].Type]));
    if (fread(PtrList[Index].Ptr, (int)(PtrList[Index].Size), (int)(Sizes[PtrList[Index].Type]), FileDescr) <= sizeof(int))
        printf("ffgets_21 failed in ReadLat.cpp");
#ifdef DEBUG
    printf("Get Binary @%d block : Size : %d ; Type : %d ; Elem size : %d\n", Index,
           PtrList[Index].Size, PtrList[Index].Type, Sizes[PtrList[Index].Type]);
#endif
}

void ProcessLattice(FILE *FileDescr, _lattice *lattice)
{
    //char Msg[1024];
    int InLev = Level;
    int Token, SubToken;
    while (Level >= InLev)
    {
        GetToken(FileDescr, &Token, &SubToken);
        switch (Token)
        {
        case SKIP:
            break;
        case NDIM:
            GetToken(FileDescr, &Token, &SubToken);
            lattice->nDim = SubToken;
            break;
        case DIMS:
            GetToken(FileDescr, &Token, &SubToken);
            lattice->dims = SubToken;
            PtrList[SubToken].Type = LONG;
            break;
        case DATA:
            GetToken(FileDescr, &Token, &SubToken);
            lattice->data = SubToken;
            break;
        case COORD:
            GetToken(FileDescr, &Token, &SubToken);
            lattice->coord = SubToken;
            break;
        default:
            ErrorDisplay((char *)"ProcessLat", Token, SubToken);
            break;
        }
    }
#ifdef DEBUG
    printf("Done ProcessLat %d %d\n", lattice->nDim, lattice->data);
#endif
}

void ProcessData(FILE *FileDescr, _cxData *cxData)
{

    int InLev2;
    int InLev = Level;
    int Token, SubToken;
    int Index;
    while (Level >= InLev)
    {
        GetToken(FileDescr, &Token, &SubToken);
        switch (Token)
        {
        case SKIP:
            break;
        case NDIM:
            GetToken(FileDescr, &Token, &SubToken);
            cxData->nDim = SubToken;
            break;
        case DIMS:
            GetToken(FileDescr, &Token, &SubToken);
            cxData->dims = SubToken;
            ErrorPrint((char *)"Dims not in @list", Token != AT);
            PtrList[SubToken].Type = LONG;
            PtrList[SubToken].Size = sizeof(int);
            break;
        case NDATAVAR:
            GetToken(FileDescr, &Token, &SubToken);
            cxData->nDataVar = SubToken;
            break;
        case PRIMTYPE:
            InLev2 = Level;
            while (Level >= InLev2)
            {
                GetToken(FileDescr, &Token, &SubToken);
                switch (Token)
                {
                case SKIP:
                    break;
                case NUMBER:
                    cxData->primType = SubToken;
                    break;
                }
            }
            break;
        case D:
            InLev2 = Level;
            while (Level >= InLev2)
            {
                GetToken(FileDescr, &Token, &SubToken);
                switch (Token)
                {
                case SKIP:
                    break;
                case VALUES:
                    GetToken(FileDescr, &Token, &SubToken);
                    Index = SubToken;
                    cxData->data = Index;
                    PtrList[Index].Type = cxData->primType;
                    break;
                }
            }
            break;
        default:
            ErrorDisplay((char *)"cxData", Token, SubToken);
            break;
        }
    }
#ifdef DEBUG
    printf("Done ProcessData %d %d %d\n", cxData->nDim, cxData->dims, cxData->nDataVar);
#endif
}

void ProcessCoord(FILE *FileDescr, _cxCoord *cxCoord)
{

    int InLev = Level;
    int InLev2;
    int Token, SubToken;
    int Index;
    while (Level >= InLev)
    {
        GetToken(FileDescr, &Token, &SubToken);
        switch (Token)
        {
        case SKIP:
            break;
        case NDIM:
            GetToken(FileDescr, &Token, &SubToken);
            cxCoord->nDim = SubToken;
            break;
        case DIMS:
            GetToken(FileDescr, &Token, &SubToken);
            cxCoord->dims = SubToken;
            ErrorPrint((char *)"Dims not in @list", Token != AT);
            PtrList[SubToken].Type = LONG;
            break;
        case COORDTYPE:
            InLev2 = Level;
            while (Level >= InLev2)
            {
                GetToken(FileDescr, &Token, &SubToken);
                switch (Token)
                {
                case SKIP:
                    break;
                case NUMBER:
                    cxCoord->coordType = SubToken;
                    break;
                }
            }
            break;
        case C:
            InLev2 = Level;
            while (Level >= InLev2)
            {
                GetToken(FileDescr, &Token, &SubToken);
                switch (Token)
                {
                case SKIP:
                    break;
                case SUMCOORD:
                    GetToken(FileDescr, &Token, &SubToken);
                    cxCoord->sumCoord = SubToken;
                    break;
                case PERIMCOORD:
                    GetToken(FileDescr, &Token, &SubToken);
                    cxCoord->perimCoord = SubToken;
                    PtrList[cxCoord->perimCoord].Type = FLOAT;
                    break;
                case BBOX:
                    GetToken(FileDescr, &Token, &SubToken);
                    cxCoord->bBox = SubToken;
                    PtrList[cxCoord->bBox].Type = FLOAT;
                    break;
                case VALUES:
                    GetToken(FileDescr, &Token, &SubToken);
                    Index = SubToken;
                    cxCoord->values = Index;
                    PtrList[Index].Type = FLOAT;
                    break;
                case NCOORDVAR:
                    GetToken(FileDescr, &Token, &SubToken);
                    cxCoord->nCoordVar = SubToken;
                    break;
                }
            }
            break;
        default:
            ErrorDisplay((char *)"cxCoord", Token, SubToken);
            break;
        }
    }
}

void GetToken(FILE *FileDescr, int *Token, int *SubToken)
{
    char Char;
    int I_Char;
    int T;

    if (ClosePar)
    {
        *Token = SKIP;
        ClosePar = 0;
        return;
    }

    ItemLng = 0;
    *Token = READ_LAT_ERROR;
    while (((I_Char = getc(FileDescr)) != EOF) && !feof(FileDescr))
    {
        Char = I_Char;
        switch (Char)
        {
        case ']':
            break;
        case ')':
            Level--;
            if (!ItemLng)
            {
                *Token = SKIP;
                return;
            }
            ClosePar = 1;
        case ' ':
            if (!ItemLng)
                break;
            switch (*Token)
            {
            case AT:
            case BRACKET:
                T = Decode(Item, SubToken, ItemLng);
                ItemLng = 0;
                if (T != EQUAL)
                    return;
                break;
            default:
                T = Decode(Item, Token, ItemLng);
                if (Char == ')')
                    *SubToken = -1;
                else
                    *SubToken = 0;
#ifdef DEBUG
                printf("Decoded %d %d %d\n", T, *Token, *SubToken);
#endif
                if (T == NUMBER)
                {
                    *SubToken = *Token;
                    *Token = T;
                }
                ItemLng = 0;
                if (T != EQUAL)
                    return;
                break;
            }
        case '\t':
            break;
        case '\n':
            break;
        case '\f':
            break;
        case '{':
            break;
        case '}':
            break;
        case '(':
            Level++;
            *Token = SKIP;
            return;
        case '@':
            *Token = AT;
            break;
        case '[':
            *Token = BRACKET;
            break;
        default:
            Item[ItemLng++] = Char;
            break;
        }
    }
    *Token = EOF;
}

int Decode(char *String, int *Value, int Lng)
{
    *Value = READ_LAT_ERROR;
    String[Lng] = '\0';
    if (!strncmp(String, "=", 1))
        *Value = EQUAL;
    if (!strncmp(String, "=cxData", Lng))
        *Value = CXDATA;
    if (!strncmp(String, "=cxCoord", Lng))
        *Value = CXCOORD;
    if (!strncmp(String, "cxLattice", Lng))
        *Value = LATTICE;
    if (!strncmp(String, "nDim", Lng))
        *Value = NDIM;
    if (!strncmp(String, "dims", Lng))
        *Value = DIMS;
    if (!strncmp(String, "data", Lng))
        *Value = DATA;
    if (!strncmp(String, "coord", Lng))
        *Value = COORD;
    if (!strncmp(String, "nDataVar", Lng))
        *Value = NDATAVAR;
    if (!strncmp(String, "primType", Lng))
        *Value = PRIMTYPE;
    if (!strncmp(String, "coordType", Lng))
        *Value = COORDTYPE;
    if (!strncmp(String, "values", Lng))
        *Value = VALUES;
    if (!strncmp(String, "bBox", Lng))
        *Value = BBOX;
    if (!strncmp(String, "sumCoord", Lng))
        *Value = SUMCOORD;
    if (!strncmp(String, "perimCoord", Lng))
        *Value = PERIMCOORD;
    if (!strncmp(String, "nCoordVar", Lng))
        *Value = NCOORDVAR;
    if (!strncmp(String, "d", Lng))
        *Value = D;
    if (!strncmp(String, "c", Lng))
        *Value = C;
    if (*Value == READ_LAT_ERROR)
    {
        if (sscanf(String, "%d", Value) != 1)
        {
            if (!strncmp(String, "cx_prim_byte", Lng))
                *Value = CHAR;
            if (!strncmp(String, "cx_prim_short", Lng))
                *Value = SHORT;
            if (!strncmp(String, "cx_prim_int", Lng))
                *Value = LONG;
            if (!strncmp(String, "cx_prim_float", Lng))
                *Value = FLOAT;
            if (!strncmp(String, "cx_prim_double", Lng))
                *Value = DOUBLE;
            if (!strncmp(String, "cx_coord_uniform", Lng))
                *Value = 0;
            if (!strncmp(String, "cx_coord_perimeter", Lng))
                *Value = 1;
            if (!strncmp(String, "cx_coord_curvilinear", Lng))
                *Value = 2;
            if (*Value == READ_LAT_ERROR)
                ErrorPrint((char *)"Not A Number", -1);
        }
#ifdef DEBUG
        printf("Number Item %s; Value %d ; Lng %d\n", String, *Value, Lng);
#endif
        return (NUMBER);
    }
#ifdef DEBUG
    printf("Item %s; Value %d ; Lng %d\n", String, *Value, Lng);
#endif
    return (*Value);
}

void ProcessAt(FILE *FileDescr, _cxData *cxData, _cxCoord *cxCoord, int Index)
{
    int Token, SubToken;
    int i;
    int Val;
    unsigned char *CPtr;
    short *SPtr;
    int *LPtr;
    float *FPtr;
    double *DPtr;

    GetToken(FileDescr, &Token, &SubToken);
    switch (Token)
    {
    case CXDATA:
        ProcessData(FileDescr, cxData);
        return;
    case CXCOORD:
        ProcessCoord(FileDescr, cxCoord);
        return;
    case EOF:
        return;
    case BRACKET:
        ErrorPrint("Too many @ blocks", Index >= PTRLISTSIZ);
        PtrList[Index].Size = SubToken;
        ErrorPrint("@ block without description", PtrList[Index].Size <= 0);
        PtrList[Index].Ptr = (char *)malloc((int)(PtrList[Index].Size * Sizes[PtrList[Index].Type]));
        ErrorPrint("Malloc", PtrList[Index].Ptr == 0);
        switch (PtrList[Index].Type)
        {
        case CHAR:
            CPtr = (unsigned char *)PtrList[Index].Ptr;
            for (i = 0; i < SubToken; i++)
            {
                if (fscanf(FileDescr, "%d", &Val) != 1)
                {
                    fprintf(stderr, "fscanf_1 failed in ReadLat.cpp");
                }
                *CPtr++ = (unsigned char)Val;
            }
            break;
        case SHORT:
            SPtr = (short *)PtrList[Index].Ptr;
            for (i = 0; i < SubToken; i++)
                if (fscanf(FileDescr, "%hd", SPtr++) != 1)
                {
                    fprintf(stderr, "fscanf_2 failed in ReadLat.cpp");
                }
            break;
        case LONG:
            LPtr = (int *)PtrList[Index].Ptr;
            for (i = 0; i < SubToken; i++)
                if (fscanf(FileDescr, "%d", LPtr++) != 1)
                {
                    fprintf(stderr, "fscanf_3 failed in ReadLat.cpp");
                }
            break;
        case FLOAT:
            FPtr = (float *)PtrList[Index].Ptr;
            for (i = 0; i < SubToken; i++)
                if (fscanf(FileDescr, "%f", FPtr++) != 1)
                {
                    fprintf(stderr, "fscanf_4 failed in ReadLat.cpp");
                }
            break;
        case DOUBLE:
            DPtr = (double *)PtrList[Index].Ptr;
            for (i = 0; i < SubToken; i++)
                if (fscanf(FileDescr, "%lf", DPtr++) != 1)
                {
                    fprintf(stderr, "fscanf_5 failed in ReadLat.cpp");
                }
            break;
        default:
            ErrorPrint("Bad block type", -1);
            break;
        }
    }
#ifdef DEBUG
    printf("Done ProcessAt : %d %d %d %x\n", Index, PtrList[Index].Size, PtrList[Index].Type, PtrList[Index].Ptr);
#endif
}

void ErrorDisplay(const char *String, int Token, int SubToken)
{
    Item[ItemLng] = '\0';
    print_comment(__LINE__, __FILE__, "%s Err @ level %d: Item= %s,Token = %d,SubToken=%d", String, Level, Item, Token, SubToken);
}

void ErrorPrint(const char *String, int Code)
{
    if (!Code)
        return;
    print_comment(__LINE__, __FILE__, "Error : %s ; code is %d", String, Code);
}
