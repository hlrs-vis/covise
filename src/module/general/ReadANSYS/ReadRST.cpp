/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// =============================================================================
// READRFL Klasse zum lesen von ANSYS RFL-Ergebnisfiles (FLOWTRAN)
// -----------------------------------------------------------------------------
// 17.9.2001  BjÃ¶rn Sander
// 16.1.2002  Sergio Leseduarte
// =============================================================================

const char *dofname[32] = {
    "UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ",
    "AX", "AY", "AZ", "VX", "VY", "VZ",
    "unused1", "unused2", "unused3", "unused4", "unused5", "unused6",
    "PRES", "TEMP", "VOLT", "MAG", "ENKE", "ENDS", "EMF",
    "CURR", "SP01", "SP02", "SP03", "SP04", "SP05", "SP06"
};

const char *exdofname[28] = {
    "DENS", "VISC", "EVIS", "COND", "ECON", "LMD1", "LMD2", "LMD3",
    "LMD4", "LMD5", "LMD6", "EMD1", "EMD2", "EMD3", "EMD4", "EMD5",
    "EMD6", "PTOT", "TTOT", "PCOE", "MACH", "STRM", "HFLU", "HFLM",
    "YPLU", "TAUW", "SPHT", "CMUV"
};

#include "ReadRST.h"
#include "ANSYS.h"
#include "ReadANSYS.h"
#include <util/coviseCompat.h>

// =============================================================================
// Konstruktor / Destruktor
// =============================================================================

// -----------------------------------------------
// Konstruktor
// -----------------------------------------------

const double ReadRST::DImpossible_ = pow(2.0, 100.0);
const float ReadRST::FImpossible_ = (float)pow(2.0, 100.0);

ReadRST::ReadRST(void)
{
    DOFData_ = NULL;
    DerivedData_ = NULL;
    rfp_ = NULL;
    ety_ = NULL;
    node_ = NULL;
    element_ = NULL;
    anznodes_ = 0;
    anzety_ = 0;
    anzelem_ = 0;
    nodeindex_ = NULL;
    elemindex_ = NULL;
    timetable_ = NULL;
    SwitchEndian_ = DO_NOT_SWITCH;
    mmap_flag_ = 0;
    mode64_ = false;
}

// -----------------------------------------------
// Destruktor
// -----------------------------------------------
ReadRST::~ReadRST()
{
    Reset(RADIKAL);
}

// =============================================================================
// Interne Methoden
// =============================================================================

// -----------------------------------------------
// Reset
// -----------------------------------------------
int
ReadRST::Reset(int message)
{
    if (!(message & PRESERVE_DOF_DATA))
    {
        delete DOFData_;
        DOFData_ = NULL;
    }

    delete DerivedData_;
    DerivedData_ = NULL;

    if (message & RADIKAL)
    {
        delete DOFData_;
        DOFData_ = NULL;

        delete[] node_;
        node_ = NULL;
        anznodes_ = 0;

        delete[] ety_;
        ety_ = NULL;
        anzety_ = 0;

        delete[] element_;
        element_ = NULL;
        anzelem_ = 0;

        memset(&header_, 0, sizeof(BinHeader));
        memset(&rstheader_, 0, sizeof(RstHeader));

        if (rfp_ != NULL)
        {
            fclose(rfp_);
            rfp_ = NULL;
        }
        delete[] nodeindex_;
        nodeindex_ = NULL;
        delete[] elemindex_;
        elemindex_ = NULL;
        delete[] timetable_;
        timetable_ = NULL;

        solheader_.clean();
    }

    return (0);
}

// -----------------------------------------------
// OpenFile
// -----------------------------------------------
// Oeffnet den Ergebnisfile, liest den header und setzt
// den Read-File-Pointer rfp;
// Rueckgabe:
// 0    : alles OK
// 1    : File not found
// 2    : could not read header
// 3    : Read Error Nodal equivalence
// 4    : Read Error Element equivalence
// 5    : Read Error Time table
int
ReadRST::OpenFile(const std::string &filename)
{
    if (rfp_ && nodeindex_ && rstheader_.numsets_ != 0)
        return 0;

    if ((rfp_ = fopen(filename.c_str(), "rb")) == NULL)
    {
        return (1);
    }

    file_des_ = fileno(rfp_);

    if (get_file_size() != 0)
    {
        Covise::sendError("Could not get file size");
        return (1);
    }

    // File offen und OK
    // erst mal zwei INTs lesen (lead-in):
    // int 1 ergibt die Laenge des folgenden Records in bytes
    // int 2 ergibt den Typ des Records (z.B. 0x64==Integer)

    /****************************************************************************/
    /* Notes:                                                                                                           */
    /*                                                                                                                      */
    /* ANSYS binary files are written in Fortran which means that data is written   */
    /* blockwise.  Each block or record is delimited by markers, namely a header  */
    /* and trailer, both of them being identical and usually 4 bytes long.  These     */
    /* markers indicate the size of the record.  ANSYS uses particularly variable-  */
    /* length records, however it has been observed that the format of the          */
    /* binary files varies depending on the ANSYS version.                                     */
    /*                                                                                                                      */
    /* In older versions of ANSYS (i.e. V5.6) the size of the records is denoted      */
    /* in BYTES while the second integer after the record header corresponds to  */
    /* the size of the Standard ANSYS File Header which is 100 or 0x64 integers  */
    /* long.  This value is followed by the actual 100 values (integers) or items      */
    /* contained in the ANSYS File Header.                                                             */
    /*                                                                                                                     */
    /* In newer versions (i.e. V11.0) it has been observed that the size of the      */
    /* records is expressed in INTs, that is 0x64 items.  The second integer has  */
    /* the value 2147483648 or 0x80000000 which may be a reference to the    */
    /* data type of the elements contained in the ANSYS File Header, namely        */
    /* integers (signed int ?).  Nevertheless it seems this integer is not counted   */
    /* in the record size.  For more information refer to "Programmer's                */
    /* Manual for Mechanical APDL" (ANSYS Release 12.1)                                     */
    /***************************************************************************/

    int *buf = new int[1024];
    char version[150];

    // Reading the first 4 bytes of the binary file
    size_t iret = fread(buf, sizeof(int), 1, rfp_);

    if (iret != 1)
    {
        Covise::sendInfo("Error reading rtp_");
    }

    int header_offset = 0;
    if (buf[0] == 100 || buf[0] == 1677721600) // 100 and 1677721600 to recognize offset in little endian and big endian
    {
        header_offset = 1;
    }

    // Reading the header
    if (fread(buf + 1 + header_offset, sizeof(int), 102, rfp_) != 102)
    {
        return (2);
    }

    int subversion;
    version[4] = 0;
    memcpy(version, &buf[11 + header_offset], 4);
    SwitchEndian(version, 4);

    int ret = sscanf(version, "%d.%d", &header_.version_, &subversion);

    if (ret != 2)
    {
        header_.version_ = 9;
    }
    Covise::sendInfo("File generated by ANSYS V%d.%d", header_.version_, subversion);

    // alle INT-Elemente umdrehen

    header_.filenum_ = SwitchEndian(buf[2]);
    header_.format_ = SwitchEndian(buf[3]);
    header_.time_ = SwitchEndian(buf[4]);
    header_.date_ = SwitchEndian(buf[5]);
    header_.unit_ = SwitchEndian(buf[10]);
    header_.ansysdate_ = SwitchEndian(buf[12]);
    memcpy(header_.machine_, &buf[13], 3 * sizeof(int));
    header_.machine_[12] = '\0';
    SwitchEndian(header_.machine_, 12);
    memcpy(header_.jobname_, &buf[16], 2 * sizeof(int));
    header_.jobname_[8] = '\0';
    SwitchEndian(header_.jobname_, 8);
    memcpy(header_.product_, &buf[18], 2 * sizeof(int));
    header_.product_[8] = '\0';
    SwitchEndian(header_.product_, 8);
    memcpy(header_.label_, &buf[20], 1 * sizeof(int));
    header_.label_[4] = '\0';
    SwitchEndian(header_.label_, 4);
    memcpy(header_.user_, &buf[21], 3 * sizeof(int));
    header_.user_[12] = '\0';
    SwitchEndian(header_.user_, 12);
    memcpy(header_.machine2_, &buf[24], 3 * sizeof(int));
    header_.machine2_[12] = '\0';
    SwitchEndian(header_.machine2_, 12);
    header_.recordsize_ = SwitchEndian(buf[27]);
    header_.maxfilelen_ = SwitchEndian(buf[28]);
    header_.maxrecnum_ = SwitchEndian(buf[29]);
    header_.cpus_ = SwitchEndian(buf[30]);
    memcpy(header_.title_, &buf[42], 20 * sizeof(int));
    header_.title_[80] = '\0';
    SwitchEndian(header_.title_, 80);
    memcpy(header_.subtitle_, &buf[62], 20 * sizeof(int));
    header_.subtitle_[80] = '\0';
    SwitchEndian(header_.subtitle_, 80);
    // File Pointer steht jetzt auf Position (100+3)*4=412 Bytes

    Covise::sendInfo("Machine: %s   Jobname: %s", header_.machine_, header_.jobname_);
    Covise::sendInfo("Product: %s   Label: %s", header_.product_, header_.label_);
    Covise::sendInfo("Title: %s   Subtitle: %s", header_.title_, header_.subtitle_);

    // jetzt den RST-File Header reinbasteln
    if (fread(buf, sizeof(int), 43, rfp_) != 43)
    {
        return (2);
    }

    if (SwitchEndian(buf[2]) != 12)
    {
        ChangeSwitch();
        if (SwitchEndian(buf[2]) == 12)
        {
            // file can be read using byteswap, prepare to try again
            delete[] buf;
            fclose(rfp_);
            // try again
            Covise::sendInfo("File is byteswapped, trying again ...");
            return OpenFile(filename);
        }
        else
        {
            // file cannot be read at all
            Covise::sendError("Cannot correctly read file header length");
            return 2;
        }
    }

    // Werte zuweisen
    rstheader_.fun12_ = SwitchEndian(buf[2]);
    rstheader_.maxnodes_ = SwitchEndian(buf[3]);
    rstheader_.usednodes_ = SwitchEndian(buf[4]);
    rstheader_.maxres_ = SwitchEndian(buf[5]);
    rstheader_.numdofs_ = SwitchEndian(buf[6]);
    rstheader_.maxelement_ = SwitchEndian(buf[7]);
    rstheader_.numelement_ = SwitchEndian(buf[8]);
    rstheader_.analysis_ = SwitchEndian(buf[9]);
    rstheader_.numsets_ = SwitchEndian(buf[10]);
    rstheader_.ptr_eof_ = SwitchEndian(buf[11]);
    rstheader_.ptr_dsi_ = SwitchEndian(buf[12]);
    rstheader_.ptr_time_ = SwitchEndian(buf[13]);
    rstheader_.ptr_load_ = SwitchEndian(buf[14]);
    rstheader_.ptr_elm_ = SwitchEndian(buf[15]);
    rstheader_.ptr_node_ = SwitchEndian(buf[16]);
    rstheader_.ptr_geo_ = SwitchEndian(buf[17]);
    rstheader_.units_ = SwitchEndian(buf[21]);
    rstheader_.numsectors_ = SwitchEndian(buf[22]);
    rstheader_.ptr_end_ = 0;

    if (SwitchEndian_ == DO_NOT_SWITCH)
    {
        rstheader_.ptr_end_ = (long long)(buf[23]) << 32 | buf[24];
    }
    else
    {
        rstheader_.ptr_end_ = (long long)(SwitchEndian(buf[24])) << 32 | SwitchEndian(buf[23]);
    }
    delete[] buf;
    // Header fertig gelesen

    // Jetzt noch die Indextabellen laden
    int true_used_nodes;
    if (header_.version_ < 10)
    {
        int offset = rstheader_.ptr_node_ * sizeof(int);
        int buf[2];
        fseek(rfp_, offset, SEEK_SET);
        IntRecord(buf, 2);
        true_used_nodes = buf[1];
        if (true_used_nodes > rstheader_.usednodes_)
        {
            true_used_nodes = rstheader_.usednodes_;
        }
    }
    else
    {
        true_used_nodes = rstheader_.usednodes_;
    }

    int offset = (rstheader_.ptr_node_ + 2) * sizeof(int);
    fseek(rfp_, offset, SEEK_SET);

    nodeindex_ = new int[true_used_nodes /* rstheader_.usednodes_ */];
    // NEW
    // if(IntRecord(nodeindex_,rstheader_.usednodes_) != rstheader_.usednodes_){
    if (IntRecord(nodeindex_, true_used_nodes) != true_used_nodes)
    {
        return (3);
    }

    // das selbe fÃ¼r die Elemente
    offset = (rstheader_.ptr_elm_ + 2) * sizeof(int);
    fseek(rfp_, offset, SEEK_SET);
    elemindex_ = new int[rstheader_.numelement_];
    if (IntRecord(elemindex_, rstheader_.numelement_) != rstheader_.numelement_)
    {
        return (3);
    }

    // Timetable lesen
    timetable_ = new double[rstheader_.maxres_];
    offset = (rstheader_.ptr_time_ + 2) * sizeof(int);
    fseek(rfp_, offset, SEEK_SET);
    if (DoubleRecord(timetable_, rstheader_.maxres_)
        != rstheader_.maxres_)
    {
        return (6);
    }

    // That's all folks
    return (0);
}

int
ReadRST::IntRecord(int *buf, int len)
{
    int ret = len;
    if (mmap_flag_)
    {
        memcpy(buf, (char *)mmap_ini_ + actual_off_, len * sizeof(int));
    }
    else
    {
        ret = fread(buf, sizeof(int), len, rfp_);
        if (ret != len)
            return ret;
    }
    int item;
    for (item = 0; item < len; ++item)
    {
        buf[item] = SwitchEndian(buf[item]);
    }
    return ret;
}

int
ReadRST::DoubleRecord(double *buf, int len)
{
    int ret = len;
    if (mmap_flag_)
    {
        memcpy(buf, (char *)mmap_ini_ + actual_off_, len * sizeof(double));
    }
    else
    {
        ret = fread(buf, sizeof(double), len, rfp_);
        if (ret != len)
            return ret;
    }
    int item;
    for (item = 0; item < len; ++item)
    {
        buf[item] = SwitchEndian(buf[item]);
    }
    return ret;
}

// -----------------------------------------------
// SwitchEndian
// -----------------------------------------------
int
ReadRST::SwitchEndian(int value)
{
    if (SwitchEndian_ == SWITCH)
    {
        int ret = 0;
        int bytes[4];

        bytes[0] = (value & 0x000000FF) << 24;
        bytes[1] = (value & 0x0000FF00) << 8;
        bytes[2] = (value & 0x00FF0000) >> 8;
        bytes[3] = (value & 0xFF000000) >> 24;

        ret = bytes[0] | bytes[1] | bytes[2] | bytes[3];
        return (ret);
    }
    return (value);
}

unsigned int
ReadRST::SwitchEndian(unsigned int value)
{
    if (SwitchEndian_ == SWITCH)
    {
        int ret = 0;
        int bytes[4];

        bytes[0] = (value & 0x000000FF) << 24;
        bytes[1] = (value & 0x0000FF00) << 8;
        bytes[2] = (value & 0x00FF0000) >> 8;
        bytes[3] = (value & 0xFF000000) >> 24;

        ret = bytes[0] | bytes[1] | bytes[2] | bytes[3];
        return (ret);
    }
    return (value);
}

void
ReadRST::SwitchEndian(char *buf, int length)
{
    if (SwitchEndian_ == DO_NOT_SWITCH)
    {
        int index(0);
        while (index < length)
        {
            char tmp1 = buf[index];
            char tmp2 = buf[index + 1];
            buf[index] = buf[index + 3];
            buf[index + 1] = buf[index + 2];
            buf[index + 2] = tmp2;
            buf[index + 3] = tmp1;
            index += 4;
        }
    }
}

void
ReadRST::ChangeSwitch()
{
    if (SwitchEndian_ == SWITCH)
    {
        SwitchEndian_ = DO_NOT_SWITCH;
    }
    else
    {
        SwitchEndian_ = SWITCH;
    }
}

double
ReadRST::SwitchEndian(double dval)
{
    if (SwitchEndian_ == SWITCH)
        byteSwap(dval);
    return dval;
}

// =============================================================================
// externe Methoden
// =============================================================================

// -----------------------------------------------
// Read
// -----------------------------------------------
int
ReadRST::Read(const std::string &filename, int num, std::vector<int> &codes)
{
    Reset(0); // alles lÃ¶schen, wenn nÃ¶tig
    // Mal keine Fehlerabfrage, FIXME!
    int problems = OpenFile(filename);
    switch (problems)
    {
    // 0    : alles OK
    // 1    : File not found
    // 2    : could not read header
    // 3    : Read Error Nodal equivalence
    // 4    : Read Error Element equivalence
    // 5    : Read Error Time table
    case 0:
        //      printf("Open file OK\n");
        break;

    case 1:
        Covise::sendError("Open file: file not found \n");
        break;

    case 2:
        Covise::sendError("Open file: Read Error, header \n");
        break;

    case 3:
        Covise::sendError("Open file: Read Error, nodal equ tabular \n");
        break;

    case 4:
        Covise::sendError("Open file: Read Error, element equi tabular \n");
        break;

    case 5:
        Covise::sendError("Open file: Read Error, time table \n");
        break;
    }
    if (problems)
        return problems;
    problems = GetDataset(num, codes);
    switch (problems)
    {
    // 1        : File ist nicht offen/initialisiert
    // 2        : Read Error DSI-Tabelle
    // 3        : Num ist nicht im Datensatz
    // 4        : Read Error Solution Header
    // 5        : Read Error DOFs
    // 6        : Read Error exDOFs
    case 0:
        //      printf("GetData : OK\n");
        break;

    case 1:
        Covise::sendError("GetData : file not open\n");
        break;

    case 2:
        Covise::sendError("GetData : read error: DSI\n");
        break;

    case 3:
        Covise::sendError("GetData : num exeeds limits\n");
        break;

    case 4:
        Covise::sendError("GetData : read error solution header\n");
        break;

    case 5:
        Covise::sendError("GetData : read error DOFs\n");
        break;

    case 6:
        Covise::sendError("GetData : read error exDOF\n");
        break;
    }
    if (problems)
        return problems;
    problems = GetNodes();
    switch (problems)
    {
    // 1        : Read Error Geometrieheader
    // 2        : Read Error Nodes
    // 3        : Read Error Elementbeschreibung
    // 4        : Read Error ETYs
    // 5        : Read Error Elementtabelle
    // 6        : Read Error Elemente
    case 0:
        //      printf("GetNodes : ok\n");
        break;

    case 1:
        Covise::sendError("GetNodes : read error geo\n");
        break;

    case 2:
        Covise::sendError("GetNodes : read error nodes\n");
        break;

    case 3:
        Covise::sendError("GetNodes : read error element description\n");
        break;

    case 4:
        Covise::sendError("GetNodes : read error ety\n");
        break;

    case 5:
        Covise::sendError("GetNodes : read error element tabular\n");
        break;

    case 6:
        Covise::sendError("GetNodes : read error elements\n");
        break;
    }
    //  fclose(rfp);
    //  rfp=NULL;
    return (problems);
}

// -----------------------------------------------
// GetDataset
// -----------------------------------------------
// Liest die Ergebnisdaten fuer einen Datensatz aus dem File
// Rueckgabe:
// 0        : alles OK
// 1        : File ist nicht offen/initialisiert
// 2        : Read Error DSI-Tabelle
// 3        : Num ist nicht im Datensatz
// 4        : Read Error Solution Header
// 5        : Read Error DOFs
// 6        : Read Error exDOFs
int
ReadRST::GetDataset(int num, std::vector<int> &codes)
{
    //  int *buf=NULL;
    int size, i, j, sumdof;
    long long offset;
    double *dof;

    // File sollte offen sein
    if (rfp_ == NULL)
        return (1);

    // Out of range check
    if (num > rstheader_.numsets_ /*rstheader_.maxres_*/)
        return (3);

    // Eventuell alte Daten LÃ¶schen
    // DOF-Liste loeschen
    delete DOFData_;
    DOFData_ = NULL;
    /* FIXME ??????????????????? !!!!!!!!!!!!!!
     delete [] node_;
     node_ = NULL;
     anznodes_=0;
     delete [] ety_;
     ety_ = NULL;
     anzety_=0;
     delete [] element_;
     element_ = NULL;
     anzelem_=0;
   */
    // so, alles gelÃ¶scht
    ReadSHDR(num);

    // jetzt Daten einlesen
    // but first read record length...

    if (solheader_.ptr_nodalsol_ == 0) // no DOF data at all
    {
        solheader_.numnodesdata_ = 0;
        DOFData_ = new DOFData; //CreateNewDOFList();
        DOFData_->anz_ = 0;
        DOFData_->nodesdataanz_ = 0;
        //    DOFData_->data_ = new double[0];
        //    DOFData_->nodesdata_ = new int[0];
        return 0;
    }

    offset = solheader_.offset_ + solheader_.ptr_nodalsol_ * 4;
    fseek(rfp_, (long)offset, SEEK_SET);
    int front[2];
    if (IntRecord(front, 2) != 2)
    {
        return (5);
    }
    sumdof = solheader_.numdofs_ + solheader_.numexdofs_;
    if (header_.version_ < 10)
    {
        solheader_.numnodesdata_ = (front[0] - 4) / (sizeof(double) * sumdof);
    }
    else
    {
        solheader_.numnodesdata_ = solheader_.numnodes_;
    }

    /*
   cerr << "SHD numnodes "<<solheader_.numnodes_ << ' '
        << "SHD numnodesdata "<<solheader_.numnodesdata_ << endl;
   */

    int frontTeoric = sizeof(int) + sizeof(double) * solheader_.numnodes_ * sumdof;

    size = solheader_.numnodesdata_ * (sumdof);
    dof = new double[size]; // Her damit!
    if (DoubleRecord(dof, size) != size)
    {
        return (5);
    }

    // Erst mal pro DOF einen double-Array erstellen
    DOFData_ = new DOFData; //CreateNewDOFList();
    DOFData_->anz_ = codes.size() * solheader_.numnodesdata_;
    DOFData_->data_ = new double[DOFData_->anz_];
    DOFData_->displacements_ = new double[3 * solheader_.numnodesdata_];
    DOFData_->nodesdataanz_ = solheader_.numnodesdata_;
    DOFData_->nodesdata_ = new int[DOFData_->nodesdataanz_];

    // Be careful, may be there is no output for all nodes!!!
    if (header_.version_ > 9)
    {
        front[0] = frontTeoric;
    }
    else if (front[0] < frontTeoric) // FIXME !!!!!!!!!!!!!!!!!!!
    {
        fseek(rfp_, 3 * sizeof(int), SEEK_CUR); // jump over tail of the last record
        if (IntRecord(DOFData_->nodesdata_, DOFData_->nodesdataanz_)
            != DOFData_->nodesdataanz_)
        {
            return 5;
        }
    }

    // fill displacements
    for (i = 0; i < 3; ++i)
    {
        int code_order;
        for (code_order = 0; code_order < solheader_.numdofs_; ++code_order)
        {
            if (solheader_.dof_[code_order] == (i + 1))
                break;
        }
        if (!(solheader_.mask_ & 0x200)) // Nur Teilbereich der Knoten wurde verwendet, assume all nodes have output -> FIXME
        {
            int base = i * solheader_.numnodesdata_;
            if (code_order == solheader_.numdofs_)
            {
                for (j = 0; j < solheader_.numnodesdata_; ++j)
                {
                    DOFData_->displacements_[base + j] = 0.0;
                }
            }
            else
            {
                for (j = 0; j < solheader_.numnodesdata_; ++j)
                {
                    DOFData_->displacements_[base + j] = dof[j * sumdof + code_order];
                }
            }
        }
        else
        {
            int datanum;
            int base = i * solheader_.numnodesdata_;
            for (datanum = 0; datanum < solheader_.numnodesdata_; ++datanum)
            {
                DOFData_->displacements_[datanum + base] = 0.0;
            }
        }
    }

    // fill requested DOFs
    for (i = 0; i < codes.size(); ++i)
    {
        DOFData_->dataset_ = num;
        int code_order;
        if (codes[i] < ReadANSYS::EX_OFFSET) // non-extra normal scalar DOF
        {
            // find position for this code
            for (code_order = 0; code_order < solheader_.numdofs_; ++code_order)
            {
                if (solheader_.dof_[code_order] == codes[i])
                    break;
            }
            if (code_order == solheader_.numdofs_)
            {
                Covise::sendError("The code of a degree of freedom was not found");
                return 5;
            }

            DOFData_->typ_ = codes[i];
            DOFData_->exdof_ = false;
        }
        else
        {
            // extra DOF
            DOFData_->typ_ = codes[i] - ReadANSYS::EX_OFFSET;
            for (code_order = 0; code_order < solheader_.numexdofs_; ++code_order)
            {
                if (solheader_.exdof_[code_order] == DOFData_->typ_)
                    break;
            }
            // !!!!!!!
            if (code_order == solheader_.numdofs_)
                abort();
            code_order += solheader_.numdofs_; // ??????? !!!!!!!!
            DOFData_->exdof_ = true;
        }
        //    cout << "Nodes !solheader_.mask_: "<< !(solheader_.mask_ & 0x200) << endl;
        if (!(solheader_.mask_ & 0x200)) // Nur Teilbereich der Knoten wurde verwendet, assume all nodes have output -> FIXME
        {
            int base = i * solheader_.numnodesdata_;
            for (j = 0; j < solheader_.numnodesdata_; ++j)
            {
                DOFData_->data_[base + j] = dof[j * sumdof + code_order];
            }
        }
        else
        {
            int datanum;
            for (datanum = 0; datanum < DOFData_->anz_; ++datanum)
            {
                DOFData_->data_[datanum] = DImpossible_;
            }
        }
        // ACHTUNG: Extradofs werden falsch bezeichnet!
        // ACHTUNG: Bei Teilbereich sind alle Daten Null!
    }
    delete[] dof;
    // Alle dofs gelesen und gespeichert!

    return (0);
}

// -----------------------------------------------
// ReadSHDR
// -----------------------------------------------
int
ReadRST::ReadSHDR(int num)
{
    unsigned int *buf = NULL, solbuf[103];
    int size, i;
    long long offset;
    double dsol[100];

    //  SOLUTIONHEADER shdr;

    // File sollte offen sein
    if (rfp_ == NULL)
        return (1);

    // Out of range check
    if (num > rstheader_.numsets_ /* rstheader_.maxres_ */)
        return (3);

    // Springe erst mal zu der DSI-Tabelle
    offset = rstheader_.ptr_dsi_ * 4;
    fseek(rfp_, (long)offset, SEEK_SET); // im pointer steht die Anzahl der int-Elemente vom Anfang

    // Jetzt sollte man die Tabelle Lesen koennen. Die ist mitunter aber recht gross
    // gewoehnlich beinhaltet sie 2*1000 Eintraege (besser 1000 64-bit Pointer)
    // dazu kommt dann immer noch der Lead-in (2 Ints) und er Lead-out (1 int)
    size = 2 * rstheader_.maxres_ + 3;
    buf = new unsigned int[size];

    if (fread(buf, sizeof(unsigned int), size, rfp_) != size)
    {
        return (2);
    }
    // jetzt mal diese Tabelle auswerten
    solheader_.offset_ = 0;
    // Hi/Lo lesen umdrehen und einfuegen
    if (SwitchEndian_ == DO_NOT_SWITCH)
    {
        solheader_.offset_ = ((long long)(buf[num + 2 + rstheader_.maxres_]) << 32 | buf[num + 1]) * 4;
        if (num < rstheader_.numsets_)
        {
            solheader_.next_offset_ = ((long long)(buf[num + 3 + rstheader_.maxres_]) << 32 | buf[num + 2]) * 4;
        }
        else
        {
            solheader_.next_offset_ = file_size_;
        }
    }
    else
    {
        solheader_.offset_ = ((long long)(SwitchEndian(buf[num + 2 + rstheader_.maxres_])) << 32 | SwitchEndian(buf[num + 1])) * 4;
        if (num < rstheader_.numsets_)
        {
            solheader_.next_offset_ = ((long long)(SwitchEndian(buf[num + 3 + rstheader_.maxres_])) << 32 | SwitchEndian(buf[num + 2])) * 4;
        }
        else
        {
            solheader_.next_offset_ = file_size_;
        }
    }
    // jetzt da hin springen und dort einen Solution Header lesen
    fseek(rfp_, (long)solheader_.offset_, SEEK_SET);
    if (fread(solbuf, sizeof(unsigned int), 103, rfp_) != 103)
    {
        return (4);
    }
    // Jetzt die Werte decodieren und zuweisen
    solheader_.numelements_ = SwitchEndian(solbuf[3]);
    solheader_.numnodes_ = SwitchEndian(solbuf[4]);
    solheader_.mask_ = SwitchEndian(solbuf[5]);
    solheader_.loadstep_ = SwitchEndian(solbuf[6]);
    solheader_.iteration_ = SwitchEndian(solbuf[7]);
    solheader_.sumiteration_ = SwitchEndian(solbuf[8]);
    solheader_.numreact_ = SwitchEndian(solbuf[9]);
    solheader_.maxesz_ = SwitchEndian(solbuf[10]);
    solheader_.nummasters_ = SwitchEndian(solbuf[11]);
    solheader_.ptr_nodalsol_ = SwitchEndian(solbuf[12]);
    solheader_.ptr_elemsol_ = SwitchEndian(solbuf[13]);
    solheader_.ptr_react_ = SwitchEndian(solbuf[14]);
    solheader_.ptr_masters_ = SwitchEndian(solbuf[15]);
    solheader_.ptr_bc_ = SwitchEndian(solbuf[16]);
    solheader_.extrapolate_ = SwitchEndian(solbuf[17]);
    solheader_.mode_ = SwitchEndian(solbuf[18]);
    solheader_.symmetry_ = SwitchEndian(solbuf[19]);
    solheader_.complex_ = SwitchEndian(solbuf[20]);
    solheader_.numdofs_ = SwitchEndian(solbuf[21]);
    // jetzt Titel und Subtitel lesen
    memcpy(solheader_.title_, &solbuf[52], sizeof(int) * 20);
    solheader_.title_[79] = '\0';
    memcpy(solheader_.subtitle_, &solbuf[72], sizeof(int) * 20);
    solheader_.subtitle_[79] = '\0';
    // weiter gehts
    solheader_.changetime_ = SwitchEndian(solbuf[92]);
    solheader_.changedate_ = SwitchEndian(solbuf[93]);
    solheader_.changecount_ = SwitchEndian(solbuf[94]);
    solheader_.soltime_ = SwitchEndian(solbuf[95]);
    solheader_.soldate_ = SwitchEndian(solbuf[96]);
    solheader_.ptr_onodes_ = SwitchEndian(solbuf[97]);
    solheader_.ptr_oelements_ = SwitchEndian(solbuf[98]);
    solheader_.numexdofs_ = SwitchEndian(solbuf[99]);
    solheader_.ptr_extra_a_ = SwitchEndian(solbuf[100]);
    solheader_.ptr_extra_t_ = SwitchEndian(solbuf[101]);

    // DOFs einlesen
    delete[] solheader_.dof_;
    solheader_.dof_ = new int[solheader_.numdofs_];
    // erst mal die normalen DOFs kopieren
    for (i = 0; i < solheader_.numdofs_; ++i)
        solheader_.dof_[i] = SwitchEndian(solbuf[22 + i]);
    // exdofs reinhauen
    delete[] solheader_.exdof_;
    solheader_.exdof_ = new int[solheader_.numexdofs_];

    // Jetzt die TIME Varioable aus dem folgenden Daten besorgen
    // Zwei ints Ã¼berspringen
    fseek(rfp_, 2 * 4, SEEK_CUR);
    if (DoubleRecord(dsol, 100) != 100)
    {
        return (7);
    }
    // Time Variable sollte an erster Stelle stehen
    solheader_.time_ = dsol[0];

    offset = solheader_.offset_ + (solheader_.ptr_extra_a_ + 2) * 4;
    fseek(rfp_, (long)offset, SEEK_SET);
    int exdofbuf[64];
    if (IntRecord(exdofbuf, 64) != 64)
    {
        return (6);
    }
    for (i = 0; i < solheader_.numexdofs_; ++i)
        solheader_.exdof_[i] = exdofbuf[i];

    delete[] buf;

    return (0);
}

// -----------------------------------------------
// GetNodes
// -----------------------------------------------
// Liest die Koordinaten der Knoten ein
// Rueckgabe:
// 0        : alles ok
// 1        : Read Error Geometrieheader
// 2        : Read Error Nodes
// 3        : Read Error Elementbeschreibung
// 4        : Read Error ETYs
// 5        : Read Error Elementtabelle
// 6        : Read Error Elemente
int
ReadRST::GetNodes(void)
{
    GeometryHeader ghdr;
    long long offset;
    int size, *buf, i, j, *etybuf;

    static const float DGR_TO_RAD = (float)M_PI / 180.0f;

    if (node_)
        return 0;

    // Springe erst mal zu der Geometrie-Tabelle
    offset = rstheader_.ptr_geo_ * 4;
    fseek(rfp_, (long)offset, SEEK_SET); // im pointer steht die Anzahl der int-Elemente vom Anfang

    size = 43;
    buf = new int[size];

    if (fread(buf, sizeof(int), size, rfp_) != size)
    {
        return (1);
    }

    // Werte zuweisen
    ghdr.maxety_ = SwitchEndian(buf[3]);
    ghdr.maxrl_ = SwitchEndian(buf[4]);
    ghdr.nodes_ = SwitchEndian(buf[5]);
    ghdr.elements_ = SwitchEndian(buf[6]);
    ghdr.maxcoord_ = SwitchEndian(buf[7]);
    ghdr.ptr_ety_ = SwitchEndian(buf[8]);
    ghdr.ptr_rel_ = SwitchEndian(buf[9]);
    ghdr.ptr_nodes_ = SwitchEndian(buf[10]);
    ghdr.ptr_sys_ = SwitchEndian(buf[11]);
    ghdr.ptr_eid_ = SwitchEndian(buf[12]);
    ghdr.ptr_mas_ = SwitchEndian(buf[17]);
    ghdr.coordsize_ = SwitchEndian(buf[18]);
    ghdr.elemsize_ = SwitchEndian(buf[19]);
    ghdr.etysize_ = SwitchEndian(buf[20]);
    ghdr.rcsize_ = SwitchEndian(buf[21]);

    if (ghdr.ptr_ety_ == 0 && ghdr.ptr_rel_ == 0 && ghdr.ptr_nodes_ == 0 && ghdr.ptr_sys_ == 0 && ghdr.ptr_eid_ == 0)
    {
        ghdr.ptr_ety_ = SwitchEndian(buf[22]);
        ghdr.ptr_nodes_ = SwitchEndian(buf[28]);
        ghdr.ptr_eid_ = SwitchEndian(buf[30]);
        mode64_ = true;
#ifdef DEBUG
        cout << "ReadRST:: switch to 64bit mode" << endl;
#endif
    }

    // Jetzt zu den KNoten springen und diese lesen (Lead in ueberspringen)
    offset = (ghdr.ptr_nodes_ + 2) * 4;
    fseek(rfp_, (long)offset, SEEK_SET);

    // Jetzt die NODES definieren
    node_ = new Node[ghdr.nodes_];
    for (i = 0; i < ghdr.nodes_; ++i)
    {
        double nodeInfo[7];
        if (DoubleRecord(nodeInfo, 7) != 7)
            return (2);

        // Werte jetzt umdrehen
        node_[i].id_ = nodeInfo[0];
        node_[i].x_ = nodeInfo[1];
        node_[i].y_ = nodeInfo[2];
        node_[i].z_ = nodeInfo[3];
        node_[i].thxy_ = DGR_TO_RAD * nodeInfo[4];
        node_[i].thyz_ = DGR_TO_RAD * nodeInfo[5];
        node_[i].thzx_ = DGR_TO_RAD * nodeInfo[6];
        node_[i].MakeRotation();
        if (mode64_)
        {
            fseek(rfp_, 12, SEEK_CUR);
        }
    }
    // das war's!
    anznodes_ = ghdr.nodes_;

    // Jetzt die Elemente lesen: zuerst mal zu ETY um die Elementbeschreibungen zu laden
    offset = (ghdr.ptr_ety_ + 2) * 4;
    fseek(rfp_, (long)offset, SEEK_SET);

    delete[] buf;

    buf = new int[ghdr.maxety_];
    if (fread(buf, sizeof(int), ghdr.maxety_, rfp_) != ghdr.maxety_)
    {
        return (3);
    }

    // ETYs im Objekt erstellen
    ety_ = new EType[ghdr.maxety_];
    anzety_ = ghdr.maxety_;
    etybuf = new int[ghdr.etysize_];

    for (i = 0; i < ghdr.maxety_; ++i)
    {
        if (mode64_)
        {
            offset = (SwitchEndian(buf[i]) + ghdr.ptr_ety_ + 2) * 4;
        }
        else
        {
            offset = (SwitchEndian(buf[i]) + 2) * 4;
        }
        fseek(rfp_, offset, SEEK_SET);

        if (fread(etybuf, sizeof(int), ghdr.etysize_, rfp_) != ghdr.etysize_)
        {
            return (4);
        }
        // Daten jetzt in den ety bringen
        ety_[i].id_ = SwitchEndian(etybuf[0]);
        ety_[i].routine_ = SwitchEndian(etybuf[1]);
        ety_[i].keyops_[0] = SwitchEndian(etybuf[2]);
        ety_[i].keyops_[1] = SwitchEndian(etybuf[3]);
        ety_[i].keyops_[2] = SwitchEndian(etybuf[4]);
        ety_[i].keyops_[3] = SwitchEndian(etybuf[5]);
        ety_[i].keyops_[4] = SwitchEndian(etybuf[6]);
        ety_[i].keyops_[5] = SwitchEndian(etybuf[7]);
        ety_[i].keyops_[6] = SwitchEndian(etybuf[8]);
        ety_[i].keyops_[7] = SwitchEndian(etybuf[9]);
        ety_[i].keyops_[8] = SwitchEndian(etybuf[10]);
        ety_[i].keyops_[9] = SwitchEndian(etybuf[11]);
        ety_[i].keyops_[10] = SwitchEndian(etybuf[12]);
        ety_[i].keyops_[11] = SwitchEndian(etybuf[13]);
        ety_[i].dofpernode_ = SwitchEndian(etybuf[33]);
        ety_[i].nodes_ = SwitchEndian(etybuf[60]);
        ety_[i].nodeforce_ = SwitchEndian(etybuf[62]);
        ety_[i].nodestress_ = SwitchEndian(etybuf[93]);
    }
    delete[] buf;
    delete[] etybuf;
    // Passt schon!
    // Jetzt die Elemente selber einlesen
    element_ = new Element[ghdr.elements_];

    anzelem_ = ghdr.elements_;

    buf = new int[2 * ghdr.elements_];

    // hinsurfen und lesen
    offset = (ghdr.ptr_eid_ + 2) * 4;
    fseek(rfp_, offset, SEEK_SET);

    if (fread(buf, sizeof(int), 2 * ghdr.elements_, rfp_) != 2 * ghdr.elements_)
    {
        return (5);
    }

    etybuf = new int[10];
    // Element accounting for user information
    int populations[ANSYS::LIB_SIZE];
    memset(populations, 0, sizeof(int) * ANSYS::LIB_SIZE);
    // Jetzt mit Schleife alle Elemente packen
    for (i = 0; i < ghdr.elements_; ++i)
    {
        if (mode64_)
        {
            offset = (SwitchEndian(buf[2 * i]) + ghdr.ptr_eid_ + 2) * 4;
        }
        else
        {
            offset = (SwitchEndian(buf[i]) + 2) * 4;
        }

        fseek(rfp_, offset, SEEK_SET);
        if (fread(etybuf, sizeof(int), 10, rfp_) != 10)
        {
            return (6);
        }
        // Jetzt Daten zuweisen
        element_[i].material_ = SwitchEndian(etybuf[0]);
        element_[i].type_ = SwitchEndian(etybuf[1]);
        element_[i].real_ = SwitchEndian(etybuf[2]);
        element_[i].section_ = SwitchEndian(etybuf[3]);
        element_[i].coord_ = SwitchEndian(etybuf[4]);
        element_[i].death_ = SwitchEndian(etybuf[5]);
        element_[i].solidmodel_ = SwitchEndian(etybuf[6]);
        element_[i].shape_ = SwitchEndian(etybuf[7]);
        element_[i].num_ = SwitchEndian(etybuf[8]);

        for (j = 0; j < anzety_; ++j)
        {
            if (ety_[j].id_ == element_[i].type_) // find ety position -> make faster
            {
                element_[i].anznodes_ = ety_[j].nodes_;
                ++(populations[ety_[j].routine_]);
            }
        }
        element_[i].nodes_ = new int[element_[i].anznodes_];
        if (IntRecord(element_[i].nodes_, element_[i].anznodes_)
            != element_[i].anznodes_)
        {
            return (6);
        }
    }

    // Some statistics
    Covise::sendInfo("Total number of nodes: %d", ghdr.nodes_);
    Covise::sendInfo("Total number of elements: %d", ghdr.elements_);

    // std::string InfoMessage("Number of elements in each element category:");
    for (i = 0; i < ANSYS::LIB_SIZE; ++i)
    {
        if (populations[i])
        {
            // char numBuf[128];
            // sprintf(numBuf," in routine %d, %d elements;",i,populations[i]);
            // InfoMessage += numBuf;

            Covise::sendInfo("Element type [%d]: %d elements", i, populations[i]);
        }
    }
    // size_t pos = InfoMessage.find_last_of(";");
    // if(pos != std::string::npos)
    //   InfoMessage.resize(pos-1);
    // InfoMessage += ".";
    // Covise::sendInfo("%s", InfoMessage.c_str());

    delete[] buf;
    delete[] etybuf;
    return (0);
}

// -----------------------------------------------
// GetTime
// -----------------------------------------------
// Return:
// <0   Fehler und zwar:
// -1.0     : Index out of Range
// -2.0     : Keine Zeittafel vorhanden
double
ReadRST::GetTime(int pos)
{
    if (pos > rstheader_.numsets_) // index ausserhalb
        return (-1.0);

    if (timetable_ == NULL) // keine Zeittabelle gelesen
    {
        return pos;
        // return(-2.0);
    }

    return (timetable_[pos]); // Zeitwert zurÃ¼ck
}
