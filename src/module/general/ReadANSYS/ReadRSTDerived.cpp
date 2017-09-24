/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReadRST.h"
#include <util/coviseCompat.h>
#include "ANSYS.h"
#include <api/coModule.h>
using namespace covise;

#ifndef _WIN32
#include <sys/mman.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>

#ifndef MAP_FILE
#define MAP_FILE 0
#endif

int
ReadRST::ReadDerived(const std::string &filename,
                     int num,
                     DerivedType derivType)
{
    Reset(PRESERVE_DOF_DATA); // alles loeschen, wenn noetig
    int problems = OpenFile(filename);
    // Mal keine Fehlerabfrage, FIXME!
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
    problems = GetDatasetDerived(num, derivType);
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
    return (problems);
}

int
ReadRST::GetDatasetDerived(int num,
                           DerivedType derType)
{
    long long offset;

    // File sollte offen sein
    if (rfp_ == NULL)
        return (1);

    // Out of range check
    if (num > rstheader_.numsets_ /*rstheader_.maxres_*/)
        return (3);
    // Eventuell alte Daten Loeschen
    // DOF-Liste loeschen
    delete DerivedData_;
    DerivedData_ = NULL;
    /*
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
    // so, alles geloescht
    ReadSHDR(num);

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
    // jetzt Daten einlesen
    // but first read record length...
    if (solheader_.ptr_elemsol_ == 0) // no DOF data at all
    {
        fprintf(stderr, "no DOF data at all!\n");
        DerivedData_ = new DerivedData;
        DerivedData_->anz_ = 0;
        DerivedData_->data_ = NULL;
        // return 0;  // Why 0? Returning 0 leads to a crash; if reading no DOFs is an error then the return value should be 5
        return 5;
    }

    offset = solheader_.offset_ + solheader_.ptr_elemsol_ * 4 + 2 * sizeof(int);

#ifdef DEBUG
    cout << "ESL offset " << offset << endl;
#endif
#ifdef WIN32
	_fseeki64(rfp_, offset, SEEK_SET);
#else
	fseek(rfp_, offset, SEEK_SET);
#endif

    // Read now the entire element solutions index table
    int *ESLptrs = new int[solheader_.numelements_];
    if (!mode64_)
    {
        if (IntRecord(ESLptrs, solheader_.numelements_) != solheader_.numelements_)
        {
            return 5;
        }
    }
    else
    {
        int tmp[2];
        for (int w = 0; w < solheader_.numelements_; w++)
        {
            IntRecord(&tmp[0], 2);
            ESLptrs[w] = tmp[0];
        }
    }
    int elem;
    int *ESLSolptrs = new int[solheader_.numelements_];
    memset(ESLSolptrs, 0, sizeof(int) * solheader_.numelements_);
//  cout << (solheader_.mask_ & 0x800) << endl;
#ifdef WIN32
    mmap_flag_ = 0;
#else
    mmap_flag_ = 1;
#endif

    if (mmap_flag_)
    {
#ifndef WIN32
        long pageSize = sysconf(_SC_PAGESIZE);
        mmap_off_ = pageSize * (solheader_.offset_ / pageSize);
        actual_off_ = solheader_.offset_ - mmap_off_;
        mmap_len_ = solheader_.next_offset_ - mmap_off_;
        mmap_ini_ = mmap(0, mmap_len_, PROT_READ, MAP_SHARED, file_des_, mmap_off_);
        if (mmap_ini_ == MAP_FAILED)
        {
            Covise::sendWarning("Could not map file to memory");
            return 2;
        }
#endif
    }

    if (1 || (solheader_.mask_ & 0x800)) // there are derived data
    { //  Note: the previous condition is always true!! Mask is useless. Why code like this?
        // get pointers ptrENS... for each element
        if (mmap_flag_)
        {
            for (elem = 0; elem < solheader_.numelements_; ++elem)
            {
                if (!ESLptrs[elem])
                    continue;
#ifdef DEBUG
                if (elem < 1)
                {
                    const char *ansptr[] = { "ptrEMS", "ptrENF", "ptrENS", "ptrENG", "ptrEGR",
                                             "ptrEEL", "ptrEPL", "ptrECR", "ptrETH", "ptrEUL",
                                             "ptrEFX", "ptrELF", "ptrEMN", "ptrECD", "ptrENL",
                                             "ptrEHC", "ptrEPT", "ptrESF", "ptrETB", "ptrECT",
                                             "ptrEXY", "ptrEBA", "ptrESV", "0" };

                    for (int j = 0; j < 23; j++)
                    {
                        int wert;
                        actual_off_ = offset - mmap_off_ + (ESLptrs[elem] + j) * 4;
                        IntRecord(&wert, 1);

                        cout << "EPOI " << ansptr[j] << " " << wert << endl;
                    }
                }
#endif
                if (!mode64_)
                {
                    actual_off_ = solheader_.offset_ - mmap_off_ + (ESLptrs[elem] + derType) * 4;
                }
                else
                {
                    if (header_.version_ >= 10.)
                    {
                        actual_off_ = offset - mmap_off_ + (ESLptrs[elem] + derType - 2) * 4;
                    }
                    else
                    {
                        actual_off_ = offset - mmap_off_ + (ESLptrs[elem] + derType) * 4;
                    }
                }
                if (IntRecord(&ESLSolptrs[elem], 1) != 1)
                {
                    return 5;
                }
#ifdef DEBUG
                if (elem < 30)
                    cout << "EE " << elem << " " << ESLSolptrs[elem] << " " << offset + (ESLptrs[elem] + derType - 2) * 4 << " base: " << offset + ESLptrs[elem] * 4 << endl;
#endif
            }
        }
        else
        {
            for (elem = 0; elem < solheader_.numelements_; ++elem)
            {
                if (!ESLptrs[elem])
                    continue;
                if (!mode64_)
                {
#ifdef WIN32
					_fseeki64(rfp_, solheader_.offset_ + (ESLptrs[elem] + derType) * 4, SEEK_SET);
#else
                    fseek(rfp_, solheader_.offset_ + (ESLptrs[elem] + derType) * 4, SEEK_SET);
#endif
                }
                else
                {
                    if (header_.version_ >= 10.)
                    {
#ifdef WIN32
						_fseeki64(rfp_, offset + (ESLptrs[elem] + derType - 2) * 4, SEEK_SET);
#else
                        fseek(rfp_, offset + (ESLptrs[elem] + derType - 2) * 4, SEEK_SET);
#endif
                    }
                    else
                    {
#ifdef WIN32
						_fseeki64(rfp_, offset + (ESLptrs[elem] + derType) * 4, SEEK_SET);
#else
						fseek(rfp_, offset + (ESLptrs[elem] + derType) * 4, SEEK_SET);
#endif
                    }
                }
                if (IntRecord(&ESLSolptrs[elem], 1) != 1)
                {
                    return 5;
                }
            }
        }
    }
    // We never reach this point
    else
    {
        Covise::sendInfo("No derived data");
    }

    DerivedData_ = new DerivedData;
    DerivedData_->anz_ = solheader_.numelements_;
    fprintf(stderr, "DerivedData_->anz_ = solheader_.numelements_ = %d\n", DerivedData_->anz_);
    DerivedData_->data_ = new std::vector<double>[DerivedData_->anz_];
    // initialise data_ with impossible
    for (elem = 0; elem < solheader_.numelements_; ++elem)
    {
        if (ESLSolptrs[elem] <= 0)
        {
            DerivedData_->data_[elem].push_back(ReadRST::DImpossible_);
            continue;
        }
        else
        {
            // Determine length of ENS record
            int lengthBytes;
            if (mmap_flag_)
            {
                if (!mode64_)
                {
                    actual_off_ = solheader_.offset_ - mmap_off_ + ESLSolptrs[elem] * 4;
                }
                else
                {
                    actual_off_ = offset + ESLptrs[elem] * 4 - mmap_off_ + ESLSolptrs[elem] * 4 - 8;
                }
            }
            else
            {
                if (!mode64_)
                {
#ifdef WIN32
					_fseeki64(rfp_, offset + ESLptrs[elem] * 4 + ESLSolptrs[elem] * 4 - 8, SEEK_SET);
#else
                    fseek(rfp_, solheader_.offset_ + ESLSolptrs[elem] * 4, SEEK_SET);
#endif
                }
                else
                {
#ifdef WIN32
					_fseeki64(rfp_, offset + ESLptrs[elem] * 4 + ESLSolptrs[elem] * 4 - 8, SEEK_SET);
#else
                    fseek(rfp_, offset + ESLptrs[elem] * 4 + ESLSolptrs[elem] * 4 - 8, SEEK_SET);
#endif
                }
            }
            if (IntRecord(&lengthBytes, 1) != 1)
            {
                return 5;
            }
#ifdef DEBUG
            cout << "BBB "
                 << " " << offset + ESLptrs[elem] * 4 + ESLSolptrs[elem] * 4 - 8 << " " << lengthBytes << endl;
#endif

            if (lengthBytes < 0)
            {
                return 5;
            }

            int lengthDoubles = 0;
            if (header_.version_ < 9)
            {
                lengthDoubles = (lengthBytes - sizeof(int)) / sizeof(double);
            }
            else
            {
                EType etype = ety_[element_[elem].type_ - 1];
                lengthDoubles = getLengthOfElemRecord(derType, etype);
            }

#ifdef DEBUG
            if (elem < 25)
                cout << "BBB "
                     << " " << offset + ESLptrs[elem] * 4 + ESLSolptrs[elem] * 4 - 8 << " " << lengthDoubles << " " << lengthBytes << endl;
#endif
            DerivedData_->data_[elem].reserve(lengthDoubles);

            if (mmap_flag_)
            {
                actual_off_ += 2 * sizeof(int);
            }
            else
            {
                fseek(rfp_, sizeof(int), SEEK_CUR);
            }
            double *derived = new double[lengthDoubles];
            DoubleRecord(derived, lengthDoubles);
            int item;
            for (item = 0; item < lengthDoubles; ++item)
            {
                DerivedData_->data_[elem].push_back(derived[item]);

#ifdef DEBUG
                if (elem < 25)
                    cout << "CCC " << derived[item] << " ";
#endif
            }
#ifdef DEBUG
            if (elem < 3)
                cout << endl;
#endif
            delete[] derived;
        }
    }
    delete[] ESLptrs;
    delete[] ESLSolptrs;

    if (mmap_flag_)
    {
#ifndef WIN32
        if (munmap(mmap_ini_, mmap_len_))
        {
            Covise::sendWarning("Could not unmap file from memory");
            return 2;
        }
#endif
        mmap_flag_ = 0;
    }
    return 0;
}

int
ReadRST::getLengthOfElemRecord(DerivedType derType, EType &etype)
{
    ANSYS &elem_db_ = ANSYS::get_handle();
    int nb_per_node = 0;
    switch (derType)
    {
    case STRESS:

        switch (elem_db_.getStressSupport(etype.routine_))
        {
        case ANSYS::SOLID:
            nb_per_node = etype.nodestress_ * 11;
            break;
        case ANSYS::LINK:
            nb_per_node = 1;
            break;
        case ANSYS::BEAM3:
            nb_per_node = 3;
            break;
        case ANSYS::BEAM4:
            nb_per_node = 5;
            break;
        case ANSYS::PLANE:
            nb_per_node = etype.nodestress_ * 11;
            break;
        case ANSYS::AXI_SHELL:
        case ANSYS::SHELL:
            nb_per_node = 2 * etype.nodestress_ * 11;
            break;
        case ANSYS::NO_STRESS:
            break;
        default:
            break;
        }
        break;

    case E_EL:
    case E_PLAS:
    case E_CREEP:
    case E_THERM:
        nb_per_node = 7 * etype.nodes_;
        break;

    case VOL_ENERGY:
        nb_per_node = 11;
        break;

    case FIELD_FLUX:
        nb_per_node = 3 * etype.nodestress_;
        break;

    case E_TEMP:
        switch (elem_db_.getStressSupport(etype.routine_))
        {
        case ANSYS::SOLID:
            nb_per_node = etype.nodeforce_; //nodfor
            break;
        case ANSYS::LINK:
            nb_per_node = 1;
            break;
        case ANSYS::BEAM3:
            nb_per_node = 3;
            break;
        case ANSYS::BEAM4:
            nb_per_node = 5;
            break;
        case ANSYS::PLANE:
            nb_per_node = 1;
            break;
        case ANSYS::AXI_SHELL:
        case ANSYS::SHELL:
            nb_per_node = 2 * etype.nodestress_;
            break;
        case ANSYS::NO_STRESS:
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
    return nb_per_node;
}

int
ReadRST::get_file_size()
{
    struct stat file_stats;

    if (fstat(file_des_, &file_stats))
        return -1;

    file_size_ = file_stats.st_size;
    return 0;
}
