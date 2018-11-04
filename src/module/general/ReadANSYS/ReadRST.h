/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS ReadRST
//
//  This class reads raw data from the .rst file
//
//  Initial version: 2001-??-?? Björn Sander
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes: Modified and adapted for structural mechanics by S. Leseduarte

#ifndef _READ_RST_H_
#define _READ_RST_H_

#include "BinHeader.h"
#include "DOFData.h"
#include "DerivedData.h"
#include "Element.h"
#include "Node.h"
#include "RstHeader.h"
#include "EType.h"
#include "GeometryHeader.h"
#include "RstHeader.h"
#include "SolutionHeader.h"
#include <util/coviseCompat.h>
#include <vector>

class ReadRST
{
public:
    enum DerivedType
    {
        STRESS = 4,
        E_EL = 7,
        E_PLAS = 8,
        E_CREEP = 9,
        E_THERM = 10,
        FIELD_FLUX = 12,
        VOL_ENERGY = 5,
        M_FLUX_DENS = 12,
        E_TEMP = 18
    };
    /** Reads file header and sets pointer to FILE
       * @param fname file name
       * @return error code
       */
    int OpenFile(const std::string &string);
    /** Reads raw DOF data required for a magnitude as given in codes
       *  from a file and a given set
       * @param fname file name
       * @param nset set number
       * @param codes contains 1 int for a scalar or 3 for a vector
       * @return error code
       */
    int Read(const std::string &fname, int nset, std::vector<int> &codes);
    /** Reads raw data required for a derived magnitude
       * @param fname file name
       * @param nset set number
       * @param kind kind of derived data (stresses, elastic strains, ...)
       * @return error code
       */
    int ReadDerived(const std::string &fname, int nset, DerivedType kind);
    /// Do not used
    int WriteData(char *); // Schreibt
    /// delivers real time for a given set
    double GetTime(int nset);
    /// Constructor
    ReadRST(void);
    /// Destructor
    ~ReadRST();
    /** Get number of sets
       * @return number of sets
       */
    int getNumTimeSteps()
    {
        return rstheader_.numsets_;
    }
    /** Get number of elements
       * @return number of elements
       */
    int getNumElement()
    {
        return anzelem_;
    }
    /** Get number of nodes
       * @return number of nodes
       */
    int getNumNodes()
    {
        return anznodes_;
    }
    /** Get array to element structures
       * @return array to element structures
       */
    const Element *getElements()
    {
        return element_;
    }
    /** Get array to node structures
       * @return array to node structures
       */
    const Node *getNodes()
    {
        return node_;
    }
    /** Get array to element type structures
       * @return array to element type structures
       */
    const EType *getETypes()
    {
        return ety_;
    }

    /** Get ANSYS version
       * @return ansys version with which the rst was simulated
       */
    int getVersion()
    {
        return header_.version_;
    }

    // Yeah, the style rules are here repeatedly violated.
    // So was it from the very beginning...
    SolutionHeader solheader_;
    DOFData *DOFData_;
    DerivedData *DerivedData_;

    /** Read solution header for a set
       * @param nset set number
       * @return error code
       */
    int ReadSHDR(int nset);

    /** Get array with node ANSYS/user labels
       * @return array with node ANSYS/user labels
       */
    const int *getNodeIndex()
    {
        return nodeindex_;
    }
    /** Get array with element ANSYS/user labels
       * @return array with element ANSYS/user labels
       */
    const int *getElemIndex()
    {
        return elemindex_;
    }
    static const double DImpossible_;
    static const float FImpossible_;
    static const int RADIKAL = 1;
    static const int PRESERVE_DOF_DATA = 2;
    /** Reset object state
       * @param message determines which parts of the object are reset
       */
    int Reset(int message);

protected:
private:
    enum SWITCH_ENDIAN
    {
        SWITCH,
        DO_NOT_SWITCH
    } SwitchEndian_;
    void ChangeSwitch();
    BinHeader header_;
    RstHeader rstheader_;
    Node *node_;
    EType *ety_;
    Element *element_;
    FILE *rfp_; // Result file pointer
    bool mode64_; // file written by 64bit system

    int mmap_flag_;
    int file_des_;
    size_t file_size_;
    void *mmap_ini_;
	size_t mmap_off_;
	size_t actual_off_;
    size_t mmap_len_;

    int anznodes_;
    int anzety_;
    int anzelem_;

    int *nodeindex_;
    int *elemindex_;

    double *timetable_;

    // interne Methoden
    int SwitchEndian(int); // Dreht die Byte-Folge um
    unsigned int SwitchEndian(unsigned int); // Dreht die Byte-Folge um
    double SwitchEndian(double); // Dreht die Byte-Folge um
    void SwitchEndian(char *buf, int length);
    int IntRecord(int *buf, int len);
    int DoubleRecord(double *buf, int len);

    //    DOFROOT *CreateNewDOFList(void); // Erstellt eine neue DOF-Liste
    int GetNodes(void);
    int GetDataset(int, std::vector<int> &); // Liste die Ergebnisdaten
    int GetDatasetDerived(int, DerivedType); // Liste die Ergebnisdaten
    int getLengthOfElemRecord(DerivedType, EType &);
    int get_file_size();
};
#endif
