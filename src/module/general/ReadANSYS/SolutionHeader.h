/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SOLUTION_HEADER_H_
#define _SOLUTION_HEADER_H_

struct SolutionHeader
{
    int numelements_; // Anzahl Elemente
    int numnodes_; // Anzahl der Knoten
    int numnodesdata_; // Anzahl der Knoten mit Daten
    unsigned int mask_; // Bitmask
    int loadstep_; // eben jener
    int iteration_; // Iterationsnummer
    int sumiteration_; // cummulative Iterationsnummer
    int numreact_; // Anzahl der Reaktionskräfte
    int maxesz_; // nicht dokumentiert
    int nummasters_; // Anzahl der Master (?)
    int ptr_nodalsol_; // Offset zur nodal solution
    int ptr_elemsol_; // Offset zur element solution
    int ptr_react_; // Offset zu den Reaktionskräften
    int ptr_masters_; // Offset zu den Masters(?)
    int ptr_bc_; // Offset zu den Boundary conditions
    int extrapolate_; // 0=move, 1=extra unless non-linear, 2=extra always
    int mode_; // Mode number of harminic loads
    int symmetry_; // für harmonische Last _;)
    int complex_; // 0=real, 1=complex
    int numdofs_; // Anzahl der DOFs in diesem Datensatz
    int *dof_; // DOF Referenzzahl, muss numdofs lang sein
    int *exdof_; // Eigenbau: enthält die exdofs (numexdofs lang)
    int changetime_; // compact: letzte Veränderung
    int changedate_; // compact: letzte Veränderung
    int changecount_; // wie oft wurde der Datensatz geändert?
    int soltime_; // compact: Ergebniszeit
    int soldate_; // compact: Ergebnisdatum
    int ptr_onodes_; // Offset zur Liste der geordneten Knoten (load case)
    int ptr_oelements_; // Offset zur Liste der geordneten Elemente (load case)
    int numexdofs_; // Anzahl zusätzlicher DOFs für Flotran
    int ptr_extra_a_; // Offset zurm EXA-Header
    int ptr_extra_t_; // Offset zum EXT_Header
    char title_[80];
    char subtitle_[80];
    long long offset_; // Fileoffset
    long long next_offset_; // Fileoffset
    // Jetzt noch ein paar Daten aus dem Double Datensatz
    double time_; // Zeitpunkt bei transienten Daten
    SolutionHeader()
    {
        dof_ = NULL;
        exdof_ = NULL;
    }
    void clean()
    {
        delete[] dof_;
        delete[] exdof_;
        dof_ = NULL;
        exdof_ = NULL;
    }
    ~SolutionHeader()
    {
        clean();
    }
};
#endif
