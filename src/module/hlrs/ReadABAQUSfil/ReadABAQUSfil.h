/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef READABAQUSFIL_H
#define READABAQUSFIL_H
/****************************************************************************\ 
 **                                                                          **
 **                                                                          **
 ** Description:                                                             **
 **                                                                          **
 ** Name:        Read Unstruct                                               **
 ** Category:    IO                                                          **
 **                                                                          **
 ** Author:                                                                  **
 **                                                                          **
 ** History:  								     **
 **              					       		     **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <api/coSimpleModule.h>
#include <string>
using namespace covise;

class ReadABAQUSfil : public coSimpleModule
{

private:
    //////////  member functions

    /// this module has only the compute call-back
    virtual int compute(const char *port);
    virtual void param(const char *, bool);

    // ------------------------------------------------------------
    // -- parameters

    // File selector for .fil result file
    coFileBrowserParam *p_filFile;

    // Drop down list for element scalar results
    coChoiceParam *p_elemres;
    // Drop down list for element tensor results
    coChoiceParam *p_telemres;
    // Drop down list for nodal results
    coChoiceParam *p_nodalres;

    // String input field for Element Set selection
    coStringParam *p_selectedSets;
    
    // ------------------------------------------------------------
    // -- ports

    // Grid ----------------------
    coOutputPort *p_gridOutPort;
    // SetGrid -------------------
    coOutputPort *p_SetgridOutPort;

    // Element results -----------
    coOutputPort *p_eresOutPort;
    // Element results -----------
    coOutputPort *p_tresOutPort;
    // Nodal results -------------
    coOutputPort *p_nresOutPort;
    // SetGrid Element results ---
    coOutputPort *p_SetgridResPort;
    // SetGrid Element results ---
    coOutputPort *p_SetgridTResPort;
    // SetGrid Nodal results -----
    coOutputPort *p_SetgridnResPort;

    // Global Variables to store the .fil file in memeory
    // initialized in : param
    // used in        : compute
    int64_t *fil_array;
    int64_t data_length;

    // Ensure equivalence of loaded .fil file and selected
    // parameters. Needed in case of reload of stored map
    const char *fil_name;

    // Flags to ensure decoupling between param and compute
    // (Really needed ???)
    bool inMapLoading;
    // hack: set this flag if param called from compute
    bool computeRunning;

    struct t_jobhead
    {

      char version[9];
      char date[16];
      char time[8];
      
      int no_conn;
      int no_nodes;
      int no_elems, no_sup_elems;
      
      int no_node_sets, no_elem_sets;

      float typical_el_length;

      int no_steps;

    } jobhead;

    typedef struct tSets
    {
      string type;
      string cname;
      int    cref;
      string name;

      int      no_nodes;
      int      no_elems;
      int*     elem_numbers;
      int*     node_numbers;
      int      no_conn;

    } tSets;

    typedef struct tCref
    {
      int    cref;
      string name;
    } tCref;

    // Container for set parameters
    vector<tSets> vsets;

public:
    ReadABAQUSfil(int argc, char *argv[]);
};
#endif
