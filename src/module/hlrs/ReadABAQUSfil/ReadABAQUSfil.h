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

    // Drop down list for element results
    coChoiceParam *p_elemres;

    // ------------------------------------------------------------
    // -- ports

    // Grid ----------------------
    coOutputPort *p_gridOutPort;
    // Element results -----------
    coOutputPort *p_eresOutPort;

    // Global Variables to store the .fil file in memeory
    // initialized in : param
    // used in        : compute
    int64_t *fil_array;
    int64_t data_length;

    // Ensure equivalence of loaded .file file and selected
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

public:
    ReadABAQUSfil(int argc, char *argv[]);
};
#endif
