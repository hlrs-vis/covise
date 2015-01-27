/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _FIELD_FILE_H_
#define _FIELD_FILE_H_

#define MAX_SCAL_FIELDS 70
#define MAX_VECT_FIELDS 15
#define MAX_FIELD_LENGHT 300

class Fields
{

private:
    int nbvect, nbscal, nbfields;
    char **VectArray;
    char **ScalArray;
    char **VectFieldArray;

public:
    // Member functions

    Fields(char **, int);
    ~Fields();
    void getAddresses(const char *const **, const char *const **, const char *const **, int *, int *, int *);
    void get_fields(char ***, char ***, int *, int *);
};
#endif
