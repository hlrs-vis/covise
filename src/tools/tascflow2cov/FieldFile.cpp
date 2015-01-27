/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FieldFile.h"
#include <string.h>

Fields::Fields(char **AllFields, int nb_tot_fields)
    : nbvect(0)
    , nbscal(0)
    , nbfields(0)
{
    char c;
    int is_vector, is_found, index = 0;
    char buffer[MAX_FIELD_LENGHT + 1];
    int i, j, k;

    ScalArray = new char *[MAX_SCAL_FIELDS];
    VectArray = new char *[MAX_VECT_FIELDS];
    VectFieldArray = new char *[MAX_SCAL_FIELDS];
    for (i = 0; i < MAX_SCAL_FIELDS; i++)
    {
        VectFieldArray[i] = NULL;
    }

    VectArray[nbvect] = new char[5];
    ScalArray[nbscal] = new char[5];
    strcpy(VectArray[nbvect++], "none");
    strcpy(ScalArray[nbscal++], "none");

    buffer[MAX_FIELD_LENGHT] = '\0';

    for (i = 0; i < nb_tot_fields; i++)
    {
        strcpy(buffer, AllFields[i]);
        c = AllFields[i][0];
        index = 0;
        is_vector = 0;
        is_found = 0;

        for (j = 0; j < nbfields; j++)
            if (VectFieldArray[j])
                if (!strcmp(VectFieldArray[j], buffer))
                    is_found = 1;
        while ((c != ' ') && (c != '\0') && (!is_vector) && (!is_found))
        {
            switch (c)
            {
            case 'X':
                buffer[index] = 'Y';
                j = 0;
                while ((j < nb_tot_fields) && (!is_vector))
                {
                    if (!strcmp(buffer, AllFields[j]))
                    {
                        is_vector = 1;
                        j = nb_tot_fields;
                        for (k = 0; k < 3; k++)
                            VectFieldArray[nbfields + k] = new char[MAX_FIELD_LENGHT + 1];
                        buffer[index] = 'X';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'Y';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'Z';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        for (k = MAX_FIELD_LENGHT - 1; k >= index + 4; k--)
                            buffer[k] = buffer[k - 4];
                        buffer[index] = '_';
                        buffer[index + 1] = 'X';
                        buffer[index + 2] = 'Y';
                        buffer[index + 3] = 'Z';
                        buffer[index + 4] = '_';
                        VectArray[nbvect] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectArray[nbvect++], buffer);
                    }
                    j++;
                }
                if (!is_vector)
                    buffer[index] = c;
                break;
            case 'Y':
                buffer[index] = 'X';
                j = 0;
                while ((j < nb_tot_fields) && (!is_vector))
                {
                    if (!strcmp(buffer, AllFields[j]))
                    {
                        is_vector = 1;
                        j = nb_tot_fields;
                        for (k = 0; k < 3; k++)
                            VectFieldArray[nbfields + k] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'Y';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'Z';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        for (k = MAX_FIELD_LENGHT - 1; k >= index + 4; k--)
                            buffer[k] = buffer[k - 4];
                        buffer[index] = '_';
                        buffer[index + 1] = 'X';
                        buffer[index + 2] = 'Y';
                        buffer[index + 3] = 'Z';
                        buffer[index + 4] = '_';
                        VectArray[nbvect] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectArray[nbvect++], buffer);
                    }
                    j++;
                }
                if (!is_vector)
                    buffer[index] = c;
                break;
            case 'Z':
                buffer[index] = 'X';
                j = 0;
                while ((j < nb_tot_fields) && (!is_vector))
                {
                    if (!strcmp(buffer, AllFields[j]))
                    {
                        is_vector = 1;
                        j = nb_tot_fields;
                        for (k = 0; k < 3; k++)
                            VectFieldArray[nbfields + k] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'Y';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'Z';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        for (k = MAX_FIELD_LENGHT - 1; k >= index + 4; k--)
                            buffer[k] = buffer[k - 4];
                        buffer[index] = '_';
                        buffer[index + 1] = 'X';
                        buffer[index + 2] = 'Y';
                        buffer[index + 3] = 'Z';
                        buffer[index + 4] = '_';
                        VectArray[nbvect] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectArray[nbvect++], buffer);
                    }
                    j++;
                }
                if (!is_vector)
                    buffer[index] = c;
                break;
            case 'U':
                buffer[index] = 'V';
                j = 0;
                while ((j < nb_tot_fields) && (!is_vector))
                {
                    if (!strcmp(buffer, AllFields[j]))
                    {
                        is_vector = 1;
                        j = nb_tot_fields;
                        for (k = 0; k < 3; k++)
                            VectFieldArray[nbfields + k] = new char[MAX_FIELD_LENGHT + 1];
                        buffer[index] = 'U';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'V';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'W';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        for (k = MAX_FIELD_LENGHT - 1; k >= index + 4; k--)
                            buffer[k] = buffer[k - 4];
                        buffer[index] = '_';
                        buffer[index + 1] = 'U';
                        buffer[index + 2] = 'V';
                        buffer[index + 3] = 'W';
                        buffer[index + 4] = '_';
                        VectArray[nbvect] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectArray[nbvect++], buffer);
                    }
                    j++;
                }
                if (!is_vector)
                    buffer[index] = c;
                break;
            case 'V':
                buffer[index] = 'U';
                j = 0;
                while ((j < nb_tot_fields) && (!is_vector))
                {
                    if (!strcmp(buffer, AllFields[j]))
                    {
                        is_vector = 1;
                        j = nb_tot_fields;
                        for (k = 0; k < 3; k++)
                            VectFieldArray[nbfields + k] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'V';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'W';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        for (k = MAX_FIELD_LENGHT - 1; k >= index + 4; k--)
                            buffer[k] = buffer[k - 4];
                        buffer[index] = '_';
                        buffer[index + 1] = 'U';
                        buffer[index + 2] = 'V';
                        buffer[index + 3] = 'W';
                        buffer[index + 4] = '_';
                        VectArray[nbvect] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectArray[nbvect++], buffer);
                    }
                    j++;
                }
                if (!is_vector)
                    buffer[index] = c;
                break;
            case 'W':
                buffer[index] = 'U';
                j = 0;
                while ((j < nb_tot_fields) && (!is_vector))
                {
                    if (!strcmp(buffer, AllFields[j]))
                    {
                        is_vector = 1;
                        j = nb_tot_fields;
                        for (k = 0; k < 3; k++)
                            VectFieldArray[nbfields + k] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'V';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'W';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        for (k = MAX_FIELD_LENGHT - 1; k >= index + 4; k--)
                            buffer[k] = buffer[k - 4];
                        buffer[index] = '_';
                        buffer[index + 1] = 'U';
                        buffer[index + 2] = 'V';
                        buffer[index + 3] = 'W';
                        buffer[index + 4] = '_';
                        VectArray[nbvect] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectArray[nbvect++], buffer);
                    }
                    j++;
                }
                if (!is_vector)
                    buffer[index] = c;
                break;

            case 'I':
                buffer[index] = 'J';
                j = 0;
                while ((j < nb_tot_fields) && (!is_vector))
                {
                    if (!strcmp(buffer, AllFields[j]))
                    {
                        is_vector = 1;
                        j = nb_tot_fields;
                        for (k = 0; k < 3; k++)
                            VectFieldArray[nbfields + k] = new char[MAX_FIELD_LENGHT + 1];
                        buffer[index] = 'I';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'J';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'K';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        for (k = MAX_FIELD_LENGHT - 1; k >= index + 4; k--)
                            buffer[k] = buffer[k - 4];
                        buffer[index] = '_';
                        buffer[index + 1] = 'I';
                        buffer[index + 2] = 'J';
                        buffer[index + 3] = 'K';
                        buffer[index + 4] = '_';
                        VectArray[nbvect] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectArray[nbvect++], buffer);
                    }
                    j++;
                }
                if (!is_vector)
                    buffer[index] = c;
                break;
            case 'J':
                buffer[index] = 'I';
                j = 0;
                while ((j < nb_tot_fields) && (!is_vector))
                {
                    if (!strcmp(buffer, AllFields[j]))
                    {
                        is_vector = 1;
                        j = nb_tot_fields;
                        for (k = 0; k < 3; k++)
                            VectFieldArray[nbfields + k] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'J';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'K';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        for (k = MAX_FIELD_LENGHT - 1; k >= index + 4; k--)
                            buffer[k] = buffer[k - 4];
                        buffer[index] = '_';
                        buffer[index + 1] = 'I';
                        buffer[index + 2] = 'J';
                        buffer[index + 3] = 'K';
                        buffer[index + 4] = '_';
                        VectArray[nbvect] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectArray[(nbvect)++], buffer);
                    }
                    j++;
                }
                if (!is_vector)
                    buffer[index] = c;
                break;
            case 'K':
                buffer[index] = 'I';
                j = 0;
                while ((j < nb_tot_fields) && (!is_vector))
                {
                    if (!strcmp(buffer, AllFields[j]))
                    {
                        is_vector = 1;
                        j = nb_tot_fields;
                        for (k = 0; k < 3; k++)
                            VectFieldArray[nbfields + k] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'J';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        buffer[index] = 'K';
                        strcpy(VectFieldArray[nbfields++], buffer);
                        for (k = MAX_FIELD_LENGHT - 1; k >= index + 4; k--)
                            buffer[k] = buffer[k - 4];
                        buffer[index] = '_';
                        buffer[index + 1] = 'I';
                        buffer[index + 2] = 'J';
                        buffer[index + 3] = 'K';
                        buffer[index + 4] = '_';
                        VectArray[nbvect] = new char[MAX_FIELD_LENGHT + 1];
                        strcpy(VectArray[nbvect++], buffer);
                    }
                    j++;
                }
                if (!is_vector)
                    buffer[index] = c;
                break;
            default:
                ;
            } //switch
            index++;
            c = AllFields[i][index];

        } //while
        if (!is_vector && !is_found)
        {
            ScalArray[nbscal] = new char[MAX_FIELD_LENGHT + 1];
            strcpy(ScalArray[nbscal++], buffer);
        }
    } //for
}

Fields::~Fields()
{
    int i;
    for (i = 0; i < nbscal; i++)
        if (ScalArray[i])
            delete[] ScalArray[i];
    delete[] ScalArray;

    for (i = 0; i < nbvect; i++)
        if (VectArray[i])
            delete[] VectArray[i];
    delete[] VectArray;

    for (i = 0; i < 3; i++)
        if (VectFieldArray[i])
            delete[] VectFieldArray[i];

    delete[] VectFieldArray;
}

void Fields::getAddresses(const char *const **VectChoiceVal, const char *const **ScalChoiceVal, const char *const **VectFieldList, int *nb_vect_list, int *nb_scal_list, int *nb_field_list)
{

    *VectChoiceVal = VectArray;
    *ScalChoiceVal = ScalArray;
    *VectFieldList = VectFieldArray;
    /* int i;
    for (i=0;i<nbvect;i++)
       (*VectChoiceVal)[i] = VectArray[i];
    for (i=0;i<nbscal;i++)
       (*ScalChoiceVal)[i] = ScalArray[i];
    for (i=0;i<nbfields;i++)
       (*VectFieldList)[i] = VectFieldArray[i]; */

    *nb_vect_list = nbvect;
    *nb_scal_list = nbscal;
    *nb_field_list = nbfields;
}

void Fields::get_fields(char ***VectChoiceVal, char ***ScalChoiceVal, int *nb_vect, int *nb_scal)
{
    *VectChoiceVal = new char *[MAX_VECT_FIELDS];
    *ScalChoiceVal = new char *[MAX_SCAL_FIELDS];

    int i;
    for (i = 0; i < nbvect; i++)
    {
        (*VectChoiceVal)[i] = new char[strlen(VectArray[i]) + 1];
        strcpy((*VectChoiceVal)[i], VectArray[i]);
    }
    for (i = 0; i < nbscal; i++)
    {
        (*ScalChoiceVal)[i] = new char[strlen(ScalArray[i]) + 1];
        strcpy((*ScalChoiceVal)[i], ScalArray[i]);
    }

    *nb_vect = nbvect;
    *nb_scal = nbscal;
}
