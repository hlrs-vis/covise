/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "covise_msg.h"
#include <shm/covise_shm.h>

#include <util/coErr.h>
#include <util/byteswap.h>
#include <util/unixcompat.h>

/*
 $Log: covise_msg.C,v $
Revision 1.3  1994/03/23  18:07:03  zrf30125
Modifications for multiple Shared Memory segments have been finished
(not yet for Cray)

Revision 1.2  93/10/08  19:19:15  zrhk0125
some fixed type sizes with sizeof calls replaced

Revision 1.1  93/09/25  20:47:21  zrhk0125
Initial revision
*/

/***********************************************************************\
 **                                                                     **
 **   Message classes Routines                     Version: 1.1         **
 **                                                                     **
 **                                                                     **
 **   Description  : The basic message structure as well as ways to     **
 **                  initialize messages easily are provided.           **
 **                  Subclasses for special types of messages           **
 **                  can be introduced.                                 **
 **                                                                     **
 **   Classes      : Message, ShmMessage                                **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  15.04.93  Ver 1.0                                  **
 **                  15.04.93  Ver 1.1 adopted to shm-malloc handling   **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

using namespace covise;

namespace
{
struct Checker
{
    Checker()
    {
        assert((int)COVISE_MESSAGE_EMPTY == (int)Message::EMPTY);
        assert((int)COVISE_MESSAGE_SOCKET_CLOSED == (int)Message::SOCKET_CLOSED);
        assert((int)COVISE_MESSAGE_CLOSE_SOCKET == (int)Message::CLOSE_SOCKET);
        assert((int)COVISE_MESSAGE_HOSTID == (int)Message::HOSTID);
        assert((int)COVISE_MESSAGE_UI == (int)Message::UI);
        assert((int)COVISE_MESSAGE_RENDER == (int)Message::RENDER);

        assert((int)UNDEFINED == (int)Message::UNDEFINED);
        assert((int)STDINOUT == (int)Message::STDINOUT);
    }
};

Checker check;
}

#undef DEBUG
ShmMessage::ShmMessage(coShmPtr *ptr)
{
    if (ptr != NULL)
    {
        type = COVISE_MESSAGE_MALLOC_OK;
        int *idata = new int[2];
        idata[0] = ptr->shm_seq_no;
        idata[1] = ptr->offset;
        data = DataHandle((char *)idata, 2 * sizeof(int));
        //cerr << "Message coShmPtr: " << ptr->shm_seq_no <<
        //     ": " << ptr->offset << "\n";
    }
    else
    {
        type = COVISE_MESSAGE_MALLOC_FAILED;
        data = DataHandle();
    }
    //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
    print();
}

ShmMessage::ShmMessage(data_type *d, long *count, int no)
    : Message()
{
    int i, j;
#ifdef DEBUG
    char tmp_str[255];
#endif

    conn = NULL;
    type = COVISE_MESSAGE_SHM_MALLOC_LIST;
#ifdef DEBUG
    sprintf(tmp_str, "ShmMessage::ShmMessage");
    LOGINFO(tmp_str);
    sprintf(tmp_str, "no: %d", no);
    LOGINFO(tmp_str);
#endif
    data = DataHandle(no * (sizeof(data_type) + sizeof(long)));
    for (i = 0, j = 0; i < no; i++)
    {
        *(data_type *)(data.accessData() + j) = d[i];
#ifdef DEBUG
        sprintf(tmp_str, "data_type: %ld", *(data_type *)(data + j));
        LOGINFO(tmp_str);
#endif
        j += sizeof(data_type);
        *(long *)(data.accessData() + j) = count[i];
#ifdef DEBUG
        sprintf(tmp_str, "size: %ld", *(long *)(data + j));
        LOGINFO(tmp_str);
#endif
        j += sizeof(long);
    }
#ifdef DEBUG
    sprintf(tmp_str, "j: %d", j);
    LOGINFO(tmp_str);
#endif
}

ShmMessage::ShmMessage(char *name, int otype, data_type *d, long *count, int no)
    : Message()
{
    int i, j;
    char *tmp_data;
    int start_data;
#ifdef DEBUG
    char tmp_str[255];
#endif

    conn = NULL;
    type = COVISE_MESSAGE_NEW_OBJECT_SHM_MALLOC_LIST;
#ifdef DEBUG
    sprintf(tmp_str, "ShmMessage::ShmMessage");
    LOGINFO(tmp_str);
    sprintf(tmp_str, "no: %d", no);
    LOGINFO(tmp_str);
#endif
    int l = strlen(name) + 1;
    if (l % SIZEOF_ALIGNMENT)
        start_data = sizeof(long) + (l / SIZEOF_ALIGNMENT + 1) * SIZEOF_ALIGNMENT;
    else
        start_data = sizeof(long) + l;
    l = sizeof(long) + start_data + no * (sizeof(data_type) + sizeof(long));
    data = DataHandle(l);
    memset(data.accessData(), '\0', data.length());
    *(int *)data.accessData() = otype;
    strcpy(&data.accessData()[sizeof(int)], name);
    tmp_data = data.accessData() + start_data;
    for (i = 0, j = 0; i < no; i++)
    {
        *(data_type *)(tmp_data + j) = d[i];
#ifdef DEBUG
        sprintf(tmp_str, "data_type: %ld", *(data_type *)(tmp_data + j));
        LOGINFO(tmp_str);
#endif
        j += sizeof(data_type);
        *(long *)(tmp_data + j) = count[i];
#ifdef DEBUG
        sprintf(tmp_str, "size: %ld", *(long *)(tmp_data + j));
        LOGINFO(tmp_str);
#endif
        j += sizeof(long);
    }
#ifdef DEBUG
    sprintf(tmp_str, "j: %d", j);
    LOGINFO(tmp_str);
#endif
}

void CtlMessage::init_list()
{
    char *tmpchptr = NULL;
    char **tmpcharlist;
    char *tmpname = NULL;
    char *tmptype = NULL;
    char *tmpstring = NULL;
    int no, i, j, sel, len;

    m_name = NULL;
    char *p = data.accessData();
    tmpchptr = strsep(&p, "\n");
    if (tmpchptr)
    {
        m_name = new char[strlen(tmpchptr) + 1];
        strcpy(m_name, tmpchptr);
    }

    //   cerr << "   CtlMessage::init_list()  m_name " << m_name << endl;

    inst_no = NULL;
    tmpchptr = strsep(&p, "\n");
    if (tmpchptr)
    {
        inst_no = new char[strlen(tmpchptr) + 1];
        strcpy(inst_no, tmpchptr);
    }

    //   cerr << "   CtlMessage::init_list()  inst_no " << inst_no << endl;

    h_name = NULL;
    tmpchptr = strsep(&p, "\n");
    if (tmpchptr)
    {
        h_name = new char[strlen(tmpchptr) + 1];
        strcpy(h_name, tmpchptr);
    }

    //   cerr << "   CtlMessage::init_list()  h_name " << h_name << endl;

    no_of_objects = 0;
    tmpchptr = strsep(&p, "\n");
    if (tmpchptr)
    {
        no_of_objects = atoi(tmpchptr);
    }
    else
    {
        if (m_name)
            no_of_objects = 1;
    }

    no_of_params_in = 0;
    tmpchptr = strsep(&p, "\n");
    if (tmpchptr)
    {
        no_of_params_in = atoi(tmpchptr);
    }

    port_names = new char *[no_of_objects];
    object_names = new char *[no_of_objects];
    port_connected = new int[no_of_objects];
    object_types = new char *[no_of_objects];
    for (i = 0; i < no_of_objects; i++)
    {
        tmpchptr = strsep(&p, "\n");
        port_names[i] = new char[strlen(tmpchptr) + 1];
        strcpy(port_names[i], tmpchptr);
        //sprintf(tmp_str, "getting %s", port_names[i]);
        //LOGINFO( tmp_str);

        tmpchptr = strsep(&p, "\n");
        object_names[i] = new char[strlen(tmpchptr) + 1];
        strcpy(object_names[i], tmpchptr);
        //sprintf(tmp_str, "getting %s", object_names[i]);
        //LOGINFO( tmp_str);

        tmpchptr = strsep(&p, "\n");
        object_types[i] = new char[strlen(tmpchptr) + 1];
        strcpy(object_types[i], tmpchptr);
        //sprintf(tmp_str, "getting %s", object_types[i]);
        //LOGINFO( tmp_str);

        // new aw 25-07-2000 "CONNECTED" message
        tmpchptr = strsep(&p, "\n");
        port_connected[i] = (strcmp(tmpchptr, "CONNECTED") == 0);
    }

    params_in = new Param *[no_of_params_in];
    memset(params_in, 0, sizeof(Param *) * no_of_params_in);
    for (i = 0; i < no_of_params_in; i++)
    {
        tmpname = strsep(&p, "\n");
        tmptype = strsep(&p, "\n");
        no = atoi(strsep(&p, "\n"));

        if (!strcmp(tmptype, "FloatScalar"))
        {
            tmpchptr = strsep(&p, "\n");
            tmpstring = new char[strlen(tmpchptr) + 1];
            strcpy(tmpstring, tmpchptr);
            params_in[i] = (Param *)new ParamFloatScalar(tmpname, tmpstring);
            delete[] tmpstring;
        }

        else if (!strcmp(tmptype, "IntScalar"))
        {
            tmpchptr = strsep(&p, "\n");
            tmpstring = new char[strlen(tmpchptr) + 1];
            strcpy(tmpstring, tmpchptr);
            params_in[i] = (Param *)new ParamIntScalar(tmpname, tmpstring);
            delete[] tmpstring;
        }

        else if (!strcmp(tmptype, "FloatSlider"))
        {
            if (no != 3)
            {
                LOGINFO("wrong number of slider-parameters");
            }
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamFloatSlider(tmpname, no, tmpcharlist);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }

        else if (!strcmp(tmptype, "IntSlider"))
        {
            if (no != 3)
            {
                LOGINFO("wrong number of slider-parameters");
            }
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamIntSlider(tmpname, no, tmpcharlist);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }

        else if (!strcmp(tmptype, "String"))
        {
            tmpchptr = strsep(&p, "\n");
            tmpstring = new char[strlen(tmpchptr) + 1];
            strcpy(tmpstring, tmpchptr);
            params_in[i] = (Param *)new ParamString(tmpname, tmpstring);
            delete[] tmpstring;
        }

        else if (!strcmp(tmptype, "Material"))
        {
            tmpchptr = strsep(&p, "\n");
            tmpstring = new char[strlen(tmpchptr) + 1];
            strcpy(tmpstring, tmpchptr);
            params_in[i] = (Param *)new ParamMaterial(tmpname, tmpstring);
            delete[] tmpstring;
            for (int j = 0; j < no - 1; ++j)
                strsep(&p, "\n");
        }

        else if (!strcmp(tmptype, "Choice"))
        {
            sel = atoi(strsep(&p, "\n")); // the selected no
            tmpcharlist = new char *[no - 1];
            for (j = 0; j < no - 1; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamChoice(tmpname, no - 1, sel, tmpcharlist);
            for (j = 0; j < no - 1; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }

        else if (!strcmp(tmptype, "Color"))
        {
            if (no != 4)
            {
                LOGINFO("wrong number of color-parameters");
            }
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamColor(tmpname, no, tmpcharlist);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }

        else if (!strcmp(tmptype, "ColormapChoice"))
        {
            sel = atoi(strsep(&p, "\n")); // the selected no
            tmpcharlist = new char *[no - 1];
            for (j = 0; j < no - 1; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamColormapChoice(tmpname, no - 1, sel, tmpcharlist);
            for (j = 0; j < no - 1; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }

        else if (!strcmp(tmptype, "Colormap"))
        {
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamColormap(tmpname, no, tmpcharlist);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }

        else if (!strcmp(tmptype, "Browser"))
        {
            tmpchptr = strsep(&p, "\n");
            tmpstring = new char[strlen(tmpchptr) + 1];
            strcpy(tmpstring, tmpchptr);
            params_in[i] = (Param *)new ParamBrowser(tmpname, tmpstring);
            delete[] tmpstring;
        }

        else if (!strcmp(tmptype, "Boolean"))
        {
            tmpchptr = strsep(&p, "\n");
            tmpstring = new char[strlen(tmpchptr) + 1];
            strcpy(tmpstring, tmpchptr);
            params_in[i] = (Param *)new ParamBoolean(tmpname, tmpstring);
            delete[] tmpstring;
        }

        else if (!strcmp(tmptype, "FloatVector"))
        {
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamFloatVector(tmpname, no, tmpcharlist);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }

        else if (!strcmp(tmptype, "IntVector"))
        {
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamIntVector(tmpname, no, tmpcharlist);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }

        else if (!strcmp(tmptype, "Text"))
        {
            len = atoi(strsep(&p, "\n"));
            tmpcharlist = new char *[no - 1];
            for (j = 0; j < no - 1; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamText(tmpname, tmpcharlist, no - 1, len);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }

        else if (!strcmp(tmptype, "Timer"))
        {
            if (no != 3)
            {
                LOGINFO("wrong number of timer-parameters");
            }
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamTimer(tmpname, no, tmpcharlist);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }

        else if (!strcmp(tmptype, "Passwd"))
        {
            if (no != 3)
            {
                LOGINFO("wrong number of timer-parameters");
            }
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamPasswd(tmpname, no, tmpcharlist);
            delete[] tmpcharlist;
        }

        else if (!strcmp(tmptype, "BrowserFilter"))
        {
            // define dummy Param
            params_in[i] = new Param(tmpname, STRING, 1);
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
            }
        }

        else
        {
            std::cerr << "Unsupported parameter type: " << tmptype << ", name: " << tmpname << std::endl;
            assert("Unsupported parameter type" == 0);
        }
    }
}

char *CtlMessage::get_object_name(const char *port)
{
    int i;

    for (i = 0; i < no_of_objects && strcmp(port_names[i], port); i++)
        ;
    if (i == no_of_objects)
        return NULL;
    else
        return object_names[i];
}

char *CtlMessage::getObjectType(const char *port)
{
    int i;

    for (i = 0; i < no_of_objects && strcmp(port_names[i], port); i++)
        ;
    if (i == no_of_objects)
        return NULL;
    else
        return object_types[i];
}

// check whether port is connected
int CtlMessage::is_port_connected(const char *port_name)
{
    int i = 0;

    // find port with this name
    while (i < no_of_objects && strcmp(port_names[i], port_name))
        i++;
    if (i == no_of_objects)
        return 0;
    else
        return port_connected[i];
}

//// ---------- GET PARAMETERS ----------

int CtlMessage::get_scalar_param(const char *param_name, long *value)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *value = atol(((ParamIntScalar *)params_in[i])->list);
        return ((ParamIntScalar *)params_in[i])->no;
    }
}

int CtlMessage::get_scalar_param(const char *param_name, float *value)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *value = (float)atof(((ParamFloatScalar *)params_in[i])->list);
        return ((ParamFloatScalar *)params_in[i])->no;
    }
}

int CtlMessage::get_vector_param(const char *param_name, int pos, long *list)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *list = atol(((ParamIntVector *)params_in[i])->list[pos]);
        return ((ParamIntScalar *)params_in[i])->no;
    }
}

int CtlMessage::get_vector_param(const char *param_name, int pos, float *list)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *list = (float)atof(((ParamFloatVector *)params_in[i])->list[pos]);
        return ((ParamFloatVector *)params_in[i])->no;
    }
}

int CtlMessage::get_string_param(const char *param_name, char **string)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *string = ((ParamString *)params_in[i])->list;
        return ((ParamString *)params_in[i])->no;
    }
}

int CtlMessage::get_browser_param(const char *param_name, char **file)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *file = ((ParamBrowser *)params_in[i])->list;
        return ((ParamBrowser *)params_in[i])->no;
    }
}

int CtlMessage::get_text_param(const char *param_name, char ***text, int *line_num)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *text = ((ParamText *)params_in[i])->list;
        *line_num = ((ParamText *)params_in[i])->get_line_number();
        return ((ParamText *)params_in[i])->no;
    }
}

int CtlMessage::get_boolean_param(const char *param_name, int *value)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        if (strcmp(((ParamBoolean *)params_in[i])->list, "TRUE") == 0 || strcmp(((ParamBoolean *)params_in[i])->list, "1") == 0)
            *value = 1;
        else
            *value = 0;

        return ((ParamBoolean *)params_in[i])->no;
    }
}

int CtlMessage::get_slider_param(const char *param_name, float *min, float *max, float *val)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
    {
        *min = *max = *val = (float)0.0;
        return 0;
    }
    else
    {
        *min = (float)atof(((ParamFloatSlider *)params_in[i])->list[0]);
        *max = (float)atof(((ParamFloatSlider *)params_in[i])->list[1]);
        *val = (float)atof(((ParamFloatSlider *)params_in[i])->list[2]);
        return ((ParamFloatSlider *)params_in[i])->no;
    }
}

int CtlMessage::get_slider_param(const char *param_name, long *min, long *max, long *val)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
    {
        *min = *max = *val = 0;
        return 0;
    }
    else
    {
        *min = atol(((ParamIntSlider *)params_in[i])->list[0]);
        *max = atol(((ParamIntSlider *)params_in[i])->list[1]);
        *val = atol(((ParamIntSlider *)params_in[i])->list[2]);
        return ((ParamIntSlider *)params_in[i])->no;
    }
}

int CtlMessage::get_timer_param(const char *param_name, long *start, long *delta, long *state)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
    {
        *start = *delta = *state = 0;
        return 0;
    }
    else
    {
        *start = atol(((ParamTimer *)params_in[i])->list[0]);
        *delta = atol(((ParamTimer *)params_in[i])->list[1]);
        *state = atol(((ParamTimer *)params_in[i])->list[2]);
        return ((ParamTimer *)params_in[i])->no;
    }
}

int CtlMessage::get_passwd_param(const char *param_name, char **host, char **user,
                                 char **passwd)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *host = ((ParamPasswd *)params_in[i])->list[0];
        *user = ((ParamPasswd *)params_in[i])->list[1];
        *passwd = ((ParamPasswd *)params_in[i])->list[2];
        return ((ParamPasswd *)params_in[i])->no;
    }
}

int CtlMessage::get_choice_param(const char *param_name, char **string)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *string = ((ParamChoice *)params_in[i])->list[(((ParamChoice *)params_in[i])->sel) - 1];
        return ((ParamChoice *)params_in[i])->no;
    }
}

int CtlMessage::get_choice_param(const char *param_name, int *sel)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *sel = ((ParamChoice *)params_in[i])->sel;
        return ((ParamChoice *)params_in[i])->no;
    }
}

int CtlMessage::get_colormapchoice_param(const char *param_name, char **string)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *string = ((ParamColormapChoice *)params_in[i])->list[(((ParamColormapChoice *)params_in[i])->sel) - 1];
        return ((ParamColormapChoice *)params_in[i])->no;
    }
}

int CtlMessage::get_material_param(const char *param_name, char **string)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *string = ((ParamMaterial *)params_in[i])->list;
        return ((ParamMaterial *)params_in[i])->no;
    }
}

int CtlMessage::get_colormapchoice_param(const char *param_name, int *sel)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *sel = ((ParamColormapChoice *)params_in[i])->sel;
        return ((ParamColormapChoice *)params_in[i])->no;
    }
}

int CtlMessage::get_colormap_param(const char *param_name, float *min, float *max, int *len, colormap_type *type)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
    {
        *min = *max = (float)0.0;
        *len = 0;
        *type = RGBAX;
        return 0;
    }
    else
    {
        *min = (float)atof(((ParamColormap *)params_in[i])->list[0]);
        *max = (float)atof(((ParamColormap *)params_in[i])->list[1]);
        if (!strcmp(((ParamColormap *)params_in[i])->list[2], "RGBAX"))
        {
            *len = atoi(((ParamColormap *)params_in[i])->list[3]);
            *type = RGBAX;
        }
        else if (!strcmp(((ParamColormap *)params_in[i])->list[2], "VIRVO"))
        {
            *type = VIRVO;
        }
        else
        {
            std::cerr << "invalid colormap type " << ((ParamColormap *)params_in[i])->list[3] << std::endl;
            *type = RGBAX;
        }
        return ((ParamColormap *)params_in[i])->no;
    }
}

int CtlMessage::get_color_param(const char *param_name, float *r, float *g, float *b, float *a)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
    {
        *r = *g = *b = 0.5;
        *a = 1.0;
        return 0;
    }
    else
    {
        *r = (float)atof(((ParamColor *)params_in[i])->list[0]);
        *g = (float)atof(((ParamColor *)params_in[i])->list[1]);
        *b = (float)atof(((ParamColor *)params_in[i])->list[2]);
        *a = (float)atof(((ParamColor *)params_in[i])->list[3]);
        return ((ParamColor *)params_in[i])->no;
    }
}

int CtlMessage::set_vector_param(const char *pname, int len, long *val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamIntVector(pname, len, val);
    no_of_params_out++;
    return ((ParamIntVector *)params_out[i])->no;
}

int CtlMessage::set_vector_param(const char *pname, int len, float *val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamFloatVector(pname, len, val);
    no_of_params_out++;
    return ((ParamFloatVector *)params_out[i])->no;
}

int CtlMessage::set_boolean_param(const char *pname, int val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamBoolean(pname, val);
    no_of_params_out++;
    return ((ParamBoolean *)params_out[i])->no;
}

int CtlMessage::set_scalar_param(const char *pname, long val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamIntScalar(pname, val);
    no_of_params_out++;
    return ((ParamIntScalar *)params_out[i])->no;
}

int CtlMessage::set_scalar_param(const char *pname, float val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamFloatScalar(pname, val);
    no_of_params_out++;
    return ((ParamFloatScalar *)params_out[i])->no;
}

int CtlMessage::set_choice_param(const char *pname, int len, char **list, int pos)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamChoice(pname, len, list, pos);
    no_of_params_out++;
    return ((ParamChoice *)params_out[i])->no;
}

int CtlMessage::set_string_param(const char *pname, char *val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamString(pname, val);
    no_of_params_out++;
    return ((ParamString *)params_out[i])->no;
}

int CtlMessage::set_browser_param(const char *pname, char *file, char * /*wildcard*/)
{
    int i = 0;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;

    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamBrowser(pname, file);
    no_of_params_out++;
    return ((ParamBrowser *)params_out[i])->no;
}

int CtlMessage::set_text_param(const char *pname, char *text, int linenum)
{
    int i, j;
    char *tmpchptr;
    char **tmpcharlist;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    tmpcharlist = new char *[linenum];

    char *p = text;
    for (j = 0; j < linenum - 1; j++)
    {
        tmpchptr = strsep(&p, "\n");
        tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
        strcpy(tmpcharlist[j], tmpchptr);
    }
    params_out[i] = (Param *)new ParamText(pname, tmpcharlist, linenum, (int)strlen(text));
    delete[] tmpcharlist;
    no_of_params_out++;
    return ((ParamText *)params_out[i])->no;
}

int CtlMessage::set_slider_param(const char *pname, float min, float max, float val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamFloatSlider(pname, min, max, val);
    no_of_params_out++;
    return ((ParamFloatSlider *)params_out[i])->no;
}

int CtlMessage::set_slider_param(const char *pname, long min, long max, long val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamIntSlider(pname, min, max, val);
    no_of_params_out++;
    return ((ParamIntSlider *)params_out[i])->no;
}

int CtlMessage::set_timer_param(const char *pname, long start, long delta, long state)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamTimer(pname, start, delta, state);
    no_of_params_out++;
    return ((ParamTimer *)params_out[i])->no;
}

int CtlMessage::set_passwd_param(const char *pname, char *host, char *user, char *passwd)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamPasswd(pname, host, user, passwd);
    no_of_params_out++;
    return ((ParamPasswd *)params_out[i])->no;
}

int CtlMessage::set_save_object(const char *sname)
{
    char **tmpptr;
    int i;

    if (no_of_save_objects % 10 == 0)
    {
        tmpptr = new char *[no_of_save_objects + 10];
        for (i = 0; i < no_of_save_objects; i++)
            tmpptr[i] = save_names[i];
        save_names = tmpptr;
        // delete [] save_names;
    }

    save_names[no_of_save_objects] = new char[strlen(sname) + 1];
    strcpy(save_names[no_of_save_objects], sname);
    no_of_save_objects++;
    return 1;
}

int CtlMessage::set_release_object(const char *sname)
{
    char **tmpptr;
    int i;

    if (no_of_release_objects % 10 == 0)
    {
        tmpptr = new char *[no_of_release_objects + 10];
        for (i = 0; i < no_of_release_objects; i++)
            tmpptr[i] = release_names[i];
        release_names = tmpptr;
        // delete [] release_names;
    }

    release_names[no_of_release_objects] = new char[strlen(sname) + 1];
    strcpy(release_names[no_of_release_objects], sname);
    no_of_release_objects++;
    return 1;
}

int CtlMessage::create_finall_message()
{
    int i, j;
    size_t size;
    char numbuf[15];

    size = 1; // final '\0'
    size += strlen(m_name) + 1;
    size += strlen(inst_no) + 1;
    size += strlen(h_name) + 1;
    size += 5; // no_of_params_out
    size += 5; // no_of_save_objects
    size += 5; // no_of_release_objects

    for (i = 0; i < no_of_params_out; i++)
    {
        switch (params_out[i]->type)
        {

        case STRING:
            size += strlen(params_out[i]->name) + 1 + strlen("String") + 1 + 5 + 1 + strlen(((ParamString *)params_out[i])->list) + 1;
            break;

        case TEXT:
            size += strlen(params_out[i]->name) + 1 + strlen("Text") + 1 + 5 + 1 + 10 + 1;
            for (j = 0; j < (params_out[i]->no) - 1; j++)
                size += strlen(((ParamText *)params_out[i])->list[j]) + 1;
            break;

        case COVISE_BOOLEAN:
            size += strlen(params_out[i]->name) + 1 + strlen("Boolean") + 1 + 5 + 1 + strlen(((ParamBoolean *)params_out[i])->list) + 1;
            break;

        case BROWSER:
            size += strlen(params_out[i]->name) + 1 + strlen("Browser") + 1 + 5 + 1 + strlen(((ParamBrowser *)params_out[i])->list) + 1;
            break;

        case FLOAT_SLIDER:
            size += strlen(params_out[i]->name) + 1 + strlen("FloatSlider") + 1 + 5 + 1 + strlen(((ParamFloatSlider *)params_out[i])->list[0]) + 1 + strlen(((ParamFloatSlider *)params_out[i])->list[1]) + 1 + strlen(((ParamFloatSlider *)params_out[i])->list[2]) + 1;
            break;

        case INT_SLIDER:
            size += strlen(params_out[i]->name) + 1 + strlen("IntSlider") + 1 + 5 + 1 + strlen(((ParamIntSlider *)params_out[i])->list[0]) + 1 + strlen(((ParamIntSlider *)params_out[i])->list[1]) + 1 + strlen(((ParamIntSlider *)params_out[i])->list[2]) + 1;
            break;

        case COLORMAP_MSG:
            size += strlen(params_out[i]->name) + 1 + strlen("Colormap") + 1 + 5 + 1;
            for (j = 0; j < params_out[i]->no; j++)
                size += strlen(((ParamColormap *)params_out[i])->list[j]) + 1;
            break;

        case CHOICE:
            size += strlen(params_out[i]->name) + 1 + strlen("Choice") + 1 + 5 + 1 + 5 + 1;
            for (j = 0; j < (params_out[i]->no); j++)
                size += strlen(((ParamChoice *)params_out[i])->list[j]) + 1;
            break;

        case FLOAT_VECTOR:
            size += strlen(params_out[i]->name) + 1 + strlen("FloatVector") + 1 + 5 + 1;
            for (j = 0; j < params_out[i]->no; j++)
                size += strlen(((ParamFloatVector *)params_out[i])->list[j]) + 1;
            break;

        case INT_VECTOR:
            size += strlen(params_out[i]->name) + 1 + strlen("IntVector") + 1 + 5 + 1;
            for (j = 0; j < params_out[i]->no; j++)
                size += strlen(((ParamIntVector *)params_out[i])->list[j]) + 1;
            break;

        case FLOAT_SCALAR:
            size += strlen(params_out[i]->name) + 1 + strlen("FloatScalar") + 1 + 5 + 1 + strlen(((ParamFloatScalar *)params_out[i])->list) + 1;
            break;

        case INT_SCALAR:
            size += strlen(params_out[i]->name) + 1 + strlen("IntScalar") + 1 + 5 + 1 + strlen(((ParamIntScalar *)params_out[i])->list) + 1;
            break;

        case TIMER:
            size += strlen(params_out[i]->name) + 1 + strlen("Timer") + 1 + 5 + 1 + strlen(((ParamTimer *)params_out[i])->list[0]) + 1 + strlen(((ParamTimer *)params_out[i])->list[1]) + 1 + strlen(((ParamTimer *)params_out[i])->list[2]) + 1;
            break;

        case PASSWD:
            size += strlen(params_out[i]->name) + 1 + strlen("Passwd") + 1 + 5 + 1 + strlen(((ParamString *)params_out[0])->list) + 1 + strlen(((ParamString *)params_out[1])->list) + 1 + strlen(((ParamString *)params_out[2])->list) + 1;
            break;
        }
    }
    for (i = 0; i < no_of_save_objects; i++)
        size += strlen(save_names[i]) + 1;
    for (i = 0; i < no_of_release_objects; i++)
        size += strlen(release_names[i]) + 1;

    char *d = new char[size];

    strcpy(d, m_name);
    strcat(d, "\n");

    strcat(d, inst_no);
    strcat(d, "\n");

    strcat(d, h_name);
    strcat(d, "\n");

    sprintf(numbuf, "%d\n", no_of_params_out);
    strcat(d, numbuf);
    sprintf(numbuf, "%d\n", no_of_save_objects);
    strcat(d, numbuf);
    sprintf(numbuf, "%d\n", no_of_release_objects);
    strcat(d, numbuf);

    for (i = 0; i < no_of_params_out; i++)
    {
        strcat(d, params_out[i]->name);
        strcat(d, "\n");

        switch (params_out[i]->type)
        {

        case STRING:
            strcat(d, "String");
            strcat(data.accessData(), "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(d, numbuf);
            strcat(d, ((ParamString *)params_out[i])->list);
            strcat(d, "\n");
            break;

        case TEXT:
            strcpy(d, "Text");
            strcat(d, "\n");
            // no of tokens
            sprintf(numbuf, "%d\n", (params_out[i]->no) + 1);
            strcat(d, numbuf);
            // string length
            sprintf(numbuf, "%d\n", ((ParamText *)params_out[i])->length);
            strcat(d, numbuf);
            for (j = 0; j < ((ParamText *)params_out[i])->line_num; j++)
            {
                strcat(d, ((ParamText *)params_out[i])->list[j]);
                strcat(d, "\n");
            }
            break;

        case COVISE_BOOLEAN:
            strcat(d, "Boolean");
            strcat(d, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(d, numbuf);
            strcat(d, ((ParamBoolean *)params_out[i])->list);
            strcat(d, "\n");
            break;

        case BROWSER:
            strcat(d, "Browser");
            strcat(d, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(d, numbuf);
            strcat(d, ((ParamBrowser *)params_out[i])->list);
            strcat(d, "\n");
            break;

        case FLOAT_SLIDER:
            strcat(d, "FloatSlider");
            strcat(d, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(d, numbuf);
            strcat(d, ((ParamFloatSlider *)params_out[i])->list[0]);
            strcat(d, "\n");
            strcat(d, ((ParamFloatSlider *)params_out[i])->list[1]);
            strcat(d, "\n");
            strcat(d, ((ParamFloatSlider *)params_out[i])->list[2]);
            strcat(d, "\n");
            break;

        case INT_SLIDER:
            strcat(d, "IntSlider");
            strcat(d, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(d, numbuf);
            strcat(d, ((ParamIntSlider *)params_out[i])->list[0]);
            strcat(d, "\n");
            strcat(d, ((ParamIntSlider *)params_out[i])->list[1]);
            strcat(d, "\n");
            strcat(d, ((ParamIntSlider *)params_out[i])->list[2]);
            strcat(d, "\n");
            break;

        case COLORMAP_MSG:
            strcat(d, "Colormap");
            strcat(d, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(d, numbuf);
            for (j = 0; j < params_out[i]->no; j++)
            {
                strcat(d, ((ParamColormap *)params_out[i])->list[j]);
                strcat(d, "\n");
            }
            break;

        case COLOR_MSG:
            strcat(d, "Color");
            strcat(d, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(d, numbuf);
            for (j = 0; j < params_out[i]->no; j++)
            {
                strcat(d, ((ParamColormap *)params_out[i])->list[j]);
                strcat(d, "\n");
            }
            break;

        case CHOICE:
            strcat(d, "Choice");
            strcat(d, "\n");
            sprintf(numbuf, "%d\n", (params_out[i]->no) + 1);
            strcat(d, numbuf);
            sprintf(numbuf, "%d\n", ((ParamChoice *)params_out[i])->sel);
            strcat(d, numbuf);
            for (j = 0; j < params_out[i]->no; j++)
            {
                strcat(d, ((ParamChoice *)params_out[i])->list[j]);
                strcat(d, "\n");
            }

            break;

        case FLOAT_VECTOR:
            strcat(d, "FloatVector");
            strcat(d, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(d, numbuf);
            for (j = 0; j < params_out[i]->no; j++)
            {
                strcat(d, ((ParamFloatVector *)params_out[i])->list[j]);
                strcat(d, "\n");
            }
            break;

        case INT_VECTOR:
            strcat(d, "IntVector");
            strcat(d, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(d, numbuf);
            for (j = 0; j < params_out[i]->no; j++)
            {
                strcat(d, ((ParamIntVector *)params_out[i])->list[j]);
                strcat(d, "\n");
            }
            break;

        case FLOAT_SCALAR:
            strcat(d, "FloatScalar");
            strcat(d, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(d, numbuf);
            strcat(d, ((ParamFloatScalar *)params_out[i])->list);
            strcat(d, "\n");
            break;

        case INT_SCALAR:
            strcat(d, "IntScalar");
            strcat(d, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(d, numbuf);
            strcat(d, ((ParamIntScalar *)params_out[i])->list);
            strcat(d, "\n");
            break;

        case TIMER:
            strcat(d, "Timer");
            strcat(d, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(d, numbuf);
            strcat(d, ((ParamTimer *)params_out[i])->list[0]);
            strcat(d, "\n");
            strcat(d, ((ParamTimer *)params_out[i])->list[1]);
            strcat(d, "\n");
            strcat(d, ((ParamTimer *)params_out[i])->list[2]);
            strcat(d, "\n");
            break;

        case PASSWD:
            strcat(d, "Passwd");
            strcat(d, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(d, numbuf);
            strcat(d, ((ParamPasswd *)params_out[i])->list[0]);
            strcat(d, "\n");
            strcat(d, ((ParamPasswd *)params_out[i])->list[1]);
            strcat(d, "\n");
            strcat(d, ((ParamPasswd *)params_out[i])->list[2]);
            strcat(d, "\n");
            break;
        }
    }
    for (i = 0; i < no_of_save_objects; i++)
    {
        strcat(d, save_names[i]);
        strcat(d, "\n");
    }
    for (i = 0; i < no_of_release_objects; i++)
    {
        strcat(d, release_names[i]);
        strcat(d, "\n");
    }

    type = COVISE_MESSAGE_FINALL;
    data = DataHandle(d, strlen(d) + 1);

    return 1;
}

int CtlMessage::create_finpart_message()
{
    int size;

    size = 1; // final '\0'
    size += int(strlen(m_name) + 1);
    size += int(strlen(inst_no) + 1);
    size += int(strlen(h_name) + 1);

    char *d = new char[size];

    strcpy(d, m_name);
    strcat(d, "\n");

    strcat(d, inst_no);
    strcat(d, "\n");

    strcat(d, h_name);
    strcat(d, "\n");

    type = COVISE_MESSAGE_FINPART;
    data = DataHandle(d, strlen(d) + 1);

    return 1;
}

CtlMessage::~CtlMessage()
{
    int i;

    delete[] m_name;
    delete[] h_name;

    for (i = 0; i < no_of_objects; i++)
    {
        delete[] port_names[i];
        delete[] object_names[i];
        delete[] object_types[i];
    }
    delete[] port_names;
    delete[] object_names;
    delete[] object_types;

    for (i = 0; i < no_of_save_objects; i++)
    {
        delete[] save_names[i];
    }
    delete[] save_names;

    for (i = 0; i < no_of_release_objects; i++)
    {
        delete[] release_names[i];
    }
    delete[] release_names;

    for (i = 0; i < no_of_params_in; i++)
    {
        delete params_in[i];
    }
    delete[] params_in;

    for (i = 0; i < no_of_params_out; i++)
    {
        delete params_out[i];
    }
    delete[] params_out;

    delete[] inst_no;
}

char *RenderMessage::get_part(char *chdata)
{
    static char *data_ptr = NULL;
    char *part, *part_ptr;
    int i;

    if (chdata)
        data_ptr = chdata;
    if (data_ptr == NULL)
    {
        LOGINFO("no data to get part of");
        return NULL;
    }
    for (i = 0; data_ptr[i] != '\n'; i++)
        ;
    part_ptr = part = new char[i + 1];
    while ((*part_ptr++ = *data_ptr++) != '\n')
        ;
    *(part_ptr - 1) = '\0';
    return part;
}

void RenderMessage::init_list()
{
    char *tmpchptr;
    int i;
    //char tmp_str[255];

    m_name = get_part(data.accessData());
    inst_no = get_part();
    h_name = get_part();
    no_of_objects = atoi(get_part());

    object_names = new char *[no_of_objects];

    for (i = 0; i < no_of_objects; i++)
    {
        tmpchptr = get_part();
        object_names[i] = new char[strlen(tmpchptr) + 1];
        strcpy(object_names[i], tmpchptr);
        //sprintf(tmp_str, "getting %s", object_names[i]);
        //LOGINFO( tmp_str);
    }
}

RenderMessage::~RenderMessage()
{
    int i;

    for (i = 0; i < no_of_objects; i++)
    {
        delete[] object_names[i];
    }
    delete[] object_names;
}

ParamFloatScalar::ParamFloatScalar(const char *na, char *l)
    : Param(na, FLOAT_SCALAR, 1)
{
    list = new char[strlen(l) + 1];
    strcpy(list, l);
}

ParamFloatScalar::ParamFloatScalar(const char *na, float val)
    : Param(na, FLOAT_SCALAR, 1)
{
    char *buf = new char[255];
    sprintf(buf, "%f", val);
    list = new char[strlen(buf) + 1];
    strcpy(list, buf);
    delete[] buf;
}

ParamIntScalar::ParamIntScalar(const char *na, char *l)
    : Param(na, INT_SCALAR, 1)
{
    list = new char[strlen(l) + 1];
    strcpy(list, l);
}

ParamIntScalar::ParamIntScalar(const char *na, long val)
    : Param(na, INT_SCALAR, 1)
{
    char *buf = new char[255];
    sprintf(buf, "%ld", val);
    list = new char[strlen(buf) + 1];
    strcpy(list, buf);
    delete[] buf;
}

ParamIntScalar::~ParamIntScalar()
{
    delete[] list;
}

ParamFloatVector::ParamFloatVector(const char *na, int num, char **l)
    : Param(na, FLOAT_VECTOR, num)
{
    list = new char *[num];
    for (int j = 0; j < num; j++)
    {
        list[j] = new char[strlen(l[j]) + 1];
        strcpy(list[j], l[j]);
    }
}

ParamFloatVector::ParamFloatVector(const char *na, int num, float *l)
    : Param(na, FLOAT_VECTOR, num)
{
    char *buf = new char[255];
    list = new char *[num];
    for (int j = 0; j < num; j++)
    {
        sprintf(buf, "%f", *(l + j));
        list[j] = new char[strlen(buf) + 1];
        strcpy(list[j], buf);
    }
    delete[] buf;
}

ParamFloatVector::~ParamFloatVector()
{
    for (int j = 0; j < no_of_items(); j++)
    {
        delete[] list[j];
    }
    delete[] list;
}

ParamIntVector::ParamIntVector(const char *na, int num, char **l)
    : Param(na, INT_VECTOR, num)
{
    list = new char *[num];
    for (int j = 0; j < num; j++)
    {
        list[j] = new char[strlen(l[j]) + 1];
        strcpy(list[j], l[j]);
    }
}

ParamIntVector::ParamIntVector(const char *na, int num, long *l)
    : Param(na, INT_VECTOR, num)
{
    char *buf = new char[255];
    list = new char *[num];
    for (int j = 0; j < num; j++)
    {
        sprintf(buf, "%ld", *(l + j));
        list[j] = new char[strlen(buf) + 1];
        strcpy(list[j], buf);
    }
    delete[] buf;
}

ParamIntVector::~ParamIntVector()
{
    for (int j = 0; j < no_of_items(); j++)
    {
        delete[] list[j];
    }
    delete[] list;
}

ParamBrowser::ParamBrowser(const char *na, char *l)
    : Param(na, BROWSER, 1)
{
    list = new char[strlen(l) + 1];
    strcpy(list, l);
}

ParamBrowser::~ParamBrowser()
{
    delete[] list;
}

ParamString::ParamString(const char *na, char *l)
    : Param(na, STRING, 1)
{
    list = new char[strlen(l) + 1];
    strcpy(list, l);
}

ParamString::~ParamString()
{
    delete[] list;
}

ParamText::ParamText(const char *na, char **l, int lineno, int len)
    : Param(na, TEXT, lineno)
{
    for (int j = 0; j < lineno; j++)
    {
        list[j] = new char[strlen(l[j]) + 1];
        strcpy(list[j], l[j]);
    }
    line_num = lineno;
    length = len;
}

ParamText::~ParamText()
{
    for (int j = 0; j < no_of_items(); j++)
    {
        delete[] list[j];
    }
    delete[] list;
}

ParamBoolean::ParamBoolean(const char *na, char *l)
    : Param(na, COVISE_BOOLEAN, 1)
{
    list = new char[strlen(l) + 1];
    strcpy(list, l);
}

ParamBoolean::ParamBoolean(const char *na, int val)
    : Param(na, COVISE_BOOLEAN, 1)
{
    if (val == 0)
    {
        list = new char[strlen("FALSE") + 1];
        strcpy(list, "FALSE");
    }
    else
    {
        list = new char[strlen("TRUE") + 1];
        strcpy(list, "TRUE");
    }
}

ParamBoolean::~ParamBoolean()
{
    delete[] list;
}

ParamFloatSlider::ParamFloatSlider(const char *na, int num, char **l)
    : Param(na, FLOAT_SLIDER, num)
{
    list = new char *[num];
    for (int j = 0; j < num; j++)
    {
        list[j] = new char[strlen(l[j]) + 1];
        strcpy(list[j], l[j]);
    }
}

ParamFloatSlider::ParamFloatSlider(const char *na, float min, float max, float val)
    : Param(na, FLOAT_SLIDER, 3)
{
    char *buf = new char[255];
    list = new char *[3];
    sprintf(buf, "%f", min);
    list[0] = new char[strlen(buf) + 1];
    strcpy(list[0], buf);
    sprintf(buf, "%f", max);
    list[1] = new char[strlen(buf) + 1];
    strcpy(list[1], buf);
    sprintf(buf, "%f", val);
    list[2] = new char[strlen(buf) + 1];
    strcpy(list[2], buf);
    delete[] buf;
}

ParamFloatSlider::~ParamFloatSlider()
{
    for (int j = 0; j < no_of_items(); j++)
        delete[] list[j];
    delete[] list;
}

ParamIntSlider::ParamIntSlider(const char *na, int num, char **l)
    : Param(na, INT_SLIDER, num)
{
    list = new char *[num];
    for (int j = 0; j < num; j++)
    {
        list[j] = new char[strlen(l[j]) + 1];
        strcpy(list[j], l[j]);
    }
}

ParamIntSlider::ParamIntSlider(const char *na, long min, long max, long val)
    : Param(na, INT_SLIDER, 3)
{
    char *buf = new char[255];
    list = new char *[3];
    sprintf(buf, "%ld", min);
    list[0] = new char[strlen(buf) + 1];
    strcpy(list[0], buf);
    sprintf(buf, "%ld", max);
    list[1] = new char[strlen(buf) + 1];
    strcpy(list[1], buf);
    sprintf(buf, "%ld", val);
    list[2] = new char[strlen(buf) + 1];
    strcpy(list[2], buf);
    delete[] buf;
}

ParamIntSlider::~ParamIntSlider()
{
    for (int j = 0; j < no_of_items(); j++)
        delete[] list[j];
    delete[] list;
}

ParamTimer::ParamTimer(const char *na, int num, char **l)
    : Param(na, TIMER, num)
{
    list = new char *[num];
    for (int j = 0; j < num; j++)
    {
        list[j] = new char[strlen(l[j]) + 1];
        strcpy(list[j], l[j]);
    }
}

ParamTimer::ParamTimer(const char *na, long start, long delta, long state)
    : Param(na, TIMER, 3)
{
    char *buf = new char[255];
    list = new char *[3];
    sprintf(buf, "%ld", start);
    list[0] = new char[strlen(buf) + 1];
    strcpy(list[0], buf);
    sprintf(buf, "%ld", delta);
    list[1] = new char[strlen(buf) + 1];
    strcpy(list[1], buf);
    sprintf(buf, "%ld", state);
    list[2] = new char[strlen(buf) + 1];
    strcpy(list[2], buf);
    delete[] buf;
}

ParamTimer::~ParamTimer()
{
    for (int j = 0; j < no_of_items(); j++)
        delete[] list[j];
    delete[] list;
}

ParamPasswd::ParamPasswd(const char *na, int num, char **l)
    : Param(na, PASSWD, num)
{
    list = new char *[num];
    for (int j = 0; j < num; j++)
    {
        list[j] = new char[strlen(l[j]) + 1];
        strcpy(list[j], l[j]);
    }
}

ParamPasswd::ParamPasswd(const char *na, char *host, char *user, char *passwd)
    : Param(na, PASSWD, 3)
{
    list = new char *[3];
    list[0] = new char[strlen(host) + 1];
    strcpy(list[0], host);
    list[1] = new char[strlen(user) + 1];
    strcpy(list[1], user);
    list[2] = new char[strlen(passwd) + 1];
    strcpy(list[2], passwd);
}

ParamPasswd::~ParamPasswd()
{
    for (int j = 0; j < no_of_items(); j++)
        delete[] list[j];
    delete[] list;
}

ParamChoice::ParamChoice(const char *na, int num, int s, char **l)
    : Param(na, CHOICE, num)
{
    sel = s;
    list = new char *[num];
    for (int j = 0; j < num; j++)
    {
        list[j] = new char[strlen(l[j]) + 1];
        strcpy(list[j], l[j]);
    }
}

ParamChoice::ParamChoice(const char *na, int num, char **l, int s)
    : Param(na, CHOICE, num)
{
    sel = s;
    list = new char *[num];
    for (int j = 0; j < num; j++)
    {
        list[j] = new char[strlen(l[j]) + 1];
        strcpy(list[j], l[j]);
    }
}

ParamChoice::~ParamChoice()
{
    for (int j = 0; j < no_of_items(); j++)
    {
        delete[] list[j];
    }
    delete[] list;
}

ParamColormapChoice::ParamColormapChoice(const char *na, int num, int s, char **l)
    : Param(na, COLORMAPCHOICE_MSG, num)
{
    sel = s;
    list = new char *[num];
    for (int j = 0; j < num; j++)
    {
        list[j] = new char[strlen(l[j]) + 1];
        strcpy(list[j], l[j]);
    }
}

ParamColormapChoice::ParamColormapChoice(const char *na, int num, char **l, int s)
    : Param(na, COLORMAPCHOICE_MSG, num)
{
    sel = s;
    list = new char *[num];
    for (int j = 0; j < num; j++)
    {
        list[j] = new char[strlen(l[j]) + 1];
        strcpy(list[j], l[j]);
    }
}

ParamColormapChoice::~ParamColormapChoice()
{
    for (int j = 0; j < no_of_items(); j++)
    {
        delete[] list[j];
    }
    delete[] list;
}

ParamMaterial::ParamMaterial(const char *na, char *l)
    : Param(na, MATERIAL_MSG, 1)
{
    list = new char[strlen(l) + 1];
    strcpy(list, l);
}

ParamMaterial::~ParamMaterial()
{
    delete[] list;
}

ParamColormap::ParamColormap(const char *na, int num, char **l)
    : Param(na, COLORMAP_MSG, num)
{
    list = new char *[num];
    for (int j = 0; j < num; j++)
    {
        list[j] = new char[strlen(l[j]) + 1];
        strcpy(list[j], l[j]);
    }
}

ParamColormap::~ParamColormap()
{
    for (int j = 0; j < no_of_items(); j++)
        delete[] list[j];
    delete[] list;
}

ParamColor::ParamColor(const char *na, int num, char **l)
    : Param(na, COLOR_MSG, num)
{
    list = new char *[num];
    for (int j = 0; j < num; j++)
    {
        list[j] = new char[strlen(l[j]) + 1];
        strcpy(list[j], l[j]);
    }
}

ParamColor::ParamColor(const char *na, float r, float g, float b, float a)
    : Param(na, COLOR_MSG, 4)
{
    char *buf = new char[255];
    list = new char *[3];
    sprintf(buf, "%f", r);
    list[0] = new char[strlen(buf) + 1];
    strcpy(list[0], buf);
    sprintf(buf, "%f", g);
    list[1] = new char[strlen(buf) + 1];
    strcpy(list[1], buf);
    sprintf(buf, "%f", b);
    list[2] = new char[strlen(buf) + 1];
    strcpy(list[2], buf);
    sprintf(buf, "%f", a);
    list[3] = new char[strlen(buf) + 1];
    strcpy(list[3], buf);
    delete[] buf;
}

ParamColor::~ParamColor()
{
    for (int j = 0; j < no_of_items(); j++)
        delete[] list[j];
    delete[] list;
}
