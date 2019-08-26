/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_MESSAGE_H
#define COVISE_MESSAGE_H

#include <string.h>
#include <stdio.h>
#include <iostream>

#include <util/coExport.h>
#include <util/byteswap.h>

#include <net/message.h>
#include <net/message_types.h>

//#include <shm/covise_shm.h>

#ifdef _WIN64
#define __WORDSIZE 64
#endif

/*
 $Log:  $
 * Revision 1.4  1994/03/23  18:07:03  zrf30125
 * Modifications for multiple Shared Memory segments have been finished
 * (not yet for Cray)
 *
 * Revision 1.3  93/10/11  09:22:19  zrhk0125
 * new types DM_CONTACT_DM and APP_CONTACT_DM included
 *
 * Revision 1.2  93/10/08  19:18:06  zrhk0125
 * data type sizes introduced
* some fixed type sizes with sizeof calls replaced
 *
 * Revision 1.1  93/09/25  20:47:03  zrhk0125
 * Initial revision
 *
 */

/***********************************************************************\
 **                                                                     **
 **   Message classes                              Version: 1.1         **
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
 **                        HOSTID                                             **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  15.04.93  Ver 1.0                                  **
 **                  15.04.93  Ver 1.1 new Messages and type added      **
 **                                    sender and send_type added       **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

namespace covise
{

class DataManagerProcess;
class coShmPtr;

enum colormap_type
{
    RGBAX,
    VIRVO
};

typedef long data_type;

class COVISEEXPORT ShmMessage : public Message // message especially for memory allocation
{
public: // at the datamanager
    ShmMessage(coShmPtr *ptr);
    // constructor with encoding into data field:
    ShmMessage(data_type d, long count)
        : Message()
    {
        conn = NULL;
        type = COVISE_MESSAGE_SHM_MALLOC;
        int l = sizeof(data_type) + sizeof(long);
        data = DataHandle(l);
        *(data_type *)data.accessData() = d;
        *(long *)(&data.accessData()[sizeof(data_type)]) = count;
    };
    ShmMessage(data_type *d, long *count, int no);
    ShmMessage(char *n, int t, data_type *d, long *count, int no);
    ShmMessage()
        : Message(){};
    //    ~ShmMessage() {
    //	delete [] data;
    //	data = NULL;
    //    };  // destructor
    int process_new_object_list(DataManagerProcess *dmgr);
    int process_list(DataManagerProcess *dmgr);
    data_type get_data_type() // data type of msg
    {
        return *(data_type *)data.data();
    };
    long get_count() // length of msg
    {
        return *(long *)(data.data() + sizeof(data_type));
    };
    int get_seq_no()
    {
        if (type == COVISE_MESSAGE_MALLOC_OK)
            return *(int *)(data.data());
        else
            return -1;
    };
    int get_offset()
    {
        if (type == COVISE_MESSAGE_MALLOC_OK)
            return *(int *)(data.data() + sizeof(data_type));
        else
            return -1;
    };
};

class COVISEEXPORT Param
{
    friend class CtlMessage;
    char *name;
    int type;
    int no;

public:
    Param(const char *na, int t, int n)
    {
        name = new char[strlen(na) + 1];
        strcpy(name, na);
        type = t;
        no = n;
    }
    char *getName() const
    {
        return name;
    }
    int no_of_items()
    {
        return no;
    }
    int get_type()
    {
        return type;
    }
    virtual ~Param()
    {
        delete[] name;
    }
};

class COVISEEXPORT ParamFloatScalar : public Param
{
    friend class CtlMessage;
    char *list;

public:
    ParamFloatScalar(const char *na, char *l);
    ParamFloatScalar(const char *na, float val);
    ~ParamFloatScalar()
    {
        delete[] list;
    }
};

class COVISEEXPORT ParamIntScalar : public Param
{
    friend class CtlMessage;
    char *list;

public:
    ParamIntScalar(const char *na, char *l);
    ParamIntScalar(const char *na, long val);
    ~ParamIntScalar();
};

class COVISEEXPORT ParamFloatVector : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamFloatVector(const char *na, int num, char **l);
    ParamFloatVector(const char *na, int num, float *l);
    ~ParamFloatVector();
};

class COVISEEXPORT ParamIntVector : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamIntVector(const char *na, int num, char **l);
    ParamIntVector(const char *na, int num, long *l);
    ~ParamIntVector();
};

class COVISEEXPORT ParamBrowser : public Param
{
    friend class CtlMessage;
    char *list;

public:
    ParamBrowser(const char *na, char *l);
    /*ParamBrowser(const char *na, char *file, char *wildcard) : Param(na, BROWSER, 2)
      {
         list = new char *[2];
         list[0] =  new char[ strlen(file)+1];
         strcpy(list[0],file);
         list[1] =  new char[ strlen(wildcard)+1];
         strcpy(list[1],wildcard);
      }*/
    ~ParamBrowser();
};

class COVISEEXPORT ParamString : public Param
{
    friend class CtlMessage;
    char *list;

public:
    ParamString(const char *na, char *l);
    char *get_length()
    {
        return (char *)strlen(list);
    };
    ~ParamString();
};

class COVISEEXPORT ParamText : public Param
{
    friend class CtlMessage;
    char **list;
    int line_num;
    int length;

public:
    ParamText(const char *na, char **l, int lineno, int len);
    int get_line_number()
    {
        return line_num;
    };
    int get_length()
    {
        return length;
    };
    ~ParamText();
};

class COVISEEXPORT ParamBoolean : public Param
{
    friend class CtlMessage;
    char *list;

public:
    ParamBoolean(const char *na, char *l);
    ParamBoolean(const char *na, int val);
    ~ParamBoolean();
};

class COVISEEXPORT ParamFloatSlider : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamFloatSlider(const char *na, int num, char **l);
    ParamFloatSlider(const char *na, float min, float max, float val);
    ~ParamFloatSlider();
};

class COVISEEXPORT ParamIntSlider : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamIntSlider(const char *na, int num, char **l);
    ParamIntSlider(const char *na, long min, long max, long val);
    ~ParamIntSlider();
};

class COVISEEXPORT ParamTimer : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamTimer(const char *na, int num, char **l);
    ParamTimer(const char *na, long start, long delta, long state);
    ~ParamTimer();
};

class COVISEEXPORT ParamPasswd : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamPasswd(const char *na, int num, char **l);
    ParamPasswd(const char *na, char *host, char *user, char *passwd);
    ~ParamPasswd();
};

class COVISEEXPORT ParamChoice : public Param
{
    friend class CtlMessage;
    char **list;
    int sel;

public:
    ParamChoice(const char *na, int num, int s, char **l);
    ParamChoice(const char *na, int num, char **l, int s);
    ~ParamChoice();
};

class COVISEEXPORT ParamColormapChoice : public Param
{
    friend class CtlMessage;
    char **list;
    int sel;

public:
    ParamColormapChoice(const char *na, int num, int s, char **l);
    ParamColormapChoice(const char *na, int num, char **l, int s);
    ~ParamColormapChoice();
};

class COVISEEXPORT ParamMaterial : public Param
{
    friend class CtlMessage;
    char *list;

public:
    ParamMaterial(const char *na, char *l);
    char *get_length()
    {
        return (char *)strlen(list);
    };

    ~ParamMaterial();
};

class COVISEEXPORT ParamColormap : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamColormap(const char *na, int num, char **l);

    ~ParamColormap();
};

class COVISEEXPORT ParamColor : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamColor(const char *na, int num, char **l);
    ParamColor(const char *na, float r, float g, float b, float a);
    ~ParamColor();
};

// holds everything necessary to work
// easily with messages from the controller
// to an application module
class COVISEEXPORT CtlMessage : public Message
{
private:
    char *m_name;
    char *h_name;
    char *inst_no;
    int no_of_objects;
    int no_of_save_objects;
    int no_of_release_objects;
    int no_of_params_in;
    int no_of_params_out;
    int MAX_OUT_PAR;
    char **port_names;
    char **object_names;
    char **object_types;
    int *port_connected;
    char **save_names;
    char **release_names;
    Param **params_in;
    Param **params_out;
    void init_list();

public:
    CtlMessage(Message *m)
        : Message()
    {
        sender = m->sender;
        send_type = m->send_type;
        type = m->type;
        data = DataHandle(strlen(m->data.data()) + 1);
        strcpy(data.accessData(), m->data.data());
        conn = m->conn; // never use this

        m_name = h_name = NULL;
        inst_no = NULL;
        object_types = port_names = object_names = save_names = release_names = NULL;
        no_of_save_objects = no_of_release_objects = 0;
        no_of_params_in = no_of_params_out = 0;
        MAX_OUT_PAR = 100;
        params_out = new Param *[MAX_OUT_PAR];
        params_in = 0;
        init_list();
    }
    ~CtlMessage();

    void get_header(char **m, char **h, char **inst)
    {
        *m = m_name;
        *h = h_name, *inst = inst_no;
    };

    int get_scalar_param(const char *, long *);
    int get_scalar_param(const char *, float *);
    int get_vector_param(const char *, int, long *);
    int get_vector_param(const char *, int, float *);
    int get_string_param(const char *, char **);
    int get_text_param(const char *, char ***, int *line_num);
    int get_boolean_param(const char *, int *);
    int get_slider_param(const char *, long *min, long *max, long *value);
    int get_slider_param(const char *, float *min, float *max, float *value);
    int get_browser_param(const char *, char **);
    int get_timer_param(const char *, long *start, long *delta, long *state);
    int get_passwd_param(const char *, char **host, char **user, char **passwd);
    int get_choice_param(const char *, int *);
    int get_choice_param(const char *, char **);
    int get_material_param(const char *, char **);
    int get_colormapchoice_param(const char *, int *);
    int get_colormapchoice_param(const char *, char **);
    int get_colormap_param(const char *, float *min, float *max, int *len, colormap_type *type);
    int get_color_param(const char *, float *r, float *g, float *b, float *a);

    int set_scalar_param(const char *, long);
    int set_scalar_param(const char *, float);
    int set_vector_param(const char *, int num, long *);
    int set_vector_param(const char *, int num, float *);
    int set_string_param(const char *, char *);
    int set_text_param(const char *, char *, int);
    int set_boolean_param(const char *, int);
    int set_slider_param(const char *, long min, long max, long value);
    int set_slider_param(const char *, float min, float max, float value);
    int set_choice_param(const char *, int, char **, int);
    int set_browser_param(const char *, char *, char *);
    int set_timer_param(const char *, long start, long delta, long state);
    int set_passwd_param(const char *, char *host, char *user, char *passwd);

    char *get_object_name(const char *name);
    char *getObjectType(const char *name);

    int set_save_object(const char *name);
    int set_release_object(const char *name);

    int create_finpart_message();
    int create_finall_message();

    int is_port_connected(const char *name);
};

class COVISEEXPORT RenderMessage : public Message // holds everything necessary to work
{
    // easily with messages from the controller
    // to the renderer
    char *m_name;
    char *h_name;
    char *inst_no;
    void init_list();

public:
    int no_of_objects;
    char **object_names;
    RenderMessage(Message *m)
        : Message()
    {
        sender = m->sender;
        send_type = m->send_type;
        type = m->type;
        data = m->data;
        conn = m->conn;
        delete m;
        m_name = h_name = NULL;
        inst_no = NULL;
        no_of_objects = 0;
        object_names = NULL;
        init_list();
    }
    ~RenderMessage();
    char *get_part(char *chdata = NULL);
};
}
#endif
