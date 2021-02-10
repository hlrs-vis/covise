/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _OBJECT_LIST_H
#define _OBJECT_LIST_H

/**************************************************************************\
**                                                           (C)1995 RUS  **
**                                                                        **
** Description: object list class for COVISE renderer modules             **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author:                                                                **
**                                                                        **
**                             Uwe Woessner                               **
**                Computer Center University of Stuttgart                 **
**                            Allmandring 30                              **
**                            70550 Stuttgart                             **
**                                                                        **
** Date:  11.09.95  V1.0                                                  **
\**************************************************************************/

#include <covise/Covise_Util.h>
#include <list>
#include <appl/RenderInterface.h>


enum OutputMode
{
    VRML97 = 0,
    X3DOM = 1
};

using namespace covise;
//================================================================
// Object
//================================================================

class LObject
{

public:
    char *name;
    char *rootname;
    char *real_root;
    int m_min_timestep;
    int m_max_timestep;
    int m_timestep;
    int m_is_timestep;
    int m_new;
    float bbox[6];

    CharBuffer *objPtr;

    LObject(const char *na, const char *ex, CharBuffer *obj);
    LObject(const char *na, const char *ex, void *obj);

    LObject();

    void set_timestep(const char *name);
    void set_timestep(int i)
    {
        m_timestep = i;
    }
    void set_minmax(int min, int max)
    {
        m_min_timestep = min;
        m_max_timestep = max;
    };

    void set_name(const char *str);
    void set_rootname(const char *str);
    void set_real_root(const char *str);
    void set_objPtr(CharBuffer *obj)
    {
        objPtr = obj;
    };
    void set_objPtr(void *obj)
    {
        objPtr = (CharBuffer *)obj;
    };
    char *get_name()
    {
        return (name);
    };
    char *get_rootname()
    {
        return (rootname);
    };
    int get_timestep(void)
    {
        return m_timestep;
    };
    CharBuffer *get_objPtr()
    {
        return (objPtr);
    };
    void write(FILE *fp)
    {
        fprintf(fp, "%s", (const char *)objPtr);
    };
    void set_boundingbox(float *boundingbox);
    void get_boundingbox(float *boundingbox);

    ~LObject()
    {
        delete[] name;
        delete[] rootname;
        delete objPtr;
    };
};

//================================================================
// ObjectList
//================================================================

class ObjectList : public std::list<std::unique_ptr<LObject>>
{
    int m_length;
    char *m_buff;
    int m_index;

private:
    float translation[3];
    float scale[3];
    float rotation[4];

    char *m_prev_camera;
    char *m_cameraMsg;
    char *m_cam_position;
    char *m_cam_orientation;

    char *m_telepMsg;
    int m_crt_timestep;
    int m_no_sw;

    void printViewPoint(FILE *fp, char *desc, float fov,
                        float p_x, float p_y, float p_z,
                        float o_x, float o_y, float o_z, float o_d);

public:
    ObjectList();
    ~ObjectList()
    {
        if (m_prev_camera != NULL)
            delete[] m_prev_camera;
        if (m_cameraMsg != NULL)
            delete[] m_cameraMsg;
        if (m_cam_position != NULL)
            delete[] m_cam_position;
        if (m_cam_orientation != NULL)
            delete[] m_cam_orientation;
        if (m_telepMsg != NULL)
            delete[] m_telepMsg;
    };
    OutputMode outputMode;
    void write(FILE *fp);
    void removeone(const char *);
    void removeall(const char *);
    void parseObjects(void);
    //int         send_chunk(char* buff,int l);
    void send_obj(const Connection *conn, char *obj, int add_length = 0);
    int sendObjects(const Connection *conn);
    int sendNewObjects(const Connection *conn);
    void set_crt_timestep(int timestep);
    void incr_no_sw(void);
    void decr_no_sw(void);
    int sendTimestep(const Connection *conn);
    int setViewPoint(char *camera);
    int setSequencer(char *seq);
    char *getViewPoint(void);
    void setTelepointer(char *telep);
    int sendTelepointer(const Connection *conn);
    void TransformViewPoint(char *);

    void setTranslation(float x, float y, float z);
    void setScale(float x, float y, float z);
    void setRotation(float x, float y, float z);
    void setRotationAngle(float d);
    void setOutputMode(OutputMode om)
    {
        outputMode = om;
    }
};

#endif
