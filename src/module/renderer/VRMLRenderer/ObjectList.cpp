/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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
 **                             Dirk Rantzau                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  11.09.95  V1.0                                                  **
\**************************************************************************/

//
// C stuff
//
#include <covise/covise.h>
#include <net/covise_connect.h>

#include "ObjectList.h"
#include "Slider_VRML.h"

#include <float.h>

#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

//
//##########################################################################
// Object
//##########################################################################
//

//==========================================================================
//  constructor : use default arguments
//==========================================================================
LObject::LObject()
{
    name = NULL;
    rootname = NULL;
    real_root = NULL;
    objPtr = NULL;
    m_min_timestep = 0;
    m_max_timestep = 0;
    m_timestep = -1;
    m_is_timestep = 0;
    m_new = 1;
}

//=========================================================================
// constructor : setup name , type and data pointer
//=========================================================================
LObject::LObject(const char *na, const char *rn, CharBuffer *ptr)
{
    name = new char[strlen(na) + 1];
    strcpy(name, na);
    rootname = NULL;
    if (rn)
    {
        rootname = new char[strlen(rn) + 1];
        strcpy(rootname, rn);
    }
    objPtr = ptr;
    m_min_timestep = 0;
    m_max_timestep = 0;
    m_timestep = -1;
    m_is_timestep = 0;
    real_root = NULL;
    m_new = 1;
}

//=========================================================================
// constructor : setup name , type and data pointer
//=========================================================================
LObject::LObject(const char *na, const char *rn, void *ptr)
{
    name = new char[strlen(na) + 1];
    strcpy(name, na);
    rootname = NULL;
    if (rn)
    {
        rootname = new char[strlen(rn) + 1];
        strcpy(rootname, rn);
    }
    objPtr = (CharBuffer *)ptr;
    m_min_timestep = 0;
    m_max_timestep = 0;
    m_timestep = -1;
    m_is_timestep = 0;
    real_root = NULL;
    m_new = 1;
}

//=========================================================================
// set timestep nb from object name
//=========================================================================

void LObject::set_timestep(const char *name)
{
    char *str_timestep;

    if (name == NULL)
        return;
    str_timestep = (char *)strrchr(name, '_');
    if (str_timestep == NULL)
        return;
    str_timestep++; //skip '_'
    m_timestep = atoi(str_timestep);
}

//=========================================================================
// set object name
//=========================================================================
void LObject::set_name(const char *str)
{
    delete[] name;
    name = new char[strlen(str) + 1];
    strcpy(name, str);
}

//=========================================================================
// set rootobject name
//=========================================================================
void LObject::set_rootname(const char *str)
{
    delete[] rootname;
    rootname = NULL;
    if (str)
    {
        rootname = new char[strlen(str) + 1];
        strcpy(rootname, str);
    }
}

//=========================================================================
// set rootobject name for an group/switch object
//=========================================================================
void LObject::set_real_root(const char *str)
{
    if (real_root)
        delete[] real_root;
    real_root = NULL;
    if (str)
    {
        real_root = new char[strlen(str) + 1];
        strcpy(real_root, str);
    }
}

void LObject::set_boundingbox(float *boundingbox)
{
    memcpy(bbox, boundingbox, 6 * sizeof(float));
}

void LObject::get_boundingbox(float *boundingbox)
{
    memcpy(boundingbox, bbox, 6 * sizeof(float));
}

//#########################################################################
// ObjectList
//#########################################################################
//

//=========================================================================
// constructor : setup head and tail of list
//=========================================================================

ObjectList::ObjectList()
{
    m_length = 0;
    m_buff = NULL;
    m_index = 0;
    m_crt_timestep = 0;
    m_no_sw = 0;
    m_prev_camera = NULL;
    m_cameraMsg = NULL;
    m_cam_position = NULL;
    m_cam_orientation = NULL;
    m_telepMsg = NULL;

    translation[0] = translation[1] = translation[2] = 0;
    scale[0] = scale[1] = scale[2] = 1;
    rotation[0] = rotation[1] = rotation[3] = 0;
    rotation[2] = 1;
}

void ObjectList::incr_no_sw(void)
{
    m_no_sw++;
    //cerr << endl << "--- incr_no_sw(): " << m_no_sw;
};

void ObjectList::decr_no_sw(void)
{
    m_no_sw--;
    //cerr << endl << "--- decr_no_sw(): " << m_no_sw;
};

void ObjectList::set_crt_timestep(int timestep)
{
    //cerr << endl << " $$$$ set_crt_timestep: " << timestep << endl;
    m_crt_timestep = timestep;
}

void ObjectList::printViewPoint(FILE *fp, char *desc, float fov,
                                float p_x, float p_y, float p_z,
                                float o_x, float o_y, float o_z, float o_d)
{
    if (outputMode == OutputMode::VRML97)
    {
        fprintf(fp, "\nViewpoint {\n    fieldOfView %f\n    jump TRUE\n    position %f %f %f\n    orientation %f %f %f %f\n    description \"%s\"\n}\n", fov,
            p_x, p_y, p_z, o_x, o_y, o_z, o_d, desc);
    }
    else
    {
        fprintf(fp, "\n<viewpoint\n    fieldOfView='%f'\n    jump='TRUE'\n    position='%f %f %f'\n    orientation='%f %f %f %f'\n    description='%s'></viewpoint>\n", fov,
            p_x, p_y, p_z, o_x, o_y, o_z, o_d, desc);
    }

   
}

void ObjectList::write(FILE *fp)
{
    int hastime = 0, numt = 0, numbeg = 10000, numtime = 20, i;
    CharBuffer *cb;
    char *bufs[200];
    int numb = 0;
    if (outputMode == OutputMode::VRML97)
    {
        fprintf(fp, "NavigationInfo {\n    type        \"EXAMINE\"\n}");
    }
    else
    {
        fprintf(fp, "<navigationInfo type='\"EXAMINE\"'></navigationInfo>\n");
    }

    float bbox[6] = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

    if (outputMode == OutputMode::VRML97)
    {
    fprintf(fp, "\n\nTransform {\n  translation %f %f %f\n  scale %f %f %f\n  rotation %f %f %f %f\n  children [\n",
            translation[0], translation[1], translation[2], scale[0], scale[1], scale[2],
            rotation[0], rotation[1], rotation[2], rotation[3] / 360.0 * 2 * M_PI);
    }
    else
    {
        fprintf(fp, "\n\n<transform\n  translation='%f %f %f'\n  scale='%f %f %f'\n  rotation='%f %f %f %f'\n>\n",
            translation[0], translation[1], translation[2], scale[0], scale[1], scale[2],
            rotation[0], rotation[1], rotation[2], rotation[3] / 360.0 * 2 * M_PI);
        //fprintf(fp, "\n\n<switch whichChoice = '0' id = 'sliderVariants'>");
    }

    for (const auto& it : *this)
    {
        if (strcmp("Endset", it->name) == 0)
        {
            if (outputMode == OutputMode::VRML97)
            {
                fprintf(fp, "]\n}\n");
            }
            else
            {
                fprintf(fp, "</switch>\n");
            }
            numbeg--;
            if (numbeg == 0)
                numtime = numt;
        }
        else if (strcmp("Beginset", it->name) == 0)
        {
            if (outputMode == OutputMode::VRML97)
            {
                fprintf(fp, "Group {\n    children [\n");
            }
            else
            {
                fprintf(fp, "<transform>\n");
            }
            if (numbeg == 1)
                numt++;
            numbeg++;
        }
        else if (strcmp("BeginTimeset", it->name) == 0)
        {
            if (outputMode == OutputMode::VRML97)
            {

                bufs[numb] = new char[200];
                sprintf(bufs[numb], "\nROUTE SCR.switchValue TO SW_%s.set_whichChoice\n", it->rootname);
                numb++;
                fprintf(fp, "DEF SW_%s Switch {\n    choice [\n", it->rootname);
            }
            else
            {
                bufs[numb] = new char[200];
                // TODO add code to switch timesteps sprintf(bufs[numb], "\nROUTE SCR.switchValue TO SW_%s.set_whichChoice\n", it->rootname);

                //numb++;
                fprintf(fp, "<switch id=\"SW_%s\">\n", it->rootname);
            }
            numt = 0;
            hastime++;
            numbeg = 1;
        }
        else
        {
            float object_bbox[6];
            it->get_boundingbox(object_bbox);
            for (i = 0; i < 3; i++)
                if (object_bbox[i] < bbox[i])
                    bbox[i] = object_bbox[i];
            for (i = 3; i < 6; i++)
                if (object_bbox[i] > bbox[i])
                    bbox[i] = object_bbox[i];

            cb = it->get_objPtr();
            const char *tmp;
            if (cb)
            {
                if (outputMode == OutputMode::VRML97)
                {
                    fprintf(fp, "\n\n# Name: %s Group: %s\n\n", it->name, it->rootname);
                }
                else
                {
                    fprintf(fp, "\n\n<!-- Name: %s Group: %s -->\n\n", it->name, it->rootname);
                }
                tmp = (const char*)(*cb);
                fprintf(fp, "%s", tmp);
            }
            if (numbeg == 1)
                numt++;
        }
    } // end while

    float fov = 0.785398f;

    float middle[3];
    middle[0] = bbox[3] - (bbox[3] - bbox[0]) / 2;
    middle[1] = bbox[4] - (bbox[4] - bbox[1]) / 2;
    middle[2] = bbox[5] - (bbox[5] - bbox[2]) / 2;

    float distance[3];

    distance[0] = 4.0f / 3.0f * ((bbox[3] - bbox[0])) / (2 * tan(fov / 2.0f));
    distance[1] = 4.0f / 3.0f * ((bbox[4] - bbox[1])) / (2 * tan(fov / 2.0f));
    distance[2] = 4.0f / 3.0f * ((bbox[5] - bbox[2])) / (2 * tan(fov / 2.0f));

    float dist = MAX(distance[0], MAX(distance[1], distance[2]));

    if (outputMode == OutputMode::VRML97)
    {
        fprintf(fp, "  ] # transform children\n} # transform\n");
    }
    else
    {
        //fprintf(fp, "</switch>\n");
        fprintf(fp, "</transform>\n"); 
    }
    if (hastime > 0)
    {
        if (outputMode == OutputMode::VRML97)
        {
            fprintf(fp, SLIDER);
            fprintf(fp, SLIDER2);
            fprintf(fp, SLIDER3);
            fprintf(fp, SLIDER4);

            //      fprintf(fp, "\nDEF StaticCaveSlider Transform {\n translation %f %f %f\n scale %f %f %f\n children [\n", middle[0], bbox[2], middle[2], dist / 6, dist / 6, dist / 6);
            fprintf(fp, "\nDEF StaticCaveSlider Transform {\n translation %f %f %f\n scale %f %f %f\n children [\n", 10.88, -600.0, -1.15, 300.0, 300.0, 300.0);
            fprintf(fp, SLIDER5);
            fprintf(fp, SLIDER6);
            fprintf(fp, "\n] }\n");

            fprintf(fp, SLIDER7);
            fprintf(fp, SLIDER8);
            fprintf(fp, "field SFInt32 sizeOfSwitch %d\n}\n", numtime);
            fprintf(fp, ROUTES);
        }
        else
        {
            // add java script code for animation control

        }
    }
    for (i = 0; i < numb; i++)
    {
        fprintf(fp, "%s", bufs[i]);
        delete[] bufs[i];
    }

    printViewPoint(fp, (char *)"Front", fov, middle[0], middle[1], middle[2] + dist,
                   1, 0, 0, 0);

    printViewPoint(fp, (char *)"Back", fov, middle[0], middle[1], middle[2] - dist,
                   0, 1, 0, (float)M_PI);

    printViewPoint(fp, (char *)"Right", fov, middle[0] + dist, middle[1], middle[2],
                   0, 1, 0, (float)(M_PI / 2.0));

    printViewPoint(fp, (char *)"Left", fov, middle[0] - dist, middle[1], middle[2],
                   0, -1, 0, (float)(M_PI / 2.0));

    printViewPoint(fp, (char *)"Top", fov, middle[0], middle[1] + dist, middle[2],
                   -1, 0, 0, (float)(M_PI / 2.0));

    printViewPoint(fp, (char *)"Bottom", fov, middle[0], middle[1] - dist, middle[2],
                   1, 0, 0, (float)(M_PI / 2.0));
}

//=========================================================================
//=========================================================================
void ObjectList::removeone(const char *n)
{
    for(auto it = begin();it!=end();it++)
    {
        if (strcmp((*it)->name, n) == 0)
        {
            if ((*it)->m_is_timestep)
                decr_no_sw();
            it=erase(it);
            return;
        }
        if ((*it)->rootname && (*it)->name && (strcmp((*it)->rootname, n) == 0) && (strncmp((*it)->name, "Begin", 5) == 0))
        {
            int numi = 1;
            while (it!=end())
            {
                if (strncmp((*it)->name, "Begin", 5) == 0)
                    numi++;
                if (strcmp((*it)->name, "Endset") == 0)
                    numi--;
                if (numi == 0)
                {
                    if ((*it)->m_is_timestep)
                        decr_no_sw();
                    it = erase(it);
                    break;
                }
                if ((*it)->m_is_timestep)
                    decr_no_sw();
                it = erase(it);
            }
            return;
        }
    }
}

void ObjectList::removeall(const char *n)
{
    //char *currname="irgendwas";
    char buf[500];
    for (const auto& it : *this)
    {
        if (it->rootname)
        {
            if (it->rootname && (strcmp(it->rootname, n) == 0))
            {
                strcpy(buf, it->name);
                removeone(buf);
            }
        }
    }
}

/*______________________________________________________________________*/
int ObjectList::setViewPoint(char *camera)
{
    char cam[500], tmp[500], tmpx[500];
    char *str;
    float fieldOfView;

    if (camera == NULL)
    {
        return 0;
    }

    if (m_prev_camera != NULL)
    {
        if (strcmp(m_prev_camera, camera) == 0)
        {
            return 0; // message not different than previous
        }
    }

    // message different than previous
    if (m_prev_camera != NULL)
        delete[] m_prev_camera;
    m_prev_camera = new char[strlen(camera) + 1];
    strcpy(m_prev_camera, camera);

    if (m_cam_position != NULL)
        delete[] m_cam_position;
    if (m_cam_orientation != NULL)
        delete[] m_cam_orientation;
    if (m_cameraMsg != NULL)
        delete[] m_cameraMsg;

    m_cam_position = NULL;
    m_cam_orientation = NULL;
    m_cameraMsg = NULL;

    strcpy(cam, camera);
    strcpy(tmp, "DEF COVISE Viewpoint {\n");
    strtok(cam, "\n");
    strtok(NULL, "\n");
    str = strtok(NULL, "\n"); // read position
    strcat(tmp, str);
    strcat(tmp, " \n");
    m_cam_position = new char[strlen(str) + 1];
    strcpy(m_cam_position, str);
    if (strstr(camera, "orientation") != NULL)
    {
        str = strtok(NULL, "\n"); // read orientation
        strcat(tmp, str);
        strcat(tmp, " \n");
        m_cam_orientation = new char[strlen(str) + 1];
        strcpy(m_cam_orientation, str);
    }
    strtok(NULL, "\n"); // skip nearDistance
    strtok(NULL, "\n"); // skip farDistance
    strtok(NULL, "\n"); // skip focalDistance
    int retval;
    retval = sscanf(str, "%s %f", cam, &fieldOfView); // read fieldOfView
    if (retval != 2)
    {
        std::cerr << "ObjectList::setViewPoint: sscanf failed" << std::endl;
        return 0;
    }
    fieldOfView = 1.; // force fieldOfView ?!
    sprintf(tmpx, "   fieldOfView %f \n    description \"COVISE\" \n} \n", fieldOfView);
    strcat(tmp, tmpx);

    m_cameraMsg = new char[strlen(tmp) + 1];
    strcpy(m_cameraMsg, tmp);

    return 1;
}

/*______________________________________________________________________*/
int ObjectList::setSequencer(char *seq)
{
    int timestep, x;

    int retval;
    retval = sscanf(seq, "%d %d %d %d %d\n", &timestep, &x, &x, &x, &x);
    if (retval != 5)
    {
        std::cerr << "ObjectList::setSequencer: sscanf failed" << std::endl;
        return 0;
    }
    set_crt_timestep(timestep - 1);

    return 1;
}

/*______________________________________________________________________*/
char *ObjectList::getViewPoint(void)
{
    if (m_cameraMsg == NULL)
    {
        m_cameraMsg = new char[300];
        strcpy(m_cameraMsg, "DEF COVISE Viewpoint {\n position \t0.5 0.5 3 \n orientation \t0.0 0.0 1.0  0.0 \n fieldOfView 1.0 \n description \"COVISE\" \n}\n");
    }
    return m_cameraMsg;
}

/*______________________________________________________________________*/
void ObjectList::setTelepointer(char *telep)
{
    char tmp[1000];
    char host[100];
    int press;
    float px, py, pz, asp;

    if (m_telepMsg != NULL)
        delete[] m_telepMsg;
    m_telepMsg = NULL;

    int retval;
    retval = sscanf(telep, "%s %d %f %f %f %f", &host[0], &press, &px, &py, &pz, &asp);
    if (retval != 6)
    {
        std::cerr << "ObjectList::setTelepointer: sscanf failed" << std::endl;
        return;
    }
    //pz = 0;
    if (press)
    {
        sprintf(tmp, "11 Transform {\n translation %f %f %f\n children Shape{  appearance Appearance { material Material {diffuseColor  1 0 0 }}\n geometry Text {  string \"<%s\"\n fontStyle FontStyle{ size 0.3} } }\n}\n", px, py, pz, host);
    }
    else
    {
        sprintf(tmp, "12 Transform {\n translation %f %f %f\n children Shape{  appearance Appearance { material Material {diffuseColor  1 0 0 }}\n geometry Text {  string \"<%s\"\n fontStyle FontStyle{ size 0.3} } }\n}\n", px, py, pz, host);
    }

    m_telepMsg = new char[strlen(tmp) + 1];
    strcpy(m_telepMsg, tmp);
}

int ObjectList::sendTelepointer(const Connection *conn)
{
    if (m_telepMsg != NULL)
    {
        send_obj(conn, m_telepMsg);
    }
    return 1;
}

void
ObjectList::TransformViewPoint(char *output)
{
    char tmp[100], cam[1000];

    if (m_cameraMsg == NULL)
    {
        strcpy(output, "NO CAM");
        return;
    }
    strcpy(cam, m_cameraMsg);
    strcpy(output, "DEF World Viewpoint {\n");
    char *str = strtok(cam, "\n");
    strtok(NULL, "\n");
    str = strtok(NULL, "\n"); //position
    strcat(output, str);
    strcat(output, "\n");
    if (strstr(m_cameraMsg, "orientation") != NULL)
    {
        str = strtok(NULL, "\n"); // orientation
        strcat(output, str);
        strcat(output, "\n");
    }
    strtok(NULL, "\n"); // skip nearDistance
    strtok(NULL, "\n"); // skip farDistance
    str = strtok(NULL, "\n"); // read focalDistance
    float fieldofView = 1.;
    int retval;
    retval = sscanf(str, "%s %f", cam, &fieldofView);
    if (retval != 2)
    {
        std::cerr << "ObjectList::TransformViewPoint: sscanf failed" << std::endl;
        return;
    }
    fieldofView = 1.;
    sprintf(tmp, "   fieldOfView %f\n    description \"COVISE\" } \n", fieldofView);
    strcat(output, tmp);
}

void ObjectList::parseObjects(void)
{
    m_length = 0;

    int hastime = 0, numt = 0, numbeg = 10000, numtime = 20, i;
    CharBuffer *cb;
    char *bufs[200];
    int numb = 0;

    char tmp_buff[10000];

    sprintf(tmp_buff, "%s", " 2 #VRML V2.0 utf8  \n");
    m_length += (int)strlen(tmp_buff);
    for (const auto& it : *this)
    {
        if (strcmp("Endset", it->name) == 0)
        {
            sprintf(tmp_buff, "%s", "]\n}\n");
            m_length += (int)strlen(tmp_buff);
            numbeg--;
            if (numbeg == 0)
                numtime = numt;
        }
        else if (strcmp("Beginset", it->name) == 0)
        {
            sprintf(tmp_buff, "%s", "Group {\n    children [\n");
            m_length += (int)strlen(tmp_buff);
            if (numbeg == 1)
                numt++;
            numbeg++;
        }
        else if (strcmp("BeginTimeset", it->name) == 0)
        {
            bufs[numb] = new char[200];
            sprintf(bufs[numb], "\nROUTE SCR.switchValue TO SW_%s.set_whichChoice\n", it->rootname);
            numb++;
            sprintf(tmp_buff, "DEF SW_%s Switch {\n    choice [\n", it->rootname);
            m_length += (int)strlen(tmp_buff);
            numt = 0;
            hastime++;
            numbeg = 1;
        }
        else
        {
            cb = it->get_objPtr();
            const char *tmp;
            if (cb)
            {
                tmp = (const char *)(*cb);
                sprintf(tmp_buff, "\n\n# Name: %s Group: %s\n\n", it->name, it->rootname);
                m_length += (int)strlen(tmp_buff);
                //fprintf(fp,tmp);
                m_length += (int)strlen(tmp);
            }
            if (numbeg == 1)
                numt++;
        }
    } // end while
    if (hastime > 0)
    {
        sprintf(tmp_buff, "%s%s%s%s%s%s%s%sfield SFInt32 sizeOfSwitch %d\n}\n%s",
                SLIDER, SLIDER2, SLIDER3, SLIDER4, SLIDER5, SLIDER6, SLIDER7, SLIDER8, numtime, ROUTES);
        m_length += (int)strlen(tmp_buff);
    }
    for (i = 0; i < numb; i++)
    {
        m_length += (int)strlen(bufs[i]);
        delete[] bufs[i];
    }
}

void ObjectList::send_obj(const Connection *conn, char *obj, int add_length)
{
    int intbuff[4];
    int length;

    length = (int)strlen(obj);

    intbuff[0] = conn->get_sender_id();
    intbuff[1] = conn->get_sendertype();
    intbuff[2] = COVISE_MESSAGE_OBJECT_TRANSFER;
    intbuff[3] = length + add_length;

#ifdef BYTESWAP
    swap_bytes((unsigned int *)intbuff, 4); // alignement with send_msg
#endif

    conn->send(intbuff, 4 * sizeof(int)); // send header

    // send the rest

    conn->send(obj, length);
}

int ObjectList::sendObjects(const Connection *conn)
{
    int hastime = 0, numt = 0, numbeg = 10000, i;
    CharBuffer *cb;
    char *bufs[200];
    int numb = 0;

    char tmp_buff[10000];

    for (const auto& it : *this)
    {
        if (strcmp("Endset", it->name) == 0)
        {
            numbeg--;
        }
        else if (strcmp("Beginset", it->name) == 0)
        {
            // add Group
            if (it->real_root)
            {
                sprintf(tmp_buff, " 7 %s@%s#%d\n", it->rootname, it->real_root, it->get_timestep());
            }
            else
            {
                sprintf(tmp_buff, " 7 %s@ROOT#%d\n", it->rootname, it->get_timestep());
            }
            it->m_new = 0;
            send_obj(conn, tmp_buff);

            if (numbeg == 1)
                numt++;
            numbeg++;
        }
        else if (strcmp("BeginTimeset", it->name) == 0)
        {

            bufs[numb] = new char[200];
            sprintf(bufs[numb], "\nROUTE SCR.switchValue TO SW_%s.set_whichChoice\n", it->rootname);
            numb++;
            sprintf(tmp_buff, "DEF SW_%s Switch {\n    choice [\n", it->rootname);

            // add Switch
            if (it->real_root)
            {
                sprintf(tmp_buff, " 8 %s@%s#%d#%d\n", it->rootname, it->real_root, it->m_min_timestep, it->m_max_timestep);
            }
            else
            {
                sprintf(tmp_buff, " 8 %s@ROOT#%d#%d\n", it->rootname, it->m_min_timestep, it->m_max_timestep);
            }
            it->m_new = 0;
            send_obj(conn, tmp_buff);

            numt = 0;
            hastime++;
            numbeg = 1;
        }
        else
        {
            cb = it->get_objPtr();
            const char *tmp;
            if (cb)
            {
                tmp = (const char *)(*cb);
                sprintf(tmp_buff, "\n\n# Name: %s Group: %s\n\n", it->name, it->rootname);

                // add Geometry
                if (it->rootname)
                {
                    sprintf(tmp_buff, " 6 %s@%s#%d\n", it->name, it->rootname, it->get_timestep());
                }
                else
                {
                    sprintf(tmp_buff, " 6 %s@ROOT#%d\n", it->name, it->get_timestep());
                }
                int add_l = (int)strlen(tmp);
                send_obj(conn, tmp_buff, add_l);
                it->m_new = 0;
                conn->send(tmp, add_l);
            }
            if (numbeg == 1)
                numt++;
        }
    } // end while
    for (i = 0; i < numb; i++)
    {
        delete[] bufs[i];
    }

    return 1;
}

int ObjectList::sendNewObjects(const Connection *conn)
{

    int hastime = 0, numt = 0, numbeg = 10000, i;
    CharBuffer *cb;
    char *bufs[200];
    int numb = 0;

    char tmp_buff[10000];

    for (const auto& it : *this)
    {
        if (it->m_new)
        {
            if (strcmp("Endset", it->name) == 0)
            {
                numbeg--;
            }
            else if (strcmp("Beginset", it->name) == 0)
            {
                // add Group
                if (it->real_root)
                {
                    sprintf(tmp_buff, " 7 %s@%s#%d\n", it->rootname, it->real_root, it->get_timestep());
                }
                else
                {
                    sprintf(tmp_buff, " 7 %s@ROOT#%d\n", it->rootname, it->get_timestep());
                }
                it->m_new = 0;
                send_obj(conn, tmp_buff);
                if (numbeg == 1)
                    numt++;
                numbeg++;
            }
            else if (strcmp("BeginTimeset", it->name) == 0)
            {

                bufs[numb] = new char[200];
                sprintf(bufs[numb], "\nROUTE SCR.switchValue TO SW_%s.set_whichChoice\n", it->rootname);
                numb++;
                sprintf(tmp_buff, "DEF SW_%s Switch {\n    choice [\n", it->rootname);

                // add Switch
                if (it->real_root)
                {
                    sprintf(tmp_buff, " 8 %s@%s#%d#%d\n", it->rootname, it->real_root, it->m_min_timestep, it->m_max_timestep);
                }
                else
                {
                    sprintf(tmp_buff, " 8 %s@ROOT#%d#%d\n", it->rootname, it->m_min_timestep, it->m_max_timestep);
                }
                it->m_new = 0;
                send_obj(conn, tmp_buff);
                numt = 0;
                hastime++;
                numbeg = 1;
            }
            else
            {
                cb = it->get_objPtr();
                const char *tmp;
                if (cb)
                {
                    tmp = (const char *)(*cb);
                    sprintf(tmp_buff, "\n\n# Name: %s Group: %s\n\n", it->name, it->rootname);
                    // add Geometry
                    if (it->rootname)
                    {
                        sprintf(tmp_buff, " 6 %s@%s#%d\n", it->name, it->rootname, it->get_timestep());
                    }
                    else
                    {
                        sprintf(tmp_buff, " 6 %s@ROOT#%d\n", it->name, it->get_timestep());
                    }
                    int add_l = (int)strlen(tmp);
                    send_obj(conn, tmp_buff, add_l);
                    it->m_new = 0;
                    conn->send(tmp, add_l);
                }
                if (numbeg == 1)
                    numt++;
            }
        } // end new object
    } // end while

    for (i = 0; i < numb; i++)
    {
        delete[] bufs[i];
    }

    return 1;
}

int ObjectList::sendTimestep(const Connection *conn)
{
    char tmp_buff[10];

    if (m_no_sw > 0)
    {
        sprintf(tmp_buff, "10 %d", m_crt_timestep);
        send_obj(conn, tmp_buff);
    }
    return 1;
}

void ObjectList::setTranslation(float x, float y, float z)
{

    translation[0] = x;
    translation[1] = y;
    translation[2] = z;
}

void ObjectList::setScale(float x, float y, float z)
{

    scale[0] = x;
    scale[1] = y;
    scale[2] = z;
}

void ObjectList::setRotation(float x, float y, float z)
{

    rotation[0] = x;
    rotation[1] = y;
    rotation[2] = z;
}

void ObjectList::setRotationAngle(float d)
{

    rotation[3] = d;
}
