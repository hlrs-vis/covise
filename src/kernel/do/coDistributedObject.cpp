/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDistributedObject.h"
#include <covise/covise.h>
#include <covise/covise_global.h>
#include <covise/covise_appproc.h>
#include "coDoData.h"
#include "coDoGeometry.h"
#include "coDoUniformGrid.h"
#include "coDoRectilinearGrid.h"
#include "coDoStructuredGrid.h"
#include "coDoUnstructuredGrid.h"
#include "coDoSet.h"
#include "coDoIntArr.h"

#undef DEBUG

/***********************************************************************\ 
 **                                                                     **
 **   Distributed Object class Routines            Version: 1.0         **
 **                                                                     **
 **                                                                     **
 **   Description  : The base class for all objects that are            **
 **                  distributed between processes.                     **
 **                  Basic functionality to use shared storage is       **
 **                  provided for its subclasses.                       **
 **                                                                     **
 **   Classes      : coDistributedObject                                  **
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
 **                  26.05.93  Ver 1.1 new structure of data in shm     **
 **                                    type handling                    **
 **                                    recursive objects                **
 **                                                                     **
\***********************************************************************/

static int object_exists;
static int *free_list;
static int current_free;

//static coShmArray *getShmArray(const char *name);

namespace covise
{

static coShmArray *getShmArray(const char *name)
{
    if (!name)
    {
        print_comment(__LINE__, __FILE__, "tried getShmArray with name == NULL");
        return NULL;
    }

    if (!ApplicationProcess::approc)
        return NULL;

    int len = (int)strlen(name) + 1;
    char *tmpptr = new char[len];
    strcpy(tmpptr, name);
    Message msg{ COVISE_MESSAGE_GET_OBJECT, DataHandle{tmpptr, len} };
    ApplicationProcess::approc->exch_data_msg(&msg, 2, COVISE_MESSAGE_OBJECT_FOUND, COVISE_MESSAGE_OBJECT_NOT_FOUND);
    coShmArray *shmarr = NULL;
    // this is a local message, so no conversion is necessary
    if (msg.type == COVISE_MESSAGE_OBJECT_FOUND)
    {
        shmarr = new coShmArray(*(int *)msg.data.data(),
                                *(shmSizeType *)(&msg.data.data()[sizeof(int)]));
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "shmarr: %d %d",
                      *(int *)msg->data,
                      *(shmSizeType *)(&msg->data[sizeof(int)]));
#endif
    } // else we probably have a socket closed message and should quit.
    return shmarr;
}
}

using namespace covise;

int coDistributedObject::set_vconstr(const char *t,
                                     coDistributedObject *(*vcfunct)(coShmArray *))
{
    VirtualConstructor *tmpptr;

    int t_no = calcType(t);
    //    cerr << "in set_vconstr with " << t_no << endl;

    if (vconstr_list == NULL)
    {
        //print_comment(__LINE__, __FILE__, "vconstr_list == NULL");
        coDistributedObject::vconstr_list = new List<VirtualConstructor>;
    }
    vconstr_list->reset();
    while ((tmpptr = vconstr_list->next()))
    {
        if (tmpptr->type == t_no)
            return 1;
    }
    tmpptr = new VirtualConstructor(t_no, vcfunct);
    vconstr_list->add(tmpptr);
    return 1;
}

const coDistributedObject *coDistributedObject::createFromShm(const coObjInfo &info)
{
    coShmArray *shmarr = ::getShmArray(info.getName());
    if (!shmarr)
        return NULL;

    const coDistributedObject *obj = createUnknown(shmarr);
    if (!obj)
        return NULL;

    if (obj->objectOk())
    {
        return obj;
    }

    delete obj;
    return NULL;
}

const coDistributedObject *coDistributedObject::createUnknown(int seg, shmSizeType offs)
{
    coShmArray *arr = new coShmArray(seg, offs);
    const coDistributedObject *obj = createUnknown(arr);
    delete arr;
    return obj;
}

const coDistributedObject *coDistributedObject::createUnknown(coShmArray *arr)
{
    VirtualConstructor *tmpptr;
    coShmArray *tmp_arr;
    int *iptr = (int *)arr->getPtr(); // pointer to the structure data
    int ltype = *iptr;

    tmp_arr = new coShmArray(arr->shm_seq_no, arr->offset);
    if (vconstr_list == NULL)
    {
        //print_comment(__LINE__, __FILE__, "vconstr_list == NULL");
        coDistributedObject::vconstr_list = new List<VirtualConstructor>;
    }
    vconstr_list->reset();
    while ((tmpptr = vconstr_list->next()))
    {
        if (tmpptr->type == ltype)
            return tmpptr->vconstr(tmp_arr);
    }
    delete tmp_arr;
    print_comment(__LINE__, __FILE__, "createUnknown failed for type id %d", ltype);
    return NULL;
}

const coDistributedObject *coDistributedObject::createUnknown() const
{
    if (name == NULL)
    {
        print_comment(__LINE__, __FILE__, "Object has no name");
        return NULL;
    }
    if (!shmarr)
        getShmArray();
    if (shmarr)
        return createUnknown(shmarr);
    else
        return NULL;
}

coDistributedObject::~coDistributedObject()
{

    if (NULL != attribs)
    {
        delete[] attribs; // if get_all_attributes was called, free up allocated space
        attribs = NULL;
    }

    if (name)
    {
        Message msg{ COVISE_MESSAGE_OBJECT_NO_LONGER_USED, DataHandle{name, strlen(name) + 1, false} };
        if (ApplicationProcess::approc)
            ApplicationProcess::approc->send_data_msg(&msg);
    }
    delete shmarr;
}

int coDistributedObject::getObjectInfo(coDoInfo **info_list) const
{
    int *iptr, *tmpiptr;
    int count, i, len;
    char *tmpcptr;
    coShmArray *tmparray;
    coDoInfo *il;
    coDoHeader *header;
    int shmarr_count;

    iptr = (int *)shmarr->getPtr(); // pointer to the structure data
    header = (coDoHeader *)iptr;
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "getObjectInfo");
//    header->print();
#endif
    count = header->get_number_of_elements();
    *info_list = new coDoInfo[count];
    shmarr_count = header->getIntHeaderSize();

    for (i = 0; i < count; i++)
    {
        il = &(*info_list)[i];
        il->type = iptr[shmarr_count++];
        il->obj_name = NULL;
        switch (il->type)
        {
        case CHARSHM:
            il->type_name = "Char";
            il->ptr = &iptr[shmarr_count];
            shmarr_count++;
            break;

        case SHORTSHM:
            il->type_name = "Short";
            il->ptr = &iptr[shmarr_count];
            shmarr_count++;
            break;

        case INTSHM:
            il->type_name = "Integer";
            il->ptr = &iptr[shmarr_count];
            shmarr_count++;
            break;

        case LONGSHM:
            il->type_name = "Long";
            il->ptr = &iptr[shmarr_count];
            // alignment !!
            shmarr_count += sizeof(long) / sizeof(int);
            if (sizeof(long) % sizeof(int))
                shmarr_count++;
            break;

        case FLOATSHM:
            il->type_name = "Float";
            il->ptr = &iptr[shmarr_count];
            // alignment !!
            shmarr_count += sizeof(float) / sizeof(int);
            if (sizeof(float) % sizeof(int))
                shmarr_count++;
            break;

        case DOUBLESHM:
            il->type_name = "Double";
            il->ptr = &iptr[shmarr_count];
            // alignment !!
            shmarr_count += sizeof(double) / sizeof(int);
            if (sizeof(double) % sizeof(int))
                shmarr_count++;
            break;

        case COVISE_NULLPTR:
            il->type_name = "Nullpointer";
            il->ptr = NULL;
            shmarr_count += 2;
            break;

        case SHMPTR:
            tmparray = new coShmArray(iptr[shmarr_count], *((shmSizeType *)&iptr[shmarr_count + 1]));
            il->ptr = tmparray->getPtr();
            tmpiptr = (int *)il->ptr;
            il->type = tmpiptr[0];
            delete tmparray;
            shmarr_count += 1+(sizeof(shmSizeType)/sizeof(int));
            switch (il->type)
            {
            case CHARSHMARRAY:
                il->type_name = "Char Array";
                break;
            case SHORTSHMARRAY:
                il->type_name = "Short Array";
                break;
            case INTSHMARRAY:
                il->type_name = "Integer Array";
                break;
            case LONGSHMARRAY:
                il->type_name = "Long Array";
                break;
            case FLOATSHMARRAY:
                il->type_name = "Float Array";
                break;
            case DOUBLESHMARRAY:
                il->type_name = "Double Array";
                break;
            case STRINGSHMARRAY:
                il->type_name = "String Array";
                break;
            case SHMPTRARRAY:
                il->type_name = "Shared Memory Pointer Array";
                break;
            default:
                il->type_name = calcTypeString(il->type);
                tmparray = new coShmArray(tmpiptr[12], tmpiptr[13]);
                tmpcptr = (char *)tmparray->getPtr();
                len = (int)strlen(&tmpcptr[8]) + 1;
                il->obj_name = new char[len];
                strcpy(il->obj_name, &tmpcptr[8]);
                delete tmparray;
                break;
            }
            break;
        }
    }

    count = getObjInfo(count, info_list);

    return count;
}

int coDistributedObject::destroy()
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "destroying object %s", name);
#endif
    Message msg{ COVISE_MESSAGE_DESTROY_OBJECT, DataHandle{name, strlen(name) + 1, false} };
    // next line changed from send_data_msg
    ApplicationProcess::approc->exch_data_msg(&msg, 2, COVISE_MESSAGE_MSG_OK, COVISE_MESSAGE_MSG_FAILED);
    //    msg->data = NULL;
    //    ApplicationProcess::approc->recv_data_msg(msg);
    if (msg.type == COVISE_MESSAGE_MSG_OK)
    {
        new_ok = 0;
        return 1;
    }
    else
    {
        print_comment(__LINE__, __FILE__, "DESTROY_OBJECT failed for %s", name);
        return 0;
    }
}

char *coDistributedObject::object_on_hosts() const
{
    char *data;

    Message msg{ COVISE_MESSAGE_OBJECT_ON_HOSTS, DataHandle{name, strlen(name) + 1, false} };
    ApplicationProcess::approc->exch_data_msg(&msg, 1, COVISE_MESSAGE_OBJECT_ON_HOSTS);
    if (msg.type == COVISE_MESSAGE_OBJECT_ON_HOSTS)
    {
        return msg.data.accessData();
    }
    else
    {
        return nullptr;
    }
}

int coDistributedObject::access(access_type acc)
{

    char *data;
    int length;

    length = sizeof(int) + (int)strlen(name) + 1;
    data = new char[length];
    *(int *)data = acc;
    sprintf(&data[sizeof(int)], "%s", name);
    Message msg(COVISE_MESSAGE_SET_ACCESS, DataHandle(data, length));
    ApplicationProcess::approc->send_data_msg(&msg);
    msg.data = DataHandle();
    ApplicationProcess::approc->recv_data_msg(&msg);
    if (acc == (ACC_READ_AND_WRITE | ACC_WRITE_ONLY))
    {
        if (msg.type == COVISE_MESSAGE_MSG_OK)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
    print_comment(__LINE__, __FILE__, "access error");
    return 0;
}

int coDistributedObject::getShmArray() const
{
    shmarr = ::getShmArray(name);
    if (shmarr)
    {
        header = (coDoHeader *)shmarr->getPtr();
    }

    return shmarr ? 0 : 1;
}

void coDistributedObject::init_header(int *size, int *no_of_allocs, int count,
                                      data_type **dt, long **ct)
{
    *size = 0;
    *no_of_allocs = 1;
    *dt = new data_type[count + 3]; // we will need two additional entries for
    *ct = new long[count + 3]; // the structure itself and the name
}

int *coDistributedObject::store_header(int size, int no_of_allocs, int count,
                                       int *shmarr_count, data_type *dt, long *ct, covise::DataHandle& idata)
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "size without header: %d", size);
    print_comment(__LINE__, __FILE__, "header size: %d", coDoHeader::getHeaderSize());
#endif

    size += coDoHeader::getHeaderSize();

    // for the data of the whole structure
    dt[0] = CHARSHMARRAY;
    ct[0] = size;
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "type: %d  size: %d", CHARSHMARRAY, size);
#endif
    // for the name of the object
    dt[no_of_allocs] = CHARSHMARRAY;
    ct[no_of_allocs] = (int)strlen(name) + 1;
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "type: %d  size: %d", CHARSHMARRAY, strlen(name) + 1);
#endif
    no_of_allocs++;
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "no_of_allocs: %d", no_of_allocs);
#endif
// now we will register the new object under its name

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "name of new  object %s", name);
#endif
    int otype = type_no;
    ShmMessage shmmsg{ name, otype, dt, ct, no_of_allocs };
    ApplicationProcess::approc->exch_data_msg(&shmmsg, 2, COVISE_MESSAGE_NEW_OBJECT_OK, COVISE_MESSAGE_NEW_OBJECT_FAILED);
    delete[] ct;
    delete[] dt;

    if (shmmsg.type != COVISE_MESSAGE_NEW_OBJECT_OK)
    {
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "error in store_header of distributed object %s", name);
#endif
        return 0; // we can do this here, all memory given back
    }

    // In the following the array that has been allocated for the structure
    // is filled. iptr points as an integer pointer to this array.
    // A char pointer could be used also, but using the int pointer makes
    // it more transparent and avoids the alignment problems that could occur
    // if a char pointer were used.

    //idata = (int *)shmmsg.data.data(); // pointer to shm-pointers //error with datahandle

    idata = shmmsg.data;
    const int* idataArray = (const int*)idata.data();

    shmarr = new coShmArray((idataArray)[0], *((shmSizeType *)&(idataArray)[1]));
    int *iptr = (int *)shmarr->getPtr(); // pointer to the structure data
    header = (coDoHeader *)iptr;
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Header before initialization");
//    header->print();
#endif

    // number of this objects type
    if (header->getObjectType() == type_no) // object existed already
    {
        object_exists = 1;
        free_list = new int[count * 2];
        current_free = 0;
        loc_version = header->get_version(); //get new version number, set by datamanager
        header->incRefCount(); // must be increased, since we are attaching

        // the name of the object should be already set and in place
        // as should be the attributes field.
        //	scount = 13; // expected below for shmarr_count
        // why 13?
    }
    else
    {
        object_exists = 0;
        header->set_object_type(type_no);
        header->set_number_of_elements(count);
        // local version number; location (iptr[5]) also used on update of object
        // in process_list of ShmMessage
        header->set_version(1);
        loc_version = 1;
        version.setPtr(shmarr->shm_seq_no,
                       shmarr->offset + header->get_version_offset());
        header->set_refcount(1);
        refcount.setPtr(shmarr->shm_seq_no,
                        shmarr->offset + header->get_refcount_offset());

        header->set_name((idataArray)[(no_of_allocs - 1) * 2],
                         *(shmSizeType *)(&(idataArray)[(no_of_allocs - 1) * 2 + 1]), name);
        header->addAttributes(0, 0);
    }
    *shmarr_count = header->getIntHeaderSize();
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Header before initialization");
// header->print();
#endif
    return iptr;
}

#ifdef INSURE
extern "C" {
void _Insight_set_option(char *, char *);
}
#endif

int coDistributedObject::store_shared_dl(int count, covise_data_list *dl)
{
    int i;
    data_type *dt = nullptr;
    long *ct = nullptr;
    int no_of_allocs, shmarr_count;
    coShmArray *tmparray = nullptr;
    coShmPtr *tmpptr = nullptr;
    int retval = 1;
    int *iptr = nullptr;

    int alignBytes; // Needed for correct alignment in first switch
    // statement. The problem occured first on Cray T3E
    // Size of allocated memory segment was to small
    // leading to corrupted data.
    // SHOULD BE CLEANED OUT SOON!!!
    // J. Rodemann, 28.05.97

    // First we determine how large the array for the structure must be
    // and how many elements we will have to put into it

    init_header(&size, &no_of_allocs, count, &dt, &ct);
    for (i = 0; i < count; i++)
    {
        switch (dl[i].type)
        {
        case CHARSHM:
            print_comment(__LINE__, __FILE__, "CHARSHM");
            size += sizeof(int) + sizeof(int); // alignment !!
            break;

        case SHORTSHM:
            print_comment(__LINE__, __FILE__, "SHORTSHM");
            size += sizeof(int) + sizeof(int); // alignment !!
            break;

        case INTSHM:
            print_comment(__LINE__, __FILE__, "INTSHM");
            size += sizeof(int) + sizeof(int);
            break;

        case LONGSHM:
            print_comment(__LINE__, __FILE__, "LONGSHM");
            size += sizeof(long) + sizeof(int);
            alignBytes = sizeof(long) % sizeof(int);
            if (alignBytes)
                size += alignBytes;
            break;

        case FLOATSHM:
            print_comment(__LINE__, __FILE__, "FLOATSHM");
            size += sizeof(float) + sizeof(int);
            alignBytes = sizeof(float) % sizeof(int);
            if (alignBytes)
                size += alignBytes;
            break;

        case DOUBLESHM:
            print_comment(__LINE__, __FILE__, "DOUBLESHM");
            size += sizeof(double) + sizeof(int);
            alignBytes = sizeof(double) % sizeof(int);
            if (alignBytes)
                size += alignBytes;
            break;

        case CHARSHMARRAY:
        case SHORTSHMARRAY:
        case INTSHMARRAY:
        case LONGSHMARRAY:
        case FLOATSHMARRAY:
        case DOUBLESHMARRAY:
        case STRINGSHMARRAY:
        case SHMPTRARRAY:
        {
            print_comment(__LINE__, __FILE__, "ARRAY");
            tmparray = (coShmArray *)dl[i].ptr;
            int numElem = tmparray->get_length() + 1;
            if (numElem < 1)
                numElem = 1;
            dt[no_of_allocs] = dl[i].type;
            ct[no_of_allocs] = numElem;
            no_of_allocs++;
            size += 2 * sizeof(int)+sizeof(shmSizeType);
            break;
        }

        case DISTROBJ:
            print_comment(__LINE__, __FILE__, "DISTROBJ");
            size += 2 * sizeof(int)+sizeof(shmSizeType);

        default:
            print_comment(__LINE__, __FILE__, "default: Type Error");
            break;
        }
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "type: %d  size: %d", dl[i].type, size);
#endif
    }

#ifdef INSURE

    _Insight_set_option("runtime", "off");
#endif

    DataHandle idata;
    ///// This calls the CRB and gets the object !!!
    iptr = store_header(size, no_of_allocs, count, &shmarr_count, dt, ct, idata);
    const int* idataArray = (const int *)idata.data();
    if (iptr == NULL)
    {
        print_comment(__LINE__, __FILE__, "Error in store_header called by store_shared");
        return 0;
    }

    // here we count the usage of the allocated elements:
    int alloc_count = 0;

    for (i = 0; i < count; i++)
    {
        iptr[shmarr_count] = dl[i].type;
        shmarr_count++;
        switch (dl[i].type)
        {
        case CHARSHM:
            tmpptr = (coShmPtr *)dl[i].ptr;
            tmpptr->setPtr(shmarr->shm_seq_no,
                           shmarr->offset + (shmarr_count - 1) * sizeof(int));
            shmarr_count++; // alignment !!
            break;

        case SHORTSHM:
            tmpptr = (coShmPtr *)dl[i].ptr;
            tmpptr->setPtr(shmarr->shm_seq_no,
                           shmarr->offset + (shmarr_count - 1) * sizeof(int));
            shmarr_count++; // alignment !!
            break;

        case INTSHM:
            tmpptr = (coShmPtr *)dl[i].ptr;
            tmpptr->setPtr(shmarr->shm_seq_no,
                           shmarr->offset + (shmarr_count - 1) * sizeof(int));
            shmarr_count++;
            break;

        case LONGSHM:
            tmpptr = (coShmPtr *)dl[i].ptr;
            tmpptr->setPtr(shmarr->shm_seq_no,
                           shmarr->offset + (shmarr_count - 1) * sizeof(int));
            // alignment !!
            shmarr_count += sizeof(long) / sizeof(int);
            if (sizeof(long) % sizeof(int))
                shmarr_count++;
            break;

        case FLOATSHM:
            tmpptr = (coShmPtr *)dl[i].ptr;
            tmpptr->setPtr(shmarr->shm_seq_no,
                           shmarr->offset + (shmarr_count - 1) * sizeof(int));
            // alignment !!
            shmarr_count += sizeof(float) / sizeof(int);
            if (sizeof(float) % sizeof(int))
                shmarr_count++;
            break;

        case DOUBLESHM:
            tmpptr = (coShmPtr *)dl[i].ptr;
            tmpptr->setPtr(shmarr->shm_seq_no,
                           shmarr->offset + (shmarr_count - 1) * sizeof(int));
            // alignment !!
            shmarr_count += sizeof(double) / sizeof(int);
            if (sizeof(double) % sizeof(int))
                shmarr_count++;
            break;

        // Arrays are always stored indirectly:
        // a SHMPTR that points to the actual Array
        case CHARSHMARRAY:
        case SHORTSHMARRAY:
        case INTSHMARRAY:
        case LONGSHMARRAY:
        case FLOATSHMARRAY:
        case DOUBLESHMARRAY:
        case STRINGSHMARRAY:
        case SHMPTRARRAY:
            if (object_exists)
            {
                free_list[current_free++] = iptr[shmarr_count];
                free_list[current_free++] = iptr[shmarr_count + 1];
            }
            iptr[shmarr_count - 1] = SHMPTR;
            iptr[shmarr_count++] = idataArray[2 + alloc_count * 2];
            *(shmSizeType *)(&iptr[shmarr_count++]) = *(shmSizeType *)(&idataArray[2 + alloc_count * 2 + 1]);
            if(sizeof(shmSizeType) > sizeof(int))
                shmarr_count++;

            tmparray = (coShmArray *)dl[i].ptr;
            tmparray->setPtr(idataArray[2 + alloc_count * 2],
                             *(shmSizeType *)(&idataArray[2 + alloc_count * 2 + 1]));
            alloc_count++;
            break;

        case DISTROBJ:
            if (dl[i].ptr != NULL)
            {
                tmparray = ((coDistributedObject *)dl[i].ptr)->shmarr;
                iptr[shmarr_count - 1] = SHMPTR;
                iptr[shmarr_count++] = tmparray->shm_seq_no;
                *(shmSizeType *)(&iptr[shmarr_count++]) = tmparray->offset;
                if(sizeof(shmSizeType) > sizeof(int))
                    shmarr_count++;
            }
            else
            {
                iptr[shmarr_count - 1] = COVISE_NULLPTR;
                iptr[shmarr_count++] = 0;
                *(shmSizeType *)(&iptr[shmarr_count++]) = 0;
                if(sizeof(shmSizeType) > sizeof(int))
                    shmarr_count++;
            }
            break;
        }
    }

#ifdef INSURE
    _Insight_set_option("runtime", "on");
#endif
    // current_free = number of entries in free_list ( = 2 * number of elements to free)

    if (current_free)
    {
        Message msg(COVISE_MESSAGE_SHM_FREE, DataHandle((char *)free_list, current_free * sizeof(int)));
        ApplicationProcess::approc->send_data_msg(&msg);
    }
    delete free_list;

    return retval;
}

int coDistributedObject::update_shared_dl(int count, covise_data_list *dl)
{
    int i;
    coShmArray *tmparray;
    int retval = 1;

    // In the following the array that has been allocated for the structure
    // is filled. iptr points as an integer pointer to this array.
    // A char pointer could be used also, but using the int pointer makes
    // it more transparent and avoids the alignment problems that could occur
    // if a char pointer were used.

    int *iptr = (int *)shmarr->getPtr(); // pointer to the structure data
    coDoHeader *header = (coDoHeader *)iptr;
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "update_shared");
//    header->print();
#endif

    header->set_object_type(type_no);
    header->set_number_of_elements(count);
    header->set_version(++loc_version);
    int shmarr_count = header->getIntHeaderSize();

    for (i = 0; i < count; i++)
    {
        iptr[shmarr_count] = dl[i].type;
        shmarr_count++;
        switch (dl[i].type)
        {
        case CHARSHM:
            shmarr_count++; // alignment !!
            break;
        case SHORTSHM:
            shmarr_count++; // alignment !!
            break;
        case INTSHM:
            shmarr_count++;
            break;
        case LONGSHM:
            // alignment !!
            shmarr_count += sizeof(long) / sizeof(int);
            if (sizeof(long) % sizeof(int))
                shmarr_count++;
            break;
        case FLOATSHM:
            // alignment !!
            shmarr_count += sizeof(float) / sizeof(int);
            if (sizeof(float) % sizeof(int))
                shmarr_count++;
            break;
        case DOUBLESHM:
            // alignment !!
            shmarr_count += sizeof(double) / sizeof(int);
            if (sizeof(double) % sizeof(int))
                shmarr_count++;
            break;
        // Arrays are always stored indirectly:
        // a SHMPTR that points to the actual Array
        case CHARSHMARRAY:
        case SHORTSHMARRAY:
        case INTSHMARRAY:
        case LONGSHMARRAY:
        case FLOATSHMARRAY:
        case DOUBLESHMARRAY:
        case STRINGSHMARRAY:
        case SHMPTRARRAY:
            iptr[shmarr_count - 1] = SHMPTR;
            tmparray = (coShmArray *)dl[i].ptr;
            iptr[shmarr_count++] = tmparray->shm_seq_no;
            *(shmSizeType *)(&iptr[shmarr_count++]) = tmparray->offset;
            if(sizeof(shmSizeType) > sizeof(int))
                shmarr_count++;
            break;
        case DISTROBJ:
            if (dl[i].ptr != NULL)
            {
                tmparray = ((coDistributedObject *)dl[i].ptr)->shmarr;
                iptr[shmarr_count - 1] = SHMPTR;
                iptr[shmarr_count++] = tmparray->shm_seq_no;
                *(shmSizeType *)(&iptr[shmarr_count++]) = tmparray->offset;
                if(sizeof(shmSizeType) > sizeof(int))
                    shmarr_count++;
            }
            else
            {
                iptr[shmarr_count - 1] = COVISE_NULLPTR;
                iptr[shmarr_count++] = 0;
                *(shmSizeType *)(&iptr[shmarr_count++]) = 0;
                if(sizeof(shmSizeType) > sizeof(int))
                    shmarr_count++;
            }
            break;
        }
    }

    // now we will register the new version of this object under its name

    int len = (int)strlen(name) + 1;
    char *data = new char[len];
    strcpy(data, name);
    Message msg(COVISE_MESSAGE_NEW_OBJECT_VERSION, DataHandle(data, len));
    ApplicationProcess::approc->send_data_msg(&msg);
    return retval;
}

int coDistributedObject::restore_header(int **iptr, int count, int *shmarr_count,
                                        int *sn, shmSizeType *of)
{
    int *tmp_iptr = (int *)shmarr->getPtr();
    *of = shmarr->offset;
    *sn = shmarr->shm_seq_no;

    const char *tmpname;
    coDoHeader *header = (coDoHeader *)tmp_iptr;
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "restore_header");
//    header->print();
#endif

    if (header->get_number_of_elements() != count)
    {
        print_comment(__LINE__, __FILE__, "number of elements to read <> count");
        print_comment(__LINE__, __FILE__, "%d instead of %d", header->get_number_of_elements(), count);
        return 0;
    }

    if (header->get_version())
    {
        version.setPtr(*sn, *of + header->get_version_offset());
        loc_version = version;
    }
    else
        return 0;

    if (header->get_refcount())
        refcount.setPtr(*sn, *of + header->get_refcount_offset());
    else
        return 0;

    if ((tmpname = header->getName()))
    {
        if (name == NULL)
        {
            int len = (int)strlen(tmpname) + 1;
            name = new char[len];
            strcpy(name, tmpname);
        }
        else if (strcmp(tmpname, name) != 0)
        {
            print_comment(__LINE__, __FILE__, "%s to read instead of %s ", tmpname, name);
            return 0;
        }
    }
    else
        return 0;

    if (header->get_attr_type() == SHMPTR)
    {
        attributes = header->getAttributes();
        if (attributes == NULL)
            return 0;
    }
    else
        attributes = NULL;

    *iptr = tmp_iptr;
    *shmarr_count = header->getIntHeaderSize();

    return 1;
}

int coDistributedObject::restore_shared_dl(int count, covise_data_list *dl)
{
    /// if we call a c'tor without SHM array, this is not ok
    if (shmarr == NULL || shmarr == (coShmArray *)-1)
    {
        new_ok = 0;
        return 0;
    }

    coShmPtr *tmpptr;
    coShmArray *tmparray;
    int i, *iptr, shmarr_count;
    int sn;
    shmSizeType of;

    if (restore_header(&iptr, count, &shmarr_count, &sn, &of) == 0)
    {
        return 0;
    }

    for (i = 0; i < count; i++)
    {
        if (iptr[shmarr_count] == dl[i].type)
        {
            shmarr_count++;
            switch (dl[i].type)
            {
            case CHARSHM:
                tmpptr = (coShmPtr *)dl[i].ptr;
                tmpptr->setPtr(sn, of + (shmarr_count - 1) * sizeof(int));
                shmarr_count++; // alignment!!
                break;
            case SHORTSHM:
                tmpptr = (coShmPtr *)dl[i].ptr;
                tmpptr->setPtr(sn, of + (shmarr_count - 1) * sizeof(int));
                shmarr_count++; // alignment!!
                break;
            case INTSHM:
                tmpptr = (coShmPtr *)dl[i].ptr;
                tmpptr->setPtr(sn, of + (shmarr_count - 1) * sizeof(int));
                shmarr_count++;
                break;
            case LONGSHM:
                tmpptr = (coShmPtr *)dl[i].ptr;
                tmpptr->setPtr(sn, of + (shmarr_count - 1) * sizeof(int));
                // alignment!!
                shmarr_count += sizeof(long) / sizeof(int);
                if (sizeof(long) % sizeof(int))
                    shmarr_count++;
                break;
            case FLOATSHM:
                tmpptr = (coShmPtr *)dl[i].ptr;
                tmpptr->setPtr(sn, of + (shmarr_count - 1) * sizeof(int));
                // alignment!!
                shmarr_count += sizeof(float) / sizeof(int);
                if (sizeof(float) % sizeof(int))
                    shmarr_count++;
                break;
            case DOUBLESHM:
                tmpptr = (coShmPtr *)dl[i].ptr;
                tmpptr->setPtr(sn, of + (shmarr_count - 1) * sizeof(int));
                // alignment!!
                shmarr_count += sizeof(double) / sizeof(int);
                if (sizeof(double) % sizeof(int))
                    shmarr_count++;
                break;
            }
        }
        else
        {
            if (iptr[shmarr_count] == SHMPTR || iptr[shmarr_count] == COVISE_NULLPTR)
            {
                shmarr_count++;

                // if I do the following dependent on the type, type-checking
                // will be done in setPtr automatically

                if (iptr[shmarr_count - 1] & 0x80 && iptr[shmarr_count] == 0)
                {
                    // this is an empty array, data not yet available
                    *(void **)dl[i].ptr = NULL;
                    shmarr_count += 2;
                }
                else
                {
                    switch (dl[i].type)
                    {
                    case CHARSHMARRAY:
                        ((coCharShmArray *)dl[i].ptr)->setPtr(iptr[shmarr_count], *(shmSizeType *)(&iptr[shmarr_count + 1]));
                        shmarr_count += 1+(sizeof(shmSizeType)/sizeof(int));
                        break;
                    case SHORTSHMARRAY:
                        ((coShortShmArray *)dl[i].ptr)->setPtr(iptr[shmarr_count], *(shmSizeType *)(&iptr[shmarr_count + 1]));
                        shmarr_count += 1+(sizeof(shmSizeType)/sizeof(int));
                        break;
                    case INTSHMARRAY:
                        ((coIntShmArray *)dl[i].ptr)->setPtr(iptr[shmarr_count], *(shmSizeType *)(&iptr[shmarr_count + 1]));
                        shmarr_count += 1+(sizeof(shmSizeType)/sizeof(int));
                        break;
                    case LONGSHMARRAY:
                        ((coLongShmArray *)dl[i].ptr)->setPtr(iptr[shmarr_count], *(shmSizeType *)(&iptr[shmarr_count + 1]));
                        shmarr_count += 1+(sizeof(shmSizeType)/sizeof(int));
                        break;
                    case FLOATSHMARRAY:
                        ((coFloatShmArray *)dl[i].ptr)->setPtr(iptr[shmarr_count], *(shmSizeType *)(&iptr[shmarr_count + 1]));
                        shmarr_count += 1+(sizeof(shmSizeType)/sizeof(int));
                        break;
                    case DOUBLESHMARRAY:
                        ((coDoubleShmArray *)dl[i].ptr)->setPtr(iptr[shmarr_count], *(shmSizeType *)(&iptr[shmarr_count + 1]));
                        shmarr_count += 1+(sizeof(shmSizeType)/sizeof(int));
                        break;
                    case STRINGSHMARRAY:
                        ((coStringShmArray *)dl[i].ptr)->setPtr(iptr[shmarr_count], *(shmSizeType *)(&iptr[shmarr_count + 1]));
                        shmarr_count += 1+(sizeof(shmSizeType)/sizeof(int));
                        break;
                    case SHMPTRARRAY:
                        ((coShmPtrArray *)dl[i].ptr)->setPtr(iptr[shmarr_count], *(shmSizeType *)(&iptr[shmarr_count + 1]));
                        shmarr_count += 1+(sizeof(shmSizeType)/sizeof(int));
                        break;
                    case DISTROBJ:
                        tmparray = new coShmArray(iptr[shmarr_count],
                                                  *(shmSizeType *)(&iptr[shmarr_count + 1]));
                        ((coDistributedObject *)dl[i].ptr)->shmarr = tmparray;
                        ((coDistributedObject *)dl[i].ptr)->header = (coDoHeader *)tmparray->getPtr();
                        if (((coDistributedObject *)dl[i].ptr)->rebuildFromShm() == 0)
                        {
                            print_comment(__LINE__, __FILE__, "rebuildFromShm failed");
                            return 0;
                        }
                        shmarr_count += 1+(sizeof(shmSizeType)/sizeof(int));
                        break;
                    case UNKNOWN:
                    case COVISE_OPTIONAL:
                        if (iptr[shmarr_count - 1] == COVISE_NULLPTR)
                        {
                            // does not work on 64bit *(long *)dl[i].ptr = 0;
                            *(void **)dl[i].ptr = NULL;
                        }
                        else
                        {
                            tmparray = new coShmArray(iptr[shmarr_count],
                                                      *(shmSizeType *)(&iptr[shmarr_count + 1]));
                            *((const coDistributedObject **)dl[i].ptr) = createUnknown(tmparray);
                            if (!(*(coDistributedObject **)dl[i].ptr)->objectOk()) // rebuildFromShm already done in reateUnknown, don't do it again
                            {
                                print_comment(__LINE__, __FILE__, "rebuildFromShm failed");
                                return 0;
                            }
                        }
                        shmarr_count += 1+(sizeof(shmSizeType)/sizeof(int));
                        break;
                    }
                }
            }
            else
            {
                print_comment(__LINE__, __FILE__, "type inconsistency while restoring object from shared memory");
                print_comment(__LINE__, __FILE__, "found %d instead of %ld", iptr[shmarr_count], dl[i].type);
                return 0;
            }
        }
    }
    return 1;
}

void coDistributedObject::setType(const char *tname, const char *long_name)
{
    char tno[7];
    int i;

    strncpy(type_name, tname, 6);
    type_name[6] = '\0';
    tno[6] = '\0';
    for (i = 0; i < 6; i++)
    {
        if (type_name[i] >= 'a' && type_name[i] <= 'z')
        {
            type_name[i] -= 'a' - 'A';
        }
        else if (type_name[i] < 'A' || type_name[i] > '_')
        {
            type_name[i] = '_';
        }
        tno[i] = type_name[i] - 'A';
    }
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Type name : %s", type_name);
#endif
    type_no = 0;
    type_no |= tno[0];
    type_no |= tno[1] << 5;
    type_no |= tno[2] << 10;
    type_no |= tno[3] << 15;
    type_no |= tno[4] << 20;
    type_no |= tno[5] << 25;
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Type no   : %d", type_no);
#endif
    int len = sizeof(int) + 7 + (int)strlen(long_name) + 1;
    char *data = new char[len];
    *(int *)data = type_no;
    memcpy(&data[sizeof(int)], tno, 7);
    memcpy(&data[sizeof(int) + 7], long_name, strlen(long_name) + 1);

    delete[] data;
}

/*
int coDistributedObject::get_transfer_arrays()
{
    return xfer_arrays;
}
*/

int coDistributedObject::calcType(const char *tname)
{
    char tmp_name[7] = { '\0', '\0', '\0', '\0', '\0', '\0', '\0' };
    char tno[6];
    int tmp_no;
    int i;

    strncpy(tmp_name, tname, 6);
    tmp_name[6] = '\0';
    for (i = 0; i < 6; i++)
    {
        if (tmp_name[i] >= 'a' && tmp_name[i] <= 'z')
        {
            tmp_name[i] -= 'a' - 'A';
        }
        else if (tmp_name[i] < 'A' || tmp_name[i] > '_')
        {
            tmp_name[i] = '_';
        }
        tno[i] = tmp_name[i] - 'A';
    }
    tmp_no = 0;
    tmp_no |= tno[0];
    tmp_no |= tno[1] << 5;
    tmp_no |= tno[2] << 10;
    tmp_no |= tno[3] << 15;
    tmp_no |= tno[4] << 20;
    tmp_no |= tno[5] << 25;

    return tmp_no;
}

char *coDistributedObject::calcTypeString(int tno)
{
    char *tmp_name = new char[7];
#ifdef DEBUG
    int tmp_tno = tno;
#endif

    tmp_name[0] = 'A' + (tno & 0x1f);
    tno >>= 5;
    tmp_name[1] = 'A' + (tno & 0x1f);
    tno >>= 5;
    tmp_name[2] = 'A' + (tno & 0x1f);
    tno >>= 5;
    tmp_name[3] = 'A' + (tno & 0x1f);
    tno >>= 5;
    tmp_name[4] = 'A' + (tno & 0x1f);
    tno >>= 5;
    tmp_name[5] = 'A' + (tno & 0x1f);
    tno >>= 5;
    tmp_name[6] = '\0';

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Type no : %d name: %s", tmp_tno, tmp_name);
#endif

    return tmp_name;
}

void coDistributedObject::addAttribute(const char *attr_name, const char *attr_val)
{
    int attr_len, sn, shmfree[8];
    shmSizeType of;
    long ct[2];
    data_type dt[2];
    ShmMessage *shmmsg;
    coStringShmArray *tmparr;
    coCharShmArray *charr;
    coDoHeader *header;
    char *tmpstr;
    if (NULL == attr_name)
    {
        print_comment(__LINE__, __FILE__, "error in addAttribute for distributed object ");
        return;
    }
    if (NULL == attr_val)
    {
        print_comment(__LINE__, __FILE__, "error in addAttribute for attr_name=%s, attr_val==NULL", attr_name);
        return;
    }

    attr_len = (int)strlen(attr_name) + 1 + (int)strlen(attr_val) + 1;
    if (attributes)
    {
        print_comment(__LINE__, __FILE__, "attributes != NULL");
        ct[0] = attributes->get_length() + 1;
    }
    else
    {
        print_comment(__LINE__, __FILE__, "attributes == NULL");
        ct[0] = 1;
    }
    dt[0] = STRINGSHMARRAY;
    ct[1] = attr_len;
    dt[1] = CHARSHMARRAY;
    shmmsg = new ShmMessage(dt, ct, 2);
    ApplicationProcess::approc->exch_data_msg(shmmsg, 2, COVISE_MESSAGE_MALLOC_LIST_OK, COVISE_MESSAGE_MALLOC_FAILED);
    if (shmmsg->type != COVISE_MESSAGE_MALLOC_LIST_OK)
    {
        print_comment(__LINE__, __FILE__, "error in addAttribute for distributed object %s", name);
        delete shmmsg;
        return;
    }
    const char *cdata = shmmsg->data.data(); // pointer to shm-pointers
    int seq = *(int *)cdata;
    cdata +=sizeof(int);
    shmSizeType offset = *(shmSizeType *)cdata;
    cdata +=sizeof(shmSizeType);
    tmparr = new coStringShmArray(seq, offset);
    seq = *(int *)cdata;
    cdata +=sizeof(int);
    offset = *(shmSizeType *)cdata;
    cdata +=sizeof(shmSizeType);
    charr = new coCharShmArray(seq, offset);
    delete shmmsg;
    tmpstr = new char[attr_len];
    sprintf(tmpstr, "%s:%s", attr_name, attr_val);

    // convert attribute name to upper case
    for (char *p = tmpstr; *p != ':'; ++p)
        *p = toupper(*p);

    // print_comment(__LINE__, __FILE__, tmpstr);
    charr->setString(tmpstr);
    delete[] tmpstr;
    if (attributes)
    {
        print_comment(__LINE__, __FILE__, "attributes != NULL");
        ArrayLengthType i;
        for (i = 0; i < attributes->get_length(); i++)
        {
            attributes->stringPtrGet(i, &sn, &of);
            tmparr->stringPtrSet(i, sn, of);
            attributes->stringPtrSet(i, 0, 0); // clear reference
        }
        tmparr->stringPtrSet(i, charr->get_shm_seq_no(), charr->get_offset());
        shmfree[0] = attributes->get_shm_seq_no();
        *(shmSizeType *)(&shmfree[1]) = attributes->get_offset();
        Message msg{ COVISE_MESSAGE_SHM_FREE, DataHandle{(char*)shmfree, sizeof(int) + sizeof(shmSizeType), false} };
        ApplicationProcess::approc->send_data_msg(&msg);
    }
    else
    {
        print_comment(__LINE__, __FILE__, "attributes == NULL");
        tmparr->stringPtrSet(0, charr->get_shm_seq_no(), charr->get_offset());
    }
    delete attributes;
    attributes = tmparr;
    delete charr;

    header = (coDoHeader *)shmarr->getPtr();
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "addAttribute");
//    header->print();
#endif
    header->addAttributes(attributes->get_shm_seq_no(), attributes->get_offset());
}

void coDistributedObject::addAttributes(int no, const char *const *attr_name,
                                        const char *const *attr_val)
{
    int *attr_len, *idata, sn, shmfree[8];
    shmSizeType of;
    long *ct;
    data_type *dt;
    ShmMessage *shmmsg;
    coStringShmArray *tmparr;
    coCharShmArray *charr;
    coDoHeader *header;
    char *tmpstr;

    print_comment(__LINE__, __FILE__, "ATTR set for %s:", name);
    for (int i = 0; i < no; i++)
    {
        print_comment(__LINE__, __FILE__, "          %s -> %s", attr_name[i], attr_val[i]);
    }
    attr_len = new int[no];
    for (int i = 0; i < no; i++)
        attr_len[i] = (int)strlen(attr_name[i]) + 1 + (int)strlen(attr_val[i]) + 1;
    ct = new long[no + 1];
    dt = new data_type[no + 1];

    if (attributes)
        ct[0] = attributes->get_length() + no;
    else
        ct[0] = no;
    dt[0] = STRINGSHMARRAY;
    for (int i = 0; i < no; i++)
    {
        ct[i + 1] = attr_len[i];
        dt[i + 1] = CHARSHMARRAY;
    }
    shmmsg = new ShmMessage(dt, ct, no + 1);
    ApplicationProcess::approc->exch_data_msg(shmmsg, 2, COVISE_MESSAGE_MALLOC_LIST_OK, COVISE_MESSAGE_MALLOC_FAILED);
    if (shmmsg->type != COVISE_MESSAGE_MALLOC_LIST_OK)
    {
        print_comment(__LINE__, __FILE__, "error in addAttribute for distributed object %s", name);
        delete shmmsg;
        return;
    }
    idata = (int *)shmmsg->data.data(); // pointer to shm-pointers
    
    const char *cdata = shmmsg->data.data(); // pointer to shm-pointers
    int seq = *(int *)cdata;
    cdata +=sizeof(int);
    shmSizeType offset = *(shmSizeType *)cdata;
    cdata +=sizeof(shmSizeType);

    tmparr = new coStringShmArray(seq, offset);
    if (attributes)
    {
        ArrayLengthType i;
        for (i = 0; i < attributes->get_length(); i++)
        {
            attributes->stringPtrGet(i, &sn, &of);
            print_comment(__LINE__, __FILE__, "sn: %d of: %d", sn, of);
            tmparr->stringPtrSet(i, sn, of);
            attributes->stringPtrSet(i, 0, 0); // clear reference
        }
        for (int j = 0; j < no; j++)
        {
            seq = *(int*)cdata;
            cdata += sizeof(int);
            offset = *(shmSizeType*)cdata;
            cdata += sizeof(shmSizeType);
            charr = new coCharShmArray(seq, offset);
            tmpstr = new char[attr_len[j]];
            sprintf(tmpstr, "%s:%s", attr_name[j], attr_val[j]);

            // convert attribute name to upper case
            for (char* p = tmpstr; *p != ':'; ++p)
                * p = toupper(*p);

            //print_comment(__LINE__, __FILE__, tmpstr);
            charr->setString(tmpstr);
            delete[] tmpstr;
            tmparr->stringPtrSet(i + j, charr->get_shm_seq_no(),
                charr->get_offset());
            delete charr;
        }
        // local message: no conversion necessary:
        shmfree[0] = attributes->get_shm_seq_no();
        *(shmSizeType*)(&shmfree[1]) = attributes->get_offset();
        Message msg{ COVISE_MESSAGE_SHM_FREE, DataHandle{(char*)shmfree, sizeof(int) + sizeof(shmSizeType), false } };
        ApplicationProcess::approc->send_data_msg(&msg);
    }
    else
    {
        for (int j = 0; j < no; j++)
        {
            seq = *(int *)cdata;
            cdata +=sizeof(int);
            offset = *(shmSizeType *)cdata;
            cdata +=sizeof(shmSizeType);
            charr = new coCharShmArray(seq, offset);;
            tmpstr = new char[attr_len[j]];
            sprintf(tmpstr, "%s:%s", attr_name[j], attr_val[j]);

            // convert attribute name to upper case
            for (char *p = tmpstr; *p != ':'; ++p)
                *p = toupper(*p);

            //print_comment(__LINE__, __FILE__, tmpstr);
            charr->setString(tmpstr);
            delete[] tmpstr;
            tmparr->stringPtrSet(j, charr->get_shm_seq_no(),
                                 charr->get_offset());
            delete charr;
        }
    }
    attributes = tmparr;
    delete shmmsg;
    delete[] attr_len;
    delete[] ct;
    delete[] dt;

    header = (coDoHeader *)shmarr->getPtr();
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "addAttributes");
//    header->print();
#endif
    header->addAttributes(attributes->get_shm_seq_no(), attributes->get_offset());
}

/* New Version by Uwe Woessner below!
char *coDistributedObject::getAttribute(char *attr_name) {
    int i;
    coShmArray *tmp_array;
    coCharShmArray *tmp_ch;
    int *idata;
    char *chdata, *tmpstr, *attr_ptr;
    char *tmp_str;

    if(!attributes) {
   // FUCK! sprintf(tmp_str, "ATTR get for %s: %s -> failed", name, attr_name);
// D.R. print_comment(__LINE__, __FILE__, tmp_str);
return NULL;
}
tmp_array = new coShmArray(attributes->get_shm_seq_no(), attributes->get_offset());
idata = (int *)tmp_array->getDataPtr();
delete tmp_array;
for(i = 0;i < attributes->get_length();i++) {
attr_ptr = attr_name;
tmp_ch = new coCharShmArray(idata[2 * i], idata[2 * i + 1]);
chdata = (char *)tmp_ch->getDataPtr();
while(*attr_ptr == *chdata && *chdata != ':') {
attr_ptr++;
chdata++;
}
if(*chdata == ':') {
chdata++;
tmpstr = new char[strlen(chdata) + 1];
strcpy(tmpstr, chdata);
delete tmp_ch;
tmp_str = new char[strlen(tmpstr)+strlen(attr_name)+strlen(name) + 100];
//sprintf(tmp_str, "ATTR get: %s -> %s object: %s", attr_name, tmpstr,name);
//fprintf(stderr,"%s\n",tmp_str);
print_comment(__LINE__, __FILE__, tmp_str);
delete tmp_str;
return tmpstr;
}
delete tmp_ch;
}
// FUCK! sprintf(tmp_str, "ATTR get: %s -> failed", attr_name);
// D.R.  print_comment(__LINE__, __FILE__, tmp_str);
return NULL;
} */

const char *coDistributedObject::getAttribute(const char *attr_name) const
{
    int i;
    coShmArray *tmp_array;
    coCharShmArray *tmp_ch;
    int *idata, len;
    char *chdata;

    if (!attributes)
        return NULL;
    tmp_array = new coShmArray(attributes->get_shm_seq_no(), attributes->get_offset());
    idata = (int *)tmp_array->getDataPtr();
    delete tmp_array;
    len = (int)strlen(attr_name);
    int es = (sizeof(int)+sizeof(shmSizeType))/sizeof(int);

    // run backwards: get most recent attrib if multiple were attached
    for (i = attributes->get_length() - 1; i >= 0; i--)
    {
        tmp_ch = new coCharShmArray(idata[es * i], *(shmSizeType *)(&idata[es * i + 1]));
        chdata = (char *)tmp_ch->getDataPtr();
        delete tmp_ch;
        if (strncasecmp(chdata, attr_name, len) == 0)
        {
            //while(*chdata != ':')
            //   chdata++;
            // falsch !!!!!
            chdata += len;
            if (*chdata == ':') // wichtig, da oben nur ein strncmp
            {
                chdata++;
                return (chdata);
            }
        }
    }
    return NULL;
}

int coDistributedObject::getNumAttributes() const
{
    if (attributes)
        return attributes->get_length();
    return 0;
}

int coDistributedObject::getAllAttributes(const char ***name,
                                          const char ***content) const
{
    int no_of_attr;
    int i;
    coShmArray *tmp_array;
    coCharShmArray *tmp_ch;
    int *idata, size = 0;
    char *chdata, **chdatas, *cp;

    if (!attributes)
        return 0;

    if (attribs != NULL)
    {
        delete[] attribs;
        attribs = NULL;
    }

    tmp_array = new coShmArray(attributes->get_shm_seq_no(), attributes->get_offset());
    idata = (int *)tmp_array->getDataPtr();
    delete tmp_array;
    no_of_attr = attributes->get_length();
    chdatas = new char *[no_of_attr];
    
    int es = (sizeof(int)+sizeof(shmSizeType))/sizeof(int);

    for (i = 0; i < no_of_attr; i++)
    {
        tmp_ch = new coCharShmArray(idata[es * i], *(shmSizeType *)(&idata[es * i + 1]));
        chdatas[i] = (char *)tmp_ch->getDataPtr();
        size += (int)strlen(chdatas[i]) + 1;
        delete tmp_ch;
    }

    // Platz fuer Attribute und Pointer, die zurueckgegeben werden
    // Dieser Platz wird beim Delete des Objekts wieder freigegeben

    cp = attribs = new char[size + 2 * no_of_attr * sizeof(char *) + 16];
    *name = (const char **)(cp + size + SIZEOF_ALIGNMENT - ((unsigned long long)(cp + size)) % SIZEOF_ALIGNMENT);
    *content = (*name + no_of_attr);
    //*name = new char *[no_of_attr];
    //*content = new char *[no_of_attr];

    for (i = 0; i < no_of_attr; i++)
    {
        strcpy(cp, chdatas[i]);
        //fprintf(stderr,"get_all_a %d:%s     %s\n",i,chdatas[i],name);
        (*name)[i] = chdata = cp;
        while (*chdata != ':')
            chdata++;
        *chdata = '\0';
        chdata++;
        (*content)[i] = chdata;
        cp += strlen(chdatas[i]) + 1;
    }
    delete[] chdatas;
    return no_of_attr;
}

namespace covise
{

void PackElement_print(PackElement *th)
{
    print_comment(__LINE__, __FILE__, "--------------- PackElement------------");
    print_comment(__LINE__, __FILE__, "shm_seq_no: %d", th->shm_seq_no);
    print_comment(__LINE__, __FILE__, "offset:     %d", th->offset);
    print_comment(__LINE__, __FILE__, "type:       %d", th->type);
    print_comment(__LINE__, __FILE__, "size:       %d", th->size);
    print_comment(__LINE__, __FILE__, "length:     %d", th->length);
    print_comment(__LINE__, __FILE__, "---------------------------------------");
}
}

void coDoHeader::print()
{
    print_comment(__LINE__, __FILE__, "--------------- Objectheader ------------");
    print_comment(__LINE__, __FILE__, "object_type:                      %d", object_type);
    print_comment(__LINE__, __FILE__, "number_of_bytes:                  %d", number_of_bytes);
    print_comment(__LINE__, __FILE__, "number_of_elements_type:          %d", number_of_elements_type);
    print_comment(__LINE__, __FILE__, "number_of_elements:               %d", number_of_elements);
    print_comment(__LINE__, __FILE__, "version_type:                     %d", version_type);
    print_comment(__LINE__, __FILE__, "version:                          %d", version);
    print_comment(__LINE__, __FILE__, "refcount_type:                    %d", refcount_type);
    print_comment(__LINE__, __FILE__, "refcount:                         %d", refcount);
    print_comment(__LINE__, __FILE__, "name_type:                        %d", name_type);
    print_comment(__LINE__, __FILE__, "name_shm_seq_no:                  %d", name_shm_seq_no);
    print_comment(__LINE__, __FILE__, "name_offset:                      %d", name_offset);
    print_comment(__LINE__, __FILE__, "attr_type:                        %d", attr_type);
    print_comment(__LINE__, __FILE__, "attr_shm_seq_no:                  %d", attr_shm_seq_no);
    print_comment(__LINE__, __FILE__, "attr_offset:                      %d", attr_offset);
    print_comment(__LINE__, __FILE__, "-----------------------------------------");
}

const char *coDoHeader::getName() const
{
    coCharShmArray *tmpcarr;
    char *tmpcptr;

    if (name_type != SHMPTR)
    {
        print_error(__LINE__, __FILE__, "wrong type found in SDS");
        return 0;
    }
    tmpcarr = new coCharShmArray(name_shm_seq_no, name_offset);
    tmpcptr = (char *)tmpcarr->getDataPtr();
    delete tmpcarr;
    return tmpcptr;
}

// shared memory for the namestring must have been reserved before
// this call

void coDoHeader::set_name(int sn, shmSizeType o, char *n)
{
    coCharShmArray *tmpcarr;

    name_type = SHMPTR;
    name_shm_seq_no = sn;
    name_offset = o;
    tmpcarr = new coCharShmArray(name_shm_seq_no, name_offset);
    tmpcarr->setString(n);
    delete tmpcarr;
}

coStringShmArray *coDoHeader::getAttributes()
{
    coStringShmArray *tmpsarr;

    if (attr_shm_seq_no)
    {
        if (attr_type == SHMPTR)
        {
            tmpsarr = new coStringShmArray(attr_shm_seq_no, attr_offset);
            return tmpsarr;
        }
        else
        {
            print_error(__LINE__, __FILE__, "wrong attribute type");
            return NULL;
        }
    }
    else
        return NULL;
}

/////////
void coDistributedObject::copyAllAttributes(const coDistributedObject *src)
{
    if (src && src != this)
    {
        const char **name, **setting;
        int n = src->getAllAttributes(&name, &setting);
        if (n > 0)
            addAttributes(n, name, setting);
    }
}

//// Common function for all read-Constructors
// copies getShmArray() for performance reasons...
void coDistributedObject::getObjectFromShm()
{
    if (!name)
    {
        new_ok = 0;
        return;
    }
    int len;
    char *tmpptr;

    len = (int)strlen(name) + 1;
    tmpptr = new char[len];
    strcpy(tmpptr, name);
    Message msg(COVISE_MESSAGE_GET_OBJECT, DataHandle(tmpptr, len));
    ApplicationProcess::approc->exch_data_msg(&msg, 2, COVISE_MESSAGE_OBJECT_FOUND, COVISE_MESSAGE_OBJECT_NOT_FOUND);
    // this is a local message, so no conversion is necessary
    if (msg.type == COVISE_MESSAGE_OBJECT_FOUND)
    {
        const char *cPtr = msg.data.data();
        shmarr = new coShmArray(*(int *)cPtr, *(shmSizeType *)(cPtr+sizeof(int)));
        header = (coDoHeader *)shmarr->getPtr();
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "shmarr: %d %d", iPtr[0], iPtr[1]);
#endif
    }
    else
    {
        new_ok = 0;
        return;
    }

    /// Virtual function call: calls class-specific routine
    if (rebuildFromShm() == 0)
    {
        print_comment(__LINE__, __FILE__, "rebuildFromShm == 0");
    }
}

/// Get my location in shared memory
void coDistributedObject::getShmLocation(int &shmSegNo, shmSizeType &offset) const
{
    shmSegNo = shmarr->get_shm_seq_no();
    offset = shmarr->get_offset();
}

bool coDistributedObject::checkObject() const
{
    //cerr << "coDistributedObject::checkObject(" << name << ")" << endl;
    bool printed = false;
    return checkObj(shmarr->get_shm_seq_no(), shmarr->get_offset(), printed);
}

// check a shm object
bool coDistributedObject::checkObj(int shmSeg, shmSizeType shmOffs, bool &printed) const
{
    coShmPtr sPtr(shmSeg, shmOffs);
    int *iPtr = (int *)sPtr.getPtr();
    //cerr << "checkObj ptr to: " << iPtr[0] << endl;

    int type = iPtr[0];

    //////////////////////////////////////////////////////////////////////////////
    /// Distributed Object types

    if (type > 255) // only DOs reach up that far
    {
        //char *tName = calcTypeString(iPtr[0]);
        //cerr << "              " << tName << endl;
        //delete [] tName;

        // minimal check
        if (iPtr[2] != COVISE_OBJECTID)
        {
            //cerr << "Found no header for Obj: \"" << name << "\"" << endl;
            return false;
        }

        /// Check DO Header
        coDoHeader *hdr = (coDoHeader *)iPtr;
        int numObj = hdr->get_number_of_elements();
        if (numObj <= 0)
        {
            //cerr << "Problems found checking header of Obj: \"" << name << "\"" << endl;
            return false;
        }

        int i;
        // check name and attributes
        iPtr += coDoHeader::getIntHeaderSize() - 6;
        for (i = 0; i < numObj + 2; i++)
        {

            switch (iPtr[0])
            {
            case SHMPTR:
            {
                //cerr << "   found PTR type: " << iPtr[0] << endl;
                if (iPtr[1] && !checkObj(iPtr[1], *(shmSizeType *)(&iPtr[2]), printed))
                {
                    if (!printed) // do not print for parents of failed children
                    {
                        printed = true;
                        const char *currObjName = hdr->getName();
                        cerr << "\n";
                        cerr << "  !!!   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
                        cerr << "  !!!   +++ Object Integrity check failed\n";
                        cerr << "  !!!   +++\n";
                        if (currObjName)
                            cerr << "  !!!   +++ Failed object: " << currObjName << "\n";

                        char *tName = calcTypeString(type);
                        cerr << "  !!!   +++ Object Type:   " << tName << endl;
                        if (i == 0)
                            cerr << "  !!!   +++ Failed field:  Object Name\n";
                        else if (i == 1)
                            cerr << "  !!!   +++ Failed field:  Attributes\n";
                        else
                        {
                            // try to create a 'dummy' object to get class structure
                            coDistributedObject *obj = NULL;
                            VirtualConstructor *tmpptr;
                            vconstr_list->reset();
                            while ((tmpptr = vconstr_list->next()))
                            {
                                if (tmpptr->type == type)
                                {
                                    // trick out the != NULL test but fail at restore_shared_dl
                                    obj = tmpptr->vconstr((coShmArray *)-1);
                                }
                            }

                            // find the element - does not work for coDoSet
                            if (0 != strcmp(tName, "SETELE"))
                            {
                                coDoInfo *info = new coDoInfo[numObj];
                                if (obj && numObj == obj->getObjInfo(numObj, &info))
                                    cerr << "  !!!   +++ Failed field:  " << info[i - 2].description << "\n";
                            }
                        }
                        cerr << "  !!!   +++\n";
                        cerr << "  !!!   +++ Distributed memory is corrupted: Save map,\n";
                        cerr << "  !!!   +++ do not try to re-execute in this session.\n";

                        cerr << "  !!!   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" << endl;

                        delete[] tName;
                        return false;
                    }
                }
                iPtr += 2 + sizeof(shmSizeType)/sizeof(int);
                break;
            }
            case COVISE_NULLPTR:
            {
                iPtr += 2 + sizeof(shmSizeType)/sizeof(int);
                break;
            }
            default:
            {
                iPtr += 2;
                break;
            }
            }
        }

        return true;
    }

    //////////////////////////////////////////////////////////////////////////////
    ////// ARRAY types

    if (type > 12 && type < 20)
    {
        int msize = iPtr[1];

        // Arrays are built :  | type | length | data | ... | data | type /
        // <type> at end is always int-aligned

        //cerr << "   found ARRAY type: " << type
        //     << " size= " << msize << endl;

        shmSizeType size = sizeof(int); // all shm vars start with type

        // calculate space (from ../dmgr_lib/dmgr_process.cpp)
        switch (type)
        {
        case FLOATSHM:
            size += sizeof(float);
            break;
        case DOUBLESHM:
            size += sizeof(double);
            break;
        case CHARSHM:
            size += sizeof(char);
            break;
        case SHORTSHM:
            size += sizeof(short);
            break;
        case LONGSHM:
            size += sizeof(long);
            break;
        case INTSHM:
            size += sizeof(int);
            break;
        case FLOATSHMARRAY:
            size += msize * sizeof(float) + 2 * sizeof(int);
            break;
        case DOUBLESHMARRAY:
            size += msize * sizeof(double) + 2 * sizeof(int);
            break;

        // All these add 2 ints: one for length and one for safety value.
        // the additional third int is for alignment whwnwver this may be doubtful
        case STRINGSHMARRAY:
            size += msize * 2 * sizeof(int) + 2 * sizeof(int);
            break;
        case CHARSHMARRAY:
            size += msize * sizeof(char) + 3 * sizeof(int);
            break;
        case SHORTSHMARRAY:
            size += msize * sizeof(short) + 3 * sizeof(int);
            break;
        case LONGSHMARRAY:
            size += msize * sizeof(long) + 3 * sizeof(int);
            break;
        case INTSHMARRAY:
            size += msize * sizeof(int) + 2 * sizeof(int);
            break;
        case SHMPTRARRAY:
            size += msize * 2 * sizeof(int) + 2 * sizeof(int);
            break;
        default:
            print_comment(__LINE__, __FILE__, "unknown type %d for shm_alloc\ncannot provide memory\n", type);
            break;
        };

        // only use aligned memory sizes
        int alignRest = size % SIZEOF_ALIGNMENT;
        if (alignRest)
            size += (SIZEOF_ALIGNMENT - alignRest);

        // size in 'ints'
        int intSize = size / sizeof(int);

        // the protection-word test for our array
        if (iPtr[intSize - 1] != type)
        {
            cerr << "error in type check: type=" << type << ", should be: " << iPtr[intSize - 1] << endl;
            return false;
        }
        //else
        //cerr << "type check passed" << endl;

        // recurse over SHM-Arrays
        if (type == SHMPTRARRAY)
        {
            for (int i = 0; i < msize; i++)
            {
                iPtr += 1 + sizeof(shmSizeType)/sizeof(int);
                if (iPtr[0] && !checkObj(iPtr[0], *(shmSizeType *)(&iPtr[1]), printed))
                    return false;
            }
        }
    }

    return true;
}

void coDistributedObject::copyObjInfo(coObjInfo *info) const
{
    info->blockNo = info->numBlocks = -1;
    info->timeStep = info->numTimeSteps = -1;
    info->time = 0.;
}

coDistributedObject *coDistributedObject::clone(const coObjInfo &info) const
{
    coDistributedObject *obj = cloneObject(info);
    obj->copyAllAttributes(this);
    return obj;
}
