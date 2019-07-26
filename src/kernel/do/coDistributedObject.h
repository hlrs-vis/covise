/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DISTRIBUTED_OBJECT_H
#define CO_DISTRIBUTED_OBJECT_H

#include <util/covise_list.h>
#include <util/coObjID.h>
#include "coShmPtrArray.h"

/*
 $Log:  $
 * Revision 1.1  1993/09/25  20:42:21  zrhk0125
 * Initial revision
 *
*/

/***********************************************************************\
 **                                                                     **
 **   Distributed Object class                     Version: 1.1         **
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
 **                  26.05.93  Ver 1.1  PackElement introduced for      **
 **                                     sending whole Objects between   **
 **                                     Data Managers                   **
 **                                                                     **
\***********************************************************************/
namespace covise
{
class DataHandle;
class ApplicationProcess;

DOEXPORT void PackElement_print(class PackElement *);

class DOEXPORT PackElement
{
public:
    int shm_seq_no;
    shmSizeType offset; // offset to the type header
    int type;
    shmSizeType size; // size in bytes
    int length; // number of elements, if array or any-type
    char *ptr; // pointer to the data (not the type-header
    PackElement()
    {
        length = shm_seq_no = offset = type = size = 0;
        ptr = NULL;
    }
    void print()
    {
        PackElement_print(this);
    };
};

class DOEXPORT VirtualConstructor
{
public:
    int type;
    coDistributedObject *(*vconstr)(coShmArray *arr);
    VirtualConstructor(int t_no, coDistributedObject *(*vc)(coShmArray *arr))
    {
        type = t_no;
        vconstr = vc;
    };
    void print(){};
};

// structure for the _dl functions
struct covise_data_list
{
    data_type type;
    const void *ptr;
};

const int NUMBER_OF_INT_HEADER_ELEMENTS = 13;
const int NUMBER_OF_SHM_SIZE_HEADER_ELEMENTS = 4;

class DOEXPORT coDoHeader
{
    // REMEMBER: upon insertion of new elements adjust getHeaderSize
    // and getIntHeaderSize, and modify:
    //     DataManagerProcess::create_object_from_msg
    //     coDistributedObject::getObjectInfo(coDoInfo **info_list)
    // AND correct coDistributedObject::checkObj

private:
    int object_type;
    shmSizeType number_of_bytes;
    int objectid_type;
    int objectid_h;
    int objectid_t;
    int number_of_elements_type; // INTSHM
    int number_of_elements;
    int version_type; // INTSHM
    int version;
    int refcount_type; // INTSHM
    int refcount;
    int name_type; // CHARSH
    int name_shm_seq_no;
    shmSizeType name_offset;
    int attr_type; // SHMPTR or NULLPTR
    int attr_shm_seq_no;
    shmSizeType attr_offset;
    // the following only for partitioned Objects!!
    int part_object_type_type; // INTSHM
    int part_object_type; // Object type
    int part_max_number_of_parts_type; // INTSHM
    int part_max_number_of_parts; // Max number of parts
    int part_curr_number_of_parts_type; // INTSHM
    int part_curr_number_of_parts; // Current number of parts
    int part_address_list_type; // SHMPTR or NULLPTR
    int part_address_list_shm_seq_no;
    shmSizeType part_address_list_offset;

public:
    static int getHeaderSize()
    {
        return getIntHeaderSize() * sizeof(int);
    }; // size must be adjusted manually!!

    static int getIntHeaderSize()
    {
        return NUMBER_OF_INT_HEADER_ELEMENTS + (NUMBER_OF_SHM_SIZE_HEADER_ELEMENTS * sizeof(shmSizeType)/sizeof(int));
    }; // size must be adjusted manually!!

    int getObjectType()
    {
        return object_type;
    };

    int get_number_of_elements()
    {
        if (number_of_elements_type == INTSHM)
            return number_of_elements;
        else
        {
            print_error(__LINE__, __FILE__,
                        "numberOFElements not found");
            return 0;
        }
    }

    int get_version()
    {
        if (version_type == INTSHM)
            return version;
        else
        {
            print_error(__LINE__, __FILE__,
                        "version number not found");
            return 0;
        }
    }

    int increase_version()
    {
        return ++version;
    }

    int get_refcount()
    {
        if (refcount_type == INTSHM)
            return refcount;
        else
        {
            print_error(__LINE__, __FILE__,
                        "reference counter not found");
            return 0;
        }
    };

    int incRefCount()
    {
        return ++refcount;
    };
    int decRefCount()
    {
        return --refcount;
    };
    const char *getName() const; // do not delete the resulting pointer
    coStringShmArray *getAttributes();
    int get_attr_type()
    {
        return attr_type;
    };
    int getObjectType_offset()
    {
        return 0;
    };
    int get_number_of_elements_offset()
    {
        return (int)((&number_of_elements_type - &object_type) * sizeof(int));
    };
    int get_version_offset()
    {
        return (int)((&version_type - &object_type) * sizeof(int));
    };
    int get_refcount_offset()
    {
        return (int)((&refcount_type - &object_type) * sizeof(int));
    };
    void get_objectid(int *h, int *t)
    {
        *h = objectid_h;
        *t = objectid_t;
    };
    void set_object_type(int ot)
    {
        object_type = ot;
    };
    void set_objectid(int h, int t)
    {
        objectid_type = COVISE_OBJECTID;
        objectid_h = h;
        objectid_t = t;
    };
    void set_number_of_elements(int noe)
    {
        number_of_elements_type = INTSHM;
        number_of_elements = noe;
    };
    void set_version(int v)
    {
        version_type = INTSHM;
        version = v;
    };
    void set_refcount(int rc)
    {
        refcount_type = INTSHM;
        refcount = rc;
    };
    void set_name(int sn, shmSizeType o, char *n); // namelength must fit!!
    void addAttributes(int sn, shmSizeType o)
    {
        if (sn == 0)
            attr_type = COVISE_NULLPTR;
        else
            attr_type = SHMPTR;
        attr_shm_seq_no = sn;
        attr_offset = o;
    };
    void print();
};

class DOEXPORT coDoInfo
{
public:
    int type;
    const char *type_name;
    const char *description;
    char *obj_name; //optional, only in the case of DISTROBJ
    void *ptr;
    coDoInfo()
        : type(0)
        , type_name(NULL)
        , description(NULL)
        , obj_name(NULL)
        , ptr(NULL){};
    ~coDoInfo()
    {
        delete obj_name;
    }
    void print()
    {
        cout << "Type: " << type << ",  " << type_name << std::endl;
        if (description)
            cout << "Description: " << description << std::endl;
        if (obj_name)
            cout << "Object Name: " << obj_name << std::endl;
        cout << "Pointer: " << ptr << std::endl;
    }
};

class DOEXPORT coDistributedObject
{
    friend class ApplicationProcess;
    friend class coShmArrayPtr;
    //friend class DO_PartitionedObject;
    friend void coShmPtrArray::set(int i, const class coDistributedObject *elem);
    static List<VirtualConstructor> *vconstr_list;

    static int xfer_arrays;

protected:
    coIntShm version;
    mutable coIntShm refcount;
    mutable coShmArray *shmarr;
    coStringShmArray *attributes = nullptr;
    mutable coDoHeader *header = nullptr;
    char type_name[7];
    int type_no = 0;
    char *name = nullptr;
    int loc_version = -1;
    bool new_ok;
    int size = 0;
    mutable char *attribs = nullptr; // Data space for Attributes
    int getShmArray() const;
    int createFromShm(coShmArray *arr)
    {
        shmarr = arr;
        return rebuildFromShm();
    };
    virtual int rebuildFromShm() = 0;
    virtual int getObjInfo(int, coDoInfo **) const
    {
        print_comment(__LINE__, __FILE__,
                      "getObjInfo called from coDistributedObject");
        return 0;
    };

    /// Check object in shared memory
    bool checkObj(int shmSegNo, shmSizeType shmOffs, bool &printed) const;
    virtual coDistributedObject *cloneObject(const coObjInfo &newinfo) const = 0;

public:
    /// Get my location in shared memory
    void getShmLocation(int &shmSegNo, shmSizeType &offset) const;

    /// Attach an attribute to an object
    void addAttribute(const char *, const char *);

    /// Attach multiple attributes to an object
    void addAttributes(int, const char *const *, const char *const *);

    /// get one attribute
    const char *getAttribute(const char *) const;

    /// get number of attributes
    int getNumAttributes() const;

    /// get all attributes
    int getAllAttributes(const char ***name, const char ***content) const;

    /// copy all attributes from src to this object
    void copyAllAttributes(const coDistributedObject *src);

    /// get the object's name
    char *getName() const
    {
        return name;
    }

    /// get the object's type
    const char *getType() const
    {
        return type_name;
    }

    /// check whether this is a certain type
    int isType(const char *reqType) const
    {
        return (0 == strcmp(reqType, type_name));
    }

    /// check whether object was created or received ok
    bool objectOk() const
    {
        return new_ok;
    }

    coDistributedObject()
    {
        size = 0;
        loc_version = 0;
        new_ok = true;
        xfer_arrays = 1;
        attributes = NULL;
        attribs = NULL;
        shmarr = NULL;
        type_name[0] = '\0';
        type_no = 0;
        header = NULL;
        name = NULL;
        //	printf("---- new coDistributedObject: %x\n", this);
    };

    coDistributedObject(const coObjInfo &info)
    {
        size = 0;
        loc_version = 0;
        new_ok = true;
        xfer_arrays = 1;
        attributes = NULL;
        attribs = NULL;
        shmarr = NULL;
        header = NULL;
        type_name[0] = '\0';
        type_no = 0;
        const char *n = info.getName();
        if (n)
        {
            name = new char[strlen(n) + 1];
            strcpy(name, n);
        }
        else
            name = NULL;
    };

    coDistributedObject(const coObjInfo &info, const char *t)
    {
        //	printf("---- new coDistributedObject: %x\n", this);
        size = 0;
        loc_version = 0;
        new_ok = true;
        xfer_arrays = 1;
        attributes = NULL;
        attribs = NULL;
        shmarr = NULL;
        header = NULL;
        strncpy(type_name, t, 7);
        type_name[6] = '\0';
        type_no = calcType(type_name);
        const char *n = info.getName();
        if (n)
        {
            name = new char[strlen(n) + 1];
            strcpy(name, n);
        }
        else
            name = NULL;
    };

    coDistributedObject(const coObjInfo &info, int shmSeg, shmSizeType offs, char *t)
    {
        size = 0;
        loc_version = 0;
        new_ok = true;
        xfer_arrays = 1;
        attributes = NULL;
        attribs = NULL;
        shmarr = new coShmArray(shmSeg, offs);
        header = (coDoHeader *)shmarr->getPtr();
        strncpy(type_name, t, 7);
        type_name[6] = '\0';
        type_no = calcType(type_name);
        attribs = NULL;
        const char *n = info.getName();
        if (n)
        {
            name = new char[strlen(n) + 1];
            strcpy(name, n);
        }
        else
            name = NULL;

        if (createFromShm(shmarr) == 0)
        {
            print_comment(__LINE__, __FILE__, "createFromShm == 0");
            new_ok = false;
        }
    };

    virtual ~coDistributedObject();
    coDistributedObject *clone(const coObjInfo &newinfo) const;
    /// retrieve a data object from shm/dmgr by name/coObjInfo
    /// (replaces (new coDistributedObject(newinfo)->createUnknown())
    static const coDistributedObject *createFromShm(const coObjInfo &newinfo);
    static const coDistributedObject *createUnknown(coShmArray *);
    static const coDistributedObject *createUnknown(int seg, shmSizeType offs);
    void copyObjInfo(coObjInfo *info) const;

    const coDistributedObject *createUnknown() const;

    int *store_header(int, int, int, int *, data_type *, long *, covise::DataHandle &idata);
    int restore_header(int **, int, int *, int *, shmSizeType *);
    void init_header(int *, int *, int, data_type **, long **);

    int update_shared_dl(int count, covise_data_list *dl);
    int store_shared_dl(int count, covise_data_list *dl);
    int restore_shared_dl(int count, covise_data_list *dl);

    void setType(const char *, const char *);
    static int calcType(const char *);
    static char *calcTypeString(int);
    int getObjectInfo(coDoInfo **) const;
    int get_type_no() const
    {
        return type_no;
    }
    int access(access_type);
    //  access_type set_access(access_type);
    //  access_type set_access_block(access_type);
    //  access_type get_access() { return current_access; };
    int destroy();
    char *object_on_hosts() const;
    //    int incRefCount() { return header->incRefCount(); };
    int incRefCount() const
    {
        return refcount = refcount + 1;
    }
    //    int decRefCount() { return header->decRefCount(); }
    int decRefCount() const
    {
        return refcount = refcount - 1;
    }
    int getRefCount() const
    {
        return refcount;
    }
    void print() const
    {
    }

    static int set_vconstr(const char *, coDistributedObject *(*)(coShmArray *));

    /// Common function for all read-Constructors:
    //           contact CRB, get Obj, call restore_shared.
    void getObjectFromShm();

    /// Check object: return true if valid, false if not
    bool checkObject() const;
};
}
#endif
