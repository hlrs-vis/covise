/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_SHM_H
#define EC_SHM_H

#include <covise/covise.h>

#if !defined(_WIN32)
#define SHARED_MEMORY
#endif

#ifdef _CRAYT3E
#define HANDLE unsigned int
#endif

#include <util/coTypes.h>
#include <util/coLog.h>

#include <util/covise_list.h>
//#include <net/covise_msg.h>
#include <net/message.h>
#include <covise/covise_global.h>
#ifdef _WIN32
#include <windows.h>
#include <windowsx.h>
#endif

/***********************************************************************\ 
 **                                                                     **
 **   Shared Memory Classes                       Version: 2.0          **
 **                                                                     **
 **                                                                     **
 **   Description  : All classes that deal with the creation and        **
 **                  administration of shared memory.                   **
 **                  The SharedMemory class handles the operating system **
 **		    part of the shared memory management.              **
 **		    is a utility class to organize the used            **
 **		    and unused parts of the shared memory.             **
 **		    ShmAccess allows only the access to the shared     **
 **		    memory areas, not the allocation or return of      **
 **		    allocated regions.                                 **
 **		    coShmAlloc does all the administration of the shared **
 **		    memory regions, using trees with nodes which point **
 **		    to used and free parts. Here all allocation of     **
 **		    regions in the shared memory takes place.          **
 **                  coShmPtr and its subclassses provide an easy         **
 **                  access to and initialization of pointers in        **
 **                  the shared memory areas.                           **
 **                                                                     **
 **   Classes      : SharedMemory, ShmAccess, coShmAlloc,                 **
 **                  coShmPtr, coCharcoShmPtr, coShortcoShmPtr, coIntcoShmPtr,        **
 **                  LongcoShmPtr, coFloatcoShmPtr, DoublecoShmPtr              **
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
 **                  26.05.93  Ver 2.0 design reworked, basic data      **
 **                                    types clarified, compound data   **
 **                                    types added.                     **
 **                                                                     **
\***********************************************************************/
namespace covise
{
class SharedMemory;
class DataManagerProcess;

const int SIZEOF_ALIGNMENT = 8;

enum access_type
{
    ACC_DENIED = 0x0,
    ACC_NONE = 0x1,
    ACC_READ_ONLY = 0x2,
    ACC_WRITE_ONLY = 0x4,
    ACC_READ_AND_WRITE = 0x8,
    ACC_READ_WRITE_DESTROY = 0x10,
    ACC_REMOTE_DATA_MANAGER = 0x20
};

// IDs for the data type encoding (for IPC)
#ifndef YAC
const int NONE = 0;
//const int CHAR            =  1;
//const int SHORT           =  2;
//const int INT             =  3;
//const int LONG            =  4;
//const int FLOAT           =  5;
//const int DOUBLE          =  6;
//const int CHARSHMPTR      =  7;
//const int SHORTSHMPTR     =  8;
//const int INTSHMPTR       =  9;
//const int LONGSHMPTR      = 10;
//const int FLOATSHMPTR     = 11;
//const int DOUBLESHMPTR    = 12;
const int CHARSHMARRAY = 13;
const int SHORTSHMARRAY = 14;
const int INTSHMARRAY = 15;
#ifdef __GNUC__
#undef LONGSHMARRAY
#endif
const int LONGSHMARRAY = 16;
const int FLOATSHMARRAY = 17;
const int DOUBLESHMARRAY = 18;
const int EMPTYCHARSHMARRAY = 13 | 0x80;
const int EMPTYSHORTSHMARRAY = 14 | 0x80;
const int EMPTYINTSHMARRAY = 15 | 0x80;
const int EMPTYLONGSHMARRAY = 16 | 0x80;
const int EMPTYFLOATSHMARRAY = 17 | 0x80;
const int EMPTYDOUBLESHMARRAY = 18 | 0x80;
const int SHMPTRARRAY = 19;
const int CHARSHM = 20;
const int SHORTSHM = 21;
const int INTSHM = 22;
#ifdef __GNUC__
#undef LONGSHM
#endif
const int LONGSHM = 23;
const int FLOATSHM = 24;
const int DOUBLESHM = 25;
const int USERDEFINED = 26;
const int SHMPTR = 27;
const int COVISE_OBJECTID = 28;
const int DISTROBJ = 29;
const int STRINGSHMARRAY = 30;
const int STRING = 31; // CHARPTR == STRING
const int UNKNOWN = 37;
const int COVISE_NULLPTR = 38;
const int COVISE_OPTIONAL = 39;
const int I_SLIDER = 40;
const int F_SLIDER = 41;
const int PER_FACE = 42;
const int PER_VERTEX = 43;
const int OVERALL = 44;
const int FLOAT_SLIDER = 45;
const int FLOAT_VECTOR = 46;
const int COVISE_BOOLEAN = 47;
const int BROWSER = 48;
const int CHOICE = 49;
const int FLOAT_SCALAR = 50;
const int COMMAND = 51;
const int MMPANEL = 52;
const int TEXT = 53;
const int TIMER = 54;
const int PASSWD = 55;
const int CLI = 56;
const int ARRAYSET = 57;
// do not exceed 127 (see EMPTY... =   | 0x80)
const int COLORMAP_MSG = 58;
const int INT_SLIDER = 59;
const int INT_SCALAR = 60;
const int INT_VECTOR = 61;
const int COLOR_MSG = 62;
const int COLORMAPCHOICE_MSG = 63;
const int MATERIAL_MSG = 64;

#endif

const int START_EVEN = 0;
const int START_ODD = 4;

const int OBJ_OVERWRITE = 0;
const int OBJ_NO_OVERWRITE = 1;

const int SET_CREATE = 0;

class SHMEXPORT MMapEntry
{
public:
    MMapEntry()
    {
    }
    ~MMapEntry()
    {
    }
    char *ptr;
    int fd;
    long size;
    void print()
    {
    }
};

class SHMEXPORT Malloc_tmp
{
public:
    static List<MMapEntry> *mmaplist;
    static void *large_new(long size);
    static void large_delete(void *);
};
typedef unsigned int shmSizeType;
typedef unsigned int ArrayLengthType;
typedef void(shmCallback)(int shmKey, shmSizeType size, char *address);

extern SHMEXPORT SharedMemory *get_shared_memory();

class SHMEXPORT SharedMemory
{
    friend SHMEXPORT SharedMemory *get_shared_memory();
    friend class DataManagerProcess;
    static class SharedMemory **shm_array;
    static List<SharedMemory> *shmlist;
    static int global_seq_no;
    class SharedMemory *next;
#if defined SHARED_MEMORY
    int shmfd; // POSIX shared memory
    int shmid; // SysV shared memory
#elif defined(_WIN32)
    HANDLE handle;
    HANDLE filemap;
#endif
    shmSizeType size;
    char *data;
    enum state
    {
        attached,
        detached,
        invalid,
        valid
    };
    state shmstate;
    int key;
    int seq_no;
    int noDelete;

public:
    SharedMemory(){};
    SharedMemory(int shm_key, shmSizeType shm_size, int noDelete = 0);
    SharedMemory(int *shm_key, shmSizeType shm_size);
    ~SharedMemory();
    static shmCallback *shmC;
#if defined(__hpux) || defined(_SX)
    void *get_pointer(int no);
#else
    void *get_pointer(int no)
    {
        if (SharedMemory::shmlist)
        {
            return &(shm_array[no - 1]->data[2 * sizeof(int)]);
        }
        print_comment(__LINE__, __FILE__, "getting pointer: 0x0");
        return NULL;
    };
#endif
    void *get_pointer()
    {
        return &(data[2 * sizeof(int)]);
    };
    SharedMemory *get_next_shm()
    {
        return next;
    }
    int get_seq_no()
    {
        return seq_no;
    };
    int is_attached()
    {
        return (shmstate == attached);
    };
    int detach();
    int get_key()
    {
        return key;
    };
    shmSizeType get_size()
    {
        return size;
    };
    void get_shmlist(int *);
    void print(){};
    static int num_attached()
    {
        return global_seq_no;
    }
};
// minimal allocation size for SHM segments - 64 MB

class SHMEXPORT ShmConfig
{
private:
    ShmConfig();
    ~ShmConfig();
    size_t minSegSize;
    static ShmConfig *theShmConfig;

public:
    static ShmConfig *the();
    static covise::shmSizeType getMallocSize();
};

const int MAX_NO_SHM = 1000;

const int COMPARE_ADDRESS = 1;
const int COMPARE_SIZE = 2;

class coShmPtr;
class coShmAlloc;

class SHMEXPORT ShmAccess
{
protected:
    static SharedMemory *shm;

public:
    ShmAccess(int k);
    ShmAccess(int *k);
    ShmAccess(char *, int noDelete = 1);
    ~ShmAccess();
    void add_new_segment(int k, shmSizeType size);
    void *get_pointer(int no)
    {
        return shm->get_pointer(no);
    };
    void *get_pointer()
    {
        return shm->get_pointer();
    };
    int get_key()
    {
        return shm->get_key();
    };
};

class PackElement;
class coShmPtrArray;
class coDistributedObject;

class SHMEXPORT coShmItem
{
    friend int covise_decode_list(List<PackElement> *, char *, DataManagerProcess *, char);
    friend coShmPtr *covise_extract_list(List<PackElement> *pack_list, char);
    friend class coShmPtrArray;
    friend class coDistributedObject; //__alpha
protected:
    int shm_seq_no;
    shmSizeType offset;
    int type;
    void *ptr;

public:
    coShmItem()
    {
        shm_seq_no = 0;
        offset = 0;
        type = NONE;
        ptr = NULL;
    };
    int get_shm_seq_no()
    {
        return shm_seq_no;
    };
    shmSizeType get_offset()
    {
        return offset;
    };
    int get_type()
    {
        return type;
    };
    void *getPtr()
    {
        return ptr;
    };
    void print();
};

class SHMEXPORT coShmPtr : public coShmItem
{
protected:
    friend class ShmMessage;
    friend class DmgrMessage;
    friend class coShmAlloc;
    friend class coDistributedObject;
    friend class DataManagerProcess;
    static SharedMemory *shmptr;
    inline void recalc()
    {
        if (shmptr == NULL)
            shmptr = get_shared_memory();
        ptr = (void *)((char *)shmptr->get_pointer(shm_seq_no) + offset);
        type = *(int *)ptr; // the type is an integer at the beginning of the memory area
    };

public:
    coShmPtr()
        : coShmItem(){};
    coShmPtr(int no, shmSizeType o)
    {
        shm_seq_no = no;
        offset = o;
        if (!(shm_seq_no == 0 && offset == 0))
            recalc();
    };
    coShmPtr(Message *msg)
    {
        shm_seq_no = *(int *)msg->data.data();
        offset = *(int *)(&msg->data.data()[sizeof(int)]);
        recalc();
    };
    void setPtr(int no, shmSizeType o)
    {
        shm_seq_no = no;
        offset = o;
        recalc();
    };
    void *getDataPtr() const
    {
        return (void *)((char *)ptr + sizeof(int)); // extra int is the type
    };
    // first integer holds type
};

template <typename DataType, int typenum>
class coDataShm : public coShmPtr
{
public:
    coDataShm()
    {
    }
    coDataShm(int no, shmSizeType o)
        : coShmPtr(no, o)
    {
        if (type != typenum)
        {
            cerr << "wrong type from shared memory in coDataShm constructor: was" << type << ", expected " << typenum << std::endl;
            print_exit(__LINE__, __FILE__, 1);
        }
    };
    DataType get() const
    {
        return *((DataType *)(((char *)ptr) + sizeof(int)));// int is the type
    }
    operator DataType() const
    {
        return *((DataType *)(((char *)ptr) + sizeof(int)));// extra int is the type
    }
    DataType set(DataType val)
    {
        return (*((DataType *)(((char *)ptr) + sizeof(int))) = val);// extra int is the type
    }
    DataType &operator=(const DataType &c)
    {
        return *((DataType *)(((char *)ptr) + sizeof(int))) = c;// extra int is the type
    }
    void setPtr(int no, shmSizeType o)
    {
        coShmPtr::setPtr(no, o);
        if (type != typenum)
        {
            cerr << "wrong type associated in coDataShm->setPtr: was" << type << ", expected " << typenum << std::endl;
            print_exit(__LINE__, __FILE__, 1);
        }
    };
};

INST_TEMPLATE2(template class SHMEXPORT coDataShm<char, CHARSHM>)
typedef coDataShm<char, CHARSHM> coCharShm;
INST_TEMPLATE2(template class SHMEXPORT coDataShm<short, SHORTSHM>)
typedef coDataShm<short, SHORTSHM> coShortShm;
INST_TEMPLATE2(template class SHMEXPORT coDataShm<int, INTSHM>)
typedef coDataShm<int, INTSHM> coIntShm;
INST_TEMPLATE2(template class SHMEXPORT coDataShm<long, LONGSHM>)
typedef coDataShm<long, LONGSHM> coLongShm;
INST_TEMPLATE2(template class SHMEXPORT coDataShm<float, FLOATSHM>)
typedef coDataShm<float, FLOATSHM> coFloatShm;
INST_TEMPLATE2(template class SHMEXPORT coDataShm<double, DOUBLESHM>)
typedef coDataShm<double, DOUBLESHM> coDoubleShm;

class SHMEXPORT coShmArray : public coShmItem
{
protected:
    friend class Message;
    friend class ShmMessage;
    friend class coShmAlloc;
    friend class coDistributedObject;
    ArrayLengthType length;
    static SharedMemory *shmptr;
    inline void recalc()
    {
        if (shmptr == NULL)
            shmptr = get_shared_memory();
        if (shm_seq_no != 0)
        {
            ptr = (void *)((char *)shmptr->get_pointer(shm_seq_no) + offset);
            type = *(int *)ptr;
            length = *(int *)((char *)ptr + sizeof(int)); // int type ArrayLengthType length
        }
        else
        {
            ptr = NULL;
            type = *(int *)ptr & 0x7F; // empty array here
            length = 0; // there is no array yet!!
        }
    };

public:
    coShmArray()
        : coShmItem()
    {
        length = 0;
    }

    coShmArray(int no, shmSizeType o)
    {
        shm_seq_no = no;
        offset = o;
        if (!(shm_seq_no == 0 && offset == 0))
            recalc();
    }

    coShmArray(Message *msg)
    {
        shm_seq_no = *(int *)msg->data.data();
        offset = *(int *)(&msg->data.data()[sizeof(int)]);
        recalc();
    }
    void set_length(ArrayLengthType l)
    {
        length = l;
    }
    ArrayLengthType get_length() const
    {
        return length;
    }
    void setPtr(int no, shmSizeType o)
    {
        shm_seq_no = no;
        offset = o;
        recalc();
    }
    void *getDataPtr() const
    {
        if (ptr)
            return (void *)((char *)ptr + sizeof(int) + sizeof(ArrayLengthType));
        // first integer holds type, second holds length
        else
        {
            // inform datamanager that array is needed now!!
            return NULL; // for now!!
        }
    }
};

template <typename DataType, int typenum>
class coDataShmArray : public coShmArray
{
public:
    coDataShmArray()
        : coShmArray()
    {
    }
    coDataShmArray(int no, shmSizeType o)
        : coShmArray(no, o)
    {
        if (type != typenum)
        {
            cerr << "wrong type in coDataShmArray constructor from shared memory: expected " << typenum << ", was " << type << std::endl;
            print_exit(__LINE__, __FILE__, 1);
        }
    }
    void setPtr(int no, shmSizeType o)
    {
        coShmArray::setPtr(no, o);
        if (type != typenum)
        {
            cerr << "wrong type in coDataShmArray->setPtr constructor from shared memory: expected " << typenum << ", was " << type << std::endl;
            print_exit(__LINE__, __FILE__, 1);
        }
        if (length < 0)
        {
            cerr << "error in array length (< 0)\n";
        }
    };
    DataType &operator[](size_t i)
    {
        if (i >= 0 && i < length)
            return ((DataType *)(((char *)ptr) + sizeof(int)+sizeof(ArrayLengthType)))[i];
        // else
        cerr << "Access error for coDataShmArray\n"
             << i << " not in 0.." << length - 1 << std::endl;
        assert(i >= 0 && i < length);
        return null_return;
    }

    const DataType &operator[](size_t i) const
    {
        if (i >= 0 && i < length)
            return ((DataType *)(((char *)ptr) + 2 * sizeof(int)))[i];
        // else
        cerr << "Access error for coDataShmArray\n"
             << i << " not in 0.." << length - 1 << std::endl;
        assert(i >= 0 && i < length);
        return null_return;
    }

private:
    static DataType null_return;
};

template <typename DataType, int typenum>
DataType coDataShmArray<DataType, typenum>::null_return = DataType();

class SHMEXPORT coCharShmArray : public coDataShmArray<char, CHARSHMARRAY>
{
public:
    coCharShmArray()
    {
    }
    coCharShmArray(int no, shmSizeType o)
        : coDataShmArray<char, CHARSHMARRAY>(no, o)
    {
    }
    int setString(const char *c)
    {
        char *chptr = (char *)getDataPtr();
        if (chptr == NULL)
            return 0;

        ArrayLengthType i = 0;
        while (c[i] != '\0' && i < length)
        {
            chptr[i] = c[i];
            i++;
        }
        if (i < length)
            chptr[i] = '\0';
        return i;
    }
};

INST_TEMPLATE2(template class SHMEXPORT coDataShmArray<short, SHORTSHMARRAY>)
typedef coDataShmArray<short, SHORTSHMARRAY> coShortShmArray;
INST_TEMPLATE2(template class SHMEXPORT coDataShmArray<int, INTSHMARRAY>)
typedef coDataShmArray<int, INTSHMARRAY> coIntShmArray;
INST_TEMPLATE2(template class SHMEXPORT coDataShmArray<long, LONGSHMARRAY>)
typedef coDataShmArray<long, LONGSHMARRAY> coLongShmArray;
INST_TEMPLATE2(template class SHMEXPORT coDataShmArray<float, FLOATSHMARRAY>)
typedef coDataShmArray<float, FLOATSHMARRAY> coFloatShmArray;
INST_TEMPLATE2(template class SHMEXPORT coDataShmArray<double, DOUBLESHMARRAY>)
typedef coDataShmArray<double, DOUBLESHMARRAY> coDoubleShmArray;

class SHMEXPORT coStringShmArray : public coDataShmArray<char *, STRINGSHMARRAY>
{
public:
    coStringShmArray()
    {
    }
    coStringShmArray(int no, shmSizeType o)
        : coDataShmArray<char *, STRINGSHMARRAY>(no, o)
    {
    }
    char *operator[](unsigned int i);
    const char *operator[](unsigned int i) const;
    void stringPtrSet(int no, int sn, shmSizeType of)
    {
        char *cptr = (char *)ptr;
        int pos = 2*sizeof(int)/*(seq_nr+key)*/ + no*(sizeof(int)+sizeof(shmSizeType));
        *((int *)(cptr+pos)) = sn;
        *((int *)(cptr+pos+sizeof(int))) = of;
    }
    void stringPtrGet(int no, int *sn, shmSizeType *of)
    {
        char *cptr = (char *)ptr;
        int pos = 2*sizeof(int)/*(seq_nr+key)*/ + no*(sizeof(int)+sizeof(shmSizeType));
        *sn = *((int *)(cptr+pos));
        *of = *((int *)(cptr+pos+sizeof(int)));
    }
};
}
#endif
