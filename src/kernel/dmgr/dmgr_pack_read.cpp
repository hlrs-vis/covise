/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "dmgr_packer.h"
#include <do/coDistributedObject.h>

#undef DEBUG
/* the object header is organized in the following way:

 int object_type;
                 here the type calculated from the six character
                 type abbreviation is stored
 int number_of_bytes;
                 the length of the sturcture in the shared memory
 int objectid_type;
                 Identifier (OBJECTID) for the unique object id
 int objectid_h;
 int objectid_t;
the two parts of the object id
(host and time, just to be unique)
int number_of_elements_type;
Identifier (INTSHM) for the element count
int number_of_elements;
how many Data elements does this object
have (excludes header)
int version_type;
Identifier (INTSHM) for the version field
int version;
increases when object is modified (to allow easy
checking for consistency)
int refcount_type;
Identifier (INTSHM) for the reference counter
int refcount;
increases for each process attached to object
int name_type;
Identifier (SHMPTR) for the name of the object
int name_shm_seq_no;
int name_offset;
shm id and offset pointing to the name string
int attr_type;
Identifier (SHMPTR or NULLPTR) for attribute
int attr_shm_seq_no;
int attr_offset;
shm id and offset pointing to the attribute object
both zero if attr_type == NULLPTR, shm id and offset
pointing to the ShmString otherwise

In general single data elements (char, short, int, etc.) are written
into an 8 byte chunk (SIZEOF_ALIGNMENT) to make it easier to read them
on the Cray. Arrays of these elements are always packed. This wastes
some space on the SGIs, but since the most part of really large data
objects is in arrays anyway it doesn't matter.

Usually data is just copied from the shared memory to the write buffer.
In case of the Cray this can involve conversion of short, int, float etc.
Standard net format is IEEE!! If two Crays are communicating, no conversion
is performed, i.e. data is written as is. If Crays send to none-Crays
always IEEE is enforced. This leads to the ugly #ifdef constructs.
*/

#ifdef _WIN32
#undef CRAY
#endif

#define SET_CHUNK 20

using namespace covise;

Packer::Packer(Message *m, DataManagerProcess *dm)
{
    shm_ptr = nullptr;
    shm_obj_ptr = nullptr;
    convert = m->conn->convert_to;
    buffer = new PackBuffer(dm, m);
    number_of_data_elements = 0;
    datamgr = dm;
}

void PackBuffer::receive()
{
    conn->recv_msg(msg);
    if (msg->type == COVISE_MESSAGE_OBJECT_FOLLOWS)
    {
        buffer = msg->data;
        msg->data = DataHandle{};
        print_comment(__LINE__, __FILE__, "msg->data.length(): %d", buffer.length());
        //intbuffer_size = buffer.length() / sizeof(int) + ((buffer.length() % sizeof(int)) ? 1 : 0);
        intbuffer_ptr = 0;
    }
    else
        print_error(__LINE__, __FILE__, "wrong message received");
}

#ifndef CRAY
inline
#endif
    void
    PackBuffer::read_int(int &rd)
{
#ifdef DEBUG
    char tmp_str[100];
#endif
#ifdef BYTESWAP
    unsigned int *urd;
#endif
    if (intbuffer_ptr >= intbuffer_size()) // if end of buffer (all read)
        receive();
#ifdef CRAY
    if (convert) // at the moment cv == IEEE <=> !cv
#ifdef _CRAYT3E
        converter.exch_to_int((char *)&intbuffer()[intbuffer_ptr], &rd);
#else
        conv_single_int_i4c8(intbuffer()[intbuffer_ptr], &rd);
#endif
    else
        rd = intbuffer[intbuffer_ptr];
    intbuffer_ptr += 1;
    return;
#else
    rd = intbuffer()[intbuffer_ptr];
#ifdef BYTESWAP
    urd = (unsigned int *)&rd;
    swap_byte(*urd);
#endif
    intbuffer_ptr += 2;
#endif
#ifdef DEBUG
    sprintf(tmp_str, "PackBuffer::read_int %d", rd);
    print_comment(__LINE__, __FILE__, tmp_str);
#endif
}

inline void PackBuffer::put_back_int()
{
    // only allowed iummediately after read_int()

    if (intbuffer_ptr > 0)
#ifdef CRAY
        intbuffer_ptr--;
#else
        intbuffer_ptr -= 2;
#endif
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "PackBuffer::put_back_int");
#endif
}

char *PackBuffer::get_current_pointer_for_n_bytes(int &n)
{
    int tmp_intbuffer_ptr;
#ifdef DEBUG
    char tmp_str[100];
#endif

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "PackBuffer::get_current_pointer_for_n_bytes");
    sprintf(tmp_str, "intbuffer_ptr: %d  intbuffer_size: %d",
            intbuffer_ptr, intbuffer_size());
    print_comment(__LINE__, __FILE__, tmp_str);
#endif
    if (intbuffer_ptr >= intbuffer_size()) // pointer is at the end
    {
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "receiving new data");
#endif
        receive();
        intbuffer_ptr = 0;
    }
    if ((intbuffer_size() - intbuffer_ptr) * (int)sizeof(int) < n)
    {
#ifdef DEBUG
        sprintf(tmp_str, "only returning partial data: %d bytes of %d",
                (intbuffer_size() - intbuffer_ptr) * sizeof(int), n);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        n = (intbuffer_size() - intbuffer_ptr) * sizeof(int);
        tmp_intbuffer_ptr = intbuffer_ptr;
        intbuffer_ptr = intbuffer_size();
    }
    else
    {
        tmp_intbuffer_ptr = intbuffer_ptr;
        intbuffer_ptr += n / sizeof(int) + (n % sizeof(int) ? 1 : 0);
        //#ifndef CRAY         muesste so allgemeingueltig sein:
        // assure alignment to SIZEOF_ALIGNMENT (already guaranteed on Cray)
        if (intbuffer_ptr % (SIZEOF_ALIGNMENT / sizeof(int)))
            intbuffer_ptr++;
//#endif
#ifdef DEBUG
        sprintf(tmp_str, "returning complete data: %d bytes", n);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
    }
#ifdef DEBUG
    sprintf(tmp_str, "end: intbuffer_ptr: %d  intbuffer_size: %d",
            intbuffer_ptr, intbuffer_size);
    print_comment(__LINE__, __FILE__, tmp_str);
#endif
    return (char *)&intbuffer()[tmp_intbuffer_ptr];
}

// read_object assumes that all pointers are prepared correctly
// especially shm_obj_ptr points to the object that is to be read

static int level = 0;

coShmPtr *Packer::read_object(char **tmp_name)
{
    int i, no_of_els;
    coShmPtr *shm_ptr;
    int *tmp_int_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_object");
#endif
    shm_ptr = read_header(tmp_name);
    level++;

    // the following is necessary, since number_of_data_elements can be changed
    // in a recursive call of write_object!! (guess how I know and how
    // long it took to find out ;-)

    no_of_els = number_of_data_elements;
    for (i = 0; i < no_of_els; i++)
    {
        print_comment(__LINE__, __FILE__, "reading element no. %d of level %d", i, level);
        buffer->read_int(*shm_obj_ptr);
        switch (*shm_obj_ptr++)
        {
        case CHARSHM:
            read_char();
            break;
        case SHORTSHM:
            read_short();
            break;
        case INTSHM:
            read_int();
            break;
        case LONGSHM:
            read_long();
            break;
        case FLOATSHM:
            read_float();
            break;
        case DOUBLESHM:
            read_double();
            break;
        case SHMPTR:
            read_shm_pointer();
            break;
        case COVISE_NULLPTR:
            read_null_pointer();
            break;
        default:
            print_error(__LINE__, __FILE__, "unidentifiable Type in read_object: %d", *(shm_obj_ptr - 1));
            return 0;
            //	    break;
        };
    }
    print_comment(__LINE__, __FILE__, "finished level %d", level);
    level--;
    tmp_int_ptr = (int *)shm_ptr->getPtr();
    tmp_int_ptr[1] = int((shm_obj_ptr - tmp_int_ptr) * int(sizeof(int)));
    return shm_ptr;
}

coShmPtr *Packer::read_header(char **tmp_name)
{
    int tmp_shm_obj_array[8];
    int shm_obj_array_size;
    char *tmp_name_ptr;
    coShmArray *shm_arr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_header");
#endif

    // set shm_obj_ptr to temporary storage
    shm_obj_ptr = tmp_shm_obj_array;
    read_type();
    read_object_id();
    buffer->read_int(*shm_obj_ptr++);
    read_number_of_elements(); // number of elements
    shm_obj_array_size = coDoHeader::getHeaderSize() + number_of_data_elements * SIZE_PER_TYPE_ENTRY;
    if (!datamgr)
        return 0L;
    // allocate storage in shared memory for data object
    shm_ptr = datamgr->shm_alloc(CHARSHMARRAY, shm_obj_array_size);
    shm_obj_ptr = (int *)shm_ptr->getPtr();
    // copy data from temporary storage to shared memory
    *shm_obj_ptr++ = tmp_shm_obj_array[0]; // type
    shm_obj_ptr++; // skip length; will be filled in later
    *shm_obj_ptr++ = tmp_shm_obj_array[1]; // OBJECTID
    *shm_obj_ptr++ = tmp_shm_obj_array[2]; // part 1
    *shm_obj_ptr++ = tmp_shm_obj_array[3]; // part 2
    *shm_obj_ptr++ = tmp_shm_obj_array[4]; // INTSHM
    *shm_obj_ptr++ = tmp_shm_obj_array[5]; // number of elements
    buffer->read_int(*shm_obj_ptr++);
    read_int(); // version
    buffer->read_int(*shm_obj_ptr++);
    read_int(); // reference count

    buffer->read_int(*shm_obj_ptr);
    if (*shm_obj_ptr == SHMPTR)
    {
        shm_obj_ptr++;
        read_shm_pointer(); // object name; internal check_buffer_size
        shm_arr = new coShmArray(*(shm_obj_ptr - 2), *(shm_obj_ptr - 1));
        tmp_name_ptr = (char *)((coShmArray *)shm_arr)->getDataPtr();
        *tmp_name = new char[strlen(tmp_name_ptr) + 1];
        strcpy(*tmp_name, tmp_name_ptr);
        print_comment(__LINE__, __FILE__, "object name read in header");
        //print_comment(__LINE__, __FILE__, *tmp_name);
        delete shm_arr;
    }
    else
        return 0L;
    buffer->read_int(*shm_obj_ptr);
    if (*shm_obj_ptr == SHMPTR)
    {
        shm_obj_ptr++;
        read_shm_pointer(); // object attributes; internal check_buffer_size
    }
    else if (*shm_obj_ptr == COVISE_NULLPTR)
    {
        shm_obj_ptr++;
        read_null_pointer(); // object attributes; internal check_buffer_size
    }
    else
        return 0L;
    return shm_ptr;
}

int Packer::read_type()
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_type");
#endif
    buffer->read_int(*shm_obj_ptr);
    shm_obj_ptr++;
    return 1;
}

int Packer::read_number_of_elements()
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_number_of_elements");
#endif
    if (read_int())
    {
        number_of_data_elements = *(shm_obj_ptr - 1);
        return 1;
    }
    else
        return 0;
}

int Packer::read_char()
{
    int tmp_int;
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_char");
#endif

    // type is already read and routine is only called when appropriate!!!
    // (must be guaranteed by programmer!!!!

    buffer->read_int(tmp_int); // automatic conversion to int
    *(char *)shm_obj_ptr = tmp_int;
    shm_obj_ptr++; // proceed to next
    return 1;
}

int Packer::read_short()
{
    int tmp_int;
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_short");
#endif

    // type is already read and routine is only called when appropriate!!!
    // (must be guaranteed by programmer!!!!

    buffer->read_int(tmp_int); // automatic conversion to int
    *(short *)shm_obj_ptr = tmp_int;
    shm_obj_ptr++; // proceed to next
    return 1;
}

int Packer::read_int()
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_int");
#endif

    // type is already read and routine is only called when appropriate!!!
    // (must be guaranteed by programmer!!!!

    buffer->read_int(*shm_obj_ptr);
    shm_obj_ptr++; // proceed to next
    return 1;
}

int Packer::read_long()
{
    int bytes_needed;
    char *tmp_char_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_long");
#endif

// type is already read and routine is only called when appropriate!!!
// (must be guaranteed by programmer!!!!

#ifdef CRAY
    //  int == long on Cray!!
    if (convert) // at the moment cv == IEEE <=> !cv
    {
        bytes_needed = SIZEOF_IEEE_LONG;
        tmp_char_ptr = buffer->get_current_pointer_for_n_bytes(bytes_needed);
        if (bytes_needed != SIZEOF_IEEE_LONG)
            print_error(__LINE__, __FILE__, "No Buffer Space available for receiving long");
#ifdef _CRAYT3E
        converter.exch_to_long(tmp_char_ptr, (long *)shm_obj_ptr);
#else
        conv_single_int_i4c8(*(int *)tmp_char_ptr, shm_obj_ptr);
#endif
    }
    else
#endif
    {
        bytes_needed = sizeof(long);
        tmp_char_ptr = buffer->get_current_pointer_for_n_bytes(bytes_needed);
        if (bytes_needed != sizeof(long))
            print_error(__LINE__, __FILE__, "No Buffer Space available for sending long");
        int no_of_ints = sizeof(long) / sizeof(int);
        swap_bytes((unsigned int *)tmp_char_ptr, no_of_ints);
        *(long *)shm_obj_ptr = *(long *)tmp_char_ptr;
    }
    shm_obj_ptr++; // proceed to next
    return 1;
}

int Packer::read_float()
{
    int bytes_needed;
    char *tmp_char_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_float");
#endif

// type is already read and routine is only called when appropriate!!!
// (must be guaranteed by programmer!!!!

#ifdef CRAY
    if (convert) // at the moment cv == IEEE <=> !cv
    {
        bytes_needed = SIZEOF_IEEE_FLOAT;
        tmp_char_ptr = buffer->get_current_pointer_for_n_bytes(bytes_needed);
        if (bytes_needed != SIZEOF_IEEE_FLOAT)
            print_error(__LINE__, __FILE__, "No Buffer Space available for sending long");
#ifdef _CRAYT3E
        converter.exch_to_float(tmp_char_ptr, (float *)shm_obj_ptr);
#else
        conv_single_float_i4c8(*(int *)tmp_char_ptr, shm_obj_ptr);
#endif
    }
    else
#endif
    {
        bytes_needed = sizeof(float);
        tmp_char_ptr = buffer->get_current_pointer_for_n_bytes(bytes_needed);
        if (bytes_needed != sizeof(float))
            print_error(__LINE__, __FILE__, "No Buffer Space available for sending long");
        int no_of_ints = sizeof(float) / sizeof(int);
        swap_bytes((unsigned int *)tmp_char_ptr, no_of_ints);
        *(float *)shm_obj_ptr = *(float *)tmp_char_ptr;
    }
    shm_obj_ptr++; // proceed to next
    return 1;
}

int Packer::read_double()
{
    int bytes_needed;
    char *tmp_char_ptr;
#ifdef CRAY
    double tmp_double;
#endif

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_double");
#endif

// type is already read and routine is only called when appropriate!!!
// (must be guaranteed by programmer!!!!

#ifdef CRAY
    if (convert) // at the moment cv == IEEE <=> !cv
    {
        bytes_needed = SIZEOF_IEEE_DOUBLE;
        tmp_char_ptr = buffer->get_current_pointer_for_n_bytes(bytes_needed);
        tmp_double = *(double *)tmp_char_ptr;
        if (bytes_needed != SIZEOF_IEEE_DOUBLE)
            print_error(__LINE__, __FILE__, "No Buffer Space available for sending long");
#ifdef _CRAYT3E
        converter.exch_to_double(tmp_char_ptr, (double *)shm_obj_ptr);
#else
        conv_single_float_i8c8((int)tmp_double, shm_obj_ptr);
#endif
    }
    else
#endif
    {
        bytes_needed = sizeof(double);
        tmp_char_ptr = buffer->get_current_pointer_for_n_bytes(bytes_needed);
        if (bytes_needed != sizeof(double))
            print_error(__LINE__, __FILE__, "No Buffer Space available for sending long");
        int no_of_ints = sizeof(double) / sizeof(int);
        swap_bytes((unsigned int *)tmp_char_ptr, no_of_ints);
        *(double *)shm_obj_ptr = *(double *)tmp_char_ptr;
    }
    shm_obj_ptr++; // proceed to next
    return 1;
}

int Packer::read_object_id()
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_object_id");
#endif
    buffer->read_int(*shm_obj_ptr);
    shm_obj_ptr++; // skip OBJECTID
    buffer->read_int(*shm_obj_ptr); // automatic conversion to int
    shm_obj_ptr++; // 1st part
    buffer->read_int(*shm_obj_ptr); // automatic conversion to int
    shm_obj_ptr++; // 2nd part
    return 1;
}

int Packer::read_char_array()
{
    int bytes_needed, rest, length;
    char *tmp_shm_obj_ptr;
    char *tmp_char_ptr;
    coShmPtr *shm_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_char_array");
#endif

    // type is already read and routine is only called when appropriate!!!
    // (must be guaranteed by programmer!!!!

    buffer->read_int(length);
    shm_ptr = datamgr->shm_alloc(CHARSHMARRAY, length);
    *shm_obj_ptr++ = shm_ptr->get_shm_seq_no();
    *shm_obj_ptr++ = shm_ptr->get_offset();
    rest = length * sizeof(char);
    tmp_shm_obj_ptr = (char *)((coShmArray *)(void *)shm_ptr)->getDataPtr();
    while (rest > 0) // there is still something to receive
    {
        bytes_needed = rest;
        tmp_char_ptr = buffer->get_current_pointer_for_n_bytes(bytes_needed);
        memcpy(tmp_shm_obj_ptr, tmp_char_ptr, bytes_needed);
        rest -= bytes_needed;
        tmp_shm_obj_ptr += bytes_needed;
    }
    return 1;
}

int Packer::read_short_array()
{
    int bytes_needed, rest, length;
    char *tmp_shm_obj_ptr;
    char *tmp_char_ptr;
#ifdef DEBUG
    char tmp_str[100];
#endif
    coShmPtr *shm_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_short_array");
#endif

    // type is already read and routine is only called when appropriate!!!
    // (must be guaranteed by programmer!!!!

    buffer->read_int(length);
#ifdef DEBUG
    sprintf(tmp_str, "short array with %d elements", length);
    print_comment(__LINE__, __FILE__, tmp_str);
#endif
    shm_ptr = datamgr->shm_alloc(SHORTSHMARRAY, length);
    *shm_obj_ptr++ = shm_ptr->get_shm_seq_no();
    *shm_obj_ptr++ = shm_ptr->get_offset();
    tmp_shm_obj_ptr = (char *)((coShmArray *)(void *)shm_ptr)->getDataPtr();
#ifdef CRAY
    if (!convert)
        rest = length * sizeof(short);
    else
#endif
        rest = length * SIZEOF_IEEE_SHORT;
    while (rest > 0) // there is still something to receive
    {
        bytes_needed = rest;
        tmp_char_ptr = buffer->get_current_pointer_for_n_bytes(bytes_needed);
#ifdef CRAY
        if (convert)
#ifdef _CRAYT3E
            converter.exch_to_short_array(tmp_char_ptr,
                                          (short *)tmp_shm_obj_ptr,
                                          bytes_needed / SIZEOF_IEEE_SHORT);
#else
            conv_array_int_i2c8((int *)tmp_char_ptr, (int *)tmp_shm_obj_ptr,
                                bytes_needed / SIZEOF_IEEE_SHORT, START_EVEN);
#endif
        else
#endif
            memcpy(tmp_shm_obj_ptr, tmp_char_ptr, bytes_needed);
        swap_short_bytes((short unsigned int *)tmp_char_ptr, bytes_needed / sizeof(short));
        rest -= bytes_needed;
        tmp_shm_obj_ptr += bytes_needed;
    }
    return 1;
}

int Packer::read_int_array()
{
    int bytes_needed, rest, length;
    char *tmp_shm_obj_ptr;
    char *tmp_char_ptr;
#ifdef DEBUG
    char tmp_str[100];
#endif
    coShmPtr *shm_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_int_array");
#endif

    // type is already read and routine is only called when appropriate!!!
    // (must be guaranteed by programmer!!!!

    buffer->read_int(length);
#ifdef DEBUG
    sprintf(tmp_str, "int array with %d elements", length);
    print_comment(__LINE__, __FILE__, tmp_str);
#endif
    shm_ptr = datamgr->shm_alloc(INTSHMARRAY, length);
    *shm_obj_ptr++ = shm_ptr->get_shm_seq_no();
    *shm_obj_ptr++ = shm_ptr->get_offset();
    tmp_shm_obj_ptr = (char *)((coShmArray *)(void *)shm_ptr)->getDataPtr();
#ifdef CRAY
    if (!convert)
        rest = length * sizeof(int);
    else
#endif
        rest = length * SIZEOF_IEEE_INT;
    while (rest > 0) // there is still something to receive
    {
        bytes_needed = rest;
        tmp_char_ptr = buffer->get_current_pointer_for_n_bytes(bytes_needed);
#ifdef CRAY
        if (convert)
#ifdef _CRAYT3E
            converter.exch_to_int_array(tmp_char_ptr,
                                        (int *)tmp_shm_obj_ptr,
                                        bytes_needed / SIZEOF_IEEE_INT);
#else
            conv_array_int_i4c8((int *)tmp_char_ptr, (int *)tmp_shm_obj_ptr,
                                bytes_needed / SIZEOF_IEEE_INT, START_EVEN);
#endif
        else
#endif
            memcpy(tmp_shm_obj_ptr, tmp_char_ptr, bytes_needed);
        swap_bytes((unsigned int *)tmp_shm_obj_ptr, bytes_needed / sizeof(int));
        rest -= bytes_needed;
        tmp_shm_obj_ptr += bytes_needed;
    }
    return 1;
}

int Packer::read_long_array()
{
    int bytes_needed, rest, length;
    char *tmp_shm_obj_ptr;
    char *tmp_char_ptr;
#ifdef DEBUG
    char tmp_str[100];
#endif
    coShmPtr *shm_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_long_array");
#endif

    // type is already read and routine is only called when appropriate!!!
    // (must be guaranteed by programmer!!!!

    buffer->read_int(length);
#ifdef DEBUG
    sprintf(tmp_str, "long array with %d elements", length);
    print_comment(__LINE__, __FILE__, tmp_str);
#endif
    shm_ptr = datamgr->shm_alloc(LONGSHMARRAY, length);
    *shm_obj_ptr++ = shm_ptr->get_shm_seq_no();
    *shm_obj_ptr++ = shm_ptr->get_offset();
    tmp_shm_obj_ptr = (char *)((coShmArray *)(void *)shm_ptr)->getDataPtr();
#ifdef CRAY
    if (!convert)
        rest = length * sizeof(long);
    else
#endif
        rest = length * SIZEOF_IEEE_LONG;
    while (rest > 0) // there is still something to receive
    {
        bytes_needed = rest;
        tmp_char_ptr = buffer->get_current_pointer_for_n_bytes(bytes_needed);
#ifdef CRAY
        if (convert)
#ifdef _CRAYT3E
            converter.exch_to_long_array(tmp_char_ptr,
                                         (long *)tmp_shm_obj_ptr,
                                         bytes_needed / SIZEOF_IEEE_LONG);
#else
            conv_array_int_i4c8((int *)tmp_char_ptr, (int *)tmp_shm_obj_ptr,
                                bytes_needed / SIZEOF_IEEE_LONG, START_EVEN);
#endif
        else
#endif
            memcpy(tmp_shm_obj_ptr, tmp_char_ptr, bytes_needed);
        swap_bytes((unsigned int *)tmp_shm_obj_ptr, bytes_needed / sizeof(int));
        rest -= bytes_needed;
        tmp_shm_obj_ptr += bytes_needed;
    }
    return 1;
}

int Packer::read_float_array()
{
    int bytes_needed, rest, length;
    char *tmp_shm_obj_ptr;
    char *tmp_char_ptr;
#ifdef DEBUG
    char tmp_str[100];
#endif
    coShmPtr *shm_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_float_array");
#endif

    // type is already read and routine is only called when appropriate!!!
    // (must be guaranteed by programmer!!!!

    buffer->read_int(length);
#ifdef DEBUG
    sprintf(tmp_str, "float array with %d elements", length);
    print_comment(__LINE__, __FILE__, tmp_str);
#endif
    shm_ptr = datamgr->shm_alloc(FLOATSHMARRAY, length);
    *shm_obj_ptr++ = shm_ptr->get_shm_seq_no();
    *shm_obj_ptr++ = shm_ptr->get_offset();
    tmp_shm_obj_ptr = (char *)((coShmArray *)(void *)shm_ptr)->getDataPtr();
#ifdef CRAY
    if (!convert)
        rest = length * sizeof(float);
    else
#endif
        rest = length * SIZEOF_IEEE_FLOAT;
    while (rest > 0) // there is still something to receive
    {
        bytes_needed = rest;
        tmp_char_ptr = buffer->get_current_pointer_for_n_bytes(bytes_needed);
#ifdef CRAY
        if (convert)
#ifdef _CRAYT3E
            converter.exch_to_float_array(tmp_char_ptr,
                                          (float *)tmp_shm_obj_ptr,
                                          bytes_needed / SIZEOF_IEEE_FLOAT);
#else
            conv_array_float_i4c8((int *)tmp_char_ptr, (int *)tmp_shm_obj_ptr,
                                  bytes_needed / SIZEOF_IEEE_FLOAT, START_EVEN);
#endif
        else
#endif
            memcpy(tmp_shm_obj_ptr, tmp_char_ptr, bytes_needed);
        swap_bytes((unsigned int *)tmp_shm_obj_ptr, bytes_needed / sizeof(int));
        rest -= bytes_needed;
        tmp_shm_obj_ptr += bytes_needed;
    }
    return 1;
}

int Packer::read_double_array()
{
    int bytes_needed, rest, length;
    char *tmp_shm_obj_ptr;
    char *tmp_char_ptr;
#ifdef DEBUG
    char tmp_str[100];
#endif
    coShmPtr *shm_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_double_array");
#endif

    // type is already read and routine is only called when appropriate!!!
    // (must be guaranteed by programmer!!!!

    buffer->read_int(length);
#ifdef DEBUG
    sprintf(tmp_str, "double array with %d elements", length);
    print_comment(__LINE__, __FILE__, tmp_str);
#endif
    shm_ptr = datamgr->shm_alloc(DOUBLESHMARRAY, length);
    *shm_obj_ptr++ = shm_ptr->get_shm_seq_no();
    *shm_obj_ptr++ = shm_ptr->get_offset();
    tmp_shm_obj_ptr = (char *)((coShmArray *)(void *)shm_ptr)->getDataPtr();
#ifdef CRAY
    if (!convert)
        rest = length * sizeof(double);
    else
#endif
        rest = length * SIZEOF_IEEE_DOUBLE;
    while (rest > 0) // there is still something to receive
    {
        bytes_needed = rest;
        tmp_char_ptr = buffer->get_current_pointer_for_n_bytes(bytes_needed);
#ifdef CRAY
        if (convert)
#ifdef _CRAYT3E
            converter.exch_to_double_array(tmp_char_ptr,
                                           (double *)tmp_shm_obj_ptr,
                                           bytes_needed / SIZEOF_IEEE_DOUBLE);
#else
            conv_array_float_c8i8((int *)tmp_char_ptr, (int *)tmp_shm_obj_ptr,
                                  bytes_needed / SIZEOF_IEEE_DOUBLE);
#endif
        else
#endif
            memcpy(tmp_shm_obj_ptr, tmp_char_ptr, bytes_needed);
        swap_bytes((unsigned int *)tmp_shm_obj_ptr, bytes_needed / sizeof(int));
        rest -= bytes_needed;
        tmp_shm_obj_ptr += bytes_needed;
    }
    return 1;
}

/*
int Packer::read_double_array() {
    int bytes_needed, rest;
    char *tmp_char_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_double_array");
#endif
    if(*shm_obj_ptr == DOUBLESHMARRAY) {  // *shm_obj_ptr == type
      buffer->read_int(*shm_obj_ptr);
      shm_obj_ptr++; // skip DOUBLESHMARRAY
}
else {
print_error(__LINE__, __FILE__, "No DOUBLESHMARRAY found");
return 0;
}
buffer->read_int(*shm_obj_ptr);  // *shm_obj_ptr == length
#ifdef CRAY
if(!convert)
rest = *shm_obj_ptr * sizeof(double);
else
#endif
rest = *shm_obj_ptr * SIZEOF_IEEE_DOUBLE;
shm_obj_ptr++; // skip number of ints
while(rest > 0) { // there is still something to send
bytes_needed = rest;
tmp_char_ptr = buffer->get_ptr_for_n_bytes(bytes_needed);
#ifdef CRAY
if(convert)
#ifdef _CRAYT3E
converter.exch_to_double_array(tmp_char_ptr,
(double*)shm_obj_ptr,
bytes_needed/SIZEOF_IEEE_DOUBLE);
#else
conv_array_float_c8i8(tmp_char_ptr, shm_obj_ptr,
bytes_needed/SIZEOF_IEEE_DOUBLE, START_EVEN);
#endif
else
#endif
memcpy(tmp_char_ptr, shm_obj_ptr, bytes_needed);
rest -= bytes_needed;
shm_obj_ptr += (bytes_needed/sizeof(int) + (bytes_needed % sizeof(int) ? 1 : 0));
}
return 1;
}
*/

int Packer::read_null_pointer()
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_null_pointer");
#endif

    // type is already read and routine is only called when appropriate!!!
    // (must be guaranteed by programmer!!!!)

    *shm_obj_ptr++ = 0; // skip nulls
    *shm_obj_ptr++ = 0; // skip nulls
    return 1;
}

int Packer::read_shm_pointer()
{
    int type, *tmp_shm_obj_ptr;
    char *tmp_name;
    coShmPtr *shm_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_shm_pointer");
#endif

    // type (SHMPTR) is already read and routine is only called when appropriate!!!
    // shm_obj_ptr points to the address that the SHMPTR points to (shm_seq_no, offset)
    // therefore we can read the type.
    // (must be guaranteed by programmer!!!!)

    buffer->read_int(type);
    switch (type)
    {
    case CHARSHMARRAY:
        read_char_array();
        break;
    case SHORTSHMARRAY:
        read_short_array();
        break;
    case INTSHMARRAY:
        read_int_array();
        break;
    case LONGSHMARRAY:
        read_long_array();
        break;
    case FLOATSHMARRAY:
        read_float_array();
        break;
    case DOUBLESHMARRAY:
        read_double_array();
        break;
    case STRINGSHMARRAY:
        read_shm_string_array();
        break;
    case SHMPTRARRAY:
        read_shm_pointer_array();
        break;
    default: // presumably complete data object
        // set buffer back to the beginning of the object
        buffer->put_back_int();
        tmp_shm_obj_ptr = shm_obj_ptr;
        shm_ptr = read_object(&tmp_name);

        // The object could be added to object tree here if necessary.
        // If not it cannot be handled by the datamanager on request
        // and a new copy is transferred over the network and created.

        shm_obj_ptr = tmp_shm_obj_ptr;
        *shm_obj_ptr++ = shm_ptr->get_shm_seq_no();
        *shm_obj_ptr++ = shm_ptr->get_offset();
        break;
    }
    return 1;
}

int Packer::read_shm_string_array()
{
    int length, i, type;
    int *tmp_shm_obj_ptr;
    coShmPtr *shm_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_shm_string_array");
#endif

    // type is already read and routine is only called when appropriate!!!
    // (must be guaranteed by programmer!!!!)

    buffer->read_int(length); // *shm_obj_ptr == length

    shm_ptr = datamgr->shm_alloc(STRINGSHMARRAY, length);
    *shm_obj_ptr++ = shm_ptr->get_shm_seq_no();
    *shm_obj_ptr++ = shm_ptr->get_offset();

    tmp_shm_obj_ptr = shm_obj_ptr;

    shm_obj_ptr = (int *)((coShmArray *)(void *)shm_ptr)->getDataPtr();

    //
    //   +----------------+--------+------+------+------+------+----
    //   + STRINGSHMARRAY | length | shm1 | off1 | shm2 | off2 | ...
    //   +----------------+--------+------+------+------+------+----
    //
    //  length == Number of strings (not number of bytes or ints)
    //

    for (i = 0; i < length; i++)
    {
        buffer->read_int(type);
        if (type == CHARSHMARRAY)
            read_char_array();
        else
            return 0;
    }

    shm_obj_ptr = tmp_shm_obj_ptr;
    return 1;
}

int Packer::read_shm_pointer_array()
{
    int length, i, max, type;
    int *tmp_shm_obj_ptr;
    coShmPtr *shm_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::read_shm_pointer_array");
#endif

    // type is already read and routine is only called when appropriate!!!
    // (must be guaranteed by programmer!!!!)

    buffer->read_int(length); // *shm_obj_ptr == length

    max = (length / SET_CHUNK + 1) * SET_CHUNK;
    shm_ptr = datamgr->shm_alloc(SHMPTRARRAY, max);
    *shm_obj_ptr++ = shm_ptr->get_shm_seq_no();
    *shm_obj_ptr++ = shm_ptr->get_offset();

    tmp_shm_obj_ptr = shm_obj_ptr;

    shm_obj_ptr = (int *)((coShmArray *)(void *)shm_ptr)->getDataPtr();
    //
    //   +-------------+--------+------+------+------+------+----
    //   + SHMPTRARRAY | length | shm1 | off1 | shm2 | off2 | ...
    //   +-------------+--------+------+------+------+------+----
    //
    //  length == Number of objects (not number of bytes or ints)
    //

    for (i = 0; i < length; i++)
    {
        buffer->read_int(type);
        if (type == SHMPTR)
            read_shm_pointer();
        else
            return 0;
    }

    shm_obj_ptr = tmp_shm_obj_ptr;
    return 1;
}
