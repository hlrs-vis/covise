/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "dmgr_packer.h"

#undef DEBUG
/*
  -----------------------------------------------------------------------
   the object header is organized in the following way:

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
into an 8 byte chunk (SIZEOFALIGNMENT) to make it easier to read them
on the Cray. Arrays of these elements are always packed. This wastes
some space on the SGIs, but since the most part of really large data
objects is in arrays anyway it doesn't matter.

Usually data is just copied from the shared memory to the write buffer.
In case of the Cray this can involve conversion of short, int, float etc.
Standard net format is IEEE!! If two Crays are communicating, no conversion
is performed, i.e. data is written as is. If Crays send to none-Crays
always IEEE is enforced. This leads to the ugly #ifdef constructs.

-----------------------------------------------------------------------*/
#ifdef _WIN32
#undef CRAY
#endif

using namespace covise;

Packer::Packer(Message *m, int s, int o)
{
    coShmPtr *shmptr;

    shmptr = new coShmPtr(s, o);
    shm_obj_ptr = (int *)shmptr->getPtr();
    delete shmptr;
    convert = m->conn->convert_to;
    buffer = new PackBuffer(m);
    number_of_data_elements = 0;
}

#if !defined(CRAY) && !defined(__hpux) && !defined(_SX)
int* covise::PackBuffer::intbuffer()
{
    return (int*)buffer.accessData();
}
int covise::PackBuffer::intbuffer_size()
{
    return buffer.length() / sizeof(int);
}
inline
#endif
    void
    PackBuffer::send()
{
    if (intbuffer_ptr != 0)
    {
        msg->data = buffer;
        msg->data.setLength(intbuffer_ptr * sizeof(int));
        print_comment(__LINE__, __FILE__, "msg->data.length(): %d", msg->data.length());
        conn->send_msg(msg);
    }
}

void Packer::flush()
{
    buffer->send();
}

#ifndef CRAY
inline
#endif
    void
    PackBuffer::write_int(int wi)
{
#ifdef DEBUG
    char tmp_str[100];
#endif
#ifdef BYTESWAP
    unsigned int *ui;
#endif

#ifdef DEBUG
    sprintf(tmp_str, "PackBuffer::write_int %d", wi);
    print_comment(__LINE__, __FILE__, tmp_str);
#endif
    if (intbuffer_ptr >= intbuffer_size()) // if buffer full
    {
        send();
        intbuffer_ptr = 0;
    }
#ifdef CRAY
    if (convert) // at the moment cv == IEEE <=> !cv
#ifdef _CRAYT3E
        converter.int_to_exch(wi, (char *)&intbuffer()[intbuffer_ptr]);
#else
        conv_single_int_c8i4(wi, &intbuffer()[intbuffer_ptr]);
#endif
    else
        intbuffer()[intbuffer_ptr] = wi;
    intbuffer_ptr += 1;
    return;
#else
    intbuffer()[intbuffer_ptr] = wi;
#ifdef BYTESWAP
    ui = (unsigned int *)&intbuffer()[intbuffer_ptr];
    swap_byte(*ui);
#endif
    intbuffer_ptr += 2;
#endif
}

char *PackBuffer::get_ptr_for_n_bytes(int &n) // always aligned
{
    int tmp_ptr;
#ifdef DEBUG
    char tmp_str[100];
#endif

#ifdef DEBUG
    sprintf(tmp_str, "want pointer for %d bytes", n);
    print_comment(__LINE__, __FILE__, tmp_str);
    sprintf(tmp_str, "intbuffer_ptr: %d  intbuffer_size: %d",
            intbuffer_ptr, intbuffer_size);
    print_comment(__LINE__, __FILE__, tmp_str);
#endif
    // a) n < available space: return pointer and increment counter
    // b) n >= avialable space: send buffer and reset counter
    //   1) n > buffersize: return pointer and set n to buffersize
    //   2) n <= buffersize: a)

    if (intbuffer_ptr == intbuffer_size())
    {
        send();
        intbuffer_ptr = 0;
    }

    tmp_ptr = intbuffer_ptr;
    if (n <= (intbuffer_size() - intbuffer_ptr) * (int)sizeof(int))
    {
        intbuffer_ptr += n / sizeof(int) + (n % sizeof(int) ? 1 : 0);
    }
    else
    {
        n = (intbuffer_size() - intbuffer_ptr) * sizeof(int);
        intbuffer_ptr = intbuffer_size();
    }
    //#ifndef CRAY        muesste so allgemeingueltig sein:
    // assure alignment to SIZEOF_ALIGNMENT (already guaranteed on Cray)
    if (intbuffer_ptr % (SIZEOF_ALIGNMENT / sizeof(int)))
        intbuffer_ptr++;
//#endif
#ifdef DEBUG
    sprintf(tmp_str, "got pointer for %d bytes", n);
    print_comment(__LINE__, __FILE__, tmp_str);
    sprintf(tmp_str, "intbuffer_ptr: %d  intbuffer_size: %d",
            intbuffer_ptr, intbuffer_size);
    print_comment(__LINE__, __FILE__, tmp_str);
#endif
    return (char *)&intbuffer()[tmp_ptr];
}

inline int Packer::write_int()
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_int");
#endif
    buffer->write_int(*shm_obj_ptr);
    shm_obj_ptr++; // skip INTSHM
    buffer->write_int(*shm_obj_ptr);
    shm_obj_ptr++; // proceed to next
    return 1;
}

inline int Packer::write_type()
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_type");
#endif
    buffer->write_int(*shm_obj_ptr);
    shm_obj_ptr++;
    return 1;
}

inline int Packer::write_number_of_elements()
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_number_of_elements");
#endif
    number_of_data_elements = *(shm_obj_ptr + 1);
    return write_int();
}

inline int Packer::write_char()
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_char");
#endif
    buffer->write_int(*shm_obj_ptr);
    shm_obj_ptr++; // skip CHARSHM
    buffer->write_int(*(char *)shm_obj_ptr); // automatic conversion to int
    shm_obj_ptr++; // proceed to next
    return 1;
}

inline int Packer::write_short()
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_short");
#endif
    buffer->write_int(*shm_obj_ptr);
    shm_obj_ptr++; // skip SHORTSHM
    buffer->write_int(*(short *)shm_obj_ptr); // automatic conversion to int
    shm_obj_ptr++; // proceed to next
    return 1;
}

int Packer::write_long()
{
    int bytes_needed;
    char *tmp_char_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_long");
#endif
    buffer->write_int(*shm_obj_ptr);
    shm_obj_ptr++; // skip LONGSHM
#ifdef CRAY
    //  int == long on Cray!!
    if (convert) // at the moment cv == IEEE <=> !cv
    {
        bytes_needed = SIZEOF_IEEE_LONG;
        tmp_char_ptr = buffer->get_ptr_for_n_bytes(bytes_needed);
        if (bytes_needed != SIZEOF_IEEE_LONG)
            print_error(__LINE__, __FILE__, "No Buffer Space available for sending long");
#ifdef _CRAYT3E
        converter.long_to_exch(*(long *)shm_obj_ptr, tmp_char_ptr);
#else
        conv_single_int_c8i4((int)*(long *)shm_obj_ptr, (int *)tmp_char_ptr);
#endif
    }
    else
#endif
    {
        bytes_needed = sizeof(long);
        tmp_char_ptr = buffer->get_ptr_for_n_bytes(bytes_needed);
        if (bytes_needed != sizeof(long))
            print_error(__LINE__, __FILE__, "No Buffer Space available for sending long");
        *(long *)tmp_char_ptr = *(long *)shm_obj_ptr;
        int no_of_ints = sizeof(long) / sizeof(int);
        swap_bytes((unsigned int *)tmp_char_ptr, no_of_ints);
    }
    shm_obj_ptr++; // proceed to next
    return 1;
}

int Packer::write_float()
{
    int bytes_needed;
    char *tmp_char_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_float");
#endif
    buffer->write_int(*shm_obj_ptr);
    shm_obj_ptr++; // skip FLOATSHM
#ifdef CRAY
    if (convert) // at the moment cv == IEEE <=> !cv
    {
        bytes_needed = SIZEOF_IEEE_FLOAT;
        tmp_char_ptr = buffer->get_ptr_for_n_bytes(bytes_needed);
        if (bytes_needed != SIZEOF_IEEE_FLOAT)
            print_error(__LINE__, __FILE__, "No Buffer Space available for sending long");
#ifdef _CRAYT3E
        converter.float_to_exch(*(float *)shm_obj_ptr, tmp_char_ptr);
#else
        conv_single_float_c8i4((int)*(float *)shm_obj_ptr, (int *)tmp_char_ptr);
#endif
    }
    else
#endif
    {
        bytes_needed = sizeof(float);
        tmp_char_ptr = buffer->get_ptr_for_n_bytes(bytes_needed);
        if (bytes_needed != sizeof(float))
            print_error(__LINE__, __FILE__, "No Buffer Space available for sending long");
        *(float *)tmp_char_ptr = *(float *)shm_obj_ptr;
        int no_of_ints = sizeof(float) / sizeof(int);
        swap_bytes((unsigned int *)tmp_char_ptr, no_of_ints);
    }
    shm_obj_ptr++; // proceed to next
    return 1;
}

int Packer::write_double()
{
    int bytes_needed;
    char *tmp_char_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_double");
#endif
    buffer->write_int(*shm_obj_ptr);
    shm_obj_ptr++; // skip DOUBLESHM
#ifdef CRAY
    if (convert) // at the moment cv == IEEE <=> !cv
    {
        bytes_needed = SIZEOF_IEEE_DOUBLE;
        tmp_char_ptr = buffer->get_ptr_for_n_bytes(bytes_needed);
        if (bytes_needed != SIZEOF_IEEE_DOUBLE)
            print_error(__LINE__, __FILE__, "No Buffer Space available for sending long");
#ifdef _CRAYT3E
        converter.double_to_exch(*(double *)shm_obj_ptr, tmp_char_ptr);
#else
        conv_single_float_c8i8((int)*(double *)shm_obj_ptr, (int *)tmp_char_ptr);
#endif
    }
    else
#endif
    {
        bytes_needed = sizeof(double);
        tmp_char_ptr = buffer->get_ptr_for_n_bytes(bytes_needed);
        if (bytes_needed != sizeof(double))
            print_error(__LINE__, __FILE__, "No Buffer Space available for sending long");
        *(double *)tmp_char_ptr = *(double *)shm_obj_ptr;
        int no_of_ints = sizeof(double) / sizeof(int);
        swap_bytes((unsigned int *)tmp_char_ptr, no_of_ints);
    }
    shm_obj_ptr++; // proceed to next
    return 1;
}

inline int Packer::write_object_id()
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_object_id");
#endif
    buffer->write_int(*shm_obj_ptr);
    shm_obj_ptr++; // skip OBJECTID
    buffer->write_int(*shm_obj_ptr); // automatic conversion to int
    shm_obj_ptr++; // 1st part
    buffer->write_int(*shm_obj_ptr); // automatic conversion to int
    shm_obj_ptr++; // 2nd part
    return 1;
}

int Packer::write_char_array()
{
    int bytes_needed, rest;
    char *tmp_char_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_char_array");
#endif
    if (*shm_obj_ptr == CHARSHMARRAY) // *shm_obj_ptr == type
    {
        buffer->write_int(*shm_obj_ptr);
        shm_obj_ptr++; // skip CHARSHMARRAY
    }
    else
    {
        print_error(__LINE__, __FILE__, "No CHARSHMARRAY found");
        return 0;
    }
    buffer->write_int(*shm_obj_ptr); // *shm_obj_ptr == length
    rest = *shm_obj_ptr * sizeof(char);
    shm_obj_ptr++; // skip number of characters
    while (rest > 0) // there is still something to send
    {
        bytes_needed = rest;
        tmp_char_ptr = buffer->get_ptr_for_n_bytes(bytes_needed);
        memcpy(tmp_char_ptr, shm_obj_ptr, bytes_needed);
        rest -= bytes_needed;
        shm_obj_ptr += (bytes_needed / sizeof(int) + (bytes_needed % sizeof(int) ? 1 : 0));
    }
    return 1;
}

int Packer::write_short_array()
{
    int bytes_needed, rest;
    char *tmp_char_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_short_array");
#endif
    if (*shm_obj_ptr == SHORTSHMARRAY) // *shm_obj_ptr == type
    {
        buffer->write_int(*shm_obj_ptr);
        shm_obj_ptr++; // skip SHORTSHMARRAY
    }
    else
    {
        print_error(__LINE__, __FILE__, "No SHORTSHMARRAY found");
        return 0;
    }
    buffer->write_int(*shm_obj_ptr); // *shm_obj_ptr == length
#ifdef CRAY
    if (!convert)
        rest = *shm_obj_ptr * sizeof(short);
    else
#endif
        rest = *shm_obj_ptr * SIZEOF_IEEE_SHORT;
    shm_obj_ptr++; // skip number of shorts
    while (rest > 0) // there is still something to send
    {
        bytes_needed = rest;
        tmp_char_ptr = buffer->get_ptr_for_n_bytes(bytes_needed);
#ifdef CRAY
        if (convert)
#ifdef _CRAYT3E
            converter.short_array_to_exch((short *)shm_obj_ptr,
                                          tmp_char_ptr,
                                          bytes_needed / SIZEOF_IEEE_SHORT);
#else
            conv_array_int_c8i2(shm_obj_ptr, (int *)tmp_char_ptr,
                                bytes_needed / SIZEOF_IEEE_SHORT, START_EVEN);
#endif
        else
#endif
            memcpy(tmp_char_ptr, shm_obj_ptr, bytes_needed);
        swap_short_bytes((short unsigned int *)tmp_char_ptr, bytes_needed / sizeof(short));

        rest -= bytes_needed;
        shm_obj_ptr += (bytes_needed / sizeof(int) + (bytes_needed % sizeof(int) ? 1 : 0));
    }
    return 1;
}

int Packer::write_int_array()
{
    int bytes_needed, rest;
    char *tmp_char_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_int_array");
#endif
    if (*shm_obj_ptr == INTSHMARRAY) // *shm_obj_ptr == type
    {
        buffer->write_int(*shm_obj_ptr);
        shm_obj_ptr++; // skip INTSHMARRAY
    }
    else
    {
        print_error(__LINE__, __FILE__, "No INTSHMARRAY found");
        return 0;
    }
    buffer->write_int(*shm_obj_ptr); // *shm_obj_ptr == length
#ifdef CRAY
    if (!convert)
        rest = *shm_obj_ptr * sizeof(int);
    else
#endif
        rest = *shm_obj_ptr * SIZEOF_IEEE_INT;
    shm_obj_ptr++; // skip number of ints
    while (rest > 0) // there is still something to send
    {
        bytes_needed = rest;
        tmp_char_ptr = buffer->get_ptr_for_n_bytes(bytes_needed);
#ifdef CRAY
        if (convert)
#ifdef _CRAYT3E
            converter.int_array_to_exch((int *)shm_obj_ptr,
                                        tmp_char_ptr,
                                        bytes_needed / SIZEOF_IEEE_INT);
#else
            conv_array_int_c8i4(shm_obj_ptr, (int *)tmp_char_ptr,
                                bytes_needed / SIZEOF_IEEE_INT, START_EVEN);
#endif
        else
#endif
            memcpy(tmp_char_ptr, shm_obj_ptr, bytes_needed);
        swap_bytes((unsigned int *)tmp_char_ptr, bytes_needed / sizeof(int));
        rest -= bytes_needed;
        shm_obj_ptr += (bytes_needed / sizeof(int) + (bytes_needed % sizeof(int) ? 1 : 0));
    }
    return 1;
}

int Packer::write_long_array()
{
    int bytes_needed, rest;
    char *tmp_char_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_long_array");
#endif
    if (*shm_obj_ptr == LONGSHMARRAY) // *shm_obj_ptr == type
    {
        buffer->write_int(*shm_obj_ptr);
        shm_obj_ptr++; // skip LONGSHMARRAY
    }
    else
    {
        print_error(__LINE__, __FILE__, "No LONGSHMARRAY found");
        return 0;
    }
    buffer->write_int(*shm_obj_ptr); // *shm_obj_ptr == length
#ifdef CRAY
    if (!convert)
        rest = *shm_obj_ptr * sizeof(long);
    else
#endif
        rest = *shm_obj_ptr * SIZEOF_IEEE_LONG;
    shm_obj_ptr++; // skip number of longs
    while (rest > 0) // there is still something to send
    {
        bytes_needed = rest;
        tmp_char_ptr = buffer->get_ptr_for_n_bytes(bytes_needed);
#ifdef CRAY
        if (convert)
#ifdef _CRAYT3E
            converter.long_array_to_exch((long *)shm_obj_ptr,
                                         tmp_char_ptr,
                                         bytes_needed / SIZEOF_IEEE_LONG);
#else
            conv_array_int_c8i4(shm_obj_ptr, (int *)tmp_char_ptr,
                                bytes_needed / SIZEOF_IEEE_LONG, START_EVEN);
#endif
        else
#endif
            memcpy(tmp_char_ptr, shm_obj_ptr, bytes_needed);
        swap_bytes((unsigned int *)tmp_char_ptr, bytes_needed / sizeof(int));
        rest -= bytes_needed;
        shm_obj_ptr += (bytes_needed / sizeof(int) + (bytes_needed % sizeof(int) ? 1 : 0));
    }
    return 1;
}

int Packer::write_float_array()
{
    int bytes_needed, rest;
    char *tmp_char_ptr;
    int no_of_floats;
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_float_array");
#endif
    if (*shm_obj_ptr == FLOATSHMARRAY) // *shm_obj_ptr == type
    {
        buffer->write_int(*shm_obj_ptr);
        shm_obj_ptr++; // skip FLOATSHMARRAY
    }
    else
    {
        print_error(__LINE__, __FILE__, "No FLOATSHMARRAY found");
        return 0;
    }
    buffer->write_int(*shm_obj_ptr); // *shm_obj_ptr == length
    no_of_floats = *shm_obj_ptr;
#ifdef CRAY
    if (!convert)
        rest = no_of_floats * sizeof(float);
    else
#endif
        rest = no_of_floats * SIZEOF_IEEE_FLOAT;
    shm_obj_ptr++; // skip number of floats
    while (rest > 0) // there is still something to send
    {
        bytes_needed = rest;
        print_comment(__LINE__, __FILE__, "bytes needed: %d", bytes_needed);
        tmp_char_ptr = buffer->get_ptr_for_n_bytes(bytes_needed);
        print_comment(__LINE__, __FILE__, "bytes got: %d", bytes_needed);
#ifdef CRAY
        if (convert)
        {
#ifdef _CRAYT3E
            converter.float_array_to_exch((float *)shm_obj_ptr,
                                          tmp_char_ptr,
                                          bytes_needed / SIZEOF_IEEE_FLOAT);
#else
            conv_array_float_c8i4(shm_obj_ptr, (int *)tmp_char_ptr,
                                  bytes_needed / SIZEOF_IEEE_FLOAT, START_EVEN);
#endif
            rest -= bytes_needed;
            shm_obj_ptr += (bytes_needed / SIZEOF_IEEE_FLOAT + (bytes_needed % SIZEOF_IEEE_FLOAT ? 1 : 0));
        }
        else
#endif
        {
            memcpy(tmp_char_ptr, shm_obj_ptr, bytes_needed);
            swap_bytes((unsigned int *)tmp_char_ptr, bytes_needed / sizeof(int));
            rest -= bytes_needed;
            shm_obj_ptr += (bytes_needed / sizeof(int) + (bytes_needed % sizeof(int) ? 1 : 0));
        }
    }
    return 1;
}

int Packer::write_double_array()
{
    int bytes_needed, rest;
    char *tmp_char_ptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_double_array");
#endif
    if (*shm_obj_ptr == DOUBLESHMARRAY) // *shm_obj_ptr == type
    {
        buffer->write_int(*shm_obj_ptr);
        shm_obj_ptr++; // skip DOUBLESHMARRAY
    }
    else
    {
        print_error(__LINE__, __FILE__, "No DOUBLESHMARRAY found");
        return 0;
    }
    buffer->write_int(*shm_obj_ptr); // *shm_obj_ptr == length
#ifdef CRAY
    if (!convert)
        rest = *shm_obj_ptr * sizeof(double);
    else
#endif
        rest = *shm_obj_ptr * SIZEOF_IEEE_DOUBLE;
    shm_obj_ptr++; // skip number of ints
    while (rest > 0) // there is still something to send
    {
        bytes_needed = rest;
        tmp_char_ptr = buffer->get_ptr_for_n_bytes(bytes_needed);
#ifdef CRAY
        if (convert)
#ifdef _CRAYT3E
            converter.double_array_to_exch((double *)shm_obj_ptr,
                                           tmp_char_ptr,
                                           bytes_needed / SIZEOF_IEEE_DOUBLE);
#else
            conv_array_float_c8i8(shm_obj_ptr, (int *)tmp_char_ptr,
                                  bytes_needed / SIZEOF_IEEE_DOUBLE);
#endif
        else
#endif
            memcpy(tmp_char_ptr, shm_obj_ptr, bytes_needed);
        swap_bytes((unsigned int *)tmp_char_ptr, bytes_needed / sizeof(int));
        rest -= bytes_needed;
        shm_obj_ptr += (bytes_needed / sizeof(int) + (bytes_needed % sizeof(int) ? 1 : 0));
    }
    return 1;
}

#if !defined(CRAY) && !defined(__hpux) && !defined(_SX)
inline
#endif
    int
    Packer::write_null_pointer()
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_null_pointer");
#endif
    if (*shm_obj_ptr == COVISE_NULLPTR) // *shm_obj_ptr == type
    {
        buffer->write_int(*shm_obj_ptr);
        shm_obj_ptr++; // skip NULLPTR
    }
    else
    {
        print_error(__LINE__, __FILE__, "No NULLPTR found");
        return 0;
    }
    shm_obj_ptr++; // skip nulls
    shm_obj_ptr++; // skip nulls
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Ende Packer::write_null_pointer");
#endif
    return 1;
}

//int Packer::write_shm_pointer(int transfer_array)
int Packer::write_shm_pointer()
{
    coShmPtr *shmptr;
    int *tmp_shm_buffer;
    int retval;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_shm_pointer");
#endif
    if (*shm_obj_ptr == SHMPTR) // *shm_obj_ptr == type
    {
        buffer->write_int(*shm_obj_ptr);
        shm_obj_ptr++; // skip SHMPTR
    }
    else
    {
        if (*shm_obj_ptr == COVISE_NULLPTR)
            return write_null_pointer();
        else
        {
            print_error(__LINE__, __FILE__, "No SHMPTR found");
            return 0;
        }
    }
    shmptr = new coShmPtr(*shm_obj_ptr, *(shm_obj_ptr + 1));
    shm_obj_ptr += 2;
    tmp_shm_buffer = shm_obj_ptr;
    shm_obj_ptr = (int *)shmptr->getPtr();
    delete shmptr;
    retval = write_shm_pointer_direct(); // aw: was (transfer_array)
    shm_obj_ptr = tmp_shm_buffer;
    return retval;
}

//int Packer::write_shm_pointer_direct(int /*transfer_array*/) {
int Packer::write_shm_pointer_direct()
{

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_shm_pointer_direct");
#endif
    switch (*shm_obj_ptr)
    {
    case CHARSHMARRAY:
        write_char_array();
        break;
    case SHORTSHMARRAY:
        write_short_array();
        break;
    case INTSHMARRAY:
        write_int_array();
        break;
    case LONGSHMARRAY:
        write_long_array();
        break;
    case FLOATSHMARRAY:
        write_float_array();
        break;
    case DOUBLESHMARRAY:
        write_double_array();
        break;
    case STRINGSHMARRAY:
        write_shm_string_array();
        break;
    case SHMPTRARRAY:
        write_shm_pointer_array();
        break;
    default: // presumably complete data object
        write_object();
        break;
    }
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Ende Packer::write_shm_pointer");
#endif
    return 1;
}

int Packer::write_shm_string_array()
{
    int length, i;
    int *tmp_shm_obj_ptr;
    coShmPtr *shmptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_shm_string_array");
#endif
    if (*shm_obj_ptr == STRINGSHMARRAY) // *shm_obj_ptr == type
    {
        buffer->write_int(*shm_obj_ptr);
        shm_obj_ptr++; // skip STRINGSHMARRAY
    }
    else
    {
        print_error(__LINE__, __FILE__, "No STRINGSHMARRAY found");
        return 0;
    }
    buffer->write_int(*shm_obj_ptr); // *shm_obj_ptr == length
    length = *shm_obj_ptr;
    shm_obj_ptr++; // skip length

    //
    //   +----------------+--------+------+------+------+------+----
    //   + STRINGSHMARRAY | length | shm1 | off1 | shm2 | off2 | ...
    //   +----------------+--------+------+------+------+------+----
    //
    //  length == Number of strings (not number of bytes or ints)
    //

    for (i = 0; i < length; i++)
    {
        shmptr = new coShmPtr(*shm_obj_ptr, *(shm_obj_ptr + 1));
        tmp_shm_obj_ptr = shm_obj_ptr + 2;
        shm_obj_ptr = (int *)shmptr->getPtr();
        delete shmptr;
        write_char_array();
        shm_obj_ptr = tmp_shm_obj_ptr;
    }
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Ende Packer::write_shm_string_array");
#endif
    return 1;
}

int Packer::write_shm_pointer_array()
{
    int length, i;
    int *tmp_shm_obj_ptr;
    coShmPtr *shmptr;

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_shm_pointer_array");
#endif
    if (*shm_obj_ptr == SHMPTRARRAY) // *shm_obj_ptr == type
    {
        buffer->write_int(*shm_obj_ptr);
        shm_obj_ptr++; // skip SHMPTRARRAY
    }
    else
    {
        print_error(__LINE__, __FILE__, "No SHMPTRARRAY found");
        return 0;
    }

    length = *shm_obj_ptr;
    shm_obj_ptr++; // skip length
    for (i = 0; i < length; i++)
        if (shm_obj_ptr[2 * i] == 0)
            break;
    length = i;
    buffer->write_int(length); // *shm_obj_ptr == length

    //
    //   +-------------+--------+------+------+------+------+----
    //   + SHMPTRARRAY | length | shm1 | off1 | shm2 | off2 | ...
    //   +-------------+--------+------+------+------+------+----
    //
    //  length == maximal Number of objects (not number of bytes or ints)
    //  actual length determined by the first 0 entry (see above)
    //

    for (i = 0; i < length; i++)
    {
        if (*shm_obj_ptr == 0)
        {
            print_comment(__LINE__, __FILE__, "only %d items of SHM_PTR_ARRAY of length %d used", i, length);
            break;
        }
        print_comment(__LINE__, __FILE__, "itm %d of %d, SHM_PTR_ARRAY", i, length);
        shmptr = new coShmPtr(*shm_obj_ptr, *(shm_obj_ptr + 1));
        tmp_shm_obj_ptr = shm_obj_ptr + 2;
        shm_obj_ptr = (int *)shmptr->getPtr();
        delete shmptr;
        buffer->write_int(SHMPTR);
        write_shm_pointer_direct();
        shm_obj_ptr = tmp_shm_obj_ptr;
    }
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Ende Packer::write_shm_pointer_array");
#endif
    return 1;
}

// write_object assumes that all pointers are prepared correctly
// especially shm_obj_ptr points to the object that is to be written

#ifdef DEBUG
static int level = 0;
#endif

int Packer::write_object()
{
    int i, no_of_els;
//int transfer_array = coDistributedObject::get_transfer_arrays();
#ifdef DEBUG
    char tmp_str[100];
#endif

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_object");
    level++;
#endif
    write_header();

    // the following is necessary, since number_of_data_elements can be changed
    // in a recursive call of write_object!!

    no_of_els = number_of_data_elements;
    for (i = 0; i < no_of_els; i++)
    {
#ifdef DEBUG
        sprintf(tmp_str, "writing element no. %d of level %d", i, level);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        switch (*shm_obj_ptr)
        {
        case CHARSHM:
            write_char();
            break;
        case SHORTSHM:
            write_short();
            break;
        case INTSHM:
            write_int();
            break;
        case LONGSHM:
            write_long();
            break;
        case FLOATSHM:
            write_float();
            break;
        case DOUBLESHM:
            write_double();
            break;
        case SHMPTR:
            //write_shm_pointer(transfer_array);
            write_shm_pointer(); //
            break;
        case COVISE_NULLPTR:
            write_null_pointer();
            break;
        default:
            print_error(__LINE__, __FILE__, "unidentifiable Type in write_object");
#ifdef DEBUG
            level--;
            print_comment(__LINE__, __FILE__, "Ende Packer::write_object");
#endif
            return 0;
            //	    break;
        };
    }
#ifdef DEBUG
    sprintf(tmp_str, "finished level %d", level);
    print_comment(__LINE__, __FILE__, tmp_str);
    level--;
    print_comment(__LINE__, __FILE__, "Ende Packer::write_object");
#endif
    return 1;
}

int Packer::write_header()
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Packer::write_header");
#endif
    write_type();
    shm_obj_ptr++; // skip local byte-length information
    write_object_id();
    write_number_of_elements(); // number of elements
    write_int(); // version
    write_int(); // reference count
    write_shm_pointer(); // object name; internal check_buffer_size
    write_shm_pointer(); // object attributes; internal check_buffer_size
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "Ende Packer::write_header");
#endif
    return 1;
}
