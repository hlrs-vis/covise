/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_PACKER_H
#define EC_PACKER_H

#include "dmgr.h"
#include <covise/covise.h>
#include <net/dataHandle.h>
#ifdef shm_ptr
#undef shm_ptr
#endif

namespace covise
{

const int SIZEOF_IEEE_CHAR = 1;
const int SIZEOF_IEEE_SHORT = 2;
const int SIZEOF_IEEE_INT = 4;
const int SIZEOF_IEEE_LONG = 4;
const int SIZEOF_IEEE_FLOAT = 4;
const int SIZEOF_IEEE_DOUBLE = 8;

const int IOVEC_MIN_SIZE = 10000;
#ifdef CRAY
const int OBJECT_BUFFER_SIZE = 25000 * SIZEOF_ALIGNMENT;
const int IOVEC_MAX_LENGTH = 1;
#else
const int OBJECT_BUFFER_SIZE = 50000 * SIZEOF_ALIGNMENT;
const int IOVEC_MAX_LENGTH = 16;
#endif

// the following computes the size of a type entry for a data object
// usually: TYPE + Data (for char, short, int, etc.) or
//          TYPE + SHM_SEQ_NO + OFFSET (for shmptr, arrays, etc.)
// size is in integers

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
const int MAX_INT_PER_DATA = MAX((sizeof(double) / sizeof(int)), 2);
const int SIZE_PER_TYPE_ENTRY = sizeof(int) + MAX_INT_PER_DATA * sizeof(int);

class DMGREXPORT PackBuffer
{
private:
    covise::DataHandle buffer; // Buffer for write
    int *intbuffer(); // integer pointer to write buffer
    int intbuffer_ptr; // current field for integer write buffer
    int intbuffer_size(); // integer size of write buffer
    int convert; // conversion necessary?
    Message *msg; // message that will be sent
    Connection *conn; // connection through which the message will be sent
    DataManagerProcess *datamgr; // to allow shm_alloc
public:
    //initialize for receive
    PackBuffer(DataManagerProcess *dm, Message *m)
    {
        datamgr = dm;
        msg = m;
        conn = msg->conn;
        convert = conn->convert_to;
        buffer = msg->data;
        msg->data = DataHandle{};
        intbuffer_ptr = 0;
    };
    PackBuffer(Message *m) // initialize for send
    {
        datamgr = 0L;
        msg = m;
        conn = msg->conn;
        convert = conn->convert_to;
        buffer = DataHandle{ OBJECT_BUFFER_SIZE };
        intbuffer_ptr = 0;
    };
    ~PackBuffer()
    {
        //	delete msg;
    };
    void send();
    void receive();
    char *get_ptr_for_n_bytes(int &n); // returns pointer to buffer and
    // sets n to length of available space (always aligned to SIZEOF_ALIGNMENT)
    void write_int(int i);
    void read_int(int &i);
    void put_back_int();
    char *get_current_pointer_for_n_bytes(int &n);
    void skip_n_bytes(int n); // returns pointer to buffer and
};

class DMGREXPORT Packer
{
    PackBuffer *buffer; // Buffer to fill for sending/to empty for receiving
    int *shm_obj_ptr; // pointer to the current working position in the object
    int convert; // conversion flag
    int number_of_data_elements; // number of elements to process (does not
    // include header)
    coShmPtr *shm_ptr;
    DataManagerProcess *datamgr; // to allow shm_alloc
#ifndef CRAY
    static int iovcovise_arr[IOVEC_MAX_LENGTH];
#endif
    int get_buffer_ptr(int);
    int write_object();
    int write_header();
    int write_type();
    int write_object_id();
    int write_char();
    int write_short();
    int write_int();
    int write_long();
    int write_float();
    int write_double();
    int write_char_array();
    int write_short_array();
    int write_int_array();
    int write_long_array();
    int write_float_array();
    int write_double_array();
    int write_shm_string_array();
    int write_shm_pointer_array();
    int write_null_pointer();
    //int write_shm_pointer(int transfer_array = 1);
    //int write_shm_pointer_direct(int transfer_array = 1);
    int write_shm_pointer();
    int write_shm_pointer_direct();
    int write_number_of_elements();
    coShmPtr *read_object(char **);
    coShmPtr *read_header(char **);
    int read_type();
    int read_object_id();
    int read_char();
    int read_short();
    int read_int();
    int read_long();
    int read_float();
    int read_double();
    int read_char_array();
    int read_short_array();
    int read_int_array();
    int read_long_array();
    int read_float_array();
    int read_double_array();
    int read_shm_string_array();
    int read_shm_pointer_array();
    int read_null_pointer();
    int read_shm_pointer();
    int read_number_of_elements();

public:
    Packer(Message *m, int s, int o);
    Packer(Message *m, DataManagerProcess *dm);
    Packer();
    ~Packer()
    {
        delete buffer;
    };
    int pack()
    {
        return write_object();
    };
    coShmPtr *unpack(char **tmp_name)
    {
        return read_object(tmp_name);
    };
    void flush();
};
}
#endif
