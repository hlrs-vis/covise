/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "dmgr.h"
#define AVL_EXTERN extern
#include "dmgr_mem_avltrees.h"
#undef AVL_EXTERN

#ifdef shm_ptr
#undef shm_ptr
#endif

using namespace covise;

coShmAlloc::coShmAlloc(int *key, DataManagerProcess *d)
    : ShmAccess(key)
{
    dmgrproc = d;
    MemChunk *mnode = new_memchunk(shm->get_seq_no(),
                                   shm->get_pointer(shm->get_seq_no()), ShmConfig::getMallocSize());

    used_list = new AddressOrderedTree();

    free_list = new AddressOrderedTree();
    free_list->insert_chunk(mnode);
    free_size_list = new SizeOrderedTree();
    free_size_list->insert_chunk(mnode);
#ifdef DEBUG
    print();
#endif
}

coShmPtr *coShmAlloc::malloc(shmSizeType size)
{
    int *msg_data = new int[2];
    int tmp_key = 0;
    shmSizeType new_size;
    SharedMemory *new_shm;
    MemChunk *mnode;
    MemChunk *new_used_node;
    Message *msg;

    if (size % SIZEOF_ALIGNMENT != 0)
        size += (SIZEOF_ALIGNMENT - (size % SIZEOF_ALIGNMENT));
#ifdef DEBUG
    sprintf(tmp_str, "malloc size: %d", size);
    print_comment(__LINE__, __FILE__, tmp_str);
#endif
    MemChunk *free_node = free_size_list->get_chunk(size);
    if (!free_node)
    {
        print_comment(__LINE__, __FILE__, "new SharedMemory");
        if (size > ShmConfig::getMallocSize())
        {
            uint64_t tmpSize = (size / ShmConfig::getMallocSize() + 1) * ShmConfig::getMallocSize();
            
            new_size = (shmSizeType)tmpSize;
            if (tmpSize != new_size)
            {
                new_size = size;
            }
        }
        else
        {
            new_size = ShmConfig::getMallocSize();
        }
        new_shm = new SharedMemory(&tmp_key, new_size);
        print_comment(__LINE__, __FILE__, "key: %d  size: %d", tmp_key, new_size);
        print_comment(__LINE__, __FILE__, "seq_no: %d  ptr: %llx", new_shm->get_seq_no(),
                      ( unsigned long long)new_shm->get_pointer());
        mnode = new_memchunk(new_shm->get_seq_no(),
                             new_shm->get_pointer(), new_size);
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "new Memnode:");
        mnode->print();
#endif
        free_list->insert_chunk(mnode);
        free_size_list->insert_chunk(mnode);
        msg_data[0] = tmp_key;
        msg_data[1] = new_size;
        msg = new Message(COVISE_MESSAGE_NEW_SDS, DataHandle((char *)&msg_data[0], 2 * sizeof(int)));
        print_comment(__LINE__, __FILE__, "dmgrproc->send_to_all_connections");
        dmgrproc->send_to_all_connections(msg);
        free_node = free_size_list->get_chunk(size);
    }
    free_list->remove_chunk(free_node);
    if (free_node->get_plain_size() != size)
    {
        new_used_node = free_node->split(size);
        free_size_list->insert_chunk(free_node); // resort the changed list
        free_list->insert_chunk(free_node); // resort the changed list
    }
    else
    {
        new_used_node = free_node;
    }
    used_list->insert_chunk(new_used_node);
#ifdef DEBUG
    print();
    new_used_node->print();
    new_used_node->getAddress()->print();
#endif

    return new_used_node->getAddress();
}

void coShmAlloc::free(int shm_seq_no, shmSizeType offset)
{
    char *tmpptr = (char *)shm->get_pointer(shm_seq_no);
    char *shm_ptr = tmpptr + offset;
    MemChunk *next_chunk, *used_node, s_node;
    static int garbage_count = 0;

    s_node.set(shm_seq_no, shm_ptr, 0);
    used_node = used_list->remove_chunk(&s_node);
    if (used_node == 0L)
        return;

    s_node.set(shm_seq_no, (char *)(shm_ptr + used_node->get_plain_size()), 0);
    next_chunk = free_list->remove_chunk(&s_node);
    if (next_chunk)
    {
        used_node->increase_size(next_chunk->get_plain_size());
        free_size_list->remove_chunk(next_chunk);
        delete_memchunk(next_chunk);
    }
    free_list->insert_chunk(used_node);
    free_size_list->insert_chunk(used_node);
    if (garbage_count == 1000)
    {
        collect_garbage();
        garbage_count = 0;
    }
}

//extern int covise_list_size;

static int covise_list_size;

void coShmAlloc::print()
{
    covise_list_size = 0;
    //    free_list->print("free list");
    print_comment(__LINE__, __FILE__, "free list: %d bytes ===============", covise_list_size);
    covise_list_size = 0;
    //    free_size_list->print("free size list");
    print_comment(__LINE__, __FILE__, "free size list: %d bytes ==========", covise_list_size);
    covise_list_size = 0;
    //    used_list->print("used list");
    print_comment(__LINE__, __FILE__, "used list: %d bytes ======================", covise_list_size);
}

void coShmAlloc::new_desk(void)
{
    MemChunk *mnode;
    SharedMemory *p_shm;
    int seq_no;
    shmSizeType size;

    if (used_list)
        used_list->empty_trees(1);
    if (free_list)
        free_list->empty_trees(0);
    if (free_size_list)
        free_size_list->empty_tree();
    p_shm = get_shared_memory();
    while (p_shm)
    {
        if (p_shm->is_attached())
        {
            seq_no = p_shm->get_seq_no();
            size = p_shm->get_size();
            size -= 2 * (sizeof(int));
            mnode = new_memchunk(seq_no, p_shm->get_pointer(seq_no), size);
            free_list->insert_chunk(mnode);
            free_size_list->insert_chunk(mnode);
        }
        p_shm = p_shm->get_next_shm();
    }
}

void MemChunk::print()
{
    print_comment(__LINE__, __FILE__, "address: %llx  size: %lld ", (unsigned long long)&address, (long long)size);
    covise_list_size += size;
}
