/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef DMGR_AVL_TREE_H
#define DMGR_AVL_TREE_H

#include <shm/covise_shm.h>

//*****************************************************************//
// search for a special value in a binary tree. return value is
//  if search==-2:  the biggest node with value < data,
//  if search==+2:  the smallest node with value > data,
//  if search==-1:  the biggest node with value <= data,
//  if search==+1:  the smallest node with value >= data,
//  if search== 0:  the node with value == data,
//        or NULL, if the condition cannot be satisfied.
//*****************************************************************//

/*
const int GREATER_THAN = 2;
const int GT_EQUAL = 1;
const int EQUAL = 0;
const int LS_EQUAL = -1;
const int LESS_THAN = -2;
*/

#ifndef GREATER_THAN
#define GREATER_THAN 2
#endif
#ifndef GT_EQUAL
#define GT_EQUAL 1
#endif
#ifndef EQUAL
#define EQUAL 0
#endif
#ifndef LS_EQUAL
#define LS_EQUAL -1
#endif
#ifndef LESS_THAN
#define LESS_THAN -2
#endif

namespace covise
{

class CO_MemAddAVLTree;

const int NO_OF_TREES = 256;
const int NO_OF_MEMCHUNKS = 1000;

class DMGREXPORT MemChunk
{
    friend class AddressOrderedTree;
    friend class CO_MemSizeAVLNode;
    friend class CO_MemAddAVLTree;
    friend class CO_MemSizeAVLTree;
    int seq_no;
    shmSizeType size;
    char *address;

public:
    class MemChunk *next;
    MemChunk()
        : seq_no(0)
        , size(0)
        , address(0L)
        , next(0L){};
    MemChunk(int no, void *add, shmSizeType s)
        : seq_no(no)
        , size(s)
        , address((char *)add)
        , next(0L){};
    ~MemChunk(){};
    MemChunk *split(shmSizeType s)
    {
        MemChunk *new_node = new MemChunk(seq_no, address, s);
        size = size - s;
        address = address + s;
        return new_node;
    };
    coShmPtr *getAddress()
    {
        SharedMemory *shm = get_shared_memory();
        coShmPtr *ptr = new coShmPtr(seq_no, covise::shmSizeType(address - (char *)shm->get_pointer(seq_no)));
        return ptr;
    };
    char *get_plain_address()
    {
        return address;
    };
    shmSizeType get_plain_size()
    {
        return size;
    };
    void increase_size(shmSizeType incr)
    {
        size += incr;
    };
    void print();
    void set(int no, void *add, shmSizeType s)
    {
        seq_no = no;
        size = s;
        address = (char *)add;
    };
};

AVL_EXTERN MemChunk *new_memchunk();
AVL_EXTERN MemChunk *new_memchunk(int no, void *add, shmSizeType s);
AVL_EXTERN void delete_memchunk(MemChunk *);

class DMGREXPORT CO_MemAVLNode /* structure for AVL-trees */
{
public:
    CO_MemAVLNode *left; /* pointer to left subtree */
    CO_MemAVLNode *right; /* pointer to right subtree */
    CO_MemAVLNode *up; /* pointer to father node */
    int balance; /* balance of subtrees =h(R)-h(L), normally -1..1 */
    MemChunk *data; /* data the tree is sorted by */
    CO_MemAVLNode(MemChunk *d)
    {
        data = d;
        left = 0L;
        right = 0L;
        up = 0L;
        balance = 0;
    };
    CO_MemAVLNode() // does not delete data!!
    {
        data = 0L;
        delete left;
        delete right;
    };
    void print()
    {
        if (left)
            left->print();
        //	if(data) data->print();
        if (right)
            right->print();
    };
    void remove_nod(int dispo_chunk)
    {
        if (left)
        {
            left->remove_nod(dispo_chunk);
            delete left;
        }
        if (right)
        {
            right->remove_nod(dispo_chunk);
            delete right;
        }
        if (dispo_chunk && data)
            delete_memchunk(data);
    };
};

class DMGREXPORT CO_MemAVLTree
{
    friend class CO_MemAddAVLTree;
    friend class CO_MemSizeAVLTree;
    int m_reb_active; //rebalance of the tree active - default true
public:
    CO_MemAVLTree() // standard initialization
    {
        m_reb_active = 1;
    };
    ~CO_MemAVLTree(){};

    void activate_reb(void)
    {
        m_reb_active = 1;
    };
    void deactivate_reb(void)
    {
        m_reb_active = 0;
    };
    int is_reb_active(void)
    {
        return m_reb_active;
    };

    void show_tree(CO_MemAVLNode *curr_node);
};

class DMGREXPORT CO_MemAddAVLTree : public CO_MemAVLTree
{
    CO_MemAVLNode *root;
    CO_MemAVLNode *best_node; /* after a search the found node can be found here */
public:
    CO_MemAddAVLTree()
        : root(0L){};
    ~CO_MemAddAVLTree(){};
    void rebalance_tree(CO_MemAVLNode *tree_node, int add_balance,
                        int grow_shrink);
    MemChunk *search_node(MemChunk *data, int search);
    MemChunk *search_and_remove_node(MemChunk *data, int search);
    int insert_node(MemChunk *data);
    MemChunk *remove_node(MemChunk *data);
    void empty_tree(int dispo_chunk)
    {
        if (root)
        {
            root->remove_nod(dispo_chunk);
            delete root;
            root = NULL;
        }
    };
};
#ifdef WIN32
#pragma warning (push)
#pragma warning (disable : 4311)
#pragma warning (disable : 4302)
#endif
class DMGREXPORT AddressOrderedTree
{
private:
    CO_MemAddAVLTree trees[NO_OF_TREES];

public:
    AddressOrderedTree(){};
    ~AddressOrderedTree(){};
    MemChunk *search_chunk(MemChunk *data, int search)
    {
        return trees[(long)data->address & 0xff00 >> 8].search_node(data, search);
    };
    int insert_chunk(MemChunk *data)
    {
        return trees[(long)data->address & 0xff00 >> 8].insert_node(data);
    };
    MemChunk *remove_chunk(MemChunk *data)
    {
        return trees[(long)data->address & 0xff00 >> 8].search_and_remove_node(data, EQUAL);
    };
    void empty_trees(int dispo_chunk)
    {
        for (int i = 0; i < NO_OF_TREES; i++)
            trees[i].empty_tree(dispo_chunk);
    };
};

#ifdef WIN32
#pragma warning (pop)
#endif

class DMGREXPORT CO_MemSizeAVLNode /* structure for AVL-trees */
{
public:
    CO_MemSizeAVLNode *left; /* pointer to left subtree */
    CO_MemSizeAVLNode *right; /* pointer to right subtree */
    CO_MemSizeAVLNode *up; /* pointer to father node */
    int balance; /* balance of subtrees =h(R)-h(L), normally -1..1 */
    shmSizeType size; /* data the tree is sorted by */
    int number_of_chunks;
    MemChunk *node_list;
    CO_MemSizeAVLNode(MemChunk *d)
        : left(0L)
        , right(0L)
        , up(0)
        , balance(0)
        , number_of_chunks(1)
    {
        d->next = 0L;
        node_list = d;
        size = d->size;
    };
    ~CO_MemSizeAVLNode()
    {
        while (node_list)
        {
            fprintf(stderr, "Fehler!!!\n");
            MemChunk *tmp = node_list->next;
            delete_memchunk(node_list);
            node_list = tmp;
        }
        // Uwe delete left;
        // delete right;
    }; // does not delete data!!
    void add_chunk(MemChunk *d)
    {
        d->next = node_list;
        node_list = d;
        number_of_chunks++;
    };
    MemChunk *remove_chunk()
    {
        MemChunk *tmpptr = node_list;
        node_list = node_list->next;
        number_of_chunks--;
        return tmpptr;
    };
    MemChunk *remove_chunk(MemChunk *data)
    {
        MemChunk *tmpptr = node_list;
        if (tmpptr == data)
        {
            node_list = node_list->next;
            number_of_chunks--;
            return tmpptr;
        }
        while (tmpptr->next)
        {
            if (tmpptr->next == data)
            {
                tmpptr->next = data->next;
                number_of_chunks--;
                return data;
            }
            else
                tmpptr = tmpptr->next;
        }
        return 0L;
    };
    void print()
    {
        if (left)
            left->print();
        //	if(data) data->print();
        if (right)
            right->print();
    };
    void remove_nod(void)
    {
        if (left)
        {
            left->remove_nod();
            delete left;
        }
        if (right)
        {
            right->remove_nod();
            delete right;
        }
        while (node_list)
        {
            MemChunk *tmp = node_list->next;
            delete_memchunk(node_list);
            node_list = tmp;
        }
    };
};

class DMGREXPORT CO_MemSizeAVLTree : public CO_MemAVLTree
{
    CO_MemSizeAVLNode *root;
    CO_MemSizeAVLNode *best_node; /* after a search the found node can be found here */
public:
    CO_MemSizeAVLTree()
        : root(0L){};
    ~CO_MemSizeAVLTree(){};
    void rebalance_tree(CO_MemSizeAVLNode *tree_node, int add_balance,
                        int grow_shrink);
    MemChunk *search_and_remove_node(shmSizeType size, int search);
    MemChunk *remove_node(MemChunk *data);
    int insert_node(MemChunk *data);
    void empty_tree(void)
    {
        if (root)
        {
            root->remove_nod();
            delete root;
            root = NULL;
        }
    };
};

class DMGREXPORT SizeOrderedTree
{
private:
    CO_MemSizeAVLTree tree;

public:
    SizeOrderedTree(){};
    ~SizeOrderedTree(){};
    MemChunk *get_chunk(shmSizeType size_wanted)
    {
        return tree.search_and_remove_node(size_wanted, GT_EQUAL);
    };
    MemChunk *remove_chunk(MemChunk *data)
    {
        return tree.remove_node(data);
    };
    int insert_chunk(MemChunk *data)
    {
        return tree.insert_node(data);
    };
    void empty_tree(void)
    {
        tree.empty_tree();
    };
};
}
#endif
