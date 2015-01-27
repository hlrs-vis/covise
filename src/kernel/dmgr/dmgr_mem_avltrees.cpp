/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <limits.h>
#define AVL_EXTERN
#include <shm/covise_shm.h>
#include "dmgr_mem_avltrees.h"
#include "dmgr.h"

/*--------------------------------------------------------------------------*\ 
 **                                                                          **
 ** AVL Tree Handling                                  Version: 1.3          **
 **                                                                          **
 ** Description : search for a node in an AVL tree,                          **
 **               insertion of a node into an AVL tree, and                  **
 **               removal of a node from an AVL tree;                        **
 **               both do rebalancing of the tree                            **
 **                                                                          **
 ** Call        :                                                            **
 **               void insert_node(tree,new_node)                            **
 **                       struct NODE **tree, *new_node;                     **
 **                                                                          **
 **               void remove_node(tree,old_node)                            **
 **                       struct NODE **tree, *old_node;                     **
 **                                                                          **
 **               struct NODE *search_node(tree,data,search)                 **
 **                       struct NODE **tree;                                **
 **                       char *data;                                        **
 **                       int search;                                        **
 **                                                                          **
 ** Parameters  : tree:  pointer to tree root pointer                        **
 **               new_node: node to insert into tree                         **
 **               old_node: node to remove from tree                         **
 **               search: indicate what node to search for:                  **
 **                       -1==LESS_EQUAL, 0==EQUAL, 1==GT_EQUAL              **
 **               data: data to search for                                   **
 **                                                                          **
 ** Result      : insert_node inserts a node into the tree,                  **
 **               remove_node removes a node from the tree,                  **
 **               search_node searches for a node in the tree, where         **
 **                 NULL is returned if no node is found that meets the      **
 **                 search criteria; otherwise a pointer to the node.        **
 **                 for EQUAL search, the data must match exactly,           **
 **                 for LESS_EQUAL search, the node with largest data        **
 **                 less or equal to search-data is returned, and            **
 **                 for GT_EQUAL, the node with smallest data greater        **
 **                 or equal to the search-data.                             **
 **                                                                          **
\*--------------------------------------------------------------------------*/

/* ERR_NODE definieren, wenn bei einem speziellen Knoten im Baum Fehler */
#define ERR_NODE 0x100403AC
#undef ERR_NODE

#undef DEBUG
#undef debug
#ifndef NULL
#define NULL 0L
#endif

//static const int GROW = 1;
//static const int SHRINK = -1;

namespace covise
{

static int n_node = 0;
//static int depth=0;

static MemChunk *chunk_list = 0L;

void fill_chunk_list()
{
    MemChunk *chunk_ptr;
    int i;

    chunk_list = chunk_ptr = new MemChunk[NO_OF_MEMCHUNKS];
    for (i = 0; i < NO_OF_MEMCHUNKS - 1; i++)
    {
        chunk_ptr[i].next = &chunk_ptr[i + 1];
    }
    chunk_ptr[i].next = 0L;
}

MemChunk *new_memchunk()
{
    MemChunk *tmpptr;

    if (!chunk_list)
        fill_chunk_list();
    tmpptr = chunk_list;
    chunk_list = chunk_list->next;
    return tmpptr;
}

MemChunk *new_memchunk(int no, void *add, long s)
{
    MemChunk *tmpptr;

    if (!chunk_list)
        fill_chunk_list();
    tmpptr = chunk_list;
    tmpptr->set(no, add, s);
    chunk_list = chunk_list->next;
    tmpptr->next = 0L; // Uwe Woessner
    return tmpptr;
}

void delete_memchunk(MemChunk *chunk_ptr)
{
    chunk_ptr->next = chunk_list;
    chunk_list = chunk_ptr;
}
}
/* rebalance an AVL-tree */

using namespace covise;

#define GROW 1
#define SHRINK -1

void CO_MemAddAVLTree::rebalance_tree(CO_MemAVLNode *tree_node,
                                      int add_balance,
                                      int grow_shrink)
{
    CO_MemAVLNode *node, *father, *Left, *Right, *Left_right, *Right_left;
    int add_b, bal;

    if (!m_reb_active)
        return; //rebalance not active

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "AVLTree<T>::rebalance_tree");
#endif
#ifdef debug
    if (grow_shrink == GROW)
    {
        fprintf(stdout, "\n rebalance_growing_tree(%8x,%8x,%d):",
                this, tree_node, add_balance);
    }
    else
    {
        fprintf(stdout, "\n rebalance_shrinking_tree(%8x,%8x,%d):",
                this, tree_node, add_balance);
        if ((int *)this == (int *)0x10032c88 && (int *)tree_node == (int *)0x100b4858)
            int i = 0;
    }
#endif

    if (tree_node == NULL)
        return; /* nothing to do ???? */

    node = tree_node;
    add_b = add_balance;
    while (node)
    {
#ifdef ERR_NODE
        if (node == ERR_NODE)
        {
            printf("[node=%x, data=%x]", node, node->data);
        }
#endif
#ifdef debug
        fprintf(stdout, " (node=%x,add_b=%d)", node, add_b);
#endif
        bal = (node->balance += add_b);
        if (((bal == 0) && (grow_shrink == GROW))
                /* balance o.k. */
            || ((bal == add_b) && (grow_shrink == SHRINK)))
        {
            break;
        }
        if (((bal != add_b) && (grow_shrink == GROW))
            || ((bal != 0) && (grow_shrink == SHRINK))) /* must rotate */
        {
            Left = node->left;
            Right = node->right;
            if (add_b == -1) /* left subtree has grown */
            {
#ifdef debug
                if (Left == NULL)
                {
                    printf("Left==NULL \n");
                    print("Left==NULL");
                    exit(0);
                }
#endif
                if ((Left->balance == -1) || (Left->balance == 0))
                {
                    /* R-ROTATE */
                    /*****************************************\ 
                *         F                  F          *
                *         |                  |          *
                *        node      ==>       L          *
                *       /    \             /   \        *
                *      L      R          LL     node    *
                *     / \                       /  \    *
                *   LL   LR                   LR    R   *
                \*****************************************/
                    father = node->up;
                    if (father)
                    {
                        if (father->left == node)
                        {
                            father->left = Left;
                        }
                        else
                        {
                            father->right = Left;
                        }
                    }
                    else /* it's the root */
                    {
                        root = Left;
                    }
                    (Left->up) = (node->up);
                    (node->left) = (Left->right);
                    if (Left->right)
                    {
                        (Left->right)->up = node;
                    }
                    Left->right = node;
                    node->up = Left;

                    if (Left->balance == 0)
                    {
                        node->balance = -1; /* new balances */
                        Left->balance = 1;
                        break; /* tree is balanced now! */
                    }
                    else /* Left->balance == -1 */
                    {
                        node->balance = 0; /* new balances */
                        Left->balance = 0;
                    }

                    node = Left;
                }
                else /* Left->balance == +1 */
                {
                    /* LR-ROTATE */
                    /*********************************************************\ 
                *         F                 F                 F         *
                *         |                 |                 |         *
                *        node              node               LR        *
                *       /    \    ==>     /    \    ==>     /    \      *
                *      L      R         LR      R         L      node   *
                *     / \              /  \              / \      / \   *
                *   LL   LR           L    LRR         LL  LRL  LRR  R  *
                *       /  \         / \                                *
                *     LRL  LRR     LL  LRL                              *
                \*********************************************************/
                    Left_right = Left->right;
#ifdef debug
                    if (Left_right == NULL)
                    {
                        printf("Left_right==NULL \n");
                        print("Left_right==NULL");
                        exit(0);
                    }
#endif
                    father = node->up;
                    if (father)
                    {
                        if (father->left == node)
                        {
                            father->left = Left_right;
                        }
                        else
                        {
                            father->right = Left_right;
                        }
                    }
                    else /* root */
                    {
                        root = Left_right;
                    }

                    Left_right->up = father;
                    (Left->right) = (Left_right->left);
                    if (Left_right->left)
                    {
                        (Left_right->left)->up = Left;
                    }
                    Left_right->left = Left;
                    Left->up = Left_right;
                    (node->left) = (Left_right->right);
                    if (Left_right->right)
                    {
                        (Left_right->right)->up = node;
                    }
                    Left_right->right = node;
                    node->up = Left_right;

                    if (Left_right->balance == -1)
                        node->balance = 1;
                    else
                        node->balance = 0;
                    if (Left_right->balance == 1)
                        Left->balance = -1;
                    else
                        Left->balance = 0;
                    Left_right->balance = 0;

                    node = Left_right;
                }
            }
            else /* right subtree has grown */
            {
#ifdef debug
                if (Right == NULL)
                {
                    printf("Right==NULL \n");
                    print("Right==NULL");
                    exit(0);
                }
#endif
                if ((Right->balance == 1) || (Right->balance == 0))
                {
                    /* L-ROTATE */
                    /*****************************************\ 
                *       F                    F          *
                *       |                    |          *
                *      node      ==>         R          *
                *     /    \               /   \        *
                *    L      R          node     RR      *
                *          / \         /  \             *
                *        RL   RR      L    RL           *
                \*****************************************/
                    father = node->up;
                    if (father)
                    {
                        if (father->left == node)
                        {
                            father->left = Right;
                        }
                        else
                        {
                            father->right = Right;
                        }
                    }
                    else /* root */
                    {
                        root = Right;
                    }
                    (Right->up) = (node->up);
                    (node->right) = (Right->left);
                    if (Right->left)
                    {
                        (Right->left)->up = node;
                    }
                    Right->left = node;
                    node->up = Right;

                    if (Right->balance == 0)
                    {
                        node->balance = 1;
                        Right->balance = -1;
                        break; /* tree height didn't change. */
                    }
                    else
                    {
                        node->balance = 0;
                        Right->balance = 0;
                    }

                    node = Right;
                }
                else
                {
                    /* RL-ROTATE */
                    /*********************************************************\ 
                *     F                 F                      F        *
                *     |                 |                      |        *
                *    node              node                    RL       *
                *   /    \     ==>    /    \      ==>        /   \      *
                *  L      R          L      RL           node      R    *
                *        / \               /  \          / \      / \   *
                *      RL   RR           RLL   R        L  RLL  RLR  RR *
                *     /  \                    / \                       *
                *   RLL  RLR               RLR   RR                     *
                \*********************************************************/
                    Right_left = Right->left;
#ifdef debug
                    if (Right_left == NULL)
                    {
                        printf("Right_left==NULL \n");
                        print("Right_left==NULL");
                        exit(0);
                    }
#endif
                    father = node->up;
                    if (father)
                    {
                        if (father->left == node)
                        {
                            father->left = Right_left;
                        }
                        else
                        {
                            father->right = Right_left;
                        }
                    }
                    else /* root */
                    {
                        root = Right_left;
                    }
                    Right_left->up = father;
                    (Right->left) = (Right_left->right);
                    if (Right_left->right)
                    {
                        (Right_left->right)->up = Right;
                    }
                    Right_left->right = Right;
                    Right->up = Right_left;
                    (node->right) = (Right_left->left);
                    if (Right_left->left)
                    {
                        (Right_left->left)->up = node;
                    }
                    Right_left->left = node;
                    node->up = Right_left;

                    if (Right_left->balance == -1)
                        Right->balance = 1;
                    else
                        Right->balance = 0;
                    if (Right_left->balance == 1)
                        node->balance = -1;
                    else
                        node->balance = 0;
                    Right_left->balance = 0;

                    node = Right_left;
                }
            }

            if (grow_shrink == GROW) /* tree is balanced now */
            {
                break;
            }
        }
        else
        {
            /* no rotation here, go up in tree */
        }

        father = node->up;
        if (father)
        {
            if ((father->left) == node)
            {
                add_b = -grow_shrink; /* left subtree of father is changing */
            }
            else
            {
                add_b = grow_shrink; /* right subtree of father changes */
            }
        }
        node = father;
    }

} /* end rebalance_tree */

/**********************************************************************/

void CO_MemSizeAVLTree::rebalance_tree(CO_MemSizeAVLNode *tree_node,
                                       int add_balance,
                                       int grow_shrink)
{
    CO_MemSizeAVLNode *node, *father, *Left, *Right, *Left_right, *Right_left;
    int add_b, bal;

    if (!m_reb_active)
        return; //rebalance not active

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "AVLTree<T>::rebalance_tree");
#endif
#ifdef debug
    if (grow_shrink == GROW)
    {
        fprintf(stdout, "\n rebalance_growing_tree(%8x,%8x,%d):",
                this, tree_node, add_balance);
    }
    else
    {
        fprintf(stdout, "\n rebalance_shrinking_tree(%8x,%8x,%d):",
                this, tree_node, add_balance);
        if ((int *)this == (int *)0x10032c88 && (int *)tree_node == (int *)0x100b4858)
            int i = 0;
    }
#endif

    if (tree_node == NULL)
        return; /* nothing to do ???? */

    node = tree_node;
    add_b = add_balance;
    while (node)
    {
#ifdef ERR_NODE
        if (node == ERR_NODE)
        {
            printf("[node=%x, data=%x]", node, node->data);
        }
#endif
#ifdef debug
        fprintf(stdout, " (node=%x,add_b=%d)", node, add_b);
#endif
        bal = (node->balance += add_b);
        if (((bal == 0) && (grow_shrink == GROW))
                /* balance o.k. */
            || ((bal == add_b) && (grow_shrink == SHRINK)))
        {
            break;
        }
        if (((bal != add_b) && (grow_shrink == GROW))
            || ((bal != 0) && (grow_shrink == SHRINK))) /* must rotate */
        {
            Left = node->left;
            Right = node->right;
            if (add_b == -1) /* left subtree has grown */
            {
#ifdef debug
                if (Left == NULL)
                {
                    printf("Left==NULL \n");
                    print("Left==NULL");
                    exit(0);
                }
#endif
                if ((Left->balance == -1) || (Left->balance == 0))
                {
                    /* R-ROTATE */
                    /*****************************************\ 
                *         F                  F          *
                *         |                  |          *
                *        node      ==>       L          *
                *       /    \             /   \        *
                *      L      R          LL     node    *
                *     / \                       /  \    *
                *   LL   LR                   LR    R   *
                \*****************************************/
                    father = node->up;
                    if (father)
                    {
                        if (father->left == node)
                        {
                            father->left = Left;
                        }
                        else
                        {
                            father->right = Left;
                        }
                    }
                    else /* it's the root */
                    {
                        root = Left;
                    }
                    (Left->up) = (node->up);
                    (node->left) = (Left->right);
                    if (Left->right)
                    {
                        (Left->right)->up = node;
                    }
                    Left->right = node;
                    node->up = Left;

                    if (Left->balance == 0)
                    {
                        node->balance = -1; /* new balances */
                        Left->balance = 1;
                        break; /* tree is balanced now! */
                    }
                    else /* Left->balance == -1 */
                    {
                        node->balance = 0; /* new balances */
                        Left->balance = 0;
                    }

                    node = Left;
                }
                else /* Left->balance == +1 */
                {
                    /* LR-ROTATE */
                    /*********************************************************\ 
                *         F                 F                 F         *
                *         |                 |                 |         *
                *        node              node               LR        *
                *       /    \    ==>     /    \    ==>     /    \      *
                *      L      R         LR      R         L      node   *
                *     / \              /  \              / \      / \   *
                *   LL   LR           L    LRR         LL  LRL  LRR  R  *
                *       /  \         / \                                *
                *     LRL  LRR     LL  LRL                              *
                \*********************************************************/
                    Left_right = Left->right;
#ifdef debug
                    if (Left_right == NULL)
                    {
                        printf("Left_right==NULL \n");
                        print("Left_right==NULL");
                        exit(0);
                    }
#endif
                    father = node->up;
                    if (father)
                    {
                        if (father->left == node)
                        {
                            father->left = Left_right;
                        }
                        else
                        {
                            father->right = Left_right;
                        }
                    }
                    else /* root */
                    {
                        root = Left_right;
                    }

                    Left_right->up = father;
                    (Left->right) = (Left_right->left);
                    if (Left_right->left)
                    {
                        (Left_right->left)->up = Left;
                    }
                    Left_right->left = Left;
                    Left->up = Left_right;
                    (node->left) = (Left_right->right);
                    if (Left_right->right)
                    {
                        (Left_right->right)->up = node;
                    }
                    Left_right->right = node;
                    node->up = Left_right;

                    if (Left_right->balance == -1)
                        node->balance = 1;
                    else
                        node->balance = 0;
                    if (Left_right->balance == 1)
                        Left->balance = -1;
                    else
                        Left->balance = 0;
                    Left_right->balance = 0;

                    node = Left_right;
                }
            }
            else /* right subtree has grown */
            {
#ifdef debug
                if (Right == NULL)
                {
                    printf("Right==NULL \n");
                    print("Right==NULL");
                    exit(0);
                }
#endif
                if ((Right->balance == 1) || (Right->balance == 0))
                {
                    /* L-ROTATE */
                    /*****************************************\ 
                *       F                    F          *
                *       |                    |          *
                *      node      ==>         R          *
                *     /    \               /   \        *
                *    L      R          node     RR      *
                *          / \         /  \             *
                *        RL   RR      L    RL           *
                \*****************************************/
                    father = node->up;
                    if (father)
                    {
                        if (father->left == node)
                        {
                            father->left = Right;
                        }
                        else
                        {
                            father->right = Right;
                        }
                    }
                    else /* root */
                    {
                        root = Right;
                    }
                    (Right->up) = (node->up);
                    (node->right) = (Right->left);
                    if (Right->left)
                    {
                        (Right->left)->up = node;
                    }
                    Right->left = node;
                    node->up = Right;

                    if (Right->balance == 0)
                    {
                        node->balance = 1;
                        Right->balance = -1;
                        break; /* tree height didn't change. */
                    }
                    else
                    {
                        node->balance = 0;
                        Right->balance = 0;
                    }

                    node = Right;
                }
                else
                {
                    /* RL-ROTATE */
                    /*********************************************************\ 
                *     F                 F                      F        *
                *     |                 |                      |        *
                *    node              node                    RL       *
                *   /    \     ==>    /    \      ==>        /   \      *
                *  L      R          L      RL           node      R    *
                *        / \               /  \          / \      / \   *
                *      RL   RR           RLL   R        L  RLL  RLR  RR *
                *     /  \                    / \                       *
                *   RLL  RLR               RLR   RR                     *
                \*********************************************************/
                    Right_left = Right->left;
#ifdef debug
                    if (Right_left == NULL)
                    {
                        printf("Right_left==NULL \n");
                        print("Right_left==NULL");
                        exit(0);
                    }
#endif
                    father = node->up;
                    if (father)
                    {
                        if (father->left == node)
                        {
                            father->left = Right_left;
                        }
                        else
                        {
                            father->right = Right_left;
                        }
                    }
                    else /* root */
                    {
                        root = Right_left;
                    }
                    Right_left->up = father;
                    (Right->left) = (Right_left->right);
                    if (Right_left->right)
                    {
                        (Right_left->right)->up = Right;
                    }
                    Right_left->right = Right;
                    Right->up = Right_left;
                    (node->right) = (Right_left->left);
                    if (Right_left->left)
                    {
                        (Right_left->left)->up = node;
                    }
                    Right_left->left = node;
                    node->up = Right_left;

                    if (Right_left->balance == -1)
                        Right->balance = 1;
                    else
                        Right->balance = 0;
                    if (Right_left->balance == 1)
                        node->balance = -1;
                    else
                        node->balance = 0;
                    Right_left->balance = 0;

                    node = Right_left;
                }
            }

            if (grow_shrink == GROW) /* tree is balanced now */
            {
                break;
            }
        }
        else
        {
            /* no rotation here, go up in tree */
        }

        father = node->up;
        if (father)
        {
            if ((father->left) == node)
            {
                add_b = -grow_shrink; /* left subtree of father is changing */
            }
            else
            {
                add_b = grow_shrink; /* right subtree of father changes */
            }
        }
        node = father;
    }

} /* end rebalance_tree */

/**********************************************************************/

MemChunk *CO_MemAddAVLTree::search_node(MemChunk *data, int search)
{
#ifdef DEBUG
    char tmp_str[255];

    sprintf(tmp_str, "AVLTree<T>::search_node, mode %d", search);
    print(tmp_str);
    print_comment(__LINE__, __FILE__, "searching node:");
    data->print();
#endif

    CO_MemAVLNode *node;
    //CO_MemAVLNode *ptr = root;
    MemChunk *best_data;

    if (root == NULL)
        return (NULL);

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "root != NULL");
#endif
    // ptr = root;

    node = root; /* we start searching at the tree's root */

    best_data = 0L;
    best_node = 0L;
    if (search == LS_EQUAL || search == LESS_THAN) /* search for "<=" or "<" */
    {
        while (node)
        {
            if (node->data->address <= data->address)
            {
                if (best_data == 0L)
                {
                    best_data = node->data;
                    best_node = node;
                }
                if (node->data->address >= best_data->address)
                {
                    best_data = node->data;
                    best_node = node;
                    if ((search == LS_EQUAL) && (best_data->address == data->address))
                        break;
                }
                node = node->right;
            }
            else
            {
                node = node->left;
            }
        }
    }
    /* search for ">=" or ">" */
    else if (search == GT_EQUAL || search == GREATER_THAN)
    {
        while (node)
        {
            if (node->data->address >= data->address)
            {
                if (best_data == 0L)
                {
                    best_data = node->data;
                    best_node = node;
                }
                if (node->data->address <= best_data->address)
                {
                    best_data = node->data;
                    best_node = node;
                    if ((search == GT_EQUAL) && (best_data->address == data->address))
                        break;
                }
                node = node->left;
            }
            else
            {
                node = node->right;
            }
        }
    }
    else if (search == EQUAL) /* search for "==" */
    {
        while (node)
        {
            if (node->data->address >= data->address)
            {
                if (node->data->address == data->address)
                {
                    best_data = node->data;
                    best_node = node;
                    break;
                }
                node = node->left;
            }
            else
            {
                node = node->right;
            }
        }
    }
    else /* wrong parameter for search */
    {
        fprintf(stderr, "Sorry, wrong parameter for search\n");
        return 0L;
    }
/* now, best_data contains the return value. */
/* and best_node the corresponding node */
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "found node:");
    if (best_data)
        best_data->print();
    else
        print_comment(__LINE__, __FILE__, "found nothing");
#endif
    return (best_data);
} /* end search_node */

MemChunk *CO_MemAddAVLTree::search_and_remove_node(MemChunk *data, int search)
{
#ifdef DEBUG
    char tmp_str[255];

    sprintf(tmp_str, "AVLTree<T>::search_node, mode %d", search);
    print(tmp_str);
    print_comment(__LINE__, __FILE__, "searching node:");
    data->print();
#endif

    CO_MemAVLNode *node, *old_node, *father;
    //CO_MemAVLNode *ptr = root;
    MemChunk *best_data;
    //MemChunk *retchunk;
    int add_balance = 0;

    if (root == NULL)
        return (NULL);

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "root != NULL");
#endif
    //ptr = root;

    node = root; /* we start searching at the tree's root */

    best_data = 0L;
    best_node = 0L;
    if (search == EQUAL) /* search for "==" */
    {
        while (node)
        {
            if (node->data->address >= data->address)
            {
                if (node->data->address == data->address)
                {
                    best_data = node->data;
                    best_node = node;
                    break;
                }
                node = node->left;
            }
            else
            {
                node = node->right;
            }
        }
    }
    /* search for "<=" or "<" */
    else if (search == LS_EQUAL || search == LESS_THAN)
    {
        while (node)
        {
            if (node->data->address <= data->address)
            {
                if (best_data == 0L)
                {
                    best_data = node->data;
                    best_node = node;
                }
                if (node->data->address >= best_data->address)
                {
                    best_data = node->data;
                    best_node = node;
                    if ((search == LS_EQUAL) && (best_data->address == data->address))
                        break;
                }
                node = node->right;
            }
            else
            {
                node = node->left;
            }
        }
    }
    /* search for ">=" or ">" */
    else if (search == GT_EQUAL || search == GREATER_THAN)
    {
        while (node)
        {
            if (node->data->address >= data->address)
            {
                if (best_data == 0L)
                {
                    best_data = node->data;
                    best_node = node;
                }
                if (node->data->address <= best_data->address)
                {
                    best_data = node->data;
                    best_node = node;
                    if ((search == GT_EQUAL) && (best_data->address == data->address))
                        break;
                }
                node = node->left;
            }
            else
            {
                node = node->right;
            }
        }
    }
    else /* wrong parameter for search */
    {
        fprintf(stderr, "Sorry, wrong parameter for search\n");
        return 0L;
    }

/* now, best_data contains the return value. */
/* and best_node the corresponding node */

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "found node:");
    if (best_data)
        best_data->print();
    else
        print_comment(__LINE__, __FILE__, "found nothing");
#endif

    if (best_node == 0L)
    {
        return 0L;
    }

    node = old_node = best_node;

    if (node->left == NULL) /* replace node by the right subtree */
    {
#ifdef debug
        fprintf(stdout, " case 1");
#endif
        /*   F              *\ 
       *   |          F   *
       *   N    ==>   |   *
       *    \         R   *
      \*     R            */
        father = node->up;
        if (father)
        {
            if (father->left == node)
            {
#ifdef debug
                fprintf(stdout, ".1");
#endif
                (father->left) = (node->right);
                add_balance = 1;
            }
            else
            {
#ifdef debug
                fprintf(stdout, ".2");
#endif
                (father->right) = (node->right);
                add_balance = -1;
            }
        }
        else /* root */
        {
#ifdef debug
            fprintf(stdout, ".3");
#endif
            root = (node->right);
        }
        if (node->right)
            (node->right)->up = father;
        // untested!!!
        delete node;
    }
    else if (node->right == 0L) /* replace node by left subtree */
    {
#ifdef debug
        fprintf(stdout, " case 2");
#endif
        /*    F             *\ 
       *    |         F   *
       *    N   ==>   |   *
       *   /          L   *
       \*  L               */
        father = node->up;
        if (father)
        {
            if (father->left == node)
            {
                (father->left) = (node->left);
                add_balance = 1;
            }
            else
            {
                (father->right) = (node->left);
                add_balance = -1;
            }
        }
        else /* root */
        {
            root = (node->left);
        }
        if (node->left)
            (node->left)->up = father;
        // untested!!!
        delete node;
    }
    else /* must search downward for a node to remove */
    {
        /*      F             F    *\ 
       *      |             |    *
       *      N             Y    *
       *     / \    ==>    / \   *
       *    V   W         V   W  *
       *   / \           / \     *
       *  X   Y         X   Z    *
       *     /                   *
       \*    Z                    */
        if (node->balance < 1) /* left subtree of node is higher than right */
        {
#ifdef debug
            fprintf(stdout, " case 3");
#endif
            node = (node->left); /* search for largest node in left subtree   */
            while (node->right)
            {
                node = (node->right);
#ifdef ERR_NODE
                if (node == ERR_NODE)
                {
                    printf("[node=%x, data=%x]", node, node->data);
                }
#endif
            }

            father = (node->up); /* node durch linken subtree ersetzen */

#ifdef debug
            if (father == NULL)
            {
                printf("\n  father==NULL; tree=%8x; node=%8x : \n", root, node);
                print("father==NULL");
                exit(0);
            }
#endif

            if (father->right == node)
            {
                (father->right) = (node->left);
                add_balance = -1;
            }
            else
            {
                (father->left) = (node->left);
                add_balance = 1;
            }
            if (node->left)
                (node->left)->up = father;
        }
        else /* right subtree of node is higher than left */
        {
#ifdef debug
            fprintf(stdout, " case 4");
#endif
            node = (node->right); /* search for smallest node in right subtree */
            while (node->left)
            {
                node = (node->left);
            }

            father = (node->up); /* node durch rechten subtree ersetzen */
            if (father->right == node)
            {
                (father->right) = (node->right);
                add_balance = -1;
            }
            else
            {
                (father->left) = (node->right);
                add_balance = 1;
            }
            if (node->right)
                (node->right)->up = father;
        }

        /* old_node durch node ersetzen */
        if (old_node->up)
        {
            if ((old_node->up)->left == old_node)
                (old_node->up)->left = node;
            else
                (old_node->up)->right = node;
            (node->up) = (old_node->up);
        }
        else /* root */
        {
            root = node;
            node->up = NULL;
        }
        node->left = old_node->left;
        if (node->left)
            (node->left)->up = node;
        node->right = old_node->right;
        if (node->right)
            (node->right)->up = node;
        (node->balance) = (old_node->balance);
        if (father == old_node)
            father = node;
        // untested!!!
        delete old_node;
    }

    /*  rebalance_shrinking_tree( tree, father, add_balance );*/

    rebalance_tree(father, add_balance, SHRINK);

    return (best_data);

} /* end search_node */

int CO_MemAddAVLTree::insert_node(MemChunk *data)
{
    CO_MemAVLNode *new_node, *node, *father;
    MemChunk *new_data;
    int add_balance = 0;
#ifdef DEBUG
    char tmp_str[255];
#endif

    if (data == 0L)
        return 0;

#ifdef DEBUG
    print("AVLTree<T>::insert_node, before");
    print_comment(__LINE__, __FILE__, "inserting node:");
    data->print();
#endif

    new_node = new CO_MemAVLNode(data);

    if (root == 0L) /* empty tree */
    {
        root = new_node;
        return 1;
    }

    new_data = new_node->data;
    node = root; /* we start with the root */
    while (node)
    {
        father = node;
        if (node->data->address >= new_data->address)
        {
            node = node->left;
        }
        else
        {
            node = node->right;
        }
    }

    /* now, the new node can be inserted as a son of father */

    new_node->up = father;
    if (father->data->address >= new_data->address)
    {
        father->left = new_node;
        add_balance = -1;
    }
    else
    {
        father->right = new_node;
        add_balance = 1;
    }

    rebalance_tree(father, add_balance, GROW);

    return 1;
} /* end insert_node */

void CO_MemAVLTree::show_tree(CO_MemAVLNode *curr_node)
{
    char tmp_str[1000];

    if (curr_node == curr_node->up)
    {
        print_comment(__LINE__, __FILE__, " .... recursive; ERROR!");
        printf(" .... recursive; ERROR!");
        return;
    }

    if (curr_node->right)
        show_tree(curr_node->right);
    //    printf("\n ");
    //    for (i=0; i<depth; i++)
    //	printf("-");
    sprintf(tmp_str, " Node %4d @@%p: bal=%2d  data=%p",
            n_node, curr_node, curr_node->balance,
            (void *)curr_node->data);
    //    printf(tmp_str);
    print_comment(__LINE__, __FILE__, " Node %4d @@%p: bal=%2d  data=%p",
                  n_node, curr_node, curr_node->balance,
                  (void *)curr_node->data);
    if (curr_node->left)
        show_tree(curr_node->left);
    //    depth--;
    return;
}

MemChunk *CO_MemAddAVLTree::remove_node(MemChunk *data)
{
    CO_MemAVLNode *node, *old_node, *father;
    MemChunk *ret_data;
    int add_balance = 0;
#ifdef DEBUG
    char tmp_str[255];
#endif

    if (data == NULL)
        return 0;
    if (search_node(data, EQUAL))
        old_node = best_node; // side effect of search_node()!!
    else
    {
        print_comment(__LINE__, __FILE__, "node not found for removal");
        return 0L;
    }

    node = old_node;
    ret_data = node->data;
    if (node->left == NULL) /* replace node by the right subtree */
    {
#ifdef debug
        fprintf(stdout, " case 1");
#endif
        /*   F              *\ 
       *   |          F   *
       *   N    ==>   |   *
       *    \         R   *
      \*     R            */
        father = node->up;
        if (father)
        {
            if (father->left == node)
            {
#ifdef debug
                fprintf(stdout, ".1");
#endif
                (father->left) = (node->right);
                add_balance = 1;
            }
            else
            {
#ifdef debug
                fprintf(stdout, ".2");
#endif
                (father->right) = (node->right);
                add_balance = -1;
            }
        }
        else /* root */
        {
#ifdef debug
            fprintf(stdout, ".3");
#endif
            root = (node->right);
        }
        if (node->right)
            (node->right)->up = father;
        // untested!!!
        delete node;
    }
    else if (node->right == 0L) /* replace node by left subtree */
    {
#ifdef debug
        fprintf(stdout, " case 2");
#endif
        /*    F             *\ 
       *    |         F   *
       *    N   ==>   |   *
       *   /          L   *
       \*  L               */
        father = node->up;
        if (father)
        {
            if (father->left == node)
            {
                (father->left) = (node->left);
                add_balance = 1;
            }
            else
            {
                (father->right) = (node->left);
                add_balance = -1;
            }
        }
        else /* root */
        {
            root = (node->left);
        }
        if (node->left)
            (node->left)->up = father;
        // untested!!!
        delete node;
    }
    else /* must search downward for a node to remove */
    {
        /*      F             F    *\ 
       *      |             |    *
       *      N             Y    *
       *     / \    ==>    / \   *
       *    V   W         V   W  *
       *   / \           / \     *
       *  X   Y         X   Z    *
       *     /                   *
       \*    Z                    */
        if (node->balance < 1) /* left subtree of node is higher than right */
        {
#ifdef debug
            fprintf(stdout, " case 3");
#endif
            node = (node->left); /* search for largest node in left subtree   */
            while (node->right)
            {
                node = (node->right);
#ifdef ERR_NODE
                if (node == ERR_NODE)
                {
                    printf("[node=%x, data=%x]", node, node->data);
                }
#endif
            }

            father = (node->up); /* node durch linken subtree ersetzen */

#ifdef debug
            if (father == NULL)
            {
                printf("\n  father==NULL; tree=%8x; node=%8x : \n", root, node);
                print("father==NULL");
                exit(0);
            }
#endif

            if (father->right == node)
            {
                (father->right) = (node->left);
                add_balance = -1;
            }
            else
            {
                (father->left) = (node->left);
                add_balance = 1;
            }
            if (node->left)
                (node->left)->up = father;
        }
        else /* right subtree of node is higher than left */
        {
#ifdef debug
            fprintf(stdout, " case 4");
#endif
            node = (node->right); /* search for smallest node in right subtree */
            while (node->left)
            {
                node = (node->left);
            }

            father = (node->up); /* node durch rechten subtree ersetzen */
            if (father->right == node)
            {
                (father->right) = (node->right);
                add_balance = -1;
            }
            else
            {
                (father->left) = (node->right);
                add_balance = 1;
            }
            if (node->right)
                (node->right)->up = father;
        }

        /* old_node durch node ersetzen */
        if (old_node->up)
        {
            if ((old_node->up)->left == old_node)
                (old_node->up)->left = node;
            else
                (old_node->up)->right = node;
            (node->up) = (old_node->up);
        }
        else /* root */
        {
            root = node;
            node->up = NULL;
        }
        node->left = old_node->left;
        if (node->left)
            (node->left)->up = node;
        node->right = old_node->right;
        if (node->right)
            (node->right)->up = node;
        (node->balance) = (old_node->balance);
        if (father == old_node)
            father = node;
        // untested!!!
        delete old_node;
    }

    /*  rebalance_shrinking_tree( tree, father, add_balance );*/
    rebalance_tree(father, add_balance, SHRINK);

#ifdef DEBUG
    print("tree after:");
#endif

    return ret_data;

} /* end remove_node */

/* insert a node into an AVL-tree, including re-balancing of the tree */

int CO_MemSizeAVLTree::insert_node(MemChunk *data)
{
    CO_MemSizeAVLNode *new_node, *node, *father;
    int add_balance = 0;
    int new_size;

    if (data == 0L)
        return 0;

    if (root == 0L) /* empty tree */
    {
        root = new CO_MemSizeAVLNode(data);
        return 1;
    }

    new_size = data->size;
    node = root; /* we start with the root */
    while (node)
    {
        father = node;
        if (node->size > new_size)
            node = node->left;
        else if (node->size < new_size)
            node = node->right;
        else
        {
            father->add_chunk(data);
            return 1;
        }
    }

    new_node = new CO_MemSizeAVLNode(data);

    /* now, the new node can be inserted as a son of father */

    new_node->up = father;
    if (father->size >= new_size)
    {
        father->left = new_node;
        add_balance = -1;
    }
    else
    {
        father->right = new_node;
        add_balance = 1;
    }

    rebalance_tree(father, add_balance, GROW);

    return 1;
} /* end insert_node */

MemChunk *CO_MemSizeAVLTree::search_and_remove_node(int chunk_size, int search)
{
#ifdef DEBUG
    char tmp_str[255];

    sprintf(tmp_str, "AVLTree<T>::search_node, mode %d", search);
    print(tmp_str);
    print_comment(__LINE__, __FILE__, "searching node:");
    data->print();
#endif

    CO_MemSizeAVLNode *node, *old_node, *father;
    //CO_MemSizeAVLNode *ptr = root;
    MemChunk *retchunk;
    int best_size, add_balance = 0;

    if (root == NULL)
        return (NULL);

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "root != NULL");
#endif
    //ptr = root;

    node = root; /* we start searching at the tree's root */

    best_size = 0;
    best_node = 0L;
    /* search for ">=" or ">" */
    if (search == GT_EQUAL || search == GREATER_THAN)
    {
        while (node)
        {
            if (node->size >= chunk_size)
            {
                if (best_size == 0)
                {
                    best_size = node->size;
                    best_node = node;
                }
                if (node->size <= best_size)
                {
                    best_size = node->size;
                    best_node = node;
                    if ((search == GT_EQUAL) && (best_size == chunk_size))
                        break;
                }
                node = node->left;
            }
            else
            {
                node = node->right;
            }
        }
    }
    /* search for "<=" or "<" */
    else if (search == LS_EQUAL || search == LESS_THAN)
    {
        while (node)
        {
            if (node->size <= chunk_size)
            {
                if (best_size == 0)
                {
                    best_size = node->size;
                    best_node = node;
                }
                if (node->size >= best_size)
                {
                    best_size = node->size;
                    best_node = node;
                    if ((search == LS_EQUAL) && (best_size == chunk_size))
                        break;
                }
                node = node->right;
            }
            else
            {
                node = node->left;
            }
        }
    }
    else if (search == EQUAL) /* search for "==" */
    {
        while (node)
        {
            if (node->size >= chunk_size)
            {
                if (node->size == chunk_size)
                {
                    best_size = node->size;
                    best_node = node;
                    break;
                }
                node = node->left;
            }
            else
            {
                node = node->right;
            }
        }
    }
    else /* wrong parameter for search */
    {
        fprintf(stderr, "Sorry, wrong parameter for search\n");
        return 0L;
    }

    /* now, best_data contains the return value. */
    /* and best_node the corresponding node */

    if (best_node)
    {
        retchunk = best_node->remove_chunk();
        if (best_node->number_of_chunks > 0)
            return retchunk;
    }
    else
    {
        return 0L;
    }

    node = old_node = best_node;

    if (node->left == NULL) /* replace node by the right subtree */
    {
#ifdef debug
        fprintf(stdout, " case 1");
#endif
        /*   F              *\ 
       *   |          F   *
       *   N    ==>   |   *
       *    \         R   *
      \*     R            */
        father = node->up;
        if (father)
        {
            if (father->left == node)
            {
#ifdef debug
                fprintf(stdout, ".1");
#endif
                (father->left) = (node->right);
                add_balance = 1;
            }
            else
            {
#ifdef debug
                fprintf(stdout, ".2");
#endif
                (father->right) = (node->right);
                add_balance = -1;
            }
        }
        else /* root */
        {
#ifdef debug
            fprintf(stdout, ".3");
#endif
            root = (node->right);
        }
        if (node->right)
            (node->right)->up = father;
        // untested!!!
        delete node;
    }
    else if (node->right == 0L) /* replace node by left subtree */
    {
#ifdef debug
        fprintf(stdout, " case 2");
#endif
        /*    F             *\ 
       *    |         F   *
       *    N   ==>   |   *
       *   /          L   *
       \*  L               */
        father = node->up;
        if (father)
        {
            if (father->left == node)
            {
                (father->left) = (node->left);
                add_balance = 1;
            }
            else
            {
                (father->right) = (node->left);
                add_balance = -1;
            }
        }
        else /* root */
        {
            root = (node->left);
        }
        if (node->left)
            (node->left)->up = father;
        // untested!!!
        delete node;
    }
    else /* must search downward for a node to remove */
    {
        /*      F             F    *\ 
       *      |             |    *
       *      N             Y    *
       *     / \    ==>    / \   *
       *    V   W         V   W  *
       *   / \           / \     *
       *  X   Y         X   Z    *
       *     /                   *
       \*    Z                    */
        if (node->balance < 1) /* left subtree of node is higher than right */
        {
#ifdef debug
            fprintf(stdout, " case 3");
#endif
            node = (node->left); /* search for largest node in left subtree   */
            while (node->right)
            {
                node = (node->right);
#ifdef ERR_NODE
                if (node == ERR_NODE)
                {
                    printf("[node=%x, data=%x]", node, node->data);
                }
#endif
            }

            father = (node->up); /* node durch linken subtree ersetzen */

#ifdef debug
            if (father == NULL)
            {
                printf("\n  father==NULL; tree=%8x; node=%8x : \n", root, node);
                print("father==NULL");
                exit(0);
            }
#endif

            if (father->right == node)
            {
                (father->right) = (node->left);
                add_balance = -1;
            }
            else
            {
                (father->left) = (node->left);
                add_balance = 1;
            }
            if (node->left)
                (node->left)->up = father;
        }
        else /* right subtree of node is higher than left */
        {
#ifdef debug
            fprintf(stdout, " case 4");
#endif
            node = (node->right); /* search for smallest node in right subtree */
            while (node->left)
            {
                node = (node->left);
            }

            father = (node->up); /* node durch rechten subtree ersetzen */
            if (father->right == node)
            {
                (father->right) = (node->right);
                add_balance = -1;
            }
            else
            {
                (father->left) = (node->right);
                add_balance = 1;
            }
            if (node->right)
                (node->right)->up = father;
        }

        /* old_node durch node ersetzen */
        if (old_node->up)
        {
            if ((old_node->up)->left == old_node)
                (old_node->up)->left = node;
            else
                (old_node->up)->right = node;
            (node->up) = (old_node->up);
        }
        else /* root */
        {
            root = node;
            node->up = NULL;
        }
        node->left = old_node->left;
        if (node->left)
            (node->left)->up = node;
        node->right = old_node->right;
        if (node->right)
            (node->right)->up = node;
        (node->balance) = (old_node->balance);
        if (father == old_node)
            father = node;
        // untested!!!
        delete old_node;
    }

    rebalance_tree(father, add_balance, SHRINK);

    return (retchunk);

} /* end search_and_remove_node */

MemChunk *CO_MemSizeAVLTree::remove_node(MemChunk *data)
{
#ifdef DEBUG
    char tmp_str[255];

    sprintf(tmp_str, "AVLTree<T>::search_node, mode %d", search);
    print(tmp_str);
    print_comment(__LINE__, __FILE__, "searching node:");
    data->print();
#endif

    CO_MemSizeAVLNode *node, *old_node, *father;
    //CO_MemSizeAVLNode *ptr = root;
    MemChunk *retchunk;
    int chunk_size, add_balance = 0; // best_size,

    if (root == NULL)
        return (NULL);

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "root != NULL");
#endif
    //ptr = root;

    node = root; /* we start searching at the tree's root */

    //best_size = 0;
    chunk_size = data->size;
    best_node = 0L;

    while (node)
    {
        if (node->size >= chunk_size)
        {
            if (node->size == chunk_size)
            {
                //best_size = node->size;
                best_node = node;
                break;
            }
            node = node->left;
        }
        else
        {
            node = node->right;
        }
    }

    /* now, best_data contains the return value. */
    /* and best_node the corresponding node */

    if (best_node)
    {
        retchunk = best_node->remove_chunk(data);
        if (best_node->number_of_chunks > 0)
            return retchunk;
    }
    else
    {
        return 0L;
    }

    node = old_node = best_node;

    if (node->left == NULL) /* replace node by the right subtree */
    {
#ifdef debug
        fprintf(stdout, " case 1");
#endif
        /*   F              *\ 
       *   |          F   *
       *   N    ==>   |   *
       *    \         R   *
      \*     R            */
        father = node->up;
        if (father)
        {
            if (father->left == node)
            {
#ifdef debug
                fprintf(stdout, ".1");
#endif
                (father->left) = (node->right);
                add_balance = 1;
            }
            else
            {
#ifdef debug
                fprintf(stdout, ".2");
#endif
                (father->right) = (node->right);
                add_balance = -1;
            }
        }
        else /* root */
        {
#ifdef debug
            fprintf(stdout, ".3");
#endif
            root = (node->right);
        }
        if (node->right)
            (node->right)->up = father;
        // untested!!!
        delete node;
    }
    else if (node->right == 0L) /* replace node by left subtree */
    {
#ifdef debug
        fprintf(stdout, " case 2");
#endif
        /*    F             *\ 
       *    |         F   *
       *    N   ==>   |   *
       *   /          L   *
       \*  L               */
        father = node->up;
        if (father)
        {
            if (father->left == node)
            {
                (father->left) = (node->left);
                add_balance = 1;
            }
            else
            {
                (father->right) = (node->left);
                add_balance = -1;
            }
        }
        else /* root */
        {
            root = (node->left);
        }
        if (node->left)
            (node->left)->up = father;
        // untested!!!
        delete node;
    }
    else /* must search downward for a node to remove */
    {
        /*      F             F    *\ 
       *      |             |    *
       *      N             Y    *
       *     / \    ==>    / \   *
       *    V   W         V   W  *
       *   / \           / \     *
       *  X   Y         X   Z    *
       *     /                   *
       \*    Z                    */
        if (node->balance < 1) /* left subtree of node is higher than right */
        {
#ifdef debug
            fprintf(stdout, " case 3");
#endif
            node = (node->left); /* search for largest node in left subtree   */
            while (node->right)
            {
                node = (node->right);
#ifdef ERR_NODE
                if (node == ERR_NODE)
                {
                    printf("[node=%x, data=%x]", node, node->data);
                }
#endif
            }

            father = (node->up); /* node durch linken subtree ersetzen */

#ifdef debug
            if (father == NULL)
            {
                printf("\n  father==NULL; tree=%8x; node=%8x : \n", root, node);
                print("father==NULL");
                exit(0);
            }
#endif

            if (father->right == node)
            {
                (father->right) = (node->left);
                add_balance = -1;
            }
            else
            {
                (father->left) = (node->left);
                add_balance = 1;
            }
            if (node->left)
                (node->left)->up = father;
        }
        else /* right subtree of node is higher than left */
        {
#ifdef debug
            fprintf(stdout, " case 4");
#endif
            node = (node->right); /* search for smallest node in right subtree */
            while (node->left)
            {
                node = (node->left);
            }

            father = (node->up); /* node durch rechten subtree ersetzen */
            if (father->right == node)
            {
                (father->right) = (node->right);
                add_balance = -1;
            }
            else
            {
                (father->left) = (node->right);
                add_balance = 1;
            }
            if (node->right)
                (node->right)->up = father;
        }

        /* old_node durch node ersetzen */
        if (old_node->up)
        {
            if ((old_node->up)->left == old_node)
                (old_node->up)->left = node;
            else
                (old_node->up)->right = node;
            (node->up) = (old_node->up);
        }
        else /* root */
        {
            root = node;
            node->up = NULL;
        }
        node->left = old_node->left;
        if (node->left)
            (node->left)->up = node;
        node->right = old_node->right;
        if (node->right)
            (node->right)->up = node;
        (node->balance) = (old_node->balance);
        if (father == old_node)
            father = node;
        // untested!!!
        delete old_node;
    }

    rebalance_tree(father, add_balance, SHRINK);

    return (retchunk);

} /* end remove_node */
