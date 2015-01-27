/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_NEW_AVL_TREE_H
#define EC_NEW_AVL_TREE_H

//*****************************************************************//
// search for a special value in a binary tree. return value is
//  if search==-2:  the biggest node with value < data,
//  if search==+2:  the smallest node with value > data,
//  if search==-1:  the biggest node with value <= data,
//  if search==+1:  the smallest node with value >= data,
//  if search== 0:  the node with value == data,
//        or NULL, if the condition cannot be satisfied.
//*****************************************************************//

#include <covise/covise.h>

#ifdef __GNUC__
#define COVISE_GREATER_THAN 2
#define COVISE_GT_EQUAL 1
#define COVISE_EQUAL 0
#define COVISE_LS_EQUAL -1
#define COVISE_LESS_THAN -2
#else
const int COVISE_GREATER_THAN = 2;
const int COVISE_GT_EQUAL = 1;
const int COVISE_EQUAL = 0;
const int COVISE_LS_EQUAL = -1;
const int COVISE_LESS_THAN = -2;
#endif

namespace covise
{

int covise_std_compare(char *, char *); // default comparison function

template <class T>
class AVLTree;

template <class T>
class CO_AVL_Node /* structure for AVL-trees */
{
    friend class CO_AVL_Tree;

public:
    CO_AVL_Node<T> *left; /* pointer to left subtree */
    CO_AVL_Node<T> *right; /* pointer to right subtree */
    CO_AVL_Node<T> *up; /* pointer to father node */
    int balance; /* balance of subtrees =h(R)-h(L), normally -1..1 */
    T *data; /* data the tree is sorted by */
    CO_AVL_Node(T *d)
    {
        data = d;
        left = 0L;
        right = 0L;
        up = 0L;
        balance = 0;
    };
    ~CO_AVL_Node()
    {
        data = 0L;
    };

    void print()
    {
        if (left)
            left->print();
        if (data)
            data->print();
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
        if (data)
            delete data;
    }
#if defined(__hpux) || defined(_SX)
    CO_AVL_Node<T> *search_identical_node(T *d);
#else
    CO_AVL_Node<T> *search_identical_node(T *d)
    {
        CO_AVL_Node<T> *retval = 0L;

        if (d == data)
            return this;
        if (left)
        {
            retval = left->search_identical_node(d);
            if (retval)
                return retval;
        }
        if (right)
        {
            retval = right->search_identical_node(d);
        }
        return retval;
    };
#endif
};

template <class T>
class AVLTree
{
private:
    int (*compare)(T *a, T *b); // compare function
    CO_AVL_Node<T> *root;
    CO_AVL_Node<T> *best_node; /* after a search the found node can be found here */
    const char *name;
    int count; /* (number of elements in tree) */
public:
    AVLTree()
    {
        root = 0L;
        compare = (int (*)(T *, T *))covise_std_compare;
        count = 0; // standard initialization
    };
    AVLTree(int (*comp)(T *a, T *b))
    {
        root = 0L;
        compare = comp;
        name = "dummy";
        count = 0;
    };
    // initialization with special comp. function
    AVLTree(int (*comp)(T *a, T *b), const char *n)
    {
        root = 0L;
        compare = comp;
        name = n;
        count = 0;
    };
    // initialization with special comp. function
    ~AVLTree()
    {
        if (name)
            delete name;
    };

    CO_AVL_Node<T> *get_root(void)
    {
        return root;
    };

    void empty_tree(void)
    {
        if (root)
        {
            root->remove_nod();
            delete root;
            root = NULL;
        }
        count = 0;
    };
    T *search_node(T *data, int search);
    void rebalance_tree(CO_AVL_Node<T> *tree_node, int add_balance,
                        int grow_shrink);
    int insert_node(T *data);
    T *remove_node(T *data);
    T *remove_node_compare(T *data);
    CO_AVL_Node<T> *search_identical_node(T *d, CO_AVL_Node<T> *start);
    void print(char *);
    void show_tree(CO_AVL_Node<T> *curr_node);
};

template <class T>
CO_AVL_Node<T> *AVLTree<T>::search_identical_node(T *d, CO_AVL_Node<T> *start)
{
    CO_AVL_Node<T> *retval = 0L;

    if (start == 0L)
        return 0L;

    if (start->data == d)
        return start;
    if (start->left)
    {
        if (compare((T *)start->left->data, d) < 0)
        {
            retval = search_identical_node(d, start->left->right);
            if (retval)
                return retval;
        }
        else
        {
            retval = search_identical_node(d, start->left);
            if (retval)
                return retval;
        }
    }
    if (start->right)
    {
        if (compare((T *)start->right->data, d) > 0)
        {
            retval = search_identical_node(d, start->right->left);
            if (retval)
                return retval;
        }
        else
        {
            retval = search_identical_node(d, start->right);
            if (retval)
                return retval;
        }
    }
    return 0L;
};

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
 **                       -1==LESS_EQUAL, 0==COVISE_EQUAL, 1==COVISE_GT_EQUAL              **
 **               data: data to search for                                   **
 **                                                                          **
 ** Result      : insert_node inserts a node into the tree,                  **
 **               remove_node removes a node from the tree,                  **
 **               search_node searches for a node in the tree, where         **
 **                 NULL is returned if no node is found that meets the      **
 **                 search criteria; otherwise a pointer to the node.        **
 **                 for COVISE_EQUAL search, the data must match exactly,           **
 **                 for LESS_EQUAL search, the node with largest data        **
 **                 less or equal to search-data is returned, and            **
 **                 for COVISE_GT_EQUAL, the node with smallest data greater        **
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

//static int covise_n_node=0;
//static int covise_depth=0;

#if defined(__hpux) || defined(_SX)
template <class T>
CO_AVL_Node<T> *CO_AVL_Node<T>::search_identical_node(T *d)
{
    CO_AVL_Node<T> *retval = 0L;

    if (d == data)
        return this;
    if (left)
    {
        retval = left->search_identical_node(d);
        if (retval)
            return retval;
    }
    if (right)
    {
        retval = right->search_identical_node(d);
    }
    return retval;
}
#endif
template <class T>
T *AVLTree<T>::search_node(T *data, int search)
{
#ifdef DEBUG
    char tmp_str[255];

    sprintf(tmp_str, "AVLTree<T>::search_node, mode %d", search);
    print(tmp_str);
    print_comment(__LINE__, __FILE__, "searching node:");
    data->print();
#endif

    CO_AVL_Node<T> *node;
    ////////    CO_AVL_Node<T> *ptr = root;
    T *best_data;

    if (root == NULL)
        return (NULL);

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "root != NULL");
#endif
    ////////ptr = root;

    node = root; /* we start searching at the tree's root */

    best_data = 0L;
    best_node = 0L;
    /* search for "<=" or "<" */
    if (search == COVISE_LS_EQUAL || search == COVISE_LESS_THAN)
    {
        while (node)
        {
            if (compare((T *)node->data, data) <= 0)
            {
                if (best_data == 0L)
                {
                    best_data = node->data;
                    best_node = node;
                }
                if (compare((T *)node->data, best_data) >= 0)
                {
                    best_data = node->data;
                    best_node = node;
                    if ((search == COVISE_LS_EQUAL) && (compare((T *)best_data, data) == 0))
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
    else if (search == COVISE_GT_EQUAL || search == COVISE_GREATER_THAN)
    {
        while (node)
        {
            if (compare((T *)node->data, data) >= 0)
            {
                if (best_data == 0L)
                {
                    best_data = node->data;
                    best_node = node;
                }
                if (compare((T *)node->data, best_data) <= 0)
                {
                    best_data = node->data;
                    best_node = node;
                    if ((search == COVISE_GT_EQUAL) && (compare((T *)best_data, data) == 0))
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
    else if (search == COVISE_EQUAL) /* search for "==" */
    {
        while (node)
        {
            int cmp = compare((T *)node->data, data);
            if (cmp >= 0)
            {
                if (cmp == 0)
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
        printf("Sorry, wrong parameter for search\n");
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

/**********************************************************************/

/* rebalance an AVL-tree */

#define GROW 1
#define SHRINK -1

template <class T>
void AVLTree<T>::rebalance_tree(CO_AVL_Node<T> *tree_node,
                                int add_balance,
                                int grow_shrink)
{
    CO_AVL_Node<T> *node, *father, *Left, *Right, *Left_right, *Right_left;
    int add_b, bal;

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

/* insert a node into an AVL-tree, including re-balancing of the tree */

template <class T>
int AVLTree<T>::insert_node(T *data)
{
    CO_AVL_Node<T> *new_node, *node, *father;
    T *new_data;
    int add_balance;
#ifdef DEBUG
    char tmp_str[255];
#endif

    if (data == 0L)
        return 0;

//  count++;

#ifdef DEBUG
    print("AVLTree<T>::insert_node, before");
    print_comment(__LINE__, __FILE__, "inserting node:");
    data->print();
#endif

    new_node = new CO_AVL_Node<T>(data);

    if (root == 0L) /* empty tree */
    {
        root = new_node;
#ifdef DEBUG
        print("tree after:");
#endif

        return 1;
    }

    new_data = new_node->data;
    node = root; /* we start with the root */
    while (node)
    {
        father = node;
        if (compare(node->data, new_data) >= 0)
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
    if (compare(father->data, new_data) >= 0)
    {
        father->left = new_node;
        add_balance = -1;
    }
    else
    {
        father->right = new_node;
        add_balance = 1;
    }

    /*  rebalance_growing_tree( tree, father, add_balance );*/
    rebalance_tree(father, add_balance, GROW);

#ifdef DEBUG
    print("tree after:");
#endif

    return 1;
} /* end insert_node */

/**********************************************************************/

/* remove a node from an AVL-tree, including re-balancing of the tree */
/* compare objects */

template <class T>
T *AVLTree<T>::remove_node_compare(T *data)
{
    CO_AVL_Node<T> *node, *old_node, *father;
    T *ret_data;
    int add_balance = 0;
#ifdef DEBUG
    char tmp_str[255];
#endif

    if (data == NULL)
        return 0;

//  count--;

#ifdef DEBUG
    print("AVLTree<T>::remove_node_compare, before");
    data->print();
#endif

    if (search_node(data, COVISE_EQUAL))
        old_node = best_node; // side effect of search_node()!!
    else
    {
        print_comment(__LINE__, __FILE__, "node not found for removal");
        return 0L;
    }

    node = old_node;
    ret_data = node->data;
#ifdef ERR_NODE
    if (node == ERR_NODE)
    {
        printf("[node=%x, data=%x]", node, node->data);
    }
#endif
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
            // !(father->left) || in the following line inserted A.W. 15.03.96
            //      if (!(father->left) || compare(father->left->data, node->data) == 0) {
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
            //      if (compare(father->left->data, node->data) == 0) {
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
            //      if (compare(father->right->data, node->data) == 0) {
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
            //      if (compare((old_node->up)->left->data, old_node->data) == 0)
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

/**********************************************************************/

/* remove a node from an AVL-tree, including re-balancing of the tree */
/* compare addresses */

template <class T>
T *AVLTree<T>::remove_node(T *data)
{
    CO_AVL_Node<T> *node, *old_node, *father, *tmp_node;
    T *ret_data;
    int add_balance = 0;
#ifdef DEBUG
    char tmp_str[255];
#endif

    if (data == NULL)
        return 0;

    search_node(data, COVISE_EQUAL);
    if (!best_node)
    {
        printf("Serious Error, did not find a node with correct data\n");
    }

    tmp_node = best_node;

    old_node = node = search_identical_node(data, tmp_node);

    if (node == 0L)
    {
        printf("remove_node failed\n");
        search_identical_node(data, tmp_node);
        return 0L;
    }

    ret_data = node->data;

    if (node->left == NULL) /* replace node by the right subtree */
    {

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
        // untested!!
        delete old_node;
    }

    /*  rebalance_shrinking_tree( tree, father, add_balance );*/
    rebalance_tree(father, add_balance, SHRINK);

#ifdef DEBUG
    print("tree after:");
#endif

    return ret_data;

} /* end remove_node */

/**********************************************************************/

template <class T>
void AVLTree<T>::print(char *str)
{
    char tmp_str[1000];

    print_comment(__LINE__, __FILE__, "------- %s: %s -------", name, str);
    //covise_n_node = 0;
    //covise_depth = 0;
    if (root == NULL)
    {
        print_comment(__LINE__, __FILE__, "Tree empty");
        printf("Tree empty");
        return;
    }

    show_tree(root);
};

template <class T>
void AVLTree<T>::show_tree(CO_AVL_Node<T> *curr_node)
{
    int i;
    char tmp_str[1000];

    if (curr_node == curr_node->up)
    {
        print_comment(__LINE__, __FILE__, " .... recursive; ERROR!");
        printf(" .... recursive; ERROR!");
        return;
    }

    //covise_depth++;
    if (curr_node->right)
        show_tree(curr_node->right);
#if 0
   covise_n_node++;
   //    printf("\n ");
   for (i=0; i<covise_depth; i++)
      //	printf("-");
      sprintf(tmp_str, " Node %4d @@%p: bal=%2d  data=%p",
         covise_n_node, curr_node, curr_node->balance,
         (void *)curr_node->data);
   //    printf(tmp_str);
   print_comment(__LINE__, __FILE__, tmp_str);
#else
    (void)tmp_str;
#endif

    if (curr_node->left)
        show_tree(curr_node->left);
    //covise_depth--;
    return;
}
}
#endif
