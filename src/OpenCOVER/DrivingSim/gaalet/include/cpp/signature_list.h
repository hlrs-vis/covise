/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __SIGNATURE_LIST_H
#define __SIGNATURE_LIST_H

namespace gaalet
{

namespace sl
{

    typedef unsigned int signature_t;
    typedef int product_signature_t;

    template <signature_t H, int GP, typename T>
    struct signature_list
    {
        static const signature_t head = H;

        static const product_signature_t geometric_product = GP;

        typedef T tail;

        static const signature_t size = tail::size + 1;
    };

    struct sl_null
    {
        //head in sl_null defined for cleaner utility implementations (e.g. insert_element, search_element), may result in error prone implementation
        static const signature_t head = 0;

        //default signature for geometric product of basis vectors
        static const product_signature_t geometric_product = 1;

        static const signature_t size = 0;
    };

    /*//get_element
template<signature_t index, typename list>
struct get_element
{
   static_assert(index < list::size, "get_element<index, list>: index not less than list size");

   typedef typename get_element<index - 1, typename list::tail>::slist slist;
};

template<typename list>
struct get_element<0, list>
{
   static_assert(0 < list::size, "get_element<index, list>: index not less than list size");

   typedef list slist;
};

//insert_element
template<signature_t element, product_signature_t GP, product_signature_t IP, product_signature_t OP,
         typename list, int op = (element==list::head) ? 0 : ((element<list::head) ? 1 : -1)>
struct insert_element
{
   typedef signature_list<list::head, list::geometric_product, list::inner_product, list::outer_product,
                           typename insert_element<element, GP, IP, OP, typename list::tail>::slist> slist;
};

template<signature_t element, product_signature_t GP, product_signature_t IP, product_signature_t OP, int op>
struct insert_element<element, GP, IP, OP, sl_null, op>
{
   typedef signature_list<element, GP, IP, OP, sl_null> slist;
};
template<signature_t element product_signature_t GP, product_signature_t IP, product_signature_t OP,>
struct insert_element<element, GP, IP, OP, sl_null, 0>
{
   typedef signature_list<element, GP, IP, OP, sl_null> slist;
};

template<signature_t element, product_signature_t GP, product_signature_t IP, product_signature_t OP, typename list>
struct insert_element<element, GP, IP, OP, list, 0>
{
   typedef list slist;
};

template<signature_t element, product_signature_t GP, product_signature_t IP, product_signature_t OP, typename list>
struct insert_element<element, GP, IP, OP, list, 1>
{
   typedef signature_list<element, GP, IP, OP, list> slist;
};


//merge lists
template<typename listlow, typename listhigh>
struct merge_lists {
   typedef typename merge_lists<typename listlow::tail, typename insert_element<listlow::head, listhigh>::slist>::slist slist;
};

template<typename listhigh>
struct merge_lists<sl_null, listhigh> {
   typedef listhigh slist;
};*/

    //search element
    template <signature_t element, typename list, bool fit = (element == list::head)>
    struct search_element
    {
        typedef typename search_element<element, typename list::tail>::slist slist;
    };

    template <signature_t element, typename list>
    struct search_element<element, list, true>
    {
        typedef list slist;
    };

    template <signature_t element, bool fit>
    struct search_element<element, sl_null, fit>
    {
        typedef sl_null slist;
    };

} //end namespace sl
} //end namespace gaalet
#endif
