/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CONFIGURATION_LIST_H
#define __CONFIGURATION_LIST_H

namespace gaalet
{

typedef unsigned long long int conf_t;

template <conf_t H, typename T>
struct configuration_list
{
    static const conf_t head = H;

    typedef T tail;

    static const conf_t size = tail::size + 1;
};

struct cl_null
{
    //head in cl_null defined for cleaner utility implementations (e.g. insert_element, search_element), may result in error prone implementation
    static const conf_t head = 0;

    static const conf_t size = 0;
};

//get_element
template <conf_t index, typename list>
struct get_element
{
    static_assert(index < list::size, "get_element<index, list>: index not less than list size");

    static const conf_t value = get_element<index - 1, typename list::tail>::value;
};

template <typename list>
struct get_element<0, list>
{
    static_assert(0 < list::size, "get_element<index, list>: index not less than list size");

    static const conf_t value = list::head;
};

//insert_element
template <conf_t element, typename list, int op = (element == list::head) ? 0 : ((element < list::head) ? 1 : -1) > struct insert_element
{
    typedef configuration_list<list::head, typename insert_element<element, typename list::tail>::clist> clist;
};

template <conf_t element, int op>
struct insert_element<element, cl_null, op>
{
    typedef configuration_list<element, cl_null> clist;
};
template <conf_t element>
struct insert_element<element, cl_null, 0>
{
    typedef configuration_list<element, cl_null> clist;
};

template <conf_t element, typename list>
struct insert_element<element, list, 0>
{
    typedef list clist;
};

template <conf_t element, typename list>
struct insert_element<element, list, 1>
{
    typedef configuration_list<element, list> clist;
};

//merge lists
template <typename listlow, typename listhigh>
struct merge_lists
{
    typedef typename merge_lists<typename listlow::tail, typename insert_element<listlow::head, listhigh>::clist>::clist clist;
};

template <typename listhigh>
struct merge_lists<cl_null, listhigh>
{
    typedef listhigh clist;
};

//search element
template <conf_t element, typename list, conf_t this_index = 0, bool fit = (element == list::head)>
struct search_element
{
    static const conf_t index = search_element<element, typename list::tail, this_index + 1>::index;
};

template <conf_t element, typename list, conf_t this_index>
struct search_element<element, list, this_index, true>
{
    static const conf_t index = this_index;
};

template <conf_t element, conf_t this_index, bool fit>
struct search_element<element, cl_null, this_index, fit>
{
    static const conf_t index = this_index;
};
template <conf_t element, conf_t this_index>
struct search_element<element, cl_null, this_index, true>
{
    static const conf_t index = this_index;
};

} //end namespace gaalet

#endif
