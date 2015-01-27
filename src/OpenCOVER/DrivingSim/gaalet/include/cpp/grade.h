/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_GRADE_H
#define __GAALET_GRADE_H

#include "utility.h"
#include "configuration_list.h"

namespace gaalet
{

template <typename list, conf_t grade, bool pass = (BitCount<list::head>::value == grade)>
struct filter_clist_for_grade
{
    typedef configuration_list<list::head, typename filter_clist_for_grade<typename list::tail, grade>::clist> clist;
};
template <typename list, conf_t grade>
struct filter_clist_for_grade<list, grade, false>
{
    typedef typename filter_clist_for_grade<typename list::tail, grade>::clist clist;
};
template <conf_t grade, bool pass>
struct filter_clist_for_grade<cl_null, grade, pass>
{
    typedef cl_null clist;
};
template <conf_t grade>
struct filter_clist_for_grade<cl_null, grade, false>
{
    typedef cl_null clist;
};

//NVCC has problems with struct grade having same name than function grade, although in different namescopes, thus capitel letter for struct Grade
template <conf_t G, class A>
struct Grade : public expression<Grade<G, A> >
{
    typedef typename filter_clist_for_grade<typename A::clist, G>::clist clist;

    typedef typename A::metric metric;

    typedef typename A::element_t element_t;

    Grade(const A &a_)
        : a(a_)
    {
    }

    template <conf_t conf>
    GAALET_CUDA_HOST_DEVICE
        element_t
            element() const
    {
        return (gaalet::search_element<conf, clist>::index != clist::size) ? a.element<conf>() : 0.0;
    }

protected:
    const A &a;
};

} //end namespace gaalet

template <gaalet::conf_t G, class A>
inline GAALET_CUDA_HOST_DEVICE
    gaalet::Grade<G, A>
    grade(const gaalet::expression<A> &a)
{
    return gaalet::Grade<G, A>(a);
}

#endif
