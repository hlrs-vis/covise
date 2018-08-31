/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVER_TIPSIFY_H
#define COVER_TIPSIFY_H

/*! \file
 \brief quickly optimize locality in triangle index list for vertex cache use

 as proposed in Sander, Nehab, Barczak: "Fast Triangle Reordering for Vertex Locality and Reduced Overdraw"
 
 \author Martin Aumüller <aumueller@hlrs.de>
 */

#include <cassert>
#include <vector>
#include <algorithm>

namespace opencover
{

namespace Tipsify {

template<class Index>
struct Neighbors {
    Index maxIndex = 0;
    std::vector<Index> use; // no. of triangles where vertex is used
    std::vector<Index> tl; // list of triangles
    std::vector<Index> offset; // offset into list of triangles where sublist for a vertex starts
};

// build neighbors
template<class Index>
Neighbors<Index> buildNeighbours(const Index *idx, size_t size, Index maxIndex) {

    Neighbors<Index> N;

    N.use.resize(maxIndex+1);
    const Index *end = idx+size;
    for (const Index *v = idx; v<end; ++v) {
        if (*v > N.maxIndex) {
            N.maxIndex = *v;
            if (N.use.size()<N.maxIndex+1)
                N.use.resize(N.maxIndex+1);
        }
        ++N.use[*v];
    }
    N.use.resize(N.maxIndex+1);

    N.offset.reserve(N.maxIndex+2);
    Index sum = 0;
    N.offset.push_back(sum);
    N.offset.push_back(sum);
    for (Index v=0; v<=N.maxIndex; ++v) {
        sum += N.use[v];
        N.offset.push_back(sum);
    }
    assert(sum == size);
    assert(N.offset[0] == 0);
    assert(N.offset[1] == 0);
    assert(N.offset[N.maxIndex+1] < sum);

    N.tl.resize(sum);
    for (size_t i=0; i<size; ++i) {
        Index v = idx[i];
        N.tl[N.offset[v+1]++] = i/3;
    }
    assert(N.offset[N.maxIndex+1] == sum);

    return N;
}

} // namespace Tipsify

template<class Index>
void tipsify(Index *idx, size_t num, int cachesize=20, int batchsize=-1) {

    enum IndexValues : Index {
        InvalidIndex = ~Index(0)
    };

    using namespace Tipsify;

    auto N = buildNeighbours<Index>(idx, num, num);

    std::vector<Index> out; // optimized index list
    out.reserve(num);

    std::vector<char> emitted(N.tl.size()); // if a triangle has already been emitted

    std::vector<Index> deadEndStack;
    std::vector<int> cachetime(N.maxIndex+1);
    int time=cachesize+1;
    size_t cursor = 0;
    Index f = idx[0]; // start vertex for triangle fans
    int remaininbatch = batchsize;
    while (f != InvalidIndex) {

        std::set<Index> candidates;
        for (Index i=N.offset[f]; i<N.offset[f+1]; ++i) {
            Index t = N.tl[i];
            if (!emitted[t]) {
                emitted[t] = true;
                for (int i=0; i<3; ++i) {
                    --remaininbatch;
                    Index v = idx[t*3+i];
                    out.push_back(v);
                    deadEndStack.push_back(v);
                    candidates.insert(v);
                    --N.use[v];
                    if (time-cachetime[v] > cachesize) {
                        // not in cache
                        cachetime[v] = time;
                        ++time;
                    }
                }
            }
        }

        f = InvalidIndex;
        bool startFresh = false;
        if (batchsize>0) {
            startFresh = remaininbatch==0;

            while (remaininbatch <= 0)
                remaininbatch += batchsize;

            if (startFresh) {
                time += batchsize;
                deadEndStack.clear();
            }
        }

        // find another vertex to fan triangles around
        if (!startFresh) {
            int maxPriority = -1;
            for (auto v: candidates) {
                // try candidates from previous fan
                if (N.use[v] == 0)
                    continue;
                int priority = 0;
                if (time - cachetime[v] + 2*N.use[v] <= cachesize && (batchsize<0 || N.use[v]<=remaininbatch))
                    priority = time-cachetime[v];
                if (priority > maxPriority) {
                    maxPriority = priority;
                    f = v;
                }
            }
            if (f == InvalidIndex) {
                // try candidates from dead-end stack
                while (!deadEndStack.empty()) {
                    Index v = deadEndStack.back();
                    deadEndStack.pop_back();
                    if (N.use[v] > 0) {
                        f = v;
                        break;
                    }
                }
            }
        }
        if (f == InvalidIndex) {
            // take another vertex from input
            for (; cursor < num; ++cursor) {
                Index v = idx[cursor];
                if (N.use[v] > 0) {
                    f = v;
                    break;
                }
            }
        }
    }

    assert(out.size() == num);
    std::copy(out.begin(), out.end(), idx);
}

}
#endif
