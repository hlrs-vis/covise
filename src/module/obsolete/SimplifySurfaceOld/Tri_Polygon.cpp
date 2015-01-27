/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:  Source Code for the triangulation of simple polygons     **
 **               without holes                                            **
 **                                                                        **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Karin Frank                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  September 1997  V1.0                                            **
\**************************************************************************/
#include "Tri_Polygon.h"
#include <util/coviseCompat.h>

TriPolygon::TriPolygon(int num_v, float (*vertices)[2])
{
    int i;

    num_vert = num_v;
    num_diag = 0;
    vertex_list = new Vertex_with_coords[num_vert];
    diag_list = new Diagonal[num_vert];

    for (i = 0; i < num_vert; i++)
    {
        vertex_list[i].set_key(i);
        vertex_list[i].set_x((double)vertices[i][0]);
        vertex_list[i].set_y((double)vertices[i][1]);
        diag_list[i].total = 0;
    }

    heap = new PQ<Vertex_with_coords>(num_vert);
    for (i = 0; i < num_vert; i++)
        heap->append(vertex_list[i]);

    heap->construct();

    status = new Tree(num_vert, vertices);
}

TriPolygon::~TriPolygon()
{
    delete heap;
    delete status;
    delete[] vertex_list;
    delete[] diag_list;
}

inline void TriPolygon::insert_diagonal(int v1, int v2)
{
    diag_list[v1].to[diag_list[v1].total] = v2;
    diag_list[v1].used[diag_list[v1].total++] = 0;
    diag_list[v2].to[diag_list[v2].total] = v1;
    diag_list[v2].used[diag_list[v2].total++] = 0;
    num_diag++;
}

int TriPolygon::get_next_diagonal(int &v, int &i)
{
    int found = 0;
    i = i + 1;
    while (v < num_vert && !found)
    {
        while (diag_list[v].used[i] && i < diag_list[v].total)
            i = i + 1;
        if (i < diag_list[v].total)
            found = 1;
        else
        {
            i = 0;
            v = v + 1;
        }
    }
    return (found);
}

inline int TriPolygon::is_above(int v, int v1)
{
    return ((vertex_list[v].get_y() - vertex_list[v1].get_y() >= 1E-06)
            || ((fabs(vertex_list[v].get_y() - vertex_list[v1].get_y()) < 1E-06) && (vertex_list[v].get_x() < vertex_list[v1].get_x())));
}

int TriPolygon::HandleStartVertex(int key, double sweep)
{
    status->insert(key, key, sweep);
    return (1);
}

int TriPolygon::HandleEndVertex(int key, double sweep)
{
    int h;

    h = status->get_helper((key - 1 + num_vert) % num_vert, sweep);
    if (h == -1)
        return (0);
    if (vertex_list[h].getType() == 2)
        insert_diagonal(h, key);
    status->remove((key - 1 + num_vert) % num_vert, sweep);
    return (1);
}

int TriPolygon::HandleSplitVertex(int key, double sweep)
{
    int l, edge;

    l = status->get_left_neighbour(key, sweep);
    if (l == -1)
        return (0);
    edge = status->get_helper(l, sweep);
    if (edge == -1)
        return (0);
    insert_diagonal(key, edge);
    status->set_helper(l, key, sweep);
    status->insert(key, key, sweep);
    return (1);
}

int TriPolygon::HandleMergeVertex(int key, double sweep)
{
    int h, l;

    h = status->get_helper((key - 1 + num_vert) % num_vert, sweep);
    if (h == -1)
        return (0);
    if (vertex_list[h].getType() == 2)
        insert_diagonal(key, h);
    status->remove((key - 1 + num_vert) % num_vert, sweep);

    l = status->get_left_neighbour(key, sweep);
    if (l == -1)
        return (0);
    h = status->get_helper(l, sweep);
    if (h == -1)
        return (0);
    if (vertex_list[h].getType() == 2)
        insert_diagonal(key, h);
    status->set_helper(l, key, sweep);
    return (1);
}

int TriPolygon::HandleRegularVertex(int key, double sweep)
{
    int h, l;

    //if (vertex_list[(key-1+num_vert)%num_vert].co_y > vertex_list[(key+1)%num_vert].co_y)
    if (is_above((key - 1 + num_vert) % num_vert, (key + 1) % num_vert))
    {
        h = status->get_helper((key - 1 + num_vert) % num_vert, sweep);
        if (h == -1)
            return (0);
        if (vertex_list[h].getType() == 2)
            insert_diagonal(key, h);
        status->remove((key - 1 + num_vert) % num_vert, sweep);
        status->insert(key, key, sweep);
    }
    else
    {
        l = status->get_left_neighbour(key, sweep);
        if (l == -1)
            return (0);
        h = status->get_helper(l, sweep);
        if (h == -1)
            return (0);
        if (vertex_list[h].getType() == 2)
            insert_diagonal(key, h);
        status->set_helper(l, key, sweep);
    }
    return (1);
}

int TriPolygon::HandleVertex(int key, double sweep)
{
    int v1, v2;
    int above_v1, above_v2;
    double interior_angle;

    v1 = (key - 1 + num_vert) % num_vert;
    v2 = (key + 1) % num_vert;
    interior_angle = IncreasesInteriorAnglePi(v1, key, v2);
    above_v1 = is_above(v1, key);
    above_v2 = is_above(v2, key);
    if (above_v1 && above_v2)
    {
        if (interior_angle < 0)
        {
            vertex_list[key].set_type(0);
            if (!HandleEndVertex(key, sweep))
                return (0);
        }
        else
        {
            vertex_list[key].set_type(2);
            if (!HandleMergeVertex(key, sweep))
                return (0);
        }
    }
    else if (!above_v1 && !above_v2)
    {
        if (interior_angle < 0)
        {
            vertex_list[key].set_type(1);
            if (!HandleStartVertex(key, sweep))
                return (0);
        }
        else
        {
            vertex_list[key].set_type(3);
            if (!HandleSplitVertex(key, sweep))
                return (0);
        }
    }
    else
    {
        vertex_list[key].set_type(4);
        if (!HandleRegularVertex(key, sweep))
            return (0);
    }
    return (1);
}

double TriPolygon::IncreasesInteriorAnglePi(int v1, int v, int v2)
{
    double n[2];
    double delta[2];
    double angle;
    double norm[2];

    // Assumption: The polygon lies on the left side of the edge (v1,v).
    // If the return value has negative sign, v2 lies on the left side of
    // the line (v1,v), otherwise it lies on its right side
    // left side == interior angle < PI
    // right side == interior angle > PI

    n[0] = vertex_list[v].get_y() - vertex_list[v1].get_y();
    n[1] = -(vertex_list[v].get_x() - vertex_list[v1].get_x());
    delta[0] = vertex_list[v2].get_x() - vertex_list[v].get_x();
    delta[1] = vertex_list[v2].get_y() - vertex_list[v].get_y();
    norm[0] = sqrt(n[0] * n[0] + n[1] * n[1]);
    norm[1] = sqrt(delta[0] * delta[0] + delta[1] * delta[1]);

    n[0] /= norm[0];
    n[1] /= norm[0];
    delta[0] /= norm[1];
    delta[1] /= norm[1];

    angle = n[0] * delta[0] + n[1] * delta[1];

    if (-1E-06 < angle && angle < 1E-06)
        angle = 0.0;
    return (angle);
}

double TriPolygon::compute_inner_angle(int v1, int v2, int v3)
{
    double n[2];
    double m[2];
    double norm[2];
    double alpha;
    double scalar;
    double angle;

    // Computes not the value of the angle (avoiding the acosf-function)
    // but a monotonely increasing funtion of the angle:
    //          - ( 1 + cos x)   x \in [0, M_PI]
    // alpha =
    //            (1 + cos x)    x \in (M_PI, 2 * M_PI)
    //

    n[0] = vertex_list[v2].get_y() - vertex_list[v1].get_y();
    n[1] = -(vertex_list[v2].get_x() - vertex_list[v1].get_x());
    m[0] = vertex_list[v2].get_y() - vertex_list[v3].get_y();
    m[1] = -(vertex_list[v2].get_x() - vertex_list[v3].get_x());

    norm[0] = sqrt(m[0] * m[0] + m[1] * m[1]);
    norm[1] = sqrt(n[0] * n[0] + n[1] * n[1]);
    for (int i = 0; i < 2; i++)
    {
        m[i] /= norm[0];
        n[i] /= norm[1];
    }

    angle = n[0] * vertex_list[v3].get_x();
    angle += n[1] * vertex_list[v3].get_y();
    angle -= n[0] * vertex_list[v2].get_x() + n[1] * vertex_list[v2].get_y();

    scalar = n[0] * m[0] + n[1] * m[1];
    alpha = 1.0 + scalar;

    if (angle < 0)
        alpha = -alpha;

    return (alpha);
}

int TriPolygon::detect_monotone_polygon(int &num, int *v_list)
{
    int v0, v1, v2;
    int index;
    int which;
    int count = 0;
    double angle, min_angle;

    v0 = 0;
    index = -1;
    num = 0;

    if (get_next_diagonal(v0, index))
    {
        v1 = v0;
        v2 = diag_list[v0].to[index];
        diag_list[v0].used[index]++;
        v_list[num++] = v0;
        v_list[num++] = v2;

        while (v2 != v0 && count < 1000)
        { // collect possible ways out of v2, choose the edge or diagonal
            // with the minimal inner angle to (v1,v2)
            which = -1;
            if (diag_list[v2].total != 0)
            {
                min_angle = compute_inner_angle(v1, v2, (v2 + 1) % num_vert);
                for (int i = 0; i < diag_list[v2].total; i++)
                {
                    if (diag_list[v2].to[i] != v1)
                    {
                        angle = compute_inner_angle(v1, v2, diag_list[v2].to[i]);
                        if (angle < min_angle)
                        {
                            which = i;
                            min_angle = angle;
                        }
                    }
                }
            }
            v1 = v2;
            if (which == -1)
            { // continue with edge (v1, v1+1)
                v2 = (v1 + 1) % num_vert;
            }
            else
            { // continue with diagonal (v1, v2)
                v2 = diag_list[v1].to[which];
                diag_list[v1].used[which]++;
            }
            v_list[num++] = v2;
            count++;
        }
        num = num - 1;
        return (num);
    }
    else
        return (0);
}

int TriPolygon::MakeMonotone()
{
    Vertex_with_coords v;

    while (heap->getSize() != 0)
    {
        v = heap->get_next();
        if (!HandleVertex(v.get_key(), v.get_y()))
            return (0);
    }
    return (1);
}

int TriPolygon::HandleMonotone(int num, int *v_list, int &num_tri, int (*triangles)[3], int optimize)
{
    MonotonePolygon *m_poly;
    double co_v[max_edges][2];
    int tri[max_edges][3];

    if (num < 3)
        return (0);

    if (num == 3)
    {
        // append triangle
        triangles[num_tri][0] = v_list[0];
        triangles[num_tri][1] = v_list[1];
        triangles[num_tri++][2] = v_list[2];
    }
    else
    {
        // triangulate monotone subpolygon
        int i;
        for (i = 0; i < num; i++)
        {
            co_v[i][0] = vertex_list[v_list[i]].get_x();
            co_v[i][1] = vertex_list[v_list[i]].get_y();
        }
        m_poly = new MonotonePolygon(num, co_v);
        if (0 == m_poly->TriangulatePolygon(tri, optimize))
        {
            delete m_poly;
            return (0);
        }

        delete m_poly;

        // append triangles
        for (i = 0; i < num - 2; i++)
        {
            triangles[num_tri][0] = v_list[tri[i][0]];
            triangles[num_tri][1] = v_list[tri[i][1]];
            triangles[num_tri++][2] = v_list[tri[i][2]];
        }
    }
    return (num_tri);
}

int TriPolygon::TriangulatePolygon(int (*triangles)[3], int optimize)
{
    int num, i;
    int v_list[max_edges];
    int num_tri = 0;

    // decompose polygon into y-monotone subpolygons
    num_diag = 0;

    if (!MakeMonotone())
        return (0);

    if (num_diag == 0)
    {
        for (i = 0; i < num_vert; i++)
            v_list[i] = i;
        if (!HandleMonotone(num_vert, v_list, num_tri, triangles, optimize))
            return (0);
    }
    else
    {
        // extract next monotone polygon
        while (detect_monotone_polygon(num, v_list))
            if (!HandleMonotone(num, v_list, num_tri, triangles, optimize))
                return (0);
    }

    if (num_tri != num_vert - 2)
    {
        cout << "Error: Possibly not all monotone polygons found!!!" << endl;
        return (0);
    }
    return (num_tri);
}

//////////////////////////////////////////////////////////////////////////////
//    CLASS MONOTONE POLYGON                                                //
//////////////////////////////////////////////////////////////////////////////

MonotonePolygon::~MonotonePolygon()
{
    delete chain;
}

MonotonePolygon::MonotonePolygon(int num_v, double (*vertices)[2])
{
    int i, j;

    num_vert = num_v;
    num_diag = 0;
    vertex_list = new Vertex_with_coords[num_vert];
    diag_list = new Diagonal[num_vert];

    Maximal_y_coordinate = vertices[0][1];
    which_max = 0;

    for (i = 0; i < num_vert; i++)
    {
        vertex_list[i].set_key(i);
        vertex_list[i].set_x(vertices[i][0]);
        vertex_list[i].set_y(vertices[i][1]);
        diag_list[i].total = 0;
        if (vertices[i][1] > Maximal_y_coordinate)
        {
            Maximal_y_coordinate = vertices[i][1];
            which_max = i;
        }
    }

    heap = new PQ<Vertex_with_coords>(num_vert);
    vertex_list[which_max].set_type(1);
    heap->append(vertex_list[which_max]);

    for (i = which_max + 1; i < which_max + num_vert; i++)
    {
        // left chain = 0
        // right chain = 1
        j = i % num_vert;
        if (vertex_list[(i - 1 + num_vert) % num_vert].get_y() - vertex_list[j].get_y() > 1E-06)
            vertex_list[j].set_type(0);
        else if (vertex_list[j].get_y() - vertex_list[(i - 1 + num_vert) % num_vert].get_y() > 1E-06)
            vertex_list[j].set_type(1);
        else
            vertex_list[j].set_type(vertex_list[(i - 1 + num_vert) % num_vert].getType());

        heap->append(vertex_list[j]);
    }

    heap->construct();
    status = new Tree(num_vert, vertices);

    chain = new Stack(max_edges);
}

inline int MonotonePolygon::get_diag_index(int v, int v1)
{
    int i = 0;
    while (i < diag_list[v].total && diag_list[v].to[i] != v1)
        i++;
    if (i < diag_list[v].total)
        return (i);
    else
        return (-1);
}

double MonotonePolygon::compute_angle_weight(int v0, int v1, int v2)
{
    double angle;
    double x0, x1, x2;
    double y0, y1, y2;
    double cosi0, cosi1, cosi2;
    double norm0, norm1, norm2;

    x0 = vertex_list[v2].get_x() - vertex_list[v0].get_x();
    x1 = vertex_list[v1].get_x() - vertex_list[v2].get_x();
    x2 = vertex_list[v0].get_x() - vertex_list[v1].get_x();

    y0 = vertex_list[v2].get_y() - vertex_list[v0].get_y();
    y1 = vertex_list[v1].get_y() - vertex_list[v2].get_y();
    y2 = vertex_list[v0].get_y() - vertex_list[v1].get_y();

    norm0 = sqrt(x0 * x0 + y0 * y0);
    norm1 = sqrt(x1 * x1 + y1 * y1);
    norm2 = sqrt(x2 * x2 + y2 * y2);

    cosi0 = (-x2 * x0 - y2 * y0) / (norm2 * norm0);
    cosi1 = (-x1 * x2 - y1 * y2) / (norm1 * norm2);
    cosi2 = (-x1 * x0 - y1 * y0) / (norm1 * norm0);

    angle = 2.0 * (cosi0 + cosi1 + cosi2) - 2.0;

    return (1.0 - angle);
}

int MonotonePolygon::detect_triangles(int (*triangles)[3])
{
    int v0, v1, v2;
    int last_di;
    int index;
    int num_tri = 0;
    int which;
    double angle, min_angle;

    v0 = 0;
    last_di = -1;

    while (get_next_diagonal(v0, last_di))
    {
        v1 = diag_list[v0].to[last_di];
        diag_list[v0].used[last_di]++;
        diag_list[v0].tri[last_di] = num_tri;

        // collect possible ways out of v1, choose the edge or diagonal
        // with the minimal inner angle to (v0,v1)
        which = -1;
        if (diag_list[v1].total != 0)
        {
            min_angle = compute_inner_angle(v0, v1, (v1 + 1) % num_vert);
            for (int i = 0; i < diag_list[v1].total; i++)
            {
                if (diag_list[v1].to[i] != v0)
                {
                    angle = compute_inner_angle(v0, v1, diag_list[v1].to[i]);
                    if (angle < min_angle)
                    {
                        which = i;
                        min_angle = angle;
                    }
                }
            }
        }

        if (which == -1)
        { // continue with edge (v1, v1+1)
            v2 = (v1 + 1) % num_vert;
            if (v2 != (v0 - 1 + num_vert) % num_vert)
            {
                index = get_diag_index(v2, v0);
                diag_list[v2].used[index]++;
                diag_list[v2].tri[index] = num_tri;
            }
        }
        else
        { // continue with diagonal (v1, v2)
            v2 = diag_list[v1].to[which];
            diag_list[v1].used[which]++;
            diag_list[v1].tri[which] = num_tri;

            if (v2 != (v0 - 1 + num_vert) % num_vert)
            {
                index = get_diag_index(v2, v0);
                diag_list[v2].used[index]++;
                diag_list[v2].tri[index] = num_tri;
            }
        }
        triangles[num_tri][0] = v0;
        triangles[num_tri][1] = v1;
        triangles[num_tri++][2] = v2;
    }
    return (num_tri);
}

void MonotonePolygon::optimize_triangulation(int (*triangles)[3])
{
    int v0, v1, v2, v3;
    int t1, t2;
    int i, j;
    int last_di;
    int index;
    double angle1, angle2;

    // Start of the procedure: all diag_list.used[]-values are 1!!!
    // In the procedure they are set to zero first

    for (i = 0; i < num_vert; i++)
        for (j = 0; j < diag_list[i].total; j++)
            diag_list[i].used[j] = 0;

    v0 = 0;
    last_di = -1;

    while (get_next_diagonal(v0, last_di))
    {
        t1 = diag_list[v0].tri[last_di];
        v1 = diag_list[v0].to[last_di];
        if (triangles[t1][0] == v0)
            v2 = triangles[t1][2];
        else if (triangles[t1][1] == v0)
            v2 = triangles[t1][0];
        else
            v2 = triangles[t1][1];

        index = get_diag_index(v1, v0);
        t2 = diag_list[v1].tri[index];
        if (triangles[t2][0] == v1)
            v3 = triangles[t2][2];
        else if (triangles[t2][1] == v1)
            v3 = triangles[t2][0];
        else
            v3 = triangles[t2][1];

        if (IncreasesInteriorAnglePi(v2, v0, v3) >= 0 || IncreasesInteriorAnglePi(v3, v1, v2) >= 0)
        {
            diag_list[v0].used[last_di] = 1;
            diag_list[v1].used[index] = 1;
        }
        else
        {
            angle1 = compute_angle_weight(v0, v1, v2);
            angle1 += compute_angle_weight(v0, v3, v1);
            angle2 = compute_angle_weight(v0, v3, v2);
            angle2 += compute_angle_weight(v3, v1, v2);

            if (angle1 > angle2)
            { // swap diagonals
                diag_list[v2].to[diag_list[v2].total] = v3;
                diag_list[v2].used[diag_list[v2].total] = 1;
                diag_list[v2].tri[diag_list[v2].total] = t1;
                diag_list[v2].total++;
                diag_list[v3].to[diag_list[v3].total] = v2;
                diag_list[v3].used[diag_list[v3].total] = 1;
                diag_list[v3].tri[diag_list[v3].total] = t2;
                diag_list[v3].total++;

                triangles[t1][0] = v3;
                triangles[t1][1] = v2;
                triangles[t1][2] = v0;
                triangles[t2][0] = v3;
                triangles[t2][1] = v1;
                triangles[t2][2] = v2;

                for (i = last_di; i < diag_list[v0].total - 1; i++)
                {
                    diag_list[v0].to[i] = diag_list[v0].to[i + 1];
                    diag_list[v0].used[i] = diag_list[v0].used[i + 1];
                    diag_list[v0].tri[i] = diag_list[v0].tri[i + 1];
                }
                diag_list[v0].total--;
                for (i = index; i < diag_list[v1].total - 1; i++)
                {
                    diag_list[v1].to[i] = diag_list[v1].to[i + 1];
                    diag_list[v1].used[i] = diag_list[v1].used[i + 1];
                    diag_list[v1].tri[i] = diag_list[v1].tri[i + 1];
                }
                diag_list[v1].total--;

                if (-1 != (i = get_diag_index(v0, v3)))
                    diag_list[v0].tri[i] = t1;

                if (-1 != (i = get_diag_index(v1, v2)))
                    diag_list[v1].tri[i] = t2;
            }
            else
            {
                diag_list[v0].used[last_di] = 1;
                diag_list[v1].used[index] = 1;
            }
        }
    }
}

int MonotonePolygon::TriangulatePolygon(int (*tri)[3], int optimize)
{
    Vertex_with_coords v;
    int v1, v2;
    Vertex_with_coords last;
    int convex, last_di;
    int num_tri = 0;

    v = heap->get_next();
    chain->push(v.get_key());
    v = heap->get_next();
    chain->push(v.get_key());

    last = v;
    while (heap->getSize() > 1)
    {
        v = heap->get_next();
        v1 = chain->pop();
        last_di = v1;
        if (vertex_list[v1].getType() != vertex_list[v.get_key()].getType())
        { // vertices are on different chains
            if (!chain->empty())
                insert_diagonal(v.get_key(), v1);
            while (!chain->empty())
            {
                v2 = chain->pop();
                if (!chain->empty())
                    insert_diagonal(v.get_key(), v2);
            }
            chain->push(last.get_key());
            chain->push(v.get_key());
        }
        else
        { // vertices are on the same chain
            convex = 1;
            while (!chain->empty() && convex)
            {
                v2 = chain->pop();
                if (vertex_list[v.get_key()].getType() == 0)
                {
                    if (IncreasesInteriorAnglePi(v.get_key(), last_di, v2) > 0)
                    //if(IncreasesInteriorAnglePi(v.get_key(), last_di, v2) > 1E-01)
                    {
                        insert_diagonal(v.get_key(), v2);
                        last_di = v2;
                    }
                    else
                    {
                        chain->push(v2);
                        convex = 0;
                    }
                }
                else
                {
                    if (IncreasesInteriorAnglePi(v.get_key(), last_di, v2) < 0)
                    //if(IncreasesInteriorAnglePi(v.get_key(), last_di, v2) < -1E-01)
                    {
                        insert_diagonal(v.get_key(), v2);
                        last_di = v2;
                    }
                    else
                    {
                        chain->push(v2);
                        convex = 0;
                    }
                }
            }
            chain->push(last_di);
            chain->push(v.get_key());
        }
        last = v;
    }
    v = heap->get_next();
    v2 = chain->pop();
    while (!chain->empty())
    {
        v2 = chain->pop();
        if (!chain->empty())
            insert_diagonal(v.get_key(), v2);
    }

    num_tri = detect_triangles(tri);

    if (optimize)
        optimize_triangulation(tri);

    return (num_tri);
}
