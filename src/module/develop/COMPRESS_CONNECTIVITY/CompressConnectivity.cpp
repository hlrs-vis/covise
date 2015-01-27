/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CompressConnectivity.h"
#include <util/coIA.h>
#include <covise/covise_unstrgrd.h>
#include <iterator>
using namespace std;

int
main(int argc, char *argv[])
{
    CompressConnectivity *application = new CompressConnectivity;
    application->start(argc, argv);
    return 1;
}

CompressConnectivity::CompressConnectivity()
{
    p_grid_in_ = addInputPort("InGrid", "UnstructuredGrid", "Input grid");

    p_grid_out_ = addOutputPort("OutGrid", "UnstructuredGrid", "output grid");
    p_data_out_ = addOutputPort("Data", "Vec3|Float", "scalar data");
}

CompressConnectivity::~CompressConnectivity()
{
}

void
catIntList(ia<int> &a, ia<int> &b)
{
    for (int i = 0; i < b.size(); i++)
    {
        a.push_back(b[i]);
    }
    b.schleifen();
}

int
CompressConnectivity::compute()
{
    ia<int> elist, tmp_elist;
    ia<int> tlist, tmp_tlist;
    ia<int> clist, tmp_clist;
    ia<float> x, y, z;

    int *el, *cl, *tl;
    float *xc, *yc, *zc;
    coDistributedObject *inObj = p_grid_in_->getCurrentObject();
    if (!inObj->isType("UNSGRD"))
    {
        p_grid_out_->setCurrentObject(new coDoUnstructuredGrid(p_grid_out_->getObjName(),
                                                               0, 0, 0, 1));
        return SUCCESS;
    }
    coDoUnstructuredGrid *inGrid = dynamic_cast<coDoUnstructuredGrid *>(inObj);
    inGrid->getAddresses(&el, &cl, &xc, &yc, &zc);
    int nele, nconn, ncoord;
    inGrid->getGridSize(&nele, &nconn, &ncoord);
    inGrid->getTypeList(&tl);

    // copy coords
    for (int i = 0; i < ncoord; i++)
    {
        x.push_back(xc[i]);
        y.push_back(yc[i]);
        z.push_back(zc[i]);
    }

    /////
    int num_neigh;
    int *neighList, *vStart;
    inGrid->getNeighborList(&num_neigh, &neighList, &vStart);

    int elem;
    int i, j;

    for (elem = 0; elem < nele; ++elem)
    {
        bool copy_hex = true;
        tmp_elist.schleifen();
        tmp_tlist.schleifen();
        tmp_clist.schleifen();

        if (tl[elem] == TYPE_HEXAEDER)
        {
            int surf_cube1[] = { 0, 1, 2, 3 };
            int surf_cube2[] = { 3, 2, 6, 7 };
            int surf_cube3[] = { 4, 5, 6, 7 };
            int surf_cube4[] = { 0, 1, 5, 4 };
            int surf_cube5[] = { 4, 0, 3, 7 };
            int surf_cube6[] = { 1, 2, 6, 5 };

            int *surf_cube[] = { &surf_cube1[0], &surf_cube2[0], &surf_cube3[0], &surf_cube4[0], &surf_cube5[0], &surf_cube6[0] };

            float middle_point[3];
            middle_point[0] = 0.5 * (x[cl[el[elem]]] + x[cl[el[elem] + 6]]);
            middle_point[1] = 0.5 * (y[cl[el[elem]]] + y[cl[el[elem] + 6]]);
            middle_point[2] = 0.5 * (z[cl[el[elem]]] + z[cl[el[elem] + 6]]);
            int middle_added = x.size(); // middle_point added to coord array
            x.push_back(middle_point[0]);
            y.push_back(middle_point[1]);
            z.push_back(middle_point[2]);

            int csurf[4];
            for (int s = 0; s < 6; s++)
            {
                /* cout << "--------------------------------------------------------------------" << endl;
	         cout << "Checking surf " << cl[el[elem]+surf_cube[s][0]] << " " <<
		                             cl[el[elem]+surf_cube[s][1]] << " " <<
					     cl[el[elem]+surf_cube[s][2]] << " " <<
					     cl[el[elem]+surf_cube[s][3]] << " " <<endl << endl;*/
                set<int> conn_elem; //list of connected elements
                set<int> base_elem; // base quad
                set<int> common;
                insert_iterator<set<int> > iter(common, common.begin());
                for (int t = 0; t < 4; t++)
                {
                    csurf[t] = cl[el[elem] + surf_cube[s][t]];
                    base_elem.insert(csurf[t]);
                    for (j = vStart[csurf[t]]; j < vStart[csurf[t] + 1]; j++)
                    {
                        if (neighList[j] != elem && tl[neighList[j]] != TYPE_HEXAEDER)
                        {
                            conn_elem.insert(neighList[j]);
                        }
                    }
                }

                set<int>::iterator it;
                set<int> query;
                set<int> diff;
                insert_iterator<set<int> > iter2(diff, diff.begin());
                bool tetra = false;

                for (it = conn_elem.begin(); it != conn_elem.end(); it++)
                {
                    common.clear();
                    query.clear();
                    diff.clear();

                    // fill query with corners of connected elem
                    int next_el = (*it == nele - 1) ? nconn : el[*it + 1];
                    for (i = el[*it]; i < next_el; i++)
                    {
                        query.insert(cl[i]);
                    }

                    set_intersection(base_elem.begin(), base_elem.end(),
                                     query.begin(), query.end(), iter);
                    //splitting cube correctly
                    if (common.size() == 3)
                    {
                        //take tetra between triangle on surface and middle point
                        set_difference(base_elem.begin(), base_elem.end(),
                                       query.begin(), query.end(), iter2);

                        if (diff.size() == 1)
                        {
                            // cout << "Touch element " << *it << "on " << elem << endl;
                            cout << "JJJJJJJJJJJJJJJJJJJJJ" << endl;
                            tetra = true;

                            copy_hex = false;
                            catIntList(tlist, tmp_tlist);
                            catIntList(elist, tmp_elist);
                            catIntList(clist, tmp_clist);

                            int not_common = *(diff.begin());

                            // first tetra
                            tlist.push_back(TYPE_TETRAHEDER);
                            elist.push_back(clist.size());
                            for (int t = 0; t < 4; t++)
                            {
                                if (csurf[t] != not_common)
                                {
                                    clist.push_back(csurf[t]);
                                }
                            }
                            clist.push_back(middle_added);
                        }
                    }
                }
                if (!tetra)
                {
                    //take pyramid between surface and middle point
                    if (!copy_hex)
                    {
                        tlist.push_back(TYPE_PYRAMID);
                        elist.push_back(clist.size());
                    }
                    else
                    {
                        tmp_tlist.push_back(TYPE_PYRAMID);
                        tmp_elist.push_back(clist.size() + tmp_clist.size());
                    }

                    for (int t = 0; t < 4; t++)
                    {
                        if (!copy_hex)
                        {
                            clist.push_back(csurf[t]);
                        }
                        else
                        {
                            tmp_clist.push_back(csurf[t]);
                        }
                    }
                    if (!copy_hex)
                    {
                        clist.push_back(middle_added);
                    }
                    else
                    {
                        tmp_clist.push_back(middle_added);
                    }
                }
            }
        }
        if (tl[elem] != TYPE_HEXAEDER || copy_hex)
        {
            //copy element
            tlist.push_back(tl[elem]);
            elist.push_back(clist.size());
            int next_el = (elem == nele - 1) ? nconn : el[elem + 1];
            for (i = el[elem]; i < next_el; i++)
            {
                clist.push_back(cl[i]);
            }
        }
    }
    p_grid_out_->setCurrentObject(new coDoUnstructuredGrid(p_grid_out_->getObjName(),
                                                           elist.size(), clist.size(), x.size(),
                                                           elist.getArray(), clist.getArray(),
                                                           x.getArray(), y.getArray(), z.getArray(), tlist.getArray()));

    float *tmp = new float[x.size()];
    memset(tmp, 0, x.size() * sizeof(float));
    p_data_out_->setCurrentObject(new coDoFloat(p_data_out_->getObjName(), x.size(), tmp));

    return SUCCESS;
}

/*elist.push_back(clist.size());
      ia<int> vertices;
      int vertex;
      ia<int> pattern;
      for(vertex=0;vertex<8;++vertex)
      {
         int corner = cl[el[elem] + vertex];
         // check if it is already repeated
         int count_old_corner;
         for(count_old_corner=0;count_old_corner<vertices.size();
            ++count_old_corner)
         {
            if(corner == vertices[count_old_corner])
            {
               break;
            }
         }
         if(count_old_corner == vertices.size())
         {
            vertices.push_back(corner);
            pattern.push_back(vertex);
         }
      }
      // add vertices array to clist
      if(vertices.size() == 4)
      {
         quad_type = TYPE_QUAD;
         // for tetras the connectivity is trivial
         for(vertex=0;vertex<4;++vertex)
         {
            clist.push_back(vertices[vertex]);
            if( cl[el[elem] + vertex] != cl[el[elem] + vertex + 4] )
            {
               quad_type = TYPE_TETRAHEDER;
            }
         }
         if( quad_type == TYPE_TETRAHEDER ) ;
         else cerr << "quad" << endl;
         tlist.push_back(quad_type);
      }
      else if(vertices.size() == 5)
      {
         tlist.push_back(TYPE_PYRAMID);
         for(vertex=0;vertex<5;++vertex)
         {
            clist.push_back(vertices[vertex]);
         }
      }
      else if(vertices.size() == 6)
      {
         tlist.push_back(TYPE_PRISM);
         // for prisms the connectivity is NOT trivial
         // we have to use the pattern

         clist.push_back(vertices[0]);
         // the bottom face may be a line
         if(pattern[2]==4)
         {
            if(pattern[1]==1)
            {
               clist.push_back(vertices[5]);
               clist.push_back(vertices[2]);
               clist.push_back(vertices[1]);
               clist.push_back(vertices[4]);
               clist.push_back(vertices[3]);
            }
            else
            {
               clist.push_back(vertices[2]);
               clist.push_back(vertices[3]);
               clist.push_back(vertices[1]);
               clist.push_back(vertices[5]);
               clist.push_back(vertices[4]);
            }
         }
         // or the top face
         else if(pattern[3]==3)
         {
            if(pattern[4]==4 && pattern[5]==5)
            {
               clist.push_back(vertices[3]);
               clist.push_back(vertices[4]);
               clist.push_back(vertices[1]);
               clist.push_back(vertices[2]);
               clist.push_back(vertices[5]);
            }
            else if(pattern[4]==6)
            {
               clist.push_back(vertices[3]);
               clist.push_back(vertices[5]);
               clist.push_back(vertices[1]);
               clist.push_back(vertices[2]);
               clist.push_back(vertices[4]);
            }
            else
            {
               clist.push_back(vertices[4]);
               clist.push_back(vertices[1]);
               clist.push_back(vertices[3]);
               clist.push_back(vertices[5]);
               clist.push_back(vertices[2]);
            }
         }
         // or one side face...
         else if(pattern[3]==4)                   // side joining
         {
            clist.push_back(vertices[1]);
            clist.push_back(vertices[2]);
            clist.push_back(vertices[3]);
            clist.push_back(vertices[4]);
            clist.push_back(vertices[5]);
         }
         else
         {
            sendError("Sorry, this is a bug");
            return FAIL;
         }
      }
      else if(vertices.size() == 8)
      {
         tlist.push_back(TYPE_HEXAEDER);
         for(vertex=0;vertex<8;++vertex)
         {
            clist.push_back(vertices[vertex]);
         }
      }
      else
      {
         sendError("Sorry, only tetras or prisms are supported for output");
         return FAIL;
      }*/
