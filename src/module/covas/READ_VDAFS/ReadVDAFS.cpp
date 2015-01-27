/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Read module for VDAFS data         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Reiner Beller                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  09.10.97  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadVDAFS.h"

//
// init covise
//

void main(int argc, char *argv[])
{
    // init
    Application *application = new Application(argc, argv);

    // and back to covise
    application->run();

    // done
}

//
// static stub callback functions calling the real class
// member functions
//

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

//
//..........................................................................
//

void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
}

void Application::compute(void *)
{
    //
    // ...... do work here ........
    //

    char buf[256];
    char *dataPath;
    char *GeometryName = NULL;
    char *curveName = NULL;
    char **consnames;
    char **fsnames;
    string elem_id;

    int fd; // file descriptor
    int n_sets = 0; // number of sets
    int n_groups = 0; // number of groups
    int n_elements_in_grp; // number of elements in a group
    int n_elements_in_set; // number of elements in a set
    int m_loops; // number of trim loops of VDAFS element FACE
    int n_cons; // number of cons elements
    int m_pairs; // number of pairs of adjacent SURF or FACE elements
    int list_type;
    int list_pos;

    int loopno;
    int fsno;

    REAL s_i;
    REAL s_e;
    REAL *w1;
    REAL *w2;

    int mode = DELETE;

    Name elem_name;
    Set act_set;
    Group act_grp;
    Vec3d temp_pnt;
    Circle temp_cir;
    Curve temp_crv;
    Face temp_fac;
    Cons temp_con;
    Surf temp_srf;
    Top temp_top;
    Cons_Ensemble *trim_loop;

    coDoPoints *points = NULL;
    DO_NurbsCurveCol *nurbs_curves = NULL;
    DO_NurbsSurfaceCol *nurbs_surfaces = NULL;
    DO_FaceCol *nurbs_faces = NULL;
    coDoSet **nurbs_set = NULL;
    coDoSet *group_set = NULL;
    coDoGeometry *geometry = NULL;

    // Input Parameters
    Covise::get_browser_param("dataPath", &dataPath);

    fd = Covise::open(dataPath, O_RDONLY);
    if (fd == -1)
    {
        sprintf(buf, "Error Opening File '%s'", dataPath);
        Covise::sendError(buf);
        return;
    }

    // get output object name
    GeometryName = Covise::get_object_name("Geometry");
    if (GeometryName == 0L)
    {
        Covise::sendError("ERROR: No object name given for geometry data object");
        return;
    }

    sprintf(buf, "%s_%d", GeometryName, 1);
    group_set = new coDoSet(buf, SET_CREATE);
    if (!group_set->objectOk())
    {
        Covise::sendError("ERROR: creation of data object 'group_set' failed");
        return;
    }

    // !!! Parsing !!!
    cerr << "\n\nPARSING ...\n";
    pars(fd);
    cerr << "\nDone parsing!" << endl;

    cerr << endl << "Sorting name space ..." << endl;
    sort_name();
    cerr << "Done sorting!" << endl << endl;

    // Setting VDAFS data objects
    n_sets = set_list.length();
    n_groups = group_list.length();

    // Assume that sets and groups are disjunct !!!!!
    nurbs_set = new coDoSet *[n_groups + n_sets];

    // groups
    for (int i = 0; i < n_groups; i++)
    {
        sprintf(buf, "%s_1_%d", GeometryName, i + 1);
        nurbs_set[i] = new coDoSet(buf, SET_CREATE);
        if (!nurbs_set[i]->objectOk())
        {
            sprintf(buf, "ERROR: creation of data object 'nurbs_set[%d]' failed", i + 1);
            Covise::sendError(buf);
            return;
        }

        // Selection of VDAFS elements and creation of NURBS objects:
        //
        // Searching names
        // Getting VDAFS object
        // Calculating NURBS representation and adding to NURBS lists

        act_grp = group_list.contents(group_list.item(i));
        n_elements_in_grp = act_grp.get_n_elements();

        cerr << "Group " << i + 1 << ": " << n_elements_in_grp << " Elements" << endl;
        ;

        for (int elem = 0; elem < n_elements_in_grp; elem++)
        {
            elem_id = act_grp.getElement_name(elem);
            elem_name = getname(elem_id);
            //cerr << elem_id << "=";
            /*
                   if ( ErrFile ) ErrFile << endl << elem_id <<endl;
                   else cerr << "File 'error.out' could not be opened!" << endl;
                   */

            list_type = elem_name.getlist_type();
            list_pos = elem_name.getpos_in_list();
            switch (list_type)
            {

            case 2: //POINT
                temp_pnt = point_list.contents(point_list.item(list_pos));
                selectedPointList.append(temp_pnt);
                break;

            case 3: //CIRCLE
                temp_cir = circle_list.contents(circle_list.item(list_pos));
                temp_cir.MakeNurbsCircle();
                break;

            case 4: //CURVE
                temp_crv = curve_list.contents(curve_list.item(list_pos));
                temp_crv.MakeNurbsCurve();
                break;

            case 5: //SURF
                temp_srf = surf_list.contents(surf_list.item(list_pos));
                temp_srf.MakeNurbsSurface(SURF);
                break;

            case 6: //CONS
                temp_con = cons_list.contents(cons_list.item(list_pos));
                elem_id = temp_con.get_curvenme();
                curveName = temp_con.get_curvenme();
                s_i = temp_con.get_s1();
                s_e = temp_con.get_s2();
                elem_name = getname(elem_id);
                list_pos = elem_name.getpos_in_list();
                temp_crv = curve_list.contents(curve_list.item(list_pos));
                temp_crv.PartOfCurve(curveName, s_i, s_e);
                break;

            case 7: //FACE
            {
                temp_fac = face_list.contents(face_list.item(list_pos));
                temp_fac.set_connectionList(mode);
                temp_fac.set_trimLoopList(mode);
                // set fill-up mode if it has not yet been done
                if (mode != ADD)
                    mode = ADD;

                m_loops = temp_fac.get_n_trimLoops();
                trim_loop = temp_fac.get_ConsEnsembles();

                for (loopno = 0; loopno < m_loops; loopno++)
                {
                    n_cons = trim_loop[loopno].get_n_cons();
                    consnames = trim_loop[loopno].get_ConsNames();
                    w1 = trim_loop[loopno].get_w1();
                    w2 = trim_loop[loopno].get_w2();

                    for (int conno = 0; conno < n_cons; conno++)
                    {
                        elem_id = consnames[conno];
                        elem_name = getname(elem_id);
                        list_pos = elem_name.getpos_in_list();
                        temp_con = cons_list.contents(cons_list.item(list_pos));
                        temp_con.PartOfCons(w1[conno], w2[conno]);
                    }
                }

                elem_id = temp_fac.get_surfnme();
                elem_name = getname(elem_id);
                list_pos = elem_name.getpos_in_list();
                temp_srf = surf_list.contents(surf_list.item(list_pos));
                temp_srf.MakeNurbsSurface(FACE);
            }
            break;

            case 10: //TOP
                // ! (so far) not considering the topology !
                // i.e. not exploiting the boundary curves
                temp_top = top_list.contents(top_list.item(list_pos));
                m_pairs = temp_top.get_m_pairs();
                fsnames = temp_top.get_fsNames();
                for (fsno = 0; fsno < 2 * m_pairs; fsno++)
                {
                    elem_id = fsnames[fsno];
                    elem_name = getname(elem_id);
                    list_pos = elem_name.getpos_in_list();
                    list_type = elem_name.getlist_type();
                    if (list_type == 7) //FACE
                    {
                        temp_fac = face_list.contents(face_list.item(list_pos));
                        temp_fac.set_connectionList(mode);
                        temp_fac.set_trimLoopList(mode);
                        // set fill-up mode if it has not yet been done
                        if (mode != ADD)
                            mode = ADD;

                        m_loops = temp_fac.get_n_trimLoops();
                        trim_loop = temp_fac.get_ConsEnsembles();

                        for (loopno = 0; loopno < m_loops; loopno++)
                        {
                            n_cons = trim_loop[loopno].get_n_cons();
                            consnames = trim_loop[loopno].get_ConsNames();
                            w1 = trim_loop[loopno].get_w1();
                            w2 = trim_loop[loopno].get_w2();

                            for (int conno = 0; conno < n_cons; conno++)
                            {
                                elem_id = consnames[conno];
                                elem_name = getname(elem_id);
                                list_pos = elem_name.getpos_in_list();
                                temp_con = cons_list.contents(cons_list.item(list_pos));
                                temp_con.PartOfCons(w1[conno], w2[conno]);
                            }
                        }

                        elem_id = temp_fac.get_surfnme();
                        elem_name = getname(elem_id);
                        list_pos = elem_name.getpos_in_list();
                        temp_srf = surf_list.contents(surf_list.item(list_pos));
                        temp_srf.MakeNurbsSurface(FACE);
                    }
                    else if (list_type == 5) //SURF
                    {
                        temp_srf = surf_list.contents(surf_list.item(list_pos));
                        temp_srf.MakeNurbsSurface(SURF);
                    }
                    else // ERROR
                    {
                        cerr << "ERROR: incorrect element in VDAFS element TOP " << endl;
                    }
                }
                break;

            default:
                cerr << "ERROR" << endl;
                break;
            }

            // cerr << " x ";
        }

        // set POINTs
        points = create_points(buf);

        // Set nurbs_curves
        nurbs_curves = create_nurbs_curves(buf);

        // Set nurbs_surfaces
        nurbs_surfaces = create_nurbs_surfaces(buf);

        // Set FACEs
        nurbs_faces = create_nurbs_faces(buf);
        mode = DELETE; // reset the position index for the connection and trim loops lists

        // Keep the POINTS and NURBS objects in a set
        nurbs_set[i]->addElement(points);
        nurbs_set[i]->addElement(nurbs_curves);
        nurbs_set[i]->addElement(nurbs_surfaces);
        nurbs_set[i]->addElement(nurbs_faces);
        group_set->addElement(nurbs_set[i]);

        delete points;
        delete nurbs_curves;
        delete nurbs_surfaces;
        delete nurbs_faces;
        delete nurbs_set[i];
    }

    // sets
    int k;
    if (n_groups == 0 && n_sets == 1)
        k = 0;
    else
        k = n_groups + 1; // 1st set not represented
    // (because of conflicts between the other sets (or groups)
    //  in the example vdafs-files)
    for (; k < n_groups + n_sets; k++)
    {
        sprintf(buf, "%s_1_%d", GeometryName, k + 1);
        nurbs_set[k] = new coDoSet(buf, SET_CREATE);
        if (!nurbs_set[k]->objectOk())
        {
            sprintf(buf, "ERROR: creation of data object 'nurbs_set[%d]' failed", k + 1);
            Covise::sendError(buf);
            return;
        }

        // Selection of VDAFS elements and creation of NURBS objects:
        //
        // Searching names
        // Getting VDAFS object
        // Calculating NURBS representation and adding to NURBS lists
        act_set = set_list.contents(set_list.item(k - n_groups));

        // solve conflicts between sets !!!!

        n_elements_in_set = act_set.get_n_elements();

        cerr << "Set " << k + 1 - n_groups << ": " << n_elements_in_set << " Elements" << endl;
        ;

        for (int elem = 0; elem < n_elements_in_set; elem++)
        {
            elem_id = act_set.getElement_name(elem);
            elem_name = getname(elem_id);
            //cerr << elem_id << "=";
            /*
           if ( ErrFile ) ErrFile << endl << elem_id <<endl;
           else cerr << "File 'error.out' could not be opened!" << endl;
         */

            list_type = elem_name.getlist_type();
            list_pos = elem_name.getpos_in_list();
            switch (list_type)
            {

            case 2: //POINT
                temp_pnt = point_list.contents(point_list.item(list_pos));
                selectedPointList.append(temp_pnt);
                break;

            case 3: //CIRCLE
                temp_cir = circle_list.contents(circle_list.item(list_pos));
                temp_cir.MakeNurbsCircle();
                break;

            case 4: //CURVE
                temp_crv = curve_list.contents(curve_list.item(list_pos));
                temp_crv.MakeNurbsCurve();
                break;

            case 5: //SURF
                temp_srf = surf_list.contents(surf_list.item(list_pos));
                temp_srf.MakeNurbsSurface(SURF);
                break;

            case 6: //CONS
                temp_con = cons_list.contents(cons_list.item(list_pos));
                elem_id = temp_con.get_curvenme();
                curveName = temp_con.get_curvenme();
                s_i = temp_con.get_s1();
                s_e = temp_con.get_s2();
                elem_name = getname(elem_id);
                list_pos = elem_name.getpos_in_list();
                temp_crv = curve_list.contents(curve_list.item(list_pos));
                temp_crv.PartOfCurve(curveName, s_i, s_e);
                break;

            case 7: //FACE
            {
                temp_fac = face_list.contents(face_list.item(list_pos));
                temp_fac.set_connectionList(mode);
                temp_fac.set_trimLoopList(mode);
                // set fill-up mode if it has not yet been done
                if (mode != ADD)
                    mode = ADD;

                m_loops = temp_fac.get_n_trimLoops();
                trim_loop = temp_fac.get_ConsEnsembles();

                for (loopno = 0; loopno < m_loops; loopno++)
                {
                    n_cons = trim_loop[loopno].get_n_cons();
                    consnames = trim_loop[loopno].get_ConsNames();
                    w1 = trim_loop[loopno].get_w1();
                    w2 = trim_loop[loopno].get_w2();

                    for (int conno = 0; conno < n_cons; conno++)
                    {
                        elem_id = consnames[conno];
                        elem_name = getname(elem_id);
                        list_pos = elem_name.getpos_in_list();
                        temp_con = cons_list.contents(cons_list.item(list_pos));
                        temp_con.PartOfCons(w1[conno], w2[conno]);
                    }
                }

                elem_id = temp_fac.get_surfnme();
                elem_name = getname(elem_id);
                list_pos = elem_name.getpos_in_list();
                temp_srf = surf_list.contents(surf_list.item(list_pos));
                temp_srf.MakeNurbsSurface(FACE);
                break;

            case 10: //TOP
                // ! (so far) not considering the topology !
                // i.e. not exploiting the boundary curves
                temp_top = top_list.contents(top_list.item(list_pos));
                m_pairs = temp_top.get_m_pairs();
                fsnames = temp_top.get_fsNames();
                for (fsno = 0; fsno < 2 * m_pairs; fsno++)
                {
                    elem_id = fsnames[fsno];
                    elem_name = getname(elem_id);
                    list_pos = elem_name.getpos_in_list();
                    list_type = elem_name.getlist_type();
                    if (list_type == 7) //FACE
                    {
                        temp_fac = face_list.contents(face_list.item(list_pos));
                        temp_fac.set_connectionList(mode);
                        temp_fac.set_trimLoopList(mode);
                        // set fill-up mode if it has not yet been done
                        if (mode != ADD)
                            mode = ADD;

                        m_loops = temp_fac.get_n_trimLoops();
                        trim_loop = temp_fac.get_ConsEnsembles();

                        for (loopno = 0; loopno < m_loops; loopno++)
                        {
                            n_cons = trim_loop[loopno].get_n_cons();
                            consnames = trim_loop[loopno].get_ConsNames();
                            w1 = trim_loop[loopno].get_w1();
                            w2 = trim_loop[loopno].get_w2();

                            for (int conno = 0; conno < n_cons; conno++)
                            {
                                elem_id = consnames[conno];
                                elem_name = getname(elem_id);
                                list_pos = elem_name.getpos_in_list();
                                temp_con = cons_list.contents(cons_list.item(list_pos));
                                temp_con.PartOfCons(w1[conno], w2[conno]);
                            }
                        }

                        elem_id = temp_fac.get_surfnme();
                        elem_name = getname(elem_id);
                        list_pos = elem_name.getpos_in_list();
                        temp_srf = surf_list.contents(surf_list.item(list_pos));
                        temp_srf.MakeNurbsSurface(FACE);
                    }
                    else if (list_type == 5) //SURF
                    {
                        temp_srf = surf_list.contents(surf_list.item(list_pos));
                        temp_srf.MakeNurbsSurface(SURF);
                    }
                    else // ERROR
                    {
                        cerr << "ERROR: incorrect element in VDAFS element TOP " << endl;
                    }
                }
                break;

            default:
                cerr << "ERROR" << endl;
                break;
            }

                //cerr << " x ";
            }
        }

        // set POINTs
        points = create_points(buf);

        // Set nurbs_curves
        nurbs_curves = create_nurbs_curves(buf);

        // Set nurbs_surfaces
        nurbs_surfaces = create_nurbs_surfaces(buf);

        // Set FACEs
        nurbs_faces = create_nurbs_faces(buf);
        mode = DELETE; // reset the position index for the connection and trim loops lists

        // Keep the POINTS and NURBS objects in a set
        nurbs_set[k]->addElement(points);
        nurbs_set[k]->addElement(nurbs_curves);
        nurbs_set[k]->addElement(nurbs_surfaces);
        nurbs_set[k]->addElement(nurbs_faces);
        group_set->addElement(nurbs_set[k]);

        delete points;
        delete nurbs_curves;
        delete nurbs_surfaces;
        delete nurbs_faces;
        delete nurbs_set[k];
    }

    // empty name space list, group and set list
    name_list.clear();
    group_list.clear();
    set_list.clear();

    geometry = new coDoGeometry(GeometryName, group_set);
    if (!geometry->objectOk())
    {
        Covise::sendError("ERROR: creation of data object 'geometry' failed");
        return;
    }

    delete[] nurbs_set;
    delete group_set;
    delete geometry;
}

void setup_POINTS(int &n_pnts)
{
    // set the dimensions of arrays for the COVISE data object coDoPoints
    n_pnts = selectedPointList.length();
}

coDoPoints *create_points(char *name)
{

    char buf[256];
    int i;
    int n_points; // number of points
    float *x, *y, *z;
    Vec3d p;
    coDoPoints *pts = NULL;

    setup_POINTS(n_points);
    sprintf(buf, "%s_%d", name, 1);
    pts = new coDoPoints(buf, n_points);
    if (!pts->objectOk())
    {
        Covise::sendError("ERROR: creation of data object 'pts' failed");
        return NULL;
    }
    pts->getAddresses(&x, &y, &z);

    i = 0;
    forall(p, selectedPointList)
    {
        x[i] = p[0];
        y[i] = p[1];
        z[i] = p[2];
        i++;
    }
    selectedPointList.clear();

    return pts;
}

void setup_NCUC(int &n_elem, int &n_knts, int &n_cpts)
{
    // set the dimensions of arrays for the COVISE data object DO_NurbsCurveCol

    NurbsCurve *cur = new NurbsCurve();

    n_knts = 0;
    n_cpts = 0;
    n_elem = nurbscurveList.length();

    forall(*cur, nurbscurveList)
    {

        n_knts += cur->get_n_knts();
        n_cpts += cur->get_n_cpts();
    }
    delete cur;
}

DO_NurbsCurveCol *create_nurbs_curves(char *name)
{
    int i, j;
    int n_k, n_c;
    int i_k, i_c;
    int n_elements;
    int n_knots;
    int n_cpoints;
    int *kvList;
    int *chList;

    char buf[256];

    float *K, *x, *y, *z, *w;

    Vec4d v;
    NurbsCurve c;
    DO_NurbsCurveCol *ncc = NULL;

    setup_NCUC(n_elements, n_knots, n_cpoints);
    sprintf(buf, "%s_%d", name, 2);
    ncc = new DO_NurbsCurveCol(buf, n_elements, n_knots, n_cpoints);
    if (!ncc->objectOk())
    {
        Covise::sendError("ERROR: creation of data object 'ncc' failed");
        return NULL;
    }
    ncc->getAddresses(&K, &x, &y, &z, &w, &kvList, &chList);
    i = 0;
    i_k = 0;
    i_c = 0;

    forall(c, nurbscurveList)
    {

        kvList[i] = i_k;
        chList[i] = i_c;

        n_k = c.get_n_knts();
        n_c = c.get_n_cpts();

        for (j = 0; j < n_k; j++)
        {
            K[j + i_k] = c.get_knot(j);
            // cout << K[j+i_k] << "\t";
        }
        // cout << endl;
        for (j = 0; j < n_c; j++)
        {
            v = c.get_controlPoint(j);
            x[j + i_c] = v[0];
            y[j + i_c] = v[1];
            z[j + i_c] = v[2];
            w[j + i_c] = v[3];
            // v.output();
        }

        i_k += n_k;
        i_c += n_c;
        i++;

        // Verify knot removal
        //c.MaximumKnotRemoval();
    }
    nurbscurveList.clear();

    return ncc;
}

void setup_NSFC(int &n_elem, int &n_Uknts, int &n_Vknts, int &n_cpts)
{
    // set the dimensions of arrays for the COVISE data object DO_NurbsSurfaceCol

    int Udim, Vdim;
    NurbsSurface *sur = new NurbsSurface();

    n_Uknts = 0;
    n_Vknts = 0;
    n_cpts = 0;
    n_elem = nurbssurfaceList.length();

    forall(*sur, nurbssurfaceList)
    {

        n_Uknts += sur->get_n_Uknts();
        n_Vknts += sur->get_n_Vknts();
        Udim = sur->get_Udim();
        Vdim = sur->get_Vdim();
        n_cpts += Udim * Vdim;
    }
    delete sur;
}

DO_NurbsSurfaceCol *create_nurbs_surfaces(char *name)
{
    int i, j, k;
    int n_c, i_c;
    int n_uk, n_vk;
    int i_uk, i_vk;
    int ud, vd;
    int n_elements;
    int n_Uknots;
    int n_Vknots;
    int n_cpoints;
    int *chList;
    int *ukvList, *vkvList;
    int *udList, *vdList;

    char buf[256];

    float *U, *V, *x, *y, *z, *w;

    Vec4d v;
    NurbsSurface s;

    DO_NurbsSurfaceCol *nsc = NULL;

    setup_NSFC(n_elements, n_Uknots, n_Vknots, n_cpoints);
    sprintf(buf, "%s_%d", name, 3);
    nsc = new DO_NurbsSurfaceCol(buf,
                                 n_elements,
                                 n_Uknots, n_Vknots,
                                 n_cpoints);

    if (!nsc->objectOk())
    {
        Covise::sendError("ERROR: creation of data object 'nsc' failed");
        return NULL;
    }
    nsc->getAddresses(&U, &V,
                      &x, &y, &z, &w,
                      &ukvList, &vkvList,
                      &udList, &vdList,
                      &chList);
    i = 0;
    i_uk = 0;
    i_vk = 0;
    i_c = 0;
    forall(s, nurbssurfaceList)
    {
        ukvList[i] = i_uk;
        vkvList[i] = i_vk;
        chList[i] = i_c;

        n_uk = s.get_n_Uknts();
        n_vk = s.get_n_Vknts();

        ud = s.get_Udim();
        vd = s.get_Vdim();
        udList[i] = ud;
        vdList[i] = vd;
        n_c = ud * vd;

        for (j = 0; j < n_uk; j++)
        {
            U[j + i_uk] = s.get_Uknot(j);
        }
        for (j = 0; j < n_vk; j++)
        {
            V[j + i_vk] = s.get_Vknot(j);
        }
        for (j = 0; j < ud; j++)
            for (k = 0; k < vd; k++)
            {
                v = s.get_controlPoint(j, k);
                x[j * vd + k + i_c] = v[0];
                y[j * vd + k + i_c] = v[1];
                z[j * vd + k + i_c] = v[2];
                w[j * vd + k + i_c] = v[3];
                //cout << j << "," << k << ": ";
                //v.output();
            }

        i_uk += n_uk;
        i_vk += n_vk;
        i_c += n_c;
        i++;
    }
    nurbssurfaceList.clear();

    return nsc;
}

void setup_FACEC(int &n_tLps, int &n_tCvs, int &n_tKnts, int &n_tCpts,
                 int &n_sfcs, int &n_Uknts, int &n_Vknts, int &n_cpts)
{
    // set the dimensions of arrays for the COVISE data object DO_FaceCol

    int Udim, Vdim;

    TrimCurve *tcur = new TrimCurve();
    NurbsSurface *sur = new NurbsSurface();

    // trim curves
    n_tKnts = 0;
    n_tCpts = 0;
    n_tCvs = curveDefList.length();

    forall(*tcur, curveDefList)
    {

        n_tKnts += tcur->get_n_knts();
        n_tCpts += tcur->get_n_cpts();
    }
    delete tcur;

    // surfaces
    n_Uknts = 0;
    n_Vknts = 0;
    n_cpts = 0;
    n_sfcs = surfaceDefList.length();

    forall(*sur, surfaceDefList)
    {

        n_Uknts += sur->get_n_Uknts();
        n_Vknts += sur->get_n_Vknts();
        Udim = sur->get_Udim();
        Vdim = sur->get_Vdim();
        n_cpts += Udim * Vdim;
    }
    delete sur;

    // trim loops
    n_tLps = trimLoopList.length();
}

DO_FaceCol *create_nurbs_faces(char *name)
{
    int i, j, k;
    int n_k, n_c; // number of knots and control points
    // of each trim curve element
    int i_k, i_c; // index items
    int n_uk, n_vk; //
    int i_uk, i_vk;
    int ud, vd;
    int i_tLoop;
    int i_tCurve;

    int n_tLoops; // number of trim loops
    int n_tCurves; // number of trim curves
    int n_tKnots;
    int n_tCpoints;
    int n_surfaces; // number of surfaces
    int n_Uknots;
    int n_Vknots;
    int n_cpoints;
    int *tkvList;
    int *tchList;
    int *ukvList, *vkvList;
    int *udList, *vdList;
    int *chList;
    int *conList;
    int *tLoopList;

    char buf[256];

    float *tK, *tU, *tV, *tW;
    float *U, *V, *x, *y, *z, *w;

    Vec3d v3d;
    Vec4d v4d;
    TrimCurve t;
    NurbsSurface s;

    DO_FaceCol *nfc = NULL;

    setup_FACEC(
        n_tLoops,
        n_tCurves, n_tKnots, n_tCpoints,
        n_surfaces, n_Uknots, n_Vknots, n_cpoints);
    sprintf(buf, "%s_%d", name, 4);
    nfc = new DO_FaceCol(buf,
                         n_tLoops,
                         n_tCurves, n_tKnots, n_tCpoints,
                         n_surfaces, n_Uknots, n_Vknots, n_cpoints);

    if (!nfc->objectOk())
    {
        Covise::sendError("ERROR: creation of data object 'nfc' failed");
        return NULL;
    }
    nfc->getAddresses(
        &tK,
        &tU, &tV, &tW,
        &tkvList, &tchList,
        &U, &V,
        &x, &y, &z, &w,
        &ukvList, &vkvList,
        &udList, &vdList,
        &chList,
        &conList,
        &tLoopList);

    // Set the trim curves building the border of the FACEs
    i = 0;
    i_k = 0;
    i_c = 0;
    forall(t, curveDefList)
    {

        tkvList[i] = i_k;
        tchList[i] = i_c;

        n_k = t.get_n_knts();
        n_c = t.get_n_cpts();

        for (j = 0; j < n_k; j++)
        {
            tK[j + i_k] = t.get_knot(j);
            // cout << tK[j+i_k] << "\t";
        }
        // cout << endl;
        for (j = 0; j < n_c; j++)
        {
            v3d = t.get_controlPoint(j);
            tU[j + i_c] = v3d[0];
            tV[j + i_c] = v3d[1];
            tW[j + i_c] = v3d[2];
            // v3d.output();
        }

        i_k += n_k;
        i_c += n_c;
        i++;
    }
    curveDefList.clear();

    // Set the surfaces defining the FACEs
    i = 0;
    i_uk = 0;
    i_vk = 0;
    i_c = 0;
    forall(s, surfaceDefList)
    {
        ukvList[i] = i_uk;
        vkvList[i] = i_vk;
        chList[i] = i_c;

        n_uk = s.get_n_Uknts();
        n_vk = s.get_n_Vknts();

        ud = s.get_Udim();
        vd = s.get_Vdim();
        udList[i] = ud;
        vdList[i] = vd;
        n_c = ud * vd;

        for (j = 0; j < n_uk; j++)
        {
            U[j + i_uk] = s.get_Uknot(j);
        }
        for (j = 0; j < n_vk; j++)
        {
            V[j + i_vk] = s.get_Vknot(j);
        }
        for (j = 0; j < ud; j++)
            for (k = 0; k < vd; k++)
            {
                v4d = s.get_controlPoint(j, k);
                x[j * vd + k + i_c] = v4d[0];
                y[j * vd + k + i_c] = v4d[1];
                z[j * vd + k + i_c] = v4d[2];
                w[j * vd + k + i_c] = v4d[3];
                //cout << j << "," << k << ": ";
                //v4d.output();
            }
        i_uk += n_uk;
        i_vk += n_vk;
        i_c += n_c;
        i++;
    }
    surfaceDefList.clear();

    // Set the connection list
    i = 0;
    forall(i_tLoop, connectionList)
    {

        if (i < n_surfaces)
            conList[i] = i_tLoop;
        else
        {
            Covise::sendError("ERROR: Error in global connection list");
            break;
        }
        i++;
    }

    connectionList.clear();

    // Set the trimLoop list
    i = 0;
    forall(i_tCurve, trimLoopList)
    {
        if (i < n_tLoops)
            tLoopList[i] = i_tCurve;
        else
        {
            Covise::sendError("ERROR: Error in global trim loop list");
            break;
        }
        i++;
    }
    trimLoopList.clear();

    return nfc;
}
