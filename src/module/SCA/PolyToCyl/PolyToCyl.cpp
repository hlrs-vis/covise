/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                    (C) 2000 VirCinity  **
 ** Description: transform polygon in the x-y-plane to a cylinder          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Name:        AssembleUsg                                               **
 ** Category:    Tools                                                     **
 **                                                                        **
 ** Author: Sven Kufer		                                          **
 **         (C)  VirCinity IT- Consulting GmbH                             **
 **         Nobelstrasse 15                                                **
 **         D- 70569 Stuttgart    			       	          **
 **                                                                        **
 **  Date:5.7.2002                                                                       **
 **                                                                        **
\**************************************************************************/

#include <iostream>
#include "PolyToCyl.h"
#include <do/coDoSet.h>

void
PolyToCyl::postInst()
{
    hparams_.push_back(h_barrelDiameter = new coHideParam(p_barrelDiameter));
}

void
PolyToCyl::preHandleObjects(coInputPort **inPorts)
{
    coInputPort *geometry = inPorts[0];
    int param;
    for (param = 0; param < hparams_.size(); ++param)
    {
        hparams_[param]->reset();
    }
    useTransformAttribute(geometry->getCurrentObject());
}

void
PolyToCyl::useTransformAttribute(const coDistributedObject *inGeo)
{
    const char *wert;
    if (inGeo == NULL)
    {
        return;
    }
    wert = inGeo->getAttribute("POLY_TO_CYL");
    if (wert == NULL) // perhaps the attribute is hidden in a set structure
    {
        if (inGeo->isType("SETELE"))
        {
            int no_elems;
            const coDistributedObject *const *setList = ((const coDoSet *)(inGeo))->getAllElements(&no_elems);
            int elem;
            for (elem = 0; elem < no_elems; ++elem)
            {
                useTransformAttribute(setList[elem]);
            }
        }
        return;
    }
    std::istringstream pvalues(wert);
    char *value = new char[strlen(wert) + 1];
    while (pvalues.getline(value, strlen(wert) + 1))
    {
        int param;
        for (param = 0; param < hparams_.size(); ++param)
        {
            hparams_[param]->load(value);
        }
    }
    delete[] value;
}

PolyToCyl::PolyToCyl()
    : coSimpleModule(0, 0, "transform polygon to a cylinder")
{

    // ports
    p_poly_in = addInputPort("poly_in", "Set_Lines|Set_Polygons", "polymetry");
    p_box_in = addInputPort("box_in", "coDoLines", "polymetry");
    p_poly_out = addOutputPort("poly_out", "Set_Lines|Set_Polygons", "polymetry");
    p_surf_out = addOutputPort("surf_out", "Set_Polygons", "surf");
    p_inlet = addOutputPort("inlet", "Set_Polygons", "inlet");

    p_torus = addBooleanParam("create_torus", "whether a torus shall be created");
    p_torus->setValue(0);
    p_barrelDiameter = addFloatParam("barrelDiameter", "width of the torus");
    p_barrelDiameter->setValue(10.6);
    p_factor = addInt32Param("factor", "multiple every line point by this factor");
    p_factor->setValue(50);
    p_usefactor = addBooleanParam("use_factor", "whether factor shall be used");
    p_usefactor->setValue(0);
}

int main(int argc, char *argv[])
{
    PolyToCyl *application = new PolyToCyl;
    application->start(argc, argv);
    return 0;
}

////// this is called whenever we have to do something
int
fcompar(const void *f1, const void *f2)
{
    float n1 = *((float *)(f1) + 1);
    float n2 = *((float *)(f2) + 1);
    if (n1 < n2)
        return -1;
    else if (n2 < n1)
        return 1;
    return 0;
}

int PolyToCyl::compute(const char *port)
{
    (void)port; // silence compiler

    const char *trans_paper = "MAT: trans.paper 0.343256 0.351851 0.351852 0.361321 0.370368 0.37037 0.398148 0.398148 0.398148 0.212963 0.212963 0.212963 0 0.1";

    const char *paper = "MAT: paper 0.343256 0.351851 0.351852 0.361321 0.370368 0.37037 0.398148 0.398148 0.398148 0.212963 0.212963 0.212963 0 0";

    Torus *torus = NULL, *torus1 = NULL, *inlet = NULL;

    if (p_box_in->getCurrentObject() == NULL || p_poly_in->getCurrentObject() == NULL)
    {
        Covise::sendWarning("Not all objects avaiable");
        return CONTINUE_PIPELINE;
    }

    bool no_transform = true;
    if (p_poly_in->getCurrentObject()->getAttribute("POLY_TO_CYL") != NULL)
    {
        no_transform = false; // only copy incoming polygon
    }

    bool makeTorus;
    if (p_torus->getValue())
    {
        makeTorus = true;
    }
    else
    {
        makeTorus = false;
    }

    //
    // get reference point out of the outer lines of a bounding box
    //

    coDoLines *box = (coDoLines *)p_box_in->getCurrentObject();

    float *x_box, *y_box, *z_box;
    int *dummy;

    float xmin, xmax, ymin, ymax, zmin, zmax;

    box->getAddresses(&x_box, &y_box, &z_box, &dummy, &dummy);

    getBoundingBox(&xmin, &xmax, &ymin, &ymax, &zmin, &zmax, x_box, y_box, z_box, box->getNumLines());

#ifdef DEBUG
    cerr << xmin << " " << xmax << " " << ymin << " " << ymax << " " << zmin << " " << zmax << endl;
#endif

    //
    //  get original polygon
    //
    coDoPolygons *in_p = NULL;
    coDoLines *in_l = NULL;
    coDistributedObject *out;

    if (p_poly_in->getCurrentObject()->isType("POLYGN"))
    {
        in_p = (coDoPolygons *)p_poly_in->getCurrentObject();
    }
    else if (p_poly_in->getCurrentObject()->isType("LINES"))
    {
        in_l = (coDoLines *)p_poly_in->getCurrentObject();
    }
    else
    {
        sendError("Received wrong type at grid port");
        return STOP_PIPELINE;
    }

    float *x_in, *y_in, *z_in;
    int *vl_in, *pl_in;
    int numVert;
    int i;

    if (in_p != NULL)
    {
        in_p->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
        numVert = in_p->getNumPoints();
    }
    else
    {
        if (no_transform || !p_usefactor->getValue())
        {
            in_l->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
            numVert = in_l->getNumPoints();
        }
        else
        {
            char objname[256];
            sprintf(objname, "%s_tmp", p_poly_out->getObjName());
            // multiple line points by FACTOR
            in_l = discreteLine(objname, in_l);
            //p_poly_out->setCurrentObject( discreteLine(p_poly_out->getObjName(), in_l) );
            //return CONTINUE_PIPELINE;
            in_l->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
            numVert = in_l->getNumPoints();
        }
    }

    if (no_transform)
    {
        if (in_p != NULL)
        {
            out = new coDoPolygons(p_poly_out->getObjName(), in_p->getNumPoints(), x_in, y_in, z_in,
                                   in_p->getNumVertices(), vl_in, in_p->getNumPolygons(), pl_in);
            out->addAttribute("MATERIAL", trans_paper);
        }
        else
        {
            out = new coDoLines(p_poly_out->getObjName(), in_l->getNumPoints(), x_in, y_in, z_in,
                                in_l->getNumVertices(), vl_in, in_l->getNumLines(), pl_in);
            out->addAttribute("COLOR", "yellow");
        }
    }

    else
    {
        float *x, *y, *z;
        int *vl, *pl;

        if (in_p != NULL)
        {
            out = new coDoPolygons(p_poly_out->getObjName(), in_p->getNumPoints(),
                                   in_p->getNumVertices(), in_p->getNumPolygons());
            out->addAttribute("MATERIAL", paper);
            ((coDoPolygons *)out)->getAddresses(&x, &y, &z, &vl, &pl);
        }
        else
        {
            out = new coDoLines(p_poly_out->getObjName(), in_l->getNumPoints(),
                                in_l->getNumVertices(), in_l->getNumLines());
            out->addAttribute("COLOR", "yellow");
            ((coDoLines *)out)->getAddresses(&x, &y, &z, &vl, &pl);
        }

        int num_el = (in_p != NULL) ? in_p->getNumPolygons() : in_l->getNumLines();
        int num_cl = (in_p != NULL) ? in_p->getNumVertices() : in_l->getNumVertices();

        for (i = 0; i < num_el; i++)
        {
            pl[i] = pl_in[i];
        }

        for (i = 0; i < num_cl; i++)
        {
            vl[i] = vl_in[i];
        }

        for (i = 0; i < numVert; i++)
        {
            x[i] = x_in[i];
            y[i] = y_in[i];
            z[i] = z_in[i];
        }

        if (makeTorus && !no_transform)
        {
            torus = new Torus();
            torus1 = new Torus();
            inlet = new Torus();
        }

        float h = ymax - ymin; // height of polygon
        if (fabs(h) <= 1.0e-20)
        {
            Covise::sendError("Height of polygon=0!");
            return STOP_PIPELINE;
        }
        float r = h / (2 * M_PI); // base radius of cylinder
        if (!p_torus->getValue())
        {
            r = 0.5 * h_barrelDiameter->getFValue();
        }
        float y0, z0; //x0,

        float *yOrder = new float[3 * numVert];
        for (i = 0; i < numVert; i++)
        {
            yOrder[3 * i] = x_in[i] - xmin;
            yOrder[3 * i + 1] = y_in[i];
            yOrder[3 * i + 2] = z_in[i];
        }
        qsort(yOrder, numVert, 3 * sizeof(float), fcompar);

        for (i = 0; i < numVert; i++)
        {
            //x0 = x[i]-xmin;
            y0 = y[i] - ymin;
            z0 = z[i] - zmin;

            y[i] = (r + z0) * cos(2 * M_PI * y0 / h);
            z[i] = (r + z0) * sin(2 * M_PI * y0 / h);
            /*
                x[i] = xmin + (r+y0) * cos( 2*M_PI*x0/h );
                y[i] = ymin + (r+y0) * sin( 2*M_PI*x0/h );
                z[i] = zmin + z0;
         */
            /*
                if( makeTorus && fabs(z[i])<=1.0e-20 ) { // calc outer surface of the cylinder

                   inlet->addVertex( x0+xmin, z0+zmin, ymin-p_torus_perc->getValue()*r, x0+xmin, zmax, ymax-p_torus_perc->getValue()*r);
                 // outer surface
                   torus->addVertex( x0+xmin, z0+zmin, ymax, x0+xmin, z0+zmin, ymin-p_torus_perc->getValue()*r );
                }
         */
            if (makeTorus && fabs(yOrder[3 * i]) <= 1.0e-6)
            {
                inlet->addVertex(xmin, yOrder[3 * i + 1],
                                 zmin - r + 0.5 * h_barrelDiameter->getFValue(),
                                 xmax, yOrder[3 * i + 1],
                                 zmin - r + 0.5 * h_barrelDiameter->getFValue());
                torus->addVertex(xmin, yOrder[3 * i + 1],
                                 yOrder[3 * i + 2], // z_in[i],
                                 xmin, yOrder[3 * i + 1],
                                 zmin - r + 0.5 * h_barrelDiameter->getFValue());
            }
            if (makeTorus && fabs(xmax - yOrder[3 * i]) <= 1.0e-6)
            {
                torus1->addVertex(xmax, yOrder[3 * i + 1],
                                  yOrder[3 * i + 2],
                                  xmax, yOrder[3 * i + 1],
                                  zmin - r + 0.5 * h_barrelDiameter->getFValue());
            }
        }
        delete[] yOrder;
    }

    if (makeTorus && !no_transform && torus && torus1)
    {
        torus->reOrder();
        torus1->reOrder();
    }

    p_poly_out->setCurrentObject(out);

    char surf_name[512];
    coDistributedObject *surf_set[3];

    if (makeTorus)
    {
        char buf[64];
        if (no_transform)
        {
            p_surf_out->setCurrentObject(new coDoPolygons(p_surf_out->getObjName(), 0, 0, 0));
        }
        else
        {
            sprintf(surf_name, "%s_top", p_surf_out->getObjName());
            coDoPolygons *top = torus->getObject(surf_name);

            sprintf(surf_name, "%s_bottom", p_surf_out->getObjName());
            sprintf(buf, "barrelDiameter %f", h_barrelDiameter->getFValue());
            surf_set[0] = top;
            surf_set[0]->addAttribute("MATERIAL", paper);
            surf_set[0]->addAttribute("POLY_TO_CYL", buf);
            surf_set[1] = torus1->getObject(surf_name);
            //surf_set[1] = movePolygons(top, 0, zmax-zmin, 0, surf_name);
            surf_set[1]->addAttribute("MATERIAL", paper);
            surf_set[1]->addAttribute("POLY_TO_CYL", buf);
            surf_set[2] = NULL;

            p_surf_out->setCurrentObject(new coDoSet(p_surf_out->getObjName(), surf_set));
            delete torus;
            delete torus1;
        }

        if (no_transform)
        {
            p_inlet->setCurrentObject(new coDoPolygons(p_inlet->getObjName(), 0, 0, 0));
        }
        else
        {
            coDoPolygons *pol_inlet = inlet->getObject(p_inlet->getObjName());
            pol_inlet->addAttribute("MATERIAL", paper);
            pol_inlet->addAttribute("POLY_TO_CYL", buf);
            p_inlet->setCurrentObject(pol_inlet);
            delete inlet;
        }
    }
    // create always objects for the output ports!!!!
    if (p_poly_out->getCurrentObject() == NULL)
    {
        p_poly_out->setCurrentObject(new coDoPolygons(p_surf_out->getObjName(), 0, 0, 0));
    }
    if (p_surf_out->getCurrentObject() == NULL)
    {
        p_surf_out->setCurrentObject(new coDoPolygons(p_surf_out->getObjName(), 0, 0, 0));
    }
    if (p_inlet->getCurrentObject() == NULL)
    {
        p_inlet->setCurrentObject(new coDoPolygons(p_inlet->getObjName(), 0, 0, 0));
    }
    return CONTINUE_PIPELINE;
}

coDoLines *
PolyToCyl::discreteLine(const char *name,
                        coDoLines *in)
{
    float *x_in, *y_in, *z_in;
    int *vl_in, *ll_in;

    float *x, *y, *z;
    int *vl, *ll;

    int i, j, k;
    int FACTOR = p_factor->getValue();

    int num_ll = 0;
    int num_vl = 0;
    int num_points = 0;
    int numVert = in->getNumVertices();

    x = new float[numVert * FACTOR];
    y = new float[numVert * FACTOR];
    z = new float[numVert * FACTOR];

    ll = new int[in->getNumLines()];
    vl = new int[in->getNumVertices() * FACTOR];

    in->getAddresses(&x_in, &y_in, &z_in, &vl_in, &ll_in);
    int next_line, next_corner;

    for (i = 0; i < in->getNumLines(); i++)
    {
        next_line = (i == in->getNumLines() - 1) ? in->getNumVertices() : ll_in[i + 1];
        //cerr << "Line " << num_ll << endl;
        ll[num_ll++] = num_vl;
        for (j = ll_in[i]; j < next_line - 1; j++)
        {

            //cerr << vl_in[j] << endl;
            for (k = 0; k < FACTOR; k++)
            {
                vl[num_vl++] = num_points;
                next_corner = (j == in->getNumVertices() - 1) ? in->getNumVertices() : vl_in[j + 1];
                x[num_points] = x_in[vl_in[j]] + ((float)k / (FACTOR - 1)) * (x_in[next_corner] - x_in[vl_in[j]]);
                y[num_points] = y_in[vl_in[j]] + ((float)k / (FACTOR - 1)) * (y_in[next_corner] - y_in[vl_in[j]]);
                z[num_points] = z_in[vl_in[j]] + ((float)k / (FACTOR - 1)) * (z_in[next_corner] - z_in[vl_in[j]]);
                //cerr << x[num_points] << " " << y[num_points] << " " <<z[num_points] << " " <<endl;
                num_points++;
            }
            //	 cerr << endl;
        }
    }
    //cerr << num_ll << " " << num_vl << " " << num_points << endl;

    coDoLines *ret = new coDoLines(name, num_points, x, y, z, num_vl, vl, num_ll, ll);
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] ll;
    delete[] vl;
    return (ret);
}

void
PolyToCyl::getBoundingBox(float *xmin,
                          float *xmax,
                          float *ymin,
                          float *ymax,
                          float *zmin,
                          float *zmax,
                          const float *x,
                          const float *y,
                          const float *z,
                          int numVert)
{
    *xmin = FLT_MAX;
    *xmax = -FLT_MAX;
    *ymin = FLT_MAX;
    *ymax = -FLT_MAX;
    *zmin = FLT_MAX;
    *zmax = -FLT_MAX;

    for (int i = 0; i < numVert; i++)
    {
        if (x[i] < *xmin)
        {
            *xmin = x[i];
        }
        if (x[i] > *xmax)
        {
            *xmax = x[i];
        }
        if (y[i] < *ymin)
        {
            *ymin = y[i];
        }
        if (y[i] > *ymax)
        {
            *ymax = y[i];
        }
        if (z[i] < *zmin)
        {
            *zmin = z[i];
        }
        if (z[i] > *zmax)
        {
            *zmax = z[i];
        }
    }
}

Torus::Torus()
{
    num_ = 0;
}

void
Torus::addVertex(float xo, float yo, float zo, float xi, float yi, float zi)
{
    Point *po = new Point(xo, yo, zo);
    Point *pi = new Point(xi, yi, zi);

    outer_line_.push_back(po);
    inner_line_.push_back(pi);

    ++num_;
}

coDoPolygons *
Torus::getObject(const char *obj_name)
{
    coDoPolygons *p = NULL;
    if (num_ > 1)
    {
        p = new coDoPolygons(obj_name, (num_ - 1) * 4, (num_ - 1) * 4, (num_ - 1) * 1);
    }
    else
    {
        p = new coDoPolygons(obj_name, 0, 0, 0);
    }
    p->addAttribute("vertexOrder", "2");

    float *x, *y, *z;
    int *pl, *cl;

    p->getAddresses(&x, &y, &z, &cl, &pl);

    int pol;
    //int num_cl=0, num_p=0;

    for (pol = 0; pol < num_ - 1; ++pol)
    {
        pl[pol] = 4 * pol;
        cl[4 * pol] = 2 * pol;
        cl[4 * pol + 1] = 2 * pol + 1;
        cl[4 * pol + 2] = 2 * pol + 3;
        cl[4 * pol + 3] = 2 * pol + 2;
        x[2 * pol] = inner_line_[pol]->x();
        y[2 * pol] = inner_line_[pol]->y();
        z[2 * pol] = inner_line_[pol]->z();
        x[2 * pol + 1] = outer_line_[pol]->x();
        y[2 * pol + 1] = outer_line_[pol]->y();
        z[2 * pol + 1] = outer_line_[pol]->z();
    }
    if (num_ < 2)
    {
        return p;
    }
    x[2 * pol] = inner_line_[pol]->x();
    y[2 * pol] = inner_line_[pol]->y();
    z[2 * pol] = inner_line_[pol]->z();
    x[2 * pol + 1] = outer_line_[pol]->x();
    y[2 * pol + 1] = outer_line_[pol]->y();
    z[2 * pol + 1] = outer_line_[pol]->z();

    /*
      for( j=0; j<1; j++ ) {
         for( i=0; i<num_-1; i++ ) {
       pl[i+j*(num_-1)] = num_cl;

       x[num_p] = outer_line_[i]->x();
       y[num_p] = outer_line_[i]->y();
       z[num_p] = outer_line_[i]->z();

       cl[num_cl++]= num_p;
       num_p++;

   x[num_p] = outer_line_[(i+1)%num_]->x();
   y[num_p] = outer_line_[(i+1)%num_]->y();
   z[num_p] = outer_line_[(i+1)%num_]->z();

   cl[num_cl++]= num_p;
   num_p++;

   x[num_p] = inner_line_[(i+1)%num_]->x();
   y[num_p] = inner_line_[(i+1)%num_]->y();
   z[num_p] = inner_line_[(i+1)%num_]->z();

   cl[num_cl++]= num_p;
   num_p++;

   x[num_p] = inner_line_[i]->x();
   y[num_p] = inner_line_[i]->y();
   z[num_p] = inner_line_[i]->z();

   cl[num_cl++]= num_p;
   num_p++;
   }
   }
   */
    /*
   for( i=0; i<num_-1; i++ ) { // inner outlet
       pl[i+2*j*(num_-1)] = num_cl;

       cl[num_cl++] = i+2;
       cl[num_cl++] = i+3;
       cl[num_cl++] = num_-1+i+2;
       cl[num_cl++] = num_-1+i+3;
   }*/
    return p;
}

coDoPolygons *
PolyToCyl::movePolygons(coDoPolygons *in,
                        float mx,
                        float my,
                        float mz,
                        const char *outname)
{
    float *x, *y, *z;
    float *x_in, *y_in, *z_in;
    int *vl, *pl;

    in->getAddresses(&x_in, &y_in, &z_in, &vl, &pl);

    x = new float[in->getNumPoints()];
    y = new float[in->getNumPoints()];
    z = new float[in->getNumPoints()];

    for (int i = 0; i < in->getNumPoints(); i++)
    {
        x[i] = x_in[i] + mx;
        y[i] = y_in[i] + my;
        z[i] = z_in[i] + mz;
    }

    coDoPolygons *out = new coDoPolygons(outname, in->getNumPoints(), x, y, z,
                                         in->getNumVertices(), vl,
                                         in->getNumPolygons(), pl);
    out->addAttribute("vertexOrder", "2");

    delete[] x;
    delete[] y;
    delete[] z;

    return out;
}

int
compar(const void *left, const void *right)
{
    Point **lfp = (Point **)left;
    Point **rgp = (Point **)right;
    Point *lf = *lfp;
    Point *rg = *rgp;
    if (lf->y() - rg->y() < 0.0)
    {
        return -1;
    }
    else if (lf->y() - rg->y() > 0.0)
    {
        return 1;
    }
    return 0;
}

Torus::~Torus()
{
    int i;
    for (i = 0; i < outer_line_.size(); ++i)
    {
        delete outer_line_[i];
        delete inner_line_[i];
    }
}

void
Torus::reOrder()
{
    qsort(&inner_line_[0], inner_line_.size(), sizeof(Point *), compar);
    qsort(&outer_line_[0], outer_line_.size(), sizeof(Point *), compar);
}
