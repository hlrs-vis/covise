#include "Gen3D.h"
#include <limits.h>

int main(int argc, char *argv[])
{
   // init
   Gen3D *application = new Gen3D;

   // and back to covise
   application->start(argc,argv);

   // done
   return 0;
}


Gen3D::Gen3D()
:coModule("Make a 3D grid out of a 2")
{
   p_inPoly = addInputPort("2D_Grid","Polygons","2D grid polygons");
   p_inVelo = addInputPort("velocity","Vec3","velocity");
   p_inPoly->setRequired(1);
   p_inVelo->setRequired(0);
   p_outGrid = addOutputPort("outGrid","UnstructuredGrid","unstructured Grid");
                                                  // 0 is as many as velocities set
   p_outVelo = addOutputPort("outVelo","Vec3","velocity");
   p_thick = addFloatParam("thick","thickness of 3D-grid");
   p_thick->setValue(0.05);
};

int Gen3D::Diagnose()
{
   const char *dtype;

   if(  p_inPoly->getCurrentObject()->isType("SETELE")  )
   {
      isset=1;
   }

   int no_pl,no_vl,no_points;
   float *xCoords,*yCoords,*zCoords;
   float *xCoordsIn,*yCoordsIn,*zCoordsIn;

   float *u_in,*v_in,*w_in;
   int *clist,*plist,*tlist;

   int nvelo;

   if (isset)
   {
      // ********* input object is a set *********

   }
   else
   {

      /*
        // ********* input object is not a set *********
        polyIn =  (coDoPolygons *) p_inPoly;
        if (!strcmp(polyIn->getType(),"POLYGN"))
        {
           Covise::sendError("ERROR: Data object 'meshIn' has wrong data type");
           return -1;
        }
        no_pl = polyIn->getNumPolygons();
        no_vl = polyIn->getNumVertices();
        no_points = polyIn->getNumPoints();
      if (  (xCoords = new float[no_points])==NULL  )
      cout << "Not enough memory for array xCoords!" << endl;
      if (  (yCoords = new float[no_points])==NULL  )
      cout << "Not enough memory for array yCoords!" << endl;
      if (  (zCoords = new float[no_points])==NULL  )
      cout << "Not enough memory for array zCoords!" << endl;
      polyIn->getAddresses(&xCoordsIn,&yCoordsIn,&zCoordsIn, &clist,&plist);

      if(p_inVelo != 0L)
      {
      veloInObj =  (coDoVec3 *) p_inVelo;
      dtype = veloInObj->getType();
      if(strcmp(dtype, "USTVDT") == 0)
      {
      veloInObj = (coDoVec3 *) p_inVelo;
      nvelo = veloInObj->getNumPoints();
      veloInObj->getAddresses(&u_in, &v_in, &w_in);
      }
      }
      */
   }

   return 0;
}


int Gen3D::compute(const char *)
{
   int i;

   isset = 0;

   if(Diagnose()<0)
      return FAIL;

   int numVelos=0;
   int numGeos=0;
   int numPress=0;

   float *xCoords,*yCoords,*zCoords;
   float *xCoordsIn,*yCoordsIn,*zCoordsIn;

   float *uIn, *vIn, *wIn;

   int *plist,*clist,*vlist,*tlist;
   int no_pl,no_vl,no_points;

   if (isset)
   {

   }
   else                                           // (isset=0)
   {
      polyIn = (coDoPolygons *)p_inPoly->getCurrentObject();

      no_pl = polyIn->getNumPolygons();
      no_vl = polyIn->getNumVertices();
      no_points = polyIn->getNumPoints();

      polyIn->getAddresses(&xCoordsIn,&yCoordsIn,&zCoordsIn,&clist,&plist);

      int nelem = no_pl;
      int nconn = 8*no_pl;
      int ncoord = 2*no_points;

      int *elem;
      int *conn;

      gridOutObj = new coDoUnstructuredGrid(p_outGrid->getObjName(),nelem,nconn,ncoord,1);
      gridOutObj->getAddresses(&elem,&conn,&xCoords,&yCoords,&zCoords);
      gridOutObj->getTypeList(&tlist);

      for(i=0;i<no_pl;i++)
      {
         tlist[i]=TYPE_HEXAGON;
      }

      // element list is easy as we just have hexas ...
      for(i=0;i<no_pl;i++)
      {
         elem[i] = 8*i;
      }

      // connectivity list made of corner list
      for(i=0;i<no_pl;i++)
      {
         conn[8*i+0] = clist[plist[i]+0];
         conn[8*i+1] = clist[plist[i]+1];
         conn[8*i+2] = clist[plist[i]+2];
         conn[8*i+3] = clist[plist[i]+3];
         conn[8*i+4] = clist[plist[i]+0]+no_points;
         conn[8*i+5] = clist[plist[i]+1]+no_points;
         conn[8*i+6] = clist[plist[i]+2]+no_points;
         conn[8*i+7] = clist[plist[i]+3]+no_points;
      }

      // the coordinate list is easy as soon as we know in which direction to expand ...

      // to find out in which coordinate plain (XY, XZ, YZ) the 2D grid lies in,
      // we get a normal vector in the first element
      float x1,x2,x3,y1,y2,y3,z1,z2,z3,nx,ny,nz;
      x1 = xCoordsIn[0]; y1 = xCoordsIn[0]; z1 = zCoordsIn[0];
      x2 = xCoordsIn[1]; y2 = xCoordsIn[1]; z2 = zCoordsIn[1];
      x3 = xCoordsIn[2]; y3 = xCoordsIn[2]; z3 = zCoordsIn[2];
      nx = (fabs) ( 1000*(y2-y1)*(z3-z1)-1000*(z2-z1)*(y3-y1) );
      ny = (fabs) ( 1000*(z2-z1)*(x3-x1)-1000*(x2-x1)*(z3-z1) );
      nz = (fabs) ( 1000*(x2-x1)*(y3-y1)-1000*(y2-y1)*(x3-x1) );

      float min = FLT_MAX;
      int min_axis[3] = {0,0,0};
      if (nx<min)
      {
         min_axis[0]=1;
         min_axis[1]=0;
         min_axis[2]=0;
      }
      if (ny<min)
      {
         min_axis[0]=0;
         min_axis[1]=1;
         min_axis[2]=0;
      }
      if (nz<min)
      {
         min_axis[0]=0;
         min_axis[1]=0;
         min_axis[2]=1;
      }

      // we'd like to have a XY-grid, so we exchange the corrdinates ...
      float *tmp;
      if (min_axis[0]==1)                         //YZ
      {
         tmp=xCoordsIn;
         xCoordsIn=zCoordsIn;
         zCoordsIn=tmp;
      }

      if (min_axis[1]==1)                         //XZ
      {
         tmp=yCoordsIn;
         yCoordsIn=zCoordsIn;
         zCoordsIn=tmp;
      }

      float delta = p_thick->getValue();
      for(i=0;i<no_points;i++)
      {
         zCoords[i]=zCoordsIn[i]-min_axis[2]*delta/2.;
         zCoords[i+no_points]=zCoordsIn[i]+min_axis[2]*delta/2.;

         yCoords[i]=yCoordsIn[i]-min_axis[1]*delta/2.;
         yCoords[i+no_points]=yCoordsIn[i]+min_axis[1]*delta/2.;

         xCoords[i]=xCoordsIn[i]-min_axis[0]*delta/2.;
         xCoords[i+no_points]=xCoordsIn[i]-min_axis[0]*delta/2.;
      }

      p_outGrid->setCurrentObject(gridOutObj);

      // velocity

      veloInObj = (coDoVec3 *)p_inVelo->getCurrentObject();
      no_points = veloInObj->getNumPoints();

      // change of axis to xy!
      veloInObj->getAddresses(&uIn,&vIn,&wIn);

      float *uOut = new float[2*no_points];
      float *vOut = new float[2*no_points];
      float *wOut = new float[2*no_points];

      if (min_axis[0]==1)                         //YZ
      {
         tmp=uIn;
         uIn=wIn;
         wIn=tmp;
      }

      if (min_axis[1]==1)                         //XZ
      {
         tmp=vIn;
         vIn=wIn;
         wIn=tmp;
      }

      for (i=0;i<no_points;i++)
      {
         uOut[i]=uIn[i];
         uOut[i+no_points]=uIn[i];

         vOut[i]=vIn[i];
         vOut[i+no_points]=vIn[i];

         wOut[i]=wIn[i];
         wOut[i+no_points]=wIn[i];
      }

      veloOutObj = new coDoVec3(p_outVelo->getObjName(),2*no_points,uOut,vOut,wOut);
      p_outVelo->setCurrentObject(veloOutObj);

   }

   return SUCCESS;

}
