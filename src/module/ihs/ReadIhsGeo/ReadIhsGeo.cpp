/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Ihs data         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  17.03.95  V1.0                                                  **
 \**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadIhsGeo.h"
int main(int argc, char *argv[])
{

   Application *application = new Application(argc,argv);

   application->run();
   return 0;

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
//
//..........................................................................
//
void Application::quit(void *)
{
   //
   // ...... delete your data here .....
   //

}


int  Application::getPnum(int k)
{
   static int onum=0;
   if(pnum[onum]<=k)
   {
      while(pnum[onum]!=k)
      {
         if(onum<n_coord)
            onum++;
         else
            onum=0;
      }
      return(onum);
   }
   else
   {
      while(pnum[onum]!=k)
      {
         if(onum>=0)
            onum--;
         else
            onum=n_coord-1;
      }
      return(onum);
   }

}


void Application::compute(void *)
{
   //
   // ...... do work here ........
   //

   // read input parameters and data object name
   FILE *grid_fp,*data_fp;
   //float p0x,p0y,p0z,p4x,p4y,p4z,p1x,p1y,p1z,p2x,p2y,p2z,p3x,p3y,p3z, d1x,  d1y,  d1z, d2x,  d2y,  d2z,  nx,  ny,  nz;
   int i,tmpi;
   float tmpf;
   char buf[600];
   char color[600];
   strcpy(color,"DarkSlateGrey");
   pnum=NULL;
   press=NULL;
   char *pattrib=NULL,*vattrib=NULL,*sattrib=NULL;

   Covise::get_browser_param("grid_path", &grid_Path);
   Covise::get_browser_param("data_path", &data_Path);

   Mesh      = Covise::get_object_name("mesh");
   Press      = Covise::get_object_name("pressure");

   if ((grid_fp = Covise::fopen(grid_Path, "r")) == NULL)
   {
      Covise::sendError("ERROR: Can't open file >> %s",grid_Path);
      return;
   }
   if ((data_fp = Covise::fopen(data_Path, "r")) == NULL)
   {
      Covise::sendWarning("WARNING: Can't open data file >> %s",data_Path);
   }

   // get rid of the header
   for(i=0;i<9;i++)
   {
      if (fgets(buf,300,grid_fp)!=NULL)
      {
         fprintf(stderr,"fgets_1 failed in ReadIHS2.cpp");
      }
      if(buf[0]=='P')
      {
         pattrib=new char[strlen(buf)+1];
         strcpy(pattrib,buf+2);
      }
      if(buf[0]=='V')
      {
         vattrib=new char[strlen(buf)+1];
         strcpy(vattrib,buf+2);
      }
      if(buf[0]=='S')
      {
         sattrib=new char[strlen(buf)+1];
         strcpy(sattrib,buf+2);
      }
   }
   strcpy(color,buf);
   if (fgets(buf,300,grid_fp)!=NULL)
   {
      fprintf(stderr,"fgets_2 failed in ReadIHS2.cpp");
   }
   if(data_fp)
   {
      for(i=0;i<10;i++)
         if (fgets(buf,300,data_fp)!=NULL)
         {
            fprintf(stderr,"fgets_3 failed in ReadIHS2.cpp");
         }
   }

   // now read in Dimensions
   if (fgets(buf,600,grid_fp)!=NULL)
   {
      fprintf(stderr,"fgets_4 failed in ReadIHS2.cpp");
   }
   sscanf(buf,"%d%d",&n_coord,&n_elem);

   if(Mesh != NULL)
   {
      mesh = new coDoPolygons(Mesh, n_coord,n_elem*4, n_elem);
      pnum=new int[n_coord];
      if (mesh->objectOk())
      {
         mesh->addAttribute("vertexOrder", "2");
         mesh->addAttribute("COLOR", color);
         if(pattrib)
            mesh->addAttribute("ROTATE_POINT",pattrib);
         if(vattrib)
         {
            mesh->addAttribute("ROTATE_VECTOR",vattrib);
         }
         if(sattrib)
            mesh->addAttribute("ROTATE_SPEED",sattrib);
         mesh->getAddresses(&x_coord,&y_coord,&z_coord,&vl,&el);
         for(i=0;i<n_coord;i++)
         {
            if(fscanf(grid_fp,"%d%f%f%f\n",pnum+i,x_coord,y_coord,z_coord)==EOF)
            {
               Covise::sendError("ERROR: unexpected end of file");
               return;
            }
            x_coord++;
            y_coord++;
            z_coord++;

         }
         for(i=0;i<n_elem;i++)
         {
            if(fscanf(grid_fp,"%d%d%d%d\n",vl,vl+1,vl+2,vl+3)==EOF)
            {
               Covise::sendError("ERROR: unexpected end of file");
               return;
            }
            *vl = getPnum(*vl);
            vl++;
            *vl = getPnum(*vl);
            vl++;
            *vl = getPnum(*vl);
            vl++;
            *vl = getPnum(*vl);
            vl++;
            *el++=i*4;
         }
      }
      else
      {
         Covise::sendError("ERROR: creation of data object 'mesh' failed");
         return;
      }
   }
   else
   {
      Covise::sendError("ERROR: object name not correct for 'mesh'");
      return;
   }

   fclose(grid_fp);

   if(( Press != 0)&&(data_fp))
   {
      press = new coDoFloat(Press, n_coord);
      if (press->objectOk())
      {

         press->getAddress(&p);
         for(i=0;i<n_coord;i++)
         {
            do
            {
               if (fgets(buf,300,data_fp)!=NULL)
               {
                  fprintf(stderr,"fgets_5 failed in ReadIHS2.cpp");
               }
               if(sscanf(buf,"%d%f%f%f%f%f%f",&tmpi,&tmpf,&tmpf,&tmpf,&tmpf,&tmpf,p)==EOF)
               {
                  Covise::sendError("ERROR: unexpected end of file");
                  return;
               }
            }
            while(tmpi != pnum[i]);
            p++;
         }

      }
      else
      {
         Covise::sendError("ERROR: creation of data object 'pressure' failed");
         return;
      }
   }
   if(data_fp)
      fclose(data_fp);
   if(press)
      delete press;
   delete[] pnum;

}
