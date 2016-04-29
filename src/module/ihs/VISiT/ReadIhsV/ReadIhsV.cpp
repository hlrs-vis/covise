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

//#include "ApplInterface.h"
#include "ReadIhsV.h"
int main(int argc, char *argv[])
{

   Application *application = new Application(argc,argv);

   application->start(argc, argv);
   return 0;

}


void Application::postInst()
{
   const char *p;
   // rei: ist voellig dreckig und nur fuer die DEMO Voith !!
   const char *x = "/mnt/fs2/projekte/visit/covise/src/application/ihs/VISiT.orig/RadialRunner/interactive_radial_demo7";

   fprintf(stderr, "Entering READIHS::postInst\n");

   sprintf(fifofilein, "%s/tmp.fenfloss.%d.fifo.in", x, getpid());
   if (mkfifo(fifofilein, 0770))
   {
      fprintf(stderr, "Named pipe file (%s) creation failed: %d (%s)\n",
         fifofilein, errno, strerror(errno));
   }
   if ((fifoin = fopen(fifofilein, "r+")) != NULL)
   {
      addSocket(fileno(fifoin));
   }
   else
      fprintf(stderr, "WARNING: Socketfile=%s: errno=%d (%s)\n",
         fifofilein, errno, strerror(errno));

   fprintf(stderr, "Leaving READIHS::postInst\n");
}


void Application::sockData(int soc)
{
   char cmd[65536];
   int len;

   len = read(soc,cmd,65535);
   if (len < 1)
   {
      fprintf(stderr, "ERROR: len=%d\n", len);
   }
   else
   {
      cmd[len] = '\0';
      fprintf(stderr, "Signal from FenflossShell arrived (%s)\n", cmd);
      // Einlesen und Anzeigen der Ergebnisse ...
      fprintf(stderr, "SockCommand done\n");
   }
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


int Application::compute(void *)
{
   //
   // ...... do work here ........
   //

   // read input parameters and data object name
   FILE *grid_fp,*data_fp;
   int i,tmpi;
   float tmpf;
   char buf[600];
   char buf2[600];
   char dp[400];
   char dpend[100];
   char gp[400];
   char gpend[100];
   int *tb, *tbt;
   int *tb2,numt,currt,t,endt,gcurrt,gendt;
   currt=0;
   int timesteps=0;
   int reuseMesh=0;
   bool twoD = false;
   bool newFormat = false;
   int elNum,elType;
   char *pattrib=NULL,*vattrib=NULL,*sattrib=NULL;

   Covise::get_browser_param("grid_path", &grid_Path);
   Covise::get_browser_param("data_path", &data_Path);
   Covise::get_scalar_param("numt", &numt);
   strcpy(dp,data_Path);
   i=strlen(dp)-1;
   while(dp[i] &&((dp[i]<'0')||(dp[i]>'9')))
      i--;
   // dp[i] ist jetzt die letzte Ziffer, alles danach ist Endung
   if(dp[i])
   {
      strcpy(dpend,dp+i+1);                       // dpend= Endung;
      dp[i+1]='\0';
   }
   else
   {
      dpend[0]='\0';
   }
   int numNumbers=0;
   bool zeros=false;
   while((dp[i]>='0')&&(dp[i]<='9'))
   {
      i--;
      numNumbers++;
      if((dp[i]>='1')&&(dp[i]<='9'))
         zeros=true;
   }
   if(dp[i])
   {
      sscanf(dp+i+1,"%d",&currt);                 //currt = Aktueller Zeitschritt
      endt=currt+numt;
      dp[i+1]=0;                                  // dp = basename
   }
   else
   {
      currt = 0;
   }

   strcpy(gp,grid_Path);
   i=strlen(gp)-1;
   while(gp[i] &&((gp[i]<'0')||(gp[i]>'9')))
      i--;
   // gp[i] ist jetzt die letzte Ziffer, alles danach ist Endung
   if(gp[i])
   {
      strcpy(gpend,gp+i+1);                       // dpend= Endung;
      gp[i+1]='\0';
   }
   else
   {
      gpend[0]='\0';
   }

   int gnumNumbers=0;
   bool gzeros=false;
   while((gp[i]>='0')&&(gp[i]<='9'))
   {
      i--;
      gnumNumbers++;
      if((gp[i]>='1')&&(gp[i]<='9'))
         gzeros=true;
   }
   if(gp[i])
   {
      sscanf(gp+i+1,"%d",&gcurrt);                //currt = Aktueller Zeitschritt
      gendt=gcurrt+numt;
      gp[i+1]=0;                                  // gp = basename
   }
   else
   {
      gcurrt = 0;
   }

   Mesh       = Covise::get_object_name("mesh");
   Veloc      = Covise::get_object_name("velocity");
   Press      = Covise::get_object_name("pressure");
   K_name     = Covise::get_object_name("K");
   EPS_name   = Covise::get_object_name("EPS");
   B_U_name   = Covise::get_object_name("B_U");
   STR_name   = Covise::get_object_name("NUt");

   coDistributedObject **Mesh_sets= new coDistributedObject*[numt+1];
   coDistributedObject **Veloc_sets= new coDistributedObject*[numt+1];
   coDistributedObject **Press_sets= new coDistributedObject*[numt+1];
   coDistributedObject **K_sets= new coDistributedObject*[numt+1];
   coDistributedObject **EPS_sets= new coDistributedObject*[numt+1];
   coDistributedObject **B_U_sets= new coDistributedObject*[numt+1];
   coDistributedObject **STR_sets= new coDistributedObject*[numt+1];
   Mesh_sets[0]=NULL;
   Veloc_sets[0]=NULL;
   Press_sets[0]=NULL;
   K_sets[0]=NULL;
   EPS_sets[0]=NULL;
   B_U_sets[0]=NULL;
   STR_sets[0]=NULL;

   int gfileNumber= currt;
   int fileNumber= currt;

   for(t=currt;t<endt;t++)
   {

      if(!reuseMesh)
      {
         if(numt>1)
         {
            int numTries=0;
            while(numTries<100)
            {
               if(gzeros)
               {
                  sprintf(buf,"%s%0*d%s",gp,gnumNumbers,gfileNumber,gpend);
                  //fprintf(stderr,"Opening file %s\n",buf);
               }
               else
                  sprintf(buf,"%s%d%s",gp,gfileNumber,gpend);
               if ((grid_fp = Covise::fopen(buf, "r")) != NULL)
               {
                  fclose(grid_fp);
                  break;
               }
               numTries++;
               gfileNumber++;
            }
            if(numTries>99)
            {
               if(t!=currt)
               {
                  reuseMesh=1;
               }
               strcpy(buf,grid_Path);
            }
         }
         else
            strcpy(buf,grid_Path);

         if ((grid_fp = Covise::fopen(buf, "r")) == NULL)
         {
            if(t==currt)
            {
               strcpy(buf2, "ERROR: Can't open file >> ");
               strcat(buf2, buf);
               Covise::sendError(buf2);
               return FAIL;
            }
            else
            {
               reuseMesh=1;
            }
         }
      }
      if(!reuseMesh)
      {

         sprintf(buf2,"Reading grid timestep %d\n",gfileNumber);
         Covise::sendInfo(buf2);
         // get rid of the header
         for(i=0;i<10;i++)
         {
            fgets(buf,300,grid_fp);
            if(strncasecmp(buf," #Dimension: 2",14)==0)
            {
               twoD = true;
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
            if(buf[0]=='A')
            {
               sattrib=new char[strlen(buf)+1];
               strcpy(sattrib,buf+2);
            }
         }
         // now read in Dimensions
         fgets(buf,300,grid_fp);
         sscanf(buf,"%d%d%d%d%d\n",&n_coord,&n_elem,&tmpi,&tmpi,&tmpi);

         if(Mesh != NULL)
         {
            tbt=tb=new int[n_coord];
            if(numt>1)
               sprintf(buf,"%s_%d",Mesh,t);
            else
               strcpy(buf,Mesh);
            //neues Datenobjekt anlegen (buf ist der Name des Objekts,
            // nelem ist die Anzahl der Hexaeder, n_coord ist die Anzahl der Koordinaten
            // die 1 am Ende muss ein! (hasTypelist muss immer true sein)
            grid=NULL;
            if(twoD)
            {
               polygons = new coDoPolygons(buf, n_coord,n_elem*4, n_elem);
               if (polygons->objectOk())
               {
                  grid = polygons;
                  polygons->addAttribute("vertexOrder", "2");
                  polygons->get_adresses(&x_coord,&y_coord,&z_coord,&vl,&el);
                  // el = Elementlist
                  // vl = Vertexlist
                  for(i=0;i<n_coord;i++)
                  {
                     if(fgets(buf,300,grid_fp)==NULL)
                     {
                        Covise::sendError("ERROR: unexpected end of file");
                        return FAIL;
                     }
                     // Einlesen der Knoten (Koordinaten), tbt ist die Knotennummer
                     sscanf(buf,"%d%f%f%f\n",tbt,x_coord,y_coord,z_coord);
                     x_coord++;
                     y_coord++;
                     z_coord++;
                     tbt++;

                  }
                  tmpi=0;
                  tbt=tb;
                  // herausfinden der groessten Knotennummer
                  for(i=n_coord-50;i<n_coord;i++)
                     if(tb[i]>tmpi)
                        tmpi=tb[i];
                  tb2=new int[tmpi+1];
                  for(i=0;i<n_coord;i++)
                  {
                     tb2[*tbt]=i;
                     tbt++;
                  }
                  // tb2[kn] enthaelt jetzt zur knotennummer kn den entsprechenden Index in die Koordinatenliste
                  // dieser Schritt ist nur notwendig, falls die Knotennummern nicht fortlaufend sind

                  // Einlesen der Elemente (Vierecke in diesem Fall)
                  for(i=0;i<n_elem;i++)
                  {
                     if(fgets(buf,300,grid_fp)==NULL)
                     {
                        Covise::sendError("ERROR: unexpected end of file");
                        return FAIL;
                     }
                     if(i==0)
                     {
                        newFormat = false;
                        if(sscanf(buf,"%d%d%d%d%d%d\n",&elNum,vl,vl+1,vl+2,vl+3,&elType)==6)
                           newFormat = true;
                     }
                     if(newFormat)
                        sscanf(buf,"%d%d%d%d%d%d\n",&elNum,vl,vl+1,vl+2,vl+3,&elType);
                     else
                        sscanf(buf,"%d%d%d%d\n",vl,vl+1,vl+2,vl+3);
                     *vl = tb2[*vl];
                     vl++;
                     *vl = tb2[*vl];
                     vl++;
                     *vl = tb2[*vl];
                     vl++;
                     *vl = tb2[*vl];
                     vl++;

                     // es git nur Hexaeder, daher immer um 8 weiter
                     *el++=i*4;

                  }
                  delete[] tb2;
                  delete[] tb;
               }

            }
            else
            {
               mesh = new coDoUnstructuredGrid(buf, n_elem,n_elem*8, n_coord, 1);
               if (mesh->objectOk())
               {
                  grid = mesh;
                  mesh->get_adresses(&el,&vl,&x_coord,&y_coord,&z_coord);
                  mesh->getTypeList(&tl);
                  // el = Elementlist
                  // vl = Vertexlist
                  // tl = Typelist
                  for(i=0;i<n_coord;i++)
                  {
                     if(fgets(buf,300,grid_fp)==NULL)
                     {
                        Covise::sendError("ERROR: unexpected end of file");
                        return FAIL;
                     }
                     // Einlesen der Knoten (Koordinaten), tbt ist die Knotennummer
                     sscanf(buf,"%d%f%f%f\n",tbt,x_coord,y_coord,z_coord);
                     x_coord++;
                     y_coord++;
                     z_coord++;
                     tbt++;

                  }
                  tmpi=0;
                  tbt=tb;
                  // herausfinden der groessten Knotennummer
                  for(i=n_coord-50;i<n_coord;i++)
                     if(tb[i]>tmpi)
                        tmpi=tb[i];
                  tb2=new int[tmpi+1];
                  for(i=0;i<n_coord;i++)
                  {
                     tb2[*tbt]=i;
                     tbt++;
                  }
                  // tb2[kn] enthaelt jetzt zur knotennummer kn den entsprechenden Index in die Koordinatenliste
                  // dieser Schritt ist nur notwendig, falls die Knotennummern nicht fortlaufend sind

                  // Einlesen der Elemente (Hexaeder in diesem Fall)
                  for(i=0;i<n_elem;i++)
                  {
                     if(fgets(buf,300,grid_fp)==NULL)
                     {
                        Covise::sendError("ERROR: unexpected end of file");
                        return FAIL;
                     }
                     if(i==0)
                     {
                        newFormat = false;
                        if(sscanf(buf,"%d%d%d%d%d%d%d%d%d%d\n",&elNum,vl,vl+1,vl+2,vl+3,vl+4,vl+5,vl+6,vl+7,&elType)==10)
                           newFormat = true;
                     }
                     if(newFormat)
                        sscanf(buf,"%d%d%d%d%d%d%d%d%d%d\n",&elNum,vl,vl+1,vl+2,vl+3,vl+4,vl+5,vl+6,vl+7,&elType);
                     else
                        sscanf(buf,"%d%d%d%d%d%d%d%d\n",vl,vl+1,vl+2,vl+3,vl+4,vl+5,vl+6,vl+7);
                     *vl = tb2[*vl];
                     vl++;
                     *vl = tb2[*vl];
                     vl++;
                     *vl = tb2[*vl];
                     vl++;
                     *vl = tb2[*vl];
                     vl++;
                     *vl = tb2[*vl];
                     vl++;
                     *vl = tb2[*vl];
                     vl++;
                     *vl = tb2[*vl];
                     vl++;
                     *vl = tb2[*vl];
                     vl++;

                     // es git nur Hexaeder, daher immer um 8 weiter
                     *el++=i*8;

                     // alles Hexaeder
                     *tl++=TYPE_HEXAEDER;

                  }
                  delete[] tb2;
                  delete[] tb;
               }
               else
               {
                  Covise::sendError("ERROR: creation of data object 'mesh' failed");
                  return FAIL;
               }
            }
         }
         else
         {
            Covise::sendError("ERROR: object name not correct for 'mesh'");
            return FAIL;
         }

         if(pattrib)
         {
            grid->addAttribute("ROTATION_POINT",pattrib);
         }
         if(vattrib)
         {
            grid->addAttribute("ROTATION_AXIS",vattrib);
         }
         if(sattrib)
         {
            grid->addAttribute("FRAME_ANGLE",sattrib);
         }

         fclose(grid_fp);

      }

      if(numt>1)
      {
         int numTries=0;
         while(numTries<100)
         {
            if(zeros)
            {
               sprintf(buf,"%s%0*d%s",dp,numNumbers,fileNumber,dpend);
               //fprintf(stderr,"Opening file %s\n",buf);
            }
            else
               sprintf(buf,"%s%d%s",dp,fileNumber,dpend);
            if ((grid_fp = Covise::fopen(buf, "r")) != NULL)
            {
               fclose(grid_fp);
               break;
            }
            numTries++;

            fileNumber++;
         }
      }
      else
         strcpy(buf,data_Path);

      if ((data_fp = Covise::fopen(buf, "r")) == NULL)
      {
         break;
      }

      // get rid of the header
      for(i=0;i<10;i++)
         fgets(buf,300,data_fp);
      if(newFormat)
         fgets(buf,300,data_fp);

      sprintf(buf2,"Reading data timestep %d\n",fileNumber);
      Covise::sendInfo(buf2);

      if( Veloc != 0)
      {
         if(numt>1)
            sprintf(buf,"%s_%d",Veloc,t);
         else
            strcpy(buf,Veloc);
         veloc = new coDoVec3(buf, n_coord);
         if (veloc->objectOk())
         {
            veloc->get_adresses(&u,&v,&w);
            if( Press != 0)
            {
               if(numt>1)
                  sprintf(buf,"%s_%d",Press,t);
               else
                  strcpy(buf,Press);
               press = new coDoFloat(buf, n_coord);
               if (press->objectOk())
               {
                  press->get_adress(&p);
                  if( K_name != 0)
                  {
                     if(numt>1)
                        sprintf(buf,"%s_%d",K_name,t);
                     else
                        strcpy(buf,K_name);
                     K = new coDoFloat(buf, n_coord);
                     if (K->objectOk())
                     {
                        K->get_adress(&k);
                        if( EPS_name != 0)
                        {
                           if(numt>1)
                              sprintf(buf,"%s_%d",EPS_name,t);
                           else
                              strcpy(buf,EPS_name);
                           EPS = new coDoFloat(buf, n_coord);
                           if (EPS->objectOk())
                           {
                              EPS->get_adress(&eps);
                              if( B_U_name != 0)
                              {
                                 if(numt>1)
                                    sprintf(buf,"%s_%d",B_U_name,t);
                                 else
                                    strcpy(buf,B_U_name);

                                 B_U = new coDoFloat(buf, n_coord);
                                 if (B_U->objectOk())
                                 {
                                    B_U->get_adress(&b_u);
                                    if( STR_name != 0)
                                    {
                                       if(numt>1)
                                          sprintf(buf,"%s_%d",STR_name,t);
                                       else
                                          strcpy(buf,STR_name);

                                       STR = new coDoFloat(buf, n_coord);
                                       if (STR->objectOk())
                                       {
                                          STR->get_adress(&str);
                                          for(i=0;i<n_coord;i++)
                                          {
                                             if(fgets(buf,300,data_fp)==NULL)
                                             {
                                                Covise::sendError("ERROR: unexpected end of file");
                                                return FAIL;
                                             }
                                             if(strlen(buf) > 30)
                                                sscanf(buf,"%d%f%f%f%f%f%f%f%f%f%f%f\n",&tmpi,u,v,w,k,eps,p,&tmpf,&tmpf,b_u,str,&tmpf);
                                             else
                                                sscanf(buf,"%d%f\n",&tmpi,p);
                                             u++;
                                             v++;
                                             w++;
                                             k++;
                                             eps++;
                                             p++;
                                             b_u++;
                                             str++;
                                          }
                                       }
                                       else
                                       {
                                          Covise::sendError("ERROR: creation of data object 'STR' failed");
                                          return FAIL;
                                       }
                                    }
                                    else
                                    {
                                       Covise::sendError("ERROR: Object name not correct for 'STR'");
                                       return FAIL;
                                    }
                                 }
                                 else
                                 {
                                    Covise::sendError("ERROR: creation of data object 'B_U' failed");
                                    return FAIL;
                                 }
                              }
                              else
                              {
                                 Covise::sendError("ERROR: Object name not correct for 'B_U'");
                                 return FAIL;
                              }
                           }
                           else
                           {
                              Covise::sendError("ERROR: creation of data object 'EPS' failed");
                              return FAIL;
                           }
                        }
                        else
                        {
                           Covise::sendError("ERROR: Object name not correct for 'EPS'");
                           return FAIL;
                        }
                     }
                     else
                     {
                        Covise::sendError("ERROR: creation of data object 'K' failed");
                        return FAIL;
                     }
                  }
                  else
                  {
                     Covise::sendError("ERROR: Object name not correct for 'K'");
                     return FAIL;
                  }
               }
               else
               {
                  Covise::sendError("ERROR: creation of data object 'pressure' failed");
                  return FAIL;
               }
            }
            else
            {
               Covise::sendError("ERROR: Object name not correct for 'pressure'");
               return FAIL;
            }
         }
         else
         {
            Covise::sendError("ERROR: creation of data object 'velocity' failed");
            return FAIL;
         }
      }
      else
      {
         Covise::sendError("ERROR: Object name not correct for 'velocity'");
         return FAIL;
      }
      for(i=0;Mesh_sets[i];i++);
      Mesh_sets[i]=grid;
      Mesh_sets[i+1]=NULL;
      if(reuseMesh)
         grid->incRefCount();
      for(i=0;Veloc_sets[i];i++);
      Veloc_sets[i]=veloc;
      Veloc_sets[i+1]=NULL;
      for(i=0;Press_sets[i];i++);
      Press_sets[i]=press;
      Press_sets[i+1]=NULL;
      for(i=0;K_sets[i];i++);
      K_sets[i]=K;
      K_sets[i+1]=NULL;
      for(i=0;EPS_sets[i];i++);
      EPS_sets[i]=EPS;
      EPS_sets[i+1]=NULL;
      for(i=0;B_U_sets[i];i++);
      B_U_sets[i]=B_U;
      B_U_sets[i+1]=NULL;
      for(i=0;STR_sets[i];i++);
      STR_sets[i]=STR;
      STR_sets[i+1]=NULL;

      timesteps++;
      fclose(data_fp);

      gfileNumber++;
      fileNumber++;
   }
   if(numt>1)
   {
      coDoSet *Mesh_set=NULL;
      coDoSet *Veloc_set= NULL;
      coDoSet *Press_set= NULL;
      coDoSet *K_set= NULL;
      coDoSet *EPS_set= NULL;
      coDoSet *B_U_set= NULL;
      coDoSet *STR_set= NULL;
      Mesh_set= new coDoSet(Mesh,Mesh_sets);
      if(Veloc_sets[0])
         Veloc_set= new coDoSet(Veloc,Veloc_sets);
      if(Press_sets[0])
         Press_set= new coDoSet(Press,Press_sets);
      if(K_sets[0])
         K_set= new coDoSet(K_name,K_sets);
      if(EPS_sets[0])
         EPS_set= new coDoSet(EPS_name,EPS_sets);
      if(B_U_sets[0])
         B_U_set= new coDoSet(B_U_name,B_U_sets);
      if(STR_sets[0])
         STR_set= new coDoSet(STR_name,STR_sets);
      Mesh_set->addAttribute("TIMESTEP","1 100");
      delete Mesh_sets[0];
      delete[] Mesh_sets;
      for(i=0;Veloc_sets[i];i++)
         delete Veloc_sets[i];
      delete[] Veloc_sets;
      for(i=0;Press_sets[i];i++)
         delete Press_sets[i];
      delete[] Press_sets;
      for(i=0;K_sets[i];i++)
         delete K_sets[i];
      delete[] K_sets;
      for(i=0;EPS_sets[i];i++)
         delete EPS_sets[i];
      delete[] EPS_sets;
      for(i=0;B_U_sets[i];i++)
         delete B_U_sets[i];
      delete[] B_U_sets;
      for(i=0;STR_sets[i];i++)
         delete STR_sets[i];
      delete[] STR_sets;
   }
   else
   {
      delete grid;
      delete veloc;
      delete press;
      delete K;
      delete EPS;
      delete B_U;
      delete STR;
   }
   return SUCCESS;
}
