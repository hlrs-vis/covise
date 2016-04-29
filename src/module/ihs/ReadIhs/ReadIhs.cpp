/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Ihs data         	                      **
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

#define NEW_HEADER

#include "ReadIhs.h"
#include <string.h>

int main(int argc, char *argv[])
{
#ifdef YAC
	coDispatcher *dispatcher = coDispatcher::Instance();
	ReadIhs *application = new ReadIhs(argc, argv);
	dispatcher->add(application);
	while (dispatcher->dispatch(1000));
	coDispatcher::deleteDispatcher();
#else
	// create the module
	ReadIhs *application = new ReadIhs(argc, argv);
	application->start(argc,argv);
#endif
	return(0);
}

ReadIhs::ReadIhs(int argc, char **argv) : coSimpleModule(argc, argv, "Read data from Ihs FENFLOSS")
{
	char buf[300];
	port_mesh = addOutputPort ("mesh","UnstructuredGrid|Polygons","Grid");
	port_velocity = addOutputPort ("velocity","Vec3","velocity");
	port_pressure = addOutputPort ("pressure","Float","pressure");
	port_K = addOutputPort ("K","Float","K");
	port_EPS = addOutputPort ("EPS","Float","EPS");
	port_RHO = addOutputPort ("RHO","Float","RHO");
	port_VLES = addOutputPort ("VLES","Float","VLES");
	port_NUt = addOutputPort ("NUt","Float","Nut");

#ifdef WIN32
   const char *defaultDir = getenv("USERPROFILE");
#else
   const char *defaultDir = getenv("HOME");
#endif
   if(defaultDir)
      sprintf(buf,"%s/",defaultDir);
   else
      sprintf(buf,"/data/");
	grid_path = addFileBrowserParam("grid_path","Grid file path");
	grid_path->setValue(buf,"*.geo;*.GEO");

	data_path = addFileBrowserParam("data_path","Geometry File");
	data_path->setValue(buf,"*sim*;*erg*;*ERG*");

	numt = addInt32Param("numt","Nuber of n_timesteps");
	numt->setValue(1);	 
}



int ReadIhs::compute(const char *)
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
        int *tb2;
        int currt,t,endt=1,gcurrt;
        //int gendt;
	int n_timesteps;
	currt=0;
	int reuseMesh=0;
	bool twoD = false;
	bool newFormat = false;
	int elNum,elType;
	char *pattrib=NULL,*vattrib=NULL,*sattrib=NULL;

	// get parameters
	n_timesteps=numt->getValue();
	strcpy(gp,grid_path->getValue());
	strcpy(dp,data_path->getValue());

	i=strlen(dp)-1;
	while(dp[i] &&((dp[i]<'0')||(dp[i]>'9')))
		i--;
	// dp[i] ist jetzt die letzte Ziffer, alles danach ist Endung
	if(dp[i]) {
		strcpy(dpend,dp+i+1);                       // dpend= Endung;
		dp[i+1]='\0';
	}
	else {
		dpend[0]='\0';
	}

	int numNumbers=0;
	bool zeros=false;
	while((dp[i]>='0')&&(dp[i]<='9'))
    {
		numNumbers++;
		if((dp[i]>='1')&&(dp[i]<='9'))
			zeros=true;
		i--;
    }
	if(dp[i])
    {
		//currt = Aktueller Zeitschritt
		if(sscanf(dp+i+1,"%d",&currt) != 1)
		{
			cerr << "ReadIhs::compute: sscanf1 failed" << endl;
		}
		endt=currt+n_timesteps;
		dp[i+1]=0;                                  // dp = basename
    }
	else
    {
		currt = 0;
    }

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
		gnumNumbers++;
		if((gp[i]>='1')&&(gp[i]<='9'))
			gzeros=true;
		i--;
    }
	if(gp[i])
    {
		//currt = Aktueller Zeitschritt
		if(sscanf(gp+i+1,"%d",&gcurrt) != 1)
		{
			cerr << "ReadIhs::compute: sscanf2 failed" << endl;
		}
                //gendt=gcurrt+n_timesteps;
		gp[i+1]=0;                                  // gp = basename
    }
	else
    {
		gcurrt = 0;
    }

	Mesh =  port_mesh->getObjName();
	Veloc = port_velocity->getObjName();
	Press = port_pressure->getObjName();
	K_name = port_K->getObjName();
	EPS_name = port_EPS->getObjName();
	RHO_name = port_RHO->getObjName();
	//VLES_name = port_VLES->getObjName();
	STR_name = port_NUt->getObjName();

	coDistributedObject **Mesh_sets= new coDistributedObject*[n_timesteps+1];
	coDistributedObject **Veloc_sets= new coDistributedObject*[n_timesteps+1];
	coDistributedObject **Press_sets= new coDistributedObject*[n_timesteps+1];
	coDistributedObject **K_sets= new coDistributedObject*[n_timesteps+1];
	coDistributedObject **EPS_sets= new coDistributedObject*[n_timesteps+1];
	coDistributedObject **RHO_sets= new coDistributedObject*[n_timesteps+1];
	// coDistributedObject **VLES_sets= new coDistributedObject*[n_timesteps+1];
	coDistributedObject **STR_sets= new coDistributedObject*[n_timesteps+1];
	Mesh_sets[0]=NULL;
	Veloc_sets[0]=NULL;
	Press_sets[0]=NULL;
	K_sets[0]=NULL;
	EPS_sets[0]=NULL;
	RHO_sets[0]=NULL;
	//  VLES_sets[0]=NULL;
	STR_sets[0]=NULL;
  
	int gfileNumber= currt;
	int fileNumber= currt;

	grid=NULL;
	veloc=NULL;
	press=NULL;
	K=NULL;
	EPS=NULL;
	RHO=NULL;
	//  VLES=NULL;
	STR=NULL;

	int length;
  
	for(t=currt;t<endt;t++)
	{
		if(!reuseMesh)
		{
			if(n_timesteps>1)
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
					if ((grid_fp = fopen(buf, "r")) != NULL)
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
					strcpy(buf,grid_path->getValue());
					//fprintf(stderr,"DEBUG gp buf=%s\n",gp);
				}
			}
			else
			{
				strcpy(buf,grid_path->getValue());
				//fprintf(stderr,"DEBUG gp buf=%s\n",gp,);
			}
			if ((grid_fp = fopen(buf, "r")) == NULL)
			{
				if(t==currt)
				{
					strcpy(buf2, "ERROR: Can't open file >> ");
					strcat(buf2, buf);
					sendError("%s",buf2);
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
			sendInfo("%s",buf2);
			// get rid of the header
#ifdef NEW_HEADER
			while(true)
			{
				if(fgets(buf,300,grid_fp) == NULL)
				{
					cerr << "ReadIhs::compute: fgets1 failed" << endl;
				}
				/*if(strncasecmp(buf," #Dimension: 2",14)==0)
				{
					twoD = true;
					}*/
				int len=strlen(buf);
				for(int j=0; j < len; j++)
					buf[j] = tolower(buf[j]);
				if (strstr(buf,"#")) {
					if (strstr(buf,"dimension")) {
						int dimension;
						char *buf2=strrchr(buf,':');
						buf2++;
						if((dimension=atoi(buf2))==2) twoD=true;
						
					}
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
				char *test=buf;
				while((*test) && ((*test)==' ' || (*test)=='\t'|| (*test)=='\r'|| (*test)=='\n'))
				{
					test++;
				}
				if(*test)
				{
					if(*test >= '0' && *test <= '9')
						break;                          // we do have a number and no comment, so this is probably the header
				}
			}
#else

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
#endif
			if(sscanf(buf,"%d%d%d%d%d\n",&n_coord,&n_elem,&tmpi,&tmpi,&tmpi) != 5)
			{
				cerr << "ReadIhs::compute: sscanf3 failed" << endl;
			}
#ifdef YAC
			if(Mesh.name != NULL)
#else
				if(Mesh != NULL)
#endif
				{
					tbt=tb=new int[n_coord];
					if(n_timesteps>1)
						sprintf(buf,"%s_%d",Mesh,t);
					else
						strcpy(buf,Mesh);
					//neues Datenobjekt anlegen (buf ist der Name des Objekts,
					// nelem ist die Anzahl der Hexaeder, n_coord ist die Anzahl der Koordinaten
					// die 1 am Ende muss ein! (hasTypelist muss immer true sein)
					grid=NULL;
					if(twoD)
					{
#ifdef YAC
						polygons = new coDoPolygons(grid->getObjName(), n_coord,n_elem*4, n_elem);
#else
						polygons = new coDoPolygons(buf, n_coord,n_elem*4, n_elem);
#endif
						if (polygons->objectOk())
						{
							grid = polygons;
							polygons->addAttribute("vertexOrder", "2");
							polygons->getAddresses(&x_coord,&y_coord,&z_coord,&vl,&el);
							// el = Elementlist
							// vl = Vertexlist
							for(i=0;i<n_coord;i++)
							{
								if(fgets(buf,300,grid_fp)==NULL)
								{
									sendError("ERROR: unexpected end of file");
									return FAIL;
								}
								// Einlesen der Knoten (Koordinaten), tbt ist die Knotennummer
								if(sscanf(buf,"%d%f%f%f\n",tbt,x_coord,y_coord,z_coord) != 4)
								{
									cerr << "ReadIhs::compute: sscanf4 failed" << endl;
								}
								x_coord++;
								y_coord++;
								z_coord++;
								tbt++;

							}
							tmpi=0;
							tbt=tb;
							// herausfinden der groessten Knotennummer
							int istart = 0;
							if(n_coord > 50) istart = n_coord-50;
							for(i=istart;i<n_coord;i++)
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
									sendError("ERROR: unexpected end of file");
									return FAIL;
								}
								if(i==0)
								{
									newFormat = false;
									if(sscanf(buf,"%d%d%d%d%d%d\n",&elNum,vl,vl+1,vl+2,vl+3,&elType)==6)
										newFormat = true;
								}
								if(newFormat)
								{
									if(sscanf(buf,"%d%d%d%d%d%d\n",&elNum,vl,vl+1,vl+2,vl+3,&elType) != 6)
									{
										cerr << "ReadIhs::compute: sscanf5 failed" << endl;
									}
								}
								else
								{
									if(sscanf(buf,"%d%d%d%d\n",vl,vl+1,vl+2,vl+3) != 4)
									{
										cerr << "ReadIhs::compute: sscanf6 failed" << endl;
									}
								}
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
							mesh->getAddresses(&el,&vl,&x_coord,&y_coord,&z_coord);
							mesh->getTypeList(&tl);
							// el = Elementlist
							// vl = Vertexlist
							// tl = Typelist
							for(i=0;i<n_coord;i++)
							{
								if(fgets(buf,300,grid_fp)==NULL)
								{
									sendError("ERROR: unexpected end of file");
									return FAIL;
								}
								// Einlesen der Knoten (Koordinaten), tbt ist die Knotennummer
								if(sscanf(buf,"%d%f%f%f\n",tbt,x_coord,y_coord,z_coord) != 4)
								{
									cerr << "ReadIhs::compute: sscanf7 failed" << endl;
								}
								x_coord++;
								y_coord++;
								z_coord++;
								tbt++;
							}
							tmpi=0;
							tbt=tb;
							// herausfinden der groessten Knotennummer
							int istart=0;
							if(n_coord > 50) istart = n_coord-50;
							for(i=istart;i<n_coord;i++)
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
									sendError("ERROR: unexpected end of file");
									return FAIL;
								}
								if(i==0)
								{
									newFormat = false;
									if(sscanf(buf,"%d%d%d%d%d%d%d%d%d%d\n",&elNum,vl,vl+1,vl+2,vl+3,vl+4,vl+5,vl+6,vl+7,&elType)==10)
										newFormat = true;
								}
								if(newFormat)
								{
									if(sscanf(buf,"%d%d%d%d%d%d%d%d%d%d\n",&elNum,vl,vl+1,vl+2,vl+3,vl+4,vl+5,vl+6,vl+7,&elType) != 10)
									{
										cerr << "ReadIhs::compute: sscanf8 failed" << endl;
									}
								}
								else
								{
									if(sscanf(buf,"%d%d%d%d%d%d%d%d\n",vl,vl+1,vl+2,vl+3,vl+4,vl+5,vl+6,vl+7) != 8)
									{
										cerr << "ReadIhs::compute: sscanf9 failed" << endl;
									}
								}
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
							sendError("ERROR: creation of data object 'mesh' failed");
							return FAIL;
						}
					}
				}
				else
				{
					sendError("ERROR: object name not correct for 'mesh'");
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

		if(n_timesteps>1)
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
#ifdef YAC
				if ((grid_fp = fopen(buf, "r")) != NULL)
#else
					if ((grid_fp = Covise::fopen(buf, "r")) != NULL)
#endif
					{
						fclose(grid_fp);
						break;
					}
				numTries++;

				fileNumber++;
			}
		}
		else
			strcpy(buf,data_path->getValue());
		if ((data_fp = fopen(buf, "r")) == NULL)
		{
			break;
		}

#ifdef NEW_HEADER
		while(true)
		{
			if(fgets(buf,300,data_fp) == NULL)
			{
				cerr << "ReadIhs::compute: fgets2 failed" << endl;
			}
			char *test=buf;
			while((*test) && ((*test)==' ' || (*test)=='\t'|| (*test)=='\r'|| (*test)=='\n'))
			{
				test++;
			}
			if(*test)
			{
				if(*test >= '0' && *test <= '9')
					break;                             // we do have a number and no comment, so this is probably the header
			}
		}

		int erg_head[6];
		int iret = sscanf(buf,"%d%d%d%d%d%d\n",&erg_head[0],&erg_head[1],&erg_head[2],&erg_head[3],&erg_head[4],&erg_head[5]);
		if(iret != 6)
			fprintf(stderr,"ReadIhs::compute(const char *) sscanf error read %d elements \n",iret);
		sprintf(buf2,"%8d%8d%8d%8d%8d%8d\n",
				erg_head[0],erg_head[1],erg_head[2],erg_head[3],erg_head[4],erg_head[5]);
		fprintf(stderr,"ReadIhs::compute(const char *) header: %s\n",buf2);
		sendInfo("%s",buf2);

#else
		// get rid of the header
		for(i=0;i<10;i++)
			fgets(buf,300,data_fp);
		if(newFormat)
			fgets(buf,300,data_fp);
#endif

		sprintf(buf2,"Reading data timestep %d\n",fileNumber);
		sendInfo("%s",buf2);

		int coord_erg = erg_head[0];
		int col_erg = erg_head[3];
      
		if( Veloc != 0)
		{
			if(n_timesteps>1)
				sprintf(buf,"%s_%d",Veloc,t);
			else
				strcpy(buf,Veloc);
			veloc = new coDoVec3(buf, n_coord);
			if (veloc->objectOk())
			{
				veloc->getAddresses(&u,&v,&w);
				if( Press != 0)
				{
					if(n_timesteps>1)
						sprintf(buf,"%s_%d",Press,t);
					else
						strcpy(buf,Press);
					press = new coDoFloat(buf, n_coord);
					if (press->objectOk())
					{
						press->getAddress(&p);
						if( K_name != 0)
						{
							if(n_timesteps>1)
								sprintf(buf,"%s_%d",K_name,t);
							else
								strcpy(buf,K_name);
							K = new coDoFloat(buf, n_coord);
							if (K->objectOk())
							{
								K->getAddress(&k);
								if( EPS_name != 0)
								{
									if(n_timesteps>1)
										sprintf(buf,"%s_%d",EPS_name,t);
									else
										strcpy(buf,EPS_name);
									EPS = new coDoFloat(buf, n_coord);
									if (EPS->objectOk())
									{
										EPS->getAddress(&eps);
										if( RHO_name != 0)
										{
											if(n_timesteps>1)
												sprintf(buf,"%s_%d",RHO_name,t);
											else
												strcpy(buf,RHO_name);

											RHO = new coDoFloat(buf, n_coord);
											if (RHO->objectOk())
											{
												RHO->getAddress(&rho);
												if( STR_name != 0)
												{
													if(n_timesteps>1)
														sprintf(buf,"%s_%d",STR_name,t);
													else
														strcpy(buf,STR_name);

													STR = new coDoFloat(buf, n_coord);
													if (STR->objectOk())
													{
														STR->getAddress(&str);
														for(i=0;i<coord_erg;i++)
														{
															if(fgets(buf,300,data_fp)==NULL)
															{
																sendError("ERROR: unexpected end of file");
																return FAIL;
															}
															if(strlen(buf) > 30)
															{
																length=sscanf(buf,"%d%f%f%f%f%f%f%f%f%f%f%f\n",&tmpi,u,v,w,k,eps,p,rho,&tmpf,&tmpf,str,&tmpf);
																if ( (length != (col_erg+1)) && (length != (col_erg+2)) )
																{
																	cerr << "ReadIhs::compute: sscanf for results failed. " << length << " columns in result file are not allowed " << col_erg << endl;
																}
															}
															else
															{
																if(sscanf(buf,"%d%f\n",&tmpi,p) != 2)
																{
																	cerr << "ReadIhs::compute: sscanf11 failed" << endl;
																}
															}
															u++;
															v++;
															w++;
															k++;
															eps++;
															p++;
															rho++;
															str++;
														} // i
													}
													else
													{
														sendError("ERROR: creation of data object 'STR' failed");
														return FAIL;
													}
												}
												else
												{
													sendError("ERROR: Object name not correct for 'STR'");
													return FAIL;
												}
											}
											else
											{
												sendError("ERROR: creation of data object 'RHO' failed");
												return FAIL;
											}
										}
										else
										{
											sendError("ERROR: Object name not correct for 'RHO'");
											return FAIL;
										}
									}
									else
									{
										sendError("ERROR: creation of data object 'EPS' failed");
										return FAIL;
									}
								}
								else
								{
									sendError("ERROR: Object name not correct for 'EPS'");
									return FAIL;
								}
							}
							else
							{
								sendError("ERROR: creation of data object 'K' failed");
								return FAIL;
							}
						}
						else
						{
							sendError("ERROR: Object name not correct for 'K'");
							return FAIL;
						}
					}
					else
					{
						sendError("ERROR: creation of data object 'pressure' failed");
						return FAIL;
					}
				}
				else
				{
					sendError("ERROR: Object name not correct for 'pressure'");
					return FAIL;
				}

			}
			else
			{
				sendError("ERROR: creation of data object 'velocity' failed");
				return FAIL;
			}
		}
		else
		{
			sendError("ERROR: Object name not correct for 'velocity'");
			return FAIL;
		}
		for(i=0;Mesh_sets[i];i++)
                   ;
		Mesh_sets[i]=grid;
		Mesh_sets[i+1]=NULL;
		if(reuseMesh)
			grid->incRefCount();
		for(i=0;Veloc_sets[i];i++)
                   ;
		Veloc_sets[i]=veloc;
		Veloc_sets[i+1]=NULL;
		for(i=0;Press_sets[i];i++)
                   ;
		Press_sets[i]=press;
		Press_sets[i+1]=NULL;
		for(i=0;K_sets[i];i++)
                   ;
		K_sets[i]=K;
		K_sets[i+1]=NULL;
		for(i=0;EPS_sets[i];i++)
                   ;
		EPS_sets[i]=EPS;
		EPS_sets[i+1]=NULL;
		for(i=0;RHO_sets[i];i++)
                   ;
		RHO_sets[i]=RHO;
		RHO_sets[i+1]=NULL;
		for(i=0;STR_sets[i];i++)
                   ;
		STR_sets[i]=STR;
		STR_sets[i+1]=NULL;

		//timesteps++; //does nothing ...
		fclose(data_fp);

		gfileNumber++;
		fileNumber++;
    }
	
	if(n_timesteps>1)
    {

       // This code doesn't seem to make sense -> if 0
#if 0
		coDoSet *Mesh_set=NULL;
                coDoSet *Veloc_set= NULL;
                coDoSet *Press_set= NULL;
                coDoSet *K_set= NULL;
                coDoSet *EPS_set= NULL;
                coDoSet *RHO_set= NULL;
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
		if(RHO_sets[0])
			RHO_set= new coDoSet(RHO_name,RHO_sets);
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
		for(i=0;RHO_sets[i];i++)
			delete RHO_sets[i];
		delete[] RHO_sets;
		for(i=0;STR_sets[i];i++)
			delete STR_sets[i];
		delete[] STR_sets;
#endif
    }
	else
    {
		delete grid;
		delete veloc;
		delete press;
		delete K;
		delete EPS;
		delete RHO;
		delete STR;
    }

	return SUCCESS;

}
