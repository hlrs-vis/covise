struct geometry
{
   int isgeo;                                     // is there a geofile?	(0/1)
   int isrb;                                      // is there a rbfile?	(0/1)
   int issim;                                     // is there a simfile?	(0/1)

   int new_rbfile;                                // new (=1) or old (=0) rb-file?

   int is3d;                                      // 3D or 2D geometry? (0/1)

   int create_boco_object;

   int numbered;                                  // numbered connectivity list?  (0/1)

   int bilanr1;
   int bilanr2;
   int bilanr3;
   int bilanr4;
   int bilanr5;

   int bila2nr1;
   int bila2nr2;
   int bila2nr3;
   int bila2nr4;
   int bila2nr5;

   int wallnr1;
   int wallnr2;
   int wallnr3;
   int wallnr4;
   int wallnr5;

   int pressnr1;
   int pressnr2;
   int pressnr3;
   int pressnr4;
   int pressnr5;

   char geofile[300];
   char rbfile[300];
   char simfile[300];

   // numbers
   int n_nodes;
   int knmaxnr;
   int elmaxnr;
   
   int n_elem_3d;                        // connectivity list 3d-geometry
   int n_elem_2d;                        // connectivity list 2d-geometry
   int n_wall;
   int n_special_wall;
   int n_wall_less;
   int n_elem_mark;                      // 3D: 4 nodes each, 2D: 2 nodes each
   int n_elem_mark2;                     // 3D: 4 nodes each, 2D: 2 nodes each
   int n_special_mark;                   // Anzahl Markierungen Port 1 Gesamt
   int n_special_mark2;                  // Anzahl Markierungen Port 2 Gesamt
   int n_in_rb;                          // Knotenrandbedingung Eintritt
   int n_press_rb;                       // Druckrandbedingungen
   int n_syme;                           // Symmetrie-Randbedingungen
   int n_kmark;

   // coordinates
   float scalingfactor;
   float *x;
   float *y;
   float *z;
   float *r;

   // connectivity
   int *elem_3d;
   int *elem_2d;

   // rb
   int *elem_mark;                                // Port 1
   int *elem_mark2;                               // Port 2
   int *special_mark;                             // Port 1
   int *special_mark2;                            // Port 2
   int *wall;
   int *special_wall;
   int *press_rb;
   float *press_rb_value;

   // cfd
   float *u;
   float *v;
   float *w;
   float *p;
   float *k;
   float *eps;
   float *p_elem;

   // boco
   int *bcin;
   int *bcout;
   int *bcperiodic1;
   int *bcperiodic2;
   int n_bcin;
   int n_bcout;
   int n_bcperiodic1;
   int n_bcperiodic2;
   int bcinnr;
   int bcoutnr;
   int bcperiodicnr1;
   int bcperiodicnr2;

   int *dirichlet_nodes;
   float *dirichlet_values;

   // bilanrlist
   int *bilanrlist;
   char **bilanames;
   int n_bilanrs;

   // wallnrlist
   int *wallnrlist;
   char **wallnames;
   int n_wallnrs;

   // presnrlist
   int *presnrlist;
   char **presnames;
   int n_presnrs;


};
