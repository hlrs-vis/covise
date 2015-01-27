/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* ------------------------------------------------------------------ 
 *
 *  animconvert.c:      
 *
 * ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Standard includes                                                  */
/* ------------------------------------------------------------------ */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Own includefiles                                                   */
/* ------------------------------------------------------------------ */
#include "anim.h"

/* ------------------------------------------------------------------ */
/* Prototypes                                                         */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv);
void convert_color(void);
int fget_line(char *, FILE *);
void initialize(int argc, char **argv);
void convert_simdata(void);
void convert_geoall(void);
void convert_elgeoall(void);
void convert_geofile(int no_first_node);
void convert_elgeofile(int no_first_node, int ntimesteps);
void convert_shift_geofile(void);
int strcnt(char *, int);
int index_to_color_polygon(int index);
int index_to_color_frame(int index);

/* ------------------------------------------------------------------ */
/* Definition of global variables                                     */
/* ------------------------------------------------------------------ */
char infile[MAXLENGTH], outfile[MAXLENGTH];
int todo;

/* ------------------------------------------------------------------ */
/* Subroutines                                                        */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    initialize(argc, argv);

    switch (todo)
    {

    case 'a':
        convert_simdata();
        break;

    case 'c':
        convert_color();
        break;

    case 'e':
        convert_elgeoall();
        break;

    case 'g':
        convert_geoall();
        break;

    case 's':
        convert_shift_geofile();
        break;
    }

    return (ERRORFREE);
}

/* ------------------------------------------------------------------ */
/* extern definitions (only for initialize) */
extern int optind;
extern char *optarg;

void initialize(int argc, char **argv)
{
    int option; /* command line option */

    if (argc != 4)
    {
        printf("usage: animconvert -[caegsh] infile outfile\n"
               "       -a ... convert simulation datafile (.str)\n");
        printf("       -c ... convert color file (.cmp)\n");
        printf("       -e ... convert elastic geometry files (.elgeoall and .elgeo)\n");
        printf("       -g ... convert geometry files (.geoall and .geo)\n");
        printf("       -s ... convert geometry files (.geo), allows numbering\n"
               "              to start at specific node number\n");
        exit(ERROR);
    }

    strcpy(infile, argv[2]);
    strcpy(outfile, argv[3]);

    while ((option = getopt(argc, argv, "acegsh")) != EOF)
    {
        switch (option)
        {

        case 'a':
            printf("converts old simulation datafile %s to new file %s\n",
                   infile, outfile);
            todo = 'a';
            break;

        case 'c':
            printf("converts old colorfile %s to new colorfile %s\n",
                   infile, outfile);
            todo = 'c';
            break;

        case 'e':
            printf("\nconverts old elastic geofile %s to new elastic geofile %s\n"
                   "        (name must be a elgeoall file)\n\n",
                   infile, outfile);
            todo = 'e';
            break;

        case 'g':
            printf("\nconverts old geofile %s to new geofile %s\n"
                   "        (name must be a geoall file)\n\n",
                   infile, outfile);
            todo = 'g';
            break;

        case 's':
            printf("\nconverts old geofile %s to new geofile %s\n"
                   "node numbering starts at specific number\n\n",
                   infile, outfile);
            todo = 's';
            break;

        case 'h':
            printf("usage: animconvert -[cagsh] infile outfile\n"
                   "       -a ... convert simulation datafile (.str)\n");
            printf("       -c ... convert color file (.cmp)\n");
            printf("       -e ... convert elastic geometry files (.elgeoall and .elgeo)\n");
            printf("       -g ... convert geometry files (.geoall and .geo)\n");
            printf("       -s ... convert geometry files (.geo), allows numbering\n"
                   "              to start at specific node number\n");
            break;
        }
    }
}

/* ------------------------------------------------------------------ */
void convert_simdata(void)
{
    int i;
    char *lne = NULL;
    float dt, value;
    int timesteps, nobodies;
    FILE *fpout, *fpin;

    /* determine number of non-void lines in file */
    fpin = fopen(infile, "r");
    OWN_CALLOC(lne, char, MAXLENGTH);
    timesteps = 0;
    while (fget_line(lne, fpin) != ERROR)
    {
        timesteps++;
    }
    (void)fclose(fpin);
    timesteps--; /* headerline */

    printf("How many bodies are in file %s?\n", infile);
    scanf("%d", &nobodies);
    timesteps = timesteps / nobodies;
    printf("%d timesteps for %d bodies to convert\n"
           "(keep in mind that you are converting 16 matrix elements to 12 matrix elements)\n",
           timesteps, nobodies);
    fflush(stdout);

    /* open file */
    if ((fpin = fopen(infile, "r")) == NULL)
    {
        printf("... unable to open the file %s\n", infile);
        exit(ERROR);
    }
    if ((fpout = fopen(outfile, "w")) == NULL)
    {
        printf("... unable to open the file %s\n", outfile);
        exit(ERROR);
    }

    /* read the animation datafile */
    (void)fgets(lne, MAXLENGTH, fpin);
    sscanf(lne, "%f", &dt);
    fprintf(fpout, "%f %d 12\n", dt, timesteps);

    for (i = 0; i < timesteps * nobodies; i++)
    {
        fscanf(fpin, "%f", &value);
        fprintf(fpout, "%f ", value);
        fscanf(fpin, "%f", &value);
        fprintf(fpout, "%f ", value);
        fscanf(fpin, "%f", &value);
        fprintf(fpout, "%f ", value);
        fscanf(fpin, "%f", &value);

        fscanf(fpin, "%f", &value);
        fprintf(fpout, "%f ", value);
        fscanf(fpin, "%f", &value);
        fprintf(fpout, "%f ", value);
        fscanf(fpin, "%f", &value);
        fprintf(fpout, "%f ", value);
        fscanf(fpin, "%f", &value);

        fscanf(fpin, "%f", &value);
        fprintf(fpout, "%f ", value);
        fscanf(fpin, "%f", &value);
        fprintf(fpout, "%f ", value);
        fscanf(fpin, "%f", &value);
        fprintf(fpout, "%f ", value);
        fscanf(fpin, "%f", &value);

        fscanf(fpin, "%f", &value);
        fprintf(fpout, "%f ", value);
        fscanf(fpin, "%f", &value);
        fprintf(fpout, "%f ", value);
        fscanf(fpin, "%f", &value);
        fprintf(fpout, "%f", value);
        fscanf(fpin, "%f", &value);

        fprintf(fpout, "\n");
    }

    (void)fclose(fpin);
    (void)fclose(fpout);
}

/* ------------------------------------------------------------------ */
void convert_color(void)
{
    int col, nr = 0; /* count colors */
    float red, green, blue;
    char *lne = NULL; /* line of text */
    FILE *fpin, *fpout;

    /* open file */
    if ((fpin = fopen(infile, "r")) == NULL)
    {
        printf("... unable to open the file %s\n", infile);
        exit(ERROR);
    }
    if ((fpout = fopen(outfile, "w")) == NULL)
    {
        printf("... unable to open the file %s\n", outfile);
        exit(ERROR);
    }

    /* read colormap-file */
    OWN_CALLOC(lne, char, MAXLENGTH);
    do
    {
        if (fget_line(lne, fpin) != ERROR)
        {
            sscanf(lne, "%d %f %f %f", &col, &red, &green, &blue);
            fprintf(fpout, "%d %f %f %f\n", col, red / 255, green / 255, blue / 255);
            nr++;
        }
    } while (!feof(fpin));

    printf("%d colors converted\n", nr);

    (void)fclose(fpin);
    (void)fclose(fpout);
    OWN_FREE(lne);
}

/* ------------------------------------------------------------------ */
void convert_elgeoall(void)
{
    int i, nfiles, ntimesteps;
    char *lne = NULL, fname[MAXLENGTH];
    FILE *fpout, *fpin;

    /* open .elgeoall files */
    if ((fpin = fopen(infile, "r")) == NULL)
    {
        printf("... unable to open the file %s\n", infile);
        exit(ERROR);
    }
    if ((fpout = fopen(outfile, "w")) == NULL)
    {
        printf("... unable to open the file %s\n", outfile);
        exit(ERROR);
    }

    /* read/write header line */
    OWN_CALLOC(lne, char, MAXLENGTH);
    if (fget_line(lne, fpin) != ERROR)
    {
        if (sscanf(lne, "%d %d", &nfiles, &ntimesteps) != 2)
        {
            printf("... problems reading the elgeoall file %s\n", infile);
            printf("    (reason maybe: are nfiles and ntimesteps not on one line?)\n");
            exit(ERROR);
        }
        fprintf(fpout, "%d %d\n", nfiles, ntimesteps);
    }
    else
    {
        printf("...error while reading header line of %s\n", infile);
        exit(ERROR);
    }
    OWN_FREE(lne);

    /* read/write file names; convert files */
    for (i = 0; i < nfiles; i++)
    {
        if (fscanf(fpin, "%s", fname) == EOF)
        {
            printf("... error while reading the elgeoall file %s (elgeofile %d)\n",
                   infile, i + 1);
            exit(ERROR);
        }
        fprintf(fpout, "new_%s\n", fname);

        /* convert geofile */
        strcpy(infile, fname);
        strcpy(outfile, "new_");
        strcat(outfile, fname);

        printf(" converting file %s to %s\n", infile, outfile);
        convert_elgeofile(0, ntimesteps);
    }

    (void)fclose(fpin);
    (void)fclose(fpout);
}

/* ------------------------------------------------------------------ */
void convert_geoall(void)
{
    int i, nfiles;
    char *lne = NULL, fname[MAXLENGTH];
    FILE *fpout, *fpin;

    /* open .geoall files */
    if ((fpin = fopen(infile, "r")) == NULL)
    {
        printf("... unable to open the file %s\n", infile);
        exit(ERROR);
    }
    if ((fpout = fopen(outfile, "w")) == NULL)
    {
        printf("... unable to open the file %s\n", outfile);
        exit(ERROR);
    }

    /* read/write header line */
    OWN_CALLOC(lne, char, MAXLENGTH);
    if (fget_line(lne, fpin) != ERROR)
    {
        sscanf(lne, "%d", &nfiles);
        fprintf(fpout, "%d\n", nfiles);
    }
    else
    {
        printf("...error while reading header line of %s\n", infile);
        exit(ERROR);
    }
    OWN_FREE(lne);

    /* read/write file names; convert files */
    for (i = 0; i < nfiles; i++)
    {
        if (fscanf(fpin, "%s", fname) == EOF)
        {
            printf("... error while reading the geoall file %s (geofile %d)\n",
                   infile, i + 1);
            exit(ERROR);
        }
        fprintf(fpout, "new_%s\n", fname);

        /* convert geofile */
        strcpy(infile, fname);
        strcpy(outfile, "new_");
        strcat(outfile, fname);

        printf(" converting file %s to %s\n", infile, outfile);
        convert_geofile(0);
    }

    (void)fclose(fpin);
    (void)fclose(fpout);
}

/* ------------------------------------------------------------------ */
void convert_shift_geofile(void)
{
    int firstnode;
    printf("    enter number of first node: ");
    /*(void)fflush(stdout)*/;
    scanf("%d", &firstnode);
    printf("\n converting file %s to %s, node numbering starts at %d\n", infile, outfile, firstnode);
    firstnode -= 1;

    convert_geofile(firstnode);
}

/* ------------------------------------------------------------------ */
void convert_geofile(int no_first_node)
{
    int nv, /* number of vertices in open file */
        nconec, /* number of connections in open file */
        j, n, nf, nftmp, nvf,
        entry, cf, cp, *face = NULL;
    char *lne = NULL;
    FILE *fpout, *fpin,
        *tmpfp = NULL; /* pointer on temporary file */
    anim_vector *vertex = NULL; /* coordinates of vertices*/

    /* open geofiles */
    if ((fpin = fopen(infile, "r")) == NULL)
    {
        printf("... unable to open the file %s\n", infile);
        exit(ERROR);
    }
    if ((fpout = fopen(outfile, "w")) == NULL)
    {
        printf("... unable to open the file %s\n", outfile);
        exit(ERROR);
    }

    /* read geofile; (the new file can't be written until number of
     faces was determined) */
    OWN_CALLOC(lne, char, MAXLENGTH);

    if (fget_line(lne, fpin) == ERROR)
    {
        printf("... error while reading the geofile %s\n",
               infile);
        OWN_FREE(lne);
        exit(ERROR);
    }

    if (sscanf(lne, "%d %d", &nv, &nconec) != 2)
    {
        printf("... error while reading header line of geofile %s\n",
               infile);
        OWN_FREE(lne);
        exit(ERROR);
    }

    /* memory allocation (for vertices) */
    OWN_CALLOC(vertex, anim_vector, nv);

    /* read vertices */
    for (j = 0; j < nv; j++)
    {
        if (fget_line(lne, fpin) == ERROR)
        {
            printf("... error while reading the file %s , vertex %d\n",
                   infile, j);
            OWN_FREE(lne);
            exit(ERROR);
        }
        if (sscanf(lne, "%f %f %f", &vertex[j][0],
                   &vertex[j][1],
                   &vertex[j][2]) != 3)
        {
            printf("... error while reading the file %s , vertex %d\n",
                   infile, j);
            OWN_FREE(lne);
            exit(ERROR);
        }
    }

    /* produce temporary file containing face information without remarks */
    if ((tmpfp = tmpfile()) == NULL)
    {
        printf("ERROR opening tmpfile");
        exit(ERROR);
    }
    nf = 0;
    while (fget_line(lne, fpin) == ERRORFREE)
    {
        fprintf(tmpfp, "%s", lne);
        nftmp = strcnt(lne, (int)'-');
        nf = nf + nftmp;
    }
    rewind(tmpfp);

    (void)fclose(fpin);

    /* header line of outfile */
    fprintf(fpout, "%d %d\n", nv, nf);

    /* write vertices */
    for (j = 0; j < nv; j++)
    {
        fprintf(fpout, "%f %f %f\n",
                vertex[j][0], vertex[j][1], vertex[j][2]);
    }

    /* memory allocation */
    OWN_CALLOC(face, int, MAXVERTICES);

    /* read/write faces/colors */
    nftmp = 0;
    nvf = 0;
    for (j = 0; j < nf; j++)
    {
        do
        {
            if (fscanf(tmpfp, "%d", &entry) != 1)
            {
                printf("... error while reading the file %s , element %d\n",
                       infile, j);
                OWN_FREE(lne);
                exit(ERROR);
            }
            if (entry != 1000)
            {
                face[nvf] = entry;
                nvf++;
            }
            if (entry < 0)
            { /* face complete */
                nvf--;
                fprintf(fpout, "%d ", nvf); /* number of nodes */
                cf = index_to_color_frame(entry); /* get color indices */
                cp = index_to_color_polygon(entry);
                fprintf(fpout, "%d ", cf);
                fprintf(fpout, "%d ", cp);

                for (n = 0; n < nvf; n++)
                { /* write vertices of face */
                    fprintf(fpout, "%d ", (face[n] + no_first_node));
                }
                nvf = 0;
                fprintf(fpout, "\n");
            }
        } while (entry >= 0);
    }

    /* nconec was sometimes negative using mkobject ??? (e.g. sphere)
  for(j=0;j<nconec;j++){ 
    if(fscanf(tmpfp,"%d",&entry)!=1){
      printf("... error while reading the file %s , element %d\n",
	     infile,j);
      OWN_FREE(lne);
      exit(ERROR);
    }
    if(entry!=1000){
      face[nvf]=entry;
      nvf++;
    }
    if(entry<0){            
      nvf--;
      fprintf(fpout,"%d ",nvf);       
      cf=index_to_color_frame(entry); 
      cp=index_to_color_polygon(entry);
      fprintf(fpout,"%d ",cf);
      fprintf(fpout,"%d ",cp);

      for(n=0;n<nvf;n++){             
	fprintf(fpout,"%d ",face[n]);
      }
      nvf=0;
      fprintf(fpout,"\n");
    }
  }
  */

    OWN_FREE(vertex);
    OWN_FREE(face);
    OWN_FREE(lne);

    (void)fclose(fpout);
}

/* ------------------------------------------------------------------ */
void convert_elgeofile(int no_first_node, int ntimesteps)
{
    int nv, /* number of vertices in open file */
        nconec, /* number of connections in open file */
        j, n, nf, nftmp, nvf,
        entry, cf, cp, *face = NULL;
    char *lne = NULL;
    FILE *fpout, *fpin,
        *tmpfp = NULL; /* pointer on temporary file */

    /* open elgeofiles */
    if ((fpin = fopen(infile, "r")) == NULL)
    {
        printf("... unable to open the elgeo file %s\n", infile);
        exit(ERROR);
    }
    if ((fpout = fopen(outfile, "w")) == NULL)
    {
        printf("... unable to open the elgeo file %s\n", outfile);
        exit(ERROR);
    }

    /* read elgeofile the first time; 
     (the new file can't be written until number of
      faces was determined) */
    OWN_CALLOC(lne, char, MAXLENGTH);

    if (fget_line(lne, fpin) == ERROR)
    {
        printf("... error while reading the elgeofile %s\n",
               infile);
        OWN_FREE(lne);
        exit(ERROR);
    }

    if (sscanf(lne, "%d %d", &nv, &nconec) != 2)
    {
        printf("... error while reading header line of elgeofile %s\n",
               infile);
        OWN_FREE(lne);
        exit(ERROR);
    }

    /* read vertices */
    for (j = 0; j < nv * ntimesteps; j++)
    {
        if (fget_line(lne, fpin) == ERROR)
        {
            printf("... error while reading the file %s , line %d\n",
                   infile, j + 1);
            OWN_FREE(lne);
            exit(ERROR);
        }
    }

    /* produce temporary file containing face information without remarks */
    if ((tmpfp = tmpfile()) == NULL)
    {
        printf("ERROR opening tmpfile");
        exit(ERROR);
    }
    nf = 0;
    while (fget_line(lne, fpin) == ERRORFREE)
    {
        fprintf(tmpfp, "%s", lne);
        nftmp = strcnt(lne, (int)'-');
        nf = nf + nftmp;
    }
    rewind(tmpfp);

    (void)fclose(fpin);

    /* read elgeofile the second time; 
     (now knowing the number of faces) */

    if ((fpin = fopen(infile, "r")) == NULL)
    {
        printf("... unable to open the elgeo file %s the second time\n", infile);
        exit(ERROR);
    }

    if (fget_line(lne, fpin) == ERROR)
    {
        printf("... error while reading the elgeofile %s\n",
               infile);
        OWN_FREE(lne);
        exit(ERROR);
    }

    if (sscanf(lne, "%d %d", &nv, &nconec) != 2)
    {
        printf("... error while reading header line of elgeofile %s\n",
               infile);
        OWN_FREE(lne);
        exit(ERROR);
    }

    /* header line of outfile */
    fprintf(fpout, "%d %d %d\n", nv, nf, ntimesteps);

    /* read vertices */
    for (j = 0; j < nv * ntimesteps; j++)
    {
        if (fget_line(lne, fpin) == ERROR)
        {
            printf("... error while reading the file %s , vertex %d\n",
                   infile, j);
            OWN_FREE(lne);
            exit(ERROR);
        }
        fprintf(fpout, "%s", lne);
    }

    /* produce temporary file containing face information without remarks */
    if ((tmpfp = tmpfile()) == NULL)
    {
        printf("ERROR opening tmpfile");
        exit(ERROR);
    }
    nf = 0;
    while (fget_line(lne, fpin) == ERRORFREE)
    {
        fprintf(tmpfp, "%s", lne);
        nftmp = strcnt(lne, (int)'-');
        nf = nf + nftmp;
    }
    rewind(tmpfp);

    (void)fclose(fpin);

    /* memory allocation */
    OWN_CALLOC(face, int, MAXVERTICES);

    /* read/write faces/colors */
    nftmp = 0;
    nvf = 0;
    for (j = 0; j < nf; j++)
    {
        do
        {
            if (fscanf(tmpfp, "%d", &entry) != 1)
            {
                printf("... error while reading the file %s , element %d\n",
                       infile, j);
                OWN_FREE(lne);
                exit(ERROR);
            }
            if (entry != 1000)
            {
                face[nvf] = entry;
                nvf++;
            }
            if (entry < 0)
            { /* face complete */
                nvf--;
                fprintf(fpout, " %d ", nvf); /* number of nodes */
                cf = index_to_color_frame(entry); /* get color indices */
                cp = index_to_color_polygon(entry);
                fprintf(fpout, "%d ", cf);
                fprintf(fpout, "%d ", cp);

                for (n = 0; n < nvf; n++)
                { /* write vertices of face */
                    fprintf(fpout, "%d ", (face[n] + no_first_node));
                }
                nvf = 0;
                fprintf(fpout, "\n");
            }
        } while (entry >= 0);
    }

    /* nconec was sometimes negative using mkobject ??? (e.g. sphere)
  for(j=0;j<nconec;j++){ 
    if(fscanf(tmpfp,"%d",&entry)!=1){
      printf("... error while reading the file %s , element %d\n",
	     infile,j);
      OWN_FREE(lne);
      exit(ERROR);
    }
    if(entry!=1000){
      face[nvf]=entry;
      nvf++;
    }
    if(entry<0){            
      nvf--;
      fprintf(fpout,"%d ",nvf);       
      cf=index_to_color_frame(entry); 
      cp=index_to_color_polygon(entry);
      fprintf(fpout,"%d ",cf);
      fprintf(fpout,"%d ",cp);

      for(n=0;n<nvf;n++){             
	fprintf(fpout,"%d ",face[n]);
      }
      nvf=0;
      fprintf(fpout,"\n");
    }
  }
  */

    OWN_FREE(face);
    OWN_FREE(lne);

    (void)fclose(fpout);
}

/* ------------------------------------------------------------------ */
int fget_line(char *lne, FILE *fp) /* read line from file
                                               free of remarks        */
{
    char *rempos; /* position of remark     */

    do
    {

        /* read line of text out of file, NULL=FALSE */
        if (fgets(lne, MAXLENGTH, fp) == NULL)
        {
            return (ERROR);
        }

        /* ignore remarks and empty lines */
        rempos = strchr(lne, '#');
        if (rempos != NULL)
        {
            *rempos = '\0';
        }
    } while (strlen(lne) == strspn(lne, " \f\n\r\t\v"));

    return (ERRORFREE);
}
/* ------------------------------------------------------------------ */
int strcnt(char *lne, int symbol) /* count number of occurences
                                           of symbol in line 'lne'    */
{
    int counter; /* counter for symbol */

    counter = 0;
    /* strchr sucht Zeichen Symbol (-) in Zeichenfolge lne, 
     bei Erfolg Zeiger auf Zeichen, sonst NULL */
    while ((lne = strchr(lne, symbol)) != NULL)
    {
        lne++;
        counter++;
    }
    return (counter);
}

/* ------------------------------------------------------------------ */
int index_to_color_frame(int index)
{
    int i;

    i = -index;
    if (i > 90000 && i < 100000)
    {
        i = i - 90000;
        i = i / 100;
    }
    else if (i > 9000000 && i < 10000000)
    {
        i = i - 9000000;
        i = i / 1000;
    }
    else
    {
        printf("error in index_to_color_frame: index=%d\n", i);
    }
    return (i);
}

/* ------------------------------------------------------------------ */
int index_to_color_polygon(int index)
{
    int i;

    i = -index;
    if (i > 90000 && i < 100000)
    {
        i = i - 90000;
        i = i % 100;
    }
    else if (i > 9000000 && i < 10000000)
    {
        i = i - 9000000;
        i = i % 1000;
    }
    else
    {
        printf("error in index_to_color_polygon: index=%d\n", i);
    }
    return (i);
}

/* ------------------------------------------------------------------ */
