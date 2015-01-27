/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <covise/covise_binary.h>
#include "domain.h"
#include "element.h"
#include <appl/ApplInterface.h>
#include <api/coModule.h>
#include "coTetin.h"
#include "coTetin__Proj.h"

#define WRITE_MODE "wb"
#define MAX_DOMAIN_IDS 100

typedef struct
{
    int *Elements; /* pointer to elements */
    int *pids; /* number of elements  */
    int n_elem; /* number of elements  */
} SURF_ELEM;
static char *curr_domainfile_name = NULL;
static char *curr_bocofile_name = NULL;
static char *tmp_domainfile_name = NULL;
static int curr_domain = -1;
static int tmp_domain = -1;
static double *Mesh = NULL;
static double *Mesh_orig = NULL;
static int n_Mesh_nodes;
static SURF_ELEM Surf_Elements[2] = { NULL, NULL, 0, NULL, NULL, 0 };
static int nodes_per_el[2] = { 3, 4 };
static char name_space[3][2000];

static void ICEM_close_dom(int id);
static void icem_def_variables(int id);
static void ICEM_write_ascii_file(char *file_name, char *text, int *ierr);
static char *ICEM_temp_name(char *dir, char *space, char *dflt);
static void ICEM_make_directory_name(char *config_dir,
                                     char *dir_name,
                                     char **dir_path_name);
void pj_covise_get_configdir(char **configfile);
void pj_covise_get_outpintffile(char **outpintffile);
void pj_covise_get_bocofile(char **bocofile);
void pj_covise_get_projected_points(float **points_x, float **points_y,
                                    float **points_z, int *n_points);
void pj_covise_free_projected_points(void);
int pj_covise_tetin(coTetin *obj);
void trans_point(const double inpnt[3],
                 const float mat[3][3], const float add[3],
                 double outpnt[3]);

static int Domain_ids[MAX_DOMAIN_IDS];
static int Domain_n_Mesh_nodes[MAX_DOMAIN_IDS];
static char *Domain_filenames[MAX_DOMAIN_IDS];
static char *Domain_bocofile_names[MAX_DOMAIN_IDS];
static double *Domain_Mesh[MAX_DOMAIN_IDS];
static double *Domain_Mesh_orig[MAX_DOMAIN_IDS];
static SURF_ELEM Domain_Surf_Elements[MAX_DOMAIN_IDS][2];
static int first = 1;

// open ICEM domain
// domainfilename : filepathname of domain file
// bocofilename   : filepathname of boco file
int ICEM_openDomain(char *domainfilename)
{
    int Res = 0;
    int i, j, i1;
    int El_node[28];
    int loop;
    int n_sections;
    int start_sec;
    int end_sec;
    int el_type;
    int dom_type;
    int cur_idx_zg;
    int idx_el_size;
    int cur_nodes_per_el;
    int cur_elem;
    int n_elements;
    int len;
    int id = -1;
    int *cur_e_array_total;
    char *bocofilename = NULL;

    if (first)
    {
        first = 0;
        for (i = 0; i < MAX_DOMAIN_IDS; i++)
        {
            Domain_ids[i] = -1;
            Domain_filenames[i] = NULL;
            Domain_bocofile_names[i] = NULL;
            Domain_Mesh[i] = NULL;
            Domain_Mesh_orig[i] = NULL;
            Domain_n_Mesh_nodes[i] = 0;
            Domain_Surf_Elements[i][0].Elements = Domain_Surf_Elements[i][1].Elements = NULL;
            Domain_Surf_Elements[i][0].pids = Domain_Surf_Elements[i][1].pids = NULL;
            Domain_Surf_Elements[i][0].n_elem = Domain_Surf_Elements[i][1].n_elem = 0;
        }
    }
    pj_covise_get_bocofile(&bocofilename);
    if (domainfilename != NULL && bocofilename != NULL)
    {
        sprintf(name_space[2], "%s", domainfilename);
        curr_domainfile_name = name_space[2];
    }
    else
    {
        Res = 2;
    }
    if (Res == 0)
    {
        sprintf(name_space[1], "%s", bocofilename);
        curr_bocofile_name = name_space[1];
        if ((curr_domain = df_open(curr_domainfile_name,
                                   MODE_READ, UNSTRUCTURED_DOMAIN))
            != -1)
        {
            /* get domain type: must be domain_mode */
            df_type(curr_domain, &dom_type);
            if (dom_type == UNSTRUCTURED_DOMAIN)
            {
                if (df_n_nodes(curr_domain, &n_Mesh_nodes) != -1 && df_n_elements(curr_domain, &n_elements) != -1)
                {
                    if (n_Mesh_nodes > 0)
                    {
                        Mesh = new double[2 * 3 * n_Mesh_nodes];
                    }
                    if (n_Mesh_nodes > 0 && Mesh != NULL)
                    {
                        Mesh_orig = Mesh + 3 * n_Mesh_nodes;
                        /* read nodes */
                        if (df_unstruct_read_nodes(curr_domain, (int)0,
                                                   n_Mesh_nodes, Mesh) != -1)
                        {
                            for (i = 0; i < 3 * n_Mesh_nodes; i++)
                            {
                                Mesh_orig[i] = Mesh[i];
                            }
                            for (loop = 0; loop < 2; loop++)
                            {
                                Surf_Elements[0].n_elem = Surf_Elements[1].n_elem = 0;
                                /* read the no of sections */
                                if (df_n_sections(curr_domain, &n_sections) != -1)
                                {
                                    for (i = 0; i < n_sections; i++)
                                    {
                                        /* read start and end no and type of elements */
                                        if (df_section_info(curr_domain, i, &start_sec,
                                                            &end_sec, &el_type) != -1)
                                        {
                                            if (el_type == TRI_3 || el_type == QUAD_4)
                                            {
                                                /* get no of nodes per element */
                                                /* size of idx_polygon */
                                                idx_el_size = ((el_type == TRI_3) ? 0 : 1);
                                                cur_nodes_per_el = nodes_per_el[idx_el_size];
                                                cur_elem = Surf_Elements[idx_el_size].n_elem;
                                                cur_idx_zg = cur_elem * cur_nodes_per_el;
                                                Surf_Elements[idx_el_size].n_elem += end_sec - start_sec + 1;
                                                if (loop == 1)
                                                {
                                                    for (j = start_sec; j <= end_sec; j++)
                                                    {
                                                        /* read next element */
                                                        if (df_read_elements(curr_domain, j,
                                                                             (int)1, El_node) != -1)
                                                        {
                                                            cur_e_array_total = &(Surf_Elements[idx_el_size].Elements[cur_idx_zg]);
                                                            for (i1 = 1; i1 <= cur_nodes_per_el;
                                                                 i1++)
                                                            {
                                                                cur_e_array_total[i1 - 1] = El_node[i1];
                                                            }
                                                            Surf_Elements[idx_el_size].pids[cur_elem] = El_node[0];
                                                            cur_elem++;
                                                            cur_idx_zg += cur_nodes_per_el;
                                                        }
                                                        else
                                                        {
                                                            Res = 7;
                                                            break;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        if (Res)
                                            break;
                                    }
                                }
                                if (loop == 0)
                                {
                                    for (i = 0; i < 2; i++)
                                    {
                                        if (Surf_Elements[i].n_elem > 0)
                                        {
                                            Surf_Elements[i].Elements = new int[Surf_Elements[i].n_elem * nodes_per_el[i]];
                                            if (Surf_Elements[i].Elements == NULL)
                                            {
                                                Res = 6;
                                                break;
                                            }
                                            Surf_Elements[i].pids = new int[Surf_Elements[i].n_elem];
                                            if (Surf_Elements[i].pids == NULL)
                                            {
                                                Res = 6;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if (Res)
                                    break;
                            }
                        }
                        else
                        {
                            Res = 5;
                        }
                    }
                    else
                    {
                        Res = 4;
                    }
                }
                else
                {
                    Res = 3;
                }
            }
            else
            {
                Res = 2;
            }
        }
        else
        {
            Res = 1;
        }
    }
    if (!Res)
    {
        for (i = 0; i < MAX_DOMAIN_IDS; i++)
        {
            if (Domain_ids[i] < 0)
            {
                id = i;
                break;
            }
        }
        if (id >= 0)
        {
            Domain_ids[id] = curr_domain;
            len = strlen(curr_domainfile_name) + 1;
            Domain_filenames[id] = new char[len];
            if (Domain_filenames[id] == NULL)
            {
                Domain_ids[id] = -1;
                id = -1;
            }
            else
            {
                sprintf(Domain_filenames[id], "%s", curr_domainfile_name);
                len = strlen(curr_bocofile_name) + 1;
                Domain_bocofile_names[id] = new char[len];
                if (Domain_bocofile_names[id] == NULL)
                {
                    Domain_ids[id] = -1;
                    delete[] Domain_filenames[id];
                    Domain_filenames[id] = NULL;
                    id = -1;
                }
                else
                {
                    sprintf(Domain_bocofile_names[id], "%s", curr_bocofile_name);
                }
            }
        }
        if (id >= 0)
        {
            Domain_Mesh[id] = Mesh;
            Domain_Mesh_orig[id] = Mesh_orig;
            Domain_n_Mesh_nodes[id] = n_Mesh_nodes;
            Domain_Surf_Elements[id][0].Elements = Surf_Elements[0].Elements;
            Domain_Surf_Elements[id][1].Elements = Surf_Elements[1].Elements;
            Domain_Surf_Elements[id][0].pids = Surf_Elements[0].pids;
            Domain_Surf_Elements[id][1].pids = Surf_Elements[1].pids;
            Domain_Surf_Elements[id][0].n_elem = Surf_Elements[0].n_elem;
            Domain_Surf_Elements[id][1].n_elem = Surf_Elements[1].n_elem;
        }
    }

    return id;
}

// close current ICEM domain and free memory
void ICEM_close_dom(int id)
{
    int i;
    if (id >= 0 && id < MAX_DOMAIN_IDS)
    {
        if (Domain_ids[id] != -1)
        {
            df_close(Domain_ids[id]);
            Domain_ids[id] = -1;
        }
        if (Domain_filenames[id] != NULL)
        {
            delete[] Domain_filenames[id];
            Domain_filenames[id] = NULL;
        }
        if (Domain_bocofile_names[id] != NULL)
        {
            delete[] Domain_bocofile_names[id];
            Domain_bocofile_names[id] = NULL;
        }
        if (Domain_Mesh[id] != NULL)
        {
            delete[] Domain_Mesh[id];
            Domain_Mesh[id] = NULL;
        }
        Domain_Mesh_orig[id] = NULL;
        Domain_n_Mesh_nodes[id] = 0;
        for (i = 0; i < 2; i++)
        {
            if (Domain_Surf_Elements[id][i].Elements != NULL)
            {
                delete[] Domain_Surf_Elements[id][i].Elements;
                Domain_Surf_Elements[id][i].Elements = NULL;
            }
            if (Domain_Surf_Elements[id][i].pids != NULL)
            {
                delete[] Domain_Surf_Elements[id][i].pids;
                Domain_Surf_Elements[id][i].pids = NULL;
            }
            Domain_Surf_Elements[id][i].n_elem = 0;
        }
    }
}

void ICEM_closeDomain(int id)
{
    ICEM_close_dom(id);
}

static void icem_def_variables(int id)
{
    if (id >= 0 && id < MAX_DOMAIN_IDS)
    {
        curr_domain = Domain_ids[id];
        curr_domainfile_name = Domain_filenames[id];
        curr_bocofile_name = Domain_bocofile_names[id];
        Mesh = Domain_Mesh[id];
        Mesh_orig = Domain_Mesh_orig[id];
        n_Mesh_nodes = Domain_n_Mesh_nodes[id];
        Surf_Elements[0].Elements = Domain_Surf_Elements[id][0].Elements;
        Surf_Elements[1].Elements = Domain_Surf_Elements[id][1].Elements;
        Surf_Elements[0].pids = Domain_Surf_Elements[id][0].pids;
        Surf_Elements[1].pids = Domain_Surf_Elements[id][1].pids;
        Surf_Elements[0].n_elem = Domain_Surf_Elements[id][0].n_elem;
        Surf_Elements[1].n_elem = Domain_Surf_Elements[id][1].n_elem;
    }
    else
    {
        curr_domain = -1;
        curr_domainfile_name = NULL;
        curr_bocofile_name = NULL;
        Mesh = NULL;
        Mesh_orig = NULL;
        n_Mesh_nodes = 0;
        Surf_Elements[0].Elements = NULL;
        Surf_Elements[1].Elements = NULL;
        Surf_Elements[0].pids = NULL;
        Surf_Elements[1].pids = NULL;
        Surf_Elements[0].n_elem = 0;
        Surf_Elements[1].n_elem = 0;
    }
}

// close (temporary) ICEM domain and call output interface
// output_name : type of output interface # case name
// type of output interface can be star, fluent or fenfloss
int ICEM_closeDomain(int id, char *solver_text_obj_name)
{
    int Res = 0;
    char *config_dir = NULL;
    char *domain_file = NULL;
    char *output_name = NULL;
    char tmp_array[4000];
    icem_def_variables(id);
    if (n_Mesh_nodes > 0 && Mesh != NULL && curr_domainfile_name != NULL)
    {
        sprintf(tmp_array, "./");
        tmp_domainfile_name = ICEM_temp_name(tmp_array,
                                             name_space[0], "/tmp/xxx_dom");
        sprintf(tmp_array, "/bin/cp %s %s", curr_domainfile_name,
                tmp_domainfile_name);
        Res = system(tmp_array);
        if (Res)
        {
            Res = 1;
        }
        else
        {
            domain_file = tmp_domainfile_name;
            if ((tmp_domain = df_open(tmp_domainfile_name,
                                      MODE_MODIFY, UNSTRUCTURED_DOMAIN)) != -1)
            {
                if (df_unstruct_update_nodes(tmp_domain, 0, n_Mesh_nodes,
                                             Mesh) == -1)
                {
                    Res = 2;
                }
            }
            else
            {
                Res = 3;
            }
        }
    }
    else
    {
        Res = 4;
    }
    if (tmp_domain != -1)
    {
        df_close(tmp_domain);
        tmp_domain = -1;
    }
    pj_covise_get_configdir(&config_dir);
    pj_covise_get_outpintffile(&output_name);
    if (config_dir == NULL || domain_file == NULL || output_name == NULL || solver_text_obj_name == NULL)
    {
        Res = 5;
    }

    if (Res == 0)
    {
        char *ptr_outpintftype = 0;
        char outpintftype[100];
        char *case_name = "VISiT";
        int ret_outp = 1;
        char *ptr_hash;
        char *env_mesh;
        char tmp_array[1000];
        tmp_array[0] = '\0';
        if ((ptr_hash = strchr(output_name, (int)'#')) == NULL)
        {
            ptr_outpintftype = output_name;
        }
        else
        {
            case_name = ptr_hash + 1;
            char *Zg = output_name;
            char *Zg1 = outpintftype;
            while ((*Zg) != '#')
            {
                *Zg1++ = *Zg++;
            }
            *Zg1 = '\0';
            ptr_outpintftype = outpintftype;
        }
        env_mesh = getenv("ICEM_ACN");
        if (ptr_outpintftype && env_mesh)
        {
            char *transfer_directory;
            char *path_case_name = 0;
            ICEM_make_directory_name(config_dir, "transfer",
                                     &transfer_directory);
            path_case_name = new char[strlen(transfer_directory) + strlen(case_name) + 2];
            sprintf(path_case_name, "%s/%s", transfer_directory,
                    case_name);
            if (strcmp(ptr_outpintftype, "star") == 0)
            {
                sprintf(tmp_array,
                        "#!/bin/csh -f\n%s/icemcfd/output-interfaces/starcd -dom %s -u -b %s %s\n",
                        env_mesh, domain_file, curr_bocofile_name, path_case_name);
            }
            else if (strcmp(ptr_outpintftype, "fluent") == 0)
            {
                sprintf(tmp_array,
                        "#!/bin/csh -f\n%s/icemcfd/output-interfaces/georampant -dom %s -b %s %s\n",
                        env_mesh, domain_file, curr_bocofile_name, path_case_name);
            }
            else if (strcmp(ptr_outpintftype, "fenfloss") == 0)
            {
                sprintf(tmp_array,
                        "#!/bin/csh -f\n%s/icemcfd/output-interfaces/fenfloss -dom %s -b %s %s\n",
                        env_mesh, domain_file, curr_bocofile_name, path_case_name);
            }
            if (strlen(tmp_array) > 0)
            {
                char name_space_tmp[1000];
                char *tmp_file_name = ICEM_temp_name(0, name_space_tmp,
                                                     "/tmp/ICEM_tmp_file");
                ICEM_write_ascii_file(tmp_file_name, tmp_array,
                                      &ret_outp);
                if (ret_outp == 0)
                {
                    int val;
                    mode_t mode;
                    // Read, write, execute by owner,
                    // Read by group, Execute by group
                    // Read by others, Execute by others
                    mode = (S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
                    val = chmod(tmp_file_name, mode);
                    ret_outp = system(tmp_file_name);
                    unlink(tmp_file_name);
                }
            }
            char *text_solver = 0;
            int char_size;
            if (ret_outp == 0)
            {
                char *format_text = "SOLVER %s\nDIR %s\nCASE %s\n";
                char_size = strlen(format_text) + strlen(transfer_directory) + strlen(case_name) + strlen(ptr_outpintftype) + 1;
                text_solver = new char[char_size];
                sprintf(text_solver, format_text,
                        ptr_outpintftype, transfer_directory,
                        case_name);
            }
            if (text_solver)
            {
                if (solver_text_obj_name)
                {
                    // create the output object
                    coDoText *text = new coDoText(solver_text_obj_name,
                                                  char_size, text_solver);
                    if (text)
                    {
                        delete text;
                    }
                }
                delete[] text_solver;
            }
            if (path_case_name)
            {
                delete[] path_case_name;
                path_case_name = 0;
            }
        }
        else
        {
            if (!env_mesh)
            {
                printf("ICEM_ACN has to be set\n");
            }
        }
    }
    ICEM_close_dom(id);
    if (tmp_domainfile_name != NULL)
    {
        unlink(tmp_domainfile_name);
        tmp_domainfile_name = NULL;
    }
    return Res;
}

// get boundary surface of current domain
// as coDoPolygons
// objectname : name of coDoPolygons object
coDoPolygons *ICEM_getDomainSurface(int id, char *objectname)
{
    coDoPolygons *Polygons = NULL;
    icem_def_variables(id);
    if ((Surf_Elements[0].n_elem > 0 || Surf_Elements[1].n_elem > 0) && n_Mesh_nodes > 0)
    {
        int *idx_nodes = new int[n_Mesh_nodes];
        if (idx_nodes != NULL)
        {
            int i, i1, j, k, l, m;
            int n_nodes = 0;
            for (i = 0; i < n_Mesh_nodes; i++)
            {
                idx_nodes[i] = -1;
            }
            for (i = 0; i < 2; i++)
            {
                int *Elements = Surf_Elements[i].Elements;
                int cur_nodes_per_el = nodes_per_el[i];
                for (j = 0; j < Surf_Elements[i].n_elem * cur_nodes_per_el; j++)
                {
                    if (Elements[j] >= 0 && Elements[j] < n_Mesh_nodes)
                    {
                        idx_nodes[Elements[j]] = 1;
                    }
                }
            }
            for (i = 0; i < n_Mesh_nodes; i++)
            {
                if (idx_nodes[i] >= 0)
                {
                    idx_nodes[i] = n_nodes;
                    n_nodes++;
                }
            }
            if (n_nodes > 0)
            {
                float *coord = new float[3 * n_nodes];
                int n_elem = Surf_Elements[0].n_elem + Surf_Elements[1].n_elem;
                int n_vertex = Surf_Elements[0].n_elem * nodes_per_el[0] + Surf_Elements[1].n_elem * nodes_per_el[1];
                int *polygon_list = new int[n_elem + n_vertex];
                int *vertex_list = polygon_list + n_elem;
                if (coord != NULL && polygon_list != NULL)
                {
                    float *x_coord = coord;
                    float *y_coord = x_coord + n_nodes;
                    float *z_coord = y_coord + n_nodes;
                    n_nodes = 0;
                    for (i = 0; i < n_Mesh_nodes; i++)
                    {
                        if (idx_nodes[i] >= 0)
                        {
                            int idx = 3 * i;
                            x_coord[n_nodes] = (float)Mesh[idx];
                            y_coord[n_nodes] = (float)Mesh[idx + 1];
                            z_coord[n_nodes] = (float)Mesh[idx + 2];
                            n_nodes++;
                        }
                    }
                    for (i = 0, l = 0, m = 0; i < 2; i++)
                    {
                        int *Elements = Surf_Elements[i].Elements;
                        int cur_nodes_per_el = nodes_per_el[i];
                        for (j = 0, k = 0; j < Surf_Elements[i].n_elem; j++)
                        {
                            int l_s = l;
                            int probl = 0;
                            polygon_list[m] = l;
                            for (i1 = k; i1 < k + cur_nodes_per_el; i1++)
                            {
                                if (idx_nodes[Elements[i1]] >= 0)
                                {
                                    vertex_list[l++] = idx_nodes[Elements[i1]];
                                }
                                else
                                {
                                    probl = 1;
                                }
                            }
                            if (probl)
                            {
                                l = l_s;
                            }
                            else
                            {
                                m++;
                            }
                            k += cur_nodes_per_el;
                        }
                    }
                    Polygons = new coDoPolygons(objectname, n_nodes,
                                                x_coord, y_coord, z_coord,
                                                l, vertex_list, m, polygon_list);
                    if (Polygons != NULL)
                    {
                        Polygons->addAttribute("vertexOrder", "2");
                    }
                }
                if (coord != NULL)
                {
                    delete[] coord;
                }
                if (polygon_list != NULL)
                {
                    delete[] polygon_list;
                }
            }
        }
        if (idx_nodes != NULL)
        {
            delete[] idx_nodes;
        }
    }
    return Polygons;
}

// transform current domain (relative to original coordinates)
// relpos : transformation vector
// rotMat : rotation matrix
void ICEM_transformDomain(int id, float relpos[3], float rotMat[3][3])
{
    icem_def_variables(id);
    for (int i = 0; i < n_Mesh_nodes; i++)
    {
        double *loc_in = &(Mesh_orig[3 * i]);
        double *loc_out = &(Mesh[3 * i]);
        trans_point(loc_in, rotMat, relpos, loc_out);
    }
}

char *cov_temp_name(char *dir, char *space, char *dflt)
{
    char *tmp_str;
    char *name = tempnam(dir, 0);
    if (name == 0)
    {
        tmp_str = strcpy(space, dflt);
    }
    else
    {
        tmp_str = strcpy(space, name);
        free(name);
    }
#ifdef _WIN32
    int len = strlen(space);
    for (int i = 0; i < len; i++)
    {
        if (space[i] == '\\')
        {
            space[i] = '/';
        }
    }
#endif
    return space;
}

// project selected surface elements of current domain onto
// geometry (nearest point)
// family_oberflaeche : family name of surface elements to project
int ICEM_projectSurfaceGrid(int id, char *family_oberflaeche, float *direct)
{
    int Res = 0;
    icem_def_variables(id);
    if (curr_domain >= 0 && n_Mesh_nodes > 0 && Mesh != NULL && family_oberflaeche != NULL)
    {
        int pid = -1;
        int i, j, k;
        int loop;
        char *name = family_oberflaeche;
        if (df_get_family_pid(curr_domain, name, &pid) == -1)
        {
            int len = strlen(family_oberflaeche);
            char *upper_name = new char[len + 1];
            for (loop = 0; loop < 2; loop++)
            {
                if (loop == 0)
                {
                    for (i = 0; i < len; i++)
                        upper_name[i] = (islower(name[i]) ? toupper(name[i]) : name[i]);
                }
                else
                {
                    for (i = 0; i < len; i++)
                        upper_name[i] = (isupper(name[i]) ? tolower(name[i]) : name[i]);
                }
                if (df_get_family_pid(curr_domain, upper_name, &pid) != -1)
                {
                    break;
                }
                else
                {
                    pid = -1;
                }
            }
            delete[] upper_name;
        }
        if (pid >= 0)
        {
            int *idx_nodes = new int[n_Mesh_nodes];
            int n_points;
            if (idx_nodes != NULL)
            {
                float *points = NULL;
                for (i = 0; i < n_Mesh_nodes; i++)
                {
                    idx_nodes[i] = 0;
                }
                for (i = 0; i < 2; i++)
                {
                    int *Elements = Surf_Elements[i].Elements;
                    int *pids = Surf_Elements[i].pids;
                    int cur_nodes_per_el = nodes_per_el[i];
                    for (j = 0; j < Surf_Elements[i].n_elem; j++)
                    {
                        if (pids[j] == pid)
                        {

                            for (k = j * cur_nodes_per_el;
                                 k < (j + 1) * cur_nodes_per_el; k++)
                            {
                                if (Elements[k] >= 0 && Elements[k] < n_Mesh_nodes)
                                {
                                    idx_nodes[Elements[k]] = 1;
                                }
                            }
                        }
                    }
                }
                for (loop = 0; loop < 2; loop++)
                {
                    n_points = 0;
                    for (i = 0; i < n_Mesh_nodes; i++)
                    {
                        if (idx_nodes[i])
                        {
                            if (loop == 1)
                            {
                                j = 3 * i;
                                k = 3 * n_points;
                                points[k] = Mesh[j];
                                points[k + 1] = Mesh[j + 1];
                                points[k + 2] = Mesh[j + 2];
                            }
                            n_points++;
                        }
                    }
                    if (loop == 0)
                    {
                        if (n_points > 0)
                        {
                            points = new float[3 * n_points];
                        }
                        else
                        {
                            break;
                        }
                    }
                }
                if (n_points > 0)
                {
                    // nearest point
                    // project to all families
                    const char **family_names = 0;
                    coTetin__Proj *Proj;
                    Proj = new coTetin__Proj(n_points, points, direct, 0,
                                             family_names);
                    coTetin *tetin = new coTetin();
                    if (tetin)
                    {
                        tetin->append(Proj);
                        delete[] points;
                        Res = pj_covise_tetin(tetin);
                        delete tetin;
                        if (Res != 0)
                        {
                            if (Res != -999)
                                Res = 1;
                        }
                        else
                        {
                            float *points_x, *points_y, *points_z;
                            int n_points_prj = 0;
                            pj_covise_get_projected_points(&points_x, &points_y,
                                                           &points_z, &n_points_prj);
                            if (points_x != NULL && points_y != NULL && points_z != NULL && n_points_prj == n_points)
                            {
                                n_points = 0;
                                for (i = 0; i < n_Mesh_nodes; i++)
                                {
                                    if (idx_nodes[i])
                                    {
                                        j = 3 * i;
                                        Mesh[j] = points_x[n_points];
                                        Mesh[j + 1] = points_y[n_points];
                                        Mesh[j + 2] = points_z[n_points];
                                        n_points++;
                                    }
                                }
                            }
                            else
                            {
                                Res = 1;
                            }
                        }
                        pj_covise_free_projected_points();
                    }
                    else
                    {
                        delete[] points;
                        Res = 2;
                    }
                }
                delete[] idx_nodes;
            }
            else
            {
                Res = 3;
            }
        }
    }
    else
    {
        Res = 4;
    }
    return Res;
}

// write ascii file
// file_name : filepathname of ascci file
// text : text to write
static void ICEM_write_ascii_file(char *file_name, char *text,
                                  int *ierr)
{
    int Res = 0;
    FILE *file = 0;

    if ((file = fopen(file_name, WRITE_MODE)))
    {
        if (fputs(text, file) <= 0)
        {
            printf("Problem in _write_ascii_file\n");
            Res = 1;
        }
    }
    else
    {
        Res = 2;
    }
    if (file)
    {
        fclose(file);
        if (Res)
        {
            unlink(file_name);
        }
    }
    *ierr = Res;
}

// create temporary filename
static char *ICEM_temp_name(char *dir, char *space, char *dflt)
{
    char *tmp_str;
    char *name = tempnam(dir, 0);
    if (name == 0)
    {
        tmp_str = strcpy(space, dflt);
    }
    else
    {
        tmp_str = strcpy(space, name);
        free(name);
    }
    return space;
}

// create directory
static void ICEM_make_directory_name(char *config_dir,
                                     char *dir_name,
                                     char **dir_path_name)
{
#ifndef _WIN32
    static char name_space[3000];
    mode_t mode;
    struct stat stat_buf;

    sprintf(name_space, "%s/%s", config_dir, dir_name);
    if (access(name_space, F_OK))
    {
        // Read, write, execute by owner,
        // Read by group, Execute by group
        // Read by others, Execute by others
        mode = (S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
        if (mkdir(name_space, mode))
        {
            printf("Directory \"%s\" cannot be created.\n", name_space);
            printf("Create %s in configuration directory\n", config_dir);
            sprintf(name_space, "%s", config_dir);
        }
    }
    else
    {
        // check if directory
        if (stat(name_space, &stat_buf))
        {
            printf("Check problems with directory %s\n", name_space);
            printf("Create %s in configuration directory\n", config_dir);
            sprintf(name_space, "%s", config_dir);
        }
        else
        {
            if ((stat_buf.st_mode & S_IFMT) != S_IFDIR)
            {
                printf("\"%s\" is not a directory.\n", name_space);
                printf("Create %s in configuration directory\n", config_dir);
                sprintf(name_space, "%s", config_dir);
            }
        }
    }
    *dir_path_name = name_space;
#endif
}
