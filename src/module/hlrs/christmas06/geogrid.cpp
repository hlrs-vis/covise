/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <include/model.h>
#include <math.h>

struct christgrid *CreateChristGrid(struct christ_model *model)
{
    struct christgrid *grid = NULL;
    if ((grid = (struct christgrid *)calloc(1, sizeof(struct christgrid))) == NULL)
    {
        fprintf(stderr, "not enough memory for struct christgrid!");
        return NULL;
    }

    grid->e = AllocElementStruct();
    grid->p = AllocPointStruct();

    float spacing;
    float x, y, z;
    float xmin = -0.5 * model->size[0];
    float ymin = -0.5 * model->size[1];
    float zmin = 0.0;

    int i, j, k;

    // Create Coordinates
    spacing = model->spacing;
    grid->nelem_x = (int)(model->size[0] / spacing);
    grid->nelem_y = (int)(model->size[1] / spacing);
    grid->nelem_z = (int)(model->size[2] / spacing);
    grid->spacing_x = model->size[0] / grid->nelem_x;
    grid->spacing_y = model->size[1] / grid->nelem_y;
    grid->spacing_z = model->size[2] / grid->nelem_z;

    grid->npoi_x = grid->nelem_x + 1;
    grid->npoi_y = grid->nelem_y + 1;
    grid->npoi_z = grid->nelem_z + 1;
    grid->nelem = grid->nelem_x * grid->nelem_y * grid->nelem_z;

    if (grid->nelem > 4500000)
    {
        fprintf(stderr, "Grid is too fine. Please enlarge spacing parameter.\n");
        return NULL;
    }
    if (grid->nelem < 10000)
    {
        fprintf(stderr, "Grid is too coarse. Please reduce spacing parameter.\n");
        return NULL;
    }

    for (i = 0; i < grid->npoi_z; i++)
    {
        for (j = 0; j < grid->npoi_y; j++)
        {
            for (k = 0; k < grid->npoi_x; k++)
            {
                x = xmin + k * grid->spacing_x;
                y = ymin + j * grid->spacing_y;
                z = zmin + i * grid->spacing_z;
                AddPoint(grid->p, x, y, z);
            }
        }
    }

    // Create connectvity list
    //int jmax = grid->nelem_y;
    //int kmax = grid->nelem_x;
    int elem[8];

    for (i = 0; i < grid->nelem_z; i++)
    {
        for (j = 0; j < grid->nelem_y; j++)
        {
            for (k = 0; k < grid->nelem_x; k++)
            {
                elem[0] = i * (grid->npoi_y * grid->npoi_x) + j * grid->npoi_x + k;
                elem[1] = elem[0] + 1;
                elem[2] = elem[1] + grid->npoi_x;
                elem[3] = elem[0] + grid->npoi_x;
                elem[4] = elem[0] + (grid->npoi_y * grid->npoi_x);
                elem[5] = elem[1] + (grid->npoi_y * grid->npoi_x);
                elem[6] = elem[2] + (grid->npoi_y * grid->npoi_x);
                elem[7] = elem[3] + (grid->npoi_y * grid->npoi_x);
                AddElement(grid->e, elem);
            }
        }
    }

    // subtract elements from grid and generate bcs
    grid->npoi = grid->p->nump;
    grid->p_to_remove = (int *)calloc(grid->npoi, sizeof(int));
    grid->e_to_remove = (int *)calloc(grid->nelem, sizeof(int));

    grid->bcwall = AllocIlistStruct(100);
    grid->bcwallpol = AllocIlistStruct(25);
    grid->bcwallvol = AllocIlistStruct(25);
    grid->bcwallvol_outer = AllocIlistStruct(25);

    grid->bcin = AllocIlistStruct(100);
    grid->bcinpol = AllocIlistStruct(25);
    grid->bcinvol = AllocIlistStruct(25);
    grid->bcinvol_outer = AllocIlistStruct(25);

    grid->bcout = AllocIlistStruct(100);
    grid->bcoutpol = AllocIlistStruct(25);
    grid->bcoutvol = AllocIlistStruct(25);
    grid->bcoutvol_outer = AllocIlistStruct(25);

    DefineBCs(grid, model, -1);
    GenerateBCs(grid, model, -1);
    //fprintf(stderr,"grid->bcwall->num=%d\n",grid->bcwall->num);
    //fprintf(stderr,"model->nobjects=%d\n",model->nobjects);
    for (i = 0; i < model->nobjects; i++)
    {
        subtractCube(grid, model, i);
        DefineBCs(grid, model, i);
        GenerateBCs(grid, model, i);
        //fprintf(stderr,"grid->bcwall->num=%d\n",grid->bcwall->num);
    }

    // clean grid coordinate and connectivity list due to removed cells and vertexes
    // 1. remove points
    int offset = 0;
    for (i = 0; i < grid->npoi; i++)
    {
        if (grid->p_to_remove[i] == 0)
        {
            grid->p->x[i - offset] = grid->p->x[i];
            grid->p->y[i - offset] = grid->p->y[i];
            grid->p->z[i - offset] = grid->p->z[i];
        }
        else if (grid->p_to_remove[i] == -1)
        {
            offset++;
        }
        else
        {
            fprintf(stderr, "error 1 in CreatechristGrid!\n");
        }
    }
    grid->p->nump -= offset;

    // 3. node replacement preparation
    offset = 0;
    grid->new_node = new int[grid->npoi];
    for (i = 0; i < grid->npoi; i++)
    {
        if (grid->p_to_remove[i] == -1)
        {
            grid->new_node[i] = -1;
            offset++;
        }
        else if (grid->p_to_remove[i] == 0)
        {
            grid->new_node[i] = i - offset;
        }
        else
        {
            fprintf(stderr, "error 2 in CreatechristGrid!\n");
        }
    }

    // 2. remove elements
    offset = 0;
    for (i = 0; i < grid->nelem; i++)
    {
        if (grid->e_to_remove[i] == 0)
        {
            for (j = 0; j < 8; j++)
            {
                grid->e->e[i - offset][j] = grid->e->e[i][j];
            }
        }
        if (grid->e_to_remove[i] == -1)
        {
            offset++;
        }
    }
    grid->e->nume -= offset;

    // 4. vertex replacement in element list
    int vertex;
    for (i = 0; i < grid->e->nume; i++)
    {
        for (j = 0; j < 8; j++)
        {
            vertex = grid->e->e[i][j];
            //if ( (grid->new_node[vertex]==-1)&&(grid->e_to_remove[i]!=-1) )
            if ((grid->new_node[vertex] == -1) && (grid->e_to_remove[i] != -1))
            {
                fprintf(stderr, "error 3 in CreatechristGrid. Element list uses node that was removed.\n");
            }
            else
            {
                grid->e->e[i][j] = grid->new_node[vertex];
            }
        }
    }

    // 5. mapping list old element nr. -> new element number
    offset = 0;
    grid->new_elem = new int[grid->nelem];
    for (i = 0; i < grid->nelem; i++)
    {
        if (grid->e_to_remove[i] == -1)
        {
            grid->new_elem[i] = -1;
            offset++;
        }
        else if (grid->e_to_remove[i] == 0)
        {
            grid->new_elem[i] = i - offset;
        }
        else
        {
            fprintf(stderr, "error 4 in CreatechristGrid!\n");
        }
    }

    // 6. remove walls where we have node-bcs (ven or airconditioning)

    // generate list (length: all nodes) that tells us which nodes are used for node-bcs
    // if all nodes of a wall element are used for node-bcs, we must remove this wall element

    // zusaetzlich ueberpruefen: wurde der Knoten bereits entfernt? Das waere eine moegliche Fehlerquelle
    int *isbcnode = (int *)calloc(grid->npoi, sizeof(int));

    int wallnodes[4];
    int wallsremoved = 0;
    int inner_removed;
    int outer_removed;
    int remove;
    int nremoved;
    for (i = 0; i < grid->bcwallvol->num; i++)
    {
        inner_removed = 0;
        outer_removed = 0;
        remove = 0;

        wallnodes[0] = grid->bcwall->list[4 * i];
        wallnodes[1] = grid->bcwall->list[4 * i + 1];
        wallnodes[2] = grid->bcwall->list[4 * i + 2];
        wallnodes[3] = grid->bcwall->list[4 * i + 3];

        // Vergleich ueber zugehoerige Element!!!
        // entfernen, falls beide zugehoerigen Volumenelement (innen und aussen)
        // entfernt wurden oder nicht existieren (= ausserhalb Geometrie)
        if (grid->bcwallvol->list[i] == -1)
            inner_removed = 1;
        /*        else if (grid->e_to_remove[grid->bcwallvol->list[i]]==-1)
               inner_removed=1;
            if (grid->e_to_remove[grid->bcwallvol_outer->list[i]]==-1)
               outer_removed=1;

            // remove if both volume elements don't exist any more
            if ( (inner_removed==1) && (outer_removed==1) )
              {
               remove=1;
            }
      */
        if (inner_removed == 1)
            remove = 1;

        // remove if a wallnode has been removed
        if ((grid->new_node[wallnodes[0]] == -1) || (grid->new_node[wallnodes[1]] == -1) || (grid->new_node[wallnodes[2]] == -1) || (grid->new_node[wallnodes[3]] == -1))
        {
            remove = 1;
        }
        if ((grid->p_to_remove[wallnodes[0]] == -1) || (grid->p_to_remove[wallnodes[1]] == -1) || (grid->p_to_remove[wallnodes[2]] == -1) || (grid->p_to_remove[wallnodes[3]] == -1))
        {
            //fprintf(stderr,"wall list uses removed node on a place where we have a node bc. Shouldn't happen.\n");
        }
        // remove if wall fully consists of bcnodes
        if ((isbcnode[wallnodes[0]]) && (isbcnode[wallnodes[1]]) && (isbcnode[wallnodes[2]]) && (isbcnode[wallnodes[3]]))
        {
            // remove it!
            remove = 1;
        }
        if (grid->bcwallvol->list[i] == -1)
        {
            // remove it!
            remove = 1;
        }
        if (grid->new_elem[grid->bcwallvol->list[i]] == -1)
        {
            remove = 1;
        }
        if (remove == 1)
        {
            wallsremoved++;
        }
        else
        {
            // take it!
            grid->bcwall->list[4 * (i - wallsremoved) + 0] = grid->new_node[wallnodes[0]];
            grid->bcwall->list[4 * (i - wallsremoved) + 1] = grid->new_node[wallnodes[1]];
            grid->bcwall->list[4 * (i - wallsremoved) + 2] = grid->new_node[wallnodes[2]];
            grid->bcwall->list[4 * (i - wallsremoved) + 3] = grid->new_node[wallnodes[3]];
            grid->bcwallvol->list[i - wallsremoved] = grid->new_elem[grid->bcwallvol->list[i]];
        }
    }
    grid->bcwall->num -= 4 * wallsremoved;
    grid->bcwallvol->num -= wallsremoved;
    grid->bcwallpol->num -= wallsremoved;

    // wenn wir noch eine Tuere reinmachen, dann wird diese besser als outlet definiert ... aber die Wand muss trotzdem weg!

    // 9. vertex replacement in bc inlet vertex list
    nremoved = 0;
    offset = 0;
    int bcelem[4];
    for (i = 0; i < grid->bcinvol->num; i++)
    {
        inner_removed = 0;
        outer_removed = 0;

        bcelem[0] = grid->bcin->list[4 * i];
        bcelem[1] = grid->bcin->list[4 * i + 1];
        bcelem[2] = grid->bcin->list[4 * i + 2];
        bcelem[3] = grid->bcin->list[4 * i + 3];

        // Vergleich ueber zugehoerige Element!!!
        // entfernen, falls beide zugehoerigen Volumenelement (innen und aussen)
        // entfernt wurden oder nicht existieren (= ausserhalb Geometrie)
        if (grid->bcinvol_outer->list[i] == -1)
            inner_removed = 1;
        else if (grid->e_to_remove[grid->bcinvol_outer->list[i]] == -1)
            inner_removed = 1;
        if (grid->e_to_remove[grid->bcinvol->list[i]] == -1)
            outer_removed = 1;

        if (inner_removed && outer_removed)
        {
            // bcelem uses nodes that have been removed! remove bc!
            remove = 1;
        }

        if (remove == 1)
        {
            // nicht uebernehmen!!
            nremoved++;
            remove = 0;
            offset++;
        }
        else
        {
            // uebernehmen!!
            grid->bcin->list[4 * i - 4 * offset] = grid->new_node[bcelem[0]];
            grid->bcin->list[4 * i - 4 * offset + 1] = grid->new_node[bcelem[1]];
            grid->bcin->list[4 * i - 4 * offset + 2] = grid->new_node[bcelem[2]];
            grid->bcin->list[4 * i - 4 * offset + 3] = grid->new_node[bcelem[3]];
            //fprintf(stderr,"bcelem, new vertices %d %d %d %d\n", grid->new_node[bcelem[0]],
            //										             grid->new_node[bcelem[1]],
            //                                                     grid->new_node[bcelem[2]],
            //                                                     grid->new_node[bcelem[3]]);
            grid->bcinvol->list[i - offset] = grid->new_elem[grid->bcinvol->list[i]];
        }
    }
    grid->bcin->num -= nremoved * 4;
    grid->bcinpol->num -= nremoved;
    grid->bcinvol->num -= nremoved;

    // 10. vertex replacement in bc outlet vertex list
    nremoved = 0;
    offset = 0;
    for (i = 0; i < grid->bcoutvol->num; i++)
    {
        inner_removed = 0;
        outer_removed = 0;

        bcelem[0] = grid->bcout->list[4 * i];
        bcelem[1] = grid->bcout->list[4 * i + 1];
        bcelem[2] = grid->bcout->list[4 * i + 2];
        bcelem[3] = grid->bcout->list[4 * i + 3];

        // Vergleich ueber zugehoerige Element!!!
        // entfernen, falls beide zugehoerigen Volumenelement (innen und aussen)
        // entfernt wurden oder nicht existieren (= ausserhalb Geometrie)
        if (grid->bcoutvol_outer->list[i] == -1)
            inner_removed = 1;
        else if (grid->e_to_remove[grid->bcoutvol_outer->list[i]] == -1)
            inner_removed = 1;
        if (grid->e_to_remove[grid->bcoutvol->list[i]] == -1)
            outer_removed = 1;

        if (inner_removed && outer_removed)
        {
            // bcelem uses nodes that have been removed! remove bc!
            remove = 1;
        }

        if (remove == 1)
        {
            // nicht uebernehmen!!
            nremoved++;
            remove = 0;
            offset++;
        }
        else
        {
            // uebernehmen!!
            grid->bcout->list[4 * i - 4 * offset] = grid->new_node[bcelem[0]];
            grid->bcout->list[4 * i - 4 * offset + 1] = grid->new_node[bcelem[1]];
            grid->bcout->list[4 * i - 4 * offset + 2] = grid->new_node[bcelem[2]];
            grid->bcout->list[4 * i - 4 * offset + 3] = grid->new_node[bcelem[3]];
            //fprintf(stderr,"bcelem, new vertices %d %d %d %d\n", grid->new_node[bcelem[0]],
            //										             grid->new_node[bcelem[1]],
            //                                                     grid->new_node[bcelem[2]],
            //                                                     grid->new_node[bcelem[3]]);
            grid->bcoutvol->list[i - offset] = grid->new_elem[grid->bcoutvol->list[i]];
        }
    }
    grid->bcout->num -= nremoved * 4;
    grid->bcoutpol->num -= nremoved;
    grid->bcoutvol->num -= nremoved;

    /*
      // 11. element replacement in bc wall volume list
      offset=0;
      for (i=0;i<grid->bcwallvol->num+wallsremoved;i++)
       {
         if (grid->new_elem[grid->bcwallvol->list[i]]==-1)
         {
            // was removed!
            offset++;
         }
         else
   {
   // take it!
   grid->bcwallvol->list[i-offset]=grid->new_elem[grid->bcwallvol->list[i]];
   }
   }
   */

    // inlet node bcs
    int *inlet_nodes = (int *)calloc(grid->npoi, sizeof(int));
    for (i = 0; i < grid->bcinvol->num; i++)
    {
        inlet_nodes[grid->bcin->list[4 * i + 0]] = 1;
        inlet_nodes[grid->bcin->list[4 * i + 1]] = 1;
        inlet_nodes[grid->bcin->list[4 * i + 2]] = 1;
        inlet_nodes[grid->bcin->list[4 * i + 3]] = 1;
    }

    grid->bcin_nodes = AllocIlistStruct(25);
    grid->bcin_velos = AllocFlistStruct(125);

    int bcnode;

    // Node-bc inlet (outer geometry)
    k = 0;

    float l = 1.; // characteristic length
    float vin_abs = sqrt(pow(model->bcin_velo[0], 2) + pow(model->bcin_velo[1], 2) + pow(model->bcin_velo[2], 2));
    float _k = 3.75 * 0.001 * pow(vin_abs, 2);
    float _eps = 9.40 * 0.0001 * pow(fabs(vin_abs / l), 3);

    float vx, vy, vz;
    float zCoord;
    for (i = 0; i < grid->p->nump; i++)
    {
        if (inlet_nodes[i] == 1)
        {
            bcnode = i;
            zCoord = grid->p->z[i];
            vx = model->bcin_velo[0] * (1 + zCoord * model->zScale);
            vy = model->bcin_velo[1] * (1 + zCoord * model->zScale);
            vz = model->bcin_velo[2] * (1 + zCoord * model->zScale);
            vin_abs = sqrt(vx * vx + vy * vy + vz * vz);
            _k = 3.75 * 0.001 * vin_abs * vin_abs;
            _eps = 9.40 * 0.0001 * pow(fabs(vin_abs / l), 3);

            Add2Ilist(grid->bcin_nodes, bcnode);
            Add2Flist(grid->bcin_velos, vx);
            Add2Flist(grid->bcin_velos, vy);
            Add2Flist(grid->bcin_velos, vz);
            Add2Flist(grid->bcin_velos, _k);
            Add2Flist(grid->bcin_velos, _eps);
        }
    }

    // 12. vertex replacement in node-bc lists (ven and airconditioning)
    // transform ven and air nodes to new notation
    /* not here in christmas06 */

    // 13. it happens that there are unremoved nodes at the intersection of objects
    //     we remove all nodes that don't appear in the element list
    int *rep_nodes = (int *)calloc(grid->npoi, sizeof(int));
    for (i = 0; i < grid->e->nume; i++)
    {
        for (j = 0; j < 8; j++)
        {
            rep_nodes[grid->e->e[i][j]]++;
        }
    }
    // nodes gives us the number of appearance in the element list of each node now
    offset = 0;
    int node;
    for (i = 0; i < grid->p->nump; i++)
    {
        if (rep_nodes[i] == 0)
        {
            rep_nodes[i] = -1;
            offset++;
        }
        else
        {
            rep_nodes[i] = i - offset;
        }
    }
    // nodes is -1 for nodes to remove, for all other nodes it gives the new number
    // replace node list now
    offset = 0;
    for (i = 0; i < grid->npoi; i++)
    {
        if (rep_nodes[i] == -1)
        {
            offset++;
        }
        else
        {
            grid->p->x[i - offset] = grid->p->x[i];
            grid->p->y[i - offset] = grid->p->y[i];
            grid->p->z[i - offset] = grid->p->z[i];
        }
    }
    grid->npoi -= offset;
    grid->p->nump -= offset;

    // we replace all lists containing nodenumbers now once again ...
    for (i = 0; i < grid->e->nume; i++)
    {
        for (j = 0; j < 8; j++)
        {
            node = grid->e->e[i][j];
            grid->e->e[i][j] = rep_nodes[node];
        }
    }
    for (i = 0; i < grid->bcin_nodes->num; i++)
    {
        node = grid->bcin_nodes->list[i];
        grid->bcin_nodes->list[i] = rep_nodes[node];
    }
    for (i = 0; i < grid->bcwallvol->num; i++)
    {
        for (j = 0; j < 4; j++)
        {
            node = grid->bcwall->list[4 * i + j];
            grid->bcwall->list[4 * i + j] = rep_nodes[node];
        }
    }
    for (i = 0; i < grid->bcinvol->num; i++)
    {
        for (j = 0; j < 4; j++)
        {
            node = grid->bcin->list[4 * i + j];
            grid->bcin->list[4 * i + j] = rep_nodes[node];
        }
    }
    for (i = 0; i < grid->bcoutvol->num; i++)
    {
        for (j = 0; j < 4; j++)
        {
            node = grid->bcout->list[4 * i + j];
            grid->bcout->list[4 * i + j] = rep_nodes[node];
        }
    }

    // realloc of grid->p arrays and grid->e
    // can be done

    // free mem
    delete[] grid -> new_node;
    free(grid->p_to_remove);
    free(grid->e_to_remove);
    free(rep_nodes);

    return grid;
}

struct covise_info *CreateGeometry4Covise(struct christ_model *model)
{
    int i;

    struct covise_info *ci = NULL;
    int ipol;
    int ipoi;

    ipoi = 0;
    ipol = 0;

    if ((ci = AllocCoviseInfo()) != NULL)
    {
        // Create Polygons
        for (i = 0; i < model->nobjects; i++)
        {
            CreateCubePolygons4Covise(ci, model, i, ipoi, ipol);
        }
    }

    // AddFloorPolygon
    ipoi = ci->p->nump;
    ipol = ci->pol->num;
    AddPoint(ci->p, -0.5 * model->size[0],
             -0.5 * model->size[1],
             0.0);
    AddPoint(ci->p, +0.5 * model->size[0],
             -0.5 * model->size[1],
             0.0);
    AddPoint(ci->p, +0.5 * model->size[0],
             +0.5 * model->size[1],
             0.0);
    AddPoint(ci->p, -0.5 * model->size[0],
             +0.5 * model->size[1],
             0.0);

    // first polygon
    Add2Ilist(ci->vx, ipoi);
    Add2Ilist(ci->vx, ipoi + 1);
    Add2Ilist(ci->vx, ipoi + 2);
    Add2Ilist(ci->vx, ipoi + 3);
    Add2Ilist(ci->pol, ipol * 4);

    return ci;
}

int read_string(char *buf, char **str, const char *separator)
// buf        ... Zeile / Quelle
// str        ... zu lesender String (Ergebnis)
// separator  ... Zeichenfolge, nach der str steht

{
    int pos, i;
    char buffer2[100];
    char *str_help;

    buf = strstr(buf, separator); // mit separator
    buf += sizeof(char) * strlen(separator); // ohne separator
    pos = 0;
    for (i = 0; i < (int)strlen(buf); i++) // Tabs etc. weg
    {
        if (!(/*( buf[i]==' ' ) || */ (buf[i] == '\t') || (buf[i] == '\n')))
        {
            buffer2[pos] = buf[i];
            pos++;
        }
    }
    buffer2[pos] = '\0';

    str_help = strdup(buffer2); // alloc memory and copy string
    *str = &str_help[0];

    return (0);
}

int read_int(char *buf, int *izahl, const char *separator)
{
    buf = strstr(buf, separator); // mit separator
    buf += sizeof(char) * strlen(separator); // ohne separator

    sscanf(buf, "%d ", izahl);

    return (0);
}

int read_double(char *buf, double *dzahl, const char *separator)
{
    buf = strstr(buf, separator); // mit separator
    buf += sizeof(char) * strlen(separator); // ohne separator

    sscanf(buf, "%lf", dzahl);

    return (0);
}

int CreateCubePolygons4Covise(struct covise_info *ci, struct christ_model *model, int number, int ipoi, int ipol)
{

    ipoi = ci->p->nump;
    ipol = ci->pol->num * 4;

    float pos[3];
    float size[3];
    pos[0] = model->cubes[number]->pos[0];
    pos[1] = model->cubes[number]->pos[1];
    pos[2] = model->cubes[number]->pos[2];
    size[0] = model->cubes[number]->size[0];
    size[1] = model->cubes[number]->size[1];
    size[2] = model->cubes[number]->size[2];

    AddPoint(ci->p, pos[0] - 0.5 * size[0],
             pos[1] - 0.5 * size[1],
             pos[2] - 0.5 * size[2]);
    AddPoint(ci->p, pos[0] + 0.5 * size[0],
             pos[1] - 0.5 * size[1],
             pos[2] - 0.5 * size[2]);
    AddPoint(ci->p, pos[0] + 0.5 * size[0],
             pos[1] + 0.5 * size[1],
             pos[2] - 0.5 * size[2]);
    AddPoint(ci->p, pos[0] - 0.5 * size[0],
             pos[1] + 0.5 * size[1],
             pos[2] - 0.5 * size[2]);

    AddPoint(ci->p, pos[0] - 0.5 * size[0],
             pos[1] - 0.5 * size[1],
             pos[2] + 0.5 * size[2]);
    AddPoint(ci->p, pos[0] + 0.5 * size[0],
             pos[1] - 0.5 * size[1],
             pos[2] + 0.5 * size[2]);
    AddPoint(ci->p, pos[0] + 0.5 * size[0],
             pos[1] + 0.5 * size[1],
             pos[2] + 0.5 * size[2]);
    AddPoint(ci->p, pos[0] - 0.5 * size[0],
             pos[1] + 0.5 * size[1],
             pos[2] + 0.5 * size[2]);

    // first polygon
    Add2Ilist(ci->vx, ipoi);
    Add2Ilist(ci->vx, ipoi + 1);
    Add2Ilist(ci->vx, ipoi + 2);
    Add2Ilist(ci->vx, ipoi + 3);
    Add2Ilist(ci->pol, ipol);
    ipol += 4;

    // second polygon
    Add2Ilist(ci->vx, ipoi);
    Add2Ilist(ci->vx, ipoi + 1);
    Add2Ilist(ci->vx, ipoi + 5);
    Add2Ilist(ci->vx, ipoi + 4);
    Add2Ilist(ci->pol, ipol);
    ipol += 4;

    // third polygon
    Add2Ilist(ci->vx, ipoi + 1);
    Add2Ilist(ci->vx, ipoi + 2);
    Add2Ilist(ci->vx, ipoi + 6);
    Add2Ilist(ci->vx, ipoi + 5);
    Add2Ilist(ci->pol, ipol);
    ipol += 4;

    // fourth polygon
    Add2Ilist(ci->vx, ipoi + 2);
    Add2Ilist(ci->vx, ipoi + 3);
    Add2Ilist(ci->vx, ipoi + 7);
    Add2Ilist(ci->vx, ipoi + 6);
    Add2Ilist(ci->pol, ipol);
    ipol += 4;

    // fifth polygon
    Add2Ilist(ci->vx, ipoi + 3);
    Add2Ilist(ci->vx, ipoi);
    Add2Ilist(ci->vx, ipoi + 4);
    Add2Ilist(ci->vx, ipoi + 7);
    Add2Ilist(ci->pol, ipol);
    ipol += 4;

    // sixth polygon
    Add2Ilist(ci->vx, ipoi + 4);
    Add2Ilist(ci->vx, ipoi + 5);
    Add2Ilist(ci->vx, ipoi + 6);
    Add2Ilist(ci->vx, ipoi + 7);
    Add2Ilist(ci->pol, ipol);
    ipol += 4;

    return (0);
}

int subtractCube(struct christgrid *grid, struct christ_model *model, int number)
{
    // subtracts cube from grid (removes cells and all interior nodes)
    // adapts coordinates of margins
    // and removes boundary conditions, if there are any
    //float spacing = model->spacing;
    float pos[3];
    pos[0] = model->cubes[number]->pos[0];
    pos[1] = model->cubes[number]->pos[1];
    pos[2] = model->cubes[number]->pos[2];
    float size[3];
    size[0] = model->cubes[number]->size[0];
    size[1] = model->cubes[number]->size[1];
    size[2] = model->cubes[number]->size[2];

    //fprintf(stderr,"subtractCube Nr. %d, ", number);

    // from cell counters to node counters
    int npoi_x = grid->nelem_x + 1;
    int npoi_y = grid->nelem_y + 1;
    int npoi_z = grid->nelem_z + 1;

    // counters
    int i, j, k;
    // i .. z-direction
    // j .. y-direction
    // k .. x-direction

    // coordinate range values min / max
    float xmin, xmax, ymin, ymax, zmin, zmax;

    // grid counter values min / max
    int ilo, ihi, jlo, jhi, klo, khi;

    float bxmin = -0.5 * model->size[0];
    float bymin = -0.5 * model->size[1];
    float bzmin = 0.;

    /*
       float bxmax = 0.5*model->size[0];
       float bymax = 0.5*model->size[1];
       float bzmax = model->size[2];
   */

    // all elements must be removed that lie completely inside the cube to be removed
    xmin = pos[0] - 0.5 * size[0];
    xmax = pos[0] + 0.5 * size[0];
    ymin = pos[1] - 0.5 * size[1];
    ymax = pos[1] + 0.5 * size[1];
    zmin = pos[2] - 0.5 * size[2];
    zmax = pos[2] + 0.5 * size[2];

    // ilo,ihi: node nrs, but highest value grid->npoi_xyz-1!!!
    ilo = (int)ceil(zmin / grid->spacing_z);
    ihi = (int)floor(zmax / grid->spacing_z);
    jlo = (int)ceil((ymin + 0.5 * model->size[1]) / grid->spacing_y);
    jhi = (int)floor((ymax + 0.5 * model->size[1]) / grid->spacing_y);
    klo = (int)ceil((xmin + 0.5 * model->size[0]) / grid->spacing_x);
    khi = (int)floor((xmax + 0.5 * model->size[0]) / grid->spacing_x);

    fprintf(stderr, "number=%2d", number);
    /*    fprintf(stderr,"imin=%2d imax=%2d jmin=%2d jmax=%2d kmin=%2d kmax=%2d\n",model->ilo,model->ihi,model->jlo,model->jhi,model->klo,model->khi);
       fprintf(stderr,"ilo =%2d ihi =%2d jlo =%2d jhi =%2d klo =%2d khi =%2d\n",ilo,ihi,jlo,jhi,klo,khi);
       fprintf(stderr,"npoi_x=%2d, npoi_y=%2d, npoi_z=%2d\n", npoi_x, npoi_y, npoi_z);
   */

    /*
      float khiexakt=(xmax+0.5*model->size[0])/grid->spacing_x;
       fprintf(stderr,"khi=%d, khiexakt=%8.4f\n", khi, khiexakt);
      float dx=((khi-khiexakt));
       if ( (dx>0.) && (dx<0.1*grid->spacing_x) )
         //khi++;
   */

    // check if complete cube lies outside grid area
    if ((ilo < 0) && (ihi < 0))
    {
        fprintf(stderr, "\nz-position to small. cube outside volume, can't be removed\n");
        return -1;
    }
    if ((ilo > grid->nelem_z) && (ihi > grid->nelem_z))
    {
        fprintf(stderr, "\nz-position to small. cube outside volume, can't be removed\n");
        return -1;
    }
    if ((jlo < 0) && (jhi < 0))
    {
        fprintf(stderr, "\ny-position to small. cube outside volume, can't be removed\n");
        return -1;
    }
    if ((jlo > grid->nelem_y) && (jhi > grid->nelem_y))
    {
        fprintf(stderr, "\ny-position to small. cube outside volume, can't be removed\n");
        return -1;
    }
    if ((klo < 0) && (khi < 0))
    {
        fprintf(stderr, "\nx-position to small. cube outside volume, can't be removed\n");
        return -1;
    }
    if ((klo > grid->nelem_x) && (khi > grid->nelem_x))
    {
        fprintf(stderr, "\nx-position to small. cube outside volume, can't be removed\n");
        return -1;
    }

    // test
    // check if part of cube lies outside grid area
    if ((ilo < 0) && (ihi >= 0) && (ihi <= npoi_z))
        ilo = 0;
    if ((ihi > npoi_z) && (ilo >= 0) && (ilo <= npoi_z))
        ihi = npoi_z;
    if ((jlo < 0) && (jhi >= 0) && (jhi <= npoi_y))
        jlo = 0;
    if ((jhi > npoi_y) && (jlo >= 0) && (jlo <= npoi_y))
        jhi = npoi_y;
    if ((klo < 0) && (khi >= 0) && (khi <= npoi_x))
        klo = 0;
    if ((khi > npoi_x) && (klo >= 0) && (klo <= npoi_x))
        khi = npoi_x;

    int dont_ilo = 0;
    int dont_ihi = 0;
    int dont_jlo = 0;
    int dont_jhi = 0;
    int dont_klo = 0;
    int dont_khi = 0;

    // test mit <=, >=
    // if cube lies outside ...
    if (ilo <= model->ilo)
    {
        ilo = model->ilo; //=0
        zmin = ilo * grid->spacing_z;
        dont_ilo = 1;
        //fprintf(stderr,"dont_ilo==1!!!\n");
    }
    if (jlo <= model->jlo)
    {
        jlo = model->jlo; //=0
        ymin = jlo * grid->spacing_y - 0.5 * model->size[1];
        dont_jlo = 1;
        //fprintf(stderr,"dont_jlo==1!!!\n");
    }
    if (klo <= model->klo)
    {
        klo = model->klo; // =0
        xmin = klo * grid->spacing_x - 0.5 * model->size[0];
        dont_jlo = 1;
        //fprintf(stderr,"dont_klo==1!!!\n");
    }
    if (ihi >= model->ihi)
    {
        ihi = model->ihi; // =npoi_z
        zmax = ihi * grid->spacing_z;
        dont_ihi = 1;
        //fprintf(stderr,"dont_ihi==1!!!\n");
    }
    if (jhi >= model->jhi)
    {
        jhi = model->jhi; // =npoi_y
        ymax = jhi * grid->spacing_y - 0.5 * model->size[1];
        dont_jhi = 1;
        //fprintf(stderr,"dont_jhi==1!!!\n");
    }
    if (khi >= model->khi)
    {
        khi = model->khi; //=npoi_x
        xmax = khi * grid->spacing_x - 0.5 * model->size[0];
        dont_khi = 1;
        //fprintf(stderr,"dont_khi==1!!!\n");
    }

    // check if high and low values fall together
    // two reasons: a) coarse spacing / small objects
    // 				b) object just touches volume at margin

    // ATTENTION! HANDLE THIS WITH CARE.
    /*
      int smalldx=0;
      int smalldy=0;
      int smalldz=0;

       if (ihi==ilo)
       {
         if (ilo*grid->spacing_z-zmin < zmax-ihi*grid->spacing_z)
            ilo--;
         else
            ihi++;
   smalldz=1;

   //    	fprintf(stderr,"z-expansion too small or object just touching volume, can't be removed!\n");
   //        return -1;

   }

   if (jhi==jlo)
   {
   if (jlo*grid->spacing_y-ymin < ymax-jhi*grid->spacing_y)
   jlo--;
   else
   jhi++;
   smalldy=1;

   //        fprintf(stderr,"y-expansion too small or object just touching volume, can't be removed!\n");
   //        return -1;

   }

   if (khi==klo)
   {
   if (klo*grid->spacing_x-xmin < xmax-khi*grid->spacing_x)
   klo--;
   else
   //khi++;
   smalldx=1;

   //        fprintf(stderr,"x-expansion too small or object just touching volume, can't be removed!\n");
   //        return -1;

   }
   */
    // check that no index is smaller than zero or bigger than possible (imin/npoints_x/jmin/npoints_y/kmin/npoints_y)
    if ((ilo < 0) || (ilo > grid->nelem_z) || (ihi < 0) || (ihi > grid->nelem_z) || (jlo < 0) || (jlo > grid->nelem_y) || (jhi < 0) || (jhi > grid->nelem_y) || (klo < 0) || (klo > grid->nelem_x) || (khi < 0) || (khi > grid->nelem_x))
    {
        fprintf(stderr, "Error in subtractCube!");
    }

    // adapt coords of cube margins so that grid fits to real cube margins
    // real margins of cube: xmin-xmax, ymin-ymax, zmin-zmax
    // margins (completely inside cube): ilo-ihi, jlo-jhi, klo-khi
    // grid->spacing_x, grid->spacing_y, grid->spacing_z
    float dzmin = ilo * grid->spacing_z - zmin;
    float dzmax = zmax - (ihi * grid->spacing_z);

    float dymin = jlo * grid->spacing_y - (ymin + 0.5 * model->size[1]);
    float dymax = ymax + 0.5 * model->size[1] - (jhi * grid->spacing_y);

    float dxmin = klo * grid->spacing_x - (xmin + 0.5 * model->size[0]);
    float dxmax = xmax + 0.5 * model->size[0] - (khi * grid->spacing_x);

    if (dzmin < 0)
        fprintf(stderr, "\nerror in subtractCube! dzmin<0\n");
    if (dzmax < 0)
        fprintf(stderr, "\nerror in subtractCube! dzmax<0\n");

    if (dymin < 0)
        fprintf(stderr, "\nerror in subtractCube! dymin<0\n");
    if (dymax < 0)
        fprintf(stderr, "\nerror in subtractCube! dymax<0\n");

    if (dxmin < 0)
        fprintf(stderr, "\nerror in subtractCube! dxmin<0\n");
    if (dxmax < 0)
        fprintf(stderr, "\nerror in subtractCube! dxmax<0\n");

    float tol = 0.4f;
    // two possibilities:
    // 1. make cells outside larger
    // 2. make cells outside smaller
    // tol < 0.5: cells outside get larger
    // tol > 0.5: cells outside get smaller: quad with angles > 90° possible!

    if ((dzmin > tol * grid->spacing_z) && (dont_ilo == 0))
    {
        // move ilo-1 upwards
        ilo--;
    }
    if ((dzmax > tol * grid->spacing_z) && (dont_ihi == 0))
    {
        // move ihi+1 downwards
        ihi++;
    }

    if ((dymin > tol * grid->spacing_y) && (dont_jlo == 0))
    {
        // move jlo-1 backwards
        jlo--;
    }
    //&& (jhi<model->jhi) )
    if ((dymax > tol * grid->spacing_y) && (dont_jhi == 0))
    {
        // move jhi+1 frontwards
        jhi++;
    }

    if ((dxmin > tol * grid->spacing_x) && (dont_klo == 0))
    {
        // move klo-1 right
        klo--;
    }
    if ((dxmax > tol * grid->spacing_x) && (dont_khi == 0))
    {
        // move khi+1 left
        khi++;
    }

    // now move it ...
    if (dont_ilo == 0)
    {
        if (dzmin <= tol * grid->spacing_z)
        {
            // move ilo dz downwards
            i = ilo;
            for (j = jlo; j <= jhi; j++)
            {
                for (k = klo; k <= khi; k++)
                {
                    //grid->p->z[i*(npoi_x*npoi_y)+j*(npoi_x)+k]-=dzmin;
                    grid->p->z[i * (npoi_x * npoi_y) + j * (npoi_x) + k] = bzmin + i * grid->spacing_z - dzmin;
                }
            }
        }
        else
        {
            // move ilo dz upwards
            i = ilo;
            for (j = jlo; j <= jhi; j++)
            {
                for (k = klo; k <= khi; k++)
                {
                    //grid->p->z[i*(npoi_x*npoi_y)+j*(npoi_x)+k]+=grid->spacing_z-dzmin;
                    grid->p->z[i * (npoi_x * npoi_y) + j * (npoi_x) + k] = bzmin + i * grid->spacing_z + grid->spacing_z - dzmin;
                }
            }
        }
    }

    if (dont_jlo == 0)
    {
        if (dymin <= tol * grid->spacing_y)
        {
            // move jlo frontwards
            j = jlo;
            for (i = ilo; i <= ihi; i++)
            {
                for (k = klo; k <= khi; k++)
                {
                    //grid->p->y[i*(npoi_x*npoi_y)+j*(npoi_x)+k]-=dymin;
                    grid->p->y[i * (npoi_x * npoi_y) + j * (npoi_x) + k] = bymin + j * grid->spacing_y - dymin;
                }
            }
        }
        else
        {
            // move jlo dy backwards
            j = jlo;
            for (i = ilo; i <= ihi; i++)
            {
                for (k = klo; k <= khi; k++)
                {
                    //grid->p->y[i*(npoi_x*npoi_y)+j*(npoi_x)+k]+=grid->spacing_y-dymin;
                    grid->p->y[i * (npoi_x * npoi_y) + j * (npoi_x) + k] = bymin + j * grid->spacing_y + grid->spacing_y - dymin;
                }
            }
        }
    }

    if (dont_klo == 0)
    {
        if (dxmin <= tol * grid->spacing_x)
        {
            // move klo dx left
            k = klo;
            for (i = ilo; i <= ihi; i++)
            {
                for (j = jlo; j <= jhi; j++)
                {
                    //grid->p->x[i*(npoi_x*npoi_y)+j*(npoi_x)+k]-=dxmin;
                    grid->p->x[i * (npoi_x * npoi_y) + j * (npoi_x) + k] = bxmin + k * grid->spacing_x - dxmin;
                }
            }
        }
        else
        {
            // move klo dx right
            k = klo;
            for (i = ilo; i <= ihi; i++)
            {
                for (j = jlo; j <= jhi; j++)
                {
                    //grid->p->x[i*(npoi_x*npoi_y)+j*(npoi_x)+k]+=grid->spacing_x-dxmin;
                    grid->p->x[i * (npoi_x * npoi_y) + j * (npoi_x) + k] = bxmin + k * grid->spacing_x + grid->spacing_x - dxmin;
                }
            }
        }
    }

    if (dont_ihi == 0)
    {
        if (dzmax <= tol * grid->spacing_z)
        {
            // move ihi dz upwards
            i = ihi;
            for (j = jlo; j <= jhi; j++)
            {
                for (k = klo; k <= khi; k++)
                {
                    //grid->p->z[i*(npoi_x*npoi_y)+j*(npoi_x)+k]+=dzmax;
                    grid->p->z[i * (npoi_x * npoi_y) + j * (npoi_x) + k] = bzmin + i * grid->spacing_z + dzmax;
                }
            }
        }
        else
        {
            // move ihi dz downwards
            i = ihi;
            for (j = jlo; j <= jhi; j++)
            {
                for (k = klo; k <= khi; k++)
                {
                    //grid->p->z[i*(npoi_x*npoi_y)+j*(npoi_x)+k]-=grid->spacing_z-dzmax;
                    grid->p->z[i * (npoi_x * npoi_y) + j * (npoi_x) + k] = bzmin + i * grid->spacing_z - (grid->spacing_z - dzmax);
                }
            }
        }
    }

    if (dont_jhi == 0)
    {
        if (dymax <= tol * grid->spacing_y)
        {
            // move jhi backwards
            j = jhi;
            for (i = ilo; i <= ihi; i++)
            {
                for (k = klo; k <= khi; k++)
                {
                    //grid->p->y[i*(npoi_x*npoi_y)+j*(npoi_x)+k]+=dymax;
                    grid->p->y[i * (npoi_x * npoi_y) + j * (npoi_x) + k] = bymin + j * grid->spacing_y + dymax;
                }
            }
        }
        else
        {
            // move jhi dy frontwards
            j = jhi;
            for (i = ilo; i <= ihi; i++)
            {
                for (k = klo; k <= khi; k++)
                {
                    //grid->p->y[i*(npoi_x*npoi_y)+j*(npoi_x)+k]-=grid->spacing_y-dymax;
                    grid->p->y[i * (npoi_x * npoi_y) + j * (npoi_x) + k] = bymin + j * grid->spacing_y - (grid->spacing_y - dymax);
                }
            }
        }
    }

    if (dont_khi == 0)
    {
        if (dxmax <= tol * grid->spacing_x)
        {
            // move khi dx right
            k = khi;
            for (i = ilo; i <= ihi; i++)
            {
                for (j = jlo; j <= jhi; j++)
                {
                    //grid->p->x[i*(npoi_x*npoi_y)+j*(npoi_x)+k]+=dxmax;
                    grid->p->x[i * (npoi_x * npoi_y) + j * (npoi_x) + k] = bxmin + k * grid->spacing_x + dxmax;
                }
            }
        }
        else
        {
            // move khi dx left
            k = khi;
            for (i = ilo; i <= ihi; i++)
            {
                for (j = jlo; j <= jhi; j++)
                {
                    //grid->p->x[i*(npoi_x*npoi_y)+j*(npoi_x)+k]-=grid->spacing_x-dxmax;
                    grid->p->x[i * (npoi_x * npoi_y) + j * (npoi_x) + k] = bxmin + k * grid->spacing_x - (grid->spacing_x - dxmax);
                }
            }
        }
    }

    /*
      // first try (is not very good, we have to do it with a bit more effort ...)
      for (i=ilo;i<=ihi;i++)	// z-dir.
       {
         for (j=jlo;j<=jhi;j++) 	// y-dir.
           {
            for (k=klo;k<=khi;k++)  // x-dir.
               {
                  if ( (i=ilo) || (i=ihi) || (j=jlo) || (j=jhi) || (k=klo) || (k=khi) ) // margin!!
                   {
                     dx=dxmin+(k-klo)*(dxmax-dxmin)/(khi-klo);
   dy=dymin+(j-jlo)*(dymax-dymin)/(jhi-jlo);
   dz=dzmin+(i-ilo)*(dzmax-dzmin)/(ihi-ilo);

   grid->p->x[i*(npoi_x*npoi_y)+j*(npoi_x)+k]+=dx;
   grid->p->y[i*(npoi_x*npoi_y)+j*(npoi_x)+k]+=dy;
   grid->p->z[i*(npoi_x*npoi_y)+j*(npoi_x)+k]+=dz;
   }
   }
   }
   }
   */

    // now: remove it!
    // we have to remove all nodes with index
    // ilo <= i <= ihi
    // jlo <= j <= jhi
    // klo <= k <= khi
    // and the corresponding elements
    int removed_nodes = 0;

    // in case our cube to subtract lies directly at the margin, we also have to subtract the outer nodes!
    int istart, iend, jstart, jend, kstart, kend;

    if (number != -1)
    {
        if (ilo == 0)
            istart = 0;
        else
            istart = ilo + 1;
        if (ihi == grid->nelem_z)
            iend = npoi_z;
        else
            iend = ihi;

        if (jlo == 0)
            jstart = 0;
        else
            jstart = jlo + 1;
        if (jhi == grid->nelem_y)
            jend = npoi_y;
        else
            jend = jhi;

        if (klo == 0)
            kstart = 0;
        else
            kstart = klo + 1;
        if (khi == grid->nelem_x)
            kend = npoi_x;
        else
            kend = khi;
        /*
         }
         else
          {
            istart=ilo+1;
            iend=ihi;
            jstart=jlo+1;
            jend=jhi;
            kstart=klo+1;
            kend=khi;
          }
      */
        for (i = istart; i < iend; i++) // z-dir.
        {
            for (j = jstart; j < jend; j++) // y-dir.
            {
                for (k = kstart; k < kend; k++) // x-dir.
                {
                    grid->p_to_remove[i * (npoi_x * npoi_y) + j * (npoi_x) + k] = -1;
                    removed_nodes++;
                }
            }
        }
        fprintf(stderr, "\tremoved %d nodes and ", removed_nodes);

        int removed_elements = 0;
        for (i = ilo; i < ihi; i++) // z-dir.
        {
            for (j = jlo; j < jhi; j++) // y-dir.
            {
                for (k = klo; k < khi; k++) // x-dir.
                {
                    grid->e_to_remove[i * (grid->nelem_x * grid->nelem_y) + j * (grid->nelem_x) + k] = -1;
                    //fprintf(stderr,"removed element nr. %d!\n",i*(grid->nelem_x*grid->nelem_y)+j*(grid->nelem_x)+k);
                    removed_elements++;
                }
            }
        }
        fprintf(stderr, "%d elements!\n", removed_elements);
    }

    model->cubes[number]->ilo = ilo;
    model->cubes[number]->ihi = ihi;
    model->cubes[number]->jlo = jlo;
    model->cubes[number]->jhi = jhi;
    model->cubes[number]->klo = klo;
    model->cubes[number]->khi = khi;

    // do we need to change existing bouundary conditions?

    // change bcs

    //fprintf(stderr,"ilo =%2d ihi =%2d jlo =%2d jhi =%2d klo =%2d khi =%2d\n",ilo,ihi,jlo,jhi,klo,khi);

    return 0;
}

int DefineBCs(struct christgrid *grid, struct christ_model *model, int number)
{
    /*
      int npoi_x = grid->npoi_x;
      int npoi_y = grid->npoi_y;
      int npoi_z = grid->npoi_z;
   */
    int nelem_x = grid->nelem_x;
    int nelem_y = grid->nelem_y;
    int nelem_z = grid->nelem_z;
    int ilo, ihi, jlo, jhi, klo, khi;

    // define bc types
    if (number == -1)
    {
        // this is the complete outside geometry!
        // walls -y ,+y, -z, +z
        // ilo,ihi,...: elements!
        model->ilo = 0;
        model->ihi = nelem_z;
        model->jlo = 0;
        model->jhi = nelem_y;
        model->klo = 0;
        model->khi = nelem_x;

        model->bc_type_minusy = INLET;
        model->bc_type_plusy = OUTLET;
        model->bc_type_minusz = WALL;
        model->bc_type_plusz = WALL; // hier muss aber noch eine Knoten-RB dazu (Klimaanlage)!

        model->bc_type_minusx = INLET;
        model->bc_type_plusx = OUTLET;

        ilo = model->ilo;
        ihi = model->ihi;
        jlo = model->jlo;
        jhi = model->jhi;
        klo = model->klo;
        khi = model->khi;
    }
    else
    {
        // walls: take care that there are no double walls!
        // ilo,ihi,...:
        ilo = model->cubes[number]->ilo;
        ihi = model->cubes[number]->ihi;
        jlo = model->cubes[number]->jlo;
        jhi = model->cubes[number]->jhi;
        klo = model->cubes[number]->klo;
        khi = model->cubes[number]->khi;

        /*
      // test
            if (ihi>=grid->npoi_z)
               model->cubes[number]->ihi=grid->npoi_z-1;
            if (jhi>=grid->npoi_y)
               model->cubes[number]->jhi=grid->npoi_y-1;
            if (khi>=grid->npoi_x)
               model->cubes[number]->khi=grid->npoi_x-1;
      */

        // z-direction
        // if ( (inside) or (outside and not normal outside wall) )

        if ((ilo != 0) || ((ilo == 0) && (model->bc_type_minusz != WALL)))
            model->cubes[number]->bc_type_minusz = WALL;
        else
            model->cubes[number]->bc_type_minusz = 0;
        if ((ihi != grid->npoi_z) || ((ihi == grid->npoi_z) && (model->bc_type_plusz != WALL)))
            model->cubes[number]->bc_type_plusz = WALL;
        else
            model->cubes[number]->bc_type_plusz = 0;

        // y-direction
        if ((jlo != 0) || ((jlo == 0) && (model->bc_type_minusy != WALL)))
            model->cubes[number]->bc_type_minusy = WALL;
        else
            model->cubes[number]->bc_type_minusy = 0;
        if ((jhi != grid->npoi_y) || ((jhi == grid->npoi_y) && (model->bc_type_plusy != WALL)))
            model->cubes[number]->bc_type_plusy = WALL;
        else
            model->cubes[number]->bc_type_plusy = 0;

        // x-direction
        if ((klo != 0) || ((klo == 0) && (model->bc_type_minusx != WALL)))
            model->cubes[number]->bc_type_minusx = WALL;
        else
            model->cubes[number]->bc_type_minusx = 0;
        if ((khi != grid->npoi_x) || ((khi == grid->npoi_x) && (model->bc_type_plusx != WALL)))
            model->cubes[number]->bc_type_plusx = WALL;
        else
            model->cubes[number]->bc_type_plusx = 0;

        if (number == 0)
        {
            model->cubes[number]->bc_special = 77;
        }
        else
        {
            model->cubes[number]->bc_special = 0;
        }
    }

    return 0;
}

int GenerateBCs(struct christgrid *grid, struct christ_model *model, int number)
{
    /*
      float xmin = -0.5*model->size[0];
       float xmax =  0.5*model->size[0];
      float ymin = -0.5*model->size[1];
       float ymax =  0.5*model->size[1];
      float zmin = -0.5*model->size[2];
       float zmax =  0.5*model->size[2];
   */
    int npoi_x = grid->npoi_x;
    int npoi_y = grid->npoi_y;
    int npoi_z = grid->npoi_z;
    int nelem_x = grid->nelem_x;
    int nelem_y = grid->nelem_y;
    //	int nelem_z = grid->nelem_z;

    int bcminusx = 0;
    int bcplusx = 0;
    int bcminusy = 0;
    int bcplusy = 0;
    int bcminusz = 0;
    int bcplusz = 0;

    int ilo, ihi, jlo, jhi, klo, khi;
    int i, j, k;

    if (number != -1)
    {
        // this is a cube to subtract
        ilo = model->cubes[number]->ilo;
        ihi = model->cubes[number]->ihi;
        jlo = model->cubes[number]->jlo;
        jhi = model->cubes[number]->jhi;
        klo = model->cubes[number]->klo;
        khi = model->cubes[number]->khi;
        bcminusx = model->cubes[number]->bc_type_minusx;
        bcplusx = model->cubes[number]->bc_type_plusx;
        bcminusy = model->cubes[number]->bc_type_minusy;
        bcplusy = model->cubes[number]->bc_type_plusy;
        bcminusz = model->cubes[number]->bc_type_minusz;
        bcplusz = model->cubes[number]->bc_type_plusz;
    }
    else
    {
        //ilo,ihi,...: elements!!!
        ilo = model->ilo;
        ihi = model->ihi;
        jlo = model->jlo;
        jhi = model->jhi;
        klo = model->klo;
        khi = model->khi;
        bcminusx = model->bc_type_minusx;
        bcplusx = model->bc_type_plusx;
        bcminusy = model->bc_type_minusy;
        bcplusy = model->bc_type_plusy;
        bcminusz = model->bc_type_minusz;
        bcplusz = model->bc_type_plusz;
    }

    int elem[6];

    // WALL WALL WALL
    // WALL WALL WALL
    // WALL WALL WALL
    // WALL WALL WALL

    // wall -x side
    if (bcminusx == WALL)
    {
        k = klo;
        for (i = ilo; i < ihi; i++)
        {
            for (j = jlo; j < jhi; j++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + npoi_x;
                elem[2] = elem[1] + (npoi_x * npoi_y);
                elem[3] = elem[0] + (npoi_x * npoi_y);
                if (number == -1)
                {
                    elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                    if (k == 0)
                        elem[5] = -1;
                    else
                        elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + (k - 1);
                }
                else
                {
                    elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                    if (k == 0)
                        elem[4] = -1;
                    else
                        elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + (k - 1);
                }
                Add2Ilist(grid->bcwallpol, grid->bcwall->num);
                Add2Ilist(grid->bcwall, elem[0]);
                Add2Ilist(grid->bcwall, elem[1]);
                Add2Ilist(grid->bcwall, elem[2]);
                Add2Ilist(grid->bcwall, elem[3]);
                //fprintf(stderr,"klo[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcwallvol, elem[4]);
                Add2Ilist(grid->bcwallvol_outer, elem[5]);
                if (elem[5] > grid->nelem)
                    fprintf(stderr, "error 1!\t");
            }
        }
    }
    /*
       // inlet node bcs
      if (number==-1)
       {
         grid->bcin_nodes=AllocIlistStruct(25);
           grid->bcin_velos=AllocFlistStruct(125);

           int bcnode;

         // Node-bc inlet (outer geometry)
           k=0;

   float l = 4.;	// characteristic length
   float _k   = 3.75 * 0.001  * pow(model->bcin_velo,2);
   float _eps = 9.40 * 0.0001 * pow(fabs(model->bcin_velo/l),3);

   for (i=ilo;i<ihi;i++)
   {
   for (j=jlo;j<jhi;j++)
   {
   bcnode = i*(npoi_x*npoi_y)+j*(npoi_x)+k;
   Add2Ilist(grid->bcin_nodes, bcnode);
   Add2Flist(grid->bcin_velos, model->bcin_velo);
   Add2Flist(grid->bcin_velos, 0.0);
   Add2Flist(grid->bcin_velos, 0.0);
   Add2Flist(grid->bcin_velos, _k);
   Add2Flist(grid->bcin_velos, _eps);
   }
   }
   }
   */

    // wall +x side
    if (bcplusx == WALL)
    {
        k = khi;
        for (i = ilo; i < ihi; i++)
        {
            for (j = jlo; j < jhi; j++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + npoi_x;
                elem[2] = elem[1] + (npoi_x * npoi_y);
                elem[3] = elem[0] + (npoi_x * npoi_y);
                if (number == -1)
                {
                    elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k - 1;
                    if (k == npoi_x - 1)
                        elem[5] = -1;
                    else
                        elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                }
                else
                {
                    elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + k - 1;
                    if (k == npoi_x - 1)
                        elem[4] = -1;
                    else
                        elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                }
                Add2Ilist(grid->bcwallpol, grid->bcwall->num);
                Add2Ilist(grid->bcwall, elem[0]);
                Add2Ilist(grid->bcwall, elem[1]);
                Add2Ilist(grid->bcwall, elem[2]);
                Add2Ilist(grid->bcwall, elem[3]);
                //fprintf(stderr,"khi[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcwallvol, elem[4]);
                Add2Ilist(grid->bcwallvol_outer, elem[5]);
                if (elem[5] > grid->nelem)
                    fprintf(stderr, "error 2!\t");
            }
        }
    }

    // wall -y side
    if (bcminusy == WALL)
    {
        j = jlo;
        for (i = ilo; i < ihi; i++)
        {
            for (k = klo; k < khi; k++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + 1;
                elem[2] = elem[1] + (npoi_x * npoi_y);
                elem[3] = elem[0] + (npoi_x * npoi_y);
                if (number == -1)
                {
                    elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                    if (j == 0)
                        elem[5] = -1;
                    else
                        elem[5] = i * (nelem_x * nelem_y) + (j - 1) * (nelem_x) + k;
                }
                else
                {
                    elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                    if (j == 0)
                        elem[4] = -1;
                    else
                        elem[4] = i * (nelem_x * nelem_y) + (j - 1) * (nelem_x) + k;
                }
                Add2Ilist(grid->bcwallpol, grid->bcwall->num);
                Add2Ilist(grid->bcwall, elem[0]);
                Add2Ilist(grid->bcwall, elem[1]);
                Add2Ilist(grid->bcwall, elem[2]);
                Add2Ilist(grid->bcwall, elem[3]);
                //fprintf(stderr,"jlo[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcwallvol, elem[4]);
                Add2Ilist(grid->bcwallvol_outer, elem[5]);
                if (elem[5] > grid->nelem)
                    fprintf(stderr, "error 3!\t");
            }
        }
    }

    // wall +y side
    if (bcplusy == WALL)
    {
        j = jhi;
        for (i = ilo; i < ihi; i++)
        {
            for (k = klo; k < khi; k++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + 1;
                elem[2] = elem[1] + (npoi_x * npoi_y);
                elem[3] = elem[0] + (npoi_x * npoi_y);
                if (number == -1)
                {
                    elem[4] = i * (nelem_x * nelem_y) + (j - 1) * (nelem_x) + k;
                    if (j == npoi_y - 1)
                        elem[5] = -1;
                    else
                        elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                }
                else
                {
                    elem[5] = i * (nelem_x * nelem_y) + (j - 1) * (nelem_x) + k;
                    if (j == npoi_y - 1)
                        elem[4] = -1;
                    else
                        elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                }
                Add2Ilist(grid->bcwallpol, grid->bcwall->num);
                Add2Ilist(grid->bcwall, elem[0]);
                Add2Ilist(grid->bcwall, elem[1]);
                Add2Ilist(grid->bcwall, elem[2]);
                Add2Ilist(grid->bcwall, elem[3]);
                //fprintf(stderr,"jhi[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcwallvol, elem[4]);
                Add2Ilist(grid->bcwallvol_outer, elem[5]);
                if (elem[5] > grid->nelem)
                    fprintf(stderr, "error 4!\t");
            }
        }
    }

    // wall -z side
    if (bcminusz == WALL)
    {
        i = ilo;
        for (j = jlo; j < jhi; j++)
        {
            for (k = klo; k < khi; k++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + 1;
                elem[2] = elem[1] + npoi_x;
                elem[3] = elem[0] + npoi_x;
                if (number == -1)
                {
                    elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                    if (i == 0)
                        elem[5] = -1;
                    else
                        elem[5] = (i - 1) * (nelem_x * nelem_y) + j * (nelem_x) + k;
                }
                else
                {
                    elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                    if (i == 0)
                        elem[4] = -1;
                    else
                        elem[4] = (i - 1) * (nelem_x * nelem_y) + j * (nelem_x) + k;
                }
                Add2Ilist(grid->bcwallpol, grid->bcwall->num);
                Add2Ilist(grid->bcwall, elem[0]);
                Add2Ilist(grid->bcwall, elem[1]);
                Add2Ilist(grid->bcwall, elem[2]);
                Add2Ilist(grid->bcwall, elem[3]);
                //fprintf(stderr,"ilo[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcwallvol, elem[4]);
                Add2Ilist(grid->bcwallvol_outer, elem[5]);
                if (elem[5] > grid->nelem)
                    fprintf(stderr, "error 5!\t");
            }
        }
    }

    // wall +z side
    if (bcplusz == WALL)
    {
        i = ihi;
        for (j = jlo; j < jhi; j++)
        {
            for (k = klo; k < khi; k++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + 1;
                elem[2] = elem[1] + npoi_x;
                elem[3] = elem[0] + npoi_x;
                if (number == -1)
                {
                    elem[4] = (i - 1) * (nelem_x * nelem_y) + j * (nelem_x) + k;
                    if (i == npoi_z - 1)
                        elem[5] = -1;
                    else
                        elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                }
                else
                {
                    elem[5] = (i - 1) * (nelem_x * nelem_y) + j * (nelem_x) + k;
                    if (i == npoi_z - 1)
                        elem[4] = -1;
                    else
                        elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                }
                Add2Ilist(grid->bcwallpol, grid->bcwall->num);
                Add2Ilist(grid->bcwall, elem[0]);
                Add2Ilist(grid->bcwall, elem[1]);
                Add2Ilist(grid->bcwall, elem[2]);
                Add2Ilist(grid->bcwall, elem[3]);
                //fprintf(stderr,"ihi[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcwallvol, elem[4]);
                Add2Ilist(grid->bcwallvol_outer, elem[5]);
                if (elem[5] > grid->nelem)
                    fprintf(stderr, "error 6!\t");
            }
        }
    }
    // airconditioning
    /* not here in christmas06 */

    // ven
    /* not here in christmas06 */

    // INLET INLET
    // INLET INLET
    // INLET INLET
    // INLET INLET

    //int *inlet_nodes = (int*)calloc(grid->npoi, sizeof(int));

    // inlet -x side
    if (bcminusx == INLET)
    {
        k = klo;
        for (i = ilo; i < ihi; i++)
        {
            for (j = jlo; j < jhi; j++)
            {

                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + npoi_x;
                elem[2] = elem[1] + (npoi_x * npoi_y);
                elem[3] = elem[0] + (npoi_x * npoi_y);
                elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                /*
            // save inlet nodes
            inlet_nodes[elem[0]]=1;
            inlet_nodes[elem[1]]=1;
            inlet_nodes[elem[2]]=1;
            inlet_nodes[elem[3]]=1;
            */
                if (k == 0)
                    elem[5] = -1;
                else
                    elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + (k - 1);
                Add2Ilist(grid->bcinpol, grid->bcin->num);
                Add2Ilist(grid->bcin, elem[0]);
                Add2Ilist(grid->bcin, elem[1]);
                Add2Ilist(grid->bcin, elem[2]);
                Add2Ilist(grid->bcin, elem[3]);
                //fprintf(stderr,"klo[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcinvol, elem[4]);
                Add2Ilist(grid->bcinvol_outer, elem[5]);
            }
        }
    }
    /*
      // inlet node bcs
      if (number==-1)
       {
         grid->bcin_nodes=AllocIlistStruct(25);
           grid->bcin_velos=AllocFlistStruct(125);

           int bcnode;

         // Node-bc inlet (outer geometry)
           k=0;

   float l = 4.;	// characteristic length
   float _k   = 3.75 * 0.001  * pow(model->bcin_velo,2);
   float _eps = 9.40 * 0.0001 * pow(fabs(model->bcin_velo/l),3);

   for (i=0; i<grid->npoi;i++)
   {
   if (inlet_nodes[i]==1)
   {
   bcnode = i;
   Add2Ilist(grid->bcin_nodes, bcnode);
   Add2Flist(grid->bcin_velos, model->bcin_velo);
   Add2Flist(grid->bcin_velos, 0.0);
   Add2Flist(grid->bcin_velos, 0.0);
   Add2Flist(grid->bcin_velos, _k);
   Add2Flist(grid->bcin_velos, _eps);
   }
   }
   }
   */

    // inlet +x side
    if (bcplusx == INLET)
    {
        k = khi;
        for (i = ilo; i < ihi; i++)
        {
            for (j = jlo; j < jhi; j++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + npoi_x;
                elem[2] = elem[1] + (npoi_x * npoi_y);
                elem[3] = elem[0] + (npoi_x * npoi_y);
                elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k - 1;
                if (k == npoi_x - 1)
                    elem[5] = -1;
                else
                    elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                Add2Ilist(grid->bcinpol, grid->bcin->num);
                Add2Ilist(grid->bcin, elem[0]);
                Add2Ilist(grid->bcin, elem[1]);
                Add2Ilist(grid->bcin, elem[2]);
                Add2Ilist(grid->bcin, elem[3]);
                //fprintf(stderr,"khi[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcinvol, elem[4]);
                Add2Ilist(grid->bcinvol_outer, elem[5]);
            }
        }
    }

    // inlet -y side
    if (bcminusy == INLET)
    {
        j = jlo;
        for (i = ilo; i < ihi; i++)
        {
            for (k = klo; k < khi; k++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + 1;
                elem[2] = elem[1] + (npoi_x * npoi_y);
                elem[3] = elem[0] + (npoi_x * npoi_y);
                elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                if (j == 0)
                    elem[5] = -1;
                else
                    elem[5] = i * (nelem_x * nelem_y) + (j - 1) * (nelem_x) + k;
                Add2Ilist(grid->bcinpol, grid->bcin->num);
                Add2Ilist(grid->bcin, elem[0]);
                Add2Ilist(grid->bcin, elem[1]);
                Add2Ilist(grid->bcin, elem[2]);
                Add2Ilist(grid->bcin, elem[3]);
                //fprintf(stderr,"jlo[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcinvol, elem[4]);
                Add2Ilist(grid->bcinvol_outer, elem[5]);
            }
        }
    }

    // inlet +y side
    if (bcplusy == INLET)
    {
        j = jhi;
        for (i = ilo; i < ihi; i++)
        {
            for (k = klo; k < khi; k++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + 1;
                elem[2] = elem[1] + (npoi_x * npoi_y);
                elem[3] = elem[0] + (npoi_x * npoi_y);
                elem[4] = i * (nelem_x * nelem_y) + (j - 1) * (nelem_x) + k;
                if (j == npoi_y - 1)
                    elem[5] = -1;
                else
                    elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                Add2Ilist(grid->bcinpol, grid->bcin->num);
                Add2Ilist(grid->bcin, elem[0]);
                Add2Ilist(grid->bcin, elem[1]);
                Add2Ilist(grid->bcin, elem[2]);
                Add2Ilist(grid->bcin, elem[3]);
                //fprintf(stderr,"jhi[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcinvol, elem[4]);
                Add2Ilist(grid->bcinvol_outer, elem[5]);
            }
        }
    }

    // inlet -z side
    if (bcminusz == INLET)
    {
        i = ilo;
        for (j = jlo; j < jhi; j++)
        {
            for (k = klo; k < khi; k++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + 1;
                elem[2] = elem[1] + npoi_x;
                elem[3] = elem[0] + npoi_x;
                elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                if (i == 0)
                    elem[5] = -1;
                else
                    elem[5] = (i - 1) * (nelem_x * nelem_y) + j * (nelem_x) + k;
                Add2Ilist(grid->bcinpol, grid->bcin->num);
                Add2Ilist(grid->bcin, elem[0]);
                Add2Ilist(grid->bcin, elem[1]);
                Add2Ilist(grid->bcin, elem[2]);
                Add2Ilist(grid->bcin, elem[3]);
                //fprintf(stderr,"ilo[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcinvol, elem[4]);
                Add2Ilist(grid->bcinvol_outer, elem[5]);
            }
        }
    }

    // inlet +z side
    if (bcplusz == INLET)
    {
        i = ihi;
        for (j = jlo; j < jhi; j++)
        {
            for (k = klo; k < khi; k++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + 1;
                elem[2] = elem[1] + npoi_x;
                elem[3] = elem[0] + npoi_x;
                elem[4] = (i - 1) * (nelem_x * nelem_y) + j * (nelem_x) + k;
                if (i == npoi_z - 1)
                    elem[5] = -1;
                else
                    elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                Add2Ilist(grid->bcinpol, grid->bcin->num);
                Add2Ilist(grid->bcin, elem[0]);
                Add2Ilist(grid->bcin, elem[1]);
                Add2Ilist(grid->bcin, elem[2]);
                Add2Ilist(grid->bcin, elem[3]);
                //fprintf(stderr,"ihi[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcinvol, elem[4]);
                Add2Ilist(grid->bcinvol_outer, elem[5]);
            }
        }
    }

    // OUTLET
    // OUTLET
    // OUTLET
    // OUTLET

    // outlet -x side
    if (bcminusx == OUTLET)
    {
        k = klo;
        for (i = ilo; i < ihi; i++)
        {
            for (j = jlo; j < jhi; j++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + npoi_x;
                elem[2] = elem[1] + (npoi_x * npoi_y);
                elem[3] = elem[0] + (npoi_x * npoi_y);
                elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                if (k == 0)
                    elem[5] = -1;
                else
                    elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + (k - 1);
                Add2Ilist(grid->bcoutpol, grid->bcout->num);
                Add2Ilist(grid->bcout, elem[0]);
                Add2Ilist(grid->bcout, elem[1]);
                Add2Ilist(grid->bcout, elem[2]);
                Add2Ilist(grid->bcout, elem[3]);
                //fprintf(stderr,"klo[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcoutvol, elem[4]);
                Add2Ilist(grid->bcoutvol_outer, elem[5]);
            }
        }
    }

    // outlet +x side
    if (bcplusx == OUTLET)
    {
        k = khi;
        for (i = ilo; i < ihi; i++)
        {
            for (j = jlo; j < jhi; j++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + npoi_x;
                elem[2] = elem[1] + (npoi_x * npoi_y);
                elem[3] = elem[0] + (npoi_x * npoi_y);
                elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k - 1;
                if (k == npoi_x - 1)
                    elem[5] = -1;
                else
                    elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                Add2Ilist(grid->bcoutpol, grid->bcout->num);
                Add2Ilist(grid->bcout, elem[0]);
                Add2Ilist(grid->bcout, elem[1]);
                Add2Ilist(grid->bcout, elem[2]);
                Add2Ilist(grid->bcout, elem[3]);
                //fprintf(stderr,"khi[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcoutvol, elem[4]);
                Add2Ilist(grid->bcoutvol_outer, elem[5]);
            }
        }
    }

    // outlet -y side
    if (bcminusy == OUTLET)
    {
        j = jlo;
        for (i = ilo; i < ihi; i++)
        {
            for (k = klo; k < khi; k++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + 1;
                elem[2] = elem[1] + (npoi_x * npoi_y);
                elem[3] = elem[0] + (npoi_x * npoi_y);
                elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                if (j == 0)
                    elem[5] = -1;
                else
                    elem[5] = i * (nelem_x * nelem_y) + (j - 1) * (nelem_x) + k;
                Add2Ilist(grid->bcoutpol, grid->bcout->num);
                Add2Ilist(grid->bcout, elem[0]);
                Add2Ilist(grid->bcout, elem[1]);
                Add2Ilist(grid->bcout, elem[2]);
                Add2Ilist(grid->bcout, elem[3]);
                //fprintf(stderr,"jlo[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcoutvol, elem[4]);
                Add2Ilist(grid->bcoutvol_outer, elem[5]);
            }
        }
    }

    // outlet +y side
    if (bcplusy == OUTLET)
    {
        j = jhi;
        for (i = ilo; i < ihi; i++)
        {
            for (k = klo; k < khi; k++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + 1;
                elem[2] = elem[1] + (npoi_x * npoi_y);
                elem[3] = elem[0] + (npoi_x * npoi_y);
                elem[4] = i * (nelem_x * nelem_y) + (j - 1) * (nelem_x) + k;
                if (j == npoi_y - 1)
                    elem[5] = -1;
                else
                    elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                Add2Ilist(grid->bcoutpol, grid->bcout->num);
                Add2Ilist(grid->bcout, elem[0]);
                Add2Ilist(grid->bcout, elem[1]);
                Add2Ilist(grid->bcout, elem[2]);
                Add2Ilist(grid->bcout, elem[3]);
                //fprintf(stderr,"jhi[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcoutvol, elem[4]);
                Add2Ilist(grid->bcoutvol_outer, elem[5]);
            }
        }
    }

    // outlet -z side
    if (bcminusz == OUTLET)
    {
        i = ilo;
        for (j = jlo; j < jhi; j++)
        {
            for (k = klo; k < khi; k++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + 1;
                elem[2] = elem[1] + npoi_x;
                elem[3] = elem[0] + npoi_x;
                elem[4] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                if (i == 0)
                    elem[5] = -1;
                else
                    elem[5] = (i - 1) * (nelem_x * nelem_y) + j * (nelem_x) + k;
                Add2Ilist(grid->bcoutpol, grid->bcout->num);
                Add2Ilist(grid->bcout, elem[0]);
                Add2Ilist(grid->bcout, elem[1]);
                Add2Ilist(grid->bcout, elem[2]);
                Add2Ilist(grid->bcout, elem[3]);
                //fprintf(stderr,"ilo[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcoutvol, elem[4]);
                Add2Ilist(grid->bcoutvol_outer, elem[5]);
            }
        }
    }

    // outlet +z side
    if (bcplusz == OUTLET)
    {
        i = ihi;
        for (j = jlo; j < jhi; j++)
        {
            for (k = klo; k < khi; k++)
            {
                elem[0] = i * (npoi_x * npoi_y) + j * (npoi_x) + k;
                elem[1] = elem[0] + 1;
                elem[2] = elem[1] + npoi_x;
                elem[3] = elem[0] + npoi_x;
                elem[4] = (i - 1) * (nelem_x * nelem_y) + j * (nelem_x) + k;
                if (i == npoi_z - 1)
                    elem[5] = -1;
                else
                    elem[5] = i * (nelem_x * nelem_y) + j * (nelem_x) + k;
                Add2Ilist(grid->bcoutpol, grid->bcout->num);
                Add2Ilist(grid->bcout, elem[0]);
                Add2Ilist(grid->bcout, elem[1]);
                Add2Ilist(grid->bcout, elem[2]);
                Add2Ilist(grid->bcout, elem[3]);
                //fprintf(stderr,"ihi[%d], added vertices %d %d %d %d\n", number, elem[0], elem[1], elem[2], elem[3]);
                Add2Ilist(grid->bcoutvol, elem[4]);
                Add2Ilist(grid->bcoutvol_outer, elem[5]);
            }
        }
    }

    return 0;
}

int CreateGeoRbFile(struct christgrid *grid, const char *geofile, const char *rbfile)
{
    int i;

    FILE *stream;
    char file[512];

    // GEOFILE
    strcpy(file, geofile);
    if ((stream = fopen(&file[0], "w")) == NULL)
    {
        printf("Can't open %s for writing!\n", file);
    }
    else
    {
        for (i = 0; i < 10; i++)
        {
            fprintf(stream, "C\n");
        }
        fprintf(stream, "%d %d %d %d %d %d %d %d\n", grid->p->nump, grid->e->nume, 0, 0, 0, 0, grid->p->nump, grid->e->nume);

        for (i = 0; i < grid->p->nump; i++)
        {
            fprintf(stream, "%6d  %8.5f  %8.5f  %8.5f\n", i + 1, grid->p->x[i], grid->p->y[i], grid->p->z[i]);
        }

        for (i = 0; i < grid->e->nume; i++)
        {
            fprintf(stream, "%10d %6d %6d %6d %6d %6d %6d %6d %6d\n", i + 1,
                    grid->e->e[i][0] + 1,
                    grid->e->e[i][1] + 1,
                    grid->e->e[i][2] + 1,
                    grid->e->e[i][3] + 1,
                    grid->e->e[i][4] + 1,
                    grid->e->e[i][5] + 1,
                    grid->e->e[i][6] + 1,
                    grid->e->e[i][7] + 1);
        }

        fclose(stream);
    }

    // RBFILE
    strcpy(file, rbfile);
    if ((stream = fopen(&file[0], "w")) == NULL)
    {
        printf("Can't open %s for writing!\n", file);
    }
    else
    {
        for (i = 0; i < 10; i++)
        {
            fprintf(stream, "C\n");
        }
        int n_bila = grid->bcinvol->num + grid->bcoutvol->num;
        int n_nodal = grid->bcin_nodes->num;

        fprintf(stream, "%d %d %d %d %d %d %d %d\n", n_nodal * 5, grid->bcwallpol->num, 0, 0, 0, 0, n_bila, 0);

        for (i = 0; i < grid->bcin_nodes->num; i++)
        {
            fprintf(stream, "%6d 1 %8.5f\n", grid->bcin_nodes->list[i] + 1, grid->bcin_velos->list[5 * i + 0]);
            fprintf(stream, "%6d 2 %8.5f\n", grid->bcin_nodes->list[i] + 1, grid->bcin_velos->list[5 * i + 1]);
            fprintf(stream, "%6d 3 %8.5f\n", grid->bcin_nodes->list[i] + 1, grid->bcin_velos->list[5 * i + 2]);
            fprintf(stream, "%6d 4 %8.5f\n", grid->bcin_nodes->list[i] + 1, grid->bcin_velos->list[5 * i + 3]);
            fprintf(stream, "%6d 5 %8.5f\n", grid->bcin_nodes->list[i] + 1, grid->bcin_velos->list[5 * i + 4]);
        }

        for (i = 0; i < grid->bcwallvol->num; i++)
        {
            fprintf(stream, "%6d %6d %6d %6d %d %d %d %6d\n", grid->bcwall->list[4 * i + 0] + 1,
                    grid->bcwall->list[4 * i + 1] + 1,
                    grid->bcwall->list[4 * i + 2] + 1,
                    grid->bcwall->list[4 * i + 3] + 1,
                    0, 0, 0,
                    grid->bcwallvol->list[i] + 1);
        }

        for (i = 0; i < grid->bcinvol->num; i++)
        {
            fprintf(stream, "%6d %6d %6d %6d %6d %6d\n", grid->bcin->list[4 * i + 0] + 1,
                    grid->bcin->list[4 * i + 1] + 1,
                    grid->bcin->list[4 * i + 2] + 1,
                    grid->bcin->list[4 * i + 3] + 1,
                    grid->bcinvol->list[i] + 1,
                    100);
        }

        for (i = 0; i < grid->bcoutvol->num; i++)
        {
            fprintf(stream, "%6d %6d %6d %6d %6d %6d\n", grid->bcout->list[4 * i + 0] + 1,
                    grid->bcout->list[4 * i + 1] + 1,
                    grid->bcout->list[4 * i + 2] + 1,
                    grid->bcout->list[4 * i + 3] + 1,
                    grid->bcoutvol->list[i] + 1,
                    110);
        }

        fclose(stream);
    }

    return 0;
}
