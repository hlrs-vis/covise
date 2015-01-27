/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FireUniversalFile.h"
#include <appl/ApplInterface.h>

#ifdef CO_linux
#define CONVERT
#endif

FireFile::FireFile(char *n)
{
    //	cerr << "opening file: " << n << endl;

    hdl = Covise::fopen(n, "r");
    if (hdl == NULL)
    {
        perror("trouble reading Fire File");
        cerr << "Wrong filename: " << n << endl;
        name = 0L;
        return;
    }
    name = new char[strlen(n) + 1];
    strcpy(name, n);
}

int FireFile::skip_block()
{
    int ident;

    if (read_line() == NULL)
        return 0;
    printf("first line in skip_block: %s\n", line);
    sscanf(line, "%d", &ident);
    while (ident != -1)
    {
        Covise::sendInfo(line);
        if (read_line() == NULL)
            return 0;
        sscanf(line, "%d", &ident);
    }
    printf("after -1: %s\n", line);
    if (read_line() == NULL)
        return 0;
    sscanf(line, "%d", &ident);
    while (ident != -1)
    {
        if (read_line() == NULL)
            return 0;
        sscanf(line, "%d", &ident);
    }
    return 1;
}

int FireFile::is_minus_one()
{
    int count = 0;

    while (line[count] == ' ' || line[count] == '\t')
        count++;

    if (line[count] == '-' && line[count + 1] == '1' && line[count + 2] == '\n')
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int FireFile::read_nodes(int &len)
{
    int ident;

    // search for -1 that indicates start of block
    if (read_line() == NULL)
        return 0;
    printf("first line in read_nodes: %s\n", line);
    while (!is_minus_one())
    {
        Covise::sendInfo(line);
        if (read_line() == NULL)
            return 0;
    }
    printf("after -1: %s\n", line);
    if (read_line() == NULL)
        return 0;
    sscanf(line, "%d", &ident);
    // verify that block has id 2411 for nodes
    if (ident != 2411)
    {
        Covise::sendError("ERROR: didn't find nodes in file (id != 2411)");
        return 0;
    }
    else
    {
        printf("found ident: %s \n", line);
    }

    // remember beginning of grid point data
    start_gridpoints = set_fseek();
    //    cerr << "start_gridpoints: " << start_gridpoints << endl;

    // now count grid points
    // always two lines: first with number, second with coordinates
    int no_of_grid_points = 0;
    if (read_line() == NULL)
        return 0;
    while (!is_minus_one())
    {
        no_of_grid_points++;
        if (read_line() == NULL)
            return 0;
        if (read_line() == NULL)
            return 0;
    }
    len = no_of_grid_points;
    return len;
}

int FireFile::read_nodes(int len, float *x, float *y, float *z)
{
    //    cerr << "going to start_gridpoints: " << start_gridpoints << endl;
    goto_fseek(start_gridpoints);

    for (int i = 0; i < len; i++)
    {
        if (read_line() == NULL)
            return 0;
        if (read_line() == NULL)
            return 0;
        sscanf(line, "%f %f %f", &x[i], &y[i], &z[i]);
    }
    return len;
}

int FireFile::read_elements(int &len)
{
    int ident;

    // search for -1 that indicates start of block
    if (read_line() == NULL)
        return 0;
    while (!is_minus_one())
    {
        Covise::sendInfo(line);
        if (read_line() == NULL)
            return 0;
    }
    if (read_line() == NULL)
        return 0;
    sscanf(line, "%d", &ident);
    // verify that block has id 2412 for elements
    if (ident != 2412)
    {
        Covise::sendError("ERROR: didn't find elements in file (id != 2412)");
        return 0;
    }
    else
    {
        printf("found ident: %s \n", line);
    }

    // remember beginning of grid point data
    start_elements = set_fseek();
    //    cerr << "start_elements: " << start_elements << endl;

    // now count elements
    // always two lines: first with number, second with coordinates
    int no_of_elements = 0;
    if (read_line() == NULL)
        return 0;
    while (!is_minus_one())
    {
        no_of_elements++;
        if (read_line() == NULL)
            return 0;
        if (read_line() == NULL)
            return 0;
    }
    len = no_of_elements;
    return len;
}

int FireFile::read_elements(int len, int *elem, int *conn)
{
    int no, v1, v2, v3, v4, v5;

    //    cerr << "going to start_elements: " << start_elements << endl;
    goto_fseek(start_elements);
    for (int i = 0; i < len; i++)
    {
        if (read_line() == NULL)
            return 0;
        //        sscanf(line, "%d %d %d %d %d %d", &no, &v1, &v2, &v3, &v4, &v5);
        //        if(v1 != 115 || v2 != 1 || v3 != 1 || v4 != 7 || v5 != 8)
        //        	cerr << "**** " << line;
        if (read_line() == NULL)
            return 0;
        elem[i] = 8 * i;
        sscanf(line, "%d %d %d %d %d %d %d %d",
               &conn[8 * i + 0],
               &conn[8 * i + 1],
               &conn[8 * i + 2],
               &conn[8 * i + 3],
               &conn[8 * i + 4],
               &conn[8 * i + 5],
               &conn[8 * i + 6],
               &conn[8 * i + 7]);

        conn[8 * i + 0]--;
        conn[8 * i + 1]--;
        conn[8 * i + 2]--;
        conn[8 * i + 3]--;
        conn[8 * i + 4]--;
        conn[8 * i + 5]--;
        conn[8 * i + 6]--;
        conn[8 * i + 7]--;
    }

    return len;
}

int FireFile::determine_data(int no_of_elements, long &data_start,
                             char **data_name, int &data_type)
{
    int dummy, i;
    int ident;
    char buffer[255];

    // search for -1 that indicates start of block
    if (read_line() == NULL)
        return 0;
    //    cout << line;
    while (!is_minus_one())
    {
        Covise::sendInfo(line);
        if (read_line() == NULL)
            return 0;
        //        cout << line;
    }
    if (read_line() == NULL)
        return 0;
    //    cout << line;
    sscanf(line, "%d", &ident);
    // verify that block has id 2414 for data
    if (ident != 2414)
    {
        Covise::sendError("ERROR: didn't find elements in file (id != 2414)");
        return 0;
    }
    else
    {
        //        printf("found ident: %s \n", line);
    }
    if (read_line() == NULL)
        return 0;
    cout << line << "now through\n";
    ;
    if (read_line() == NULL)
        return 0;
    if (read_line() == NULL)
        return 0;
    if (read_line() == NULL)
        return 0;
    if (read_line() == NULL)
        return 0;
    if (read_line() == NULL)
        return 0;
    if (read_line() == NULL)
        return 0;
    int line_end = strlen(line);
    *data_name = new char[strlen(line) + 1];
    strcpy(*data_name, line);
    char *tmp_ptr = *data_name;
    for (i = 0; i < strlen(*data_name); i++)
        if ((*data_name)[i] == ' ')
            (*data_name)[i] = '_';

    for (i = line_end - 2; (*data_name)[i] == '_'; i--)
    {
        (*data_name)[i] = 0;
    }

    //    cout << "name: " << line;
    if (read_line() == NULL)
        return 0;
    if (read_line() == NULL)
        return 0;
    if (read_line() == NULL)
        return 0;
    if (read_line() == NULL)
        return 0;
    if (read_line() == NULL)
        return 0;
    if (read_line() == NULL)
        return 0;
    data_start = set_fseek();
    if (read_line() == NULL)
        return 0;
    sscanf(line, "%d %d", &dummy, &data_type);
    //    cout << "data type: " << data_type << endl;
    if (read_line() == NULL)
        return 0;
    while (!is_minus_one())
    {
        if (read_line() == NULL)
            return 0;
        //        if(read_line() == NULL) return 0;
    }
    //    cout << "return 1\n";
    return 1;
}

int FireFile::read_data(int no_of_grid_points, long data_start,
                        float *s)
{
    //    cerr << "going to data_start: " << data_start << endl;
    goto_fseek(data_start);
    for (int i = 0; i < no_of_grid_points; i++)
    {
        if (read_line() == NULL)
            return 0;
        if (read_line() == NULL)
            return 0;
        sscanf(line, "%f", &s[i]);
    }
}

int FireFile::read_data(int no_of_grid_points, long data_start,
                        float *u, float *v, float *w)
{
    int no, t;

    //    cerr << "going to data_start: " << data_start << endl;
    goto_fseek(data_start);
    for (int i = 0; i < no_of_grid_points; i++)
    {
        if (read_line() == NULL)
            return 0;
        sscanf(line, "%d %d", &no, &t);
        if (i == 0)
            cerr << "reading data element " << no << endl;
        if (read_line() == NULL)
            return 0;
        sscanf(line, "%f %f %f", &u[i], &v[i], &w[i]);
        if (i == 0)
            cerr << u[i] << " " << v[i] << " " << w[i] << endl;
    }
}
