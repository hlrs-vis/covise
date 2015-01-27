/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "erg2cov.h"
#include "ergResultNames.h"
#include <file/covWriteFiles.h>
#include <sys/stat.h>
#include <util/coLog.h>

void readString(FILE *f, string &m)
{
    char line = 1;

    while ((line = fgetc(f)))
    {
        m += line;
    }
}

void readResultNames(vector<string> &results)
{
    int i = 0;
    results.push_back("INIT");
    while (resultNames[i] != NULL)
    {
        results.push_back(resultNames[i++]);
    }
}

void printUsage()
{
    cout << endl;
    cout << "Usage: erg2cov <filename> [outdir]" << endl << endl;
    cout << "Convert erg file <filename> into directory \"erg-covise\". This directory is by default" << endl;
    cout << "created in the directory of <filename>. Otherwise in <outdir>" << endl << endl;
}

int main(int argc, char **argv)
{
    cout << "*************************************************" << endl;
    cout << "*                                               *" << endl;
    cout << "* (C) 2005 Visenso GmbH                         *" << endl;
    cout << "*                                               *" << endl;
    cout << "*                                               *" << endl;
    cout << "*  erg2cov: convert erg file to covise project  *" << endl;
    cout << "*                                               *" << endl;
    cout << "*                                               *" << endl;
    cout << "*************************************************" << endl;

    if (argc == 1 || argc > 3)
    {
        printUsage();
        return 0;
    }

    //read variable names
    vector<string> res_names;

    readResultNames(res_names);

    //covgrp result file
    string covName;
    string project;
    string dir(argv[1]);

    // extract dir
    string::size_type d = dir.rfind("/", dir.size() - 1);
    if (d == string::npos)
    {
        project = dir;
        dir.erase();
    }
    else
    {
        project = dir.substr(d + 1, dir.size());
        dir.erase(d, dir.size());
    }

    // extract project file
    string::size_type it = project.rfind(".", project.size() - 1);
    if (it != string::npos)
    {
        project.erase(it, project.size());
    }

    string outDir;
    if (argc == 3)
    {
        outDir = string(argv[2]) + "/erg-covise/";
    }
    else
    {
        outDir = dir + "/erg-covise/";
    }

    mkdir(outDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH);

    covName = outDir + project + ".covgrp";
    FILE *covgrp = fopen(covName.c_str(), "w");
    if (!covgrp)
    {
        cerr << "Unable to open " << covName.c_str() << endl;
        cerr << "Aborted. " << endl << endl;
        return 0;
    }

    char *filename = argv[1];
    cout << endl << "Converting file ...                " << filename << endl << endl;

    FILE *file = fopen(filename, "r");
    if (!file)
    {
        cerr << "Unable to open " << filename << endl;
        cerr << "Aborted. " << endl << endl;
        return 0;
    }

    size_t iret;
    iret = fread(&Header.error, 1, 1, file);
    iret = fread(&Header.version, 1, 4, file);
    iret = fread(&Header.result_list_pos, 1, 4, file);
    iret = fread(&Header.num_nodes, 1, 4, file);
    iret = fread(&Header.num_elements, 1, 4, file);
    iret = fread(&Header.error, 1, 1, file);
    (void)iret;

    // read grid information
    MeshData mesh(Header.num_nodes, Header.num_elements);
    mesh.readData(file);
    float *x0, *y0, *z0;
    mesh.get_nodes(&x0, &y0, &z0);

    int cov_id = covOpenOutFile((char *)(outDir + "grid.covise").c_str());
    mesh.writeCovFile(cov_id);
    covCloseOutFile(cov_id);
    fputs("Gitter:grid.covise\n", covgrp);

    // read results
    fseek(file, Header.result_list_pos, SEEK_SET);

    //cout << Header.version << " " << Header.result_list_pos  << " " << Header.num_nodes << " " << Header.num_elements <<endl;

    readString(file, ResultList.remark);

    //cout <<  ResultList.remark << endl;

    iret = fread(&ResultList.num_results, 1, 4, file);
    //cout << ResultList.num_results << endl;

    int i, pos;
    char type, id, char_id;

    string title;

    for (i = 0; i < ResultList.num_results; i++)
    {
        title.erase();
        readString(file, title);

        iret = fread(&type, 1, 1, file);
        iret = fread(&pos, 1, 4, file);
        iret = fread(&id, 1, 1, file);
        iret = fread(&char_id, 1, 1, file);

        //Result::ResultType rtype = (Result::ResultType) type;

        bool success = true;

        long ff = ftell(file);
        cout << endl << "Parsing: " << title << endl << "Type: " << type << endl;

        string::size_type slash = title.find("/", 0);
        if (slash != string::npos)
        {
            title.erase(slash, title.size());
        }

        string fileName = outDir + title + ".covise";
        cov_id = covOpenOutFile((char *)fileName.c_str());

        // replace LayerNr if possible
        int resType = atoi(title.c_str());
        if (resType > 346)
        {
            i = ResultList.num_results;
            cerr << "Unknown result type" << endl;
            break;
        }

        string resultName(res_names[atoi(title.c_str())]);
        string::size_type cross = resultName.find("#", 0);
        if (cross != string::npos)
        {
            string::size_type number = title.find("-", 0);
            if (number != string::npos)
            {
                string part = title.substr(number, title.size());
                resultName.replace(cross, 1, part);
            }
        }

        if (type == Result::NODES_MORPH)
        {
            NodesMorphResult result(Header.num_nodes,
                                    resultName,
                                    pos,
                                    (Result::ResultIdentifier)id,
                                    char_id);

            result.set_node_base(x0, y0, z0);
            result.readValues(file);
            result.writeCovFile(cov_id);
        }
        else if (type == Result::NODES)
        {
            NodesResult result(Header.num_nodes,
                               resultName,
                               pos,
                               (Result::ResultIdentifier)id,
                               char_id);

            result.readValues(file);
            result.writeCovFile(cov_id);
        }
        else if (type == Result::ELEMENTS)
        {
            ElemResult result(Header.num_elements,
                              resultName,
                              pos,
                              (Result::ResultIdentifier)id,
                              char_id);

            result.readValues(file);
            result.writeCovFile(cov_id);
        }
        else if (type == Result::NODES_ANIMATION)
        {
            NodesAnimationResult result(Header.num_nodes,
                                        resultName,
                                        pos,
                                        (Result::ResultIdentifier)id,
                                        char_id);

            result.readValues(file);
            result.writeCovFile(cov_id);
        }
        else if (type == Result::VECTOR_ANIMATION)
        {
            VectorAnimationResult result(Header.num_elements,
                                         resultName,
                                         pos,
                                         (Result::ResultIdentifier)id,
                                         char_id);

            result.readValues(file, mesh);
            result.writeCovFileValues(cov_id);

            string pointFile = outDir + title + "_points.covise";
            int point_id = covOpenOutFile((char *)pointFile.c_str());
            result.writeCovFilePoints(point_id);
        }
        else
        {
            cout << "WARNING: Variable " << resultName << " not converted because type #"
                 << (int)type << " is not yet implemented" << endl;
            success = false;
        }
        fseek(file, ff, SEEK_SET);
        covCloseOutFile(cov_id);

        if (!success)
        {
            string rm_call = string("rm '") + fileName + string("'");
            int iret = system(rm_call.c_str());
            if (iret == -1)
                return -1;
        }
        else
        {
            string cov_entry = resultName + string(":") + fileName + string("\n");
            fputs(cov_entry.c_str(), covgrp);
        }
    }
    fclose(file);
    fclose(covgrp);
    cout << endl << "Finished succesfully." << endl;
    return 1;
}

MeshData::MeshData(int num_nodes, int num_elements)
{
    coords_ = new float[3 * num_nodes];
    conn_ = new int[3 * num_elements];

    num_nodes_ = num_nodes;
    num_elements_ = num_elements;

    tl_ = new int[num_elements_];
    pl_ = new int[num_elements_];
    cl_ = new int[3 * num_elements_];
    x_ = new float[num_nodes_];
    y_ = new float[num_nodes_];
    z_ = new float[num_nodes_];

    num_cl_ = 0;
}

void
MeshData::get_nodes(float **x0, float **y0, float **z0)
{
    *x0 = x_;
    *y0 = y_;
    *z0 = z_;
}

MeshData::~MeshData()
{
    delete[] coords_;
    delete[] conn_;
    delete[] tl_;
    delete[] pl_;
    delete[] cl_;
    delete[] x_;
    delete[] y_;
    delete[] z_;
}

void
MeshData::readData(FILE *file)
{
    size_t iret;
    iret = fread(coords_, 1, 12 * num_nodes_, file);
    iret = fread(conn_, 1, 12 * num_elements_, file);
    (void)iret;

    num_cl_ = 0;

    int i;

    for (i = 0; i < num_elements_; i++)
    {
        if (conn_[3 * i + 2] >= 0)
        {
            tl_[i] = TYPE_TRIANGLE;
            pl_[i] = num_cl_;
            cl_[num_cl_++] = conn_[3 * i];
            cl_[num_cl_++] = conn_[3 * i + 1];
            cl_[num_cl_++] = conn_[3 * i + 2];
        }
        else
        {
            tl_[i] = TYPE_BAR;
            pl_[i] = num_cl_;
            cl_[num_cl_++] = conn_[3 * i];
            cl_[num_cl_++] = conn_[3 * i + 1];
        }
    }

    for (i = 0; i < num_nodes_; i++)
    {
        x_[i] = coords_[3 * i];
        y_[i] = coords_[3 * i + 1];
        z_[i] = coords_[3 * i + 2];
    }
}

void
MeshData::get_middle_points(float **mx, float **my, float **mz)
{
    for (int i = 0; i < num_elements_; i++)
    {
        (*mx)[i] = x_[conn_[i * 3]];
        (*my)[i] = y_[conn_[i * 3]];
        (*mz)[i] = z_[conn_[i * 3]];
    }
}

void
MeshData::writeCovFile(int cov_id)
{

    covWriteUNSGRD(cov_id, num_elements_, num_cl_, num_nodes_,
                   pl_, cl_, tl_,
                   x_, y_, z_,
                   NULL, NULL, COUNT_ATTR);
}

Result::Result(string name, ResultType type, int block_pos, ResultIdentifier id, char char_id)
{
    name_ = name;
    type_ = type;
    block_pos_ = block_pos;
    id_ = id;
    char_id_ = char_id;
}

string
Result::get_name()
{
    return name_;
}

NodesMorphResult::NodesMorphResult(int num_nodes,
                                   string name,
                                   int block_pos,
                                   ResultIdentifier id,
                                   char char_id)
    : Result(name, Result::NODES_MORPH, block_pos, id, char_id)
{
    disp_[0] = new float[num_nodes];
    disp_[1] = new float[num_nodes];
    disp_[2] = new float[num_nodes];
    work_ = new float[3 * num_nodes];

    num_nodes_ = num_nodes;

    x0_ = NULL;
    y0_ = NULL;
    z0_ = NULL;
}

NodesMorphResult::~NodesMorphResult()
{
    delete[] disp_[0];
    delete[] disp_[1];
    delete[] disp_[2];
    delete[] work_;
}

void
NodesMorphResult::set_node_base(float *x, float *y, float *z)
{
    x0_ = x;
    y0_ = y;
    z0_ = z;
}

void
NodesMorphResult::readValues(FILE *file)
{
    fseek(file, block_pos_, SEEK_SET);

    size_t iret = fread(work_, 1, 12 * num_nodes_, file);
    if (iret == 0)
        return;

    if (x0_ && y0_ && z0_)
    {
        int i;
        for (i = 0; i < num_nodes_; i++)
        {
            disp_[0][i] = work_[3 * i] - x0_[i];
            disp_[1][i] = work_[3 * i + 1] - y0_[i];
            disp_[2][i] = work_[3 * i + 2] - z0_[i];
        }
    }
}

void
NodesMorphResult::writeCovFile(int cov_id)
{
    char *species[] = { (char *)"SPECIES" };
    char *spec_val[2];
    spec_val[0] = (char *)name_.c_str();
    covWriteUSTVDT(cov_id, num_nodes_, disp_[0], disp_[1], disp_[2],
                   species, spec_val, 1);
}

NodesResult::NodesResult(int num_nodes,
                         string name,
                         int block_pos,
                         ResultIdentifier id,
                         char char_id)
    : Result(name, Result::NODES, block_pos, id, char_id)
{
    values_ = new float[num_nodes];
    num_nodes_ = num_nodes;
}

NodesResult::~NodesResult()
{
    delete[] values_;
}

void
NodesResult::readValues(FILE *file)
{
    fseek(file, block_pos_, SEEK_SET);
    size_t iret = fread(values_, 1, 4 * num_nodes_, file);
    if (iret == 0)
        return;
}

void
NodesResult::writeCovFile(int cov_id)
{
    char *species[] = { (char *)"SPECIES" };
    char *spec_val[2];
    spec_val[0] = (char *)name_.c_str();
    covWriteUSTSDT(cov_id, num_nodes_, values_,
                   species, spec_val, 1);
}

ElemResult::ElemResult(int num_elem,
                       string name,
                       int block_pos,
                       ResultIdentifier id,
                       char char_id)
    : Result(name, Result::ELEMENTS, block_pos, id, char_id)
{
    values_ = new float[num_elem];
    num_elem_ = num_elem;
}

ElemResult::~ElemResult()
{
    delete[] values_;
}

void
ElemResult::readValues(FILE *file)
{
    fseek(file, block_pos_, SEEK_SET);
    size_t iret = fread(values_, 1, 4 * num_elem_, file);
    if (iret == 0)
        return;
}

void
ElemResult::writeCovFile(int cov_id)
{
    char *species[] = { (char *)"SPECIES" };
    char *spec_val[2];
    spec_val[0] = (char *)name_.c_str();
    covWriteUSTSDT(cov_id, num_elem_, values_,
                   species, spec_val, 1);
}

NodesAnimationResult::NodesAnimationResult(int num_nodes,
                                           string name,
                                           int block_pos,
                                           ResultIdentifier id,
                                           char char_id)
    : Result(name, Result::NODES_ANIMATION, block_pos, id, char_id)
{
    num_nodes_ = num_nodes;
    num_steps_ = 0;
}

NodesAnimationResult::~NodesAnimationResult()
{
    for (int i = 0; i < num_steps_; i++)
    {
        delete[] values_[i];
    }
}

void
NodesAnimationResult::readValues(FILE *file)
{
    fseek(file, block_pos_, SEEK_SET);

    size_t iret = fread(&num_steps_, 1, 4, file);

    string time;
    for (int i = 0; i < num_steps_; i++)
    {
        time.erase();
        readString(file, time);
        time_axis_.push_back(time);
    }

    for (int i = 0; i < num_steps_; i++)
    {
        float *tmp_values = new float[num_nodes_];
        iret = fread(tmp_values, 1, 4 * num_nodes_, file);
        values_.push_back(tmp_values);
    }

    (void)iret;
}

void
NodesAnimationResult::writeCovFile(int cov_id)
{
    char *timestep[] = { (char *)"TIMESTEP" };
    char *timestep_val[2];
    char last_step[64];

    timestep_val[0] = last_step;
    sprintf(last_step, "1 %d", num_steps_);

    covWriteSetBegin(cov_id, num_steps_);
    for (int i = 0; i < num_steps_; i++)
    {
        covWriteUSTSDT(cov_id, num_nodes_, values_[i],
                       NULL, NULL, COUNT_ATTR);
    }
    covWriteSetEnd(cov_id, timestep, (char **)timestep_val, 1);
}

VectorAnimationResult::VectorAnimationResult(int num_elements,
                                             string name,
                                             int block_pos,
                                             ResultIdentifier id,
                                             char char_id)
    : Result(name, Result::VECTOR_ANIMATION, block_pos, id, char_id)
{
    num_elements_ = num_elements;
    num_steps_ = 0;

    pointx_ = new float[num_elements_];
    pointy_ = new float[num_elements_];
    pointz_ = new float[num_elements_];
}

VectorAnimationResult::~VectorAnimationResult()
{
    for (int i = 0; i < num_steps_; i++)
    {
        delete[] values_[i][0];
        delete[] values_[i][1];
        delete[] values_[i][2];
    }
    delete[] pointx_;
    delete[] pointy_;
    delete[] pointz_;
}

void
VectorAnimationResult::readValues(FILE *file, MeshData &mesh)
{
    fseek(file, block_pos_, SEEK_SET);

    size_t iret = fread(&num_steps_, 1, 4, file);

    string time;
    int i;
    for (i = 0; i < num_steps_; i++)
    {
        time.erase();
        readString(file, time);
        time_axis_.push_back(time);
    }

    for (i = 0; i < num_steps_; i++)
    {
        float *tmp_values = new float[3 * num_elements_];
        iret = fread(tmp_values, 1, 12 * num_elements_, file);

        float *x = new float[num_elements_];
        float *y = new float[num_elements_];
        float *z = new float[num_elements_];
        for (int j = 0; j < num_elements_; j++)
        {
            x[j] = tmp_values[3 * j];
            y[j] = tmp_values[3 * j + 1];
            z[j] = tmp_values[3 * j + 2];
        }
        vector<float *> tmp_vec;
        tmp_vec.push_back(x);
        tmp_vec.push_back(y);
        tmp_vec.push_back(z);

        values_.push_back(tmp_vec);

        delete[] tmp_values;
    }

    // calculate middle point of elements
    mesh.get_middle_points(&pointx_, &pointy_, &pointz_);
    (void)iret;
}

void
VectorAnimationResult::writeCovFileValues(int cov_id)
{
    char *timestep[] = { (char *)"TIMESTEP" };
    char *timestep_val[2];
    char last_step[64];

    timestep_val[0] = last_step;
    sprintf(last_step, "1 %d", num_steps_);

    covWriteSetBegin(cov_id, num_steps_);
    for (int i = 0; i < num_steps_; i++)
    {
        covWriteUSTVDT(cov_id, num_elements_, values_[i][0], values_[i][1], values_[i][2],
                       NULL, NULL, COUNT_ATTR);
    }
    covWriteSetEnd(cov_id, timestep, (char **)timestep_val, 1);
}

void
VectorAnimationResult::writeCovFilePoints(int cov_id)
{
    char *timestep[] = { (char *)"TIMESTEP" };
    char *timestep_val[2];
    char last_step[64];

    timestep_val[0] = last_step;
    sprintf(last_step, "1 %d", num_steps_);

    covWriteSetBegin(cov_id, num_steps_);
    for (int i = 0; i < num_steps_; i++)
    {
        covWritePOINTS(cov_id, num_elements_, pointx_, pointy_, pointz_,
                       NULL, NULL, COUNT_ATTR);
    }
    covWriteSetEnd(cov_id, timestep, (char **)timestep_val, 1);
}
