/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>

struct header
{
    char error;
    int version;
    int result_list_pos;
    int num_nodes;
    int num_elements;
    char solid_or_neutral; // 1 solid, 0 neutral
} Header;

class MeshData
{
public:
    MeshData(int num_nodes, int num_elements);
    ~MeshData();

    void readData(FILE *file);
    void writeCovFile(int cov_id);
    void get_nodes(float **x0, float **y0, float **z0);

    void get_middle_points(float **mx, float **my, float **mz);

private:
    float *coords_;
    int *conn_;

    int num_nodes_;
    int num_elements_;

    int num_cl_;
    int *tl_, *pl_, *cl_;
    float *x_, *y_, *z_;
};

class Result
{

public:
    typedef enum
    {
        NODES = 1,
        NODES_ANIMATION = 2,
        VECTOR_ANIMATION = 3,
        ELEMENTS = 4,
        NODES_MORPH = 7
    } ResultType;
    typedef enum
    {
        NONE = 0,
        FILLING = 1,
        COOLING = 2,
        SAW = 3
    } ResultIdentifier;

    string get_name();

    Result(string name, ResultType type, int block_pos, ResultIdentifier id, char char_id);
    virtual ~Result(){};

    virtual void readValues(FILE *file)
    {
        (void)file;
    };
    virtual void writeCovFile(int cov_id)
    {
        (void)cov_id;
    };

protected:
    string name_;
    ResultType type_;
    int block_pos_; //pos of block in file
    ResultIdentifier id_;
    char char_id_;
};

class NodesResult : public Result
{

public:
    NodesResult(int num_nodes, string name, int block_pos, ResultIdentifier id, char char_id);
    ~NodesResult();

    void readValues(FILE *file);
    void writeCovFile(int cov_id);

private:
    float *values_;
    int num_nodes_;
};

class ElemResult : public Result
{

public:
    ElemResult(int num_elem, string name, int block_pos, ResultIdentifier id, char char_id);
    ~ElemResult();

    void readValues(FILE *file);
    void writeCovFile(int cov_id);

private:
    float *values_;
    int num_elem_;
};

class NodesMorphResult : public Result
{

public:
    NodesMorphResult(int num_nodes, string name, int block_pos, ResultIdentifier id, char char_id);
    ~NodesMorphResult();

    void readValues(FILE *file);
    void writeCovFile(int cov_id);

    void set_node_base(float *x, float *y, float *z);

private:
    int num_nodes_;
    float *disp_[3];
    float *work_, *x0_, *y0_, *z0_;
};

class NodesAnimationResult : public Result
{

public:
    NodesAnimationResult(int num_nodes, string name, int block_pos, ResultIdentifier id, char char_id);
    ~NodesAnimationResult();

    void readValues(FILE *file);
    void writeCovFile(int cov_id);

private:
    vector<float *> values_;
    vector<string> time_axis_;
    int num_nodes_;
    int num_steps_;
};

class VectorAnimationResult : public Result
{

public:
    VectorAnimationResult(int num_elements, string name, int block_pos, ResultIdentifier id, char char_id);
    ~VectorAnimationResult();

    void readValues(FILE *file)
    {
        (void)file;
    };
    void readValues(FILE *file, MeshData &mesh);
    void writeCovFileValues(int cov_id);
    void writeCovFilePoints(int cov_id);

private:
    vector<vector<float *> > values_;
    vector<string> time_axis_;
    int num_elements_;
    int num_steps_;

    float *pointx_, *pointy_, *pointz_;
};
class ElementsResult : public Result
{

public:
    ElementsResult(int num_elements, string name, int block_pos, ResultIdentifier id, char char_id);
    void readValues(FILE *file);
    float *values;
};

struct resultlist
{
    string remark;
    int num_results;
    vector<Result> results;
} ResultList;
