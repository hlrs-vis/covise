#include <openGA.hpp>
#include <fstream>

class MyGAproblem
{
public:
    std::ofstream output_file;

    struct MySolution
    {

        // FIXME: Use template to generate std::array with another size
        std::array<int,4> cam;
        std::string to_string() const;

    };

    struct MyMiddleCost
    {
        // This is where the results of simulation
        // is stored but not yet finalized.
        double objective;
};


    typedef EA::Genetic<MySolution,MyMiddleCost> GA_Type;
    typedef EA::GenerationType<MySolution,MyMiddleCost> Generation_Type;

public:
    MyGAproblem();
    ~MyGAproblem();

private:
    int myrandom();
    void init_genes(MySolution& p,const std::function<double(void)> &rnd01);
    bool eval_solution(const MySolution& p,MyMiddleCost &c);

    MySolution mutate(const MySolution& X_base, const std::function<double(void)> &rnd01,
                      double shrink_scale);

    MySolution crossover(const MySolution& X1, const MySolution& X2,const std::function<double(void)> &rnd01);

    double calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X);

    void SO_report_generation(int generation_number, const EA::GenerationType<MySolution,MyMiddleCost> &last_generation,
                              const MySolution& best_genes);



};
