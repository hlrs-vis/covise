#include "GA.h"
#include "DataManager.h"
#include "Sensor.h"
#include "Profiling.h"
#include "SensorPlacement.h"
#include <numeric>

bool GA:: maxCoverage1(const Solution& sensorNetwork, MiddleCost &c)
{
    SP_PROFILE_FUNCTION();
    
    // std::vector<int> necessarySensors{1,1,1,1,1,2,2,2,2,2};

    // std::vector<float> visMat1 {0.0f, 0.4f, 0.8f, 1.0f, 0.2f, 0.1f, 0.4f, 0.1f, 0.3f, 1.0f};
    // std::vector<float> visMat2 {0.0f, 0.0f, 0.8f, 1.0f, 0.2f, 0.1f, 0.4f, 0.1f, 0.3f, 1.0f};
    // std::vector<float> visMat3 {0.0f, 0.4f, 0.8f, 1.0f, 0.2f, 0.1f, 0.4f, 0.1f, 0.3f, 0.0f};
    // std::vector<std::vector<float>> visMat;
    // visMat.push_back(visMat1);
    // visMat.push_back(visMat2);
    // visMat.push_back(visMat3);

    std::vector<int> sensorsPerPoint(m_NumberOfObservationPoints,0);
    std::vector<float> sumVisMat(m_NumberOfObservationPoints,0.0f);

    for(const auto& sensor : sensorNetwork.sensors)
    {
        std::transform(sensor->getVisibilityMatrix().begin(), sensor->getVisibilityMatrix().end(), sensorsPerPoint.begin(), sensorsPerPoint.begin(),[](float i, int j ) {return (i == 0.0f ? j : j+1);});  // count nbrOf sensors 
                                                                                              
        std::transform(sensor->getVisibilityMatrix().begin(), sensor->getVisibilityMatrix().end(), sumVisMat.begin(), sumVisMat.begin(), std::plus<float>());                                              // add coefficients 
    }
    
    // check if visible:
    std::vector<float> coveredPoints;
    coveredPoints.reserve(m_NumberOfObservationPoints);

    auto ItsensorsPerPoint = sensorsPerPoint.begin();
    auto ItRequiredSensors = m_RequiredSensorsPerPoint.begin();

    while( ItsensorsPerPoint != sensorsPerPoint.end())
    {
        if(*ItsensorsPerPoint >= *ItRequiredSensors)
            coveredPoints.push_back(1);
        else 
            coveredPoints.push_back(0);
        
        // increment iterators
        if( ItsensorsPerPoint != sensorsPerPoint.end())
        {
            ++ItsensorsPerPoint;
            ++ItRequiredSensors;
        }
    }

    int sumCoveredPoints = std::accumulate(coveredPoints.begin(), coveredPoints.end(),0);
    
    c.objective = -(sumCoveredPoints);


    
    // std::cout<<"Cameras: "<<std::endl;
    // for(const auto& x : sensorsPerPoint)
    // {
    // std::cout<< x<<", ";
    // }
    // std::cout<<""<<std::endl;
    // std::cout<<"VisMat"<<std::endl;
    // for(const auto& x :sumVisMat)
    // {
    //     std::cout<< x <<", ";
    // }
    // std::cout<<""<<std::endl;

   
    return true;
}

bool GA::maxCoverage2(const Solution& p, MiddleCost &c)
{
    SP_PROFILE_FUNCTION();

    std::cout<<"Fitness: max Coverage 2" <<std::endl;
    return false; //FIXME
}

bool rejectSameSensorPosition(SensorPosition* newPos, SensorPosition* oldPos)
{
    return true;
}

GA::GA(FitnessFunctionType fitness)
{
    m_NumberOfSensors = calcNumberOfSensors();
    calcZonePriorities();

    SP_PROFILE_BEGIN_SESSION("SensorPlacement-Optimization","SensorPlacement-Optimization.json");

	EA::Chronometer timer;
	timer.tic();	
	using std::bind;
	using std::placeholders::_1;
	using std::placeholders::_2;
	using std::placeholders::_3;

    if(fitness == FitnessFunctionType::MaxCoverage1) // implement this here directly more down there->optimization Strategy not necessary
        m_FitnessFunction = std::bind(&GA::maxCoverage1,this,_1,_2);
    else if(fitness == FitnessFunctionType::MaxCoverage2)
        m_FitnessFunction = std::bind(&GA::maxCoverage2,this,_1,_2);

	ga_obj.problem_mode = EA::GA_MODE::SOGA;
    ga_obj.multi_threading = true;
    ga_obj.idle_delay_us = 1; // switch between threads quickly
    ga_obj.dynamic_threading = m_DynamicThreading;
    ga_obj.verbose = false;
    ga_obj.population = m_PopulationSize;
    ga_obj.generation_max = m_MaxGeneration;
    ga_obj.calculate_SO_total_fitness = std::bind( &GA::calculate_SO_total_fitness, this, _1);
    ga_obj.init_genes = std::bind( &GA::init_genes, this, _1,_2);
    ga_obj.eval_solution = std::bind( &GA::optimizationStrategy, this, _1,_2 );
    ga_obj.mutate = std::bind( &GA::mutate, this, _1,_2,_3 );
    ga_obj.crossover = std::bind( &GA::crossover, this, _1,_2,_3 );
    ga_obj.SO_report_generation = std::bind( &GA::SO_report_generation, this, _1,_2,_3 );
    ga_obj.crossover_fraction = m_CrossoverRate;
    ga_obj.mutation_rate = m_MutationRate;
    ga_obj.best_stall_max = 10;
    ga_obj.elite_count = ga_obj.population / 100 * 6; //6% of population size;
    ga_obj.solve();
	
	std::cout<<"The problem is optimized in "<<timer.toc()<<" seconds."<<std::endl;
    SP_PROFILE_END_SESSION();
}

bool GA::optimizationStrategy(const Solution& p, MiddleCost &c)
{
	return m_FitnessFunction(p,c);
}

void GA::init_genes(Solution& sensorNetwork,const std::function<double(void)> &rnd01)
{
    SP_PROFILE_FUNCTION();

    for(int i{0}; i < m_NumberOfSensors; ++i)
        sensorNetwork.sensors.push_back(m_SensorPool.getRandomOrientation(i, rnd01));
    
}

Solution GA::mutate(const Solution& X_base,const std::function<double(void)> &rnd01, double shrink_scale)
{
    SP_PROFILE_FUNCTION();

    Solution X_new = X_base;
    int count = std::ceil(shrink_scale * m_NumberOfSensors);

    while(count != 0)
    {
        int randomPos = std::roundl(rnd01()*(m_NumberOfSensors-1));
        X_new.sensors.at(randomPos) = m_SensorPool.getRandomOrientation(randomPos, rnd01);
        count--;
    }

    return X_new;
}

Solution GA::crossover(const Solution& X1, const Solution& X2,const std::function<double(void)> &rnd01)
{
    SP_PROFILE_FUNCTION();

    int cutPoint = roundl(rnd01() * m_NumberOfSensors);
    Solution X_new;
    X_new.sensors.clear(); // brauch man das ? 
    X_new.sensors.insert(X_new.sensors.begin(), X1.sensors.begin(), X1.sensors.begin() + cutPoint);
    X_new.sensors.insert(X_new.sensors.end(), X2.sensors.begin() + cutPoint,X2.sensors.end());

    return X1;
}

double GA::calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X)
{
    SP_PROFILE_FUNCTION();

    // finalize the cost
    // obtain the final cost from the middle cost
    double final_cost{0};
    final_cost += X.middle_costs.objective;
    
    return final_cost;
}

void GA::SO_report_generation(int generation_number,const EA::GenerationType<Solution,MiddleCost> &last_generation,const Solution& best_genes)
{
    std::cout
        <<"Generation ["<<generation_number<<"], "
        <<"Best="<<last_generation.best_total_cost<<", "
        <<"Average="<<last_generation.average_cost<<", "
       // <<"Best genes(cameras)=("<<best_genes.to_string()<<")"<<", "
        // <<"Best Coverage[%]="<<last_generation.chromosomes.at(last_generation.best_chromosome_index).middle_costs.coverage.to_string()<<", "
        <<"Exe_time="<<last_generation.exe_time
    <<std::endl;
}



void GA:: calcZonePriorities()
{
    for(const auto& zone : DataManager::GetSafetyZones())
    {
        int priority = 1; //FIX ME get real priority
        m_RequiredSensorsPerPoint.insert(m_RequiredSensorsPerPoint.end(),zone.get()->getNumberOfPoints(), priority);
    }
    m_NumberOfObservationPoints = m_RequiredSensorsPerPoint.size();
}


SensorPool::SensorPool()
{
    for(const auto& sensor : DataManager::GetSensors())
        m_Sensors.push_back(sensor.get());

    for(const auto& zone : DataManager::GetSensorZones())
        m_SensorZones.push_back(zone.get());
}

Orientation* SensorPool::getRandomOrientation(int pos, const std::function<double(void)> &rnd01) const
{
    Orientation* result{nullptr};

    // better Solution create Class Random Generator and use it in openGA and in Sensors Class to return random Sensor
    if( pos <= m_Sensors.size() )
    {
        int numberOfOrientations = m_Sensors.at(pos)->getNbrOfOrientations();
        int random = std::roundl(rnd01() * (numberOfOrientations - 1));

        result = m_Sensors.at(pos)->getSpecificOrientation(random);
    }
    else if(pos > m_Sensors.size() && pos <= m_SensorZones.size())
    {
        int numberOfSensors = m_SensorZones.at(pos)->getNumberOfSensors();
        int randomSensor = std::roundl(rnd01() * (numberOfSensors - 1));
        int numberOfOrientations = m_SensorZones.at(pos)->getSpecificSensor(randomSensor)->getNbrOfOrientations();
        int randomOrientation = std::roundl(rnd01() * (numberOfOrientations - 1));

        result = m_SensorZones.at(pos)->getSpecificSensor(randomSensor)->getSpecificOrientation(randomOrientation);
    }
    else
        std::cout << "No sensor at this position available" << std::endl;
    
    return result;
}

std::vector<Orientation> GA::getFinalOrientations() const
{
    std::vector<Orientation*> p_finalOrientations = ga_obj.last_generation.chromosomes.at(ga_obj.last_generation.best_chromosome_index).genes.sensors;
    std::vector<Orientation> finalOrientations;
    std::transform(std::begin(p_finalOrientations), std::end(p_finalOrientations),
                   std::back_inserter(finalOrientations),[](Orientation* orient){return *orient;});
    
    return finalOrientations; 
}
