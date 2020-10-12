#pragma once
#include <openGA.h>
#include <Sensor.h>
#include <Zone.h>

#include <fstream>

struct Solution{

    std::vector<Orientation*> sensors;
    // std::vector<SensorZone*> sensorZones;
};

struct MiddleCost{

    struct Coverage //Coverage in % 
    {
        float total;
        float prio1;
        float prio2;
        std::string to_string() const
        {
            std::string output;
            output = "total: "+std::to_string(total)+" Prio1: "+std::to_string(prio1)+" Prio2 " +std::to_string(prio2)+" ";
            return std::string("{") + output + "}";
        }
    };

    double objective;
    Coverage coverage;
};

struct SensorPool
{
public:
    SensorPool();
    Orientation* getRandomOrientation(int pos, const std::function<double(void)> &rnd01) const; 
    int getNbrOfSingleSensors()const{return m_Sensors.size();}
    std::vector<int> getNbrOfSensorsPerZone()const;
private:
    std::vector<SensorPosition*> m_Sensors;
    std::vector<SensorZone*> m_SensorZones;
};

enum class FitnessFunctionType
{
    MaxCoverage1,
    MaxCoverage2
};

struct PropertiesMaxCoverage1
{
    float weightingFactorPRIO1{2.0f};
    float Penalty{1000.0f};             //Penalty for too few cameras
    //float PenaltyFactorPRIO2{1.0};    //not necessary ?
    float thresholdVisibility{0.5};     //the sum(all Vismats)/ RequiredSensorsForThisPoint > thresholdVisibility
};

struct PropertiesMaxCoverage2
{
    float m_PenaltyConstant;
    float m_RequiredCoverage{0.8};  
    float m_ThresholdVisibility{0.5};
};

class GA
{
public:
    using FitnessFunction = std::function<bool(const Solution& p, MiddleCost &c)>;
    GA(FitnessFunctionType fitness);

    static PropertiesMaxCoverage1 s_PropsMaxCoverage1;
    static PropertiesMaxCoverage2 s_PropsMaxCoverage2;

    float m_CrossoverRate{0.7};
    float m_MutationRate{0.3};
    int m_PopulationSize{3000};
    int m_MaxGeneration{1000};
    bool m_DynamicThreading{false};

    std::vector<Orientation> getFinalOrientations() const;


private:
    FitnessFunction m_FitnessFunction; 
    std::ofstream m_Results;

    SensorPool m_SensorPool{};
    int m_NumberOfSensors;
    int m_NumberOfObservationPoints;
    int m_NumberOfPrio1Points;
    std::vector<int> m_RequiredSensorsPerPoint; 
    
    typedef EA::Genetic<Solution,MiddleCost> GA_Type;
    typedef EA::GenerationType<Solution,MiddleCost> Generation_Type;
    GA_Type ga_obj;

    // GA functions
    void init_genes(Solution& p,const std::function<double(void)> &rnd01);
    Solution mutate(const Solution& X_base,const std::function<double(void)> &rnd01,double shrink_scale);
    Solution crossover(const Solution& X1, const Solution& X2,const std::function<double(void)> &rnd01);
    double calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X);
    void SO_report_generation(int generation_number,const EA::GenerationType<Solution,MiddleCost> &last_generation,const Solution& best_genes);
    bool optimizationStrategy(const Solution& p, MiddleCost &c);

    // available Fitness Functions
    bool maxCoverage1(const Solution& p, MiddleCost &c);
    bool maxCoverage2(const Solution& p, MiddleCost &c);

    void calcCoverageProcentage(MiddleCost &c, int sumCoveredPrio1Points, int sumCoveredPrio2Points)const;
    int coverEachPointWithMin1Sensor(std::vector<int>& nbrOfCameras, std::vector<float> &sumVisMat); 
    int sumOfCoveredPrio1Points(const std::vector<int>& sensorsPerPoint, const std::vector<int>& requiredSensorsPerPoint, const std::vector<float>& sumVisMat) const;
    
    
    const Orientation* getRandomSensor(int sensorPosition ,const std::function<double(void)> &rnd01)const;
    void calcZonePriorities();
};

