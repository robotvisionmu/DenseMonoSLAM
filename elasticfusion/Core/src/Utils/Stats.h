#ifndef STATS_H_
#define STATS_H_

#include <vector>
#include <map>
#include <string>
#include <fstream>


class Stats
{
    public:
        Stats(){}
        virtual ~Stats(){}

        void record(float nid_rgb, float nid_depth, float nid_overall, int surfel_count, int num_frames_fused, bool fused = false)
        {
            m_nid_rgb.push_back(nid_rgb);
            m_nid_depth.push_back(nid_depth);
            m_nid_overall.push_back(nid_overall);

            m_surfel_count.push_back(surfel_count);
            m_frames_fused.push_back(num_frames_fused);

            m_fused.push_back(fused);
        }

        void clear()
        {
            m_nid_rgb.clear();
            m_nid_depth.clear();
            m_nid_overall.clear();

            m_surfel_count.clear();
            m_frames_fused.clear();
        }

        void write(std::string filename)
        {
            std::ofstream ofs(filename);

            if(!ofs.good())return;

            ofs << "frames fused: " << m_frames_fused.back() << "\n";
            ofs << "num surfels:" << m_surfel_count.back() << "\n";
            
            std::stringstream ss;
            ss << "fused: ";
            for(int i = 0; i < (int)m_fused.size()-1; i++)
                ss << m_fused[i] << ",";
            ss << m_fused.back();
            ss<<"\n";

            ofs << ss.str();

            std::stringstream surfel_ss;
            surfel_ss << "surfels per frame: ";
            for(int i = 0; i < (int)m_surfel_count.size()-1; i++)
                surfel_ss << m_surfel_count[i] << ",";
            surfel_ss << m_surfel_count.back();
            ofs << surfel_ss.str();
            ofs.close();
        }
    
    private:
        std::vector<float> m_nid_rgb;
        std::vector<float> m_nid_depth;
        std::vector<float> m_nid_normal;
        std::vector<float> m_nid_radius;
        std::vector<float> m_nid_overall;

        std::map<std::string, std::vector<float>> m_timings; 

        std::vector<int> m_surfel_count;
        std::vector<int> m_frames_fused;

        std::vector<bool> m_fused;
};

#endif /*STATS_H_*/