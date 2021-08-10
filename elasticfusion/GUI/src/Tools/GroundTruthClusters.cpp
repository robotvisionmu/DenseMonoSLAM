#include "GroundTruthClusters.h"

GroundTruthClusters::GroundTruthClusters(const std::string& filename) {
  std::ifstream ifs;
  std::string line;
  ifs.open(filename);

  while (!ifs.eof()) {
    uint64_t utime;
    double dtime;
    int cluster;
    std::getline(ifs, line);

    int n = sscanf(line.c_str(), "%lf,%i", &dtime, &cluster);

    utime = uint64_t(dtime);

    clusters[utime] = cluster;
  }
  std::map<int, bool> cluster_map;
  for(const auto & kv : clusters)
  {
    if(cluster_map.find(kv.second) == cluster_map.end())
      cluster_map[kv.second] = true;
  }
  std::vector<int> cluster_list;
  for(const auto & kv : cluster_map)
  {
    cluster_list.push_back(kv.first);
  }
  
  float num_clusters = (float)cluster_list.size();

  for(const auto & c : cluster_list)
  {
    float h = (360.0 * float(c)) / num_clusters;
    float s = 1.0f;
    float v = 1.0f;
    auto f = [&](float n) -> float
    {
      float k = std::fmod((n + (h/60.0)), 6.0);
      return v - (v * s * std::max(std::min(k, std::min(4.0f - k, 1.0f)), 0.0f));
    };
    float r = f(5.0f);
    float g = f(3.0f);
    float b = f(1.0f);
    cluster_colors[c] = std::make_tuple(r, g, b);
  }
}

GroundTruthClusters::~GroundTruthClusters() {}

int GroundTruthClusters::getCluster(uint64_t timestamp) {
  int cluster = 0;

  const auto& c = clusters.find(timestamp);

  if (c != clusters.end()) {
    cluster = clusters[timestamp];
    last_utime = timestamp;
  }
  return cluster;
}

std::tuple<float,float,float> GroundTruthClusters::getClusterColor(int cluster)
{
  return cluster_colors[cluster];
}
