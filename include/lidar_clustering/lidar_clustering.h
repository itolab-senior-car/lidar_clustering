#ifndef LIDAR_CLUSTERING_H_
#define LIDAR_CLUSTERING_H_

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>

#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/don.h>

#include <pcl/kdtree/kdtree.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/common/common.h>
#include <pcl/common/pca.h>

#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>

#include <pcl/segmentation/extract_clusters.h>

#include "itolab_senior_car_msgs/CloudCluster.h"

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <limits>
#include <cmath>
#include <chrono>

namespace lidar_clustering
{
  using PointI = pcl::PointXYZ;
  class LidarClustering
  {
  public:
    explicit LidarClustering();
    virtual ~LidarClustering();
    void setCloud(const pcl::PointCloud<PointI>::Ptr in_origin_cloud_pt,
                  const std::vector<int>& in_cluster_indices,
                  std_msgs::Header in_ros_header,
                  int in_id,
                  std::string in_label);
    void toRosMessage(std_msgs::Header in_ros_header,
                      itolab_senior_car_msgs::CloudCluster& out_cluster_message);

    pcl::PointCloud<PointI>::Ptr getCloud();
    pcl::PointXYZ getMinPoint();
    pcl::PointXYZ getMaxPoint();
    pcl::PointXYZ getAveragePoint();
    pcl::PointXYZ getCentroid();

    jsk_recognition_msgs::BoundingBox getBoundingBox();

    double getOrientationAngle() const;
    double getLength() const;
    double getHeight() const;
    double getWidth() const;

    int getID() const;
    std::string getLabel() const;

    Eigen::Matrix3f getEigenVectors() const;
    Eigen::Vector3f getEigenValues() const;

    bool isValid() const;
    void setValidity(bool in_valid);

    pcl::PointCloud<pcl::PointXYZ>::Ptr joinCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr);

    std::vector<double> getFpfhDescriptor(const unsigned int& in_ompnum_threads,
                                          const double& in_normal_search_radius,
                                          const double& in_fpfh_search_radius);
 
  private:
    pcl::PointCloud<PointI>::Ptr m_pointcloud;
    pcl::PointXYZ m_min_point;
    pcl::PointXYZ m_max_point;
    pcl::PointXYZ m_average_point;
    pcl::PointXYZ m_centroid;

    double m_orientation_angle = 0.0;
    double m_length = 0.0;
    double m_width = 0.0;
    double m_height = 0.0;

    jsk_recognition_msgs::BoundingBox m_bounding_box;

    std::string m_label;

    int m_id = 0;

    Eigen::Matrix3f m_eigen_vectors;
    Eigen::Vector3f m_eigen_values;

    bool m_valid_cluster = false;
    
  };
  using ClusterPtr = boost::shared_ptr<LidarClustering>;
}  // namespace lidar_clustering

#endif // LIDAR_CLUSTERING_H_
