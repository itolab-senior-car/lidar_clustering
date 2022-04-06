#ifndef LIDAR_CLUSTERING_NODE_H
#define LIDAR_CLUSTERING_NODE_H

#include "lidar_clustering/lidar_clustering.h"
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <pcl/filters/voxel_grid.h>

#include "itolab_senior_car_msgs/Centroids.h"
#include "itolab_senior_car_msgs/CloudClusterArray.h"
#include "itolab_senior_car_msgs/DetectedObjectArray.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <cmath>

namespace lidar_clustering_node
{
  class LidarClusteringNode
  {
  public:
    LidarClusteringNode();
    void publishDetectedObjects(const itolab_senior_car_msgs::CloudClusterArray &clusters);
    void publishBoundingBoxes(const itolab_senior_car_msgs::CloudClusterArray& clusters);
    void publishCloudClusters(const ros::Publisher* publisher,
                              const itolab_senior_car_msgs::CloudClusterArray& clusters,
                              const std::string& target_frame, const std_msgs::Header& header);
    void publishCentroids(const ros::Publisher *publisher,
                          const itolab_senior_car_msgs::Centroids &centroids,
                          const std::string &target_frame,
                          const std_msgs::Header &header);
    void publishCloud(const ros::Publisher *publisher,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_publish_ptr);
    void keepLanePoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_ptr,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr out_ptr,
                        float lane_threshold = 1.5);
    void clusterObjecks(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_ptr,
                        std::vector<lidar_clustering::ClusterPtr>& clusters,
                        itolab_senior_car_msgs::Centroids& centroids,
                        double in_max_cluster_distance = 0.5);
    void checkClusterMerge(const size_t& in_cluster_id, std::vector<lidar_clustering::ClusterPtr> &in_cluster,
                           std::vector<char> &visited_clusters,
                           std::vector<size_t> &out_merged_indices,
                           double in_merge_threshold);
    void mergeCluster(const std::vector<lidar_clustering::ClusterPtr> &in_cluster,
                      std::vector<lidar_clustering::ClusterPtr> &out_cluster,
                      std::vector<size_t> in_merge_indices,
                      const size_t &current_index,
                      std::vector<char> &in_out_merged_cluster);
    void checkAllForMerge(std::vector<lidar_clustering::ClusterPtr> &in_clusters,
                          std::vector<lidar_clustering::ClusterPtr> &out_clusters,
                          float in_merge_threshold);
    void segmentByDistance(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_ptr,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr out_ptr,
                           itolab_senior_car_msgs::Centroids &centroids,
                           itolab_senior_car_msgs::CloudClusterArray &clusters);
    void removeFloor(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_ptr,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr out_no_floor_ptr,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr out_only_floor_ptr,
                     float in_max_height = 0.2,
                     float in_floor_max_angle = 0.1);
    void downsampleCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_ptr,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr out_ptr,
                         float in_leaf_size = 0.2);
    void clipCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_ptr,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr out_ptr,
                   float min_width = -0.5,
                   float max_width = 0.5,
                   float min_distance = 0.2,
                   float max_distance = 3.0,
                   float min_height = -0.9,
                   float max_height = 0.5);
    void differenceNormalsSegmentation(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_ptr,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr out_ptr);
    void removePointsUpTo(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_ptr,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr out_ptr,
                        const double in_distance);
    void velodyne_callback(const sensor_msgs::PointCloud2ConstPtr& in_msg);
    void transformBoundingBox(const jsk_recognition_msgs::BoundingBox& in_boundingbox,
                              jsk_recognition_msgs::BoundingBox& out_boundingbox,
                              const std::string& in_target_frame,
                              const std_msgs::Header& in_header);
  private :
    ros::NodeHandle nh;
    ros::Publisher cluster_cloud_pub, ground_cloud_pub;
    ros::Publisher centroid_pub, cluster_message_pub, detected_object_pub;
    ros::Publisher bounding_boxes_pub, bounding_centroid_value_pub;
    
    ros::Subscriber point_cloud_sub;
    
    tf::StampedTransform* transform;
    tf::StampedTransform* velodyne_output_transform;
    tf::TransformListener* transform_listener;

    std::string output_frame = "velodyne";
    std_msgs::Header velodyne_header;

    visualization_msgs::Marker visualization_marker;

    std::list<std::vector<geometry_msgs::Point>> way_area_points;
    pcl::PointCloud<pcl::PointXYZ> sensor_cloud;

    bool downsample_cloud = true;
    bool remove_ground = true;
    bool get_ground_cloud = true;
    bool use_diffnormals = false;
    bool use_multiple_threshold = false;


    double initial_quat_w = 1.0;
    double leaf_size = 0.1;
    int cluster_min_size = 20;
    int cluster_max_size = 1000;
    
    double clip_min_width = -0.5;
    double clip_max_witdh = 0.5;
    double clip_min_distance = 0.2;
    double clip_max_distance = 3.0;
    double clip_min_height = -0.85;
    double clip_max_height = 0.5;

    double remove_points_upto = 0.0;
    double cluster_merge_threshold = 1.5;
    double cluster_distance = 0.75;

    std::string lidar_topic = "/points_raw";
    std::vector<double> clustering_distances = {0.5, 1.1, 1.6, 2.1, 2.6};
    std::vector<double> clustering_ranges = {15, 30, 45, 60};
  };

}
#endif // LIDAR_CLUSTERING_NODE_H
