#include "lidar_clustering/lidar_clustering_node.h"

namespace lidar_clustering_node
{
using PclPointXYZ = pcl::PointCloud<pcl::PointXYZ>;
LidarClusteringNode::LidarClusteringNode() : nh()
{
  // bool remove_ground = true;
  // bool use_diffnormals = false;
  // bool use_multiple_threshold = false;

  // double initial_quat_w = 1.0;
  nh.param("lidar_clustering/downsample_cloud", downsample_cloud, false);
  nh.param("lidar_clustering/leaf_size", leaf_size, 0.1);

  nh.param("lidar_clustering/get_ground_cloud", get_ground_cloud, true);

  nh.param("lidar_clustering/cluster_min_size", cluster_min_size, 20);
  nh.param("lidar_clustering/cluster_max_size", cluster_max_size, 1000);

  nh.param("lidar_clustering/clip_min_width", clip_min_width, -0.5);
  nh.param("lidar_clustering/clip_max_width", clip_max_witdh, 0.5);
  nh.param("lidar_clustering/clip_min_distance", clip_min_distance, 0.2);
  nh.param("lidar_clustering/clip_max_distance", clip_max_distance, 3.0);
  nh.param("lidar_clustering/clip_min_height", clip_min_height, -0.85);
  nh.param("lidar_clustering/clip_max_height", clip_max_height, 0.5);

  point_cloud_sub = nh.subscribe(lidar_topic, 1, &LidarClusteringNode::velodyne_callback, this);

  cluster_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_points_cluster", 1);
  ground_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar_points_ground", 1);
  centroid_pub = nh.advertise<itolab_senior_car_msgs::Centroids>("/lidar_cluster_centroids", 1);
  detected_object_pub = nh.advertise<itolab_senior_car_msgs::DetectedObjectArray>("/lidar_detected_object", 1);

  bounding_boxes_pub = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/lidar_bounding_boxes", 1);
  bounding_centroid_value_pub = nh.advertise<visualization_msgs::MarkerArray>("/lidar_centroid_val", 1);
  }

  void LidarClusteringNode::velodyne_callback(const sensor_msgs::PointCloud2ConstPtr& in_msg)
  {
    PclPointXYZ::Ptr current_sensor_cloud_ptr(new PclPointXYZ);

    pcl::fromROSMsg(*in_msg, *current_sensor_cloud_ptr);
    velodyne_header = in_msg->header;

    PclPointXYZ::Ptr downsampled_cloud_ptr(new PclPointXYZ);
    if(downsample_cloud)
      {
        downsampleCloud(current_sensor_cloud_ptr,
                        downsampled_cloud_ptr,
                        leaf_size);
      }
    else
      {
        downsampled_cloud_ptr = current_sensor_cloud_ptr;
      }

    PclPointXYZ::Ptr clipped_cloud_ptr(new PclPointXYZ);
    clipCloud(downsampled_cloud_ptr, clipped_cloud_ptr,
              clip_min_width, clip_max_witdh,
              clip_min_distance, clip_max_distance,
              clip_min_height, clip_max_height);

    PclPointXYZ::Ptr no_floor_ptr(new PclPointXYZ);
    PclPointXYZ::Ptr only_floor_cloud_ptr(new PclPointXYZ);

    if(remove_ground)
      {
        removeFloor(clipped_cloud_ptr,
                    no_floor_ptr,
                    only_floor_cloud_ptr);
      }
    else
      {
        no_floor_ptr = clipped_cloud_ptr;
      }

    if(remove_ground && get_ground_cloud)
      {
        publishCloud(&ground_cloud_pub, only_floor_cloud_ptr);
      }

    PclPointXYZ::Ptr diffnormal_cloud_ptr(new PclPointXYZ);
    if(use_diffnormals)
      {
        differenceNormalsSegmentation(no_floor_ptr, diffnormal_cloud_ptr);
      }
    else
      {
        diffnormal_cloud_ptr = no_floor_ptr;
      }

    itolab_senior_car_msgs::Centroids centroids;
    PclPointXYZ::Ptr clustered_cloud_ptr(new PclPointXYZ);
    itolab_senior_car_msgs::CloudClusterArray cloud_cluster;

    segmentByDistance(diffnormal_cloud_ptr, clustered_cloud_ptr, centroids, cloud_cluster);
    
    publishCloud(&cluster_cloud_pub, clustered_cloud_ptr);
    std::cout << "cloud published\n";

    centroids.header = velodyne_header;
    publishCentroids(&centroid_pub, centroids, output_frame, velodyne_header);
    std::cout << "centroids published\n";
    cloud_cluster.header = velodyne_header;
    publishCloudClusters(&cluster_message_pub, cloud_cluster, output_frame, velodyne_header);
    publishBoundingBoxes(cloud_cluster);
    std::cout << "cloud cluster published\n";
  }

  void LidarClusteringNode::downsampleCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_ptr,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr out_ptr, float in_leaf_size)
  {
    pcl::VoxelGrid<pcl::PointXYZ> vxgd;
    vxgd.setInputCloud(in_ptr);
    vxgd.setLeafSize(in_leaf_size, in_leaf_size, in_leaf_size);
    vxgd.filter(*out_ptr);
  }

  void LidarClusteringNode::clipCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_ptr,
                                      pcl::PointCloud<pcl::PointXYZ>::Ptr out_ptr, float min_width,
                                      float max_width, float min_distance, float max_distance,
                                      float min_height, float max_height)
  {
    out_ptr->clear();
    for(const auto& p:in_ptr->points)
      {
        if(p.x >= min_distance &&
           p.x <= max_distance &&
           p.y >= min_width &&
           p.y <= max_width &&
           p.z >= min_height &&
           p.z <= max_height)
          {
            out_ptr->points.push_back(p);
          }
      }
  }

  void LidarClusteringNode::removeFloor(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_ptr,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr out_no_floor_ptr,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr out_only_floor_ptr, float in_max_height,
                   float in_floor_max_angle)
  {
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(50);
    seg.setAxis(Eigen::Vector3f(0, 0, 1));
    seg.setEpsAngle(in_floor_max_angle);

    seg.setDistanceThreshold(in_max_height);
    seg.setOptimizeCoefficients(true);
    seg.setInputCloud(in_ptr);
    seg.segment(*inliers, *coefficients);
    if(inliers->indices.size() == 0)
      {
        std::cout << "Inliers is empty for given data. Could not estimate a planar model." << "\n";
      }

    // Filter out the floor
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(in_ptr);
    extract.setIndices(inliers);
    extract.setNegative(true); // true removes the indices, false leaves only the indices
    extract.filter(*out_no_floor_ptr);

    // Get only floor cloud
    extract.setNegative(false); // false leaves only the indices
    extract.filter(*out_only_floor_ptr);
  }

  void LidarClusteringNode::differenceNormalsSegmentation(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_ptr,
                                                          pcl::PointCloud<pcl::PointXYZ>::Ptr out_ptr)
  {
    float small_scale = 0.5;
    float large_scale = 2.0;
    float angle_threshold = 0.5;

    pcl::search::Search<pcl::PointXYZ>::Ptr tree;
    if(in_ptr->isOrganized())
      {
        tree.reset(new pcl::search::OrganizedNeighbor<pcl::PointXYZ>());
      }
    else
      {
        tree.reset(new pcl::search::KdTree<pcl::PointXYZ>(false));
      }

    tree->setInputCloud(in_ptr);

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::PointNormal> norm;
    norm.setInputCloud(in_ptr);
    norm.setSearchMethod(tree);

    norm.setViewPoint(std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max());

    pcl::PointCloud<pcl::PointNormal>::Ptr norm_small_scale(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr norm_large_scale(new pcl::PointCloud<pcl::PointNormal>);

    norm.setRadiusSearch(small_scale);
    norm.compute(*norm_small_scale);

    norm.setRadiusSearch(large_scale);
    norm.compute(*norm_large_scale);


    pcl::PointCloud<pcl::PointNormal>::Ptr diffnormal(new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud<pcl::PointXYZ, pcl::PointNormal>(*in_ptr, *diffnormal);

    pcl::DifferenceOfNormalsEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::PointNormal> diffnormal_estimator;
    diffnormal_estimator.setInputCloud(in_ptr);
    diffnormal_estimator.setNormalScaleLarge(norm_large_scale);
    diffnormal_estimator.setNormalScaleSmall(norm_small_scale);

    diffnormal_estimator.initCompute();
    diffnormal_estimator.computeFeature(*diffnormal);

    pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond(new pcl::ConditionOr<pcl::PointNormal>());
    range_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(
                                                                               new pcl::FieldComparison<pcl::PointNormal>("curvature", pcl::ComparisonOps::GT, angle_threshold)));
    pcl::ConditionalRemoval<pcl::PointNormal> cond_removal;
    cond_removal.setCondition(range_cond);
    cond_removal.setInputCloud(diffnormal);

    pcl::PointCloud<pcl::PointNormal>::Ptr diffnormal_filtered(new pcl::PointCloud<pcl::PointNormal>);

    cond_removal.filter(*diffnormal_filtered);
    pcl::copyPointCloud<pcl::PointNormal, pcl::PointXYZ>(*diffnormal_filtered, *out_ptr);
  }

  void LidarClusteringNode::clusterObjecks(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_ptr,
                                           std::vector<lidar_clustering::ClusterPtr>& clusters,
                                           itolab_senior_car_msgs::Centroids& centroids, double in_max_cluster_distance)
  {
    // std::cout << "Clustering\n";
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    PclPointXYZ::Ptr cloud_2d(new PclPointXYZ);
    pcl::copyPointCloud(*in_ptr, *cloud_2d);
    for(auto& pz:cloud_2d->points)
      {
        pz.z = 0.0;
      }

    if(cloud_2d->points.size() > 0)
      {
        tree->setInputCloud(cloud_2d);
      }

    // perform clustering on 2d cloud
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> euc;
    euc.setClusterTolerance(in_max_cluster_distance);
    euc.setMinClusterSize(cluster_min_size);
    euc.setMaxClusterSize(cluster_max_size);
    euc.setSearchMethod(tree);
    euc.setInputCloud(cloud_2d);
    euc.extract(cluster_indices);
    // use indices on 3d cloud

    unsigned int k = 0;
    clusters.reserve(cluster_indices.size());
    for(auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
      {
        lidar_clustering::ClusterPtr cluster(new lidar_clustering::LidarClustering);
        cluster->setCloud(in_ptr, it->indices, velodyne_header, k, "");
        clusters.emplace_back(cluster);

        ++k;
      }
  }
  void LidarClusteringNode::transformBoundingBox(const jsk_recognition_msgs::BoundingBox& in_boundingbox,
                            jsk_recognition_msgs::BoundingBox& out_boundingbox, const std::string& in_target_frame,
                            const std_msgs::Header& in_header)
  {
    geometry_msgs::PoseStamped pose_in, pose_out;
    pose_in.header = in_header;
    pose_in.pose = in_boundingbox.pose;
    try
    {
      transform_listener->transformPose(in_target_frame, ros::Time(), pose_in, in_header.frame_id, pose_out);
    }
    catch (tf::TransformException& ex)
    {
      ROS_ERROR("transformBoundingBox: %s", ex.what());
    }
    out_boundingbox.pose = pose_out.pose;
    out_boundingbox.header = in_header;
    out_boundingbox.header.frame_id = in_target_frame;
    out_boundingbox.dimensions = in_boundingbox.dimensions;
    out_boundingbox.value = in_boundingbox.value;
    out_boundingbox.label = in_boundingbox.label;
  }

  void LidarClusteringNode::checkAllForMerge(std::vector<lidar_clustering::ClusterPtr>& in_clusters,
                                             std::vector<lidar_clustering::ClusterPtr>& out_clusters, float in_merge_threshold)
  {
    // std::cout << "Checking for points merge\n";
    std::vector<char> visited_cluster(in_clusters.size(), false);
    std::vector<char> merged_cluster(in_clusters.size(), false);

    size_t current_index = 0;

    for(size_t idx = 0; idx < in_clusters.size(); ++idx)
      {
        if(!visited_cluster[idx])
          {
            visited_cluster[idx] = true;
            std::vector<size_t> merge_indices;
            checkClusterMerge(idx, in_clusters, visited_cluster, merge_indices, in_merge_threshold);
            mergeCluster(in_clusters, out_clusters, merge_indices, current_index++, merged_cluster);
          }
      }
    for(size_t idx = 0; idx < in_clusters.size(); ++idx)
      {
        if(!merged_cluster[idx])
          {
            out_clusters.push_back(in_clusters[idx]);
          }
      }
      // std::cout << "Finished\n";
  }

  void LidarClusteringNode::checkClusterMerge(const size_t& in_cluster_id, std::vector<lidar_clustering::ClusterPtr>& in_clusters,
                                              std::vector<char>& visited_clusters,
                                              std::vector<size_t>& out_merged_indices, double in_merge_threshold)
  {
    pcl::PointXYZ point_a = in_clusters[in_cluster_id]->getCentroid();
    for(size_t idx = 0; idx < in_clusters.size(); ++idx)
      {
        if(idx != in_cluster_id && !visited_clusters[idx])
          {
            pcl::PointXYZ point_b = in_clusters[idx]->getCentroid();
            double distance = sqrt(pow(point_b.x - point_a.x, 2) + pow(point_b.y - point_a.y, 2));
            if(distance<= in_merge_threshold)
              {
                visited_clusters[idx] = true;
                out_merged_indices.push_back(idx);
                checkClusterMerge(idx, in_clusters, visited_clusters, out_merged_indices, in_merge_threshold);
              }
          }
      }
  }
  void LidarClusteringNode::mergeCluster(const std::vector<lidar_clustering::ClusterPtr>& in_cluster,
                                         std::vector<lidar_clustering::ClusterPtr>& out_cluster, std::vector<size_t> in_merge_indices,
                                         const size_t& current_index, std::vector<char>& in_out_merged_cluster)
  {
    PclPointXYZ cloud;
    lidar_clustering::ClusterPtr merged_cluster(new lidar_clustering::LidarClustering());
    for(size_t idx = 0; idx < in_merge_indices.size(); ++idx)
      {
        cloud += *(in_cluster[in_merge_indices[idx]]->getCloud());
        in_out_merged_cluster[in_merge_indices[idx]] = true;
      }
    std::vector<int> indices(cloud.points.size(), 0);
    std::iota(indices.begin(), indices.end(), 0);

    if(cloud.points.size() > 0)
      {
        merged_cluster->setCloud(cloud.makeShared(), indices, velodyne_header, current_index, "");
        out_cluster.push_back(merged_cluster);
      }
  }

  void LidarClusteringNode::segmentByDistance(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_ptr,
                                              pcl::PointCloud<pcl::PointXYZ>::Ptr out_ptr,
                                              itolab_senior_car_msgs::Centroids& centroids,
                                              itolab_senior_car_msgs::CloudClusterArray& clusters)
  {
    // std::cout << "Segmenting\n";
    std::vector<lidar_clustering::ClusterPtr> cluster_vector;
    PclPointXYZ::Ptr cloud_ptr(new PclPointXYZ);
    if(!use_multiple_threshold)
      {
        cloud_ptr->reserve(in_ptr->size());
        // for (unsigned int idx = 0; idx < in_ptr->points.size(); ++idx)
        //   {
        //     pcl::PointXYZ current_point;
        //     current_point = in_ptr->points[idx];

        //     cloud_ptr->points.push_back(current_point);
        //   }
        for(const auto& p:in_ptr->points)
          {
            cloud_ptr->points.push_back(p);
          }
        clusterObjecks(cloud_ptr, cluster_vector, centroids, cluster_distance);
        std::cout << "Size of cluster_vector is " << cluster_vector.size() << "\n";
      }
    else
      {
        std::vector<PclPointXYZ::Ptr> cloud_segments_array(5);
        for(unsigned int idx = 0; idx < cloud_segments_array.size(); ++idx)
          {
            PclPointXYZ::Ptr tmp_cloud(new PclPointXYZ);
            cloud_segments_array[idx] = tmp_cloud;
          }

        for(size_t idx = 0; idx < in_ptr->points.size(); ++idx)
          {
            pcl::PointXYZ current_point;
            current_point = in_ptr->points[idx];
            float origin_distance = sqrt(pow(current_point.x, 2) + pow(current_point.z, 2));
            if(origin_distance < clustering_ranges[0])
              {
                cloud_segments_array[0]->points.push_back(current_point);
              }
            else if (origin_distance < clustering_ranges[1])
              {
                cloud_segments_array[1]->points.push_back(current_point);
              }
            else if (origin_distance < clustering_ranges[2])
              {
                cloud_segments_array[2]->points.push_back(current_point);
              }
            else if (origin_distance < clustering_ranges[3])
              {
                cloud_segments_array[3]->points.push_back(current_point);
              }
            else
              {
                cloud_segments_array[4]->points.push_back(current_point);
              }
          }
        for(size_t idx = 0; idx < cloud_segments_array.size(); ++idx)
          {
            std::vector<lidar_clustering::ClusterPtr> local_cluster_vector;
            clusterObjecks(cloud_segments_array[idx], local_cluster_vector,
                           centroids, clustering_distances[idx]);
            cluster_vector.insert(cluster_vector.end(),
                                  local_cluster_vector.cbegin(),
                                  local_cluster_vector.cend());
          }
      }
    std::vector<lidar_clustering::ClusterPtr> mid_cluster_vector;
    std::vector<lidar_clustering::ClusterPtr> final_cluster_vector;
    if(cluster_vector.size() > 0)
      checkAllForMerge(cluster_vector, mid_cluster_vector, cluster_merge_threshold);
    else
      mid_cluster_vector = cluster_vector;

    if(mid_cluster_vector.size() > 0)
      checkAllForMerge(mid_cluster_vector, final_cluster_vector, cluster_merge_threshold);
    else
      final_cluster_vector = mid_cluster_vector;
    std::cout << "Final cluster vector size: " << final_cluster_vector.size() << "\n";

    for(size_t idx = 0; idx < final_cluster_vector.size(); ++idx)
      {
        *out_ptr = *out_ptr + *(final_cluster_vector[idx]->getCloud());
        jsk_recognition_msgs::BoundingBox bounding_box = final_cluster_vector[idx]->getBoundingBox();

        pcl::PointXYZ center_point = final_cluster_vector[idx]->getCentroid();
        geometry_msgs::Point centroid;
        centroid.x = center_point.x;
        centroid.y = center_point.y;
        centroid.z = center_point.z;
        bounding_box.header = velodyne_header;
            
        if(final_cluster_vector[idx]->isValid())
          {
            centroids.points.push_back(centroid);
            itolab_senior_car_msgs::CloudCluster cloud_cluster;
            final_cluster_vector[idx]->toRosMessage(velodyne_header, cloud_cluster);
            clusters.clusters.push_back(cloud_cluster);
          }
      }
    // std::cout << "Done segmenting\n";
  }

  void LidarClusteringNode::publishCloudClusters(
                   const ros::Publisher* publisher,
                   const itolab_senior_car_msgs::CloudClusterArray& clusters,
                   const std::string& target_frame, const std_msgs::Header& header)
  {
    if(target_frame != header.frame_id)
      {
    //     itolab_senior_car_msgs::CloudClusterArray clusters_transformed;
    //     clusters_transformed.header = header;
    //     clusters_transformed.header.frame_id = target_frame;
    //     for(const auto& c:clusters.clusters)
    //       {
    //         itolab_senior_car_msgs::CloudCluster cluster_transform;
    //         cluster_transform.header = header;
    //         try
    //           {
    //             transform_listener->lookupTransform(target_frame, velodyne_header.frame_id,
    //                                                 ros::Time(), *transform);
    //             pcl_ros::transformPointCloud(target_frame, *transform, c.cloud,
    //                                          cluster_transform.cloud);
    //             transform_listener->transformPoint(target_frame, ros::Time(), c.min_point,
    //                                                header.frame_id, cluster_transform.min_point);
    //             transform_listener->transformPoint(target_frame, ros::Time(),
    //                                                c.max_point, header.frame_id,
    //                                                cluster_transform.max_point);
    //             transform_listener->transformPoint(target_frame, ros::Time(),
    //                                                c.avg_point, header.frame_id,
    //                                                cluster_transform.avg_point);
    //             transform_listener->transformPoint(target_frame, ros::Time(),
    //                                                c.min_point, header.frame_id,
    //                                                cluster_transform.centroid_point);
    //             cluster_transform.dimensions = c.dimensions;
    //             cluster_transform.eigen_values = c.eigen_values;
    //             cluster_transform.eigen_vectors = c.eigen_vectors;

    //             cluster_transform.bounding_box.pose.position = c.bounding_box.pose.position;
    //             cluster_transform.bounding_box.pose.orientation.w = initial_quat_w;
    //             clusters_transformed.clusters.push_back(cluster_transform);
    //           }
    //         catch(tf::TransformException &ex)
    //           {
    //             ROS_ERROR("publishCloudClusters: %s", ex.what());
    //           }
    //       }
    //     publisher->publish(clusters_transformed);
    //     publishDetectedObjects(clusters_transformed);
      }
    else
      {
        publisher->publish(clusters);
        publishDetectedObjects(clusters);
      }
  }

  void LidarClusteringNode::publishCentroids(const ros::Publisher* publisher,
                                             const itolab_senior_car_msgs::Centroids& centroids,
                                             const std::string& target_frame, const std_msgs::Header& header)
  {
    if(target_frame != header.frame_id)
      {
        itolab_senior_car_msgs::Centroids centroid_transformed;
        centroid_transformed.header = header;
        centroid_transformed.header.frame_id = target_frame;
        for(const auto& p:centroid_transformed.points)
          {
            geometry_msgs::PointStamped centroid_in, centroid_out;
            centroid_in.header = header;
            centroid_in.point = p;
            try
              {
                transform_listener->transformPoint(target_frame, ros::Time(),
                                                   centroid_in, header.frame_id,
                                                   centroid_out);
              }
            catch(tf::TransformException &ex)
              {
                ROS_ERROR("publishCentroids: %s", ex.what());
              }
          }
        publisher->publish(centroid_transformed);
      }
    else
      {
        publisher->publish(centroids);
      }
  }

  void LidarClusteringNode::publishCloud(const ros::Publisher* publisher,
                                         const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_publish_ptr)
  {
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud_to_publish_ptr, cloud_msg);
    cloud_msg.header = velodyne_header;
    publisher->publish(cloud_msg);
  }

  void LidarClusteringNode::publishDetectedObjects(const itolab_senior_car_msgs::CloudClusterArray &clusters)
  {
    itolab_senior_car_msgs::DetectedObjectArray detected_object_array;
    detected_object_array.header = clusters.header;

    for(size_t idx = 0; idx < clusters.clusters.size(); ++idx)
      {
        itolab_senior_car_msgs::DetectedObject detected_object;
        detected_object.header = clusters.header;
        detected_object.label = "unknown";
        detected_object.score = 1.;
        detected_object.space_frame = clusters.header.frame_id;
        detected_object.pose = clusters.clusters[idx].bounding_box.pose;
        detected_object.dimensions = clusters.clusters[idx].dimensions;
        detected_object.pointcloud = clusters.clusters[idx].cloud;
        detected_object.convex_hull = clusters.clusters[idx].convex_hull;
        detected_object.valid = true;

        detected_object_array.objects.push_back(detected_object);
      }
    detected_object_pub.publish(detected_object_array);
  }

  void LidarClusteringNode::publishBoundingBoxes(const itolab_senior_car_msgs::CloudClusterArray& clusters)
  {
    jsk_recognition_msgs::BoundingBoxArray boxes;
    visualization_msgs::MarkerArray markers;

    for(size_t idx = 0; idx < clusters.clusters.size(); ++idx)
      {
        visualization_msgs::Marker mark;
        jsk_recognition_msgs::BoundingBox box;
        box = clusters.clusters[idx].bounding_box;
        box.label = idx;
        boxes.boxes.push_back(box);

        mark.header.frame_id = clusters.header.frame_id;
        mark.header.stamp = clusters.header.stamp;
        mark.ns = "bb";
        mark.id = idx;
        mark.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        mark.action = visualization_msgs::Marker::ADD;
        mark.lifetime = ros::Duration(0.05);
        mark.pose.position = box.pose.position;
        mark.pose.orientation = box.pose.orientation;
        double distance = std::hypot(clusters.clusters[idx].centroid_point.point.x,
                                     clusters.clusters[idx].centroid_point.point.y);
        mark.text = std::to_string(distance);
        mark.scale.x = 1.0;
        mark.scale.y = 1.0;
        mark.scale.z = 1.0;

        mark.color.r = 0.0f;
        mark.color.g = 1.0f;
        mark.color.b = 0.0f;
        mark.color.a = 1.0;
        markers.markers.push_back(mark);
      }
    boxes.header.stamp = ros::Time();
    boxes.header.frame_id = output_frame;
    bounding_boxes_pub.publish(boxes);
    bounding_centroid_value_pub.publish(markers);
  }
}  // namespace lidar_clustering_node
