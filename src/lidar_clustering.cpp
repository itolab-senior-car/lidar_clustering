#include "lidar_clustering/lidar_clustering.h"

namespace lidar_clustering
{
  LidarClustering::LidarClustering()
  {
    m_valid_cluster = true;
  }

  jsk_recognition_msgs::BoundingBox LidarClustering::getBoundingBox()
  {
    return m_bounding_box;
  }

  pcl::PointCloud<PointI>::Ptr LidarClustering::getCloud()
  {
    return m_pointcloud;
  }

  pcl::PointXYZ LidarClustering::getMinPoint()
  {
    return m_min_point;
  }

  pcl::PointXYZ LidarClustering::getMaxPoint()
  {
    return m_max_point;
  }

  pcl::PointXYZ LidarClustering::getCentroid()
  {
    return m_centroid;
  }

  pcl::PointXYZ LidarClustering::getAveragePoint()
  {
    return m_average_point;
  }

  double LidarClustering::getOrientationAngle() const
  {
    return m_orientation_angle;
  }

  Eigen::Vector3f LidarClustering::getEigenValues() const
  {
    return m_eigen_values;
  }

  Eigen::Matrix3f LidarClustering::getEigenVectors() const
  {
    return m_eigen_vectors;
  }

  bool LidarClustering::isValid() const
  {
    return m_valid_cluster;
  }

  void LidarClustering::setValidity(bool in_valid)
  {
    m_valid_cluster = in_valid;
  }

  int LidarClustering::getID() const
  {
    return m_id;
  }

  void LidarClustering::toRosMessage(std_msgs::Header in_ros_header, itolab_senior_car_msgs::CloudCluster &out_cluster_message)
  {
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*(this->getCloud()), cloud_msg);
    cloud_msg.header = in_ros_header;

    out_cluster_message.header = in_ros_header;
    out_cluster_message.cloud = cloud_msg;

    out_cluster_message.min_point.header = in_ros_header;
    out_cluster_message.min_point.point.x = this->getMinPoint().x;
    out_cluster_message.min_point.point.y = this->getMinPoint().y;
    out_cluster_message.min_point.point.z = this->getMinPoint().z;

    out_cluster_message.max_point.header = in_ros_header;
    out_cluster_message.max_point.point.x = this->getMaxPoint().x;
    out_cluster_message.max_point.point.y = this->getMaxPoint().y;
    out_cluster_message.max_point.point.z = this->getMaxPoint().z;

    out_cluster_message.avg_point.header = in_ros_header;
    out_cluster_message.avg_point.point.x = this->getAveragePoint().x;
    out_cluster_message.avg_point.point.y = this->getAveragePoint().y;
    out_cluster_message.avg_point.point.z = this->getAveragePoint().z;

    out_cluster_message.centroid_point.header = in_ros_header;
    out_cluster_message.centroid_point.point.x = this->getCentroid().x;
    out_cluster_message.centroid_point.point.y = this->getCentroid().y;
    out_cluster_message.centroid_point.point.z = this->getCentroid().z;

    out_cluster_message.estimated_angle = this->getOrientationAngle();
    out_cluster_message.dimensions = this->getBoundingBox().dimensions;
    out_cluster_message.bounding_box = this->getBoundingBox();

    out_cluster_message.eigen_values.x = this->getEigenValues().x();
    out_cluster_message.eigen_values.y = this->getEigenValues().y();
    out_cluster_message.eigen_values.z = this->getEigenValues().z();

    Eigen::Matrix3f eigen_vectors = this->getEigenVectors();
    for(uint8_t i = 0; i < 3; ++i)
      {
        geometry_msgs::Vector3 eigen_vector;
        eigen_vector.x = static_cast<float>(eigen_vectors(i, 0));
        eigen_vector.y = static_cast<float>(eigen_vectors(i, 1));
        eigen_vector.z = static_cast<float>(eigen_vectors(i, 2));
        out_cluster_message.eigen_vectors.push_back(eigen_vector);
      }
  }

  void LidarClustering::setCloud(const pcl::PointCloud<PointI>::Ptr in_origin_cloud_ptr, const std::vector<int> &in_cluster_indices, std_msgs::Header in_ros_header, int in_id, std::string in_label)
  {
    m_label = in_label;
    m_id = in_id;

    pcl::PointCloud<PointI>::Ptr current_cluster(new pcl::PointCloud<PointI>);
    float min_x =  std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float min_y =  std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();
    float min_z =  std::numeric_limits<float>::max();
    float max_z = -std::numeric_limits<float>::max();
    float average_x = 0.0;
    float average_y = 0.0;
    float average_z = 0.0;

    for(auto iter = in_cluster_indices.begin(); iter != in_cluster_indices.end(); ++iter)
      {
        PointI p;
        p.x = in_origin_cloud_ptr->points[*iter].x;
        p.y = in_origin_cloud_ptr->points[*iter].y;
        p.z = in_origin_cloud_ptr->points[*iter].z;

        average_x += p.x;
        average_y += p.y;
        average_z += p.z;
        m_centroid.x += p.x;
        m_centroid.y += p.y;
        m_centroid.z += p.z;
        current_cluster->points.push_back(p);

        min_x = min_x * (p.x >= min_x) + p.x * (p.x < min_x);
        min_y = min_y * (p.y >= min_y) + p.y * (p.y < min_y);
        min_z = min_z * (p.z >= min_z) + p.z * (p.z < min_z);

        max_x = max_x * (p.x <= max_x) + p.x * (p.x > max_x);
        max_y = max_y * (p.y <= max_y) + p.y * (p.y > max_y);
        max_z = max_z * (p.z <= max_z) + p.z * (p.z > max_z);
        // std::cout << "x (min, max): " << min_x << " " << max_x << "\n";
        // std::cout << "y (min, max): " << min_y << " " << max_y << "\n";
        // std::cout << "z (min, max): " << min_z << " " << max_z << "\n";
      }

    m_min_point.x = min_x;
    m_min_point.y = min_y;
    m_min_point.z = min_z;

    m_max_point.x = max_x;
    m_max_point.y = max_y;
    m_max_point.z = max_z;

    if (in_cluster_indices.size() > 0)
      {
        m_centroid.x /= in_cluster_indices.size();
        m_centroid.y /= in_cluster_indices.size();
        m_centroid.z /= in_cluster_indices.size();

        average_x /= in_cluster_indices.size();
        average_y /= in_cluster_indices.size();
        average_z /= in_cluster_indices.size();
      }

    m_average_point.x = average_x;
    m_average_point.y = average_y;
    m_average_point.z = average_z;

    m_length = m_max_point.x - m_min_point.x;
    m_width  = m_max_point.y - m_min_point.y;
    m_height = m_max_point.z - m_min_point.z;

    m_bounding_box.header = in_ros_header;
    m_bounding_box.pose.position.x = m_min_point.x + m_length / 2;
    m_bounding_box.pose.position.y = m_min_point.y +  m_width / 2;
    m_bounding_box.pose.position.z = m_min_point.z + m_height / 2;
    // std::cout << "BB position(x, y, z) : " << m_bounding_box.pose.position.x << " "
    //           << m_bounding_box.pose.position.y << " "
    //           << m_bounding_box.pose.position.z << "\n";

    m_bounding_box.dimensions.x = ((m_length < 0) ? -1 * m_length : m_length);
    m_bounding_box.dimensions.y = (( m_width < 0) ? -1 *  m_width : m_width );
    m_bounding_box.dimensions.z = ((m_height < 0) ? -1 * m_height : m_height);
    // std::cout << "BB dimension(x, y, z) : " << m_bounding_box.dimensions.x << " "
    //           << m_bounding_box.dimensions.y << " "
    //           << m_bounding_box.dimensions.z << "\n";

    double rz = 0;

    tf::Quaternion quat = tf::createQuaternionFromRPY(0.0, 0.0, rz);
    tf::quaternionTFToMsg(quat, m_bounding_box.pose.orientation);

    current_cluster->width = current_cluster->points.size();
    current_cluster->height = 1;
    current_cluster->is_dense = true;

    if(current_cluster->points.size() > 3)
      {
        pcl::PCA<pcl::PointXYZ> current_cluster_pca;
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_cluster_mono(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::copyPointCloud<PointI, pcl::PointXYZ>(*current_cluster, *current_cluster_mono);

        current_cluster_pca.setInputCloud(current_cluster_mono);
        m_eigen_vectors = current_cluster_pca.getEigenVectors();
        m_eigen_values = current_cluster_pca.getEigenValues();
      }

    m_valid_cluster = true;
    m_pointcloud = current_cluster;
  }

  LidarClustering::~LidarClustering()
  {
  }
}
