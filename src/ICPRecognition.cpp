/*

Author: Kathleen Lee and Seungwook Han
 Edits by David Watkins

*/

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <cluster_segmentation/SegmentedClustersArray.h>
#include <cluster_segmentation/DetectedObject.h>
#include <cluster_segmentation/DetectedObjectsArray.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/ply_io.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/common/projection_matrix.h>
#include <tf_conversions/tf_eigen.h>
#include <string>
#include <sensor_msgs/point_cloud2_iterator.h>

#include "boost/filesystem.hpp"
#include "boost/regex.hpp"
#include <iostream>
#include <shape_msgs/Mesh.h>

#include <boost/algorithm/string/replace.hpp>


class LoadedMesh {
public:
    std::string filepath;
    std::string name;

    LoadedMesh() : name(""), filepath("") {

    }

    LoadedMesh(std::string filepath, std::string name) : name(name), filepath(filepath), meshCloud(new pcl::PointCloud<pcl::PointXYZ>) {
        pcl::io::loadPolygonFilePLY(filepath, *objectMesh);
        pcl::fromPCLPointCloud2(objectMesh->cloud, *meshCloud);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr getMeshCloud() const{
        return meshCloud->makeShared();
    }
    pcl::PolygonMesh::Ptr getObjectMesh() const {
        return objectMesh;
    }

private:
    pcl::PolygonMesh::Ptr objectMesh;
    pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud;
};


std::vector<LoadedMesh> findAllMeshesInDirectory(std::string directoryName)
{
    std::vector<LoadedMesh> meshPaths;
    boost::filesystem::path current_dir(directoryName);

    boost::regex pattern("*.ply"); // list all files starting with a
    for (boost::filesystem::recursive_directory_iterator iter(current_dir), end; iter != end; ++iter)
    {
        std::string name = iter->path().filename().string();
        if (regex_match(name, pattern)) {
            meshPaths.push_back(LoadedMesh(iter->path().string(), name));
        }
    }

    return meshPaths;
}


geometry_msgs::Pose matrix4fToPose(Eigen::Matrix4f &trans) {

    Eigen::Matrix4d trans4d = trans.cast<double>();
    Eigen::Affine3d transAffine;
    transAffine.matrix() = trans4d;
    tf::Transform tf;
    tf::transformEigenToTF(transAffine, tf);

    geometry_msgs::Pose pose;
    pose.position.x = tf.getOrigin().x();
    pose.position.y = tf.getOrigin().y();
    pose.position.z = tf.getOrigin().z();
    pose.orientation.x = tf.getRotation().x();
    pose.orientation.y = tf.getRotation().y();
    pose.orientation.z = tf.getRotation().z();
    pose.orientation.w = tf.getRotation().w();

    return pose;
}


std::pair<double, geometry_msgs::Pose> convergeICP(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn,
        const std::string &frame_id,
        const LoadedMesh &mesh
)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud = mesh.getMeshCloud();
    meshCloud->header.frame_id = frame_id;

    // Run ICP algorithm and print score
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloudIn);
    icp.setInputTarget(meshCloud);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    ROS_INFO_STREAM("Cluster has converged: " << icp.hasConverged() << " with score '" << icp.getFitnessScore() << "'");

    // Get final transformed matrix4f
    Eigen::Matrix4f trans = icp.getFinalTransformation();
    geometry_msgs::Pose pose = matrix4fToPose(trans);
    ROS_INFO_STREAM("Final transform: " << pose);

    return std::make_pair(icp.getFitnessScore(), pose);
};


void convertPolygonMeshToRosMesh(
        const pcl::PolygonMesh::Ptr polygon_mesh_ptr,
        shape_msgs::Mesh::Ptr ros_mesh_ptr
)
{
    ROS_INFO("Conversion from PCL PolygonMesh to ROS Mesh started.");

    pcl_msgs::PolygonMesh pcl_msg_mesh;

    pcl_conversions::fromPCL(*polygon_mesh_ptr, pcl_msg_mesh);

    sensor_msgs::PointCloud2Modifier pcd_modifier(pcl_msg_mesh.cloud);

    size_t size = pcd_modifier.size();

    ros_mesh_ptr->vertices.resize(size);

    ROS_INFO_STREAM("polys: " << pcl_msg_mesh.polygons.size() << " vertices: " << pcd_modifier.size());

    sensor_msgs::PointCloud2ConstIterator<float> pt_iter(pcl_msg_mesh.cloud, "x");

    for (size_t i = 0u; i < size; i++, ++pt_iter) {
        ros_mesh_ptr->vertices[i].x = pt_iter[0];
        ros_mesh_ptr->vertices[i].y = pt_iter[1];
        ros_mesh_ptr->vertices[i].z = pt_iter[2];
    }

    ROS_INFO_STREAM("Updated vertices");

    ros_mesh_ptr->triangles.resize(polygon_mesh_ptr->polygons.size());

    for (size_t i = 0u; i < polygon_mesh_ptr->polygons.size(); ++i) {
        if (polygon_mesh_ptr->polygons[i].vertices.size() < 3u) {
            ROS_WARN_STREAM("Not enough points in polygon. Ignoring it.");
            continue;
        }

        for (size_t j = 0u; j < 3u; ++j) {
            ros_mesh_ptr->triangles[i].vertex_indices[j] =
                    polygon_mesh_ptr->polygons[i].vertices[j];
        }
    }
    ROS_INFO("Conversion from PCL PolygonMesh to ROS Mesh ended.");
}




class ICPRecognition {

public:

    explicit ICPRecognition(ros::NodeHandle nh) : _nodeHandle(nh) {

        if (!nh.getParam("mesh_resource_folder_path", _meshResourceFolderPath)) {
            ROS_ERROR("Could not find 'mesh_resource_folder_path' param in param server.");
            exit(-1);
        }

        _loadedMeshes = findAllMeshesInDirectory(_meshResourceFolderPath);

        // define the subscriber and publisher
        _clusterSubscriber = _nodeHandle.subscribe("/cluster_segmentation/pcl_clusters", 1, &ICPRecognition::cluster_cb, this);
        // Visualize marker in rviz
        _detectedObjectPublisher = _nodeHandle.advertise<cluster_segmentation::DetectedObjectsArray>("/cluster_segmentation/detected_objects",1);
        _recognizedObjectMarkerPublisher = _nodeHandle.advertise<visualization_msgs::MarkerArray>("/cluster_segmentation/detected_markers",1);
    }

private:

    ros::NodeHandle _nodeHandle;
    ros::Subscriber _clusterSubscriber;
    ros::Publisher _detectedObjectPublisher;
    ros::Publisher _recognizedObjectMarkerPublisher;
    std::string _meshResourceFolderPath;
    std::vector<LoadedMesh> _loadedMeshes;

    void cluster_cb(const cluster_segmentation::SegmentedClustersArray& clusterMsg);

}; // end class definition

// define callback function
void ICPRecognition::cluster_cb (const cluster_segmentation::SegmentedClustersArray& clusterMsg)
{
    cluster_segmentation::DetectedObjectsArray detectedObjectsArray;
    visualization_msgs::MarkerArray detectedObjectsMarkers;
    int clusterCount = 0;

    for(const sensor_msgs::PointCloud2 &cluster : clusterMsg.clusters) {
        // Container for original & filtered data
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(cluster, *cloudIn);

        // Print attributes of input cloud
        ROS_INFO_STREAM("Saved " << cloudIn->points.size() << " data points to input");

        double minScore = 1000000;
        geometry_msgs::Pose minPose;
        LoadedMesh minMesh;
        bool hasConverged;

        for(const LoadedMesh& loadedMesh : _loadedMeshes) {
            std::pair<double, geometry_msgs::Pose> icp = convergeICP(cloudIn, clusterMsg.header.frame_id, loadedMesh);
            double score = icp.first;

            if(score < minScore) {
                minPose = icp.second;
                minScore = score;
                minMesh = loadedMesh;
                hasConverged = true;
            }
        }

        if(hasConverged) {
            // Convert minPose into PoseStamped
            geometry_msgs::PoseStamped poseStamped;
            poseStamped.pose = minPose;
            poseStamped.header.frame_id = clusterMsg.header.frame_id;

            // Convert meshCloud into a PointCloud2 ROS msg
            sensor_msgs::PointCloud2 minMeshROSCloud;
            pcl::toROSMsg(*minMesh.getMeshCloud(), minMeshROSCloud);

            // Convert polygon mesh into ROS mesh
            shape_msgs::MeshPtr detectedObjectMesh;
            convertPolygonMeshToRosMesh(minMesh.getObjectMesh(), detectedObjectMesh);

            visualization_msgs::Marker marker;
            marker.header.frame_id = clusterMsg.header.frame_id;
            marker.header.stamp = ros::Time();
            marker.id = clusterCount;
            marker.pose = minPose;
            marker.mesh_resource = minMesh.filepath;
            marker.type = visualization_msgs::Marker::MESH_RESOURCE;
            marker.action = visualization_msgs::Marker::ADD;
            detectedObjectsMarkers.markers.push_back(marker);

            cluster_segmentation::DetectedObject detectedObject;
            detectedObject.label = minMesh.name;
            detectedObject.cloud = minMeshROSCloud;
            detectedObject.pose = poseStamped;
            detectedObject.mesh = *detectedObjectMesh;
            detectedObjectsArray.objects.push_back(detectedObject);

            clusterCount++;
        }
    }


    // Publish marker
    _recognizedObjectMarkerPublisher.publish(detectedObjectsMarkers);
    _detectedObjectPublisher.publish(detectedObjectsArray);
}



int main (int argc, char** argv)
{
    // Initialize ROS
    ros::init (argc, argv, "ICPRecognition");
    ros::NodeHandle nh;

    // Object for ICPRecognition
    ICPRecognition recognizer(nh);

    while(ros::ok())
        ros::spin ();
}
