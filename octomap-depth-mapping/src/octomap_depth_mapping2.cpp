#include "octomap_depth_mapping.hpp"
#include "depth_conversions.hpp"

#include <cv_bridge/cv_bridge.h>
#include <rclcpp_components/register_node_macro.hpp>

#include <Eigen/Geometry>

#ifdef CUDA
#include "cuda_proj.hpp"
#include <cuda_runtime.h>
#endif

namespace ph = std::placeholders;

namespace octomap_depth_mapping
{

OctomapDemap::OctomapDemap(const rclcpp::NodeOptions &options):
    Node("octomap_demap_node", options),
    fx(524.0),
    fy(524.0),
    cx(316.8),
    cy(238.5),
    padding(1),
    width(640),
    height(480),
    encoding("mono16"),
    frame_id("map"),
    filename(""),
    save_on_shutdown(false)
{
    // Parameter Declaration
    fx = this->declare_parameter("sensor_model.fx", fx);
    fy = this->declare_parameter("sensor_model.fy", fy);
    cx = this->declare_parameter("sensor_model.cx", cx);
    cy = this->declare_parameter("sensor_model.cy", cy);
    frame_id = this->declare_parameter("frame_id", frame_id);
    padding = this->declare_parameter("padding", padding);
    filename = this->declare_parameter("filename", filename);
    encoding = this->declare_parameter("encoding", encoding);
    save_on_shutdown = this->declare_parameter("save_on_shutdown", save_on_shutdown);
    width = this->declare_parameter("width", width);
    height = this->declare_parameter("height", height);

    rclcpp::QoS qos_profile(rclcpp::KeepLast(5));
    qos_profile.best_effort();
    auto rmw_qos_profile = qos_profile.get_rmw_qos_profile();

    // Publishers
    octomap_publisher_ = this->create_publisher<octomap_msgs::msg::Octomap>("map_out", qos_profile);

    // Subscribers with TimeSynchronizer
    RCLCPP_INFO(this->get_logger(), "Setting up message_filters subscribers...");
    depth_sub_.subscribe(this, "image_in", rmw_qos_profile);
    pose_sub_.subscribe(this, "pose_in", rmw_qos_profile);

    sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Image,
        geometry_msgs::msg::PoseStamped>>(depth_sub_, pose_sub_, 5);
    sync_->registerCallback(std::bind(&OctomapDemap::demap_callback, this, ph::_1, ph::_2));

    // Services
    octomap_srv_ = this->create_service<octomap_msgs::srv::GetOctomap>("get_octomap",
        std::bind(&OctomapDemap::octomap_srv, this, ph::_1, ph::_2));
    reset_srv_ = this->create_service<std_srvs::srv::Empty>("reset",
        std::bind(&OctomapDemap::reset_srv, this, ph::_1, ph::_2));
    save_srv_ = this->create_service<std_srvs::srv::Empty>("save",
        std::bind(&OctomapDemap::save_srv, this, ph::_1, ph::_2));

    // OctoMap Initialization
    double resolution = this->declare_parameter("resolution", 0.1);
    double probHit    = this->declare_parameter("sensor_model.hit", 0.7);
    double probMiss   = this->declare_parameter("sensor_model.miss", 0.4);
    double thresMin   = this->declare_parameter("sensor_model.min", 0.12);
    double thresMax   = this->declare_parameter("sensor_model.max", 0.97);

    ocmap = std::make_shared<octomap::OcTree>(resolution);
    read_ocmap();

    ocmap->setResolution(resolution);
    ocmap->setProbHit(probHit);
    ocmap->setProbMiss(probMiss);
    ocmap->setClampingThresMin(thresMin);
    ocmap->setClampingThresMax(thresMax);

#ifdef CUDA
    pc_count = 0;
    for(int i = 0; i < width; i+=padding) {
        for(int j = 0; j < height; j+=padding) {
            pc_count+=3;
        }
    }
    pc_size = pc_count * sizeof(double);
    RCLCPP_INFO(this->get_logger(), "CUDA point cloud count: %d", pc_count);
    // CORRECTED: This call is now type-safe because gpu_depth is void*
    cudaMalloc(&gpu_depth, width * height * sizeof(float));
    cudaMalloc<double>(&gpu_pc, pc_size);
    pc = (double*)malloc(pc_size);
    block.x = 32;
    block.y = 32;
    grid.x = (width + block.x - 1) / block.x;
    grid.y = (height + block.y - 1) / block.y;
#endif

    RCLCPP_INFO(this->get_logger(), "--- Launch Parameters ---");
    RCLCPP_INFO_STREAM(this->get_logger(), "frame_id: " << frame_id << ", resolution: " << resolution);
    RCLCPP_INFO_STREAM(this->get_logger(), "Image Size: " << width << "x" << height << ", padding: " << padding);
    RCLCPP_INFO_STREAM(this->get_logger(), "Intrinsics (fx, fy, cx, cy): " << fx << ", " << fy << ", " << cx << ", " << cy);
    RCLCPP_INFO_STREAM(this->get_logger(), "Sensor Model (hit, miss, min, max): " << probHit << ", " << probMiss << ", " << thresMin << ", " << thresMax);
    RCLCPP_INFO_STREAM(this->get_logger(), "Octomap file: " << filename << ", save_on_shutdown: " << (save_on_shutdown ? "true" : "false"));
    RCLCPP_INFO(this->get_logger(), "-------------------------");
    RCLCPP_INFO(this->get_logger(), "Octomap mapping node setup is complete.");
}

OctomapDemap::~OctomapDemap()
{
    if(save_on_shutdown && !filename.empty()) {
        if(save_ocmap()) {
            RCLCPP_INFO_STREAM(this->get_logger(), "Successfully saved OctoMap to " << filename << " on shutdown.");
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to save OctoMap on shutdown.");
        }
    }

#ifdef CUDA
    cudaFree(gpu_depth);
    cudaFree(gpu_pc);
    free(pc);
#endif
}

void OctomapDemap::demap_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
    const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose_msg)
{
    RCLCPP_INFO_ONCE(this->get_logger(), "SYNC SUCCESS: First matched message pair received! Processing map updates.");
    
    try {
        update_map(depth_msg, pose_msg->pose);
        publish_all();
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in demap_callback: %s", e.what());
    }
}

// REMOVED: The definition for processDepthImage is now in the header file.

void OctomapDemap::update_map(const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg, const geometry_msgs::msg::Pose& pose)
{
    Eigen::Quaterniond rotation(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    Eigen::Vector3d translation(pose.position.x, pose.position.y, pose.position.z);
    
    octomap::pose6d sensor_to_world(
        octomap::point3d(translation.x(), translation.y(), translation.z()),
        octomath::Quaternion(rotation.w(), rotation.x(), rotation.y(), rotation.z())
    );

    auto start = this->now();

#ifdef CUDA
    double tx = translation.x();
    double ty = translation.y();
    double tz = translation.z();

    Eigen::Matrix3d rot_matrix = rotation.toRotationMatrix();
    double r11 = rot_matrix(0, 0); double r12 = rot_matrix(0, 1); double r13 = rot_matrix(0, 2);
    double r21 = rot_matrix(1, 0); double r22 = rot_matrix(1, 1); double r23 = rot_matrix(1, 2);
    double r31 = rot_matrix(2, 0); double r32 = rot_matrix(2, 1); double r33 = rot_matrix(2, 2);
    
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(depth_msg);
    size_t depth_size = cv_ptr->image.total() * cv_ptr->image.elemSize();

    cudaMemcpy(gpu_depth, cv_ptr->image.ptr(), depth_size, cudaMemcpyHostToDevice);
    
    project_depth_img((ushort*)gpu_depth, gpu_pc, width, padding,
        grid, block,
        fx, fy, cx, cy,
        r11, r12, r13,
        r21, r22, r23,
        r31, r32, r33,
        tx, ty, tz);
        
    cudaMemcpy(pc, gpu_pc, pc_size, cudaMemcpyDeviceToHost);
    
    octomap::Pointcloud point_cloud;
    for(int i = 0; i < pc_count; i += 3) {
        if(pc[i] == 0 && pc[i+1] == 0 && pc[i+2] == 0) continue;
        point_cloud.push_back(pc[i], pc[i+1], pc[i+2]);
    }
    ocmap->insertPointCloud(point_cloud, sensor_to_world.trans());
#else
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(depth_msg);
    const cv::Mat& depth_image = cv_ptr->image;

    if (depth_msg->encoding == "16UC1" || depth_msg->encoding == "mono16") {
        processDepthImage<uint16_t>(depth_image, sensor_to_world);
    } else if (depth_msg->encoding == "32FC1") {
        processDepthImage<float>(depth_image, sensor_to_world);
    } else {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Unsupported depth encoding: %s", depth_msg->encoding.c_str());
        return;
    }
#endif

    auto diff = this->now() - start;
    RCLCPP_DEBUG(this->get_logger(), "Map update took: %.4f seconds", diff.seconds());
}

void OctomapDemap::publish_all()
{
    octomap_msgs::msg::Octomap msg;
    msg_from_ocmap(msg);
    octomap_publisher_->publish(msg);
}

bool OctomapDemap::octomap_srv(
    const std::shared_ptr<octomap_msgs::srv::GetOctomap::Request> /*req*/,
    std::shared_ptr<octomap_msgs::srv::GetOctomap::Response> res)
{
    return msg_from_ocmap(res->map);
}

bool OctomapDemap::save_srv(
    const std::shared_ptr<std_srvs::srv::Empty::Request> /*req*/,
    const std::shared_ptr<std_srvs::srv::Empty::Response> /*res*/)
{
    if (filename.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Cannot save map, no filename parameter was provided.");
        return false;
    }
    if (save_ocmap()) {
        RCLCPP_INFO_STREAM(this->get_logger(), "OctoMap successfully saved to " << filename);
        return true;
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to save OctoMap.");
        return false;
    }
}

bool OctomapDemap::reset_srv(
    const std::shared_ptr<std_srvs::srv::Empty::Request> /*req*/,
    const std::shared_ptr<std_srvs::srv::Empty::Response> /*res*/)
{
    ocmap->clear();
    RCLCPP_INFO(this->get_logger(), "OctoMap has been reset.");
    publish_all();
    return true;
}

bool OctomapDemap::read_ocmap()
{
    if (filename.length() <= 3) return false;

    std::string ext = filename.substr(filename.length() - 3, 3);
    bool success = false;
    if (ext == ".bt") {
        success = ocmap->readBinary(filename);
    } else if (ext == ".ot") {
        auto tree = octomap::AbstractOcTree::read(filename);
        if (tree) {
            ocmap.reset(dynamic_cast<octomap::OcTree*>(tree));
            success = (ocmap != nullptr);
        }
    } else {
        RCLCPP_WARN_STREAM(this->get_logger(), "Ignoring file '" << filename << "' with unknown extension '" << ext << "'.");
        return false;
    }

    if (success) {
        publish_all();
        RCLCPP_INFO_STREAM(this->get_logger(), "OctoMap read successfully from " << filename);
    } else {
        RCLCPP_ERROR_STREAM(this->get_logger(), "Failed to read OctoMap from " << filename);
    }
    return success;
}

bool OctomapDemap::save_ocmap()
{
    if (filename.length() <= 3) return false;

    std::string ext = filename.substr(filename.length() - 3, 3);
    if (ext == ".bt") {
        return ocmap->writeBinary(filename);
    } else if (ext == ".ot") {
        return ocmap->write(filename);
    }
    return false;
}

} // namespace octomap_depth_mapping

RCLCPP_COMPONENTS_REGISTER_NODE(octomap_depth_mapping::OctomapDemap)