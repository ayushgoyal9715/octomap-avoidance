#include "octomap_depth_mapping.hpp"
#include "depth_conversions.hpp"

#include <cv_bridge/cv_bridge.h>

#ifdef CUDA
#include "cuda_proj.hpp"
#include <cuda_runtime.h>
#endif

namespace ph = std::placeholders;

namespace octomap_depth_mapping
{

OctomapDemap::OctomapDemap(const rclcpp::NodeOptions &options, const std::string node_name): 
    Node(node_name, options),
    fx(524),
    fy(524),
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
    fx = this->declare_parameter("sensor_model/fx", fx);
    fy = this->declare_parameter("sensor_model/fy", fy);
    cx = this->declare_parameter("sensor_model/cx", cx);
    cy = this->declare_parameter("sensor_model/cy", cy);
    frame_id = this->declare_parameter("frame_id", frame_id);
    padding = this->declare_parameter("padding", padding);
    filename = this->declare_parameter("filename", filename);
    encoding = this->declare_parameter("encoding", encoding);
    save_on_shutdown = this->declare_parameter("save_on_shutdown", save_on_shutdown);
    width = this->declare_parameter("width", width);
    height = this->declare_parameter("height", height);

    rclcpp::QoS qos(rclcpp::KeepLast(3));
    qos.best_effort();

    // pubs
    octomap_publisher_ = this->create_publisher<octomap_msgs::msg::Octomap>("map_out", qos);

    auto rmw_qos_profile = qos.get_rmw_qos_profile();
    // subs
    depth_sub_.subscribe(this, "image_in", rmw_qos_profile);
    pose_sub_.subscribe(this, "pose_in", rmw_qos_profile);

    // sync depth + pose
    sync_ = std::make_shared<message_filters::TimeSynchronizer<
        sensor_msgs::msg::Image, geometry_msgs::msg::PoseStamped>>(depth_sub_, pose_sub_, 3);
    sync_->registerCallback(std::bind(&OctomapDemap::demap_callback, this, ph::_1, ph::_2));

    // services
    octomap_srv_ = this->create_service<octomap_msgs::srv::GetOctomap>("get_octomap", 
        std::bind(&OctomapDemap::octomap_srv, this, ph::_1, ph::_2));

    reset_srv_ = this->create_service<std_srvs::srv::Empty>("reset", 
        std::bind(&OctomapDemap::reset_srv, this, ph::_1, ph::_2));

    save_srv_ = this->create_service<std_srvs::srv::Empty>("save", 
        std::bind(&OctomapDemap::save_srv, this, ph::_1, ph::_2));

    double resolution = this->declare_parameter("resolution", 0.1);
    double probHit    = this->declare_parameter("sensor_model/hit", 0.7);
    double probMiss   = this->declare_parameter("sensor_model/miss", 0.4);
    double thresMin   = this->declare_parameter("sensor_model/min", 0.12);
    double thresMax   = this->declare_parameter("sensor_model/max", 0.97);

    ocmap = std::make_shared<octomap::OcTree>(resolution);
    read_ocmap(); // read octomap if file provided

    // overwrite sensor model values
    ocmap->setResolution(resolution);
    ocmap->setProbHit(probHit);
    ocmap->setProbMiss(probMiss);
    ocmap->setClampingThresMin(thresMin);
    ocmap->setClampingThresMax(thresMax);

#ifdef CUDA
    pc_count = 0;
    for(int i = 0; i < width; i+=padding)
    {
        for(int j = 0; j < height; j+=padding)
        {
            pc_count+=3;   
        }
    }

    pc_size = pc_count * sizeof(double);
    depth_size = width*height*sizeof(ushort);

    RCLCPP_INFO(this->get_logger(), "%d", pc_count);

    cudaMalloc<ushort>(&gpu_depth, depth_size);
    cudaMalloc<double>(&gpu_pc, pc_size);
    pc = (double*)malloc(pc_size);

    block.x = 32;
    block.y = 32;
    grid.x = (width + block.x - 1) / block.x;
    grid.y = (height + block.y - 1) / block.y;
#endif

    RCLCPP_INFO(this->get_logger(), "Setup is done");
}

OctomapDemap::~OctomapDemap()
{
    if(!save_on_shutdown)
        return;

    if(save_ocmap())
    {
        RCLCPP_INFO_STREAM(this->get_logger(), "Save on shutdown successful " << filename);
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Save on shutdown failed");
    }

#ifdef CUDA
    cudaFree(gpu_depth);
    cudaFree(gpu_pc);
    free(pc);
#endif
}

bool OctomapDemap::octomap_srv(
    const std::shared_ptr<octomap_msgs::srv::GetOctomap::Request> req, 
    std::shared_ptr<octomap_msgs::srv::GetOctomap::Response> res)
{
    return msg_from_ocmap(res->map);
}

bool OctomapDemap::save_srv(
    const std::shared_ptr<std_srvs::srv::Empty::Request> req, 
    const std::shared_ptr<std_srvs::srv::Empty::Response> res)
{   
    if(save_ocmap())
    {
        RCLCPP_INFO_STREAM(this->get_logger(), "Octomap is saved to " << filename);
        return true;
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Octomap is not saved");
        return false;
    }
}

bool OctomapDemap::reset_srv(
    const std::shared_ptr<std_srvs::srv::Empty::Request> req, 
    const std::shared_ptr<std_srvs::srv::Empty::Response> res)
{
    ocmap->clear();
    RCLCPP_INFO(this->get_logger(), "Octomap reset");
    return true;
}

void OctomapDemap::demap_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg, 
    const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose_msg)
{
    auto cv_ptr = cv_bridge::toCvCopy(depth_msg, encoding);
    update_map(cv_ptr->image, pose_msg->pose);
    publish_all();
}

void OctomapDemap::publish_all()
{
    octomap_msgs::msg::Octomap msg;
    msg_from_ocmap(msg);
    octomap_publisher_->publish(msg);
}

// === Manual transformation (no tf2) ===
void OctomapDemap::update_map(const cv::Mat& depth, const geometry_msgs::msg::Pose& pose)
{
    double tx = pose.position.x;
    double ty = pose.position.y;
    double tz = pose.position.z;
    octomap::point3d origin(tx, ty, tz);

    double qx = pose.orientation.x;
    double qy = pose.orientation.y;
    double qz = pose.orientation.z;
    double qw = pose.orientation.w;

    double qx2 = qx * qx;
    double qy2 = qy * qy;
    double qz2 = qz * qz;

    double r11 = 1.0 - 2.0 * (qy2 + qz2);
    double r12 = 2.0 * (qx * qy - qw * qz);
    double r13 = 2.0 * (qx * qz + qw * qy);

    double r21 = 2.0 * (qx * qy + qw * qz);
    double r22 = 1.0 - 2.0 * (qx2 + qz2);
    double r23 = 2.0 * (qy * qz - qw * qx);

    double r31 = 2.0 * (qx * qz - qw * qy);
    double r32 = 2.0 * (qy * qz + qw * qx);
    double r33 = 1.0 - 2.0 * (qx2 + qy2);

    auto start = this->now();

#ifdef CUDA
    cudaMemcpy(gpu_depth ,depth.ptr(), depth_size, cudaMemcpyHostToDevice);

    project_depth_img(gpu_depth, gpu_pc, width, padding,
        grid, block,
        fx, fy, cx, cy,
        r11, r12, r13,
        r21, r22, r23,
        r31, r32, r33,
        tx, ty, tz);

    cudaMemcpy(pc, gpu_pc, pc_size, cudaMemcpyDeviceToHost);

    for(int i = 0, n = pc_count-3; i < n; i+=3)
    {
        if(pc[i] == 0 && pc[i+1] == 0 && pc[i+2] == 0) { continue; }
        ocmap->insertRay(origin, octomap::point3d(pc[i], pc[i+1], pc[i+2]));
    }
#else
    for(int i = 0; i < depth.rows; i+=padding)
    {
        const ushort* row = depth.ptr<ushort>(i);

        for(int j = 0; j < depth.cols; j+=padding)
        {
            const double d = depth_to_meters(row[j]);
            if(d == 0) continue;

            double px = (j - cx) * d / fx;
            double py = (i - cy) * d / fy;
            double pz = d;

            double world_x = r11 * px + r12 * py + r13 * pz + tx;
            double world_y = r21 * px + r22 * py + r23 * pz + ty;
            double world_z = r31 * px + r32 * py + r33 * pz + tz;

            ocmap->insertRay(origin, octomap::point3d(world_x, world_y, world_z));
        }
    }
#endif

    auto end = this->now();
    auto diff = end - start;
    RCLCPP_INFO(this->get_logger(), "update map time(sec) : %.4f", diff.seconds());
}

bool OctomapDemap::read_ocmap()
{
    if(filename.length() <= 3)
        return false;

    std::string ext = filename.substr(filename.length()-3, 3);

    if(ext == ".bt")
    {
        if (!ocmap->readBinary(filename))
            return false;
    }
    else if(ext == ".ot")
    {
        auto tree = octomap::AbstractOcTree::read(filename);
        octomap::OcTree *octree = dynamic_cast<octomap::OcTree*>(tree);
        ocmap = std::shared_ptr<octomap::OcTree>(octree);
    }
    else 
        return false;

    if(!ocmap)
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to read octomap");
        return false;
    }

    publish_all();
    RCLCPP_INFO_STREAM(this->get_logger(), "Octomap read from " << filename);
    return true;
}

bool OctomapDemap::save_ocmap()
{
    if(filename.length() <= 3)
        return false;

    std::string ext = filename.substr(filename.length()-3, 3);

    if(ext == ".bt")
    {
        if (!ocmap->writeBinary(filename))
            return false;
    }
    else if(ext == ".ot")
    {
        if (!ocmap->write(filename))
            return false;
    }
    else 
        return false;

    return true;
}

} // namespace octomap_depth_mapping

RCLCPP_COMPONENTS_REGISTER_NODE(octomap_depth_mapping::OctomapDemap)
