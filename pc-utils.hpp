// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
#pragma once

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
//#include "example.hpp"          // Include short list of convenience functions for rendering

#include <algorithm>            // std::min, std::max

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <Eigen/Core>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/gui/vpDisplayGDI.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/sensor/vpRealSense2.h>
#include <visp3/robot/vpRobotFranka.h>
#include <visp3/detection/vpDetectorAprilTag.h>
#include <visp3/visual_features/vpFeatureThetaU.h>
#include <visp3/visual_features/vpFeatureTranslation.h>
#include <visp3/vs/vpServo.h>
#include <visp3/vs/vpServoDisplay.h>
#include <visp3/gui/vpPlot.h>

#include <../imbs/imbs.hpp>
#include <../imbs/imbs.cpp>

#include <chrono>

typedef struct {
  Eigen::Vector3d p_tag;
  Eigen::Matrix3d R_tag;
  //threshold for filtering
  float x_min; 
  float x_max;
  float y_min;
  float y_max;
  float z_min;
  float z_max;
  //threshold for cluster
  float xc_min;
  float xc_max;
  float yc_min;
  float yc_max;
  float th_min; //threshold for number of points in cluster
  float cluster_tolerance;
  float leaf_size;
} detectionParam;

detectionParam param;

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds milliseconds;

int alignTag(Eigen::MatrixXd &T, std::string opt_robot_ip, std::string opt_eMc_filename, bool opt_adaptive_gain, double opt_tagSize, bool display_tag,
  bool opt_verbose, bool opt_plot, bool opt_task_sequencing, double threshold_t, double threshold_tu, int opt_quad_decimate);

Eigen::MatrixXd detectObjectBSP(std::string pathModel);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_to_pcl(const rs2::points& points, const rs2::video_frame& color, cv::Mat& filtered);
std::tuple<uint8_t, uint8_t, uint8_t> get_texcolor(rs2::video_frame texture, rs2::texture_coordinate texcoords);
float getZValue(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float x, float y);
void compute2D(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2D);
float getZMin(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_temp);

static Clock::time_point t0 = Clock::now();

void tic() {
  t0 = Clock::now();
}

void toc() {
    Clock::time_point t1 = Clock::now();
    milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
    std::cout <<"Elapsed time is "<< ms.count() << " milliseconds\n";
}


Eigen::MatrixXd detectObjectBSP(std::string pathModel) {
  rs2::pointcloud pc;
  rs2::points points;
  rs2::pipeline pipeline;
  rs2::pipeline_profile pipeline_profile;
  rs2::frameset frameset;

  rs2::frame color_frame;
  cv::Mat color_mat;

  rs2::frame depth_frame;
  cv::Mat depth_mat;
  

  Eigen::MatrixXd T_obj;
  T_obj.setIdentity(4,4);
  rs2::pipeline pipe;
  rs2::config cfg;
  rs2::colorizer color_map;

  //Add desired streams to configuration
  cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
  cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

  //Instruct pipeline to start streaming with the requested configuration
  pipe.start(cfg);
  rs2::align align_to_color(RS2_STREAM_COLOR);

  // Camera warmup - dropping several first frames to let auto-exposure stabilize
  rs2::frameset frames;
  for(int i = 0; i < 50; i++) {
    frames = pipe.wait_for_frames();
  }

  pcl::visualization::PCLVisualizer* viewer;
  viewer = new pcl::visualization::PCLVisualizer("3d viewer"); 
  viewer->setBackgroundColor(0.33f, 0.33f, 0.33f);
  viewer->initCameraParameters();
  viewer->setCameraPosition(0.0f, 0.0f, 0.0f, //0.0f,
            0.0f, 0.0f, 1.0f,
            0.0f, -1.0f, 0.0f);
  
  int num = 0;
  while(true) // Application still alive?
  {  
    tic();
    // Wait for the next set of frames from the camera
    frames = pipe.wait_for_frames();
    frames = align_to_color.process(frames);

    cv::Mat rgbImage(Size(640, 480), CV_8UC3, (void*)frames.get_color_frame().get_data(), Mat::AUTO_STEP);
    cv::Mat depth(Size(640, 480), CV_8UC3, (void*)frames.get_depth_frame().apply_filter(color_map).get_data(), Mat::AUTO_STEP);
    rs2::depth_frame depthImage = frames.get_depth_frame();
        
    // Retrieve Color Flame
    color_frame = frames.get_color_frame();
    if( !color_frame ){
      exit(0);
    }      
        
    // Tell pointcloud object to map to this color frame
    pc.map_to(color_frame);
    // Retrieve Depth Flame
    depth_frame = frames.get_depth_frame();
    if( !depth_frame ){
      exit(0);
    }

    // Generate the pointcloud and texture mappings
    points = pc.calculate(depth_frame);

    BackgroundSubtractorIMBS* pIMBS;
    //pIMBS = new BackgroundSubtractorIMBS(30);//, 20, 5, 500, 2, 25, 0.65, 1.15, 60., 40., 30., 10000., false);
    double hh = 40.;
    double ss = 80.;
    double aa = 0.65; //v
    pIMBS = new BackgroundSubtractorIMBS(30,20,5,500.,2,25,aa,1.15,ss,hh,30,10000.,false);
    pIMBS->loadBg(pathModel.c_str());

    cv::Mat frame = rgbImage;
    cv::Mat fgMask;
    //update the background model
    pIMBS->apply(frame, fgMask);
    //get background image
    cv::Mat bgImage;
    pIMBS->getBackgroundImage(bgImage);

    cv::Mat filteredFg;
    int morph_size = 3;
    cv::Mat element = getStructuringElement(MORPH_RECT, Size( morph_size + 1, morph_size+1 ), Point( morph_size, morph_size ) );

    /// Apply the specified morphology operation
    morphologyEx( fgMask, filteredFg, MORPH_OPEN, element );

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = points_to_pcl(points, color_frame, filteredFg);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_voxel(new pcl::PointCloud<pcl::PointXYZRGB>);
    if(param.leaf_size == 0.0){
      pcl::copyPointCloud(*cloud, *cloud_voxel);
    } else {    
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
      voxel_grid.setInputCloud(cloud);
      voxel_grid.setLeafSize(param.leaf_size,param.leaf_size,param.leaf_size);//(0.01,0.01,0.01); //0.01, 0.01, 0.01);
      voxel_grid.filter(*cloud_voxel);
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud_voxel, *cloud_temp);
    
    pcl::PassThrough<pcl::PointXYZRGB> Cloud_Filter0; // Create the filtering object
    Cloud_Filter0.setInputCloud (cloud_temp);           // Input generated cloud to filter
    Cloud_Filter0.setFilterFieldName ("z");        // Set field name to Z-coordinate
    Cloud_Filter0.setFilterLimits(param.z_min, param.z_max);// (-20, 20);      // Set accepted interval values 465
    Cloud_Filter0.filter (*cloud_temp);  
    
    pcl::PassThrough<pcl::PointXYZRGB> Cloud_Filter; // Create the filtering object
    Cloud_Filter.setInputCloud (cloud_temp);           // Input generated cloud to filter
    Cloud_Filter.setFilterFieldName ("y");        // Set field name to Z-coordinate
    Cloud_Filter.setFilterLimits(param.y_min, param.y_max);// (-0.12, 0.12);      // Set accepted interval values 465
    Cloud_Filter.filter (*cloud_temp);  

    pcl::PassThrough<pcl::PointXYZRGB> Cloud_Filter2; // Create the filtering object
    Cloud_Filter2.setInputCloud (cloud_temp);           // Input generated cloud to filter
    Cloud_Filter2.setFilterFieldName ("x");        // Set field name to Z-coordinate
    Cloud_Filter2.setFilterLimits(param.x_min, param.x_max);// (-0.175, 0.175);      // Set accepted interval values 465
    Cloud_Filter2.filter (*cloud_temp); 

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2D(new pcl::PointCloud<pcl::PointXYZRGB>);
    compute2D(cloud_temp, cloud2D);


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr subsamp_cloud(new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::copyPointCloud(*cloud_temp, *subsamp_cloud);


    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(subsamp_cloud);

    pcl::IndicesPtr indices (new std::vector <int>);
    std::vector<pcl::PointIndices> cluster_indices;
   //---------------------------------------Euclidean segmentation
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    // Set distance threshold
    ec.setClusterTolerance(param.cluster_tolerance);// (0.02);
    ec.setMinClusterSize (10);
    //ec.setMaxClusterSize (5000);
    // Set the KD tree that will be used to look for points neighbors
    ec.setSearchMethod (tree);
    ec.setInputCloud (subsamp_cloud);
    // Get the clusters (i.e., the point indices that belong to each cluster)
    std::vector <pcl::PointIndices> clusters;
    ec.extract (clusters);

    std::string base_name("Cluster");  
    int j = 0;


    for (std::vector<pcl::PointIndices>::const_iterator it = clusters.begin (); 
         it != clusters.end (); ++it)
    {
      // Create a cloud for each cluster
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
      // Iterate for each index
      for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
        cloud_cluster->points.push_back (subsamp_cloud->points[*pit]);
        
      cloud_cluster->width = cloud_cluster->points.size ();
      cloud_cluster->height = 1;
      cloud_cluster->is_dense = true;

      float zMin = getZMin(cloud_cluster);

      pcl::PassThrough<pcl::PointXYZRGB> Cloud_Filter3; // Create the filtering object
      Cloud_Filter0.setInputCloud (cloud_cluster);           // Input generated cloud to filter
      Cloud_Filter0.setFilterFieldName ("z");        // Set field name to Z-coordinate
      Cloud_Filter0.setFilterLimits(zMin, zMin+0.02); // Set accepted interval values 465
      Cloud_Filter0.filter (*cloud_cluster); 

      
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_c_2D(new pcl::PointCloud<pcl::PointXYZRGB>);
      compute2D(cloud_cluster, cloud_c_2D);

      int pp = rand();
      pcl::PointCloud<pcl::PointXYZ>::Ptr pc_temp (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::copyPointCloud(*cloud_c_2D, *pc_temp);
      

      //BB
      pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
      feature_extractor.setInputCloud (pc_temp);
      feature_extractor.compute ();

      std::vector <float> moment_of_inertia;
      std::vector <float> eccentricity;
      pcl::PointXYZ min_point_AABB;
      pcl::PointXYZ max_point_AABB;
      pcl::PointXYZ min_point_OBB;
      pcl::PointXYZ max_point_OBB;
      pcl::PointXYZ position_OBB;
      Eigen::Matrix3f rotational_matrix_OBB;
      float major_value, middle_value, minor_value;
      Eigen::Vector3f major_vector, middle_vector, minor_vector;
      Eigen::Vector3f mass_center;

      feature_extractor.getMomentOfInertia (moment_of_inertia);
      feature_extractor.getEccentricity (eccentricity);
      feature_extractor.getAABB (min_point_AABB, max_point_AABB);
      feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
      feature_extractor.getEigenValues (major_value, middle_value, minor_value);
      feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
      feature_extractor.getMassCenter (mass_center);

      std::stringstream sbuff; sbuff<<"cloud"; sbuff<<pp;
      std::stringstream sbuff_obb; sbuff_obb<<"obb"; sbuff_obb<<pp;
      viewer->addPointCloud<pcl::PointXYZ> (pc_temp, sbuff.str());
      viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, sbuff.str());
      Eigen::Vector3f position (position_OBB.x, position_OBB.y, position_OBB.z);
      Eigen::Quaternionf quat (rotational_matrix_OBB);
      float dim_x = max_point_OBB.x - min_point_OBB.x;
      float dim_y = max_point_OBB.y - min_point_OBB.y;
      //end-BB
      
      toc();
      if(num > 0 && clusters.size() > 0 && (cloud_cluster->size() > param.th_min) && (param.xc_min <= dim_x && dim_x <= param.xc_max) && (param.yc_min <= dim_y && dim_y <= param.yc_max)){
        viewer->addCube (position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, sbuff_obb.str());
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, sbuff_obb.str());
          
        pcl::PointXYZ center (mass_center (0), mass_center (1), mass_center (2));
        pcl::PointXYZ x_axis (major_vector (0) + mass_center (0), major_vector (1) + mass_center (1), major_vector (2) + mass_center (2));
        pcl::PointXYZ y_axis (middle_vector (0) + mass_center (0), middle_vector (1) + mass_center (1), middle_vector (2) + mass_center (2));
        pcl::PointXYZ z_axis (minor_vector (0) + mass_center (0), minor_vector (1) + mass_center (1), minor_vector (2) + mass_center (2));
          
        std::stringstream sbuff_m1; sbuff_m1<<"major_eigen_vector_"; sbuff_m1<<pp;
        std::stringstream sbuff_m2; sbuff_m2<<"middle_eigen_vector_"; sbuff_m2<<pp;
        std::stringstream sbuff_m3; sbuff_m3<<"minor_eigen_vector_"; sbuff_m3<<pp;
        viewer->addLine (center, x_axis, 1.0f, 0.0f, 0.0f, sbuff_m1.str());
        viewer->addLine (center, y_axis, 0.0f, 1.0f, 0.0f, sbuff_m2.str());
        viewer->addLine (center, z_axis, 0.0f, 0.0f, 1.0f, sbuff_m3.str());
        viewer->spinOnce(100); 
        //--------------------------------------
        T_obj(0,0) = rotational_matrix_OBB(0,0); T_obj(0,1) = rotational_matrix_OBB(0,1); T_obj(0,2) = rotational_matrix_OBB(0,2);
        T_obj(1,0) = rotational_matrix_OBB(1,0); T_obj(1,1) = rotational_matrix_OBB(1,1); T_obj(1,2) = rotational_matrix_OBB(1,2);
        T_obj(2,0) = rotational_matrix_OBB(2,0); T_obj(2,1) = rotational_matrix_OBB(2,1); T_obj(2,2) = rotational_matrix_OBB(2,2);
        T_obj(0,3) = position[0];
        T_obj(1,3) = position[1];
        T_obj(2,3) = getZValue(cloud, mass_center(0),mass_center (1));
        return T_obj;
      } else if (num > 5){
        return T_obj;
      }
      j++;
    }
    clusters.clear();
    indices->clear();
    num++;
    viewer->spinOnce(100); 
  }
  pipe.stop();
  return T_obj;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_to_pcl(const rs2::points& points, const rs2::video_frame& color, cv::Mat& filtered){

    // OpenCV Mat for showing the rgb color image, just as part of processing
    Mat colorr(Size(640, 480), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", colorr);
        
    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    // Config of PCL Cloud object
    cloud->width = static_cast<uint32_t>(sp.width());
    cloud->height = static_cast<uint32_t>(sp.height());
    cloud->is_dense = false;
    cloud->points.resize(points.size());

    auto tex_coords = points.get_texture_coordinates();
    auto vertices = points.get_vertices();

    int idx = 0;
    for(int i = 0; i < cloud->height; ++i){
      for(int j = 0; j < cloud->width; ++j){
        if(filtered.at<uchar>(i,j) > 250) {//!= 0){
          cloud->points[idx].x = vertices[idx].x;
          cloud->points[idx].y = vertices[idx].y;
          cloud->points[idx].z = vertices[idx].z;

          std::tuple<uint8_t, uint8_t, uint8_t> current_color;
          current_color = get_texcolor(color, tex_coords[idx]);

          // Reversed order- 2-1-0 because of BGR model used in camera
          cloud->points[idx].r = std::get<2>(current_color);
          cloud->points[idx].g = std::get<1>(current_color);
          cloud->points[idx].b = std::get<0>(current_color);
        }
        idx++;
      }
    }


    
   return cloud;
}

std::tuple<uint8_t, uint8_t, uint8_t> get_texcolor(rs2::video_frame texture, rs2::texture_coordinate texcoords)
{
    const int w = texture.get_width(), h = texture.get_height();
    
    // convert normals [u v] to basic coords [x y]
    int x = std::min(std::max(int(texcoords.u*w + .5f), 0), w - 1);
    int y = std::min(std::max(int(texcoords.v*h + .5f), 0), h - 1);

    int idx = x * texture.get_bytes_per_pixel() + y * texture.get_stride_in_bytes();
    const auto texture_data = reinterpret_cast<const uint8_t*>(texture.get_data());
    return std::tuple<uint8_t, uint8_t, uint8_t>(texture_data[idx], texture_data[idx+1], texture_data[idx+2]);
}

float getZMin(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
  pcl::PointXYZRGB point;
  float min = 1000;
  for (int i= 0; i < cloud->points.size(); i++){
    point = cloud->points[i];
    if(point.z < min){
      min = point.z;
    }
  }
  return min;
}

float getZValue(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float x, float y){
  pcl::PointXYZRGB point;
  float eps = 0.005;
  for (int i= 0; i < cloud->points.size(); i++){
    point = cloud->points[i];
    if((x-eps <= point.x && point.x <= x+eps) && (y-eps <= point.y && point.y <= y+eps)){
      return point.z;
    }
  }
  return 0.0f;
}

void compute2D(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2D){
  pcl::PointXYZRGB point;
  point.r = 255;
  point.g = 255;
  point.b = 255;
  for (int i= 0; i < cloud->points.size(); i++){
    point.x = cloud->points[i].x;
    point.y = cloud->points[i].y;
    point.z = 0.57;
    cloud2D->points.push_back(point);
  }
}

//VISP
int alignTag(Eigen::MatrixXd &T, std::string opt_robot_ip, std::string opt_eMc_filename, bool opt_adaptive_gain, double opt_tagSize, bool display_tag,
  bool opt_verbose, bool opt_plot, bool opt_task_sequencing, double threshold_t, double threshold_tu, int opt_quad_decimate){
  /*double opt_tagSize = 0.017;
  std::string opt_robot_ip = "192.168.1.1";
  bool display_tag = true;
  int opt_quad_decimate = 2;
  bool opt_verbose = false;
  bool opt_plot = true;
  bool opt_task_sequencing = false;*/
  double convergence_threshold_t = threshold_t;//0.005; //0.0005;
  double convergence_threshold_tu = vpMath::rad(threshold_tu); //vpMath::rad(0.7); //vpMath::rad(0.5);

  vpRobotFranka robot;

  try {
    robot.connect(opt_robot_ip);

    vpRealSense2 rs;
    rs2::config config;
    unsigned int width = 1280, height = 720;
    config.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGBA8, 30);
    //config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGBA8, 30);
    //config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    //config.enable_stream(RS2_STREAM_INFRARED, 640, 480, RS2_FORMAT_Y8, 30);
    rs.open(config);

    // Get camera extrinsics
    vpPoseVector ePc;
    // Set camera extrinsics default values
    ePc[0] = 0.0337731; ePc[1] = -0.00535012; ePc[2] = -0.0523339;
    ePc[3] = -0.247294; ePc[4] = -0.306729; ePc[5] = 1.53055;

    // If provided, read camera extrinsics from --eMc <file>
    if (!opt_eMc_filename.empty()) {
      ePc.loadYAML(opt_eMc_filename, ePc);
    }
    else {
      std::cout << "Warning, opt_eMc_filename is empty! Use hard coded values." << "\n";
    }
    vpHomogeneousMatrix eMc(ePc);
    //std::cout << "eMc:\n" << eMc << "\n";

    // Get camera intrinsics
    vpCameraParameters cam = rs.getCameraParameters(RS2_STREAM_COLOR, vpCameraParameters::perspectiveProjWithDistortion);
    std::cout << "cam:\n" << cam << "\n";

    vpImage<unsigned char> I(height, width);

    #if defined(VISP_HAVE_X11)
        vpDisplayX dc(I, 10, 10, "Color image");
    #elif defined(VISP_HAVE_GDI)
        vpDisplayGDI dc(I, 10, 10, "Color image");
    #endif
    vpDetectorAprilTag::vpAprilTagFamily tagFamily = vpDetectorAprilTag::TAG_36h11;
    vpDetectorAprilTag::vpPoseEstimationMethod poseEstimationMethod = vpDetectorAprilTag::HOMOGRAPHY_VIRTUAL_VS;
    //vpDetectorAprilTag::vpPoseEstimationMethod poseEstimationMethod = vpDetectorAprilTag::BEST_RESIDUAL_VIRTUAL_VS;
    vpDetectorAprilTag detector(tagFamily);
    detector.setAprilTagPoseEstimationMethod(poseEstimationMethod);
    detector.setDisplayTag(display_tag);
    detector.setAprilTagQuadDecimate(opt_quad_decimate);

    // Servo
    vpHomogeneousMatrix cdMc, cMo, oMo;

    vpHomogeneousMatrix cdMo( vpTranslationVector(param.p_tag[0],param.p_tag[1],param.p_tag[2]),
                              vpRotationMatrix( {param.R_tag(0,0), param.R_tag(0,1), param.R_tag(0,2),
                                                 param.R_tag(1,0), param.R_tag(1,1), param.R_tag(1,2), 
                                                 param.R_tag(2,0), param.R_tag(2,1), param.R_tag(2,2),} ) );


    cdMc = cdMo * cMo.inverse();
    vpFeatureTranslation t(vpFeatureTranslation::cdMc);
    vpFeatureThetaU tu(vpFeatureThetaU::cdRc);
    t.buildFrom(cdMc);
    tu.buildFrom(cdMc);

    vpFeatureTranslation td(vpFeatureTranslation::cdMc);
    vpFeatureThetaU tud(vpFeatureThetaU::cdRc);

    vpServo task;
    task.addFeature(t, td);
    task.addFeature(tu, tud);
    task.setServo(vpServo::EYEINHAND_CAMERA);
    task.setInteractionMatrixType(vpServo::CURRENT);

    if (opt_adaptive_gain) {
      vpAdaptiveGain lambda(1.5, 0.4, 30); // lambda(0)=4, lambda(oo)=0.4 and lambda'(0)=30
      task.setLambda(lambda);
    }
    else {
      task.setLambda(0.5);
    }
    vpPlot *plotter = nullptr;
    int iter_plot = 0;
    if (opt_plot) {
      plotter = new vpPlot(2, static_cast<int>(250 * 2), 500, static_cast<int>(I.getWidth()) + 80, 10, "Real time curves plotter");
      plotter->setTitle(0, "Visual features error");
      plotter->setTitle(1, "Camera velocities");
      plotter->initGraph(0, 6);
      plotter->initGraph(1, 6);
      plotter->setLegend(0, 0, "error_feat_tx");
      plotter->setLegend(0, 1, "error_feat_ty");
      plotter->setLegend(0, 2, "error_feat_tz");
      plotter->setLegend(0, 3, "error_feat_theta_ux");
      plotter->setLegend(0, 4, "error_feat_theta_uy");
      plotter->setLegend(0, 5, "error_feat_theta_uz");
      plotter->setLegend(1, 0, "vc_x");
      plotter->setLegend(1, 1, "vc_y");
      plotter->setLegend(1, 2, "vc_z");
      plotter->setLegend(1, 3, "wc_x");
      plotter->setLegend(1, 4, "wc_y");
      plotter->setLegend(1, 5, "wc_z");
    }
    bool final_quit = false;
    bool has_converged = false;
    bool send_velocities = false;
    bool servo_started = false;
    std::vector<vpImagePoint> *traj_vip = nullptr; // To memorize point trajectory

    static double t_init_servo = vpTime::measureTimeMs();

    robot.set_eMc(eMc); // Set location of the camera wrt end-effector frame
    robot.setRobotState(vpRobot::STATE_VELOCITY_CONTROL);

    while (!has_converged && !final_quit) {
      double t_start = vpTime::measureTimeMs();

      rs.acquire(I);

      vpDisplay::display(I);

      std::vector<vpHomogeneousMatrix> cMo_vec;
      detector.detect(I, opt_tagSize, cam, cMo_vec);

      std::stringstream ss;
      ss << "Left click to " << (send_velocities ? "stop the robot" : "servo the robot") << ", right click to quit.";
      vpDisplay::displayText(I, 20, 20, ss.str(), vpColor::red);

      vpColVector v_c(6);

      // Only one tag is detected
      if (cMo_vec.size() == 1) {
        cMo = cMo_vec[0];

        static bool first_time = true;
        if (first_time) {
          // Introduce security wrt tag positionning in order to avoid PI rotation
          std::vector<vpHomogeneousMatrix> v_oMo(2), v_cdMc(2);
          v_oMo[1].buildFrom(0, 0, 0, 0, 0, M_PI);
          for (size_t i = 0; i < 2; i++) {
            v_cdMc[i] = cdMo * v_oMo[i] * cMo.inverse();
          }
          if (std::fabs(v_cdMc[0].getThetaUVector().getTheta()) < std::fabs(v_cdMc[1].getThetaUVector().getTheta())) {
            oMo = v_oMo[0];
          }
          else {
            std::cout << "Desired frame modified to avoid PI rotation of the camera" << std::endl;
            oMo = v_oMo[1];   // Introduce PI rotation
          }
        }

        // Update visual features
        cdMc = cdMo * oMo * cMo.inverse();

        t.buildFrom(cdMc);
        tu.buildFrom(cdMc);

        if (opt_task_sequencing) {
          if (! servo_started) {
            if (send_velocities) {
              servo_started = true;
            }
            t_init_servo = vpTime::measureTimeMs();
          }
          v_c = task.computeControlLaw((vpTime::measureTimeMs() - t_init_servo)/1000.);
        }
        else {
          v_c = task.computeControlLaw();
        }

        // Display desired and current pose features
        vpDisplay::displayFrame(I, cdMo * oMo, cam, opt_tagSize / 1.5, vpColor::yellow, 2);
        vpDisplay::displayFrame(I, cMo,  cam, opt_tagSize / 2,   vpColor::none,   3);
        // Get tag corners
        std::vector<vpImagePoint> vip = detector.getPolygon(0);
        // Get the tag cog corresponding to the projection of the tag frame in the image
        vip.push_back(detector.getCog(0));
        // Display the trajectory of the points
        if (first_time) {
           traj_vip = new std::vector<vpImagePoint> [vip.size()];
        }
        //display_point_trajectory(I, vip, traj_vip);

        if (opt_plot) {
          plotter->plot(0, iter_plot, task.getError());
          plotter->plot(1, iter_plot, v_c);
          iter_plot++;
        }

        if (opt_verbose) {
          std::cout << "v_c: " << v_c.t() << std::endl;
        }

        vpTranslationVector cd_t_c = cdMc.getTranslationVector();
        vpThetaUVector cd_tu_c = cdMc.getThetaUVector();
        double error_t = sqrt(cd_t_c.sumSquare());
        double error_tu = vpMath::deg(sqrt(cd_tu_c.sumSquare()));

        ss.str("");
        ss << "error_t: " << error_t;
        vpDisplay::displayText(I, 20, static_cast<int>(I.getWidth()) - 150, ss.str(), vpColor::red);
        ss.str("");
        ss << "error_tu: " << error_tu;
        vpDisplay::displayText(I, 40, static_cast<int>(I.getWidth()) - 150, ss.str(), vpColor::red);

        if (opt_verbose)
          std::cout << "error translation: " << error_t << " ; error rotation: " << error_tu << std::endl;

        if (error_t < convergence_threshold_t && error_tu < convergence_threshold_tu) {
          has_converged = true;
          std::cout << "Servo task has converged." << std::endl;;
          vpDisplay::displayText(I, 100, 20, "Servo task has converged", vpColor::red);
        }

        if (first_time) {
          first_time = false;
        }
      } // end if (cMo_vec.size() == 1)
      else {
        v_c = 0;
      }

      if (!send_velocities) {
        v_c = 0;
      }

      // Send to the robot
      robot.setVelocity(vpRobot::CAMERA_FRAME, v_c);

      ss.str("");
      ss << "Loop time: " << vpTime::measureTimeMs() - t_start << " ms";
      vpDisplay::displayText(I, 40, 20, ss.str(), vpColor::red);
      vpDisplay::flush(I);

      vpMouseButton::vpMouseButtonType button;
      if (vpDisplay::getClick(I, button, false)) {
        switch (button) {
        case vpMouseButton::button1:
          send_velocities = !send_velocities;
          break;

        case vpMouseButton::button3:
          final_quit = true;
          v_c = 0;
          break;

        default:
          break;
        }
      }
    }

    std::cout << "Stop the robot " << std::endl;
    vpPoseVector wPe;
    robot.getPosition(vpRobot::END_EFFECTOR_FRAME, wPe);
    vpHomogeneousMatrix M(wPe);
    T(0,0) = M[0][0]; T(0,1) = M[0][1]; T(0,2) = M[0][2]; T(0,3) = M[0][3];
    T(1,0) = M[1][0]; T(1,1) = M[1][1]; T(1,2) = M[1][2]; T(1,3) = M[1][3];
    T(2,0) = M[2][0]; T(2,1) = M[2][1]; T(2,2) = M[2][2]; T(2,3) = M[2][3];
    T(3,0) = 0; T(3,1) = 0; T(3,2) = 0; T(3,3) = 1;
    //std::cout << "POSITION: \n" << T << endl;
    //wPe.print();
    robot.setRobotState(vpRobot::STATE_STOP);



    if (opt_plot && plotter != nullptr) {
      delete plotter;
      plotter = nullptr;
    }

    task.kill();

    if (!final_quit) {
      while (!final_quit) {
        rs.acquire(I);
        vpDisplay::display(I);

        vpDisplay::displayText(I, 20, 20, "Click to quit the program.", vpColor::red);
        vpDisplay::displayText(I, 40, 20, "Visual servo converged.", vpColor::red);

        //if (vpDisplay::getClick(I, false)) {
        //  final_quit = true;
        //}
        final_quit = true;

        vpDisplay::flush(I);
      }
    }

    if (traj_vip) {
      delete [] traj_vip;
    }

  } catch(const vpException &e) {
    std::cout << "ViSP exception: " << e.what() << std::endl;
    std::cout << "Stop the robot " << std::endl;
    robot.setRobotState(vpRobot::STATE_STOP);
    return EXIT_FAILURE;
  } catch(const franka::NetworkException &e) {
    std::cout << "Franka network exception: " << e.what() << std::endl;
    std::cout << "Check if you are connected to the Franka robot"
              << " or if you specified the right IP using --ip command line option set by default to 192.168.1.1. " << std::endl;
    return EXIT_FAILURE;
  } catch(const std::exception &e) {
    std::cout << "Franka exception: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

}



