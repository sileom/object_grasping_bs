#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <math.h>
#include <thread>

#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include "common.h"

#include "pc-utils.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <librealsense2/rs.hpp> 
#include <librealsense2/rsutil.h>    

#include <ctime>
#include <chrono>

using namespace std;
using namespace pcl;
using namespace cv;
using namespace Eigen;

int move_robot(std::string &robot_ip, Eigen::Vector3d final_point, Eigen::MatrixXd final_matrix, double ti, double tf, double vi, double vf, bool use_current_orientation);
Eigen::Vector3d computePolPos(double ti, double tc, std::array<double, 4> &coeff_pos_x, std::array<double, 4> &coeff_pos_y, std::array<double, 4> &coeff_pos_z);
Eigen::Vector3d computePolVel(double ti, double tc, std::array<double, 3> &coeff_vel_x, std::array<double, 3> &coeff_vel_y, std::array<double, 3> &coeff_vel_z);
void computeCoeffPos(double p_i, double p_f, double v_i, double v_f, double t_i, double t_f, std::array<double, 4> &coeff);
void computeCoeffVel(double p_i, double p_f, double v_i, double v_f, double t_i, double t_f, std::array<double, 3> &coeff);
Eigen::MatrixXd getR(std::array<double, 16> posa);
Eigen::VectorXd r2aa(Eigen::MatrixXd R);
Eigen::VectorXd abscissa(double tk, double tf, double ti);
Eigen::MatrixXd elemRotation(int num, double angolo);

int main(int argc, char** argv) {
    std::string robot_ip; 
    int object_type = 1; 
    std::string modelFilepath; 
    std::string opt_eMc_filename;
    bool opt_adaptive_gain = false;
    bool opt_verbose = false;
    bool opt_plot = false;
    bool opt_task_sequencing = false;
    double tMove = 10;
    double opt_tagSize = 0.017;
    int opt_quad_decimate = 2;
    bool display_tag = true;
    double convergence_th_t = 0.01; 
    double convergence_th_tu = 2; //degree


    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--ip" && i + 1 < argc) {
            robot_ip = std::string(argv[i + 1]);
        }
        else if (std::string(argv[i]) == "--inputModelPath" && i + 1 < argc) {
            modelFilepath = std::string(argv[i + 1]);
        }
        else if (std::string(argv[i]) == "--objectType" && i + 1 < argc) {
            object_type = std::stoi(argv[i + 1]);
        }
        else if (std::string(argv[i]) == "--savePath" && i + 1 < argc) {
            savePath = std::string(argv[i + 1]);
        }
        else if (std::string(argv[i]) == "--tMove") {
            tMove = std::stod(argv[i + 1]);
        }
        else if (std::string(argv[i]) == "--tAss" && i + 1 < argc) {
            t_ass = std::stod(argv[i + 1]);
        }
        else if (std::string(argv[i]) == "--opt_eMc_filename" && i + 1 < argc) {
            opt_eMc_filename = std::string(argv[i + 1]);
        }
        else if (std::string(argv[i]) == "--opt_adaptive_gain") {
            opt_adaptive_gain = true;
        } 
        else if (std::string(argv[i]) == "--tag_size" && i + 1 < argc) {
            opt_tagSize = std::stod(argv[i + 1]);
        } 
        else if (std::string(argv[i]) == "--verbose") {
            opt_verbose = true;
        }
        else if (std::string(argv[i]) == "--plot") {
            opt_plot = true;
        }
        else if (std::string(argv[i]) == "--task_sequencing") {
            opt_task_sequencing = true;
        }
        else if (std::string(argv[i]) == "--quad_decimate" && i + 1 < argc) {
            opt_quad_decimate = std::stoi(argv[i + 1]);
        }
        else if (std::string(argv[i]) == "--convergence_th_t" && i + 1 < argc) {
            convergence_th_t = std::stod(argv[i + 1]);
        }
        else if (std::string(argv[i]) == "--convergence_th_tu" && i + 1 < argc) {
            convergence_th_tu = std::stod(argv[i + 1]);
        }
        else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout << argv[0] << " [--ip <ip_robot>] \n"
                                << "[--inputModelPath <input model file>] \n"
                                << "[--objectType <string represent object>] \n"
                                << "[--tMove <movement time>] \n"
                                << "[--tAss <settling time>] \n"
                                << "[--opt_eMc_filename <eMc filepath>] \n"
                                << "[--tag_size <tag size in m>] \n"
                                << "[--quad_decimate <decimation>] \n"
                                << "[--convergence_th_t <convergence threshold on t>] \n"
                                << "[--convergence_th_tu <convergence threshold on tu in degree>] \n"
                                << "[--adaptive_gain] [--verbose] [--plot] [--task_sequencing] [--help] [-h]" 
                                << "\n";
            return EXIT_SUCCESS;
        }
    }

    // Parameters to alignment and to filter
    param.p_tag[0] = 0.1820248736; param.p_tag[1] = -0.13314309; param.p_tag[2] = 0.3962837345;
    param.R_tag(0,0) = 0.9999219085; param.R_tag(0,1) = -0.0123218041; param.R_tag(0,2) = 0.002085679898;
    param.R_tag(1,0) = -0.01232827789; param.R_tag(1,1) = -0.9999191361; param.R_tag(1,2) = 0.003120061659;
    param.R_tag(2,0) = 0.002047066454; param.R_tag(2,1) = -0.00314553085; param.R_tag(2,2) = -0.9999929576;

    param.x_min = -0.175; param.x_max = 0.168;
    param.y_min = -0.15;  param.y_max = 0.11;
    param.z_min = 0.05;    param.z_max = 0.47;

    param.xc_min = 0.005;  param.xc_max = 0.18;
    param.yc_min = 0.005;  param.yc_max = 0.15;
    param.th_min = 520; //105;

    param.leaf_size = 0.0; //for voxel
    param.cluster_tolerance = 0.02;

    // Alignment phase
    Eigen::MatrixXd T_state;
    T_state.setIdentity(4,4);    
    alignTag(T_state, robot_ip, opt_eMc_filename, opt_adaptive_gain, opt_tagSize, display_tag, opt_verbose, opt_plot, 
      opt_task_sequencing, convergence_th_t, convergence_th_tu, opt_quad_decimate);
    std::this_thread::sleep_for(1s);


    // Detection phase
    cout << "Start detection" << endl;
    Eigen::MatrixXd cMo_from_func;
    Eigen::MatrixXd cMo_f;
    cMo_from_func.setIdentity(4,4);
    cMo_f.setIdentity(4,4);

    cMo_f = detectObjectBSP(modelFilepath);


    Eigen::MatrixXd wMe;
    wMe.setIdentity(4,4);
    wMe.block<3,3>(0,0) = T_state.block<3,3>(0,0);
    wMe(0,3) = T_state(0,3);
    wMe(1,3) = T_state(1,3);
    wMe(2,3) = T_state(2,3)

    Eigen::MatrixXd eMc;
    eMc.setIdentity(4,4);
    eMc << 0.01999945272, -0.9990596861, 0.03846772089, 0.05834485203,
          0.9997803621, 0.01974315714, -0.007031030191, -0.03476564525,
          0.006264944557, 0.03859988867, 0.999235107, -0.06760482074,
          0, 0, 0, 1;

    Eigen::MatrixXd wMc;
    wMc = wMe * eMc;

    Eigen::MatrixXd wMo;
    wMo = wMe * eMc * cMo_f;

    // Grasping phase
    Eigen::MatrixXd R_angle(3,3);
    Eigen::MatrixXd R_anglef(3,3);
    Eigen::MatrixXd R_1_angle(3,3);
    Eigen::MatrixXd R_1_anglef(3,3);

    Eigen::MatrixXd R_curr(3,3);
    franka::Robot robot(robot_ip);
    franka::RobotState robot_state = robot.readOnce();
    R_curr = getR(robot_state.O_T_EE_d);

    R_anglef = elemRotation(ROTZ, (M_PI/2)) * wMo.block<3,3>(0,0);
    R_angle = R_anglef * R_curr;

    R_1_anglef = elemRotation(ROTZ, -1*(M_PI/2)) * wMo.block<3,3>(0,0);
    R_1_angle = R_1_anglef * R_curr;

    float angle, angle_1;
    angle = atan2( R_angle(1,0),  R_angle(0,0) );
    angle_1 = atan2( R_1_angle(1,0),  R_1_angle(0,0) );
    
    Eigen::MatrixXd R_des_(3,3);
    if( abs(angle) < abs(angle_1) ){
      R_des_ = R_anglef;
    } else {
      R_des_ = R_1_anglef;
    }

    // Move to approach point
    move_robot(robot_ip,  Eigen::Vector3d(wMo(0,3), wMo(1,3)-0*0.005,wMo(2,3)+0.05), R_des_, 0.0, tMove, 0.0, 0.0, false);
    // Move to grasping point
    move_robot(robot_ip,  Eigen::Vector3d(wMo(0,3), wMo(1,3)-0*0.005,wMo(2,3)), MatrixXd::Identity(3,3), 0.0, tMove, 0.0, 0.0, true);


    float dim_grasp = 0.0;
    if(object_type == 4){
      dim_grasp = 0.0075; 
    } else if (object_type == 5){
      dim_grasp = 0.022; 
    }

    franka::Gripper gripper(robot_ip);
    franka::GripperState gripper_state=gripper.readOnce();
    if (!gripper.grasp(dim_grasp, 0.1, 60)){ 
      std::cout << "Failed to grasp the object." << std::endl;
      return -1;
    }

  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }
  return 0;
}

int move_robot(std::string &robot_ip, Eigen::Vector3d final_point, Eigen::MatrixXd final_matrix, double ti, double tf, double vi, double vf, bool use_current_orientation){
  try {
    franka::Robot robot(robot_ip);
    franka::Model model = robot.loadModel();
    franka::RobotState robot_state = robot.readOnce();
    setDefaultBehavior(robot);

    if (use_current_orientation){
      final_matrix=getR(robot_state.O_T_EE_d);
    }

    // Set additional parameters always before the control loop, NEVER in the control loop!
    // Set collision behavior.
    robot.setCollisionBehavior(
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});

    std::array<double, 16> initial_pose = robot_state.O_T_EE;
    std::array<double, 16> desider_pose = initial_pose;

    desider_pose[12] = final_point[0];     //X
    desider_pose[13] = final_point[1];     //Y
    desider_pose[14] = final_point[2];     //Z


    std::array<double, 4> coeff_pos_x{};
    std::array<double, 4> coeff_pos_y{};
    std::array<double, 4> coeff_pos_z{};
    std::array<double, 3> coeff_vel_x{};
    std::array<double, 3> coeff_vel_y{};
    std::array<double, 3> coeff_vel_z{};
    computeCoeffPos(initial_pose[12], desider_pose[12], 0.0, 0.0, ti, tf, coeff_pos_x);
    computeCoeffPos(initial_pose[13], desider_pose[13], 0.0, 0.0, ti, tf, coeff_pos_y);
    computeCoeffPos(initial_pose[14], desider_pose[14], 0.0, 0.0, ti, tf, coeff_pos_z);
    computeCoeffVel(initial_pose[12], desider_pose[12], 0.0, 0.0, ti, tf, coeff_vel_x);
    computeCoeffVel(initial_pose[13], desider_pose[13], 0.0, 0.0, ti, tf, coeff_vel_y);
    computeCoeffVel(initial_pose[14], desider_pose[14], 0.0, 0.0, ti, tf, coeff_vel_z);


    Eigen::MatrixXd R0 = getR(initial_pose);
    Eigen::MatrixXd Rf = final_matrix; 

    Eigen::MatrixXd Rf0 = R0.transpose()*Rf;
    Eigen::VectorXd r_theta = r2aa(Rf0);
    Eigen::Vector3d r(r_theta[0], r_theta[1], r_theta[2]);
    double theta_f = r_theta[3]; 

    Eigen::Vector3d p_des;
    Eigen::Vector3d v_des;
    Eigen::VectorXd s_ds;
    Eigen::MatrixXd R_d;
    Eigen::Vector3d omegad;

    double time = 0.0;

    robot.control([&](const franka::RobotState& robot_state, franka::Duration period) -> franka::CartesianVelocities { 
      time += period.toSec();

      std::array<double, 7> q = robot_state.q;
      std::array<double, 7> v = robot_state.dq;
      std::array<double, 7> acc = robot_state.ddq_d;
      std::array<double, 7> tau = robot_state.tau_J;
      std::array<double, 7> cor = model.coriolis(robot_state);

      if(time < (tf-ti) ) {
        p_des = computePolPos(ti, time, coeff_pos_x, coeff_pos_y, coeff_pos_z);
        v_des = computePolVel(ti, time, coeff_vel_x, coeff_vel_y, coeff_vel_z);
        s_ds = abscissa(time, tf, ti);
        omegad = R0*r;
        omegad = omegad * (s_ds[1]*theta_f);
      } else {
        p_des = computePolPos(ti, tf, coeff_pos_x, coeff_pos_y, coeff_pos_z);
        v_des = computePolVel(ti, tf, coeff_vel_x, coeff_vel_y, coeff_vel_z);
        s_ds = abscissa(tf, tf, ti);
        omegad = R0*r;
        omegad = omegad * (s_ds[1]*theta_f);
      }

      franka::CartesianVelocities output = {{v_des[0], v_des[1], v_des[2], omegad[0], omegad[1], omegad[2]}};

      if (time >= ((tf-ti)+t_ass)) {
        std::cout << std::endl << "Finished motion, shutting down example" << std::endl;
        return franka::MotionFinished(output);
      }
      return output;
    });

    std::cout << "Done." << std::endl;

  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }
  return 0;
}

Eigen::Vector3d computePolPos(double ti, double tc, std::array<double, 4> &coeff_pos_x, std::array<double, 4> &coeff_pos_y, std::array<double, 4> &coeff_pos_z){
  Eigen::Vector3d p;
  double t = tc - ti;
  p[0] = coeff_pos_x[0] + coeff_pos_x[1] * t + coeff_pos_x[2] * pow(t,2) + coeff_pos_x[3] * pow(t,3); 
  p[1] = coeff_pos_y[0] + coeff_pos_y[1] * t + coeff_pos_y[2] * pow(t,2) + coeff_pos_y[3] * pow(t,3); 
  p[2] = coeff_pos_z[0] + coeff_pos_z[1] * t + coeff_pos_z[2] * pow(t,2) + coeff_pos_z[3] * pow(t,3);  
  return p;
}

Eigen::Vector3d computePolVel(double ti, double tc, std::array<double, 3> &coeff_vel_x, std::array<double, 3> &coeff_vel_y, std::array<double, 3> &coeff_vel_z){
  Eigen::Vector3d v;
  double t = tc - ti;
  v[0] = coeff_vel_x[0] + 2*coeff_vel_x[1] * t + 3*coeff_vel_x[2] * pow(t,2);
  v[1] = coeff_vel_y[0] + 2*coeff_vel_y[1] * t + 3*coeff_vel_y[2] * pow(t,2);
  v[2] = coeff_vel_z[0] + 2*coeff_vel_z[1] * t + 3*coeff_vel_z[2] * pow(t,2);
  return v;
}

void computeCoeffPos(double p_i, double p_f, double v_i, double v_f, double t_i, double t_f, std::array<double, 4> &coeff){
  double T = t_f - t_i;
	coeff[0] = p_i;
  coeff[1] = v_i;
		
	double a2_num = -3*(p_i-p_f)-(2*v_i+v_f)*T;
	coeff[2] = a2_num/pow(T,2.0);

	double a3_num = 2*(p_i-p_f)+(v_i+v_f)*T;
	coeff[3] = a3_num/pow(T,3.0);

}

void computeCoeffVel(double p_i, double p_f, double v_i, double v_f, double t_i, double t_f, std::array<double, 3> &coeff){
  double T = t_f - t_i;
	coeff[0] = v_i;
	
	double a2_num = -3*(p_i-p_f)-(2*v_i+v_f)*T;
	coeff[1] = a2_num/pow(T,2.0);
	
	double a3_num = 2*(p_i-p_f)+(v_i+v_f)*T;
	coeff[2] = a3_num/pow(T,3.0);

}

Eigen::MatrixXd getR(std::array<double, 16> posa){
  Eigen::MatrixXd R(3,3);
  R(0,0) = posa[0];
  R(0,1) = posa[4];
  R(0,2) = posa[8];
  R(1,0)= posa[1];
  R(1,1)= posa[5];
  R(1,2)= posa[9];
  R(2,0) = posa[2];
  R(2,1) = posa[6];
  R(2,2) = posa[10];
  return R;
}

Eigen::VectorXd r2aa(Eigen::MatrixXd R){
   Eigen::Vector3d r0(0,0,0);
   double val = ((R(0, 0)+R(1,1) + R(2,2) - 1)*0.5) + 0.0;
   double theta = acos(std::min(std::max(val,-1.0),1.0));
   
   if(abs(theta-M_PI) <= 0.00001){
     r0[0] = -1*sqrt((R(0,0)+1) * 0.5); 
     r0[1] = sqrt((R(1,1)+1) * 0.5);
     r0[2] = sqrt(1-pow(r0(0),2)-pow(r0(1), 2));
   } else {
     if(theta >= 0.00001) {
      r0[0] = (R(2,1)-R(1, 2))/(2*sin(theta));
      r0[1] = (R(0,2)-R(2, 0))/(2*sin(theta));
      r0[2] = (R(1,0)-R(0, 1))/(2*sin(theta));
     }
   }
   Eigen::VectorXd result(4);
   //cout << "\n\n\n\nAsse angolo nel metodo\n"<< r0(0) << " " << r0(1) << " " << r0(2) << " " << theta << endl << endl << endl;
   result << r0(0), r0(1), r0(2), theta;
   return result;
}

Eigen::VectorXd abscissa(double tk, double tf, double ti){
   double t = (tk-ti)/(tf-ti);
   if(t > 1.0){
     t = 1.0;
   } 
   double s = pow(t, 3)*(6*pow(t,2)-15*t+10);
   double ds = (pow(t,2)*(30*pow(t,2)-60*t+30))/tf;
   Eigen::VectorXd result(2);
   result << s, ds;
   return result;

}

Eigen::MatrixXd elemRotation(int num, double angolo){
   Eigen::MatrixXd R(3,3);
   if(num == 1){
    R << 1, 0, 0, 0, cos(angolo), -sin(angolo), 0, sin(angolo), cos(angolo);
  }
  if(num == 2){
    R << cos(angolo), 0, sin(angolo), 0, 1, 0, -sin(angolo), 0, cos(angolo);
  }
  if(num == 3){
    R << cos(angolo), -sin(angolo), 0, sin(angolo), cos(angolo), 0, 0, 0, 1;
  }

  return R;
}
