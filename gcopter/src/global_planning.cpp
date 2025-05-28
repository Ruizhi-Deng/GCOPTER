#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/sfc_gen.hpp"
#include "gcopter/trajectory.hpp"
#include "gcopter/voxel_map.hpp"
#include "misc/visualizer.hpp"

#include <controller_msgs/FlatTarget.h>
#include <gcopter/getTraj.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

struct Config {
    std::string mapTopic;
    std::string targetTopic;
    double dilateRadius;
    double voxelWidth;
    std::vector<double> mapBound;
    double timeoutRRT;
    double maxVelMag;
    double maxBdrMag;
    double maxTiltAngle;
    double minThrust;
    double maxThrust;
    double vehicleMass;
    double gravAcc;
    double horizDrag;
    double vertDrag;
    double parasDrag;
    double speedEps;
    double weightT;
    std::vector<double> chiVec;
    double smoothingEps;
    int integralIntervs;
    double relCostTol;
    int reference_type;

    Config(const ros::NodeHandle& nh_priv) {
        nh_priv.getParam("MapTopic", mapTopic);
        nh_priv.getParam("TargetTopic", targetTopic);
        nh_priv.getParam("DilateRadius", dilateRadius);
        nh_priv.getParam("VoxelWidth", voxelWidth);
        nh_priv.getParam("MapBound", mapBound);
        nh_priv.getParam("TimeoutRRT", timeoutRRT);
        nh_priv.getParam("MaxVelMag", maxVelMag);
        nh_priv.getParam("MaxBdrMag", maxBdrMag);
        nh_priv.getParam("MaxTiltAngle", maxTiltAngle);
        nh_priv.getParam("MinThrust", minThrust);
        nh_priv.getParam("MaxThrust", maxThrust);
        nh_priv.getParam("VehicleMass", vehicleMass);
        nh_priv.getParam("GravAcc", gravAcc);
        nh_priv.getParam("HorizDrag", horizDrag);
        nh_priv.getParam("VertDrag", vertDrag);
        nh_priv.getParam("ParasDrag", parasDrag);
        nh_priv.getParam("SpeedEps", speedEps);
        nh_priv.getParam("WeightT", weightT);
        nh_priv.getParam("ChiVec", chiVec);
        nh_priv.getParam("SmoothingEps", smoothingEps);
        nh_priv.getParam("IntegralIntervs", integralIntervs);
        nh_priv.getParam("RelCostTol", relCostTol);
        nh_priv.getParam("Reference_type", reference_type);
    }
};

class GlobalPlanner {
  private:
    Config config;

    ros::NodeHandle nh;
    ros::Subscriber mapSub;
    ros::Subscriber targetSub;
    ros::Publisher flatTargetPublisher;
    ros::ServiceServer trajTrigger;

    bool mapInitialized;
    voxel_map::VoxelMap voxelMap;
    Visualizer visualizer;
    std::vector<Eigen::Vector3d> startGoal;

    Trajectory<5> traj;
    double trajStamp;

  public:
    GlobalPlanner(const Config& conf, ros::NodeHandle& nh_)
        : config(conf), nh(nh_), mapInitialized(false), visualizer(nh) {
        const Eigen::Vector3i xyz(
            (config.mapBound[1] - config.mapBound[0]) / config.voxelWidth,
            (config.mapBound[3] - config.mapBound[2]) / config.voxelWidth,
            (config.mapBound[5] - config.mapBound[4]) / config.voxelWidth);

        const Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2],
                                     config.mapBound[4]);

        voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

        mapSub = nh.subscribe(config.mapTopic, 1, &GlobalPlanner::mapCallBack, this,
                              ros::TransportHints().tcpNoDelay());

        targetSub = nh.subscribe(config.targetTopic, 1, &GlobalPlanner::targetCallBack,
                                 this, ros::TransportHints().tcpNoDelay());

        flatTargetPublisher =
            nh.advertise<controller_msgs::FlatTarget>("reference/flatsetpoint", 1);

        trajTrigger =
            nh.advertiseService("get_traj", &GlobalPlanner::getTrajectoryCallBack, this);
    }

    inline void mapCallBack(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        if (!mapInitialized) {
            size_t cur = 0;
            const size_t total = msg->data.size() / msg->point_step;
            float* fdata = (float*)(&msg->data[0]);
            for (size_t i = 0; i < total; i++) {
                cur = msg->point_step / sizeof(float) * i;

                if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
                    std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
                    std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2])) {
                    continue;
                }
                voxelMap.setOccupied(
                    Eigen::Vector3d(fdata[cur + 0], fdata[cur + 1], fdata[cur + 2]));
            }

            voxelMap.dilate(std::ceil(config.dilateRadius / voxelMap.getScale()));

            mapInitialized = true;
        }
    }

    inline void plan() {
        if (startGoal.size() == 2) {
            std::vector<Eigen::Vector3d> route;
            sfc_gen::planPath<voxel_map::VoxelMap>(
                startGoal[0], startGoal[1], voxelMap.getOrigin(), voxelMap.getCorner(),
                &voxelMap, 0.01, route);
            std::vector<Eigen::MatrixX4d> hPolys;
            std::vector<Eigen::Vector3d> pc;
            voxelMap.getSurf(pc);

            sfc_gen::convexCover(route, pc, voxelMap.getOrigin(), voxelMap.getCorner(),
                                 7.0, 3.0, hPolys);
            sfc_gen::shortCut(hPolys);

            if (route.size() > 1) {
                visualizer.visualizePolytope(hPolys);

                Eigen::Matrix3d iniState;
                Eigen::Matrix3d finState;
                iniState << route.front(), Eigen::Vector3d::Zero(),
                    Eigen::Vector3d::Zero();
                finState << route.back(), Eigen::Vector3d::Zero(),
                    Eigen::Vector3d::Zero();

                gcopter::GCOPTER_PolytopeSFC gcopter;

                // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
                // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight,
                // thrust_weight]^T physicalParams = [vehicle_mass,
                // gravitational_acceleration, horitonral_drag_coeff,
                //                   vertical_drag_coeff, parasitic_drag_coeff,
                //                   speed_smooth_factor]^T
                // initialize some constraint parameters
                Eigen::VectorXd magnitudeBounds(5);
                Eigen::VectorXd penaltyWeights(5);
                Eigen::VectorXd physicalParams(6);
                // ROS_INFO("Max vel: %.2f", config.maxVelMag);
                magnitudeBounds(0) = config.maxVelMag;
                magnitudeBounds(1) = config.maxBdrMag;
                magnitudeBounds(2) = config.maxTiltAngle;
                magnitudeBounds(3) = config.minThrust;
                magnitudeBounds(4) = config.maxThrust;
                penaltyWeights(0) = (config.chiVec)[0];
                penaltyWeights(1) = (config.chiVec)[1];
                penaltyWeights(2) = (config.chiVec)[2];
                penaltyWeights(3) = (config.chiVec)[3];
                penaltyWeights(4) = (config.chiVec)[4];
                physicalParams(0) = config.vehicleMass;
                physicalParams(1) = config.gravAcc;
                physicalParams(2) = config.horizDrag;
                physicalParams(3) = config.vertDrag;
                physicalParams(4) = config.parasDrag;
                physicalParams(5) = config.speedEps;
                const int quadratureRes = config.integralIntervs;

                traj.clear();

                if (!gcopter.setup(config.weightT, iniState, finState, hPolys, INFINITY,
                                   config.smoothingEps, quadratureRes, magnitudeBounds,
                                   penaltyWeights, physicalParams)) {
                    return;
                }

                if (std::isinf(gcopter.optimize(traj, config.relCostTol))) {
                    return;
                }

                if (traj.getPieceNum() > 0) {
                    trajStamp = ros::Time::now().toSec();
                    visualizer.visualize(traj, route);
                    report = true;
                }
            }
        }
    }

    bool report = true;

    inline void targetCallBack(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        if (mapInitialized) {
            if (startGoal.size() >= 2) {
                startGoal.clear();
            }
            const double zGoal =
                config.mapBound[4] + config.dilateRadius +
                fabs(msg->pose.orientation.z) *
                    (config.mapBound[5] - config.mapBound[4] - 2 * config.dilateRadius);
            const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
            ROS_INFO("Point %.2f %.2f %.2f", goal(0), goal(1), goal(2));
            if (voxelMap.query(goal) == 0) {
                visualizer.visualizeStartGoal(goal, 0.5, startGoal.size());
                startGoal.emplace_back(goal);
            } else {
                ROS_WARN("Infeasible Position Selected !!!\n");
                return;
            }

            plan();
        }
        return;
    }

    /// @brief Publish the flatness target
    inline void process() {
        Eigen::VectorXd physicalParams(6);
        physicalParams(0) = config.vehicleMass;
        physicalParams(1) = config.gravAcc;
        physicalParams(2) = config.horizDrag;
        physicalParams(3) = config.vertDrag;
        physicalParams(4) = config.parasDrag;
        physicalParams(5) = config.speedEps;

        flatness::FlatnessMap flatmap;
        flatmap.reset(physicalParams(0), physicalParams(1), physicalParams(2),
                      physicalParams(3), physicalParams(4), physicalParams(5));

        if (traj.getPieceNum() > 0) {
            const double delta = ros::Time::now().toSec() - trajStamp;
            if (report) {
                ROS_INFO("Final Dest: %.2f %.2f %.2f",
                         traj.getPos(traj.getTotalDuration())(0),
                         traj.getPos(traj.getTotalDuration())(1),
                         traj.getPos(traj.getTotalDuration())(2));
                report = false;
            }
            if (delta > 0.0 && delta < traj.getTotalDuration()) {
                double thr;
                Eigen::Vector4d quat;
                Eigen::Vector3d omg;

                flatmap.forward(traj.getVel(delta), traj.getAcc(delta),
                                traj.getJer(delta), 0.0, 0.0, thr, quat, omg);
                double speed = traj.getVel(delta).norm();
                double bodyratemag = omg.norm();
                double tiltangle =
                    acos(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)));
                std_msgs::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
                speedMsg.data = speed;
                thrMsg.data = thr;
                tiltMsg.data = tiltangle;
                bdrMsg.data = bodyratemag;
                visualizer.speedPub.publish(speedMsg);
                visualizer.thrPub.publish(thrMsg);
                visualizer.tiltPub.publish(tiltMsg);
                visualizer.bdrPub.publish(bdrMsg);

                visualizer.visualizeSphere(traj.getPos(delta), config.dilateRadius);

                // publish flat_target
                controller_msgs::FlatTarget flat_target;
                flat_target.header.stamp = ros::Time::now();
                flat_target.header.frame_id = "map";
                flat_target.type_mask = config.reference_type;

                flat_target.position.x = traj.getPos(delta)(0);
                flat_target.position.y = traj.getPos(delta)(1);
                flat_target.position.z = traj.getPos(delta)(2);

                flat_target.velocity.x = traj.getVel(delta)(0);
                flat_target.velocity.y = traj.getVel(delta)(1);
                flat_target.velocity.z = traj.getVel(delta)(2);

                flat_target.acceleration.x = traj.getAcc(delta)(0);
                flat_target.acceleration.y = traj.getAcc(delta)(1);
                flat_target.acceleration.z = traj.getAcc(delta)(2);

                flat_target.jerk.x = traj.getJer(delta)(0);
                flat_target.jerk.y = traj.getJer(delta)(1);
                flat_target.jerk.z = traj.getJer(delta)(2);

                flatTargetPublisher.publish(flat_target);
            }
        }
    }

    bool getTrajectoryCallBack(gcopter::getTraj::Request& req,
                               gcopter::getTraj::Response& res) {
        if (mapInitialized) {
            startGoal.clear();

            const Eigen::Vector3d start(req.start_point.x, req.start_point.y,
                                        req.start_point.z);
            const Eigen::Vector3d goal(req.goal_point.x, req.goal_point.y,
                                       req.goal_point.z);

            if (voxelMap.query(start) == 0) {
                visualizer.visualizeStartGoal(start, 0.5, startGoal.size());
                startGoal.emplace_back(start);
            } else {
                ROS_WARN("Infeasible Start Position: %.2f %.2f %.2f !!!\n", start(0),
                         start(1), start(2));
                res.success = false;
                return false;
            }
            if (voxelMap.query(goal) == 0) {
                visualizer.visualizeStartGoal(goal, 0.5, startGoal.size());
                startGoal.emplace_back(goal);
            } else {
                ROS_WARN("Infeasible Goal Position : %.2f %.2f %.2f !!!\n", goal(0),
                         goal(1), goal(2));
                res.success = false;
                return false;
            }

            ROS_INFO("Traj: (%.2f %.2f %.2f) -> (%.2f %.2f %.2f)\n", start(0), start(1),
                     start(2), goal(0), goal(1), goal(2));

            try {
                plan();
            } catch (const std::exception& e) {
                ROS_ERROR("Exception: %s\n", e.what());
                res.success = false;
                return false;
            }
            res.success = true;
            return true;
        } else {
            ROS_WARN("Map Not Initialized !!!\n");
            res.success = false;
            return false;
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "global_planning_node");
    ros::NodeHandle nh_;

    GlobalPlanner global_planner(Config(ros::NodeHandle("~")), nh_);

    ros::Rate lr(1000);
    while (ros::ok()) {
        global_planner.process();
        ros::spinOnce();
        lr.sleep();
    }

    return 0;
}
