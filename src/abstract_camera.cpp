#include "abstract_camera.hpp"

namespace camera_model
{

//@ build
AbstractCamera::AbstractCamera() :
        model_(ABSTRACT) {}

AbstractCamera::AbstractCamera(Model model) :
        model_(model) {}

AbstractCamera::AbstractCamera(int width, int height, Model type) :
        model_(type), width_(width), height_(height) {}

AbstractCamera::AbstractCamera(int width, int height, double fx, double fy, double cx, double cy, Model model) :
        model_(model), width_(width), height_(height), fx_(fx), fy_(fy), cx_(cx), cy_(cy)
{
    K_ = cv::Mat::eye(3,3,CV_64FC1);
    K_.at<double>(0,0) = fx_;
    K_.at<double>(0,2) = cx_;
    K_.at<double>(1,1) = fy_;
    K_.at<double>(1,2) = cy_;
}

//@ get camera model
AbstractCamera::Model AbstractCamera::checkCameraModel(std::string calib_file)
{
    cv::FileStorage fs(calib_file.c_str(), cv::FileStorage::READ);
    LOG_ASSERT(fs.isOpened()) << "Failed to open calibration file at: " << calib_file;

    std::string str_camera_model;
    if (!fs["Camera.model"].empty())
        fs["Camera.model"] >> str_camera_model;

    fs.release();

    if(str_camera_model == "pinhole")
        return PINHOLE;
    else if(str_camera_model == "mei")
        return MEI;
    else
        return UNKNOW;
}

//@ from pixel coordiante to unit plane
Vector3d AbstractCamera::lift(const Vector2d& px) const
{
    LOG(FATAL) << "Please instantiation!!!";
}

Vector3d AbstractCamera::lift(double x, double y) const
{
    LOG(FATAL) << "Please instantiation!!!";
}

//@ from camera coordiante to pixel coordiante
Vector2d AbstractCamera::project(const Vector3d& xyz) const
{
    LOG(FATAL) << "Please instantiation!!!";
}

//@ from unit plane to pixel coordiante
Vector2d AbstractCamera::project(double x, double y, double z) const
{
    LOG(FATAL) << "Please instantiation!!!";
}

//@ remove distortion in points
void AbstractCamera::undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const
{
    LOG(FATAL) << "Please instantiation!!!";
}

//@ remove distortion on images
void AbstractCamera::undistortMat(const cv::Mat &img_dist, cv::Mat &img_udist) const
{
    LOG(FATAL) << "Please instantiation!!!";
}

inline const double AbstractCamera::xi() const
{
    LOG(FATAL) << "Please instantiation!!!";
}
}