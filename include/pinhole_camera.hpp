#ifndef _PINHOLE_CAMERA_HPP_
#define _PINHOLE_CAMERA_HPP_
#include "abstract_camera.hpp"

namespace camera_model
{
using namespace Eigen;
using namespace cv;
      
class PinholeCamera : public AbstractCamera
{
public:

    typedef std::shared_ptr<PinholeCamera> Ptr;

    virtual Vector3d lift(const Vector2d& px) const;

    virtual Vector3d lift(double x, double y) const;

    virtual Vector2d project(const Vector3d& xyz) const;

    virtual Vector2d project(double x, double y, double z=1) const;

    //! all undistort points are in the normlized plane
    virtual void undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const;

    virtual void undistortMat(const cv::Mat &img_dist, cv::Mat &img_udist) const;

    inline static PinholeCamera::Ptr create(int width, int height, double fx, double fy, double cx, double cy, double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0)
    {return PinholeCamera::Ptr(new PinholeCamera(width, height, fx, fy, cx, cy, k1, k2, p1, p2));}

    inline static PinholeCamera::Ptr create(int width, int height, const cv::Mat& K, const cv::Mat& D)
    {return PinholeCamera::Ptr(new PinholeCamera(width, height, K, D));}

    inline static PinholeCamera::Ptr create(std::string calib_file)
    {return PinholeCamera::Ptr(new PinholeCamera(calib_file));}

private:

    PinholeCamera(int width, int height, double fx, double fy, double cx, double cy,
           double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0);

    PinholeCamera(int width, int height, const cv::Mat& K, const cv::Mat& D);

    PinholeCamera(std::string calib_file);

private:

    double k1_, k2_, p1_, p2_;

};

}
#endif  // _PINHOLE_CAMERA_HPP_