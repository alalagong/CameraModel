#ifndef _MEI_CAMERA_HPP_
#define _MEI_CAMERA_HPP_

#include "abstract_camera.hpp"
#include <opencv2/ccalib/omnidir.hpp>

// #define USE_REMAP

namespace camera_model
{
using namespace Eigen;
using namespace cv;

class MEICamera : public AbstractCamera
{
public:

    typedef std::shared_ptr<MEICamera> Ptr;

    virtual const double xi() const { return xi_; }; 

    virtual Vector3d lift(const Vector2d& px) const;

    virtual Vector3d lift(double x, double y) const;

    virtual Vector2d project(const Vector3d& xyz) const;

    virtual Vector2d project(double x, double y, double z=1) const;

    //! all undistort points are in the normlized plane
    virtual void undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const;

    virtual void undistortMat(const cv::Mat &img_dist, cv::Mat &img_udist) const;

    inline static MEICamera::Ptr create(int width, int height, double xi, double fx, double fy, double cx, double cy, double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0)
    {return MEICamera::Ptr(new MEICamera(width, height, xi, fx, fy, cx, cy, k1, k2, p1, p2));}

    inline static MEICamera::Ptr create(int width, int height, double xi, const cv::Mat& K, const cv::Mat& D)
    {return MEICamera::Ptr(new MEICamera(width, height, xi, K, D));}

    inline static MEICamera::Ptr create(std::string calib_file)
    {return MEICamera::Ptr(new MEICamera(calib_file));}

private:

    MEICamera(int width, int height, double xi, double fx, double fy, double cx, double cy,
                double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0);

    MEICamera(int width, int height, double xi, const cv::Mat& K, const cv::Mat& D);

    MEICamera(std::string calib_file);

private:

    double xi_;
    double k1_, k2_, p1_, p2_;

};



} // end namespace

#endif  // _MEI_CAMERA_HPP_