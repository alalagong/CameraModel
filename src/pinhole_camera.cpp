#include "pinhole_camera.hpp"
#include <math.h>

namespace camera_model
{

PinholeCamera::PinholeCamera(int width, int height, double fx, double fy, double cx, double cy,
       double k1, double k2, double p1, double p2) :
        AbstractCamera(width, height, fx, fy, cx, cy, PINHOLE),
        k1_(k1), k2_(k2), p1_(p1), p2_(p2)
{
    distortion_ = (fabs(k1_) > 0.0000001);
    D_ = cv::Mat::zeros(1,4,CV_64FC1);
    D_.at<double>(0) = k1;
    D_.at<double>(1) = k2;
    D_.at<double>(2) = p1;
    D_.at<double>(3) = p2;
}

PinholeCamera::PinholeCamera(int width, int height, const cv::Mat& K, const cv::Mat& D):
        AbstractCamera(width, height, PINHOLE)
{
    assert(K.cols == 3 && K.rows == 3);
    assert(D.cols == 4 || D.rows == 1);
    if(K.type() == CV_64FC1)
        K_ = K.clone();
    else
        K.convertTo(K_, CV_64FC1);

    if(D.type() == CV_64FC1)
        D_ = D.clone();
    else
        D.convertTo(D_, CV_64FC1);

    fx_ = K.at<double>(0,0);
    fy_ = K.at<double>(1,1);
    cx_ = K.at<double>(0,2);
    cy_ = K.at<double>(1,2);

    k1_ = D.at<double>(0);
    k2_ = D.at<double>(1);
    p1_ = D.at<double>(2);
    p2_ = D.at<double>(3);

    distortion_ = (fabs(k1_) > 0.0000001);
}

PinholeCamera::PinholeCamera(std::string calib_file) :
    AbstractCamera(PINHOLE)
{
    cv::FileStorage fs(calib_file.c_str(), cv::FileStorage::READ);
    LOG_ASSERT(fs.isOpened()) << "Failed to open calibration file at: " << calib_file;
    
    //* model
    std::string str_camera_model;
    if (!fs["Camera.model"].empty())
        fs["Camera.model"] >> str_camera_model;

    LOG_ASSERT(str_camera_model == "pinhole") << "Wrong camera modle: " << str_camera_model;

    //* resolution 分辨率
    cv::FileNode resolution = fs["Camera.resolution"];
    LOG_ASSERT(resolution.size() == 2) << "Failed to load Camera.resolution with error size: " << resolution.size();
    width_ = resolution[0];
    height_ = resolution[1];

    //* intrinsics 内参
    cv::FileNode intrinsics = fs["Camera.intrinsics"];
    LOG_ASSERT(intrinsics.size() == 4) << "Failed to load Camera.intrinsics with error size: " << intrinsics.size();
    fx_ = intrinsics[0];
    fy_ = intrinsics[1];
    cx_ = intrinsics[2];
    cy_ = intrinsics[3];
    K_ = cv::Mat::eye(3,3,CV_64FC1);
    K_.at<double>(0,0) = fx_;
    K_.at<double>(0,2) = cx_;
    K_.at<double>(1,1) = fy_;
    K_.at<double>(1,2) = cy_;

    //* distortion_coefficients 畸变系数
    cv::FileNode distortion_coefficients = fs["Camera.distortion_coefficients"];
    LOG_ASSERT(distortion_coefficients.size() == 4) << "Failed to load Camera.distortion_coefficients with error size: " << distortion_coefficients.size();
    k1_ = distortion_coefficients[0];
    k2_ = distortion_coefficients[1];
    p1_ = distortion_coefficients[2];
    p2_ = distortion_coefficients[3];

    D_ = cv::Mat::zeros(1,4,CV_64F);
    D_.at<double>(0) = k1_;
    D_.at<double>(1) = k2_;
    D_.at<double>(2) = p1_;
    D_.at<double>(3) = p2_;

    distortion_ = (fabs(k1_) > 0.0000001);

    fs.release();
}

// 像素坐标转单位平面
Vector3d PinholeCamera::lift(double x, double y) const
{
    Vector3d xyz(0, 0, 1);
    if(distortion_)
    {
        double p[2] = {x, y};
        cv::Mat pt_d = cv::Mat(1, 1, CV_64FC2, p);
        cv::Mat pt_u = cv::Mat(1, 1, CV_64FC2, xyz.data());
        // 函数要求两通道来表示点
        cv::undistortPoints(pt_d, pt_u, K_, D_); 
    }
    else
    {
        xyz[0] = (x - cx_) / fx_;
        xyz[1] = (y - cy_) / fy_;
    }

    return xyz;
}


Vector3d PinholeCamera::lift(const Vector2d &px) const
{
    return lift(px[0], px[1]);
}

// 单位平面转像素坐标
Vector2d PinholeCamera::project(double x, double y, double z) const
{
    double x_u = x/z, y_u = y/z; 
    Vector2d px(x_u, y_u);
    if(distortion_)
    {
        const double x2 = x_u * x_u;
        const double y2 = y_u * y_u;
        const double r2 = x2 + y2;
        const double rdist = 1 + r2 * (k1_ + k2_ * r2);
        const double a1 = 2 * x_u * y_u;
        const double a2 = r2 + 2 * x_u * x_u;
        const double a3 = r2 + 2 * y_u * y_u;

        px[0] = x_u * rdist + p1_ * a1 + p2_ * a2;
        px[1] = y_u * rdist + p1_ * a3 + p2_ * a1;
    }

    px[0] = fx_ * px[0] + cx_;
    px[1] = fy_ * px[1] + cy_;
    return px;
}

Vector2d PinholeCamera::project(const Vector3d &xyz) const
{
    return project(xyz[0], xyz[1], xyz[2]);
}

void PinholeCamera::undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const
{
    cv::undistortPoints(pts_dist, pts_udist, K_, D_);
}

void PinholeCamera::undistortMat(const cv::Mat &img_dist, cv::Mat &img_undist) const
{
    if(0)  // use opencv
    { 
        cv::Mat map1, map2;
        cv::initUndistortRectifyMap(K_, D_, cv::Mat(), cv::Mat(), cv::Size(width_, height_), CV_16SC2, map1, map2);
        // cv::initUndistortRectifyMap(K_, D_, Mat(), 
        //             cv::getOptimalNewCameraMatrix(K_, D_, cv::Size(width_, height_), 1, cv::Size(width_, height_), 0),
        //             cv::Size(width_, height_), CV_16SC2, map1, map2);
        cv::remap(img_dist, img_undist, map1, map2, cv::INTER_LINEAR);   
    }
    else
    {
        assert(img_dist.type() == CV_8UC1);
        img_undist = cv::Mat(height_, width_, img_dist.type());
        for(int i=0; i<height_; i++)
        {
            for(int j=0; j<width_; j++)
            {
                cv::Point2f pt_undist, pt_dist;  // point on unit plane
                pt_undist.x = (j - cx_) / fx_;
                pt_undist.y = (i - cy_) / fy_;

                double x, y, r2, r4, r6, a1, a2, a3, cdist, xd, yd;
                x = pt_undist.x;
                y = pt_undist.y;
                r2 = x*x + y*y;
                r4 = r2*r2;
                r6 = r4*r2;
                a1 = 2*x*y;
                a2 = r2 + 2*x*x;
                a3 = r2 + 2*y*y;
                cdist = 1 +D_.at<double>(0)*r2 + D_.at<double>(1)*r4;
                pt_dist.x = x*cdist + D_.at<double>(2)*a1 + D_.at<double>(3)*a2;
                pt_dist.y = y*cdist + D_.at<double>(2)*a3 + D_.at<double>(3)*a1;
                xd = pt_dist.x*fx_ + cx_;
                yd = pt_dist.y*fy_ + cy_;

                Eigen::Vector2d pix_dist(xd, yd);
                if(isInFrame(pix_dist, 1))
                {
                    // 双线性插值
                    int xi, yi;
                    float dx, dy;
                    xi = floor(xd);
                    yi = floor(yd);
                    dx = xd - xi;
                    dy = yd - yi;

                    img_undist.at<uchar>(j, i) = ( (1-dx)*(1-dy)*img_dist.at<uchar>(xi, yi) + 
                                                 dx*(1-dy)*img_dist.at<uchar>(xi+1, yi) + 
                                                 (1-dx)*dy*img_dist.at<uchar>(xi, yi+1) +
                                                 dx*dy*img_dist.at<uchar>(xi+1, yi+1) );
                }else
                {
                    img_undist.at<uchar>(j, i) = 0;
                }
                
            }
        }    
    }
    
}


} // end namespace