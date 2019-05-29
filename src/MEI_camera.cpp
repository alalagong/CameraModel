#include "MEI_camera.hpp"
#include <math.h>

namespace camera_model
{
MEICamera::MEICamera(int width, int height, double xi, double fx, double fy, double cx, double cy,
            double k1, double k2, double p1, double p2):
            AbstractCamera(width, height, fx, fy, cx, cy, MEI),
            xi_(xi), k1_(k1), k2_(k2), p1_(p1), p2_(p2)
{
    distortion_ = (fabs(k1_) > 0.0000001);
    D_ = cv::Mat::zeros(1,4,CV_64FC1);
    D_.at<double>(0) = k1;
    D_.at<double>(1) = k2;
    D_.at<double>(2) = p1;
    D_.at<double>(3) = p2;
}
MEICamera::MEICamera(int width, int height, double xi, const cv::Mat& K, const cv::Mat& D):
            AbstractCamera(width, height, MEI), xi_(xi)
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

MEICamera::MEICamera(std::string calib_file) : 
            AbstractCamera(MEI)
{
    cv::FileStorage fs(calib_file.c_str(), cv::FileStorage::READ);
    LOG_ASSERT(fs.isOpened()) << "Failed to open calibration file at: " << calib_file;
    
    //* model
    std::string str_camera_model;
    if (!fs["Camera.model"].empty())
        fs["Camera.model"] >> str_camera_model;

    LOG_ASSERT(str_camera_model == "mei") << "Wrong camera modle: " << str_camera_model;

    //* resolution 分辨率
    cv::FileNode resolution = fs["Camera.resolution"];
    LOG_ASSERT(resolution.size() == 2) << "Failed to load Camera.resolution with error size: " << resolution.size();
    width_ = resolution[0];
    height_ = resolution[1];

    //* intrinsics 内参
    cv::FileNode intrinsics = fs["Camera.intrinsics"];
    LOG_ASSERT(intrinsics.size() == 5) << "Failed to load Camera.intrinsics with error size: " << intrinsics.size();
    xi_ = intrinsics[0];
    fx_ = intrinsics[1];
    fy_ = intrinsics[2];
    cx_ = intrinsics[3];
    cy_ = intrinsics[4];
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

// 像素坐标转单位球面（unit sphere）
Eigen::Vector3d MEICamera::lift(double x, double y) const
{
    Eigen::Vector3d xyz(0, 0, 1);
    if(distortion_)
    {
        std::vector<cv::Point2f> pt_d, pt_u;
        pt_d.push_back(cv::Point2f(x,y));
        undistortPoints(pt_d, pt_u); 
        assert(pt_u.size() == 1);
        xyz[0] = (pt_u[0].x - cx_) / fx_;
        xyz[1] = (pt_u[0].y - cy_) / fy_;
    } 
    else
    {
        xyz[0] = (x - cx_) / fx_;
        xyz[1] = (y - cy_) / fy_;
    }

    double x_u = xyz[0], y_u = xyz[1], lambda;
    if(xi_ == 1)
    {
        lambda = 2.0 / (x_u * x_u + y_u * y_u + 1.0);
        xyz << lambda*x_u, lambda*y_u, lambda - 1.0;
    }
    else
    {
        double r_2 = x_u*x_u + y_u*y_u;
        lambda = (xi_ + sqrt(1.0 + (1.0 - xi_*xi_)*(r_2)))/(r_2 + 1.0);
        xyz << lambda*x_u, lambda*y_u, lambda - xi_;
    }

    return xyz;
}

Eigen::Vector3d MEICamera::lift(const Eigen::Vector2d &px) const
{
    return lift(px[0], px[1]);
}

Eigen::Vector2d MEICamera::project(double x, double y, double z) const
{
    Eigen::Vector3d X(x, y, z);
    X.normalize();
    double x_u, y_u;
    x_u = X[0]/(X[2] + xi_);
    y_u = X[1]/(X[2] + xi_);    

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

Eigen::Vector2d MEICamera::project(const Eigen::Vector3d &xyz) const
{
    return project(xyz[0], xyz[1], xyz[2]);
}

//! 感觉去畸变的过程和xi那些操作无关的, 不需要转到Xs, 
// 迭代思想
void MEICamera::undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const
{
    if(0)
        cv::omnidir::undistortPoints(pts_dist, pts_udist, K_, D_, xi_, cv::Mat()); // 和omnidir的一样
    else
    {   
        for(auto it : pts_dist)
        {
            double x_dist, y_dist, r2, r4, a1, a2, a3, cdest_inv, deltaX, deltaY;
            double x_corr, y_corr;
            x_dist = (it.x - cx()) / fx();
            y_dist = (it.y - cy()) / fy();

            x_corr = x_dist;
            y_corr = y_dist;

            for(int i=0; i<10; i++)
            {
                r2 =x_corr*x_corr + y_corr*y_corr;
                r4 = r2 * r2;

                cdest_inv = 1 / (1.f + k1_*r2 + k2_*r4);
                a1 = 2.f * x_corr * y_corr;
                a2 = r2 + 2 * x_corr * x_corr;
                a3 = r2 + 2 * y_corr * y_corr;

                deltaX = p1_ * a1 + p2_ * a2;
                deltaY = p1_ * a3 + p2_ * a1;

                x_corr = (x_dist - deltaX) * cdest_inv;
                y_corr = (y_dist - deltaY) * cdest_inv;

            }

            double x_undist = x_corr; // * fx() + cx();
            double y_undist = y_corr; // * fy() + cy();

            pts_udist.push_back(cv::Point2f(x_undist, y_undist));

        }
    }
    assert(pts_dist.size() == pts_udist.size());
}

//TODO, difference between remap or not
void MEICamera::undistortMat(const cv::Mat &img_dist, cv::Mat &img_undist ) const
{
    cv::Size image_size(width_, height_);
    
    assert(img_dist.type() == CV_8UC1);
    img_undist = cv::Mat(height_, width_, img_dist.type());

    cv::Mat K_new = K();
    K_new.at<double>(0,0) = width()/4;
    K_new.at<double>(0,2) = width()/2;
    K_new.at<double>(1,1) = height()/4;
    K_new.at<double>(1,2) = height()/2;

    cv::Mat mapX = cv::Mat::zeros(image_size, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(image_size, CV_32F);

    for(int i=0; i<height_; i++)
    {
        for(int j=0; j<width_; j++)
        {
            cv::Point2f pt_undist, pt_dist;  // point on unit plane
            pt_undist.x = (j - K_new.at<double>(0,2)) / K_new.at<double>(0,0);
            pt_undist.y = (i - K_new.at<double>(1,2)) / K_new.at<double>(1,1);
            
            //! 通过这个可以理解了，这个射影矫正的过程，就是把原来透视球镜成像原理映射到 pinhole 上的一个过程
            //! 所以最开始的相机内参是要自己求得，原来的参数是omni模型的
            //! 而且相机成像为那种球状，不是因为畸变，而是本身的成像原理，因此开始无畸变的不需要xi。 
            
            //? 以下是错误的，留着引以为戒
            // double x_u = pt_undist.x, y_u = pt_undist.y;
            // double r_2 = x_u*x_u + y_u*y_u;
            // Eigen::Vector3d xyz;
            // double lambda = (xi_ + sqrt(1.0 + (1.0 - xi_*xi_)*(r_2)))/(r_2 + 1.0);
            // xyz << lambda*x_u, lambda*y_u, lambda - xi_;

            // xyz.normalize();
            // double Xs = xyz[0];
            // double Ys = xyz[1];
            // double Zs = xyz[2];
            
            // pt_undist 是世界坐标下的， 理解重点！
            double r = sqrt(pt_undist.x*pt_undist.x + pt_undist.y*pt_undist.y + 1);
            double Xs = pt_undist.x/r;
            double Ys = pt_undist.y/r;
            double Zs = 1/r;

            double x, y, r2, r4, a1, a2, a3, cdist, xd, yd;
            x = Xs/(Zs + xi_);
            y = Ys/(Zs + xi_);
            r2 = x*x + y*y;
            r4 = r2*r2;
            a1 = 2*x*y;
            a2 = r2 + 2*x*x;
            a3 = r2 + 2*y*y;
            cdist = 1 +D_.at<double>(0)*r2 + D_.at<double>(1)*r4;
            pt_dist.x = x*cdist + D_.at<double>(2)*a1 + D_.at<double>(3)*a2;
            pt_dist.y = y*cdist + D_.at<double>(2)*a3 + D_.at<double>(3)*a1;
            xd = pt_dist.x*fx_ + cx_;
            yd = pt_dist.y*fy_ + cy_;

            mapX.at<float>(j, i) = xd;
            mapY.at<float>(j, i) = yd;

#ifndef USE_REMAP
            Eigen::Vector2d pix_dist;
            pix_dist << xd, yd;
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
            }
            else
            {
                img_undist.at<uchar>(j, i) = 0;
            }
#endif            
            }
        }
        
#ifdef USE_REMAP
        std::cout<<"Done by remap !!!"<<std::endl;
        cv::Mat map1, map2;
        cv::convertMaps(mapX, mapY, map2, map1, CV_32FC1, false);
        cv::remap(img_dist, img_undist, map1, map2, cv::INTER_LINEAR);
#endif
}

} // end namespace 
