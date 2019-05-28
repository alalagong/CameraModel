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
    // LOG(INFO) << "["<<fx_ <<", "<<fy_<<", "<<cx_<<", "<<cy_<<"]";

    //* distortion_coefficients 畸变系数
    cv::FileNode distortion_coefficients = fs["Camera.distortion_coefficients"];
    LOG_ASSERT(distortion_coefficients.size() == 4) << "Failed to load Camera.distortion_coefficients with error size: " << distortion_coefficients.size();
    k1_ = distortion_coefficients[0];
    k2_ = distortion_coefficients[1];
    p1_ = distortion_coefficients[2];
    p2_ = distortion_coefficients[3];
    // LOG(INFO) << "["<<k1_ <<", "<<k2_<<", "<<p1_<<", "<<p2_<<"]";

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

//@ 函数返回的点是在单位平面上的，不是在图像上的
void PinholeCamera::undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const
{
    if(0) // use opencv 
    {   
        cv::fisheye::undistortPoints(pts_dist, pts_udist, K_, D_); // 返回的不是图像上的， 是单位平面的。。。
        int i=0;
        for( auto it : pts_udist)
        {
            double pix_u = fx()*it.x + cx();
            double pix_v = fy()*it.y + cy();

            pts_udist[i].x = pix_u;
            pts_udist[i].y = pix_v;
            i++;
        }
    }
    else // 自己写的和opencv有点出入， opencv是迭代求解theta
    {
        for( auto it : pts_dist)
        {
            double x_dist, y_dist, r, theta, theta2, theta4, theta6, theta8, thetad, scaling_inv;
            double x_corr, y_corr;
            x_dist = (it.x - cx()) / fx();
            y_dist = (it.y - cy()) / fy();

            x_corr = x_dist;
            y_corr = x_dist;

            for(int i=0; i<10; i++)
            {
                r =sqrt(x_corr*x_corr + y_corr*y_corr);

                theta = atan(r);
                theta2 = theta*theta;
                theta4 = theta2*theta2;
                theta6 = theta2*theta4;
                theta8 = theta4*theta4;
                thetad = theta*(1.f + k1_*theta2 + k2_*theta4 + p1_*theta6 + p2_*theta8);
                scaling_inv = (r > 1e-8) ? r / thetad  : 1.0;

                x_corr = scaling_inv*x_dist;
                y_corr = scaling_inv*y_dist;
            }

            double x_undist = x_corr; // * fx() + cx();
            double y_undist = y_corr; // * fy() + cy();

            pts_udist.push_back(cv::Point2f(x_undist, y_undist));
        }
    }
    assert(pts_dist.size() == pts_udist.size());
}

void PinholeCamera::undistortMat(const cv::Mat &img_dist, cv::Mat &img_undist) const
{
    if(0)  // use opencv
    { 
        cv::Mat map1, map2, K_new;
        // cv::getOptimalNewCameraMatrix()
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K_, D_, cv::Size(width_, height_), cv::noArray(), K_new, 0.f);
        cv::fisheye::initUndistortRectifyMap(K_, D_, Mat(), K_new, cv::Size(width_, height_), CV_16SC2, map1, map2);
        cv::remap(img_dist, img_undist, map1, map2, cv::INTER_LINEAR);   
    }
    else
    { //! 以下是和 K_new = K_ 一样， 说明畸变导致了相机内参有变化， 需要求出来这个变化
        assert(img_dist.type() == CV_8UC1);
        img_undist = cv::Mat(height_, width_, img_dist.type());
        
        //* 重新计算出考虑畸变的相机内参
        int N = 3;
        std::vector<cv::Point2f> points_sample, points_sample_undist;
        for(int i=0; i<N; i++)
            for(int j=0; j<N; j++)
            {
                points_sample.push_back(cv::Point2f((float)j*width()/(N-1), (float)i*height()/(N-1)));                
            }        
        undistortPoints(points_sample, points_sample_undist);

#ifdef _DEBUG_MODE_
        cv::Mat K_new;
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K_, D_, cv::Size(width_, height_), cv::noArray(), K_new, 0.f);
        LOG(INFO) << "compute from opencv, new fx, fy, cx, cy" << K_new.at<double>(0,0) <<", "<< K_new.at<double>(1,1)<<", "<< K_new.at<double>(0,2)<<", "<< K_new.at<double>(1,2);
#endif

// 以下两种方法都可。。。

//! ******************************* 这是 Radtan 的计算方法（cv::）*******************************
        // 这里取最小的一圈， 也可以取最大的
        float u_0 = -FLT_MAX, u_1 = FLT_MAX, v_0 = -FLT_MAX, v_1 = FLT_MAX;

        for(int i=0; i<N; i++)
            for(int j=0; j<N; j++) 
            {
                cv::Point2f it = points_sample_undist[i*N + j];

                if(j == 0)
                    u_0 = MAX(u_0, it.x);
                if(j == N-1)
                    u_1 = MIN(u_1, it.x);
                if(i == 0)
                    v_0 = MAX(v_0, it.y);
                if(i == N-1)
                    v_1 = MIN(v_1, it.y);
            }

        double fx_new, fy_new, cx_new, cy_new;
        fx_new = width() / (u_1 - u_0);
        fy_new = height() / (v_1 - v_0);
        cx_new = -fx_new * (u_0) ;
        cy_new = -fy_new * (v_0) ;



//! **********************这是 EQUI 的计算方法（cv::fisheye)**********************
/* {        
        //计算均值
        double mean_x, mean_y;
        // assert(points_sample_undist.size() == 8);
        for(auto it : points_sample_undist)
        {
            mean_x += it.x;
            mean_y += it.y;
        }
        mean_x /= (N*N) ;
        mean_y /= (N*N) ;

#ifdef _DEBUG_MODE_
        LOG(INFO) << "mean_x " <<mean_x;
        LOG(INFO) << "points_sample_undist[0].x " <<points_sample_undist[0].x;
#endif  

        // 计算比值
        double aspect_ratio = fx_ / fy_;
        mean_y *= aspect_ratio;  //bug 和opencv不同

        for(int i=0; i<N*N; i++)
        {
            points_sample_undist[i].y *= aspect_ratio;
        }

        // 计算范围
        double minx = DBL_MAX, miny = DBL_MAX, maxx = -DBL_MAX, maxy = -DBL_MAX;
        for(int i=0; i<N; ++i)
            for(int j=0; j<N; ++j) 
            {
                if(j == 0 || j == N-1)
                    minx = std::min(minx, std::abs(points_sample_undist[i*N+j].x - mean_x)); 
                if(i == 0 || i == N-1)
                    miny = std::min(miny, std::abs(points_sample_undist[i*N+j].y - mean_y));
                maxx = std::max(maxx, std::abs(points_sample_undist[i*N+j].x - mean_x));
                maxy = std::max(maxy, std::abs(points_sample_undist[i*N+j].y - mean_y));
            }

        
#ifdef _DEBUG_MODE_
        LOG(INFO) << "minx " <<minx;
        LOG(INFO) << "miny " <<miny;
#endif  

        double f1 = width() * 0.5/(minx);
        // double f2 = width() * 0.5/(maxx);
        double f3 = height() * 0.5 * aspect_ratio/(miny);
        // double f4 = height() * 0.5 * aspect_ratio/(maxy);

        double f_max, fx_new, fy_new, cx_new, cy_new;
        f_max = std::max(f1, f3);
        fx_new = f_max;
        fy_new = f_max / aspect_ratio;
        cx_new = -mean_x * fx_new + width()*0.5;
        cy_new = -mean_y * fy_new + height()*0.5;

#ifdef _DEBUG_MODE_
        LOG(INFO) << "fx_new" << fx_new;
        LOG(INFO) << "cx_new" << cx_new;
#endif

} */

        for(int i=0; i<height_; i++)
        {
            for(int j=0; j<width_; j++)
            {
                cv::Point2f pt_undist, pt_dist;  // point on unit plane
                pt_undist.x = (j - cx_new) / fx_new;
                pt_undist.y = (i - cy_new) / fy_new;

                //! radtan
                // double x, y, r2, r4, r6, a1, a2, a3, cdist, xd, yd;
                // x = pt_undist.x;
                // y = pt_undist.y;
                // r2 = x*x + y*y;
                // r4 = r2*r2;
                // r6 = r4*r2;
                // a1 = 2*x*y;
                // a2 = r2 + 2*x*x;
                // a3 = r2 + 2*y*y;
                // cdist = 1 +D_.at<double>(0)*r2 + D_.at<double>(1)*r4;
                // pt_dist.x = x*cdist + D_.at<double>(2)*a1 + D_.at<double>(3)*a2;
                // pt_dist.y = y*cdist + D_.at<double>(2)*a3 + D_.at<double>(3)*a1;
                // xd = pt_dist.x*fx_ + cx_;
                // yd = pt_dist.y*fy_ + cy_;

                //! equi
                double x, y, r, xd, yd, theta, theta2, theta4, theta6, theta8, thetad, scaling;
                x = pt_undist.x;
                y = pt_undist.y;
                r = sqrt(x*x + y*y);
                theta = atan(r);
                theta2 = theta * theta;
                theta4 = theta2 * theta2;
                theta6 = theta4 * theta2;
                theta8 = theta4 * theta4;
                thetad = theta * (1 + D_.at<double>(0) * theta2 + D_.at<double>(1) * theta4 + D_.at<double>(2) * theta6 + D_.at<double>(3) * theta8);
                scaling = (r > 1e-8) ? thetad / r : 1.0;
                xd = fx_*x*scaling + cx_;
                yd = fy_*y*scaling + cy_;

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