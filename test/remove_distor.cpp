#include "abstract_camera.hpp"
#include "MEI_camera.hpp"
#include "pinhole_camera.hpp"
#include <chrono>
#include <string>
#include <algorithm>
#include <opencv2/ccalib/omnidir.hpp>

using namespace camera_model;

static std::mt19937_64 rd;
static std::uniform_real_distribution<double> distribution(0.0, std::nextafter(1, std::numeric_limits<double>::max()));

inline int Rand(int min, int max)
{ return (((double)distribution(rd) * (max - min + 1))) + min;}


void undistortOmniByOpencv(cv::Mat src, std::vector<cv::Point2f> input_point, 
                        const cv::Mat &K, const cv::Mat &D, const double xi, const int width, const int height)
{
    //! rectify point to undistort
    std::vector<cv::Point2f> pts_undist;
    
    cv::omnidir::undistortPoints(input_point, pts_undist, K, D, xi, cv::Mat());

    LOG(INFO) << "Undistorted Point by Opencv : " << std::endl;    
    for_each(pts_undist.begin(), pts_undist.end(), [&](std::vector<cv::Point2f>::iterator it){ 
        LOG(INFO) << "(" << (*it).x << ", "<<(*it).y << ")";
    })
    LOG(INFO) << endl;

    
    //! rectify image to undistort
    cv::Mat dst(src.size(), src.type());
    cv::Mat map1, map2;

    cv::omnidir::initUndistortRectifyMap(K, D, xi, cv::Mat(), cv::Mat(), cv::Size(width, height), CV_16SC2, map1, map2);
    cv::remap(src, dst, map1, map2, cv::INTER_LINEAR);
    cv::imshow("UndistByOpencv", dst);
}


int main(int argc, char *argv[])
{
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_prefix = true;
    FLAGS_log_dir = std::string(getcwd(NULL,0))+"/../log";
    google::InitGoogleLogging(argv[0]);

    std::string calib_file = "", img_file = ""; // path
    LOG_ASSERT(!calib_file.empty()) << "Empty Calibration file input !!!";
    LOG_ASSERT(!img_file.empty()) << "Empty Image file input !!!";

    AbstractCamera::Ptr camera;

    AbstractCamera::Model model = AbstractCamera::checkCameraModel(calib_file);
    if(AbstractCamera::Model::PINHOLE == model)
    {
        PinholeCamera::Ptr pinhole_camera = PinholeCamera::create(calib_file);
        // 运行时期多态, 虚函数会根据指针来决定运行哪个
        camera = std::static_pointer_cast<AbstractCamera>(pinhole_camera); 
    }
    else if(AbstractCamera::Model::MEI == model)
    {
        MEICamera::Ptr mei_camera = MEICamera::create(calib_file);
        camera = std::static_pointer_cast<AbstractCamera>(mei_camera);
    }

    cv::Mat src = cv::imread(img_file, CV_8UC1);
    cv::Mat dst;
    std::vector<cv::Point2f> pts_dist, pts_undist;

    // test undistort points with 10 random pixels on image
    for(int i=0; i<10; i++)
    {
       pts_dist.push_back(cv::Point2i(Rand(10, camera->width()-10), Rand(10, camera->height()-10) ) );        
    }

    camera->undistortPoints(pts_dist, pts_undist);
    camera->undistortMat(src, dst);

    cv::imshow("UndistByMine", dst);
}
