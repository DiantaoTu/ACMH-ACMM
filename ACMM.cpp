#include "base.h"

// 用于debug的时候的输出控制 0会减少debug输出
#define DEBUG_ACMH 0
#define DEBUG_DERES 0

using namespace REC3D;

typedef vector<cv::Mat> WeightMap;
typedef vector<WeightMap> WeightMapArr;

namespace OPT
{
extern float fNCCThreshold;      // NCC阈值
extern unsigned nSizeHalfWindow; // NCC半窗口尺寸
// const unsigned nSizeHalfWindow = 3; // NCC窗口参数
extern unsigned nSizeStep;       // NCC计算步长
extern string sDataPath;
} // namespace OPT


// 通过单应矩阵H求对应点
cv::Point2f ProjectH(const float* H, const float* X)
{
	float invZ = 1 / (H[6] * X[0] + H[7] * X[1] + H[8]);
	return cv::Point2f((H[0] * X[0] + H[1] * X[1] + H[2])*invZ, (H[3] * X[0] + H[4] * X[1] + H[5])*invZ);

} // 通过单应矩阵H求对应点

cv::Mat DownSample(cv::Mat img)
{
    cv::Mat downSampled;
    cv::pyrDown(img, downSampled);
    return downSampled;
}

// 保存深度图，不一定是深度图只要是单通道浮点数组成的矩阵都可以
// 数字越小颜色越深
void SaveDepthMap(cv::Mat depthMap, const string fileName)
{
    float max = FLT_MIN, min = FLT_MAX;
    for (cv::MatIterator_<float> it = depthMap.begin<float>(); it != depthMap.end<float>(); it++)
    {
        if (*(it) > max)
            max = *(it);
        if (*(it) < min)
            min = *(it);
    }
    cv::Mat img = cv::Mat::zeros(depthMap.size(), CV_8U);
    float delta = max - min;
    cv::MatIterator_<float> depth_begin = depthMap.begin<float>();
    cv::MatIterator_<float> depth_end = depthMap.end<float>();
    cv::MatIterator_<uchar> img_begin = img.begin<uchar>();
    cv::MatIterator_<uchar> img_end = img.end<uchar>();
    while (depth_begin != depth_end)
    {
        // 0为黑色 255为白色 越近颜色越深
        uchar depth = (uchar)((*(depth_begin)-min) / delta * 255);
        *(img_begin) = depth;
        img_begin++;
        depth_begin++;
    }
    if (cv::imwrite(fileName, img))
        LOG(INFO) << "Save depth map to " << fileName;
    else
        LOG(INFO) << "Fail to save depth map to " << fileName;
}

// 联合双边上采样
// high - 高分辨率的彩色图  low - 低分辨率的彩色图   depthMap - 低分辨率下的深度图或法向图
cv::Mat JointBilateralUpsample(cv::Mat high, cv::Mat low, cv::Mat coarseMap, const int halfWindow = 5, const float sigma_d = 0.5, const float sigma_r = 0.1)
{
    // https://www.jianshu.com/p/ce4afe599d6a
    int width = high.cols;
    int height = high.rows;
    float factor = 2;  // 高像素与低像素图像尺度之比为2
    cv::Mat upSampled; // 上采样后的图像
    int type = 1;
    if (coarseMap.type() == CV_32F) //深度图 type=1
        upSampled = cv::Mat::zeros(high.size(), CV_32F);
    else if (coarseMap.type() == CV_32FC3) // 法向图 type=0
    {
        type = 0;
        upSampled = cv::Mat::zeros(high.size(), CV_32FC3);
    }
    else
    {
        cout << "mat type must be float or Vec3f" << endl;
        exit(0);
    }

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            cv::Vec3b p = high.at<cv::Vec3b>(i, j);
            // 确定当前像素支撑窗口的边界
            float low_i = i / factor;
            float low_j = j / factor;
            int iMax = floor(min(low.rows - 1.f, low_i + halfWindow));
            int iMin = ceil(max(0.f, low_i - halfWindow));
            int jMax = floor(min(low.cols - 1.f, low_j + halfWindow));
            int jMin = ceil(max(0.f, low_j - halfWindow));
            // 计算f函数，结果存储在spatial中
            cv::Mat lowWindow = coarseMap.rowRange(iMin, iMax + 1).colRange(jMin, jMax + 1).clone();
            cv::Mat spatial = cv::Mat::zeros(lowWindow.size(), CV_32F);
            for (int m = 0; m < spatial.rows; m++)
                for (int n = 0; n < spatial.cols; n++)
                {
                    float x = iMin + m - low_i;
                    float y = jMin + n - low_j;
                    spatial.at<float>(m, n) = exp(-(x * x + y * y) / (2 * sigma_d * sigma_d));
                }
            // highWindow是像素P在高分辨率下的支撑窗口，如果低分辨率时窗口为5*5
            // 高分辨率下应该就是10*10，但公式中要求二者一致，因此高分辨率下窗口也是5*5
            // 这就需要对窗口的数据降采样，隔一行采样一行，隔一列采样一列
            cv::Mat highWindow = cv::Mat::zeros(spatial.size(), CV_8UC3);
            for (int m = 0; m < spatial.rows; m++)
                for (int n = 0; n < spatial.cols; n++)
                {
                    highWindow.at<cv::Vec3b>(m, n) = high.at<cv::Vec3b>((iMin + m) * factor, (jMin + n) * factor);
                }

            // 计算g函数，结果存储在range里
            cv::Mat range = cv::Mat::zeros(highWindow.size(), CV_32F); // range kernel filter 也就是g函数
            cv::MatIterator_<cv::Vec3b> highBegin = highWindow.begin<cv::Vec3b>();
            cv::MatIterator_<cv::Vec3b> highEnd = highWindow.end<cv::Vec3b>();
            cv::MatIterator_<float> rangeBegin = range.begin<float>();
            while (highBegin != highEnd)
            {
                float B = ((*highBegin)[0] - p[0]) / 255.f;
                float G = ((*highBegin)[1] - p[1]) / 255.f;
                float R = ((*highBegin)[2] - p[2]) / 255.f;
                *rangeBegin = exp(-(B * B + G * G + R * R) / (2 * sigma_r * sigma_r));
                highBegin++;
                rangeBegin++;
            }

            cv::Mat spatial_range = cv::Mat::zeros(range.size(), CV_32F);
            spatial_range = spatial.mul(range);
            float Kp = cv::sum(spatial_range)[0];
            cv::MatIterator_<float> sumBegin = spatial_range.begin<float>();
            cv::MatIterator_<float> sumEnd = spatial_range.end<float>();
            if (type)
            {
                float depth = 0;
                cv::MatIterator_<float> lowBegin = lowWindow.begin<float>();
                while (sumBegin != sumEnd)
                {
                    depth += (*sumBegin) * (*lowBegin);
                    sumBegin++;
                    lowBegin++;
                }
                depth /= Kp;
                upSampled.at<float>(i, j) = depth;
            }
            else
            {
                cv::Vec3f normal(0, 0, 0);
                cv::MatIterator_<cv::Vec3f> lowBegin = lowWindow.begin<cv::Vec3f>();
                while (sumBegin != sumEnd)
                {
                    normal += (*sumBegin) * (*lowBegin);
                    sumBegin++;
                    lowBegin++;
                }
                normal /= Kp;
                upSampled.at<cv::Vec3f>(i, j) = normal;
            }
        }
    return upSampled;
}

ImageArr DownSampleImageArr(ImageArr &images)
{
    int imageNum = images.size();
    ImageArr downSampled(imageNum);
    for (int i = 0; i < imageNum; i++)
    {
        Image &imgSource = images[i];
        Image &imgDownSample = downSampled[i];
        imgDownSample.ID = imgSource.ID;
        imgDownSample.name = imgSource.name;

        imgDownSample.imageRGB = DownSample(imgSource.imageRGB);
        imgDownSample.imageGray = DownSample(imgSource.imageGray);
        imgDownSample.width = imgDownSample.imageGray.cols;
        imgDownSample.height = imgDownSample.imageGray.rows;
        ASSERT(imgDownSample.imageGray.size() == imgDownSample.imageRGB.size());

        // imgDownSample.K = cv::Matx33f::eye();
        imgDownSample.K(0, 0) = imgSource.K(0, 0) / 2.f;
        imgDownSample.K(0, 2) = imgSource.K(0, 2) / 2.f;
        imgDownSample.K(1, 1) = imgSource.K(1, 1) / 2.f;
        imgDownSample.K(1, 2) = imgSource.K(1, 2) / 2.f;
        imgDownSample.K(2, 2) = imgSource.K(2, 2);
        // R在降采样后应该不变
        imgDownSample.R = imgSource.R;

        imgDownSample.C = imgSource.C;
        // imgDownSample.C.y = imgSource.C.y / 2.f;
        // imgDownSample.C.z = imgSource.C.z / 2.f;
        imgDownSample.T = -imgDownSample.R * imgDownSample.C;
        cv::hconcat(imgDownSample.K * imgDownSample.R, cv::Mat(imgDownSample.K * imgDownSample.T), imgDownSample.P);

        // 深度在降采样后应该也不变
        imgDownSample.dMin = imgSource.dMin;
        imgDownSample.dMax = imgSource.dMax;
    }
    return downSampled;
}

cv::Vec3f GenerateRandomNormal(cv::Point2i pt, Image refImage)
{
    float v1 = 0.0f;
    float v2 = 0.0f;
    float s = 2.0f;
    cv::RNG rng(cv::getTickCount());
    while (s >= 1.0f)
    {
        v1 = 2.0f * rng.uniform(0.f, 1.f) - 1.0f;
        v2 = 2.0f * rng.uniform(0.f, 1.f) - 1.0f;
        s = v1 * v1 + v2 * v2;
    }
    cv::Vec3f normal(0, 0, 0);
    const float s_norm = sqrt(1.0f - s);
    normal[0] = 2.0f * v1 * s_norm;
    normal[1] = 2.0f * v2 * s_norm;
    normal[2] = 1.0f - 2.0f * s;

    // Make sure normal is looking away from camera.
    cv::Vec3f view_ray((pt.x - refImage.K(0, 2)) / refImage.K(0, 0), // (col-cx)/fx
                       (pt.y - refImage.K(1, 2)) / refImage.K(1, 1), // (row-cy)/fy
                       1.f);
    if (normal.dot(view_ray) > 0)
    {
        normal[0] = -normal[0];
        normal[1] = -normal[1];
        normal[2] = -normal[2];
    }
    return normal;
}

// 对每个图像的像素初始化一个深度与法向
int Init(ImageArr &images)
{
    int imageNum = images.size();
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < imageNum; i++)
    {
        Image &image = images[i];
        image.depthMap = cv::Mat(image.height, image.width, CV_32F);
        image.normalMap = cv::Mat(image.height, image.width, CV_32FC3);
        cv::RNG rng(cv::getTickCount());
        rng.fill(image.depthMap, cv::RNG::UNIFORM, image.dMin, image.dMax);
        for (int m = 0; m < image.width; m++)
            for (int n = 0; n < image.height; n++)
            {
                cv::Point2i pt(m, n);
                image.normalMap.at<cv::Vec3f>(pt) = GenerateRandomNormal(pt, image);
            }
    }
    return 0;
}

// 计算双边权重适应性NCC Bilateral Weighted Adaption of NCC
// imageInfo - srcImage对应的图像信息，也就是Hm Hl Hr
// return - NCC值[0-2] 0为最佳 2为最差
float BilateralNCC(Image refImage, Image srcImage, NeighborInfo imageInfo,
                   const cv::Point2i x, float depth, const cv::Vec3f normal)
{
    const float sigma_g(0.2); //计算NCC的公式中的两个参数
    const float sigma_x(3.f);
    const cv::Point2i lt0(x.x - OPT::nSizeHalfWindow, x.y - OPT::nSizeHalfWindow); // 以x为中心的图像块左上角
    const cv::Point2i rb0(x.x + OPT::nSizeHalfWindow, x.y + OPT::nSizeHalfWindow); // 以x为中心的图像块右下角

    const unsigned nSizeHalfWindow = 3;
    const unsigned nSizeWindow = 2 * nSizeHalfWindow + 1;
    const unsigned nTexels = nSizeWindow * nSizeWindow;

    // 在 refImage上进行计算的部分
    if (!refImage.IsInside(lt0) || !refImage.IsInside(rb0))
        return 2.f;

    Eigen::Array<float, nTexels, 1> ref_texels;      // texels存储当前图像中以x为中心的方形区域灰度值
    Eigen::Array<float, nTexels, 1> distance_square; //distance_square存储x为中心的区域中的像素点与x的距离的平方
    Eigen::Array<float, nTexels, 1> ref_texels_square;
	Eigen::Array<float, nTexels, 1> bilateral_weight;
	float center = refImage.imageGray.at<float>(x);
    int k = 0;
    for (int i = 0; i < nSizeWindow; i++)
    {
        for (int j = 0; j < nSizeWindow; j++)
        {
			cv::Point2i ptn(lt0.x + j, lt0.y + i);
			float current = refImage.imageGray.at<float>(ptn);
			float delta_g = (current - center) / 255.f;
            ref_texels(k) = current ;
            ref_texels_square(k) = current * current;
            distance_square(k) = (i - nSizeHalfWindow) * (i - nSizeHalfWindow) + (j - nSizeHalfWindow) * (j - nSizeHalfWindow);
            bilateral_weight(k) = exp(- delta_g * delta_g / (2 * sigma_g * sigma_g) - distance_square(k) / (2 * sigma_x * sigma_x)); //双边权重
			k++;
        }
    }

    // 在 srcImage上进行计算的部分
    cv::Point3f X0 = refImage.TransformPointI2C(cv::Point3f(x.x, x.y, 1));
    const cv::Matx33f H((imageInfo.Hl + cv::Vec3f(imageInfo.Hm) * normal.t() * (1.f / normal.dot(X0 * depth))) * imageInfo.Hr); // 由空间面片诱导的单应矩阵H
    Eigen::Array<float, nTexels, 1> src_texels;
    Eigen::Array<float, nTexels, 1> src_texels_square;
    k = 0;
    for (int i = 0; i < nSizeWindow; i++)
    {
        for (int j = 0; j < nSizeWindow; j++)
        {
            cv::Point2f pt = ProjectH(H.val, cv::Vec2f((float)(lt0.x + j), (float)(lt0.y + i)).val); // 通过H计算邻域图像上的对应点pt
            if (!srcImage.IsInside(pt))
                return 2.f;
            src_texels(k) = srcImage.Sample(pt); // 通过双线性插值计算灰度值
            src_texels_square(k) = src_texels(k) * src_texels(k);
            k++;
        }
    }

    const float bilateral_weight_sum = bilateral_weight.sum();
    float src_ref_avg = (ref_texels * src_texels * bilateral_weight).sum() / bilateral_weight_sum; //E(xy)
    float src_avg = (src_texels * bilateral_weight).sum() / bilateral_weight_sum;                  //E(y)
    float ref_avg = (ref_texels * bilateral_weight).sum() / bilateral_weight_sum;                  //E(x)
    float src_avg_square = (src_texels_square * bilateral_weight).sum() / bilateral_weight_sum;    //E(y^2)
    float ref_avg_square = (ref_texels_square * bilateral_weight).sum() / bilateral_weight_sum;    //E(x^2)
    float src_src_cov = src_avg_square - src_avg * src_avg;                                        //E(y^2)-E(y)^2
    float ref_ref_cov = ref_avg_square - ref_avg * ref_avg;                                        //E(x^2)-E(x)^2

    const float kMinVar = 1e-5f;
    if (src_src_cov < kMinVar || ref_ref_cov < kMinVar)
        return 2.f;

    float src_ref_cov = src_ref_avg - src_avg * ref_avg; //E(xy)-E(x)E(y)
    float ncc = src_ref_cov / sqrt(src_src_cov * ref_ref_cov);
    // 返回的评分为[0-2] 0为最佳 2为最差
    return max(0.f, min(2.f, 1 - ncc));
}

// 计算代价矩阵 一般是8*N 或者 9*N
// return - 由NCC组成的代价矩阵
cv::Mat ComputeBilateralNCC(ImageArr images, vector<pair<float, cv::Vec3f>> depth_normal, cv::Point2i pt, const int refID)
{
    int imageNum = images.size();
    cv::Mat bilateralCost = cv::Mat::zeros(depth_normal.size(), imageNum, CV_32F);
    Image refImage = images[refID];            

    vector<NeighborInfo> imageInfos;
    for (auto img : images)
    {
        NeighborInfo neibor;
        neibor.ID = img.ID;
        neibor.Hl = img.K * img.R * refImage.R.t();
        neibor.Hm = img.K * img.R * (refImage.C - img.C);
        neibor.Hr = refImage.K.inv();
        imageInfos.push_back(neibor);
    }
    for (int i = 0; i < bilateralCost.rows; i++)
    {
        float depth = depth_normal[i].first;
        cv::Vec3f normal = depth_normal[i].second;
        for (int j = 0; j < bilateralCost.cols; j++)
        {
            if (refID == j)
            {
                bilateralCost.at<float>(i, j) = 2.f;
                continue;
            }
            Image srcImage = images[j];
            NeighborInfo imageInfo = imageInfos[j];
            bilateralCost.at<float>(i, j) = BilateralNCC(refImage, srcImage, imageInfo, pt, depth, normal);
        }
    }
    return bilateralCost;
}

// 根据NCC代价矩阵计算各个视图的权重，同时选出最重要的那个视图
// last-上一次迭代中当前像素最重要的视图的ID
Eigen::ArrayXf ComputeViewWeight(cv::Mat cost, int iteration, const int refID, int &last, bool update=false)
{
    float init_good_threshold = 0.8;
    float alpha = 90;
    float beta = 0.3;
    float good_threshold = init_good_threshold * exp(-iteration * iteration / alpha);
    float bad_threshold = 1.2;
    int n1 = 2;
    int n2 = 3;
    vector<int> S_t;
    Eigen::Array<float, Eigen::Dynamic, 1> viewWeight; // 每个视图的权重，权重最大的被认为是最重要的视图
    viewWeight.resize(cost.cols);
    for (int i = 0; i < cost.cols; i++)
    {
        if (i == refID)
        {
            // viewWeight << 0.f;
            viewWeight(i) = 0;
            continue;
        }
        vector<float> S_good;
        vector<float> S_bad;

        for (int j = 0; j < cost.rows; j++)
        {
            float c = cost.at<float>(j, i);
            if (c < good_threshold)
                S_good.push_back(c);
            else if (c > bad_threshold)
                S_bad.push_back(c);
        }
        if (S_good.size() > n1 && S_bad.size() < n2)
            S_t.push_back(i);
        else
        {
            viewWeight(i) = 0.2 * (i == last);
            continue;
        }

        Eigen::Array<float, Eigen::Dynamic, 1> confidence;
        confidence.resize(S_good.size());
        for (int i = 0; i < confidence.size(); i++)
        {
            float c = S_good[i];
            float conf = exp(-c * c / (2 * beta * beta));
            confidence(i) = conf;
        }
        float weight = confidence.sum() / S_good.size();
        weight = ((i == last) + 1) * weight;
        viewWeight(i) = weight;
    }
    // 找到权重最高的view并更新last
    if(update)
    {
        Eigen::ArrayXf::Index maxIndex;
        viewWeight.maxCoeff(&maxIndex);
        last = maxIndex;
    }
    return viewWeight;
}

// 从多个猜测中选择一个最好的出来
// return - 最好的猜测的索引以及他的多视图匹配代价
// update=true 会在最后更新最重要的视图，为下次的迭代准备
// geo_consistency=true代表使用几何一致性
pair<int, float> SelectHypotheses(cv::Mat cost, Eigen::ArrayXf viewWeight, bool geo_consistency = false, cv::Mat projError = cv::Mat(1, 1, 0))
{
    // 计算各个view的代价
    Eigen::Array<float, Eigen::Dynamic, 1> viewCost;
    viewCost.resize(cost.cols);
    vector<float> photometric;
    for (int i = 0; i < cost.rows; i++)
    {
        for (int j = 0; j < cost.cols; j++)
        {
            if (geo_consistency)
                viewCost(j) = cost.at<float>(i, j) + 0.2 * projError.at<float>(i, j);
            else
                viewCost(j) = cost.at<float>(i, j);
        }
        photometric.push_back((viewCost * viewWeight).sum() / viewWeight.sum());
    }
    std::vector<float>::iterator smallest = std::min_element(photometric.begin(), photometric.end());
    int idx = smallest - photometric.begin();
    float c = photometric[idx];
    pair<int, float> idx_cost(idx, c);
    return idx_cost;
}

struct PointScore
{
    cv::Point2i pt;
    float score;
};

bool sortPointScore(const PointScore s1, const PointScore s2)
{
    return s1.score < s2.score; //升序排列
}

vector<cv::Point2i> CheckerboardSampling(Image image, const cv::Point2i pt, const cv::Mat costMap)
{
    vector<cv::Point2i> candidate;
    // 上下左右四个临近的像素
    candidate.push_back(cv::Point2i(pt.x - 1, pt.y));
    candidate.push_back(cv::Point2i(pt.x + 1, pt.y));
    candidate.push_back(cv::Point2i(pt.x, pt.y - 1));
    candidate.push_back(cv::Point2i(pt.x, pt.y + 1));
    // 四个V形区域的其他像素
    for (int i = 2; i < 5; i++)
    {
        // 左侧V形区域
        candidate.push_back(cv::Point2i(pt.x - i, pt.y + i - 1));
        candidate.push_back(cv::Point2i(pt.x - i, pt.y - i + 1));
        // 右侧V形区域
        candidate.push_back(cv::Point2i(pt.x + i, pt.y + i - 1));
        candidate.push_back(cv::Point2i(pt.x + i, pt.y - i + 1));
        // 上方V形区域
        candidate.push_back(cv::Point2i(pt.x + i - 1, pt.y - i));
        candidate.push_back(cv::Point2i(pt.x - i + 1, pt.y - i));
        // 下方V形区域
        candidate.push_back(cv::Point2i(pt.x + i - 1, pt.y + i));
        candidate.push_back(cv::Point2i(pt.x - i + 1, pt.y + i));
    }
    // 四个长条区域
    for (int i = 3; i < 25; i += 2)
    {
        candidate.push_back(cv::Point2i(pt.x, pt.y - i)); // 上
        candidate.push_back(cv::Point2i(pt.x, pt.y + i)); // 下
        candidate.push_back(cv::Point2i(pt.x - i, pt.y)); // 左
        candidate.push_back(cv::Point2i(pt.x + i, pt.y)); // 右
    }

    vector<PointScore> hypothesis;
    for (cv::Point2i point : candidate)
    {
        if (!image.IsInside(point))
            continue;
        PointScore ps;
        ps.pt = point;
        ps.score = costMap.at<float>(point);
        hypothesis.push_back(ps);
    }
    ASSERT(hypothesis.size() >= 8);
    sort(hypothesis.begin(), hypothesis.end(), sortPointScore);
    candidate.clear();
    hypothesis.resize(8);
    for (PointScore s : hypothesis)
        candidate.push_back(s.pt);
    return candidate;
}

// 计算初始的多视图匹配代价 4.1节对应的内容
// 计算参考图像中的每个像素的多视图匹配代价 匹配代价存在了confmap里
cv::Mat InitMatchingCost(ImageArr &images, const int refID)
{
    int imageNum = images.size();
    Image &refImage = images[refID];
    cv::Mat initCost = cv::Mat::zeros(refImage.height, refImage.width, CV_32F);
    // 计算当前参考图像与其他所有图像之间的H矩阵(包括自己)
    vector<NeighborInfo> imageInfos;
    for (auto img : images)
    {
        NeighborInfo neibor;
        neibor.ID = img.ID;
        neibor.Hl = img.K * img.R * refImage.R.t();
        neibor.Hm = img.K * img.R * (refImage.C - img.C);
        neibor.Hr = refImage.K.inv();
        imageInfos.push_back(neibor);
    }
    ASSERT(imageInfos.size() == imageNum);
    // 遍历每个像素 计算初始的多视图匹配代价
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < refImage.width; i++)
        for (int j = 0; j < refImage.height; j++)
        {
            cv::Point2i pt(i, j);
            cv::Vec3f normal = refImage.normalMap.at<cv::Vec3f>(pt);
            float depth = refImage.depthMap.at<float>(pt);
            vector<float> costs;
            // 计算初始的多视图匹配代价
            for (int m = 0; m < imageNum; m++)
            {
                Image srcImage = images[m];
                ASSERT(srcImage.ID == m);
                if (m == refID)
                {
                    costs.push_back(2.f);
                    continue;
                }
                float cost = BilateralNCC(refImage, srcImage, imageInfos[m], pt, depth, normal);
                costs.push_back(cost);
            }

            // 找到costs里最小的K个，然后取平均值作为最终的匹配代价
            float cost = 0;
            vector<float>::iterator minIndex;
            int K = 5;
            for (int k = 0; k < K; k++)
            {
                minIndex = min_element(costs.begin(), costs.end());
                cost += costs[minIndex - costs.begin()];
                costs.erase(minIndex);
            }
            initCost.at<float>(pt) = cost;
        }
    return initCost;
}

cv::Vec3f PerturbNormal(cv::Vec3f normal, const float perturbation)
{
    cv::RNG rng(cv::getTickCount());
    const float a1 = (rng.operator float() - 0.5f) * perturbation;
    const float a2 = (rng.operator float() - 0.5f) * perturbation;
    const float a3 = (rng.operator float() - 0.5f) * perturbation;
    const float sin_a1 = sin(a1);
    const float sin_a2 = sin(a2);
    const float sin_a3 = sin(a3);
    const float cos_a1 = cos(a1);
    const float cos_a2 = cos(a2);
    const float cos_a3 = cos(a3);

    // R = Rx * Ry * Rz
    float R[9];
    R[0] = cos_a2 * cos_a3;
    R[1] = -cos_a2 * sin_a3;
    R[2] = sin_a2;
    R[3] = cos_a1 * sin_a3 + cos_a3 * sin_a1 * sin_a2;
    R[4] = cos_a1 * cos_a3 - sin_a1 * sin_a2 * sin_a3;
    R[5] = -cos_a2 * sin_a1;
    R[6] = sin_a1 * sin_a3 - cos_a1 * cos_a3 * sin_a2;
    R[7] = cos_a3 * sin_a1 + cos_a1 * sin_a2 * sin_a3;
    R[8] = cos_a1 * cos_a2;

    cv::Vec3f prt_normal(R[0] * normal[0] + R[1] * normal[1] + R[2] * normal[2],
                         R[3] * normal[0] + R[4] * normal[1] + R[5] * normal[2],
                         R[6] * normal[0] + R[7] * normal[1] + R[8] * normal[2]);
    return prt_normal;
}

// 根据给定的depth和normal生成几个新的depth和normal
// iteration - 当前迭代次数
// pt - 像素点的坐标
vector<std::pair<float, cv::Vec3f>> GenerateDepthNormal(float depth, cv::Vec3f normal, const int iteration, Image refImage, cv::Point2i pt)
{
    cv::RNG rng(cv::getTickCount());
    float rand_depth = rng.uniform(refImage.dMin, refImage.dMax);
    cv::Vec3f rand_normal = GenerateRandomNormal(pt, refImage);
    // 产生随机扰动是借鉴了colmap
    float perturbation = 1.0f / std::pow(2.0f, iteration);
    float max_depth = (1 + perturbation) * depth;
    float min_depth = (1 - perturbation) * depth;
    float prt_depth = rng.uniform(0.f, 1.f) * (max_depth - min_depth) + min_depth;

    cv::Vec3f prt_normal = PerturbNormal(normal, perturbation * PI);
    cv::Vec3f view_ray((pt.x - refImage.K(0, 2)) / refImage.K(0, 0), // (col-cx)/fx
                       (pt.y - refImage.K(1, 2)) / refImage.K(1, 1), // (row-cy)/fy
                       1.f);
    // 扰动后的法向要和之前的法向方向相同，如果不同则重新计算，最多重算三次
    int m = 0;
    while (prt_normal.dot(view_ray) > 0)
    {
        if (m == 3)
        {
            prt_normal = normal;
            break;
        }
        prt_normal = PerturbNormal(normal, 0.5 * perturbation * PI);
        perturbation *= 0.5;
        m++;
    }
    prt_normal = cv::normalize(prt_normal);
    // 经过以上步骤后，现在有三组深度和法向，分别是
    // (depth, normal)  (rand_depth, rand_normal)  (prt_depth, prt_normal)
    // 用这三组再生成六组深度和法向
    vector<std::pair<float, cv::Vec3f>> depth_normal;
    pair<float, cv::Vec3f> dn1;
    dn1 = make_pair(depth, normal);
    depth_normal.push_back(dn1);

    pair<float, cv::Vec3f> dn2;
    dn2 = make_pair(depth, prt_normal);
    depth_normal.push_back(dn2);

    pair<float, cv::Vec3f> dn3;
    dn3 = make_pair(depth, rand_normal);
    depth_normal.push_back(dn3);

    pair<float, cv::Vec3f> dn4;
    dn4 = make_pair(rand_depth, normal);
    depth_normal.push_back(dn4);

    pair<float, cv::Vec3f> dn5;
    dn5 = make_pair(rand_depth, prt_normal);
    depth_normal.push_back(dn5);

    pair<float, cv::Vec3f> dn6;
    dn6 = make_pair(rand_depth, rand_normal);
    depth_normal.push_back(dn6);

    pair<float, cv::Vec3f> dn7;
    dn7 = make_pair(prt_depth, normal);
    depth_normal.push_back(dn7);

    pair<float, cv::Vec3f> dn8;
    dn8 = make_pair(prt_depth, rand_normal);
    depth_normal.push_back(dn8);

    pair<float, cv::Vec3f> dn9;
    dn9 = make_pair(prt_depth, prt_normal);
    depth_normal.push_back(dn9);

    return depth_normal;
}

cv::Mat ComputeReprojectionError(ImageArr &images, vector<pair<float, cv::Vec3f>> depth_normal, const int refID, cv::Point2i pt)
{
    Image refImage = images[refID];
    cv::Mat projError = cv::Mat::zeros(depth_normal.size(), images.size(), CV_32F);
    for (int m = 0; m < projError.rows; m++)
        for (int n = 0; n < projError.cols; n++)
        {
            if (n == refID)
            {
                projError.at<float>(m, n) = 3.f;
                continue;
            }
            float d = depth_normal[m].first;
            Image srcImage = images[n];
            cv::Point3f X = refImage.TransformPointI2W(cv::Point3f(pt.x, pt.y, d));
            cv::Point2i srcX = srcImage.TransformPointC2Ii(srcImage.TransformPointW2C(X));
            if (!srcImage.IsInside(srcX))
            {
                projError.at<float>(m, n) = 3.f;
                continue;
            }
            X = srcImage.TransformPointI2W(cv::Point3f(srcX.x, srcX.y, srcImage.depthMap.at<float>(srcX)));
            cv::Point2i refX = refImage.TransformPointC2Ii(refImage.TransformPointW2C(X));
            if (!refImage.IsInside(refX))
            {
                projError.at<float>(m, n) = 3.f;
                continue;
            }
            float error = sqrt((pt.x - refX.x) * (pt.x - refX.x) + (pt.y - refX.y) * (pt.y - refX.y));
            projError.at<float>(m, n) = min(error, 3.f);
        }
    return projError;
}

// geo_consistency=true 时会使用光度一致性加几何一致性
// geo_consistency=false 时只使用光度一致性
void ACMH(ImageArr &images, int refID, vector<cv::Mat> weightMapArr, bool geo_consistency = false)
{
    int imageNum = images.size();
    int maxIteration = 4;
    LOG(INFO) << "Iteration times: " << maxIteration << " reference ID: " << refID;
    // 对参考图像进行一次ACMH
    Image &refImage = images[refID];
    refImage.confMap = cv::Mat(refImage.height, refImage.width, CV_32F, cv::Scalar(2));
    LOG(INFO) << "Current scale is " << refImage.width << " x " << refImage.height;
    cv::Mat lastImportantView(refImage.height, refImage.width, CV_32S, INT_MAX); //记录上一次迭代后对于每个像素最重要的视图的ID
    // 重复迭代n次
    for (int it = 0; it < maxIteration; it++)
    {
        LOG(INFO) << "Iteration: " << it;
        cv::Mat initCost = InitMatchingCost(images, refID);
        LOG(INFO) << "Finish init matching cost";
        // 这样是为了增加并行性，一次性处理一半的点，就像棋盘格
        for (int a = 0; a < 2; a++)
        {
            for (int i = 0; i < refImage.height; i++)
                #pragma omp parallel for schedule(dynamic)
                for (int j = (i % 2 + a) % 2; j < refImage.width; j += 2)
                {
                    cv::Point2i pt(j, i);
                    // 使用初始的匹配代价从pt周围的点里选出8个比较好的，存储在good里 4.2节
                    vector<cv::Point2i> good = CheckerboardSampling(refImage, pt, initCost);
                    ASSERT(good.size() == 8);
                    if (DEBUG_ACMH)
                        LOG(INFO) << "Finish checker board sample from current point";
                    vector<pair<float, cv::Vec3f>> good_depth_normal;
                    for (cv::Point2i point : good)
                    {
                        float d = refImage.depthMap.at<float>(point);
                        cv::Vec3f n = refImage.normalMap.at<cv::Vec3f>(point);
                        pair<float, cv::Vec3f> depth_normal = make_pair(d, n);
                        good_depth_normal.push_back(depth_normal);
                    }
                    cv::Mat bilateralCost = ComputeBilateralNCC(images, good_depth_normal, pt, refID);
                    if (DEBUG_ACMH)
                        LOG(INFO) << "Finish compute cost matrix with NCC for the 1st time";

                    // 从8个里选出一个最好的猜测 4.3节
                    pair<int, float> idx_cost;
                    Eigen::ArrayXf viewWeight;
                    if (geo_consistency)
                    {
                        cv::Mat projError = ComputeReprojectionError(images, good_depth_normal, refID, pt);
                        viewWeight = ComputeViewWeight(bilateralCost, it, refID, lastImportantView.at<int>(pt));
                        idx_cost = SelectHypotheses(bilateralCost, viewWeight, geo_consistency, projError);
                    }
                    else
                    {
                        viewWeight = ComputeViewWeight(bilateralCost, it, refID, lastImportantView.at<int>(pt));
                        idx_cost = SelectHypotheses(bilateralCost, viewWeight);
                    }
                    if (DEBUG_ACMH)
                        LOG(INFO) << "Finish selecting the best hypothesis from 8 hypotheses";

                    if (idx_cost.second < refImage.confMap.at<float>(pt))
                    {
                        refImage.confMap.at<float>(pt) = idx_cost.second;
                        refImage.depthMap.at<float>(pt) = good_depth_normal[idx_cost.first].first;
                        refImage.normalMap.at<cv::Vec3f>(pt) = good_depth_normal[idx_cost.first].second;
                    }
                    if (DEBUG_ACMH)
                        LOG(INFO) << "Finish propagation";
                }
        }

        // 存储每次迭代后的结果
        char num[10];
        sprintf(num, "%d_", it);
        string snum(num);
        // iterationx_refID.jpg
        SaveDepthMap(refImage.depthMap, OPT::sDataPath + "/depthmap/iteration" + snum + refImage.name);

        // 对结果进行refine 4.4节
        # pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < refImage.height; i++)
            for(int j = 0; j < refImage.width; j++)
            {
                cv::Point2i pt(j,i);
                float depth = refImage.depthMap.at<float>(pt);
                cv::Vec3f normal = refImage.normalMap.at<cv::Vec3f>(pt);
                // 用当前点的深度和法向再生成额外的8对深度和法向
                vector<pair<float, cv::Vec3f> > refine_depth_normal = GenerateDepthNormal(depth, normal, it, refImage, pt);
                if(DEBUG_ACMH) LOG(INFO) << "Finish generating 9 hypotheses with the best hypothesis";
                cv::Mat bilateralCost = ComputeBilateralNCC(images, refine_depth_normal, pt, refID);
                ASSERT(refine_depth_normal.size() == 9 && bilateralCost.rows == 9);
                if(DEBUG_ACMH) LOG(INFO) << "Finish compute cost matrix with NCC for the 2nd time";
                Eigen::ArrayXf viewWeight;
                pair<int, float> idx_cost;
                if(geo_consistency)
                {
                    cv::Mat projError = ComputeReprojectionError(images, refine_depth_normal, refID, pt);
                    viewWeight = ComputeViewWeight(bilateralCost, it, refID, lastImportantView.at<int>(pt), true);
                    idx_cost = SelectHypotheses(bilateralCost, viewWeight, geo_consistency, projError);
                }
                else
                {
                    viewWeight = ComputeViewWeight(bilateralCost, it, refID, lastImportantView.at<int>(pt), true);
                    idx_cost = SelectHypotheses(bilateralCost, viewWeight);
                }
                if(DEBUG_ACMH) LOG(INFO) << "Finish selecting the best hypothesis from 9 hypotheses";

                // 把各个视图对当前点pt的权重存储到weightMapArr的对应位置
                ASSERT(weightMapArr.size() == viewWeight.size());
                if(DEBUG_ACMH) LOG(INFO) <<"Saving view weight";
                for(int k=0; k<imageNum; k++)
                {
                    weightMapArr[k].at<float>(pt) = viewWeight(k);
                }

                if (idx_cost.second < refImage.confMap.at<float>(pt))
                {
                    refImage.confMap.at<float>(pt) = idx_cost.second;
                    refImage.depthMap.at<float>(pt) = refine_depth_normal[idx_cost.first].first;
                    refImage.normalMap.at<cv::Vec3f>(pt) = refine_depth_normal[idx_cost.first].second;
                }
                
                if(DEBUG_ACMH) LOG(INFO) << "Finish refinment";
            }

        // 存储refine之后的结果
        SaveDepthMap(refImage.depthMap, OPT::sDataPath + "/depthmap/refine" + snum + refImage.name);
    }

    // 对最终的深度图使用 5x5的中值滤波
    cv::Mat filter;
    cv::medianBlur(refImage.depthMap, filter, 3);
    refImage.depthMap = filter.clone();
    LOG(INFO) << "Finish median filter";
}

// 细节恢复器
void DetailRestore(ImageArr &images, const int refID, WeightMap weightMap)
{
    ASSERT(weightMap[0].size() == (images[0].depthMap.size() / 2));
    Image &refImage = images[refID];

    // 这是存储 init Photometric Consistency Cost
    cv::Mat initCostMap = cv::Mat::zeros(refImage.depthMap.size(), CV_32F);
    if (DEBUG_DERES)
        LOG(INFO) << "Finish init matching cost";
    # pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < refImage.height; i++)
        for (int j = 0; j < refImage.width; j++)
        {
            cv::Point2i pt(j, i);
            float depth = refImage.depthMap.at<float>(pt);
            cv::Vec3f normal = refImage.normalMap.at<cv::Vec3f>(pt);
            vector<pair<float, cv::Vec3f>> good_depth_normal;
            good_depth_normal.push_back(make_pair(depth, normal));
            cv::Mat bilateralNCC = ComputeBilateralNCC(images, good_depth_normal, pt, refID);
            ASSERT(bilateralNCC.rows == 1);
            
            float weightSum = 0, a = 0;
            cv::Point2i ptDown(i / 2, j / 2);
            for (cv::MatIterator_<float> it = bilateralNCC.begin<float>(); it != bilateralNCC.end<float>(); it++)
            {
                float weight = weightMap[it - bilateralNCC.begin<float>()].at<float>(ptDown);
                a += *(it) * weight;
                weightSum += weight;
            }
            // 通过上采样后得到的深度和法向计算的光度一致性代价
            const float initPhotoCost = a / weightSum;
            if (DEBUG_DERES)
                LOG(INFO) << "Finish computing initial photo consistency cost, which is " << initPhotoCost;
            initCostMap.at<float>(pt) = initPhotoCost;
        }

    /*以下是使用basic MVS计算一次光度一致性代价
    其实就是ACMH，不过是只迭代一次的ACMH*/
    // 接下来ACMH要用这两个作为深度图和法向图，不能在原图上操作
    cv::Mat depthMapTmp = refImage.depthMap.clone();
    cv::Mat normalMapTmp = refImage.normalMap.clone();
    cv::Mat confMapTmp = cv::Mat::zeros(refImage.depthMap.size(), CV_32F);

    cv::Mat initCost = InitMatchingCost(images, refID);
    LOG(INFO) << "Finish init matching cost";
    // 这样是为了增加并行性，一次性处理一半的点，就像棋盘格
    for (int a = 0; a < 2; a++)
    {
        for (int i = 0; i < refImage.height; i++)
            #pragma omp parallel for schedule(dynamic)
            for (int j = (i % 2 + a) % 2; j < refImage.width; j += 2)
            {
                cv::Point2i pt(j, i);
                // 使用初始的匹配代价从pt周围的点里选出8个比较好的，存储在good里
                vector<cv::Point2i> good = CheckerboardSampling(refImage, pt, initCost);
                ASSERT(good.size() == 8);
                if (DEBUG_DERES)
                    LOG(INFO) << "Finish checker board sample";
                vector<pair<float, cv::Vec3f>> good_depth_normal;
                for (auto point : good)
                {
                    float d = refImage.depthMap.at<float>(point);
                    cv::Vec3f n = refImage.normalMap.at<cv::Vec3f>(point);
                    pair<float, cv::Vec3f> depth_normal = make_pair(d, n);
                    good_depth_normal.push_back(depth_normal);
                }
                cv::Mat bilateralCost = ComputeBilateralNCC(images, good_depth_normal, pt, refID);
                if (DEBUG_DERES)
                    LOG(INFO) << "Finish computing cost matrix with NCC for the 1st time";
                Eigen::ArrayXf viewWeight;
                pair<int, float> idx_cost;
                int tmp = INT_MAX;
                viewWeight = ComputeViewWeight(bilateralCost, 0, refID, tmp);
                idx_cost = SelectHypotheses(bilateralCost, viewWeight);
                if (DEBUG_DERES)
                    LOG(INFO) << "Find the smallset photo consistency cost";

                depthMapTmp.at<float>(pt) = good_depth_normal[idx_cost.first].first;
                normalMapTmp.at<cv::Vec3f>(pt) = good_depth_normal[idx_cost.first].second;
                confMapTmp.at<float>(pt) = idx_cost.second;
            }
    }
    // refine
    # pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < refImage.height; i++)
        for(int j = 0; j < refImage.width; j++)
        {
            cv::Point2i pt(j,i);
            float depth = refImage.depthMap.at<float>(pt);
            cv::Vec3f normal = refImage.normalMap.at<cv::Vec3f>(pt);
            // 通过最好的猜测再生成9个猜测,然后从这9个里选一个最好的
            vector<pair<float, cv::Vec3f>> refine_depth_normal = GenerateDepthNormal(depth, normal, 0, refImage, pt);
            cv::Mat refine_bilateralCost = ComputeBilateralNCC(images, refine_depth_normal, pt, refID);
            if (DEBUG_DERES)
                LOG(INFO) << "Finish generating 9 hypotheses with best depth and normal";
            cv::Mat bilateralCost = ComputeBilateralNCC(images, refine_depth_normal, pt, refID);
            int tmp = INT_MAX;
            Eigen::ArrayXf viewWeight;
            pair<int, float> idx_cost;
            viewWeight = ComputeViewWeight(bilateralCost, 0, refID, tmp);
            idx_cost = SelectHypotheses(bilateralCost, viewWeight);
            if (DEBUG_DERES)
                LOG(INFO) << "Find the smallest photo consistency cost";

            if(idx_cost.second < confMapTmp.at<float>(pt))
            {
                depthMapTmp.at<float>(pt) = refine_depth_normal[idx_cost.first].first;
                normalMapTmp.at<cv::Vec3f>(pt) = refine_depth_normal[idx_cost.first].second;
                confMapTmp.at<float>(pt) = idx_cost.second;
            }
        }

    # pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < refImage.height; i++)
        for(int j = 0; j < refImage.width; j++)   
        {
            cv::Point2i pt(j,i);
            // 如果两个代价相差大于0.1则选择相信mvs算出来的
            if ( initCostMap.at<float>(pt) - confMapTmp.at<float>(pt) > 0.1)
            {
                refImage.depthMap.at<float>(pt) = depthMapTmp.at<float>(pt);
                refImage.normalMap.at<cv::Vec3f>(pt) = normalMapTmp.at<cv::Vec3f>(pt);
            }
        }            
}

int EstimateDepthMap2(ImageArr &images)
{
    int imageNum = images.size();

    ImageArr downSample1 = DownSampleImageArr(images); //经过一次降采样后的ImageArr 原图的0.5大小
    LOG(INFO) << "finish downsample 1st time";
    ImageArr downSample2 = DownSampleImageArr(downSample1); // 经过两次降采样后的ImageArr  原图的0.25大小
    LOG(INFO) << "finish downsample 2nd time";
    Init(downSample2);
    LOG(INFO) << "initialized depth and normal for the coarsest scale";
    
    // 对最低尺度的深度计算
    {
        // weightMap是一系列的权重，vector里面的每一项对应了一个视图，矩阵每个点代表了
        // 这个视图对这个点的权重
        WeightMapArr weightMapArr(imageNum);
        for (int i = 0; i < imageNum; i++)
            weightMapArr[i] = cv::Mat::zeros(downSample2[0].height, downSample2[0].width, CV_32F);
        // 对经过两次下采样的图像序列进行ACMH
        LOG(INFO) << "start ACMH at coarsest scale";
        for (int i = 0; i < imageNum; i++)
        {
            string fileName = "depth_ACMH_0.25_";
            SaveDepthMap(downSample2[i].depthMap, OPT::sDataPath + "/depthmap/init_" + downSample2[i].name);

            WeightMap weightMap(imageNum);
            for(int j = 0; j < weightMap.size(); j++)
                weightMap[j] = cv::Mat::zeros(refImage.height, refImage.width, CV_32F);

            ACMH(downSample2, i, weightMap);

            SaveDepthMap(downSample2[i].depthMap, OPT::sDataPath + "/depthmap/" + fileName + downSample2[i].name);
            cv::imwrite(OPT::sDataPath + "/depthmap/normal_" + downSample2[i].name, (downSample2[0].normalMap + 1) / 2 * 255);
            SaveDepthMap(downSample2[i].confMap, OPT::sDataPath + "/depthmap/conf_" + downSample2[i].name);
        }
        LOG(INFO) << "finish ACMH at coarsest scale";
        LOG(INFO) << "start ACMH & geometry at coarsest scale";
        // 根据论文，需要再进行一次 ACMH+geometry constancy
        for (int i = 0; i < imageNum; i++)
        {
            WeightMap weightMap(imageNum);
            for(int j = 0; j < weightMap.size(); j++)
                weightMap[j] = cv::Mat::zeros(downSample2[0].height, downSample2[0].width, CV_32F);

            ACMH(downSample2, i, weightMap, true);
            weightMapArr[i] = weightMap;

            string fileName = "depth_ACMH+geo_0.25_";
            SaveDepthMap(downSample2[i].depthMap, OPT::sDataPath + "/depthmap/" + fileName + downSample2[i].name);
        }
        LOG(INFO) << "finish ACMH & geometry at coarsest scale";
        // 提升一个尺度
        for (int i = 0; i < imageNum; i++)
        {
            Image &high = downSample1[i];
            Image &low = downSample2[i];
            high.depthMap = cv::Mat(high.height, high.width, CV_32F);
            high.depthMap = JointBilateralUpsample(high.imageRGB, low.imageRGB, low.depthMap);
            high.normalMap = cv::Mat(high.height, high.width, CV_32FC3);
            high.normalMap = JointBilateralUpsample(high.imageRGB, low.imageRGB, low.normalMap);
            SaveDepthMap(high.depthMap, OPT::sDataPath + "/depthmap/upSample_0.25_" + downSample2[i].name);
        }
        LOG(INFO) << "finish up sample at coarsest scale";
        // exit(0);
        for (int i = 0; i < imageNum; i++)
        {
            DetailRestore(images, i, weightMapArr[i]);
        }
            
        LOG(INFO) << "finish detail restoring at coarsest scale";
    }

    // 中间尺度的深度计算
    {
        WeightMapArr weightMapArr(imageNum);
        for (int i = 0; i < imageNum; i++)
        {
            WeightMap weightMap(imageNum);
            for(int j = 0; j < weightMap.size(); j++)
                weightMap[j] = cv::Mat::zeros(downSample1[0].height, downSample1[0].width, CV_32F);

            ACMH(downSample1, i, weightMap, true);
            weightMapArr[i] = weightMap;

            string fileName = "depth_ACMH_0.5_";
            SaveDepthMap(downSample1[i].depthMap, OPT::sDataPath + "/depthmap/" + fileName + downSample1[i].name);
        }
        LOG(INFO) << "finish ACMH & geometry at middle scale";
        // 提升一个尺度
        for (int i = 0; i < imageNum; i++)
        {
            Image &high = images[i];
            Image &low = downSample1[i];
            high.depthMap = cv::Mat(high.height, high.width, CV_32F);
            high.depthMap = JointBilateralUpsample(high.imageRGB, low.imageRGB, low.depthMap);
            high.normalMap = cv::Mat(high.height, high.width, CV_32FC3);
            high.normalMap = JointBilateralUpsample(high.imageRGB, low.imageRGB, low.normalMap);
            SaveDepthMap(high.depthMap, OPT::sDataPath + "/depthmap/upSample_0.5_" + downSample1[i].name);
        }
        LOG(INFO) << "finish up sample at middle scale";
        for (int i = 0; i < imageNum; i++)
            DetailRestore(images, i, weightMapArr[i]);
        LOG(INFO) << "finish detail restoring at middle scale";
    }

    // 原始分辨率的深度计算
    {
        WeightMapArr weightMapArr(imageNum);
        
        for (int i = 0; i < imageNum; i++)
        {
            WeightMap weightMap(imageNum);
            for(int j = 0; j < weightMap.size(); j++)
                weightMap[j] = cv::Mat::zeros(images[0].height, images[0].width, CV_32F);

            ACMH(images, i, weightMap, true);
            weightMapArr[i] = weightMap;

            string fileName = "depth_ACMH_1.0_";
            SaveDepthMap(images[i].depthMap, OPT::sDataPath + "/depthmap/" + fileName + images[i].name);
        }
        LOG(INFO) << "finish ACMH & geometry at raw scale";
    }
    return 0;
}