#include "mathes.h"

// #define IMAGE_DEMO

vector<Point2f> normalizeKeypoints(const vector<Point2f>& keypoints, int h, int w)
{
    if (w <= 0 || h <= 0) {
        return {}; // or throw
    }

    float shift_x = static_cast<float>(w) / 2.0f;
    float shift_y = static_cast<float>(h) / 2.0f;
    float scale = static_cast<float>(std::max(w, h)) / 2.0f;

    // Avoid division by zero (though scale > 0 if w,h > 0)
    if (scale == 0.0f) {
        return {};
    }

    vector<Point2f> normalized;
    normalized.reserve(keypoints.size());

    for (const auto& kp : keypoints) {
        float x_norm = (kp.x - shift_x) / scale;
        float y_norm = (kp.y - shift_y) / scale;
        normalized.emplace_back(x_norm, y_norm);
    }
    return normalized;
}

Mat drawLightGlueMatches(
    const Mat& img0,
    const Mat& img1,
    const vector<Point2i>& kpts0,
    const vector<Point2i>& kpts1)
{
    if (kpts0.empty() || kpts1.empty() || kpts0.size() != kpts1.size()) {
        std::cerr << "No valid matches to draw!" << std::endl;
        return cv::Mat();
    }
    // 创建拼接图像：左右并排
    int h = std::max(img0.rows, img1.rows);
    int w = img0.cols + img1.cols;
    Mat canvas(h, w, CV_8UC3, Scalar(0, 0, 0));
    // 复制图像到画布
    img0.copyTo(canvas(Rect(0, 0, img0.cols, img0.rows)));
    img1.copyTo(canvas(Rect(img0.cols, 0, img1.cols, img1.rows)));

    // 颜色设置
    cv::RNG rng(12345); // 固定随机种子，颜色可复现
    vector<Scalar> colors(kpts0.size());
    for (size_t i = 0; i < kpts0.size(); ++i) {
        colors[i] = Scalar(rng.uniform(100, 255), rng.uniform(100, 255), rng.uniform(100, 255));
    }
    // 绘制匹配线和关键点
    for (size_t i = 0; i < kpts0.size(); ++i) {
        Point2i pt0 = kpts0[i];
        Point2i pt1 = Point2i(kpts1[i].x + img0.cols, kpts1[i].y); // x offset by img0 width
        // 跳出边界的点不画（安全检查）
        if (pt0.x < 0 || pt0.y < 0 || pt0.x >= img0.cols || pt0.y >= img0.rows) continue;
        if (pt1.x < img0.cols || pt1.y < 0 || pt1.x >= w || pt1.y >= h) continue;
        // 画连线
        cv::line(canvas, pt0, pt1, colors[i], 1, cv::LINE_AA);
        // 画关键点（小圆）
        cv::circle(canvas, pt0, 3, colors[i], -1, cv::LINE_AA);
        cv::circle(canvas, pt1, 3, colors[i], -1, cv::LINE_AA);
    }
    // 添加文字：匹配数量
    std::string text = "Matches: " + std::to_string(kpts0.size());
    putText(canvas, text, Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    return canvas;
}


void saveMatchesToFile(const std::vector<cv::Point2i>& kpts0_match, const std::vector<cv::Point2i>& kpts1_match, const std::string& filePath) {
    std::ofstream outFile(filePath);

    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing: " << filePath << std::endl;
        return;
    }

    // 确保两个向量大小相同
    if (kpts0_match.size() != kpts1_match.size()) {
        std::cerr << "Key points vectors size mismatch!" << std::endl;
        return;
    }

    // 写入匹配点对到文件
    for (size_t i = 0; i < kpts0_match.size(); ++i) {
        outFile << "Match " << i + 1 << ": "
                << "Image 1 (" << kpts0_match[i].x << ", " << kpts0_match[i].y << "), "
                << "Image 2 (" << kpts1_match[i].x << ", " << kpts1_match[i].y << ")"
                << std::endl;
    }

    outFile.close();
}


int main(int argc, char** argv)
{
    int ret;
    Mat show_img;
    LightGlueMatches results;

    // 默认模型路径（可选）
    std::string model_superpoint = "model/mysuperpoint_512_512_rk3588.rknn";
    std::string model_lightglue  = "model/mylightglue_512_512_rk3588.rknn";

    // 解析命令行参数
    if (argc == 3) {
        model_superpoint = argv[1];
        model_lightglue  = argv[2];
    } else if (argc == 1) {
        std::cout << "Using default models.\n";
    } else {
        std::cerr << "Usage: " << argv[0]
                  << " [superpoint_model.rknn] [lightglue_model.rknn]\n";
        return -1;
    }

    std::cout << "SuperPoint model: " << model_superpoint << "\n";
    std::cout << "LightGlue model:  " << model_lightglue << "\n";

    Matches match(model_superpoint.c_str(), model_lightglue.c_str());

#if defined(IMAGE_DEMO)
    Mat img1 = cv::imread("temp_cropped.jpg");
    Mat img2 = cv::imread("rot_60_scale_2.0_cropped.jpg");
    if(img1.empty() || img2.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    /* -------加载输入文件，这里对上了------- */
//     int H = 512, W = 512, C = 1;
//     size_t input_size = H * W * C;
//     std::vector<float> input1, input2;
//     if (!match.load_float_data("input1.bin", input1, input_size) ||
//         !match.load_float_data("input2.bin", input2, input_size)) {
//         std::cerr << "Failed to load inputs\n";
//         return -1;
//     }
//     printf("input1 size = %d\n", input1.size());
//
//     auto result1 = match.superpoint_infer(input1);
//     auto result2 = match.superpoint_infer(input2);

    /* -------加载输入文件，这里对上了------- */

    /* -------测试图像加载------- */
    // 先不管前处理
    Mat img_data1 = match.data_process(img1, true);
    Mat img_data2 = match.data_process(img2, false);

    auto start = chrono::steady_clock::now();
    auto result1 = match.superpoint_infer(img_data1);
    auto end = chrono::steady_clock::now();
    auto tt = chrono::duration_cast<microseconds>(end - start);
    cout << "superpoint_infer run time is = " << tt.count() << "us" << endl;

    auto result2 = match.superpoint_infer(img_data2);
    /* -------测试图像加载------- */

    vector<Point2f>& keypoints0 = result1.first;
    Mat& desc0 = result1.second;
    vector<Point2f>& keypoints1 = result2.first;
    Mat& desc1 = result2.second;

    auto kpts_normalized0 = normalizeKeypoints(keypoints0, 512, 512);
    auto kpts_normalized1 = normalizeKeypoints(keypoints1, 512, 512);

    start = chrono::steady_clock::now();
    results = match.lightglue_infer(kpts_normalized0, desc0, kpts_normalized1, desc1);
    end = chrono::steady_clock::now();
    tt = chrono::duration_cast<microseconds>(end - start);
    cout << "lightglue_infer run time is = " << tt.count() << "us" << endl;

    //saveMatchesToFile(results.kpts0_orig, results.kpts1_orig, "matches.txt"); // 保存匹配坐标到文件
    // 画图
    if (!results.kpts0_orig.empty()) {
        show_img = drawLightGlueMatches(img1, img2, results.kpts0_orig, results.kpts1_orig);
        // 保存
        cv::imwrite("matches_lightglue.png", show_img);
        std::cout << "Saved match visualization to: " << "matches_lightglue.png" << std::endl;

    }
    else {
        std::cout << "No matches found!" << std::endl;
    }

#else

    Mat frame, image;
    Rect box;
    VideoCapture capture(41);
    if (!capture.isOpened()) {
        cout << "file load is failed" << endl;
        return -1;
    }

    capture.read(frame);
    //cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    box = selectROI("frame", frame, true);
    //box = Rect(391, 286, 140, 86);
    image = frame(box).clone();

    Mat img_data1 = match.data_process(image, true);
    auto result1 = match.superpoint_infer(img_data1);
    vector<Point2f>& keypoints0 = result1.first;
    Mat& desc0 = result1.second;
    auto kpts_normalized0 = normalizeKeypoints(keypoints0, 512, 512);

    Mat img_data2;

    vector<Point2f> keypoints1;
    Mat desc1;
    vector<Point2f> kpts_normalized1;

    while (capture.read(frame)) {
        auto start = chrono::steady_clock::now();

        img_data2 = match.data_process(frame, false);
        auto result2 = match.superpoint_infer(img_data2);
        keypoints1 = result2.first;
        desc1 = result2.second;
        kpts_normalized1 = normalizeKeypoints(keypoints1, 512, 512);

        results = match.lightglue_infer(kpts_normalized0, desc0, kpts_normalized1, desc1);

        auto end = chrono::steady_clock::now();
        auto tt = chrono::duration_cast<microseconds>(end - start);
        cout << "run time is = " << tt.count() << "us" << endl;

        show_img = drawLightGlueMatches(image, frame, results.kpts0_orig, results.kpts1_orig);
        imshow("Image", show_img);
        char key = waitKey(20);
        if (key == 'q')
        {
            break;
        }
    }
    capture.release();

#endif
    return 0;
}
