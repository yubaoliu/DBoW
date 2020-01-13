#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace DBoW2;
using namespace std;
using namespace cv;

void loadFeatures(const std::vector<cv::Mat> &t_vImage, std::vector<std::vector<cv::Mat>> &t_vFeatures);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat>> &features);

int main(int argc, char **argv)
{
    string dataset_dir = argv[1];
    ifstream fin(dataset_dir + "/associations.txt");
    if (!fin)
    {
        //Example: ./demo_txt /home/yubao/data/Dataset/TUM/freiburg3/rgbd_dataset_freiburg3_sitting_static
        cout << "please generate the associate file called associations.txt!" << endl;
        return 1;
    }

    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while (!fin.eof())
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file;
        rgb_times.push_back(atof(rgb_time.c_str()));
        depth_times.push_back(atof(depth_time.c_str()));
        rgb_files.push_back(dataset_dir + "/" + rgb_file);
        depth_files.push_back(dataset_dir + "/" + depth_file);

        if (fin.good() == false)
            break;
    }
    fin.close();

    cout << "generating features ... " << endl;

    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    std::vector<cv::Mat> images;
    images.reserve(rgb_files.size());

    for (uint i = 0; i < rgb_files.size(); i++)
    {
        string rgb_file = rgb_files[i];
        std::cout << "image path: " << rgb_file << std::endl;
        Mat image = imread(rgb_file, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            std::cout << "Empty image: " << i << std::endl;
            continue;
        }
        images.push_back(image);
    }

    std::vector<std::vector<cv::Mat>> features;

    loadFeatures(images, features);
    testVocCreation(features);

    cout << "done" << endl;

    return 0;
}

void loadFeatures(const std::vector<cv::Mat> &t_vImage, std::vector<std::vector<cv::Mat>> &t_vFeatures)
{
    int IMAGES = t_vImage.size();
    std::cout << "Size of images: " << IMAGES << std::endl;

    t_vFeatures.clear();
    t_vFeatures.reserve(IMAGES);

    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    cout << "Extracting ORB features..." << endl;
    for (int i = 0; i < IMAGES; ++i)
    {
        cv::Mat image = t_vImage[i];
        if (image.empty())
        {
            std::cout << "[Error] Image is empty: " << std::endl;
            continue;
        }

        cv::Mat mask;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb->detectAndCompute(image, mask, keypoints, descriptors);
        t_vFeatures.emplace_back(vector<cv::Mat>());
        changeStructure(descriptors, t_vFeatures.back());
    }
}

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
    out.resize(plain.rows);

    for (int i = 0; i < plain.rows; ++i)
    {
        out[i] = plain.row(i);
    }
}

void testVocCreation(const vector<vector<cv::Mat>> &features)
{
    // int NIMAGES = features.size();
    // branching factor and depth levels
    const int k = 10;
    //TODO  L=6 not work
    const int L = 6;

    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;

    OrbVocabulary voc(k, L, weight, score);

    cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl
         << endl;

    // lets do something with this vocabulary
    // cout << "Matching images against themselves (0 low, 1 high): " << endl;
    // BowVector v1, v2;
    // for (int i = 0; i < NIMAGES; i++)
    // {
    //     voc.transform(features[i], v1);
    //     for (int j = 0; j < NIMAGES; j++)
    //     {
    //         voc.transform(features[j], v2);

    //         double score = voc.score(v1, v2);
    //         cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    //     }
    // }

    // save the vocabulary to disk
    cout << endl
         << "Saving vocabulary..." << endl;
    // voc.save("small_voc.yml.gz");
    voc.saveToTextFile("voc.txt");
    cout << "Done" << endl;
}
