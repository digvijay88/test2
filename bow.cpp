// Soni Nishitkumar Hiteshkumar 201002026
// Digvijay Singh 201002052
// Bag of words (surf)

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include<string>
#include<fstream>
#include<cstring>
#include<ctime>
#include<map>
#include "opencv2/legacy/legacy.hpp"
using namespace cv;
using namespace std;


int main(){
    int words = 12;
    vector<string>files;
    Mat img;
    Mat descriptors;
    Mat document_vector;
    map<string,int>Intersection;
    vector<string> Similar;
    map<string,int>file_index;
    map<string,double>Score;
    vector<vector<string> > inverted_index;
    map<string,int> :: iterator it;
    SurfFeatureDetector q_detector(400);
    SurfFeatureDetector detector(400);
    SurfDescriptorExtractor* extractor = new SurfDescriptorExtractor;
    vector<KeyPoint> keypoints;
    vector<Mat> document_vectors;
    vector<string>temp2;
    Mat image;
    Mat normalized;
    Mat q_descriptors;
    Mat normalized_document_vector;
    vector<pair<double,string> >SCORES;
    FILE * fp;
    char line[200];
    int ind;
    double score;
    fp = fopen("files", "r");
    BOWKMeansTrainer bowtrainer(words); //num clusters
    ind = 0;
    while(fgets(line, sizeof(line), fp) != NULL) {
        line[strlen(line)-1] = '\0';
        string temp(line);
        string path = "Images/" + temp;
        files.push_back(path);
        img = imread(path);
        if(!img.data) continue;
        detector.detect(img, keypoints);
        extractor->compute(img, keypoints,descriptors);
        bowtrainer.add(descriptors);
        file_index[path] = ind;
        ind++;
    }
    fclose(fp);
    Mat vocabulary = bowtrainer.cluster();
    DescriptorMatcher * matcher = new BFMatcher(NORM_L2,false);
    BOWImgDescriptorExtractor * bowide = new BOWImgDescriptorExtractor(extractor,matcher);
    bowide->setVocabulary(vocabulary);
    // document vector calculation
    fp = fopen("files", "r");
    while(fgets(line, sizeof(line), fp) != NULL) {
        line[strlen(line)-1] = '\0';
        string temp(line);
        string path = "Images/" + temp;
        img = imread(path);
        detector.detect(img, keypoints);
        if(keypoints.size() == 0)
            continue;
        bowide->compute(img, keypoints, document_vector);
        document_vectors.push_back(document_vector);
    }
    fclose(fp);
    // inverted index calculation
    for(int i = 0; i < words; i++) {
        for(int j = 0; j < document_vectors.size(); j++) {
            if(document_vectors[j].at<double>(i) > 0) {
                string fname = files[j];
                temp2.push_back(fname);
            }
        }
        inverted_index.push_back(temp2);
        temp2.clear();
    }
    // BOW queries
    fp = fopen("test_images", "r");
    int positives=0;
    int negatives=0;
    while(fgets(line, sizeof(line), fp) != NULL) {
        keypoints.clear();
        SCORES.clear();
        Similar.clear();
        Intersection.clear();
        line[strlen(line)-1] = '\0';
        string temp(line);
        string path = "test/" + temp;
        image = imread(path);
        if(!image.data) {
            cout << "empty image" << endl;
            continue;
        }
//        cout << path << endl;
        q_descriptors;
        q_detector.detect(image, keypoints);
        bowide->compute(image, keypoints, document_vector);
//        cout << document_vector.size().height << " " << document_vector.size().width << endl;
        if(document_vector.size().height == 0) continue;
        int hist_words = 0;
        for(int i = 0; i < words; i++) {
            if(document_vector.at<double>(i) > 0) {
                for(int j = 0; j < inverted_index[i].size(); j++) 
                    Intersection[inverted_index[i][j]]+=1;
                hist_words++;
            }
        }
        for(it = Intersection.begin(); it != Intersection.end(); it++) {
            if(it->second >= hist_words/2) {  
                Similar.push_back(it->first);
            }
        }
        normalize(document_vector, normalized_document_vector, 1, 0, NORM_L2, -1, Mat());
        for(int i = 0; i < Similar.size(); i++) {
            int idx = file_index[Similar[i]];
            document_vector = document_vectors[idx];
            normalize(document_vector, normalized, 1, 0, NORM_L2,-1, Mat());
            score = normalized.dot(normalized_document_vector);
            SCORES.push_back(make_pair(score, Similar[i]));
        }
        sort(SCORES.begin(), SCORES.end());
        int MATCH = 0;
        for(int i = SCORES.size()-1; i >= SCORES.size()-8 && i >= 0 && MATCH == 0; i--) {
    //        cout << SCORES[i].second.substr(7,3) << " " << temp.substr(0,3) << endl;
            if(SCORES[i].second.substr(7,3) == temp.substr(0,3)) {
                MATCH = 1;
                break;
            }
        }
        if(MATCH) positives++;
        else negatives++;
//        cout << "done " << endl;
    }
    cout << positives+negatives << endl;
    cout << positives<< " " << negatives << endl;
    fclose(fp);
    return 0;
}
