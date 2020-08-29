#include <dv-sdk/module.hpp>
#include <dv-sdk/processing.hpp>
#include <opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "pinball.hpp"
#include <iostream>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <opencv2/core/eigen.hpp>


using namespace Eigen;


class FeastDV : public dv::ModuleBase {
private:
	// user selectable refractory period in microseconds
	int n_neurons;
	float thresholdClose;
	float thresholdOpen;
	float eta;
	float tau;
	int radius;
	uint64_t  lastFrame = 0;
    uint64_t  displayFreq = 1e6;


    int rows;
	int cols;

	MatrixXf w;
//	Matrix<uint64_t,Dynamic,Dynamic> lastTime;
    MatrixXf thresholds;
    MatrixXf roiSurface;
    MatrixXf dotProduct;
//    MatrixXf outputFrame;
//    long long r;


//	cv::Mat w;
//	cv::Mat threshold;
	dv::TimeMat lastFiringTimes;
//	cv::Mat roiSurface;
//    cv::Mat dotProduct;
public:
	static void initInputs(dv::InputDefinitionList &in) {
		in.addEventInput("events");
	}

	static void initOutputs(dv::OutputDefinitionList &out) {
		out.addEventOutput("events");
//		out.addFrameOutput("frames");
	}

	static const char *initDescription() {
		return ("The Feast Layer Module");
	}

	static void initConfigOptions(dv::RuntimeConfig &config) {
		config.add("n_neurons",
			dv::ConfigOption::intOption("Number of neurons", 9, 2, 100));
        config.add("thresholdClose",
                   dv::ConfigOption::floatOption("Threshold Close", 0.01, 0, 1));
        config.add("thresholdOpen",
                   dv::ConfigOption::floatOption("Threhsold Open", 0.01, 0, 1));
        config.add("eta",
                   dv::ConfigOption::floatOption("Eta", 0.001, 0, 1));
        config.add("tau",
                   dv::ConfigOption::floatOption("Tau", 1000, 1, 2000));
        config.add("roi_x",
                   dv::ConfigOption::intOption("ROI X", 10, 2, 100));
        config.add("roi_y",
                   dv::ConfigOption::intOption("ROI Y", 10, 2, 100));
        config.add("radius",
                   dv::ConfigOption::intOption("ROI radius", 4, 2, 100));



//        config.setPriorityOptions({"tau"});
	}

	FeastDV() : n_neurons(9), thresholdClose(0.01), thresholdOpen(0.02), eta(0.01), tau(10000),radius(4),lastFiringTimes(inputs.getEventInput("events").size()) {
		outputs.getEventOutput("events").setup(1,n_neurons,"Output feast events");
//        cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE );
        rows = inputs.getEventInput("events").sizeX();
        cols = inputs.getEventInput("events").sizeY();
        w = MatrixXf::Random((2*radius+1)*(2*radius+1),n_neurons);
        w.colwise().normalize();

        roiSurface = MatrixXf::Zero(1, (2*radius+1)*(2*radius+1));
        thresholds = MatrixXf::Zero(1, n_neurons);
        dotProduct = MatrixXf::Zero(1, n_neurons);

//        double guess = sqrt(n_neurons);
//        r = std::floor(guess);
//        if(r*r == n_neurons){
//            outputFrame = MatrixXf::Zero((2*radius+1)*r, (2*radius+1)*r);
//        }
//        else {
//            r = std::ceil(guess);
//            outputFrame = MatrixXf::Zero((2*radius+1)*r, (2*radius+1)*r);
//        }
//
//        outputs.getFrameOutput("frames").setup((2*radius+1)*r,(2*radius+1)*r,"Output feast events");
//





//


	}

	void run() override {
		auto input  = inputs.getEventInput("events");
		auto output = outputs.getEventOutput("events");
//		auto frameOutput = outputs.getFrameOutput("frames");

        int event_x;
        int event_y;


        for (const auto &event : input.events()){
            event_x = event.x();
            event_y = event.y();
            lastFiringTimes.at(event.y(), event.x()) = event.timestamp();
            int winnerNeuron = -1;
            int maxDotProduct = 0;
            if(event.x()-radius < 0 || event.x()+radius > rows -1 || event.y()-radius < 0 || event.y()+radius > cols-1){
                continue;
            } else{

                for(int x=event_x-radius;x<=event_x+radius;x++){
                    for(int y=event_y-radius;y<=event_y+radius;y++){
                       roiSurface(0,(x-event_x+radius)*radius + y-event_y+radius) = static_cast<float>(expf(
                               -(static_cast<float>(event.timestamp() - lastFiringTimes.at(y,x))) / tau));
                    }
                }
                roiSurface.normalize();

                dotProduct = roiSurface*w;

                for(int n = 0; n < n_neurons; n++){
                    if(dotProduct(0,n) > maxDotProduct && dotProduct(0,n)  > thresholds(0,n)){
                        winnerNeuron = n;
                        maxDotProduct = dotProduct(0,n);
                    }
                }
                if (winnerNeuron > -1){
                    w.col(winnerNeuron) = w.col(winnerNeuron)*(1-eta) + roiSurface.transpose()*eta;
                    w.col(winnerNeuron).normalize();
//                    w_norm = cv::norm(w.col(winnerNeuron),cv::NormTypes::NORM_L2);
//                    w.col(winnerNeuron) /= w_norm;
//                    cv::normalize(w.col(winnerNeuron), w.col(winnerNeuron));
                    thresholds(0,winnerNeuron) += thresholdClose;
                    output << dv::Event(event.timestamp(), 0,winnerNeuron,true);
                }else{
                    thresholds = thresholds - MatrixXf::Constant(1, n_neurons, thresholdOpen);
                }

            }
//            cv::Mat outFrame = cv::Mat((2*radius+1)*r, (2*radius+1)*r,CV_32F,cv::Scalar(0.0));
//            cv::Mat featureMapCV;// = cv::Mat((2*radius+1), (2*radius+1),CV_32F,cv::Scalar(0.0));
//
//
//            if (event.timestamp() - lastFrame > displayFreq) {
//                for(int i = 0; i<n_neurons; i++){
//                    Matrix<float, Dynamic, Dynamic> feature = Map<const Matrix<float, Dynamic,Dynamic>>(w.col(winnerNeuron).data(), 2*radius+1, 2*radius+1);
//                    cv::eigen2cv(feature, featureMapCV);
//                    featureMapCV.copyTo(outFrame(cv::Rect(static_cast<int>(i/r)*(2*radius+1),
//                                                          static_cast<int>(i%r)*(2*radius+1),
//                                                          static_cast<int>(2*radius+1),static_cast<int>(2*radius+1))));
////                    outputFrame.block(static_cast<int>(i/r)*(2*radius+1),
////                                      static_cast<int>(i%r)*(2*radius+1),
////                                      static_cast<int>(2*radius+1),static_cast<int>(2*radius+1)) = Map<const Matrix<float, Dynamic,Dynamic>>(w.col(winnerNeuron).data(), 2*radius+1, 2*radius+1);
//                }
//
////                cv::eigen2cv(outputFrame,outFrame);
//
//                frameOutput<<event.timestamp()<<outFrame<<dv::commit;
//                lastFrame = event.timestamp();
//
//            }
		}
        output << dv::commit;

	}

	void configUpdate() override {
        n_neurons = config.getInt("n_neurons");
        tau = config.getFloat("tau");
        thresholdOpen = config.getFloat("thresholdOpen");
        thresholdClose= config.getFloat("thresholdClose");
        radius = config.getInt("radius");
        outputs.getEventOutput("events").setup(1,n_neurons,"Output feast events");
        rows = inputs.getEventInput("events").sizeX();
        cols = inputs.getEventInput("events").sizeY();

        w.resize(0,0);
        thresholds.resize(0,0);
        roiSurface.resize(0,0);
        dotProduct.resize(0,0);

        w = MatrixXf::Random((2*radius+1)*(2*radius+1),n_neurons);
        w.colwise().normalize();

        roiSurface = MatrixXf::Zero(1, (2*radius+1)*(2*radius+1));
        thresholds = MatrixXf::Zero(1, n_neurons);
        dotProduct = MatrixXf::Zero(1, n_neurons);

//        outputFrame.resize(0,0);
//
//        double guess = sqrt(n_neurons);
//        r = std::floor(guess);
//        if(r*r == n_neurons){
//            outputFrame = MatrixXf::Zero((2*radius+1)*r, (2*radius+1)*r);
//        }
//        else {
//            r = std::ceil(guess);
//            outputFrame = MatrixXf::Zero((2*radius+1)*r, (2*radius+1)*r);
//        }
//        outputs.getFrameOutput("frames").setup((2*radius+1)*r,(2*radius+1)*r,"Output feast events");
//

    }
};

registerModuleClass(FeastDV)
