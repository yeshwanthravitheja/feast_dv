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
#include <opencv2/imgproc.hpp>


using namespace Eigen;


class FeastDV : public dv::ModuleBase {
private:
	// user selectable refractory period in microseconds
	int n_neurons;
	float thresholdClose;
	float thresholdOpen;
	float eta;
	long tau;
	int radius;
	uint64_t  lastFrame = 0;
    uint64_t  displayFreq = 1e6;




	MatrixXf w;
    MatrixXf thresholds;
    MatrixXf roiSurface;
    MatrixXf dotProduct;
    MatrixXf outputFrame;
    long long r;

	dv::TimeMat lastFiringTimes;

public:
	static void initInputs(dv::InputDefinitionList &in) {
		in.addEventInput("events");
	}

	static void initOutputs(dv::OutputDefinitionList &out) {
		out.addEventOutput("events");
		out.addFrameOutput("frames");
	}

	static const char *initDescription() {
		return ("The Feast Layer Module");
	}

	static void initConfigOptions(dv::RuntimeConfig &config) {
		config.add("n_neurons",
			dv::ConfigOption::intOption("Number of neurons", 4, 2, 100));
        config.add("thresholdClose",
                   dv::ConfigOption::floatOption("Threshold Close", 0.001, 0, 1));
        config.add("thresholdOpen",
                   dv::ConfigOption::floatOption("Threhsold Open", 0.0005, 0, 1));
        config.add("eta",
                   dv::ConfigOption::floatOption("Eta", 0.005, 0, 1));
        config.add("tau",
                   dv::ConfigOption::longOption("Tau", 1 * 5e3, 4e3, 1e4));
//        config.add("roi_x",
//                   dv::ConfigOption::intOption("ROI X", 10, 2, 100));
//        config.add("roi_y",
//                   dv::ConfigOption::intOption("ROI Y", 10, 2, 100));
        config.add("radius",
                   dv::ConfigOption::intOption("ROI radius", 1, 1, 100));
        config.add("displayFreq",
                   dv::ConfigOption::longOption("display Frequency", 1e6, 5*1e4, 1e7));
        config.setPriorityOptions({"n_neurons", "radius",
                                   "displayFreq", "tau",
                                   "eta","thresholdClose",
                                   "thresholdOpen"});


//        config.setPriorityOptions({"tau"});
	}

	FeastDV() : n_neurons(9), thresholdClose(0.01), thresholdOpen(0.02), eta(0.01), tau(10000),radius(4),lastFiringTimes(inputs.getEventInput("events").size()) {

        w = MatrixXf::Random((2*radius+1)*(2*radius+1),n_neurons);
        w.colwise().normalize();

        roiSurface = MatrixXf::Zero(1, (2*radius+1)*(2*radius+1));
        thresholds = MatrixXf::Zero(1, n_neurons);
        dotProduct = MatrixXf::Zero(1, n_neurons);

        double guess = sqrt(n_neurons);
        r = std::floor(guess);
        if(r*r == n_neurons){
            outputFrame = Matrix<float, Dynamic,Dynamic>::Zero((2*radius+1)*r, (2*radius+1)*r);
        }
        else {
            r = std::ceil(guess);
            outputFrame = Matrix<float, Dynamic,Dynamic>::Zero((2*radius+1)*r, (2*radius+1)*r);
        }

//        dvModuleRegisterOutput(moduleData, "frames", "REMOVE");

        outputs.getFrameOutput("frames").setup(650,650,"Output feast frames");
        outputs.getEventOutput("events").setup(r,r,"Output feast events");

//





//


	}

	void run() override {
		auto input  = inputs.getEventInput("events");
		auto output = outputs.getEventOutput("events");


        int event_x;
        int event_y;
//        IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");


        for (const auto &event : input.events()){
            event_x = event.x();
            event_y = event.y();
            lastFiringTimes.at(event.y(), event.x()) = event.timestamp();
            int winnerNeuron = -1;
            int maxDotProduct = 0;
            if(event.x()-radius < 0 || event.x()+radius > input.sizeX() -1 || event.y()-radius < 0 || event.y()+radius > input.sizeY()-1){
                continue;
            } else{
                for(int x=0;x<2*radius+1;x++){
                    for(int y=0;y<2*radius+1;y++){
                        roiSurface(0,x*(2*radius+1) + y) = static_cast<float>(expf(
                                -(static_cast<float>(event.timestamp() - lastFiringTimes.at(y+event_y-radius,x+event_x-radius))) / tau));
                    }
                }
//                std::cout<<std::endl<<std::endl;

//                for(int x=event_x-radius;x<=event_x+radius;x++){
//                    for(int y=event_y-radius;y<=event_y+radius;y++){
//                       roiSurface(0,(x-event_x+radius)*radius + y-event_y+radius) = static_cast<float>(expf(
//                               -(static_cast<float>(event.timestamp() - lastFiringTimes.at(y,x))) / tau));
//                    }
//                }
//                std::cout<<roiSurface.format(CleanFmt)<<std::endl;

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
//                    std::cout<<roiSurface.format(CleanFmt)<<std::endl;
                    w.col(winnerNeuron).normalize();
//                    std::cout<<w.col(winnerNeuron).format(CleanFmt)<<std::endl;

                    thresholds(0,winnerNeuron) += thresholdClose;
                    output << dv::Event(event.timestamp(), static_cast<int>(winnerNeuron%r),static_cast<int>(winnerNeuron/r),true);
                }else{
                    thresholds = thresholds - MatrixXf::Constant(1, n_neurons, thresholdOpen);
                }

            }

            if (event.timestamp() - lastFrame > displayFreq) {
                auto frameOutput = outputs.getFrameOutput("frames");
                auto outFrame = frameOutput.frame();
                outFrame.setTimestamp(event.timestamp());
                cv::Mat outFrameCV;// = cv::Mat::zeros(cv::Size((2*radius+1)*r,(2*radius+1)*r), CV_32F);
                cv::Mat outFrameResizedCV;

                for(int i = 0; i<n_neurons; i++){
//                    cv::Mat feature;
//                    MatrixXf featureEigen = Map<const Matrix<float, Dynamic,Dynamic>>(w.col(i).data(), 2*radius+1, 2*radius+1);
//                    cv::eigen2cv(featureEigen,feature);
////                    std::cout << "M = " << std::endl << " " << feature << std::endl << std::endl;
//                    feature.copyTo(outFrameCV.colRange(static_cast<int>(i%r)*(2*radius+1),
//                                                       static_cast<int>(2*radius+1+(i%r)*(2*radius+1))).rowRange(static_cast<int>(i/r)*(2*radius+1),static_cast<int>(2*radius+1+(i/r)*(2*radius+1)) ));

                    outputFrame.block(static_cast<int>(i/r)*(2*radius+1),
                                      static_cast<int>(i%r)*(2*radius+1),
                                      static_cast<int>(2*radius+1),static_cast<int>(2*radius+1)) = Map<const Matrix<float, Dynamic,Dynamic>>(w.col(i).data(), 2*radius+1, 2*radius+1);

                }

                cv::eigen2cv(outputFrame,outFrameCV);
                cv::resize(outFrameCV, outFrameResizedCV, cv::Size(650, 650), 0,0, cv::INTER_NEAREST);
                cv::normalize(outFrameResizedCV, outFrameResizedCV, 0, 255, cv::NORM_MINMAX);
                outFrameResizedCV.convertTo(outFrameResizedCV, CV_8UC1);
                outFrame.setMat(outFrameResizedCV);
                outFrame.commit();


//                frameOutput<<event.timestamp()<<outFrameCV<<dv::commit;
                lastFrame = event.timestamp();

            }
		}
        output << dv::commit;

	}

	void configUpdate() override {
        tau = config.getLong("tau");
        thresholdOpen = config.getFloat("thresholdOpen");
        thresholdClose= config.getFloat("thresholdClose");
        displayFreq = config.getLong("displayFreq");

        if(n_neurons!=config.getInt("n_neurons") || radius!=config.getInt("radius")){
            n_neurons = config.getInt("n_neurons");
            radius = config.getInt("radius");
            w.resize(0,0);
            thresholds.resize(0,0);
            roiSurface.resize(0,0);
            dotProduct.resize(0,0);
            outputFrame.resize(0,0);

            w = MatrixXf::Random((2*radius+1)*(2*radius+1),n_neurons);
            w.colwise().normalize();

            roiSurface = MatrixXf::Zero(1, (2*radius+1)*(2*radius+1));
            thresholds = MatrixXf::Zero(1, n_neurons);
            dotProduct = MatrixXf::Zero(1, n_neurons);
            double guess = sqrt(n_neurons);
            r = std::floor(guess);
            if(r*r == n_neurons){
                outputFrame = Matrix<float, Dynamic,Dynamic>::Zero((2*radius+1)*r, (2*radius+1)*r);
            }
            else {
                r = std::ceil(guess);
                outputFrame = Matrix<float, Dynamic,Dynamic>::Zero((2*radius+1)*r, (2*radius+1)*r);
            }
            outputs.getEventOutput("events").setup(r,r,"Output feast events");



        }







    }
};

registerModuleClass(FeastDV)
