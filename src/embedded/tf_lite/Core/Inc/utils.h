#include <queue>
#include <iostream>
#ifndef UTILS_H
#define UTILS_H
#ifdef __cplusplus
extern "C" {
#endif


class SimpleMovingAverage 
{

public:
    /** Initialize your data structure here. */
    
    SimpleMovingAverage(int window_size);
    
    float next(int val);

private:
    unsigned int window_size_;
    int sum_ = 0.0;
    std::queue<int> q_;


};

SimpleMovingAverage::SimpleMovingAverage(int window_size)   
                                        : window_size_(window_size) {}

float SimpleMovingAverage::next(int val) {
        
    if (q_.size() == window_size_-1) {
        sum_ += val;
        q_.emplace(val);
        //std::cout<< "sum:"<<sum_<<std::endl;
        //std::cout<< "window_size:"<<q_.size()<<std::endl;
        float sma = sum_ / q_.size();
        sum_ -= q_.front();
        //std::cout<< q_.front()<<": removed"<<std::endl;
        q_.pop();
        
        //std::cout<< q_.back()<<": added"<<std::endl;
        
        //std::cout<< "SMA:"<<sma<<std::endl;
        return sma;
    }
    else{
        //std::cout<< "---------------------------------------" <<std::endl;
        q_.emplace(val);    
        sum_ += val;
        return 0.0;
    }


    }



#ifdef __cplusplus
}
#endif

#endif