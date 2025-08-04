# HDR Exposure

There is no accepted solution for auto exposure algorithms that are agnostic of fluorescence marker properties, marker patterns and all environmental factors such as antibody-fluorophore aggregates. All these variables make it extremely difficult to home in exact exposure time. An alternate strategy is to vastly increase the dynamic range of the image. This makes it so almost no matter the actual fluorescence properties of the channel, I will almost certainly have taken the image with sufficient, non-saturated exposure at some point. This strategy is called the HDR auto exposure (Brinkmann, Eva-Maria, et al. "Advanced high dynamic range fluorescence microscopy with Poisson noise modeling and integrated edge-preserving denoising." Journal of Physics Communications 5.7 (2021): 075016.). Conceptually our approach is very similar, but we differ in a couple areas of execution. Namely, we extrapolate intensities to highest exposure time taken and use simplified curves based on relative extrapolation errors and Poisson error.


<!-- You can drag and drop your images directly here
![](images/HDR_demo001.png)

e and use this template -->
![](images/<Your Image>.jpg)
![](images/<Your Image>.jpg)
![](images/<Your Image>.jpg)




* [.](testpage1.md){step}
* [.](testpage2.md){step}

