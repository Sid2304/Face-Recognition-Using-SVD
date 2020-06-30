## Face-Recognition-Using-SVD

The idea of this project was to create a representative image - out of multiple sample facial immages of person - and which will be used for facial regognitoin purpose.

Methodology: We extract the most significant characteristics from all the sample images of a person, that best describes the face. We call this out representative image.
             We find a representative image for all the subjects and define a metric of closeness (error) between two vectors
             
             In this case we use the L2 Norm or the euclidian distance between the vectors defined in he 'pixel space', i.e., we find the sum of square of the difference 
             between the values at each pixel coordinate of the two images.
             
             To classify an input image, we find its L2 norm with all of the representative images, and the one with the least value is the class of the input image
             
Concept: We use Singular Value Decomposition to extract the characteristic features from the 'face - space' matrix - a matrix formed by stacking up all the sample images
         of a subject 

Procedure: 
 Convert each image matrix into a column vector by flattening, i.e. we place all the 64 columns of the matrix one below the other. The rationale behind collapsing the 
   matrix is that the required value of the pixels is independent of their relative positions w.r.t each other. 
 Stack the 10 column vectors for a particular subject such that we obtain a (4096 x 10) matrix, which contains data of all the 10 images for that particular subject. 
   Let’s call this data matrix A.  
 Mean-shift A matrix to get accurate results while applying SVD. 
 Apply SVD on (4096 x 10) mean-shifted A matrix to obtain U, 𝚺, and VT matrices of sizes (4096 x 4096), (4096 x 10), and (10 x 10) respectively. U matrix contains the
   eigenvectors of AAT, 𝚺 contains the singular values of ATA arranged diagonally and VT contains the eigenvectors of ATA. 
   
Extracting characterisitc features:

 The vector obtained above is the representative image as the product U1*𝚺1*VT1 projects the data in a 1dimensional subspace that captures maximum variance in the data
   in Total Least Square (TLS) sense. 
 Intuitively, SVD extracts the most important characteristic features from all 10 images and combines them to form a representative image. 
 Add the mean values back and transform the vector to (64 x 64) matrix to obtain the representative image. 
 
 
