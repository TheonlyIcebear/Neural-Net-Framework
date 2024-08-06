# Neural-Net-Framework 🧠
I custom library I made for training neural networks from scratch, using numpy and scipy


# Imformation ℹ

I've been working on many neural network projects for the past year now, with my earliest project finished in April 2024

And I've learned a lot through much trial and error in the Machine Learning field and my end goal is to create a boudning box regression model similiar to `yolo-v1`.

### Purpose:

 - I noticed my old code was really limiting and innefecient:
 
   - https://github.com/TheonlyIcebear/Image-Recognition-AI
   - https://github.com/TheonlyIcebear/Tic-Tac-Toe-Machine-Learnin

 - I decided to do a complete rewrite allowing my code to be more modular much like keras' `Sequential` class allowing me to quickly add features like `Batch Normalization`


Image classification problem training loss:<br>
![Image](image.png)

# Features ⚙
 - The model is completely sequential, meaning the output from one layer will only go to a single next layer

 - Performs a 2d convolution, good for feature extraction in images

It includes, `Convolutional Layers`, `Dense Layers`, `Batch Normalization` and `Max Pooling`

# Notes

 - This is **NOT** a replacement for actual Neural Network libraries such as `Tensorflow` or `PyTorch`, this is simply a library i made because I want to understand neural networks on a deep level.

- If you value your time, please don't use this on serious projects lol

 - For some reason (aka im too dumb to fix it) I cant get the custom striding to work but I decided to leave it in anyways

 - This is CPU bound only meaning it is really slow compared to other GPU based models, I plan to add modules such as `CuPy` once I get a Nividia GPU so I can use and test CUDA, but for now multiprocessing it is.

# Sources 🔌

 - https://www.youtube.com/watch?v=Lakz2MoHy6o&t=337s
 - https://www.youtube.com/watch?v=pj9-rr1wDhM
 - https://optimization.cbe.cornell.edu/index.php?title=Adam
 - https://paperswithcode.com/method/he-initialization#:~:text=Kaiming%20Initialization%2C%20or%20He%20Initialization,magnitudes%20of%20input%20signals%20exponentially.
 - https://builtin.com/machine-learning/adam-optimization
 - https://www.youtube.com/watch?v=Lakz2MoHy6o&t=337s
